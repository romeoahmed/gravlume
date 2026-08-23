use gravlume_domain::{EquatorialEmissionModel, Observation, SceneRadiance};
use wgpu::util::DeviceExt as _;

mod input;
#[cfg(test)]
mod inspection;
mod shader;
mod shadow_coverage;

pub use input::{GpuTraceInputError, TraceUniforms};

#[cfg(test)]
pub use inspection::{SampleInspection, SampleInspectionSource, SamplePolarSide, SampleSceneValue};

use input::TraceDispatch;

use crate::{
    extent::RenderExtent,
    scientific_capture::ScientificCaptureMetadata,
    spectral_lut::{BLACKBODY_LUT_BYTE_SIZE, blackbody_log2_fraction_lut},
};
use shadow_coverage::{ShadowCoverage, ShadowTarget};

const TRACE_RECORD_PLANE_ELEMENT_SIZE: u64 = 16;
const TRACE_WORKGROUP_AXIS: u32 = 8;

pub struct TracePipeline {
    pipeline: wgpu::ComputePipeline,
    escape_map_node_pipeline: Option<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    dispatch: wgpu::Buffer,
    blackbody_lut: Option<wgpu::Buffer>,
    shadow_coverage: Option<ShadowCoverage>,
    plan: TracePlan,
    scientific_capture_metadata: Option<ScientificCaptureMetadata>,
    #[cfg(test)]
    target_kind: TraceTargetKind,
}

#[derive(Clone, Copy)]
enum TracePlan {
    AcceleratedSky,
    EquatorialBolometricSurface,
    EquatorialBlackbodySurface,
}

struct CompiledTraceInput {
    uniforms: TraceUniforms,
    plan: TracePlan,
    scientific_capture_metadata: Option<ScientificCaptureMetadata>,
}

impl CompiledTraceInput {
    fn compile(observation: &Observation) -> Result<Self, GpuTraceInputError> {
        let scene = observation.scene();
        let (surface, plan) = match scene.radiance() {
            SceneRadiance::AnalyticSky => (None, TracePlan::AcceleratedSky),
            SceneRadiance::EquatorialSurface(surface) => {
                let plan = match surface.emitter().emission_model() {
                    EquatorialEmissionModel::InverseCubeBolometricV1 => {
                        TracePlan::EquatorialBolometricSurface
                    }
                    EquatorialEmissionModel::InverseCubeBlackbodyV1 { .. } => {
                        TracePlan::EquatorialBlackbodySurface
                    }
                };
                (Some(surface), plan)
            }
        };
        let uniforms = TraceUniforms::from_observation(observation)?;
        let scientific_capture_metadata = surface.map(|surface| {
            ScientificCaptureMetadata::for_surface(scene.spacetime().mass_m(), surface)
        });
        Ok(Self {
            uniforms,
            plan,
            scientific_capture_metadata,
        })
    }
}

impl TracePlan {
    fn scratch_bytes(self, extent: RenderExtent) -> u64 {
        match self {
            Self::AcceleratedSky => shadow_coverage_scratch_bytes(extent)
                .saturating_add(escape_map_scratch_bytes(extent)),
            Self::EquatorialBolometricSurface | Self::EquatorialBlackbodySurface => 0,
        }
    }

    const fn surface_events_enabled(self) -> f64 {
        match self {
            Self::AcceleratedSky => 0.0,
            Self::EquatorialBolometricSurface | Self::EquatorialBlackbodySurface => 1.0,
        }
    }

    const fn has_blackbody_lut(self) -> bool {
        matches!(self, Self::EquatorialBlackbodySurface)
    }
}

#[derive(Clone, Copy)]
enum TraceTargetKind {
    Presentation,
    #[cfg(test)]
    Diagnostic,
}

struct TracePipelineSpec {
    shader_source: String,
    entry_point: &'static str,
    escape_map_node_entry_point: Option<&'static str>,
    has_shadow_refinement: bool,
    target_kind: TraceTargetKind,
}

pub struct TraceTimestampWrites<'a> {
    escape_map_timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'a>>,
    trace_timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'a>>,
}

impl<'a> TraceTimestampWrites<'a> {
    pub(crate) const fn new(
        escape_map_timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'a>>,
        trace_timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'a>>,
    ) -> Self {
        Self {
            escape_map_timestamp_writes,
            trace_timestamp_writes,
        }
    }

    #[cfg(test)]
    const fn untimed() -> Self {
        Self::new(None, None)
    }
}

impl TracePlan {
    fn presentation_spec(self) -> TracePipelineSpec {
        let (shader_source, entry_point, escape_map_node_entry_point, has_shadow_refinement) =
            match self {
                Self::AcceleratedSky => (
                    shader::accelerated_scene(),
                    "trace_scene_accelerated",
                    Some("trace_escape_map_nodes"),
                    true,
                ),
                Self::EquatorialBolometricSurface => (
                    shader::bolometric_surface_scene(),
                    "trace_bolometric_surface_scene",
                    None,
                    false,
                ),
                Self::EquatorialBlackbodySurface => (
                    shader::blackbody_surface_scene(),
                    "trace_blackbody_surface_scene",
                    None,
                    false,
                ),
            };
        TracePipelineSpec {
            shader_source,
            entry_point,
            escape_map_node_entry_point,
            has_shadow_refinement,
            target_kind: TraceTargetKind::Presentation,
        }
    }

    #[cfg(test)]
    fn capture_spec(self) -> TracePipelineSpec {
        let (shader_source, entry_point, has_shadow_refinement) = match self {
            Self::AcceleratedSky => (shader::trace_capture(), "capture_trace_scene", true),
            Self::EquatorialBolometricSurface => (
                shader::bolometric_surface_capture(),
                "capture_surface_trace_scene",
                false,
            ),
            Self::EquatorialBlackbodySurface => (
                shader::blackbody_surface_capture(),
                "capture_surface_trace_scene",
                false,
            ),
        };
        TracePipelineSpec {
            shader_source,
            entry_point,
            escape_map_node_entry_point: None,
            has_shadow_refinement,
            target_kind: TraceTargetKind::Diagnostic,
        }
    }

    #[cfg(test)]
    fn footprint_capture_spec(self) -> Result<TracePipelineSpec, GpuTraceInputError> {
        let shader_source = match self {
            Self::AcceleratedSky => {
                return Err(GpuTraceInputError::SurfaceFootprintRequiresSurface);
            }
            Self::EquatorialBolometricSurface => shader::bolometric_surface_footprint_capture(),
            Self::EquatorialBlackbodySurface => shader::blackbody_surface_footprint_capture(),
        };
        Ok(TracePipelineSpec {
            shader_source,
            entry_point: "capture_surface_footprint",
            escape_map_node_entry_point: None,
            has_shadow_refinement: false,
            target_kind: TraceTargetKind::Diagnostic,
        })
    }

    #[cfg(test)]
    fn transport_capture_spec(self) -> Result<TracePipelineSpec, GpuTraceInputError> {
        if matches!(self, Self::AcceleratedSky) {
            return Err(GpuTraceInputError::SurfaceTransportRequiresSurface);
        }
        Ok(TracePipelineSpec {
            entry_point: "capture_surface_transport_case",
            ..self.capture_spec()
        })
    }
}

impl TraceTargetKind {
    const fn captures_records(self) -> bool {
        match self {
            Self::Presentation => false,
            #[cfg(test)]
            Self::Diagnostic => true,
        }
    }
}

fn trace_bind_group_layout_entries(
    target_kind: TraceTargetKind,
    has_escape_map: bool,
    has_blackbody_lut: bool,
) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size_of::<TraceUniforms>()),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size_of::<TraceDispatch>()),
            },
            count: None,
        },
    ];
    if target_kind.captures_records() {
        entries.extend((3..=6).map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(TRACE_RECORD_PLANE_ELEMENT_SIZE),
            },
            count: None,
        }));
    }
    if has_escape_map {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size_of::<u32>()),
            },
            count: None,
        });
    }
    if has_blackbody_lut {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(BLACKBODY_LUT_BYTE_SIZE),
            },
            count: None,
        });
    }
    entries
}

impl TracePipeline {
    pub(crate) fn new(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let compiled = CompiledTraceInput::compile(observation)?;
        let spec = compiled.plan.presentation_spec();
        Ok(Self::from_compiled(device, compiled, spec))
    }

    #[cfg(test)]
    pub(crate) fn for_trace_capture(
        device: &wgpu::Device,
        observation: &Observation,
        subpixel: [f64; 2],
    ) -> Result<Self, GpuTraceInputError> {
        let mut compiled = CompiledTraceInput::compile(observation)?;
        compiled
            .uniforms
            .set_capture_subpixel(subpixel, "trace_capture_subpixel")?;
        let spec = compiled.plan.capture_spec();
        Ok(Self::from_compiled(device, compiled, spec))
    }

    #[cfg(test)]
    pub(crate) fn for_surface_footprint_capture(
        device: &wgpu::Device,
        observation: &Observation,
        subpixel: [f64; 2],
    ) -> Result<Self, GpuTraceInputError> {
        let mut compiled = CompiledTraceInput::compile(observation)?;
        compiled
            .uniforms
            .set_capture_subpixel(subpixel, "surface_footprint_subpixel")?;
        compiled.uniforms.use_footprint_step_policy();
        let spec = compiled.plan.footprint_capture_spec()?;
        Ok(Self::from_compiled(device, compiled, spec))
    }

    #[cfg(test)]
    pub(crate) fn for_surface_transport_capture(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let compiled = CompiledTraceInput::compile(observation)?;
        let spec = compiled.plan.transport_capture_spec()?;
        Ok(Self::from_compiled(device, compiled, spec))
    }

    #[cfg(test)]
    pub(crate) fn for_accelerated_trace_capture(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let compiled = CompiledTraceInput::compile(observation)?;
        Ok(Self::from_compiled(
            device,
            compiled,
            TracePipelineSpec {
                shader_source: shader::accelerated_capture(),
                entry_point: "capture_accelerated_trace_scene",
                escape_map_node_entry_point: Some("trace_escape_map_nodes"),
                has_shadow_refinement: true,
                target_kind: TraceTargetKind::Diagnostic,
            },
        ))
    }

    #[cfg(test)]
    pub(crate) fn for_initial_ray_capture(
        device: &wgpu::Device,
        observation: &Observation,
        subpixel: [f32; 2],
    ) -> Result<Self, GpuTraceInputError> {
        let mut compiled = CompiledTraceInput::compile(observation)?;
        compiled.uniforms.set_packed_subpixel(subpixel);
        Ok(Self::from_compiled(
            device,
            compiled,
            TracePipelineSpec {
                shader_source: shader::initial_ray_capture(),
                entry_point: "write_initial_rays",
                escape_map_node_entry_point: None,
                has_shadow_refinement: false,
                target_kind: TraceTargetKind::Diagnostic,
            },
        ))
    }

    #[cfg(test)]
    pub(crate) fn for_invariant_gate_capture(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let compiled = CompiledTraceInput::compile(observation)?;
        Ok(Self::from_compiled(
            device,
            compiled,
            TracePipelineSpec {
                shader_source: shader::invariant_gate_capture(),
                entry_point: "write_invariant_gate_cases",
                escape_map_node_entry_point: None,
                has_shadow_refinement: false,
                target_kind: TraceTargetKind::Diagnostic,
            },
        ))
    }

    #[cfg(test)]
    pub(crate) fn for_event_policy_capture(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let compiled = CompiledTraceInput::compile(observation)?;
        Ok(Self::from_compiled(
            device,
            compiled,
            TracePipelineSpec {
                shader_source: shader::event_policy_capture(),
                entry_point: "write_event_policy_cases",
                escape_map_node_entry_point: None,
                has_shadow_refinement: false,
                target_kind: TraceTargetKind::Diagnostic,
            },
        ))
    }

    fn from_compiled(
        device: &wgpu::Device,
        compiled: CompiledTraceInput,
        spec: TracePipelineSpec,
    ) -> Self {
        let CompiledTraceInput {
            uniforms,
            plan,
            scientific_capture_metadata,
        } = compiled;
        let TracePipelineSpec {
            shader_source,
            entry_point,
            escape_map_node_entry_point,
            has_shadow_refinement,
            target_kind,
        } = spec;
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/util/trait.DeviceExt.html#tymethod.create_buffer_init
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU trace uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dispatch = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU trace dispatch"),
            contents: bytemuck::bytes_of(&TraceDispatch {
                tile_origin: [0; 2],
                workgroup_count: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let blackbody_lut = plan.has_blackbody_lut().then(|| {
            let entries = blackbody_log2_fraction_lut();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blackbody spectral log2-fraction LUT"),
                contents: bytemuck::cast_slice(&entries),
                usage: wgpu::BufferUsages::STORAGE,
            })
        });
        let shadow_coverage = has_shadow_refinement.then(|| ShadowCoverage::new(device, &uniforms));
        let entries = trace_bind_group_layout_entries(
            target_kind,
            escape_map_node_entry_point.is_some(),
            blackbody_lut.is_some(),
        );
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU trace bind group layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU trace pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cartesian Kerr-Schild trace shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let pipeline_constants = [("SURFACE_EVENTS_ENABLED", plan.surface_events_enabled())];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &pipeline_constants,
                ..Default::default()
            },
            cache: None,
        });
        let escape_map_node_pipeline = escape_map_node_entry_point.map(|entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &pipeline_constants,
                    ..Default::default()
                },
                cache: None,
            })
        });

        Self {
            pipeline,
            escape_map_node_pipeline,
            bind_group_layout,
            uniforms,
            dispatch,
            blackbody_lut,
            shadow_coverage,
            plan,
            scientific_capture_metadata,
            #[cfg(test)]
            target_kind,
        }
    }

    pub(crate) fn scratch_bytes(&self, extent: RenderExtent) -> u64 {
        self.plan.scratch_bytes(extent)
    }

    pub(crate) const fn has_escape_map(&self) -> bool {
        self.escape_map_node_pipeline.is_some()
    }

    pub(crate) const fn scientific_capture_metadata(&self) -> Option<&ScientificCaptureMetadata> {
        self.scientific_capture_metadata.as_ref()
    }

    pub(crate) fn create_target(&self, device: &wgpu::Device, extent: RenderExtent) -> TraceTarget {
        let texture_usage = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-linear HDR trace target"),
            size: wgpu::Extent3d {
                width: extent.width(),
                height: extent.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: texture_usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        #[cfg(test)]
        let record_planes = self.target_kind.captures_records().then(|| {
            let size = trace_record_plane_size(extent);
            DiagnosticPlanes {
                source_time: create_record_plane(device, "trace source and time", size),
                invariant_drift: create_record_plane(device, "trace invariant drift", size),
                metadata: create_record_plane(device, "trace metadata", size),
                event: create_record_plane(device, "trace event candidates", size),
            }
        });
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.dispatch.as_entire_binding(),
            },
        ];
        #[cfg(test)]
        let capture_entries = record_planes.as_ref().map(|planes| {
            planes
                .buffers()
                .into_iter()
                .enumerate()
                .map(|(index, buffer)| wgpu::BindGroupEntry {
                    binding: u32::try_from(index).expect("diagnostic binding index fits u32") + 3,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>()
        });
        #[cfg(not(test))]
        let capture_entries: Option<Vec<wgpu::BindGroupEntry<'_>>> = None;
        let escape_map = self.escape_map_node_pipeline.as_ref().map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("packed escape-direction map"),
                size: escape_map_scratch_bytes(extent),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let escape_map_entry = escape_map.as_ref().map(|buffer| wgpu::BindGroupEntry {
            binding: 7,
            resource: buffer.as_entire_binding(),
        });
        let blackbody_lut_entry = self
            .blackbody_lut
            .as_ref()
            .map(|buffer| wgpu::BindGroupEntry {
                binding: 8,
                resource: buffer.as_entire_binding(),
            });
        let entries = entries
            .into_iter()
            .chain(capture_entries.into_iter().flatten())
            .chain(escape_map_entry)
            .chain(blackbody_lut_entry)
            .collect::<Vec<_>>();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU trace bind group"),
            layout: &self.bind_group_layout,
            entries: &entries,
        });
        let shadow_coverage = self
            .shadow_coverage
            .as_ref()
            .map(|coverage| coverage.create_target(device, &view, extent));
        TraceTarget {
            view,
            #[cfg(test)]
            record_planes,
            bind_group,
            shadow_coverage,
        }
    }

    #[cfg(test)]
    pub(crate) fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceTarget,
        tiles: TileRegion,
    ) {
        self.encode_batch_with_refinement(
            queue,
            encoder,
            target,
            tiles,
            TraceTimestampWrites::untimed(),
            true,
        );
    }

    #[cfg(test)]
    pub(crate) fn encode_base(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceTarget,
        tiles: TileRegion,
    ) {
        self.encode_batch_with_refinement(
            queue,
            encoder,
            target,
            tiles,
            TraceTimestampWrites::untimed(),
            false,
        );
    }

    pub(crate) fn encode_batch(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceTarget,
        tiles: TileRegion,
        timestamp_writes: TraceTimestampWrites<'_>,
    ) {
        self.encode_batch_with_refinement(queue, encoder, target, tiles, timestamp_writes, true);
    }

    fn encode_batch_with_refinement(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceTarget,
        tiles: TileRegion,
        timestamp_writes: TraceTimestampWrites<'_>,
        refine_final_batch: bool,
    ) {
        let TraceTimestampWrites {
            escape_map_timestamp_writes,
            trace_timestamp_writes,
        } = timestamp_writes;
        self.set_tile_dispatch(queue, tiles);
        self.encode_escape_map_pass(encoder, target, tiles, escape_map_timestamp_writes);
        self.encode_trace_pass(
            encoder,
            target,
            tiles,
            trace_timestamp_writes,
            refine_final_batch,
        );
    }

    fn encode_escape_map_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceTarget,
        tiles: TileRegion,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let Some(node_pipeline) = &self.escape_map_node_pipeline else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("escape-direction map pass"),
            timestamp_writes,
        });
        pass.set_pipeline(node_pipeline);
        pass.set_bind_group(0, &target.bind_group, &[]);
        let [workgroups_x, workgroups_y] = escape_map_node_workgroups(tiles);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    fn encode_trace_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceTarget,
        tiles: TileRegion,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
        refine_final_batch: bool,
    ) {
        let refinement = self
            .shadow_coverage
            .as_ref()
            .zip(target.shadow_coverage.as_ref())
            .filter(|(_, shadow)| refine_final_batch && tiles.finishes(shadow.extent));
        if let Some((_, shadow)) = refinement {
            ShadowCoverage::reset_control(encoder, shadow);
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GPU trace pass"),
            timestamp_writes,
        });
        pass.set_bind_group(0, &target.bind_group, &[]);
        pass.set_pipeline(&self.pipeline);
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePass.html#method.dispatch_workgroups
        let [workgroups_x, workgroups_y] = tiles.workgroups();
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        if let Some((coverage, shadow)) = refinement {
            coverage.encode(&mut pass, shadow);
        }
    }

    fn set_tile_dispatch(&self, queue: &wgpu::Queue, tiles: TileRegion) {
        let [origin_x, origin_y] = tiles.origin();
        let [workgroups_x, workgroups_y] = tiles.workgroups();
        let dispatch = TraceDispatch {
            tile_origin: [origin_x, origin_y],
            workgroup_count: [workgroups_x, workgroups_y],
        };
        // Small queue writes are staged immediately and execute before the following submission.
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html#method.write_buffer
        queue.write_buffer(&self.dispatch, 0, bytemuck::bytes_of(&dispatch));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileRegion {
    origin: [u32; 2],
    workgroups: [u32; 2],
}

impl TileRegion {
    #[cfg(any(test, feature = "gpu-benchmarks"))]
    pub(crate) const fn all(extent: RenderExtent) -> Self {
        Self {
            origin: [0, 0],
            workgroups: tile_grid(extent),
        }
    }

    pub(crate) const fn new(origin: [u32; 2], workgroups: [u32; 2]) -> Self {
        debug_assert!(workgroups[0] > 0 && workgroups[1] > 0);
        Self { origin, workgroups }
    }

    #[cfg(test)]
    pub(crate) const fn containing_pixel(pixel: [u32; 2]) -> Self {
        Self::new(
            [
                pixel[0] / TRACE_WORKGROUP_AXIS,
                pixel[1] / TRACE_WORKGROUP_AXIS,
            ],
            [1, 1],
        )
    }

    pub(crate) const fn origin(self) -> [u32; 2] {
        self.origin
    }

    pub(crate) const fn workgroups(self) -> [u32; 2] {
        self.workgroups
    }

    pub(crate) const fn len(self) -> u32 {
        self.workgroups[0] * self.workgroups[1]
    }

    pub(crate) const fn finishes(self, extent: RenderExtent) -> bool {
        let grid = tile_grid(extent);
        let row_contiguous =
            self.workgroups[1] == 1 || (self.origin[0] == 0 && self.workgroups[0] == grid[0]);
        row_contiguous
            && self.origin[0] + self.workgroups[0] == grid[0]
            && self.origin[1] + self.workgroups[1] == grid[1]
    }
}

pub const fn tile_grid(extent: RenderExtent) -> [u32; 2] {
    [
        extent.width().div_ceil(TRACE_WORKGROUP_AXIS),
        extent.height().div_ceil(TRACE_WORKGROUP_AXIS),
    ]
}

const fn escape_map_node_workgroups(tiles: TileRegion) -> [u32; 2] {
    let [tile_columns, tile_rows] = tiles.workgroups();
    [
        (tile_columns * 2 + 1).div_ceil(TRACE_WORKGROUP_AXIS),
        (tile_rows * 2 + 1).div_ceil(TRACE_WORKGROUP_AXIS),
    ]
}

pub fn escape_map_scratch_bytes(extent: RenderExtent) -> u64 {
    let columns = u64::from(extent.width().div_ceil(TRACE_WORKGROUP_AXIS)) * 2 + 1;
    let rows = u64::from(extent.height().div_ceil(TRACE_WORKGROUP_AXIS)) * 2 + 1;
    columns
        .saturating_mul(rows)
        .saturating_mul(size_of::<u32>())
}

pub fn shadow_coverage_scratch_bytes(extent: RenderExtent) -> u64 {
    shadow_coverage::scratch_bytes(extent)
}

pub struct TraceTarget {
    view: wgpu::TextureView,
    #[cfg(test)]
    record_planes: Option<DiagnosticPlanes>,
    bind_group: wgpu::BindGroup,
    shadow_coverage: Option<ShadowTarget>,
}

impl TraceTarget {
    pub(crate) const fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    #[cfg(test)]
    pub(crate) fn texture(&self) -> &wgpu::Texture {
        self.view.texture()
    }

    #[cfg(test)]
    pub(crate) const fn record_planes(&self) -> [&wgpu::Buffer; 4] {
        self.record_planes
            .as_ref()
            .expect("capture targets include diagnostic record planes")
            .buffers()
    }

    #[cfg(test)]
    pub(crate) const fn shadow_control(&self) -> &wgpu::Buffer {
        &self
            .shadow_coverage
            .as_ref()
            .expect("refined capture target contains shadow control")
            .control
    }
}

#[cfg(test)]
struct DiagnosticPlanes {
    source_time: wgpu::Buffer,
    invariant_drift: wgpu::Buffer,
    metadata: wgpu::Buffer,
    event: wgpu::Buffer,
}

#[cfg(test)]
impl DiagnosticPlanes {
    const fn buffers(&self) -> [&wgpu::Buffer; 4] {
        [
            &self.source_time,
            &self.invariant_drift,
            &self.metadata,
            &self.event,
        ]
    }
}

#[cfg(test)]
fn create_record_plane(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
pub fn trace_record_plane_size(extent: RenderExtent) -> u64 {
    u64::from(extent.width())
        .saturating_mul(u64::from(extent.height()))
        .saturating_mul(TRACE_RECORD_PLANE_ELEMENT_SIZE)
}

pub const fn size_of<T>() -> u64 {
    std::mem::size_of::<T>() as u64
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TraceTermination {
    HorizonCrossing = 1,
    Escape = 2,
    SingularityGuard = 3,
    StepExhaustion = 4,
    NumericalFailure = 5,
    Uncertain = 6,
    EquatorialSurface = 7,
}

#[cfg(test)]
impl From<TraceTermination> for u32 {
    fn from(value: TraceTermination) -> Self {
        value as Self
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown GPU trace termination discriminant {0}")]
pub struct UnknownTraceTermination(pub u32);

#[cfg(test)]
impl TryFrom<u32> for TraceTermination {
    type Error = UnknownTraceTermination;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::HorizonCrossing),
            2 => Ok(Self::Escape),
            3 => Ok(Self::SingularityGuard),
            4 => Ok(Self::StepExhaustion),
            5 => Ok(Self::NumericalFailure),
            6 => Ok(Self::Uncertain),
            7 => Ok(Self::EquatorialSurface),
            unknown => Err(UnknownTraceTermination(unknown)),
        }
    }
}

use std::borrow::Cow;

use gravlume_domain::{
    Extremality, KerrNewmanSpacetime, KerrSchildChart, Observation, ValidationReport,
};
use num_traits::ToPrimitive as _;
use wgpu::util::DeviceExt as _;

use crate::{
    extent::RenderExtent,
    shadow_coverage::{ShadowCoverage, ShadowTarget},
};

pub const TRACE_SHADER: &str = include_str!("shaders/trace.wgsl");
pub const DIRECTION_RECONSTRUCTION_SHADER: &str =
    include_str!("shaders/direction_reconstruction.wgsl");
pub const SHADOW_COVERAGE_SHADER: &str = include_str!("shaders/shadow_coverage.wgsl");
#[cfg(test)]
const DIAGNOSTIC_SHADER: &str = include_str!("shaders/trace_diagnostic.wgsl");
#[cfg(test)]
const DIRECTION_RECONSTRUCTION_DIAGNOSTIC_SHADER: &str =
    include_str!("shaders/direction_reconstruction_diagnostic.wgsl");
#[cfg(test)]
const INITIAL_RAY_CAPTURE_SHADER: &str = include_str!("shaders/initial_ray_capture.wgsl");
#[cfg(test)]
const INVARIANT_GATE_CAPTURE_SHADER: &str = include_str!("shaders/invariant_gate_capture.wgsl");
pub const INVARIANT_DRIFT_LIMIT: f32 = 0.05;
const NORMALIZED_FREQUENCY_TOLERANCE: f64 = 32.0 * f64::EPSILON;
pub const KERR_CAPTURE_OVERRIDE: &str = "KERR_CAPTURE_FAST_PATH";
const TRACE_RECORD_PLANE_ELEMENT_SIZE: u64 = 16;
pub const TRACE_WORKGROUP_WIDTH: u32 = 8;
pub const TRACE_WORKGROUP_HEIGHT: u32 = 8;

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct TraceUniforms {
    pub spacetime: [f32; 4],
    pub observer_event: [f32; 4],
    pub observer_velocity: [f32; 4],
    pub image_right: [f32; 4],
    pub image_up: [f32; 4],
    pub arrival: [f32; 4],
    pub view: [f32; 4],
    pub event_surfaces: [f32; 4],
    pub step_policy: [f32; 4],
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct TraceDispatch {
    tile_region: [u32; 4],
}

impl TraceUniforms {
    pub fn from_observation(observation: &Observation) -> Result<Self, GpuTraceInputError> {
        let scene = observation.scene();
        let physical_spacetime = *scene.spacetime();
        let mass = physical_spacetime.mass_m();
        let view = *observation.view();
        let frame = scene.observer_frame();
        let sample = view
            .sample(0, 0, 0.5, 0.5)
            .map_err(GpuTraceInputError::DomainInvariant)?;
        let observer_frequency = observation
            .initial_ray(sample)
            .map_err(GpuTraceInputError::DomainInvariant)?
            .observer_frequency();
        if (observer_frequency - 1.0).abs() > NORMALIZED_FREQUENCY_TOLERANCE {
            return Err(GpuTraceInputError::NonNormalizedObserverFrequency { observer_frequency });
        }
        let spacetime_uniform = pack4(
            [
                1.0,
                physical_spacetime.spin_m() / mass,
                physical_spacetime.charge_m() / mass,
                match physical_spacetime.chart() {
                    KerrSchildChart::Ingoing => 1.0,
                    KerrSchildChart::Outgoing => -1.0,
                },
            ],
            "spacetime",
        )?;
        let gpu_spacetime = KerrNewmanSpacetime::new(
            f64::from(spacetime_uniform[0]),
            f64::from(spacetime_uniform[1]),
            f64::from(spacetime_uniform[2]),
            physical_spacetime.chart(),
        )
        .map_err(GpuTraceInputError::DomainInvariant)?;
        let canonical_state = physical_spacetime.extremality();
        let gpu_extremality = gpu_spacetime.extremality();
        if gpu_extremality != canonical_state {
            return Err(GpuTraceInputError::ExtremalityChangedByPacking {
                canonical_state,
                gpu_extremality,
            });
        }
        let horizon = gpu_spacetime.outer_horizon_radius().unwrap_or(-1.0);
        let [_, observer_x, observer_y, observer_z] = scene.observer_event().to_txyz();

        Ok(Self {
            spacetime: spacetime_uniform,
            observer_event: pack4(
                [0.0, observer_x / mass, observer_y / mass, observer_z / mass],
                "observer_event",
            )?,
            observer_velocity: pack4(frame.four_velocity_txyz(), "observer_velocity")?,
            image_right: pack4(frame.image_right_txyz(), "image_right")?,
            image_up: pack4(frame.image_up_txyz(), "image_up")?,
            arrival: pack4(frame.arrival_direction_txyz(), "arrival")?,
            view: pack4(
                [(view.vertical_fov().radians() * 0.5).tan(), 1.0, 0.5, 0.5],
                "view",
            )?,
            event_surfaces: pack4(
                [200.0, f64::from(f32::from_bits(0x2b80_0000)), horizon, 0.0],
                "event_surfaces",
            )?,
            step_policy: [0.1, 0.005, 8.0, INVARIANT_DRIFT_LIMIT],
        })
    }
}

fn pack4(values: [f64; 4], field: &'static str) -> Result<[f32; 4], GpuTraceInputError> {
    let [Some(a), Some(b), Some(c), Some(d)] =
        values.map(|value| value.to_f32().filter(|packed| packed.is_finite()))
    else {
        return Err(GpuTraceInputError::NotRepresentable { field });
    };
    Ok([a, b, c, d])
}

#[derive(Debug, thiserror::Error)]
pub enum GpuTraceInputError {
    #[error("validated observation failed to resolve its initial ray: {0}")]
    DomainInvariant(#[source] ValidationReport),
    #[error(
        "GPU trace inputs must be normalized to observer frequency 1, got {observer_frequency}"
    )]
    NonNormalizedObserverFrequency { observer_frequency: f64 },
    #[error(
        "spacetime extremality changes from {canonical_state:?} to {gpu_extremality:?} under the GPU f32 contract"
    )]
    ExtremalityChangedByPacking {
        canonical_state: Extremality,
        gpu_extremality: Extremality,
    },
    #[error("observation field {field} is not representable by the GPU f32 contract")]
    NotRepresentable { field: &'static str },
}

pub struct RayTracer {
    pipeline: wgpu::ComputePipeline,
    reconstruction_node_pipeline: Option<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    dispatch: wgpu::Buffer,
    shadow_coverage: ShadowCoverage,
    target_kind: TraceTargetKind,
}

#[derive(Clone, Copy)]
enum TraceTargetKind {
    Presentation,
    #[cfg(test)]
    Diagnostic,
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
    has_direction_map: bool,
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
        entries.extend((3..=5).map(|binding| wgpu::BindGroupLayoutEntry {
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
    if has_direction_map {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size_of::<u32>()),
            },
            count: None,
        });
    }
    entries
}

impl RayTracer {
    pub(crate) fn new(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let uniforms = TraceUniforms::from_observation(observation)?;
        Ok(Self::from_uniforms(
            device,
            uniforms,
            direction_reconstruction_shader_source(),
            "trace_scene_direction_reconstruction",
            Some("trace_direction_reconstruction_nodes"),
            TraceTargetKind::Presentation,
        ))
    }

    #[cfg(test)]
    pub(crate) fn for_trace_capture(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let uniforms = TraceUniforms::from_observation(observation)?;
        Ok(Self::from_uniforms(
            device,
            uniforms,
            diagnostic_shader_source(),
            "capture_trace_scene",
            None,
            TraceTargetKind::Diagnostic,
        ))
    }

    #[cfg(test)]
    pub(crate) fn for_direction_reconstruction_trace_capture(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let uniforms = TraceUniforms::from_observation(observation)?;
        Ok(Self::from_uniforms(
            device,
            uniforms,
            direction_reconstruction_diagnostic_shader_source(),
            "capture_direction_reconstruction_trace_scene",
            Some("trace_direction_reconstruction_nodes"),
            TraceTargetKind::Diagnostic,
        ))
    }

    #[cfg(test)]
    pub(crate) fn for_initial_ray_capture(
        device: &wgpu::Device,
        observation: &Observation,
        subpixel: [f32; 2],
    ) -> Result<Self, GpuTraceInputError> {
        let mut uniforms = TraceUniforms::from_observation(observation)?;
        uniforms.view[2..].copy_from_slice(&subpixel);
        Ok(Self::from_uniforms(
            device,
            uniforms,
            Cow::Owned(format!(
                "{}\n{INITIAL_RAY_CAPTURE_SHADER}",
                diagnostic_shader_source()
            )),
            "write_initial_rays",
            None,
            TraceTargetKind::Diagnostic,
        ))
    }

    #[cfg(test)]
    pub(crate) fn for_invariant_gate_capture(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let uniforms = TraceUniforms::from_observation(observation)?;
        Ok(Self::from_uniforms(
            device,
            uniforms,
            Cow::Owned(format!(
                "{}\n{INVARIANT_GATE_CAPTURE_SHADER}",
                diagnostic_shader_source()
            )),
            "write_invariant_gate_cases",
            None,
            TraceTargetKind::Diagnostic,
        ))
    }

    fn from_uniforms(
        device: &wgpu::Device,
        uniforms: TraceUniforms,
        shader_source: Cow<'static, str>,
        entry_point: &'static str,
        reconstruction_node_entry_point: Option<&'static str>,
        target_kind: TraceTargetKind,
    ) -> Self {
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/util/trait.DeviceExt.html#tymethod.create_buffer_init
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU trace uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dispatch = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU trace dispatch"),
            contents: bytemuck::bytes_of(&TraceDispatch {
                tile_region: [0; 4],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let shadow_coverage = ShadowCoverage::new(device, &uniforms);
        let entries =
            trace_bind_group_layout_entries(target_kind, reconstruction_node_entry_point.is_some());
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
            source: wgpu::ShaderSource::Wgsl(shader_source),
        });
        let capture_constants = [(KERR_CAPTURE_OVERRIDE, 1.0)];
        let pipeline_constants = if reconstruction_node_entry_point.is_some() {
            capture_constants.as_slice()
        } else {
            &[]
        };
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: pipeline_constants,
                ..Default::default()
            },
            cache: None,
        });
        let reconstruction_node_pipeline = reconstruction_node_entry_point.map(|entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: pipeline_constants,
                    ..Default::default()
                },
                cache: None,
            })
        });

        Self {
            pipeline,
            reconstruction_node_pipeline,
            bind_group_layout,
            uniforms,
            dispatch,
            shadow_coverage,
            target_kind,
        }
    }

    pub(crate) fn create_target(&self, device: &wgpu::Device, extent: RenderExtent) -> TraceImage {
        let mut texture_usage =
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        if self.target_kind.captures_records() {
            texture_usage |= wgpu::TextureUsages::COPY_SRC;
        }
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
                direction_time: create_record_plane(device, "trace direction and time", size),
                invariant_drift: create_record_plane(device, "trace invariant drift", size),
                metadata: create_record_plane(device, "trace metadata", size),
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
        let direction_map = self.reconstruction_node_pipeline.as_ref().map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("packed direction reconstruction map"),
                size: direction_reconstruction_scratch_bytes(extent),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let direction_map_entry = direction_map.as_ref().map(|buffer| wgpu::BindGroupEntry {
            binding: 6,
            resource: buffer.as_entire_binding(),
        });
        let entries = entries
            .into_iter()
            .chain(capture_entries.into_iter().flatten())
            .chain(direction_map_entry)
            .collect::<Vec<_>>();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU trace bind group"),
            layout: &self.bind_group_layout,
            entries: &entries,
        });
        let shadow_coverage = self.shadow_coverage.create_target(device, &view, extent);
        TraceImage {
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
        target: &TraceImage,
        tiles: TileRegion,
    ) {
        self.encode_node_pass(queue, encoder, target, tiles, None);
        self.encode_resolve_pass(encoder, target, tiles, None, true);
    }

    #[cfg(test)]
    pub(crate) fn encode_base(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceImage,
        tiles: TileRegion,
    ) {
        self.encode_node_pass(queue, encoder, target, tiles, None);
        self.encode_resolve_pass(encoder, target, tiles, None, false);
    }

    pub(crate) fn encode_node_pass(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceImage,
        tiles: TileRegion,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        self.set_tile_region(queue, tiles);
        let Some(node_pipeline) = &self.reconstruction_node_pipeline else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("direction reconstruction node pass"),
            timestamp_writes,
        });
        pass.set_pipeline(node_pipeline);
        pass.set_bind_group(0, &target.bind_group, &[]);
        let [workgroups_x, workgroups_y] = direction_reconstruction_node_workgroups(tiles);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    pub(crate) fn encode_resolve_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceImage,
        tiles: TileRegion,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
        refine_final_batch: bool,
    ) {
        let refine_shadow = refine_final_batch && tiles.finishes(target.shadow_coverage.extent);
        if refine_shadow {
            ShadowCoverage::reset_control(encoder, &target.shadow_coverage);
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
        if refine_shadow {
            self.shadow_coverage
                .encode(&mut pass, &target.shadow_coverage);
        }
    }

    fn set_tile_region(&self, queue: &wgpu::Queue, tiles: TileRegion) {
        let [origin_x, origin_y] = tiles.origin();
        let [workgroups_x, workgroups_y] = tiles.workgroups();
        let dispatch = TraceDispatch {
            tile_region: [origin_x, origin_y, workgroups_x, workgroups_y],
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

    pub(crate) const fn origin(self) -> [u32; 2] {
        self.origin
    }

    pub(crate) const fn workgroups(self) -> [u32; 2] {
        self.workgroups
    }

    pub(crate) const fn len(self) -> u32 {
        self.workgroups[0] * self.workgroups[1]
    }

    pub const fn finishes(self, extent: RenderExtent) -> bool {
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
        extent.width().div_ceil(TRACE_WORKGROUP_WIDTH),
        extent.height().div_ceil(TRACE_WORKGROUP_HEIGHT),
    ]
}

const fn direction_reconstruction_node_workgroups(tiles: TileRegion) -> [u32; 2] {
    let [tile_columns, tile_rows] = tiles.workgroups();
    [
        (tile_columns * 2 + 1).div_ceil(TRACE_WORKGROUP_WIDTH),
        (tile_rows * 2 + 1).div_ceil(TRACE_WORKGROUP_HEIGHT),
    ]
}

pub fn direction_reconstruction_scratch_bytes(extent: RenderExtent) -> u64 {
    let columns = u64::from(extent.width().div_ceil(TRACE_WORKGROUP_WIDTH)) * 2 + 1;
    let rows = u64::from(extent.height().div_ceil(TRACE_WORKGROUP_HEIGHT)) * 2 + 1;
    columns
        .saturating_mul(rows)
        .saturating_mul(size_of::<u32>())
}

fn direction_reconstruction_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!("{TRACE_SHADER}\n{DIRECTION_RECONSTRUCTION_SHADER}"))
}

#[cfg(test)]
fn diagnostic_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!("{TRACE_SHADER}\n{DIAGNOSTIC_SHADER}"))
}

#[cfg(test)]
fn direction_reconstruction_diagnostic_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!(
        "{TRACE_SHADER}\n{DIAGNOSTIC_SHADER}\n{DIRECTION_RECONSTRUCTION_SHADER}\n{DIRECTION_RECONSTRUCTION_DIAGNOSTIC_SHADER}"
    ))
}

pub fn shadow_coverage_scratch_bytes(extent: RenderExtent) -> u64 {
    crate::shadow_coverage::scratch_bytes(extent)
}

pub struct TraceImage {
    view: wgpu::TextureView,
    #[cfg(test)]
    record_planes: Option<DiagnosticPlanes>,
    bind_group: wgpu::BindGroup,
    shadow_coverage: ShadowTarget,
}

impl TraceImage {
    pub(crate) const fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    #[cfg(test)]
    pub(crate) fn texture(&self) -> &wgpu::Texture {
        self.view.texture()
    }

    #[cfg(test)]
    pub(crate) const fn record_planes(&self) -> [&wgpu::Buffer; 3] {
        self.record_planes
            .as_ref()
            .expect("capture targets include diagnostic record planes")
            .buffers()
    }

    #[cfg(test)]
    pub(crate) const fn shadow_control(&self) -> &wgpu::Buffer {
        &self.shadow_coverage.control
    }
}

#[cfg(test)]
struct DiagnosticPlanes {
    direction_time: wgpu::Buffer,
    invariant_drift: wgpu::Buffer,
    metadata: wgpu::Buffer,
}

#[cfg(test)]
impl DiagnosticPlanes {
    const fn buffers(&self) -> [&wgpu::Buffer; 3] {
        [&self.direction_time, &self.invariant_drift, &self.metadata]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TraceTermination {
    HorizonCrossing = 1,
    Escape = 2,
    SingularityGuard = 3,
    StepExhaustion = 4,
    NumericalFailure = 5,
    Uncertain = 6,
}

impl From<TraceTermination> for u32 {
    fn from(value: TraceTermination) -> Self {
        value as Self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown GPU trace termination discriminant {0}")]
pub struct UnknownTraceTermination(pub u32);

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
            unknown => Err(UnknownTraceTermination(unknown)),
        }
    }
}

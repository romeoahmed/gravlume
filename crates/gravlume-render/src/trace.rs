use std::borrow::Cow;

use gravlume_domain::{KerrNewmanSpacetime, Observation, ParameterState, ValidationReport};
use num_traits::ToPrimitive as _;
use wgpu::util::DeviceExt as _;

use crate::extent::RenderExtent;

const TRACE_SHADER: &str = include_str!("shaders/trace.wgsl");
#[cfg(test)]
const INITIAL_RAY_CAPTURE_SHADER: &str = include_str!("shaders/initial_ray_capture.wgsl");
#[cfg(test)]
const INVARIANT_GATE_CAPTURE_SHADER: &str = include_str!("shaders/invariant_gate_capture.wgsl");
pub const INVARIANT_DRIFT_LIMIT: f32 = 0.05;
const NORMALIZED_FREQUENCY_TOLERANCE: f64 = 32.0 * f64::EPSILON;
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
    pub projection: [f32; 4],
    pub event_surfaces: [f32; 4],
    pub step_policy: [f32; 4],
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct TraceDispatch {
    pixels: [u32; 4],
}

impl TraceUniforms {
    pub fn from_observation(observation: &Observation) -> Result<Self, TraceInputError> {
        let scene = observation.scene();
        let physical_spacetime = *scene.spacetime();
        let mass = physical_spacetime.mass_m();
        let projection = *observation.projection();
        let frame = scene.observer_frame();
        let sample = projection
            .sample(0, 0, 0.5, 0.5)
            .map_err(TraceInputError::DomainInvariant)?;
        let observer_frequency = observation
            .initial_ray(sample)
            .map_err(TraceInputError::DomainInvariant)?
            .observer_frequency();
        if (observer_frequency - 1.0).abs() > NORMALIZED_FREQUENCY_TOLERANCE {
            return Err(TraceInputError::NonNormalizedObserverFrequency { observer_frequency });
        }
        let spacetime_uniform = pack4(
            [
                1.0,
                physical_spacetime.spin_m() / mass,
                physical_spacetime.charge_m() / mass,
                0.0,
            ],
            "spacetime",
        )?;
        let interactive_spacetime = KerrNewmanSpacetime::new(
            f64::from(spacetime_uniform[0]),
            f64::from(spacetime_uniform[1]),
            f64::from(spacetime_uniform[2]),
        )
        .map_err(TraceInputError::DomainInvariant)?;
        let canonical_state = physical_spacetime.parameter_state();
        let interactive_state = interactive_spacetime.parameter_state();
        if interactive_state != canonical_state {
            return Err(TraceInputError::ParameterStateChangedByPacking {
                canonical_state,
                interactive_state,
            });
        }
        let horizon = interactive_spacetime.outer_horizon_radius().unwrap_or(-1.0);
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
            projection: pack4(
                [
                    (projection.vertical_fov().radians() * 0.5).tan(),
                    1.0,
                    0.5,
                    0.5,
                ],
                "projection",
            )?,
            event_surfaces: pack4(
                [200.0, f64::from(f32::from_bits(0x2b80_0000)), horizon, 0.0],
                "event_surfaces",
            )?,
            step_policy: [0.01, 0.005, 0.5, INVARIANT_DRIFT_LIMIT],
        })
    }
}

fn pack4(values: [f64; 4], field: &'static str) -> Result<[f32; 4], TraceInputError> {
    let [Some(a), Some(b), Some(c), Some(d)] =
        values.map(|value| value.to_f32().filter(|packed| packed.is_finite()))
    else {
        return Err(TraceInputError::NotRepresentable { field });
    };
    Ok([a, b, c, d])
}

#[derive(Debug, thiserror::Error)]
pub enum TraceInputError {
    #[error("validated observation failed to resolve its initial ray: {0}")]
    DomainInvariant(#[source] ValidationReport),
    #[error(
        "interactive trace inputs must be normalized to observer frequency 1, got {observer_frequency}"
    )]
    NonNormalizedObserverFrequency { observer_frequency: f64 },
    #[error(
        "spacetime parameter state changes from {canonical_state:?} to {interactive_state:?} under the interactive f32 contract"
    )]
    ParameterStateChangedByPacking {
        canonical_state: ParameterState,
        interactive_state: ParameterState,
    },
    #[error("observation field {field} is not representable by the interactive f32 contract")]
    NotRepresentable { field: &'static str },
}

pub struct TraceCompute {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    dispatch: wgpu::Buffer,
}

impl TraceCompute {
    pub(crate) fn new(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, TraceInputError> {
        let uniforms = TraceUniforms::from_observation(observation)?;
        Ok(Self::from_uniforms(
            device,
            uniforms,
            TRACE_SHADER.into(),
            "trace_scene",
        ))
    }

    #[cfg(test)]
    pub(crate) fn for_initial_ray_capture(
        device: &wgpu::Device,
        observation: &Observation,
        subpixel: [f32; 2],
    ) -> Result<Self, TraceInputError> {
        let mut uniforms = TraceUniforms::from_observation(observation)?;
        uniforms.projection[2..].copy_from_slice(&subpixel);
        Ok(Self::from_uniforms(
            device,
            uniforms,
            Cow::Owned(format!("{TRACE_SHADER}\n{INITIAL_RAY_CAPTURE_SHADER}")),
            "write_initial_rays",
        ))
    }

    #[cfg(test)]
    pub(crate) fn for_invariant_gate_capture(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, TraceInputError> {
        let uniforms = TraceUniforms::from_observation(observation)?;
        Ok(Self::from_uniforms(
            device,
            uniforms,
            Cow::Owned(format!("{TRACE_SHADER}\n{INVARIANT_GATE_CAPTURE_SHADER}")),
            "write_invariant_gate_cases",
        ))
    }

    fn from_uniforms(
        device: &wgpu::Device,
        uniforms: TraceUniforms,
        shader_source: Cow<'static, str>,
        entry_point: &'static str,
    ) -> Self {
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/util/trait.DeviceExt.html#tymethod.create_buffer_init
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("interactive trace uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dispatch = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("interactive trace dispatch"),
            contents: bytemuck::bytes_of(&TraceDispatch { pixels: [0; 4] }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("interactive trace bind group layout"),
            entries: &[
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(TRACE_RECORD_PLANE_ELEMENT_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(TRACE_RECORD_PLANE_ELEMENT_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(TRACE_RECORD_PLANE_ELEMENT_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<TraceDispatch>()),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("interactive trace pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cartesian Kerr-Schild trace shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniforms,
            dispatch,
        }
    }

    pub(crate) fn create_target(&self, device: &wgpu::Device, extent: RenderExtent) -> TraceTarget {
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
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let record_plane_size = trace_record_plane_size(extent);
        let direction_time =
            create_record_plane(device, "trace direction and time", record_plane_size);
        let invariant_drift =
            create_record_plane(device, "trace invariant drift", record_plane_size);
        let metadata = create_record_plane(device, "trace metadata", record_plane_size);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("interactive trace bind group"),
            layout: &self.bind_group_layout,
            entries: &[
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
                    resource: direction_time.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: invariant_drift.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: metadata.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.dispatch.as_entire_binding(),
                },
            ],
        });
        TraceTarget {
            texture,
            view,
            #[cfg(test)]
            direction_time,
            #[cfg(test)]
            invariant_drift,
            #[cfg(test)]
            metadata,
            bind_group,
        }
    }

    pub(crate) fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceTarget,
        pixels: TracePixels,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        self.set_pixel_offset(queue, pixels.start());
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("interactive trace pass"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &target.bind_group, &[]);
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePass.html#method.dispatch_workgroups
        pass.dispatch_workgroups(
            1,
            pixels
                .len()
                .div_ceil(TRACE_WORKGROUP_WIDTH * TRACE_WORKGROUP_HEIGHT),
            1,
        );
    }

    fn set_pixel_offset(&self, queue: &wgpu::Queue, pixel: u32) {
        let dispatch = TraceDispatch {
            pixels: [pixel, 0, 0, 0],
        };
        // Small queue writes are staged immediately and execute before the following submission.
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html#method.write_buffer
        queue.write_buffer(&self.dispatch, 0, bytemuck::bytes_of(&dispatch));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TracePixels {
    start: u32,
    end: u32,
}

impl TracePixels {
    #[cfg(test)]
    pub(crate) const fn all(extent: RenderExtent) -> Self {
        Self {
            start: 0,
            end: extent.width() * extent.height(),
        }
    }

    pub(crate) const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    pub(crate) const fn start(self) -> u32 {
        self.start
    }

    pub(crate) const fn len(self) -> u32 {
        self.end - self.start
    }
}

#[cfg(test)]
pub const fn production_shader_source() -> &'static str {
    TRACE_SHADER
}

pub struct TraceTarget {
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "keeps the storage texture alive for its view")
    )]
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    #[cfg(test)]
    direction_time: wgpu::Buffer,
    #[cfg(test)]
    invariant_drift: wgpu::Buffer,
    #[cfg(test)]
    metadata: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl TraceTarget {
    pub(crate) const fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    #[cfg(test)]
    pub(crate) const fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    #[cfg(test)]
    pub(crate) const fn record_planes(&self) -> [&wgpu::Buffer; 3] {
        [&self.direction_time, &self.invariant_drift, &self.metadata]
    }
}

fn create_record_plane(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

pub fn trace_record_plane_size(extent: RenderExtent) -> u64 {
    u64::from(extent.width())
        .saturating_mul(u64::from(extent.height()))
        .saturating_mul(TRACE_RECORD_PLANE_ELEMENT_SIZE)
}

const fn size_of<T>() -> u64 {
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
#[error("unknown interactive trace termination discriminant {0}")]
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

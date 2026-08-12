use gravlume_domain::{Observation, ValidationReport};
use num_traits::ToPrimitive as _;
use wgpu::util::DeviceExt as _;

use crate::extent::RenderExtent;

pub const TRACE_SHADER: &str = include_str!("shaders/trace.wgsl");
pub const TRACE_RECORD_SIZE: u64 = 48;

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct TraceUniforms {
    pub spacetime: [f32; 4],
    pub observer_event: [f32; 4],
    pub observer_velocity: [f32; 4],
    pub image_right: [f32; 4],
    pub image_up: [f32; 4],
    pub arrival: [f32; 4],
    pub projection_policy: [f32; 4],
    pub integration: [f32; 4],
    pub viewport: [u32; 4],
}

impl TraceUniforms {
    pub fn from_observation(observation: &Observation) -> Result<Self, TraceInputError> {
        let scene = observation.scene();
        let spacetime = *scene.spacetime();
        let mass = spacetime.mass_m();
        let projection = *observation.projection();
        let frame = scene.observer_frame();
        let sample = projection
            .sample(0, 0, 0.5, 0.5)
            .map_err(TraceInputError::DomainInvariant)?;
        let observer_frequency = observation
            .initial_ray(sample)
            .map_err(TraceInputError::DomainInvariant)?
            .observer_frequency();
        let horizon = spacetime
            .outer_horizon_radius()
            .map_or(-1.0, |radius| radius / mass);

        Ok(Self {
            spacetime: pack4(
                [
                    1.0,
                    spacetime.spin_m() / mass,
                    spacetime.charge_m() / mass,
                    horizon,
                ],
                "spacetime",
            )?,
            observer_event: pack4(
                scene.observer_event().to_txyz().map(|value| value / mass),
                "observer_event",
            )?,
            observer_velocity: pack4(frame.four_velocity_txyz(), "observer_velocity")?,
            image_right: pack4(frame.image_right_txyz(), "image_right")?,
            image_up: pack4(frame.image_up_txyz(), "image_up")?,
            arrival: pack4(frame.arrival_direction_txyz(), "arrival")?,
            projection_policy: pack4(
                [
                    (projection.vertical_fov().radians() * 0.5).tan(),
                    200.0,
                    f64::from(f32::from_bits(0x2b80_0000)),
                    observer_frequency,
                ],
                "projection_policy",
            )?,
            integration: [0.01, 0.005, 0.5, 0.05],
            viewport: [
                projection.width().get(),
                projection.height().get(),
                2_048,
                u32::from(spacetime.outer_horizon_radius().is_some()),
            ],
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
    #[error("observation field {field} is not representable by the interactive f32 contract")]
    NotRepresentable { field: &'static str },
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct TraceRecord {
    pub direction_time: [f32; 4],
    pub invariant_drift: [f32; 4],
    pub metadata: [u32; 4],
}

pub struct TraceCompute {
    pipeline: wgpu::ComputePipeline,
    #[cfg(test)]
    initial_ray_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
}

impl TraceCompute {
    pub fn new(device: &wgpu::Device, observation: &Observation) -> Result<Self, TraceInputError> {
        let uniforms = TraceUniforms::from_observation(observation)?;
        Ok(Self::from_uniforms(device, uniforms))
    }

    #[cfg(test)]
    pub fn new_for_initial_rays(
        device: &wgpu::Device,
        observation: &Observation,
        subpixel: [f32; 2],
    ) -> Result<Self, TraceInputError> {
        let mut uniforms = TraceUniforms::from_observation(observation)?;
        uniforms.integration[..2].copy_from_slice(&subpixel);
        Ok(Self::from_uniforms(device, uniforms))
    }

    fn from_uniforms(device: &wgpu::Device, uniforms: TraceUniforms) -> Self {
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/util/trait.DeviceExt.html#tymethod.create_buffer_init
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("interactive trace uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
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
                        min_binding_size: wgpu::BufferSize::new(TRACE_RECORD_SIZE),
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
            source: wgpu::ShaderSource::Wgsl(TRACE_SHADER.into()),
        });
        let pipeline = create_pipeline(device, &pipeline_layout, &shader, "trace_scene");
        #[cfg(test)]
        let initial_ray_pipeline =
            create_pipeline(device, &pipeline_layout, &shader, "initial_ray_contract");

        Self {
            pipeline,
            #[cfg(test)]
            initial_ray_pipeline,
            bind_group_layout,
            uniforms,
        }
    }

    pub fn create_target(&self, device: &wgpu::Device, extent: RenderExtent) -> TraceTarget {
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
        let records = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interactive geometric trace records"),
            size: trace_buffer_size(extent),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
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
                    resource: records.as_entire_binding(),
                },
            ],
        });
        TraceTarget {
            extent,
            #[cfg(test)]
            texture,
            view,
            #[cfg(test)]
            records,
            bind_group,
        }
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceTarget,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        Self::encode_with_pipeline(encoder, target, timestamp_writes, &self.pipeline);
    }

    #[cfg(test)]
    pub fn encode_initial_rays(&self, encoder: &mut wgpu::CommandEncoder, target: &TraceTarget) {
        Self::encode_with_pipeline(encoder, target, None, &self.initial_ray_pipeline);
    }

    fn encode_with_pipeline(
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceTarget,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
        pipeline: &wgpu::ComputePipeline,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("interactive trace pass"),
            timestamp_writes,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &target.bind_group, &[]);
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePass.html#method.dispatch_workgroups
        pass.dispatch_workgroups(
            target.extent.width().div_ceil(8),
            target.extent.height().div_ceil(8),
            1,
        );
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &'static str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

pub struct TraceTarget {
    extent: RenderExtent,
    #[cfg(test)]
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    #[cfg(test)]
    records: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl TraceTarget {
    pub const fn extent(&self) -> RenderExtent {
        self.extent
    }

    pub const fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    #[cfg(test)]
    pub const fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    #[cfg(test)]
    pub const fn records(&self) -> &wgpu::Buffer {
        &self.records
    }
}

fn trace_buffer_size(extent: RenderExtent) -> u64 {
    u64::from(extent.width()) * u64::from(extent.height()) * TRACE_RECORD_SIZE
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

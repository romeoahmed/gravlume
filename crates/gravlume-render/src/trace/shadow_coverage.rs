//! Selective subpixel refinement for the capture/escape shadow boundary.

use wgpu::util::DeviceExt as _;

use super::{TraceUniforms, shader, size_of};
use crate::extent::RenderExtent;

const CLASSIFY_WORKGROUP_AXIS: u32 = 8;
const REFINE_WORKGROUP_WIDTH: u32 = 64;
const EDGE_PIXEL_BYTES: u64 = size_of::<u32>();
const CONTROL_BYTES: u64 = size_of::<ShadowControl>();

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(8))]
struct ShadowControl {
    count: u32,
    capacity: u32,
    padding: [u32; 2],
}

const _: () = {
    assert!(std::mem::size_of::<ShadowControl>() == 16);
    assert!(std::mem::align_of::<ShadowControl>() == 8);
    assert!(std::mem::offset_of!(ShadowControl, count) == 0);
    assert!(std::mem::offset_of!(ShadowControl, capacity) == 4);
    assert!(std::mem::offset_of!(ShadowControl, padding) == 8);
};

pub struct ShadowCoverage {
    classify_pipeline: wgpu::ComputePipeline,
    refine_pipeline: wgpu::ComputePipeline,
    classify_layout: wgpu::BindGroupLayout,
    refine_layout: wgpu::BindGroupLayout,
    uniform_bind_group: wgpu::BindGroup,
}

pub struct ShadowTarget {
    pub extent: RenderExtent,
    capacity: u32,
    pub control: wgpu::Buffer,
    classify_bind_group: wgpu::BindGroup,
    refine_bind_group: wgpu::BindGroup,
}

impl ShadowCoverage {
    pub fn new(device: &wgpu::Device, uniforms: &wgpu::Buffer) -> Self {
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow refinement uniform layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(size_of::<TraceUniforms>()),
                },
                count: None,
            }],
        });
        let classify_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow edge classification layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_buffer_layout(1, false, EDGE_PIXEL_BYTES),
                storage_buffer_layout(2, false, CONTROL_BYTES),
            ],
        });
        let refine_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow edge refinement layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                storage_buffer_layout(1, true, EDGE_PIXEL_BYTES),
                storage_buffer_layout(2, true, CONTROL_BYTES),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selective shadow coverage shader"),
            source: wgpu::ShaderSource::Wgsl(shader::shadow_coverage().into()),
        });
        let classify_pipeline = create_pipeline(
            device,
            &shader,
            "classify shadow edges",
            "classify_shadow_edges",
            &[None, Some(&classify_layout)],
        );
        let refine_pipeline = create_pipeline(
            device,
            &shader,
            "refine shadow edges",
            "refine_shadow_edges",
            &[Some(&uniform_layout), None, Some(&refine_layout)],
        );
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow refinement uniforms"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });

        Self {
            classify_pipeline,
            refine_pipeline,
            classify_layout,
            refine_layout,
            uniform_bind_group,
        }
    }

    pub fn create_target(
        &self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
        extent: RenderExtent,
    ) -> ShadowTarget {
        let capacity = edge_capacity(extent);
        let edge_pixels = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow edge pixels"),
            size: u64::from(capacity) * EDGE_PIXEL_BYTES,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let control_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        #[cfg(test)]
        let control_usage = control_usage | wgpu::BufferUsages::COPY_SRC;
        let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shadow refinement control"),
            contents: bytemuck::bytes_of(&ShadowControl {
                count: 0,
                capacity,
                padding: [0; 2],
            }),
            usage: control_usage,
        });
        let classify_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow edge classification resources"),
            layout: &self.classify_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edge_pixels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: control.as_entire_binding(),
                },
            ],
        });
        let refine_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow edge refinement resources"),
            layout: &self.refine_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edge_pixels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: control.as_entire_binding(),
                },
            ],
        });

        ShadowTarget {
            extent,
            capacity,
            control,
            classify_bind_group,
            refine_bind_group,
        }
    }

    pub fn reset_control(encoder: &mut wgpu::CommandEncoder, target: &ShadowTarget) {
        encoder.clear_buffer(&target.control, 0, Some(size_of::<u32>()));
    }

    pub fn encode<'pass>(
        &'pass self,
        pass: &mut wgpu::ComputePass<'pass>,
        target: &'pass ShadowTarget,
    ) {
        // Separate dispatches keep sampled classification and write-only refinement in distinct
        // resource-usage scopes.
        pass.set_pipeline(&self.classify_pipeline);
        pass.set_bind_group(1, &target.classify_bind_group, &[]);
        pass.dispatch_workgroups(
            target.extent.width().div_ceil(CLASSIFY_WORKGROUP_AXIS),
            target.extent.height().div_ceil(CLASSIFY_WORKGROUP_AXIS),
            1,
        );

        pass.set_pipeline(&self.refine_pipeline);
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_bind_group(2, &target.refine_bind_group, &[]);
        pass.dispatch_workgroups(target.capacity.div_ceil(REFINE_WORKGROUP_WIDTH), 1, 1);
    }
}

pub fn scratch_bytes(extent: RenderExtent) -> u64 {
    u64::from(edge_capacity(extent)) * EDGE_PIXEL_BYTES + CONTROL_BYTES
}

fn edge_capacity(extent: RenderExtent) -> u32 {
    let total_pixels = u64::from(extent.width()) * u64::from(extent.height());
    let perimeter_bound =
        8_u64.saturating_mul(u64::from(extent.width()).saturating_add(u64::from(extent.height())));
    u32::try_from(total_pixels.min(perimeter_bound)).unwrap_or(u32::MAX)
}

const fn storage_buffer_layout(
    binding: u32,
    read_only: bool,
    minimum_size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(minimum_size),
        },
        count: None,
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    label: &'static str,
    entry_point: &'static str,
    bind_group_layouts: &[Option<&wgpu::BindGroupLayout>],
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts,
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

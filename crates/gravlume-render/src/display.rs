use wgpu::util::DeviceExt as _;

use crate::{capabilities::SurfaceSelection, extent::RenderExtent};

/// egui-wgpu's non-sRGB target path writes gamma-encoded, premultiplied colors.
/// Keeping that target separate lets the final pass decode and composite it at SDR reference white.
/// Source: <https://docs.rs/crate/egui-wgpu/0.36.1/source/src/renderer.rs>
pub const UI_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct OutputUniforms {
    mapping: [f32; 4],
}

pub struct DisplayPipeline {
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    presentation_pipeline: wgpu::RenderPipeline,
    publish_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    publish_bind_group_layout: wgpu::BindGroupLayout,
    output_uniforms: wgpu::Buffer,
}

pub struct DisplayTarget {
    ui_view: wgpu::TextureView,
}

pub struct PublishedScene {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

pub struct CandidatePublication {
    bind_group: wgpu::BindGroup,
}

impl DisplayPipeline {
    pub(crate) fn new(device: &wgpu::Device, selection: SurfaceSelection) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("output bind group layout"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<OutputUniforms>()),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("output pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene and UI output shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
        });
        let output_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("output mapping uniforms"),
            contents: bytemuck::bytes_of(&OutputUniforms::from(selection)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let presentation_pipeline =
            Self::create_render_pipeline(device, &pipeline_layout, &shader, selection);
        let publish_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("candidate publication bind group layout"),
                entries: &[texture_entry(0)],
            });
        let publish_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("candidate publication pipeline layout"),
                bind_group_layouts: &[Some(&publish_bind_group_layout)],
                immediate_size: 0,
            });
        let publish_pipeline = Self::create_pipeline(
            device,
            &publish_pipeline_layout,
            &shader,
            "publish_complete_candidate",
            wgpu::TextureFormat::Rgba16Float,
        );

        Self {
            shader,
            pipeline_layout,
            presentation_pipeline,
            publish_pipeline,
            bind_group_layout,
            publish_bind_group_layout,
            output_uniforms,
        }
    }

    fn create_render_pipeline(
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        selection: SurfaceSelection,
    ) -> wgpu::RenderPipeline {
        Self::create_pipeline(
            device,
            pipeline_layout,
            shader,
            selection.fragment_entry(),
            selection.format(),
        )
    }

    fn create_pipeline(
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        fragment_entry: &'static str,
        format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("surface output pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("fullscreen_triangle"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fragment_entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    pub(crate) fn create_presentation_pipeline(
        &self,
        device: &wgpu::Device,
        selection: SurfaceSelection,
    ) -> wgpu::RenderPipeline {
        Self::create_render_pipeline(device, &self.pipeline_layout, &self.shader, selection)
    }

    pub(crate) fn install_output(
        &mut self,
        queue: &wgpu::Queue,
        selection: SurfaceSelection,
        pipeline: Option<wgpu::RenderPipeline>,
    ) {
        queue.write_buffer(
            &self.output_uniforms,
            0,
            bytemuck::bytes_of(&OutputUniforms::from(selection)),
        );
        if let Some(pipeline) = pipeline {
            self.presentation_pipeline = pipeline;
        }
    }

    pub(crate) fn create_target(device: &wgpu::Device, extent: RenderExtent) -> DisplayTarget {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SDR gamma UI overlay"),
            size: texture_extent(extent),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: UI_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        DisplayTarget {
            ui_view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }

    pub(crate) fn create_published_scene(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &DisplayTarget,
        extent: RenderExtent,
    ) -> PublishedScene {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("complete published scene HDR"),
            size: texture_extent(extent),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let scene_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("initialize published scene"),
        });
        {
            let attachment = Some(wgpu::RenderPassColorAttachment {
                view: &scene_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            });
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("initialize published scene pass"),
                color_attachments: &[attachment],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        queue.submit([encoder.finish()]);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("published scene and UI output bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&target.ui_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.output_uniforms.as_entire_binding(),
                },
            ],
        });
        PublishedScene {
            texture,
            bind_group,
        }
    }

    pub(crate) fn bind_candidate(
        &self,
        device: &wgpu::Device,
        candidate: &wgpu::TextureView,
    ) -> CandidatePublication {
        CandidatePublication {
            bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("complete candidate publication bind group"),
                layout: &self.publish_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(candidate),
                }],
            }),
        }
    }

    pub(crate) fn encode_publication(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        published: &PublishedScene,
        candidate: &CandidatePublication,
    ) {
        let view = published
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let attachment = Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("publish complete candidate pass"),
            color_attachments: &[attachment],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.publish_pipeline);
        pass.set_bind_group(0, &candidate.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    pub(crate) fn encode_presentation(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        surface_view: &wgpu::TextureView,
        scene: &PublishedScene,
    ) {
        let color_attachment = Some(wgpu::RenderPassColorAttachment {
            view: surface_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("surface output pass"),
            color_attachments: &[color_attachment],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.presentation_pipeline);
        pass.set_bind_group(0, &scene.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl DisplayTarget {
    pub(crate) const fn ui_view(&self) -> &wgpu::TextureView {
        &self.ui_view
    }
}

impl From<SurfaceSelection> for OutputUniforms {
    fn from(selection: SurfaceSelection) -> Self {
        Self {
            mapping: [
                selection.tone_map_headroom(),
                selection.reference_white_scale(),
                0.0,
                0.0,
            ],
        }
    }
}

const fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

const fn texture_extent(extent: RenderExtent) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: extent.width(),
        height: extent.height(),
        depth_or_array_layers: 1,
    }
}

const fn size_of<T>() -> u64 {
    std::mem::size_of::<T>() as u64
}

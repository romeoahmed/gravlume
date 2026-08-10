fn fragment_entry(format: wgpu::TextureFormat) -> &'static str {
    if format.is_srgb() {
        "display_to_linear_target"
    } else {
        "display_to_gamma_target"
    }
}

pub struct DisplayPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl DisplayPipeline {
    pub(crate) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Phase 0 display shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("neutral HDR display pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("fullscreen_triangle"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fragment_entry(format)),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    pub(crate) fn bind_scene(
        &self,
        device: &wgpu::Device,
        scene_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display scene HDR bind group"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(scene_view),
            }],
        })
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        surface_view: &wgpu::TextureView,
        scene_bind_group: &wgpu::BindGroup,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
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
            label: Some("neutral HDR display pass"),
            color_attachments: &[color_attachment],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::DisplayPipeline;

    const WIDTH: u32 = 3;
    const PADDED_BYTES_PER_ROW: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

    #[test]
    fn display_transfer_is_equivalent_for_srgb_and_linear_targets() {
        let hardware_encoded = render_display(wgpu::TextureFormat::Rgba8UnormSrgb);
        let shader_encoded = render_display(wgpu::TextureFormat::Rgba8Unorm);

        for pixels in [&hardware_encoded, &shader_encoded] {
            assert_eq!(&pixels[0..4], &[255, 0, 255, 255]);
            assert_eq!(&pixels[4..8], &[255, 0, 255, 255]);
            assert!(pixels[8] > pixels[9]);
            assert!(pixels[9] > pixels[10]);
            assert_eq!(pixels[11], 255);
        }

        assert!(
            hardware_encoded
                .iter()
                .zip(&shader_encoded)
                .all(|(hardware, shader)| hardware.abs_diff(*shader) <= 1),
            "hardware and shader sRGB encoding diverged: {hardware_encoded:?} != {shader_encoded:?}"
        );
    }

    fn render_display(format: wgpu::TextureFormat) -> Vec<u8> {
        let gpu = crate::test_gpu::native_gpu();
        let scene = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("display diagnostic scene input"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let half_words = [
            0xbc00_u16, 0, 0, 0x3c00, 0x7e00, 0, 0, 0x3c00, 0x4400, 0x3c00, 0, 0x3c00,
        ];
        let scene_bytes: Vec<u8> = half_words.into_iter().flat_map(u16::to_le_bytes).collect();
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &scene,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &scene_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH * 8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let output = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("display diagnostic output"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display diagnostic readback"),
            size: u64::from(PADDED_BYTES_PER_ROW),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let scene_view = scene.create_view(&wgpu::TextureViewDescriptor::default());
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
        let display = DisplayPipeline::new(&gpu.device, format);
        let bind_group = display.bind_scene(&gpu.device, &scene_view);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("display diagnostic encoder"),
            });
        display.encode(&mut encoder, &output_view, &bind_group, None);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(PADDED_BYTES_PER_ROW),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let submission = gpu.queue.submit([encoder.finish()]);
        let mapped = crate::test_gpu::read_buffer(&readback, submission);
        mapped[..WIDTH as usize * 4].to_vec()
    }
}

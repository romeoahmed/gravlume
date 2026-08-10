pub fn fragment_entry(format: wgpu::TextureFormat) -> &'static str {
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
    use super::{DisplayPipeline, fragment_entry};

    #[test]
    fn display_entry_matches_surface_transfer_responsibility() {
        assert_eq!(
            fragment_entry(wgpu::TextureFormat::Bgra8UnormSrgb),
            "display_to_linear_target"
        );
        assert_eq!(
            fragment_entry(wgpu::TextureFormat::Bgra8Unorm),
            "display_to_gamma_target"
        );
    }

    #[test]
    fn display_marks_negative_and_non_finite_radiance_as_diagnostic_magenta() {
        let pixels = pollster::block_on(probe_display());

        assert_eq!(&pixels[0..4], &[255, 0, 255, 255]);
        assert_eq!(&pixels[4..8], &[255, 0, 255, 255]);
        assert!(pixels[8] > pixels[9]);
        assert!(pixels[9] > pixels[10]);
        assert_eq!(pixels[11], 255);
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the end-to-end GPU probe keeps resource lifetimes and submission order linear"
    )]
    async fn probe_display() -> Vec<u8> {
        const WIDTH: u32 = 3;
        const PADDED_BYTES_PER_ROW: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = crate::native_backends();
        let instance = wgpu::Instance::new(descriptor);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .expect("native adapter is available");
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits::default()
            .using_resolution(adapter_limits.clone())
            .using_alignment(adapter_limits);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("display diagnostic contract device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                ..Default::default()
            })
            .await
            .expect("display contract device request succeeds");
        let scene = device.create_texture(&wgpu::TextureDescriptor {
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
        queue.write_texture(
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
        let scene_view = scene.create_view(&wgpu::TextureViewDescriptor::default());
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("display diagnostic output"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display diagnostic readback"),
            size: u64::from(PADDED_BYTES_PER_ROW),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let display = DisplayPipeline::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        let bind_group = display.bind_scene(&device, &scene_view);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
        let submission = queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("display readback poll succeeds");
        receiver
            .recv()
            .expect("display map callback runs")
            .expect("display readback maps");
        let mapped = readback
            .get_mapped_range(..)
            .expect("display mapped range is available");
        let pixels = mapped[..WIDTH as usize * 4].to_vec();
        drop(mapped);
        readback.unmap();
        pixels
    }
}

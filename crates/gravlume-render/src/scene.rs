use crate::extent::RenderExtent;

const WORKGROUP_WIDTH: u32 = 8;
const WORKGROUP_HEIGHT: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchGrid {
    x: u32,
    y: u32,
}

impl DispatchGrid {
    pub(crate) const fn for_extent(extent: RenderExtent) -> Self {
        Self {
            x: extent.width().div_ceil(WORKGROUP_WIDTH),
            y: extent.height().div_ceil(WORKGROUP_HEIGHT),
        }
    }

    pub(crate) const fn x(self) -> u32 {
        self.x
    }

    pub(crate) const fn y(self) -> u32 {
        self.y
    }
}

pub struct SceneCompute {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl SceneCompute {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene HDR bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene HDR pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Phase 0 scene compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scene.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scene HDR compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    pub(crate) fn create_target(&self, device: &wgpu::Device, extent: RenderExtent) -> SceneTarget {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-linear HDR intermediate"),
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene HDR bind group"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });

        SceneTarget {
            extent,
            #[cfg(test)]
            texture,
            view,
            bind_group,
        }
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &SceneTarget,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scene HDR compute pass"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &target.bind_group, &[]);
        let grid = DispatchGrid::for_extent(target.extent);
        pass.dispatch_workgroups(grid.x(), grid.y(), 1);
    }
}

pub struct SceneTarget {
    extent: RenderExtent,
    #[cfg(test)]
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

impl SceneTarget {
    pub(crate) const fn extent(&self) -> RenderExtent {
        self.extent
    }

    #[cfg(test)]
    const fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub(crate) const fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
}

#[cfg(test)]
const fn readback_row_layout(extent: RenderExtent) -> (u32, u32) {
    let unpadded = extent.width() * 8;
    let padded =
        unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    (unpadded, padded)
}

#[cfg(test)]
fn encode_probe(
    device: &wgpu::Device,
    compute: &SceneCompute,
    target: &SceneTarget,
    readback: &wgpu::Buffer,
    padded_bytes_per_row: u32,
) -> wgpu::CommandBuffer {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Phase 0 headless encoder"),
    });
    compute.encode(&mut encoder, target, None);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: target.texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(target.extent().height()),
            },
        },
        wgpu::Extent3d {
            width: target.extent().width(),
            height: target.extent().height(),
            depth_or_array_layers: 1,
        },
    );
    encoder.finish()
}

#[cfg(test)]
fn remove_row_padding(
    mapped: &[u8],
    extent: RenderExtent,
    unpadded_bytes_per_row: u32,
    padded_bytes_per_row: u32,
) -> Vec<u8> {
    let unpadded = usize::try_from(unpadded_bytes_per_row).expect("row length fits usize");
    let padded = usize::try_from(padded_bytes_per_row).expect("padded row length fits usize");
    let height = usize::try_from(extent.height()).expect("height fits usize");
    let mut bytes = Vec::with_capacity(unpadded * height);
    for row in mapped.chunks_exact(padded) {
        bytes.extend_from_slice(&row[..unpadded]);
    }
    bytes
}

#[cfg(test)]
fn render_scene(extent: RenderExtent) -> Vec<u8> {
    let gpu = crate::test_gpu::native_gpu();
    let compute = SceneCompute::new(&gpu.device);
    let target = compute.create_target(&gpu.device, extent);
    let (unpadded_bytes_per_row, padded_bytes_per_row) = readback_row_layout(extent);
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Phase 0 headless HDR readback"),
        size: u64::from(padded_bytes_per_row) * u64::from(extent.height()),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let commands = encode_probe(
        &gpu.device,
        &compute,
        &target,
        &readback,
        padded_bytes_per_row,
    );
    let submission = gpu.queue.submit([commands]);
    let mapped = gpu.read_buffer(&readback, submission);
    remove_row_padding(
        &mapped,
        extent,
        unpadded_bytes_per_row,
        padded_bytes_per_row,
    )
}

#[cfg(test)]
mod tests {
    use super::render_scene;
    use crate::extent::RenderExtent;

    #[test]
    fn compute_overwrites_every_pixel_at_workgroup_boundaries() {
        for (width, height) in [(1, 1), (8, 8), (9, 8), (17, 9)] {
            let extent = RenderExtent::new(width, height).expect("test extent is nonzero");
            let bytes = render_scene(extent);
            let pixels = bytes.chunks_exact(8);

            assert!(pixels.remainder().is_empty(), "{width}x{height}");
            assert_eq!(pixels.len(), (width * height) as usize, "{width}x{height}");
            assert!(
                pixels
                    .clone()
                    .all(|pixel| u16::from_le_bytes([pixel[6], pixel[7]]) == 0x3c00),
                "compute left an unwritten pixel at {width}x{height}"
            );
            assert!(
                pixels.clone().any(|pixel| {
                    let red = u16::from_le_bytes([pixel[0], pixel[1]]);
                    red > 0x3c00 && red < 0x7c00
                }),
                "scene lost its HDR range at {width}x{height}"
            );
        }
    }
}

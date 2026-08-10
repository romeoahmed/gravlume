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
#[derive(Debug, thiserror::Error)]
enum ProbeError {
    #[error("no native GPU adapter satisfies the Phase 0 request: {0}")]
    Adapter(#[from] wgpu::RequestAdapterError),
    #[error("the selected adapter is not WebGPU compliant")]
    Downlevel,
    #[error("adapter is missing Phase 0 features: {0:?}")]
    MissingFeatures(wgpu::Features),
    #[error("failed to request the Phase 0 device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("buffer map failed")]
    Map,
    #[error("GPU poll failed: {0}")]
    Poll(#[from] wgpu::PollError),
}

#[cfg(test)]
struct HeadlessProbe {
    extent: RenderExtent,
    bytes: Vec<u8>,
}

#[cfg(test)]
impl HeadlessProbe {
    const BYTES_PER_PIXEL: usize = 8;

    const fn extent(&self) -> RenderExtent {
        self.extent
    }

    fn every_alpha_is_one(&self) -> bool {
        self.bytes
            .chunks_exact(Self::BYTES_PER_PIXEL)
            .all(|pixel| u16::from_le_bytes([pixel[6], pixel[7]]) == 0x3c00)
    }

    fn has_channel_above_one(&self) -> bool {
        self.bytes.chunks_exact(Self::BYTES_PER_PIXEL).any(|pixel| {
            let red = u16::from_le_bytes([pixel[0], pixel[1]]);
            red > 0x3c00 && red < 0x7c00
        })
    }
}

#[cfg(test)]
async fn request_probe_device() -> Result<(wgpu::Device, wgpu::Queue), ProbeError> {
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
        .await?;

    if !adapter.get_downlevel_capabilities().is_webgpu_compliant() {
        return Err(ProbeError::Downlevel);
    }
    let required_features = crate::capabilities::BASELINE_FEATURES;
    let missing_features = required_features - adapter.features();
    if !missing_features.is_empty() {
        return Err(ProbeError::MissingFeatures(missing_features));
    }
    let adapter_limits = adapter.limits();
    let required_limits = wgpu::Limits::default()
        .using_resolution(adapter_limits.clone())
        .using_alignment(adapter_limits);
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Phase 0 headless contract device"),
            required_features,
            required_limits,
            ..Default::default()
        })
        .await
        .map_err(ProbeError::from)
}

#[cfg(test)]
fn readback_row_layout(extent: RenderExtent) -> (u32, u32) {
    let bytes_per_pixel =
        u32::try_from(HeadlessProbe::BYTES_PER_PIXEL).expect("HDR pixel byte width fits u32");
    let unpadded = extent.width() * bytes_per_pixel;
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
fn readback_bytes(
    device: &wgpu::Device,
    readback: &wgpu::Buffer,
    submission: wgpu::SubmissionIndex,
    extent: RenderExtent,
    unpadded_bytes_per_row: u32,
    padded_bytes_per_row: u32,
) -> Result<Vec<u8>, ProbeError> {
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;
    receiver
        .recv()
        .map_err(|_| ProbeError::Map)?
        .map_err(|_| ProbeError::Map)?;

    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .map_err(|_| ProbeError::Map)?;
    let unpadded = usize::try_from(unpadded_bytes_per_row).expect("row length fits usize");
    let padded = usize::try_from(padded_bytes_per_row).expect("padded row length fits usize");
    let height = usize::try_from(extent.height()).expect("height fits usize");
    let mut bytes = Vec::with_capacity(unpadded * height);
    for row in mapped.chunks_exact(padded) {
        bytes.extend_from_slice(&row[..unpadded]);
    }
    drop(mapped);
    readback.unmap();
    Ok(bytes)
}

#[cfg(test)]
async fn probe_headless(extent: RenderExtent) -> Result<HeadlessProbe, ProbeError> {
    let (device, queue) = request_probe_device().await?;

    let compute = SceneCompute::new(&device);
    let target = compute.create_target(&device, extent);
    let (unpadded_bytes_per_row, padded_bytes_per_row) = readback_row_layout(extent);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Phase 0 headless HDR readback"),
        size: u64::from(padded_bytes_per_row) * u64::from(extent.height()),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let commands = encode_probe(&device, &compute, &target, &readback, padded_bytes_per_row);
    let submission = queue.submit([commands]);
    let bytes = readback_bytes(
        &device,
        &readback,
        submission,
        extent,
        unpadded_bytes_per_row,
        padded_bytes_per_row,
    )?;

    Ok(HeadlessProbe { extent, bytes })
}

#[cfg(test)]
mod tests {
    use super::{DispatchGrid, probe_headless};
    use crate::extent::RenderExtent;

    #[test]
    fn dispatch_grid_ceiling_divides_edge_workgroups() {
        let cases = [
            ((1, 1), (1, 1)),
            ((8, 8), (1, 1)),
            ((9, 8), (2, 1)),
            ((1279, 719), (160, 90)),
        ];

        for ((width, height), expected) in cases {
            let extent = RenderExtent::new(width, height).expect("test extent is nonzero");
            let grid = DispatchGrid::for_extent(extent);
            assert_eq!((grid.x(), grid.y()), expected);
        }
    }

    #[test]
    fn headless_compute_writes_scene_linear_hdr_for_odd_extent() {
        let extent = RenderExtent::new(17, 9).expect("test extent is nonzero");

        let probe = pollster::block_on(probe_headless(extent))
            .expect("the native adapter should execute the Phase 0 compute shader");

        assert_eq!(probe.extent(), extent);
        assert!(probe.every_alpha_is_one());
        assert!(probe.has_channel_above_one());
    }
}

use gravlume_domain::Observation;

use crate::{
    extent::RenderExtent,
    trace::{TraceCompute, trace_record_plane_size},
};

const RECORD_FIELD_SIZE: usize = std::mem::size_of::<[u32; 4]>();
const RECORD_FIELD_COUNT: u64 = 3;

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct TraceRecord {
    pub direction_time: [f32; 4],
    pub invariant_drift: [f32; 4],
    pub metadata: [u32; 4],
}

#[derive(Clone, Copy)]
pub enum TraceEntryPoint {
    InitialRay { subpixel: [f32; 2] },
    Trace,
}

pub struct TraceCapture {
    pub records: Vec<TraceRecord>,
    hdr: Vec<u8>,
}

impl TraceCapture {
    pub fn hdr_pixel(&self, index: usize) -> [u8; 8] {
        let start = index * 8;
        self.hdr[start..start + 8]
            .try_into()
            .expect("HDR pixel contains eight bytes")
    }
}

pub fn render_trace_for_test(
    observation: &Observation,
    entry_point: TraceEntryPoint,
) -> TraceCapture {
    let gpu = crate::test_gpu::native_gpu();
    let extent = RenderExtent::new(
        observation.projection().width().get(),
        observation.projection().height().get(),
    )
    .expect("validated observation extent is nonzero");
    let compute = match entry_point {
        TraceEntryPoint::InitialRay { subpixel } => {
            TraceCompute::for_initial_ray_capture(&gpu.device, observation, subpixel)
        }
        TraceEntryPoint::Trace => TraceCompute::new(&gpu.device, observation),
    }
    .expect("observation packs for GPU");
    let target = compute.create_target(&gpu.device, extent);
    let plane_size = trace_record_plane_size(extent);
    let record_bytes = plane_size * RECORD_FIELD_COUNT;
    let hdr_offset = record_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.into())
        * u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let (unpadded_bytes_per_row, padded_bytes_per_row) = readback_row_layout(extent);
    let hdr_bytes = u64::from(padded_bytes_per_row) * u64::from(extent.height());
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("headless trace structured readback"),
        size: hdr_offset + hdr_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless trace encoder"),
        });
    match entry_point {
        TraceEntryPoint::InitialRay { .. } => compute.encode_initial_rays(&mut encoder, &target),
        TraceEntryPoint::Trace => compute.encode(&mut encoder, &target, None),
    }
    for (plane_index, plane) in target.record_planes().into_iter().enumerate() {
        encoder.copy_buffer_to_buffer(
            plane,
            0,
            &readback,
            u64::try_from(plane_index).expect("record plane index fits u64") * plane_size,
            plane_size,
        );
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: target.texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: hdr_offset,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(extent.height()),
            },
        },
        wgpu::Extent3d {
            width: extent.width(),
            height: extent.height(),
            depth_or_array_layers: 1,
        },
    );
    let submission = gpu.queue.submit([encoder.finish()]);
    let mapped = gpu.read_buffer(&readback, submission);
    let record_end = usize::try_from(record_bytes).expect("test trace size fits usize");
    let plane_size = usize::try_from(plane_size).expect("test record plane size fits usize");
    let direction_time = mapped[..plane_size].chunks_exact(RECORD_FIELD_SIZE);
    let invariant_drift = mapped[plane_size..plane_size * 2].chunks_exact(RECORD_FIELD_SIZE);
    let metadata = mapped[plane_size * 2..record_end].chunks_exact(RECORD_FIELD_SIZE);
    let records = direction_time
        .zip(invariant_drift)
        .zip(metadata)
        .map(
            |((direction_time, invariant_drift), metadata)| TraceRecord {
                direction_time: bytemuck::pod_read_unaligned(direction_time),
                invariant_drift: bytemuck::pod_read_unaligned(invariant_drift),
                metadata: bytemuck::pod_read_unaligned(metadata),
            },
        )
        .collect();
    let hdr_start = usize::try_from(hdr_offset).expect("test HDR offset fits usize");
    let hdr = remove_row_padding(
        &mapped[hdr_start..],
        extent,
        unpadded_bytes_per_row,
        padded_bytes_per_row,
    );
    TraceCapture { records, hdr }
}

const fn readback_row_layout(extent: RenderExtent) -> (u32, u32) {
    let unpadded = extent.width() * 8;
    let padded =
        unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    (unpadded, padded)
}

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

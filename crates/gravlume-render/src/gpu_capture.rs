use gravlume_domain::{ImageSample, Observation};

use crate::{
    extent::RenderExtent,
    trace::{SampleInspection, TileRegion, TracePipeline, tile_grid, trace_record_plane_size},
};

const RECORD_FIELD_SIZE: usize = std::mem::size_of::<[u32; 4]>();
const RECORD_FIELD_COUNT: u64 = 4;

#[derive(Clone, Copy, Debug)]
pub struct TraceRecord {
    pub source_time: [f32; 4],
    pub invariant_drift: [f32; 4],
    pub metadata: [u32; 4],
    pub event: [u32; 4],
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

pub fn capture_trace(observation: &Observation) -> TraceCapture {
    let gpu = crate::test_device::native_gpu();
    let trace = TracePipeline::for_trace_capture(&gpu.device, observation, [0.5, 0.5])
        .expect("observation packs for GPU");
    capture(gpu, observation, &trace, false)
}

pub fn capture_trace_sample(observation: &Observation, sample: ImageSample) -> TraceCapture {
    let gpu = crate::test_device::native_gpu();
    let trace = TracePipeline::for_trace_capture(&gpu.device, observation, sample.subpixel())
        .expect("observation packs for GPU");
    let tile = TileRegion::containing_pixel(sample.pixel());
    capture_region(gpu, observation, &trace, tile)
}

pub fn inspect_sample(observation: &Observation, sample: ImageSample) -> SampleInspection {
    let [pixel_x, pixel_y] = sample.pixel();
    let [subpixel_x, subpixel_y] = sample.subpixel();
    let sample = observation
        .view()
        .sample(pixel_x, pixel_y, subpixel_x, subpixel_y)
        .expect("inspection sample belongs to the observation view");
    let gpu = crate::test_device::native_gpu();
    let trace = TracePipeline::new(&gpu.device, observation)
        .expect("observation packs for bounded GPU inspection");
    trace.inspect_sample(gpu, observation_extent(observation), sample)
}

pub fn capture_surface_footprint_sample(
    observation: &Observation,
    sample: ImageSample,
) -> TraceCapture {
    assert!(
        sample
            .subpixel()
            .into_iter()
            .all(|coordinate| (0.25..=0.75).contains(&coordinate)),
        "surface footprint stencil must remain inside one pixel"
    );
    let gpu = crate::test_device::native_gpu();
    let trace =
        TracePipeline::for_surface_footprint_capture(&gpu.device, observation, sample.subpixel())
            .expect("surface observation packs for GPU footprint capture");
    let tile = TileRegion::containing_pixel(sample.pixel());
    capture_region(gpu, observation, &trace, tile)
}

pub fn capture_surface_transport_case(observation: &Observation) -> TraceCapture {
    let gpu = crate::test_device::native_gpu();
    let trace = TracePipeline::for_surface_transport_capture(&gpu.device, observation)
        .expect("surface observation packs for isolated GPU transport");
    capture(gpu, observation, &trace, false)
}

pub fn capture_accelerated_trace(observation: &Observation) -> TraceCapture {
    let gpu = crate::test_device::native_gpu();
    let trace = TracePipeline::for_accelerated_trace_capture(&gpu.device, observation)
        .expect("observation packs for GPU");
    capture(gpu, observation, &trace, false)
}

pub fn capture_refined_trace(observation: &Observation) -> TraceCapture {
    let gpu = crate::test_device::native_gpu();
    let trace = TracePipeline::for_trace_capture(&gpu.device, observation, [0.5, 0.5])
        .expect("observation packs for GPU");
    capture(gpu, observation, &trace, true)
}

pub fn capture_refined_edge_count(observation: &Observation, repetitions: u32) -> u32 {
    assert!(repetitions > 0, "at least one refinement is required");
    let gpu = crate::test_device::native_gpu();
    let trace = TracePipeline::for_trace_capture(&gpu.device, observation, [0.5, 0.5])
        .expect("observation packs for GPU");
    let extent = observation_extent(observation);
    let target = trace.create_target(&gpu.device, extent);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("repeated headless trace encoder"),
        });
    let tiles = TileRegion::all(extent);
    for _ in 0..repetitions {
        trace.encode(&gpu.queue, &mut encoder, &target, tiles);
    }
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shadow edge count readback"),
        size: size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        target.shadow_control(),
        0,
        &readback,
        0,
        size_of::<u32>() as u64,
    );
    let submission = gpu.queue.submit([encoder.finish()]);
    bytemuck::pod_read_unaligned(&gpu.read_buffer(&readback, submission))
}

pub fn capture_accelerated_trace_in_batches(
    observation: &Observation,
    tiles_per_batch: u32,
) -> TraceCapture {
    let gpu = crate::test_device::native_gpu();
    let trace = TracePipeline::for_accelerated_trace_capture(&gpu.device, observation)
        .expect("observation packs for GPU");
    capture_in_batches(gpu, observation, &trace, tiles_per_batch)
}

pub fn capture_initial_rays(observation: &Observation, subpixel: [f32; 2]) -> TraceCapture {
    let gpu = crate::test_device::native_gpu();
    let trace = TracePipeline::for_initial_ray_capture(&gpu.device, observation, subpixel)
        .expect("observation packs for GPU");
    capture(gpu, observation, &trace, false)
}

pub fn capture_invariant_gate_cases(observation: &Observation) -> TraceCapture {
    let gpu = crate::test_device::native_gpu();
    let trace = TracePipeline::for_invariant_gate_capture(&gpu.device, observation)
        .expect("observation packs for GPU");
    capture(gpu, observation, &trace, false)
}

pub fn capture_event_policy_cases(observation: &Observation) -> TraceCapture {
    let gpu = crate::test_device::native_gpu();
    let trace = TracePipeline::for_event_policy_capture(&gpu.device, observation)
        .expect("observation packs for GPU");
    capture(gpu, observation, &trace, false)
}

fn capture(
    gpu: &crate::test_device::TestGpu,
    observation: &Observation,
    trace: &TracePipeline,
    refine_shadow: bool,
) -> TraceCapture {
    let extent = observation_extent(observation);
    let target = trace.create_target(&gpu.device, extent);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless trace encoder"),
        });
    let tiles = TileRegion::all(extent);
    if refine_shadow {
        trace.encode(&gpu.queue, &mut encoder, &target, tiles);
    } else {
        trace.encode_base(&gpu.queue, &mut encoder, &target, tiles);
    }
    finish_capture(gpu, extent, &target, encoder)
}

fn capture_region(
    gpu: &crate::test_device::TestGpu,
    observation: &Observation,
    trace: &TracePipeline,
    tiles: TileRegion,
) -> TraceCapture {
    let extent = observation_extent(observation);
    let target = trace.create_target(&gpu.device, extent);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless trace region encoder"),
        });
    trace.encode_base(&gpu.queue, &mut encoder, &target, tiles);
    finish_capture(gpu, extent, &target, encoder)
}

fn capture_in_batches(
    gpu: &crate::test_device::TestGpu,
    observation: &Observation,
    trace: &TracePipeline,
    tiles_per_batch: u32,
) -> TraceCapture {
    assert!(tiles_per_batch > 0, "tile batches must be nonzero");
    let extent = observation_extent(observation);
    let target = trace.create_target(&gpu.device, extent);
    let [tile_columns, tile_rows] = tile_grid(extent);
    let total_tiles = tile_columns * tile_rows;
    let mut next_tile = 0;
    while next_tile < total_tiles {
        let tile_x = next_tile % tile_columns;
        let tile_y = next_tile / tile_columns;
        let workgroups_x = tiles_per_batch
            .min(tile_columns - tile_x)
            .min(total_tiles - next_tile);
        let batch = TileRegion::new([tile_x, tile_y], [workgroups_x, 1]);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless trace batch encoder"),
            });
        trace.encode_base(&gpu.queue, &mut encoder, &target, batch);
        let submission = gpu.queue.submit([encoder.finish()]);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("trace batch completes");
        next_tile += batch.len();
    }

    let encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless trace readback encoder"),
        });
    finish_capture(gpu, extent, &target, encoder)
}

fn observation_extent(observation: &Observation) -> RenderExtent {
    RenderExtent::new(
        observation.view().width().get(),
        observation.view().height().get(),
    )
    .expect("validated observation extent is nonzero")
}

fn finish_capture(
    gpu: &crate::test_device::TestGpu,
    extent: RenderExtent,
    target: &crate::trace::TraceTarget,
    mut encoder: wgpu::CommandEncoder,
) -> TraceCapture {
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
    let source_time = mapped[..plane_size].as_chunks::<RECORD_FIELD_SIZE>().0;
    let invariant_drift = mapped[plane_size..plane_size * 2]
        .as_chunks::<RECORD_FIELD_SIZE>()
        .0;
    let metadata = mapped[plane_size * 2..plane_size * 3]
        .as_chunks::<RECORD_FIELD_SIZE>()
        .0;
    let event = mapped[plane_size * 3..record_end]
        .as_chunks::<RECORD_FIELD_SIZE>()
        .0;
    let records = source_time
        .iter()
        .zip(invariant_drift)
        .zip(metadata)
        .zip(event)
        .map(
            |(((source_time, invariant_drift), metadata), event)| TraceRecord {
                source_time: bytemuck::pod_read_unaligned(source_time),
                invariant_drift: bytemuck::pod_read_unaligned(invariant_drift),
                metadata: bytemuck::pod_read_unaligned(metadata),
                event: bytemuck::pod_read_unaligned(event),
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

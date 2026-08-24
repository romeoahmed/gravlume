use gravlume_domain::ImageSample;
use wgpu::util::DeviceExt as _;

use super::{
    SampleInspectionError, SampleRetrace,
    protocol::{
        GpuInspectionRequest, INSPECTION_RECORD_BYTES, INSPECTION_REQUEST_BYTES,
        decode_corpus_readback,
    },
};
use crate::{
    extent::RenderExtent,
    test_device::TestGpu,
    trace::{TRACE_WORKGROUP_AXIS, TracePipeline, TracePlan, shader},
};

pub fn capture_sample_corpus(
    gpu: &TestGpu,
    trace: &TracePipeline,
    extent: RenderExtent,
    samples: &[ImageSample],
) -> Result<Vec<SampleRetrace>, SampleInspectionError> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    assert!(
        samples.iter().all(|sample| {
            let [pixel_x, pixel_y] = sample.pixel();
            pixel_x < extent.width() && pixel_y < extent.height()
        }),
        "sample corpus must belong to the observation extent"
    );
    let sample_count = u32::try_from(samples.len()).expect("sample count fits u32");
    let request_bytes = INSPECTION_REQUEST_BYTES
        .checked_mul(u64::from(sample_count))
        .expect("sample corpus request size fits u64");
    let record_bytes = INSPECTION_RECORD_BYTES
        .checked_mul(u64::from(sample_count))
        .expect("sample corpus record size fits u64");
    let limits = gpu.device.limits();
    let maximum_storage_binding = limits.max_storage_buffer_binding_size;
    assert!(
        request_bytes <= maximum_storage_binding && record_bytes <= maximum_storage_binding,
        "sample corpus exceeds the device storage-buffer binding limit"
    );
    assert!(
        request_bytes <= limits.max_buffer_size && record_bytes <= limits.max_buffer_size,
        "sample corpus exceeds the device buffer-size limit"
    );

    let requests = samples
        .iter()
        .copied()
        .map(|sample| GpuInspectionRequest::new(sample, extent))
        .collect::<Vec<_>>();
    let request_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sample corpus requests"),
            contents: bytemuck::cast_slice(&requests),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let record_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sample corpus records"),
        size: record_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sample corpus readback"),
        size: record_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let pipeline = create_pipeline(&gpu.device, trace);
    let bind_group = create_bind_group(
        &gpu.device,
        trace,
        &pipeline,
        &request_buffer,
        &record_buffer,
    );
    let invocations_per_workgroup = TRACE_WORKGROUP_AXIS * TRACE_WORKGROUP_AXIS;
    let workgroups = sample_count.div_ceil(invocations_per_workgroup);
    assert!(
        workgroups <= limits.max_compute_workgroups_per_dimension,
        "sample corpus exceeds the device dispatch limit"
    );

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sample corpus encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sample corpus pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One invocation owns one request/record pair. The shader bounds-checks the final partial
        // workgroup against the runtime-sized request array.
        // Source: https://docs.rs/wgpu/30.0.1/wgpu/struct.ComputePass.html#method.dispatch_workgroups
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&record_buffer, 0, &readback, 0, record_bytes);
    let submission = gpu.queue.submit([encoder.finish()]);
    let bytes = gpu.read_buffer(&readback, submission);
    let channel_model = trace
        .scientific_capture_metadata()
        .map(crate::scientific_capture::ScientificCaptureMetadata::channels);
    decode_corpus_readback(&bytes, channel_model, samples)
}

fn create_pipeline(device: &wgpu::Device, trace: &TracePipeline) -> wgpu::ComputePipeline {
    let shader_source = match trace.plan {
        TracePlan::AcceleratedSky => shader::analytic_sample_corpus(),
        TracePlan::EquatorialBolometricSurface => shader::bolometric_sample_corpus(),
        TracePlan::EquatorialBlackbodySurface => shader::blackbody_sample_corpus(),
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ordered sample corpus shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline_constants = [(
        "SURFACE_EVENTS_ENABLED",
        trace.plan.surface_events_enabled(),
    )];
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("inspect_sample_corpus"),
        layout: None,
        module: &shader,
        entry_point: Some("inspect_sample_corpus"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &pipeline_constants,
            ..Default::default()
        },
        cache: None,
    })
}

fn create_bind_group(
    device: &wgpu::Device,
    trace: &TracePipeline,
    pipeline: &wgpu::ComputePipeline,
    requests: &wgpu::Buffer,
    records: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(0);
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: trace.uniforms.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 10,
            resource: requests.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 11,
            resource: records.as_entire_binding(),
        },
    ];
    if let Some(blackbody_lut) = &trace.blackbody_lut {
        entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: blackbody_lut.as_entire_binding(),
        });
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sample corpus bind group"),
        layout: &layout,
        entries: &entries,
    })
}

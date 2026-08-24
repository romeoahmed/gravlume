use gravlume_domain::ImageSample;
use wgpu::util::DeviceExt as _;

use super::{
    SampleInspectionError, SampleRetrace,
    kernel::{SampleInspectionKernel, inspection_workgroup_count},
    protocol::{
        GpuInspectionRequest, INSPECTION_RECORD_BYTES, INSPECTION_REQUEST_BYTES,
        decode_corpus_readback,
    },
};
use crate::{extent::RenderExtent, test_device::TestGpu, trace::TracePipeline};

impl TracePipeline {
    pub(crate) fn capture_sample_corpus(
        &self,
        gpu: &TestGpu,
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
        let workgroups = inspection_workgroup_count(sample_count);
        assert!(
            workgroups <= limits.max_compute_workgroups_per_dimension,
            "sample corpus exceeds the device dispatch limit"
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

        let kernel =
            SampleInspectionKernel::new(&gpu.device, self, &request_buffer, &record_buffer);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sample corpus encoder"),
            });
        kernel.encode_samples(&mut encoder, sample_count);
        encoder.copy_buffer_to_buffer(&record_buffer, 0, &readback, 0, record_bytes);
        let submission = gpu.queue.submit([encoder.finish()]);
        let bytes = gpu.read_buffer(&readback, submission);
        let channel_model = self
            .scientific_capture_metadata()
            .map(crate::scientific_capture::ScientificCaptureMetadata::channels);
        decode_corpus_readback(&bytes, channel_model, samples)
    }
}

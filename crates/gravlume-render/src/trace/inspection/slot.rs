use std::sync::mpsc::{self, TryRecvError};

use gravlume_domain::ImageSample;

use super::{
    SampleInspectionCompletion, SampleInspectionError, SampleInspectionRequestError,
    SampleInspectionTicket,
    protocol::{
        GpuInspectionRequest, INSPECTION_READBACK_BYTES, INSPECTION_RECORD_BYTES,
        INSPECTION_REQUEST_BYTES, PUBLISHED_TEXEL_OFFSET, decode_readback,
    },
};
use crate::{extent::RenderExtent, scientific_capture::ScientificChannelModel};

use super::super::{TracePipeline, TracePlan, shader};

pub(super) struct PendingInspection {
    pub(super) receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    pub(super) ticket: SampleInspectionTicket,
    pub(super) cancelled: bool,
}

impl PendingInspection {
    fn cancelled_completion(
        &self,
        accepted_generation: Option<u64>,
    ) -> Option<SampleInspectionCompletion> {
        (self.cancelled || accepted_generation != Some(self.ticket.generation())).then_some(
            SampleInspectionCompletion::Cancelled {
                ticket: self.ticket,
            },
        )
    }
}

pub struct SampleInspectionSlot {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    request: wgpu::Buffer,
    record: wgpu::Buffer,
    readback: wgpu::Buffer,
    channel_model: Option<ScientificChannelModel>,
    pub(super) pending: Option<PendingInspection>,
}

impl SampleInspectionSlot {
    pub(crate) fn new(device: &wgpu::Device, trace: &TracePipeline) -> Self {
        let request = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sample inspection request"),
            size: INSPECTION_REQUEST_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let record = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sample inspection record"),
            size: INSPECTION_RECORD_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sample inspection readback"),
            size: INSPECTION_READBACK_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let pipeline = create_inspection_pipeline(device, trace);
        let bind_group = create_inspection_bind_group(device, trace, &pipeline, &request, &record);
        Self {
            pipeline,
            bind_group,
            request,
            record,
            readback,
            channel_model: trace
                .scientific_capture_metadata()
                .map(crate::scientific_capture::ScientificCaptureMetadata::channels),
            pending: None,
        }
    }

    pub(crate) fn request(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        published_texture: &wgpu::Texture,
        extent: RenderExtent,
        generation: u64,
        sample: ImageSample,
    ) -> Result<SampleInspectionTicket, SampleInspectionRequestError> {
        let pixel = sample.pixel();
        let extent_array = [extent.width(), extent.height()];
        if pixel[0] >= extent.width() || pixel[1] >= extent.height() {
            return Err(SampleInspectionRequestError::SampleOutsideExtent {
                pixel,
                extent: extent_array,
            });
        }
        if self.pending.is_some() {
            return Err(SampleInspectionRequestError::Busy);
        }
        let request = GpuInspectionRequest::new(sample, extent);
        let ticket = SampleInspectionTicket::new(generation, extent, sample);
        // Queue writes are staged immediately and execute before the following submission.
        // Source: https://docs.rs/wgpu/30.0.1/wgpu/struct.Queue.html#method.write_buffer
        queue.write_buffer(&self.request, 0, bytemuck::bytes_of(&request));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sample inspection encoder"),
        });
        // Zero is an invalid termination discriminant, so a missing or partial shader write can
        // never decode as the previous request's record.
        encoder.clear_buffer(&self.record, 0, Some(INSPECTION_RECORD_BYTES));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sample inspection pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.record, 0, &self.readback, 0, INSPECTION_RECORD_BYTES);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: published_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: pixel[0],
                    y: pixel[1],
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: PUBLISHED_TEXEL_OFFSET,
                    // Keep the native copy layout explicit. The portable 256-byte row stride does
                    // not add trailing padding after this copy's only row, so the eight-byte texel
                    // can immediately follow the record.
                    // Sources:
                    // - https://docs.rs/wgpu/30.0.1/wgpu/struct.TexelCopyBufferLayout.html
                    // - https://docs.rs/wgpu/30.0.1/wgpu/struct.BufferTextureCopyInfo.html#structfield.bytes_in_copy
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let (sender, receiver) = mpsc::sync_channel(1);
        // Mapping belongs to the same ordered submission as the record and published-texel copies.
        // The event loop drives the short callback with `Device::poll`.
        // Source: https://docs.rs/wgpu/30.0.1/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit
        encoder.map_buffer_on_submit(&self.readback, wgpu::MapMode::Read, .., move |result| {
            if sender.send(result).is_err() {
                tracing::debug!("sample inspection callback receiver dropped");
            }
        });
        queue.submit([encoder.finish()]);
        self.pending = Some(PendingInspection {
            receiver,
            ticket,
            cancelled: false,
        });
        Ok(ticket)
    }

    pub(crate) const fn cancel_active(&mut self) {
        if let Some(pending) = self.pending.as_mut() {
            pending.cancelled = true;
        }
    }

    pub(crate) fn poll(
        &mut self,
        accepted_generation: Option<u64>,
    ) -> Option<SampleInspectionCompletion> {
        let pending = self.pending.take()?;
        let map_result = match pending.receiver.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => {
                self.pending = Some(pending);
                return None;
            }
            Err(TryRecvError::Disconnected) => {
                return Some(Self::disconnected_completion(&pending, accepted_generation));
            }
        };

        Some(self.complete_mapping(&pending, accepted_generation, map_result))
    }

    fn complete_mapping(
        &self,
        pending: &PendingInspection,
        accepted_generation: Option<u64>,
        map_result: Result<(), wgpu::BufferAsyncError>,
    ) -> SampleInspectionCompletion {
        let cancelled = pending.cancelled_completion(accepted_generation);
        if let Err(error) = map_result {
            // The callback grants CPU access only with `Ok`; a failed mapping has no mapped range
            // to release. Source: https://docs.rs/wgpu/30.0.1/wgpu/struct.Buffer.html#method.map_async
            return cancelled.unwrap_or_else(|| SampleInspectionCompletion::Failed {
                ticket: pending.ticket,
                error: error.into(),
            });
        }
        if let Some(cancelled) = cancelled {
            self.readback.unmap();
            return cancelled;
        }

        let result = self.read_inspection(pending.ticket);
        self.readback.unmap();
        match result {
            Ok(inspection) => SampleInspectionCompletion::Completed {
                ticket: pending.ticket,
                inspection,
            },
            Err(error) => SampleInspectionCompletion::Failed {
                ticket: pending.ticket,
                error,
            },
        }
    }

    fn disconnected_completion(
        pending: &PendingInspection,
        accepted_generation: Option<u64>,
    ) -> SampleInspectionCompletion {
        pending.cancelled_completion(accepted_generation).unwrap_or(
            SampleInspectionCompletion::Failed {
                ticket: pending.ticket,
                error: SampleInspectionError::CallbackDisconnected,
            },
        )
    }

    #[cfg(test)]
    pub(super) fn wait_for_completion(
        &mut self,
        accepted_generation: Option<u64>,
    ) -> SampleInspectionCompletion {
        let pending = self
            .pending
            .take()
            .expect("test inspection has a pending mapping");
        let Ok(map_result) = pending.receiver.recv() else {
            return Self::disconnected_completion(&pending, accepted_generation);
        };
        self.complete_mapping(&pending, accepted_generation, map_result)
    }

    fn read_inspection(
        &self,
        ticket: SampleInspectionTicket,
    ) -> Result<super::SampleInspection, SampleInspectionError> {
        let mapped = self.readback.get_mapped_range(..)?;
        let result = decode_readback(&mapped, self.channel_model, ticket);
        drop(mapped);
        result
    }

    pub(crate) const fn is_pending(&self) -> bool {
        self.pending.is_some()
    }
}

fn create_inspection_pipeline(
    device: &wgpu::Device,
    trace: &TracePipeline,
) -> wgpu::ComputePipeline {
    let shader_source = match trace.plan {
        TracePlan::AcceleratedSky => shader::analytic_sample_inspection(),
        TracePlan::EquatorialBolometricSurface => shader::bolometric_sample_inspection(),
        TracePlan::EquatorialBlackbodySurface => shader::blackbody_sample_inspection(),
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bounded sample inspection shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline_constants = [(
        "SURFACE_EVENTS_ENABLED",
        trace.plan.surface_events_enabled(),
    )];
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("inspect_sample"),
        // The inspection module owns this pipeline and its sole bind group; no layout is shared
        // with presentation or capture pipelines. Derive the private layout from this entry point.
        // Source: https://docs.rs/wgpu/30.0.1/wgpu/struct.ComputePipelineDescriptor.html#structfield.layout
        layout: None,
        module: &shader,
        entry_point: Some("inspect_sample"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &pipeline_constants,
            ..Default::default()
        },
        cache: None,
    })
}

fn create_inspection_bind_group(
    device: &wgpu::Device,
    trace: &TracePipeline,
    pipeline: &wgpu::ComputePipeline,
    request: &wgpu::Buffer,
    record: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(0);
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: trace.uniforms.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: record.as_entire_binding(),
        },
    ];
    if let Some(blackbody_lut) = &trace.blackbody_lut {
        entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: blackbody_lut.as_entire_binding(),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: 9,
        resource: request.as_entire_binding(),
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sample inspection bind group"),
        layout: &layout,
        entries: &entries,
    })
}

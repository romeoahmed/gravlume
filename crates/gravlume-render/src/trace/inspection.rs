use std::sync::mpsc::{self, TryRecvError};

use gravlume_domain::ImageSample;

use super::{TracePipeline, TracePlan, shader, size_of};
use crate::{
    extent::RenderExtent,
    scientific_capture::{ScientificChannelModel, ScientificTexel},
};

const INSPECTION_REQUEST_BYTES: u64 = size_of::<GpuInspectionRequest>();
const INSPECTION_RECORD_BYTES: u64 = size_of::<GpuInspectionRecord>();
const PUBLISHED_TEXEL_OFFSET: u64 = INSPECTION_RECORD_BYTES;
const PUBLISHED_TEXEL_BYTES: u64 = 8;
const INSPECTION_READBACK_BYTES: u64 = PUBLISHED_TEXEL_OFFSET + PUBLISHED_TEXEL_BYTES;
const _: () = {
    assert!(PUBLISHED_TEXEL_BYTES == size_of::<[u16; 4]>());
    assert!(PUBLISHED_TEXEL_OFFSET.is_multiple_of(PUBLISHED_TEXEL_BYTES));
};
const KNOWN_NUMERICAL_FLAGS: u32 = 1 | 2 | 4;
const KNOWN_EVENT_CANDIDATES: u32 = 1 | 2 | 4 | 8;

const HORIZON_TAG: u32 = 0.0_f32.to_bits();
const ANALYTIC_ESCAPE_TAG: u32 = 1.0_f32.to_bits();
const SURFACE_RADIANCE_TAG: u32 = 2.0_f32.to_bits();
const SINGULARITY_FAILURE_TAG: u32 = (-3.0_f32).to_bits();
const STEP_EXHAUSTION_FAILURE_TAG: u32 = (-4.0_f32).to_bits();
const NUMERICAL_FAILURE_TAG: u32 = (-5.0_f32).to_bits();
const UNCERTAIN_FAILURE_TAG: u32 = (-6.0_f32).to_bits();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TraceTermination {
    HorizonCrossing = 1,
    Escape = 2,
    SingularityGuard = 3,
    StepExhaustion = 4,
    NumericalFailure = 5,
    Uncertain = 6,
    EquatorialSurface = 7,
}

impl From<TraceTermination> for u32 {
    fn from(value: TraceTermination) -> Self {
        value as Self
    }
}

impl TryFrom<u32> for TraceTermination {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::HorizonCrossing),
            2 => Ok(Self::Escape),
            3 => Ok(Self::SingularityGuard),
            4 => Ok(Self::StepExhaustion),
            5 => Ok(Self::NumericalFailure),
            6 => Ok(Self::Uncertain),
            7 => Ok(Self::EquatorialSurface),
            unknown => Err(unknown),
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// Immutable target captured when one inspection request is admitted.
///
/// This live renderer ticket is not a persisted artifact identity. Persisted evidence needs its
/// own canonical observation, producer revision, and backend identity.
pub struct SampleInspectionTicket {
    generation: u64,
    extent: [u32; 2],
    sample: ImageSample,
}

impl SampleInspectionTicket {
    const fn new(generation: u64, extent: RenderExtent, sample: ImageSample) -> Self {
        Self {
            generation,
            extent: [extent.width(), extent.height()],
            sample,
        }
    }

    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }

    #[must_use]
    pub const fn extent(self) -> [u32; 2] {
        self.extent
    }

    #[must_use]
    pub const fn sample(self) -> ImageSample {
        self.sample
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplePolarSide {
    Negative,
    Equatorial,
    Positive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleBranchKey {
    pub(crate) initial_polar_side: SamplePolarSide,
    pub(crate) radial_turnings: u32,
    pub(crate) equatorial_crossings: u32,
    pub(crate) azimuth_winding: i32,
}

impl SampleBranchKey {
    #[must_use]
    pub const fn initial_polar_side(self) -> SamplePolarSide {
        self.initial_polar_side
    }

    #[must_use]
    pub const fn radial_turnings(self) -> u32 {
        self.radial_turnings
    }

    #[must_use]
    pub const fn equatorial_crossings(self) -> u32 {
        self.equatorial_crossings
    }

    #[must_use]
    pub const fn azimuth_winding(self) -> i32 {
        self.azimuth_winding
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SampleSurfaceEvaluation {
    Radiance([f32; 3]),
    NumericalFailure { visible_rgb: [f32; 3] },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SampleTraceOutcome {
    Horizon {
        branch: SampleBranchKey,
    },
    Escape {
        branch: SampleBranchKey,
        /// A normalized orientation used by the analytic sky preview, not a physical spectrum.
        unit_direction: [f32; 3],
        preview_rgb: [f32; 3],
    },
    EquatorialSurface {
        branch: SampleBranchKey,
        radius_over_m: f32,
        azimuth_radians: f32,
        frequency_ratio: f32,
        channels: ScientificChannelModel,
        evaluation: SampleSurfaceEvaluation,
    },
    SingularityGuard {
        branch: SampleBranchKey,
        visible_rgb: [f32; 3],
    },
    StepExhausted {
        branch_prefix: SampleBranchKey,
        visible_rgb: [f32; 3],
    },
    NumericalFailure {
        visible_rgb: [f32; 3],
    },
    Uncertain {
        visible_rgb: [f32; 3],
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleTraceDiagnostics {
    coordinate_time_delta_over_m: f32,
    event_candidates: u32,
    event_residual: f32,
    steps: u32,
    numerical_flags: u32,
    maximum_invariant_drift: [f32; 4],
}

impl SampleTraceDiagnostics {
    #[must_use]
    pub const fn coordinate_time_delta_over_m(self) -> f32 {
        self.coordinate_time_delta_over_m
    }

    #[must_use]
    pub const fn event_candidate_bits(self) -> u32 {
        self.event_candidates
    }

    #[must_use]
    pub const fn event_residual(self) -> f32 {
        self.event_residual
    }

    #[must_use]
    pub const fn steps(self) -> u32 {
        self.steps
    }

    #[must_use]
    pub const fn numerical_flag_bits(self) -> u32 {
        self.numerical_flags
    }

    #[must_use]
    pub const fn maximum_invariant_drift(self) -> [f32; 4] {
        self.maximum_invariant_drift
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleRetrace {
    effective_subpixel: [f32; 2],
    outcome: SampleTraceOutcome,
    diagnostics: SampleTraceDiagnostics,
}

impl SampleRetrace {
    /// Identifies the fixed full Kerr-Schild RK4 retrace and its WGSL binary32 arithmetic domain.
    pub const METHOD_ID: &str = "gpu-ks-rk4-v1/full-kerr-schild-retrace/wgsl-binary32";

    #[must_use]
    pub const fn effective_subpixel(self) -> [f32; 2] {
        self.effective_subpixel
    }

    #[must_use]
    pub const fn outcome(self) -> SampleTraceOutcome {
        self.outcome
    }

    #[must_use]
    pub const fn diagnostics(self) -> SampleTraceDiagnostics {
        self.diagnostics
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleInspection {
    published_texel: ScientificTexel,
    fresh_retrace: SampleRetrace,
}

impl SampleInspection {
    /// Returns the exact `Rgba16Float` texel copied from the bound published generation.
    #[must_use]
    pub const fn published_texel(&self) -> ScientificTexel {
        self.published_texel
    }

    /// Returns the binary32 evidence from the fresh full Kerr-Schild retrace.
    ///
    /// This is deliberately separate from [`Self::published_texel`], which may include a
    /// conservative accelerator, shadow refinement, and `Rgba16Float` rounding.
    #[must_use]
    pub const fn fresh_retrace(&self) -> SampleRetrace {
        self.fresh_retrace
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SampleInspectionError {
    #[error("unknown GPU trace termination discriminant {0}")]
    UnknownTermination(u32),
    #[error("unknown GPU sample initial polar-side discriminant {0}")]
    UnknownPolarSide(u32),
    #[error("GPU sample inspection returned an invalid {field} record")]
    InvalidRecord { field: &'static str },
    #[error("GPU sample inspection readback mapping failed: {0}")]
    Map(#[from] wgpu::BufferAsyncError),
    #[error("GPU sample inspection mapped range was unavailable: {0}")]
    MappedRange(#[from] wgpu::MapRangeError),
    #[error("GPU sample inspection callback channel disconnected")]
    CallbackDisconnected,
    #[error("GPU sample inspection readback had an invalid byte count")]
    InvalidReadback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SampleInspectionRequestError {
    #[error("the renderer has no complete publication for its current generation")]
    NoCurrentPublication,
    #[error("sample pixel {pixel:?} lies outside the published extent {extent:?}")]
    SampleOutsideExtent { pixel: [u32; 2], extent: [u32; 2] },
    #[error("the fixed sample inspection slot is still in flight")]
    Busy,
}

#[derive(Debug)]
pub struct SampleInspectionCompletion {
    ticket: SampleInspectionTicket,
    disposition: SampleInspectionDisposition,
}

impl SampleInspectionCompletion {
    const fn new(ticket: SampleInspectionTicket, disposition: SampleInspectionDisposition) -> Self {
        Self {
            ticket,
            disposition,
        }
    }

    #[must_use]
    pub const fn ticket(&self) -> SampleInspectionTicket {
        self.ticket
    }

    #[must_use]
    pub const fn disposition(&self) -> &SampleInspectionDisposition {
        &self.disposition
    }
}

#[derive(Debug)]
pub enum SampleInspectionDisposition {
    Completed(SampleInspection),
    Cancelled,
    Failed(SampleInspectionError),
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
struct GpuInspectionRequest {
    pixel_extent: [u32; 4],
    subpixel: [f32; 4],
}

impl GpuInspectionRequest {
    const fn new(sample: ImageSample, extent: RenderExtent) -> Self {
        let [pixel_x, pixel_y] = sample.pixel();
        let [subpixel_x, subpixel_y] = binary32_subpixel(sample);
        Self {
            pixel_extent: [pixel_x, pixel_y, extent.width(), extent.height()],
            subpixel: [subpixel_x, subpixel_y, 0.0, 0.0],
        }
    }
}

const fn binary32_subpixel(sample: ImageSample) -> [f32; 2] {
    let [subpixel_x, subpixel_y] = sample.subpixel();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "ImageSample guarantees finite subpixel coordinates in [0, 1], so binary32 rounding is total"
    )]
    {
        [subpixel_x as f32, subpixel_y as f32]
    }
}

const _: () = {
    assert!(std::mem::size_of::<GpuInspectionRequest>() == 32);
    assert!(std::mem::align_of::<GpuInspectionRequest>() == 16);
    assert!(std::mem::offset_of!(GpuInspectionRequest, pixel_extent) == 0);
    assert!(std::mem::offset_of!(GpuInspectionRequest, subpixel) == 16);
};

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
struct GpuInspectionRecord {
    // (termination, failure flags, steps, event candidates)
    metadata: [u32; 4],
    branch_key: [u32; 4],
    source_time: [f32; 4],
    scene_value: [f32; 4],
    // (event residual, reserved, reserved, reserved)
    event_diagnostics: [f32; 4],
    maximum_invariant_drift: [f32; 4],
}

// Six vec4 lanes give the storage record an exact, padding-free 16-byte layout.
// Source: https://www.w3.org/TR/WGSL/#alignment-and-size
const _: () = {
    assert!(std::mem::size_of::<GpuInspectionRecord>() == 96);
    assert!(std::mem::align_of::<GpuInspectionRecord>() == 16);
    assert!(std::mem::offset_of!(GpuInspectionRecord, metadata) == 0);
    assert!(std::mem::offset_of!(GpuInspectionRecord, branch_key) == 16);
    assert!(std::mem::offset_of!(GpuInspectionRecord, source_time) == 32);
    assert!(std::mem::offset_of!(GpuInspectionRecord, scene_value) == 48);
    assert!(std::mem::offset_of!(GpuInspectionRecord, event_diagnostics) == 64);
    assert!(std::mem::offset_of!(GpuInspectionRecord, maximum_invariant_drift) == 80);
};

struct PendingInspection {
    receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ticket: SampleInspectionTicket,
    cancelled: bool,
}

impl PendingInspection {
    fn disposition(&self, accepted_generation: Option<u64>) -> Option<SampleInspectionDisposition> {
        if self.cancelled || accepted_generation != Some(self.ticket.generation()) {
            Some(SampleInspectionDisposition::Cancelled)
        } else {
            None
        }
    }
}

pub struct SampleInspector {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    request: wgpu::Buffer,
    record: wgpu::Buffer,
    readback: wgpu::Buffer,
    channel_model: Option<ScientificChannelModel>,
    pending: Option<PendingInspection>,
}

impl SampleInspector {
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
        let disposition = pending.disposition(accepted_generation);
        if let Err(error) = map_result {
            // The callback grants CPU access only with `Ok`; a failed mapping has no mapped range
            // to release. Source: https://docs.rs/wgpu/30.0.1/wgpu/struct.Buffer.html#method.map_async
            return SampleInspectionCompletion::new(
                pending.ticket,
                disposition.unwrap_or_else(|| SampleInspectionDisposition::Failed(error.into())),
            );
        }
        if let Some(disposition) = disposition {
            self.readback.unmap();
            return SampleInspectionCompletion::new(pending.ticket, disposition);
        }

        let result = self.read_inspection(pending.ticket);
        self.readback.unmap();
        let disposition = match result {
            Ok(inspection) => SampleInspectionDisposition::Completed(inspection),
            Err(error) => SampleInspectionDisposition::Failed(error),
        };
        SampleInspectionCompletion::new(pending.ticket, disposition)
    }

    fn disconnected_completion(
        pending: &PendingInspection,
        accepted_generation: Option<u64>,
    ) -> SampleInspectionCompletion {
        let disposition = pending.disposition(accepted_generation).unwrap_or(
            SampleInspectionDisposition::Failed(SampleInspectionError::CallbackDisconnected),
        );
        SampleInspectionCompletion::new(pending.ticket, disposition)
    }

    #[cfg(test)]
    fn wait_for_completion(
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
    ) -> Result<SampleInspection, SampleInspectionError> {
        let mapped = self.readback.get_mapped_range(..)?;
        let record_bytes = mapped
            .get(..std::mem::size_of::<GpuInspectionRecord>())
            .ok_or(SampleInspectionError::InvalidReadback)?;
        let record = bytemuck::try_pod_read_unaligned(record_bytes)
            .map_err(|_| SampleInspectionError::InvalidReadback)?;
        let texel_start = usize::try_from(PUBLISHED_TEXEL_OFFSET)
            .map_err(|_| SampleInspectionError::InvalidReadback)?;
        let texel_end = usize::try_from(PUBLISHED_TEXEL_OFFSET + PUBLISHED_TEXEL_BYTES)
            .map_err(|_| SampleInspectionError::InvalidReadback)?;
        let texel_bytes = mapped
            .get(texel_start..texel_end)
            .ok_or(SampleInspectionError::InvalidReadback)?;
        let rgba16_float_bits = std::array::from_fn(|channel| {
            let offset = channel * std::mem::size_of::<u16>();
            u16::from_le_bytes([texel_bytes[offset], texel_bytes[offset + 1]])
        });
        drop(mapped);
        decode_record(
            record,
            self.channel_model,
            ticket,
            ScientificTexel::from_rgba16_float_bits(rgba16_float_bits),
        )
    }

    pub(crate) const fn has_pending_request(&self) -> bool {
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

#[cfg(test)]
impl TracePipeline {
    pub(crate) fn inspect_sample(
        &self,
        gpu: &crate::test_device::TestGpu,
        extent: RenderExtent,
        sample: ImageSample,
    ) -> SampleInspection {
        let published = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-only sample inspection published texel"),
            size: wgpu::Extent3d {
                width: extent.width(),
                height: extent.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut inspector = SampleInspector::new(&gpu.device, self);
        inspector
            .request(&gpu.device, &gpu.queue, &published, extent, 1, sample)
            .expect("test inspection request is accepted");
        let fence = gpu.queue.submit([]);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(fence),
                timeout: None,
            })
            .expect("test inspection submission completes");
        match inspector.wait_for_completion(Some(1)).disposition {
            SampleInspectionDisposition::Completed(inspection) => inspection,
            disposition => {
                panic!("test inspection must complete, got {disposition:?}")
            }
        }
    }
}

fn decode_record(
    raw: GpuInspectionRecord,
    channel_model: Option<ScientificChannelModel>,
    ticket: SampleInspectionTicket,
    published_texel: ScientificTexel,
) -> Result<SampleInspection, SampleInspectionError> {
    if !raw
        .source_time
        .iter()
        .chain(&raw.scene_value)
        .chain(&raw.event_diagnostics)
        .chain(&raw.maximum_invariant_drift)
        .all(|value| value.is_finite())
    {
        return Err(SampleInspectionError::InvalidRecord {
            field: "non-finite value",
        });
    }
    if raw.event_diagnostics[1..] != [0.0; 3] {
        return Err(SampleInspectionError::InvalidRecord {
            field: "event diagnostics",
        });
    }

    let termination = TraceTermination::try_from(raw.metadata[0])
        .map_err(SampleInspectionError::UnknownTermination)?;
    let numerical_flags = raw.metadata[1];
    if numerical_flags & !KNOWN_NUMERICAL_FLAGS != 0 {
        return Err(SampleInspectionError::InvalidRecord {
            field: "failure flags",
        });
    }
    let event_candidates = raw.metadata[3];
    if event_candidates & !KNOWN_EVENT_CANDIDATES != 0 {
        return Err(SampleInspectionError::InvalidRecord {
            field: "event candidates",
        });
    }

    let outcome = decode_outcome(
        termination,
        decode_branch_key(termination, raw.branch_key)?,
        raw.source_time,
        raw.scene_value,
        channel_model,
    )?;
    Ok(SampleInspection {
        published_texel,
        fresh_retrace: SampleRetrace {
            effective_subpixel: binary32_subpixel(ticket.sample()),
            outcome,
            diagnostics: SampleTraceDiagnostics {
                coordinate_time_delta_over_m: raw.source_time[3],
                event_candidates,
                event_residual: raw.event_diagnostics[0],
                steps: raw.metadata[2],
                numerical_flags,
                maximum_invariant_drift: raw.maximum_invariant_drift,
            },
        },
    })
}

fn decode_branch_key(
    termination: TraceTermination,
    words: [u32; 4],
) -> Result<Option<SampleBranchKey>, SampleInspectionError> {
    if termination == TraceTermination::NumericalFailure {
        if words != [0; 4] {
            return Err(SampleInspectionError::InvalidRecord {
                field: "numerical-failure branch",
            });
        }
        return Ok(None);
    }

    let initial_polar_side = match words[3] {
        0 => SamplePolarSide::Negative,
        1 => SamplePolarSide::Equatorial,
        2 => SamplePolarSide::Positive,
        unknown => return Err(SampleInspectionError::UnknownPolarSide(unknown)),
    };
    if termination == TraceTermination::Uncertain {
        return Ok(None);
    }
    Ok(Some(SampleBranchKey {
        initial_polar_side,
        radial_turnings: words[0],
        equatorial_crossings: words[1],
        azimuth_winding: i32::from_ne_bytes(words[2].to_ne_bytes()),
    }))
}

fn decode_outcome(
    termination: TraceTermination,
    branch: Option<SampleBranchKey>,
    source_time: [f32; 4],
    value: [f32; 4],
    channel_model: Option<ScientificChannelModel>,
) -> Result<SampleTraceOutcome, SampleInspectionError> {
    let [source_x, source_y, source_z, _] = source_time;
    let [red, green, blue, alpha] = value;
    let rgb = [red, green, blue];
    let tag = alpha.to_bits();

    match termination {
        TraceTermination::HorizonCrossing => {
            require_scene_tag(tag, HORIZON_TAG)?;
            if rgb.map(f32::to_bits) != [0; 3] {
                return Err(SampleInspectionError::InvalidRecord {
                    field: "horizon scene value",
                });
            }
            Ok(SampleTraceOutcome::Horizon {
                branch: require_branch(branch)?,
            })
        }
        TraceTermination::Escape => {
            require_scene_tag(tag, ANALYTIC_ESCAPE_TAG)?;
            Ok(SampleTraceOutcome::Escape {
                branch: require_branch(branch)?,
                unit_direction: [source_x, source_y, source_z],
                preview_rgb: rgb,
            })
        }
        TraceTermination::EquatorialSurface => {
            if source_x <= 0.0 || source_z <= 0.0 {
                return Err(SampleInspectionError::InvalidRecord {
                    field: "surface source",
                });
            }
            let channels = channel_model.ok_or(SampleInspectionError::InvalidRecord {
                field: "surface channel model",
            })?;
            let evaluation = if tag == SURFACE_RADIANCE_TAG {
                if rgb.into_iter().any(|channel| channel < 0.0) {
                    return Err(SampleInspectionError::InvalidRecord {
                        field: "surface radiance",
                    });
                }
                SampleSurfaceEvaluation::Radiance(rgb)
            } else if tag == NUMERICAL_FAILURE_TAG {
                SampleSurfaceEvaluation::NumericalFailure { visible_rgb: rgb }
            } else {
                return Err(SampleInspectionError::InvalidRecord {
                    field: "termination/scene tag",
                });
            };
            Ok(SampleTraceOutcome::EquatorialSurface {
                branch: require_branch(branch)?,
                radius_over_m: source_x,
                azimuth_radians: source_y,
                frequency_ratio: source_z,
                channels,
                evaluation,
            })
        }
        TraceTermination::SingularityGuard => {
            require_scene_tag(tag, SINGULARITY_FAILURE_TAG)?;
            Ok(SampleTraceOutcome::SingularityGuard {
                branch: require_branch(branch)?,
                visible_rgb: rgb,
            })
        }
        TraceTermination::StepExhaustion => {
            require_scene_tag(tag, STEP_EXHAUSTION_FAILURE_TAG)?;
            Ok(SampleTraceOutcome::StepExhausted {
                branch_prefix: require_branch(branch)?,
                visible_rgb: rgb,
            })
        }
        TraceTermination::NumericalFailure => {
            require_scene_tag(tag, NUMERICAL_FAILURE_TAG)?;
            Ok(SampleTraceOutcome::NumericalFailure { visible_rgb: rgb })
        }
        TraceTermination::Uncertain => {
            require_scene_tag(tag, UNCERTAIN_FAILURE_TAG)?;
            Ok(SampleTraceOutcome::Uncertain { visible_rgb: rgb })
        }
    }
}

fn require_branch(
    branch: Option<SampleBranchKey>,
) -> Result<SampleBranchKey, SampleInspectionError> {
    branch.ok_or(SampleInspectionError::InvalidRecord {
        field: "missing branch",
    })
}

const fn require_scene_tag(actual: u32, expected: u32) -> Result<(), SampleInspectionError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SampleInspectionError::InvalidRecord {
            field: "termination/scene tag",
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use proptest::prelude::*;

    use super::{
        INSPECTION_READBACK_BYTES, INSPECTION_RECORD_BYTES, INSPECTION_REQUEST_BYTES,
        PendingInspection, SampleBranchKey, SampleInspectionDisposition, SampleInspectionError,
        SampleInspectionRequestError, SampleInspectionTicket, SampleInspector, SamplePolarSide,
        SampleRetrace, SampleTraceOutcome, decode_branch_key,
    };
    use crate::{
        error::GpuErrorScopes,
        extent::RenderExtent,
        test_device::native_gpu,
        trace::{TracePipeline, TraceTermination},
    };

    fn branch_counter() -> impl Strategy<Value = u32> {
        prop_oneof![Just(0), Just(u32::MAX), any::<u32>()]
    }

    fn branch_winding() -> impl Strategy<Value = i32> {
        prop_oneof![Just(i32::MIN), Just(0), Just(i32::MAX), any::<i32>()]
    }

    fn assert_same_ticket(actual: SampleInspectionTicket, expected: SampleInspectionTicket) {
        assert_eq!(actual.generation(), expected.generation());
        assert_eq!(actual.extent(), expected.extent());
        assert_eq!(actual.sample(), expected.sample());
    }

    #[test]
    fn numerical_failure_uses_an_explicit_zero_branch_sentinel() {
        assert_eq!(
            decode_branch_key(TraceTermination::NumericalFailure, [0; 4])
                .expect("zero failure sentinel decodes"),
            None
        );
    }

    #[test]
    fn fixed_buffers_match_the_documented_logical_budget() {
        assert_eq!(INSPECTION_REQUEST_BYTES, 32);
        assert_eq!(INSPECTION_RECORD_BYTES, 96);
        assert_eq!(INSPECTION_READBACK_BYTES, 104);
        assert_eq!(
            INSPECTION_REQUEST_BYTES + INSPECTION_RECORD_BYTES + INSPECTION_READBACK_BYTES,
            232
        );
    }

    #[test]
    fn map_failure_does_not_emit_a_secondary_unmap_validation_error() {
        const SURFACE_OBSERVABLE: &str =
            include_str!("../../../gravlume-reference/fixtures/v2/kerr-surface-observable.toml");
        let fixture = gravlume_reference::FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
            .expect("repository surface fixture parses")
            .into_surface_observation()
            .expect("fixture is a surface observation");
        let observation = fixture.observation();
        let extent = RenderExtent::new(
            observation.view().width().get(),
            observation.view().height().get(),
        )
        .expect("fixture extent is nonzero");
        let gpu = native_gpu();
        let trace = TracePipeline::new(&gpu.device, observation)
            .expect("fixture observation enters the GPU profile");
        let mut inspector = SampleInspector::new(&gpu.device, &trace);
        let ticket = SampleInspectionTicket::new(23, extent, fixture.sample());
        let (sender, receiver) = mpsc::sync_channel(1);
        sender
            .send(Err(wgpu::BufferAsyncError))
            .expect("synthetic map completion is delivered");
        inspector.pending = Some(PendingInspection {
            receiver,
            ticket,
            cancelled: false,
        });
        let scopes = GpuErrorScopes::push(&gpu.device);

        let completion = inspector
            .poll(Some(23))
            .expect("map failure produces one terminal event");

        assert_same_ticket(completion.ticket(), ticket);
        assert!(matches!(
            completion.disposition(),
            SampleInspectionDisposition::Failed(SampleInspectionError::Map(_))
        ));
        let secondary_error = pollster::block_on(scopes.finish());
        assert!(
            secondary_error.is_ok(),
            "the typed map failure must be the only reported error: {secondary_error:?}"
        );
    }

    #[test]
    fn cancelled_request_drains_before_the_fixed_slot_is_reused() {
        const SURFACE_OBSERVABLE: &str =
            include_str!("../../../gravlume-reference/fixtures/v2/kerr-surface-observable.toml");
        let fixture = gravlume_reference::FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
            .expect("repository surface fixture parses")
            .into_surface_observation()
            .expect("fixture is a surface observation");
        let observation = fixture.observation();
        let extent = RenderExtent::new(
            observation.view().width().get(),
            observation.view().height().get(),
        )
        .expect("fixture extent is nonzero");
        let gpu = native_gpu();
        let trace = TracePipeline::new(&gpu.device, observation)
            .expect("fixture observation enters the GPU profile");
        let published = published_texture(&gpu.device, extent);
        let mut inspector = SampleInspector::new(&gpu.device, &trace);

        let request = inspector
            .request(
                &gpu.device,
                &gpu.queue,
                &published,
                extent,
                7,
                fixture.sample(),
            )
            .expect("the fixed slot accepts its first request");
        assert!(matches!(
            inspector.request(
                &gpu.device,
                &gpu.queue,
                &published,
                extent,
                7,
                fixture.sample(),
            ),
            Err(SampleInspectionRequestError::Busy)
        ));
        inspector.cancel_active();
        assert!(inspector.has_pending_request());

        let fence = gpu.queue.submit([]);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(fence),
                timeout: None,
            })
            .expect("cancelled inspection submission drains");
        let completion = inspector.wait_for_completion(Some(7));
        assert_same_ticket(completion.ticket(), request);
        assert!(matches!(
            completion.disposition(),
            SampleInspectionDisposition::Cancelled
        ));
        assert!(!inspector.has_pending_request());

        inspector
            .request(
                &gpu.device,
                &gpu.queue,
                &published,
                extent,
                7,
                fixture.sample(),
            )
            .expect("the slot is reusable only after cancellation drains");
    }

    #[test]
    fn completion_binds_ticket_and_fixed_retrace_method() {
        const SURFACE_OBSERVABLE: &str =
            include_str!("../../../gravlume-reference/fixtures/v2/kerr-surface-observable.toml");
        let fixture = gravlume_reference::FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
            .expect("repository surface fixture parses")
            .into_surface_observation()
            .expect("fixture is a surface observation");
        let observation = fixture.observation();
        let extent = RenderExtent::new(
            observation.view().width().get(),
            observation.view().height().get(),
        )
        .expect("fixture extent is nonzero");
        let gpu = native_gpu();
        let trace = TracePipeline::new(&gpu.device, observation)
            .expect("fixture observation enters the GPU profile");
        let published = published_texture(&gpu.device, extent);
        let mut inspector = SampleInspector::new(&gpu.device, &trace);

        let request = inspector
            .request(
                &gpu.device,
                &gpu.queue,
                &published,
                extent,
                11,
                fixture.sample(),
            )
            .expect("inspection request is accepted");
        let fence = gpu.queue.submit([]);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(fence),
                timeout: None,
            })
            .expect("inspection submission completes");
        let completion = inspector.wait_for_completion(Some(11));
        assert_same_ticket(completion.ticket(), request);
        let SampleInspectionDisposition::Completed(inspection) = completion.disposition() else {
            panic!(
                "completed GPU work must decode as a completed inspection, got {:?}",
                completion.disposition()
            );
        };

        assert_eq!(request.generation(), 11);
        assert_eq!(request.extent(), [extent.width(), extent.height()]);
        assert_eq!(request.sample(), fixture.sample());
        assert_eq!(
            SampleRetrace::METHOD_ID,
            "gpu-ks-rk4-v1/full-kerr-schild-retrace/wgsl-binary32"
        );
        assert!(matches!(
            inspection.fresh_retrace().outcome(),
            SampleTraceOutcome::EquatorialSurface { .. }
        ));
        assert_eq!(
            inspection.published_texel().kind(),
            crate::ScientificPixelKind::Horizon,
            "the zero-initialized published texel remains distinct from the fresh retrace"
        );
    }

    #[test]
    fn publication_mismatch_discards_the_result_once_and_releases_the_slot() {
        const SURFACE_OBSERVABLE: &str =
            include_str!("../../../gravlume-reference/fixtures/v2/kerr-surface-observable.toml");
        let fixture = gravlume_reference::FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
            .expect("repository surface fixture parses")
            .into_surface_observation()
            .expect("fixture is a surface observation");
        let observation = fixture.observation();
        let extent = RenderExtent::new(
            observation.view().width().get(),
            observation.view().height().get(),
        )
        .expect("fixture extent is nonzero");
        let gpu = native_gpu();
        let trace = TracePipeline::new(&gpu.device, observation)
            .expect("fixture observation enters the GPU profile");
        let published = published_texture(&gpu.device, extent);
        let mut inspector = SampleInspector::new(&gpu.device, &trace);

        let request = inspector
            .request(
                &gpu.device,
                &gpu.queue,
                &published,
                extent,
                17,
                fixture.sample(),
            )
            .expect("inspection request is accepted");
        let fence = gpu.queue.submit([]);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(fence),
                timeout: None,
            })
            .expect("inspection submission completes");

        let completion = inspector.wait_for_completion(Some(18));
        assert_same_ticket(completion.ticket(), request);
        assert!(matches!(
            completion.disposition(),
            SampleInspectionDisposition::Cancelled
        ));
        assert!(inspector.poll(Some(18)).is_none());
        assert!(!inspector.has_pending_request());
    }

    fn published_texture(device: &wgpu::Device, extent: RenderExtent) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sample inspection published-texel fixture"),
            size: wgpu::Extent3d {
                width: extent.width(),
                height: extent.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn branch_decoder_preserves_arbitrary_committed_values(
            radial_turnings in branch_counter(),
            equatorial_crossings in branch_counter(),
            azimuth_winding in branch_winding(),
        ) {
            let polar_sides = [
                (0, SamplePolarSide::Negative),
                (1, SamplePolarSide::Equatorial),
                (2, SamplePolarSide::Positive),
            ];
            let committed_terminations = [
                TraceTermination::HorizonCrossing,
                TraceTermination::Escape,
                TraceTermination::SingularityGuard,
                TraceTermination::StepExhaustion,
                TraceTermination::EquatorialSurface,
            ];

            for (polar_side_word, initial_polar_side) in polar_sides {
                let words = [
                    radial_turnings,
                    equatorial_crossings,
                    u32::from_ne_bytes(azimuth_winding.to_ne_bytes()),
                    polar_side_word,
                ];
                for termination in committed_terminations {
                    let branch = decode_branch_key(termination, words)
                        .expect("known branch decodes")
                        .expect("committed termination retains its branch");
                    prop_assert_eq!(branch, SampleBranchKey {
                        initial_polar_side,
                        radial_turnings,
                        equatorial_crossings,
                        azimuth_winding,
                    });
                }
                prop_assert_eq!(
                    decode_branch_key(TraceTermination::Uncertain, words)
                        .expect("provisional uncertain branch is recognized"),
                    None
                );
            }
        }

        #[test]
        fn branch_decoder_rejects_invalid_protocol_words(
            payload in (any::<[u32; 4]>(), 0_usize..4),
            radial_turnings: u32,
            equatorial_crossings: u32,
            azimuth_winding: i32,
            unknown_side in 3_u32..=u32::MAX,
        ) {
            let (mut words, nonzero_index) = payload;
            words[nonzero_index] |= 1;

            prop_assert!(matches!(
                decode_branch_key(TraceTermination::NumericalFailure, words),
                Err(SampleInspectionError::InvalidRecord {
                    field: "numerical-failure branch"
                })
            ), "nonzero branch words {words:?} must not decode as a failure sentinel");
            let words = [
                radial_turnings,
                equatorial_crossings,
                u32::from_ne_bytes(azimuth_winding.to_ne_bytes()),
                unknown_side,
            ];

            prop_assert!(matches!(
                decode_branch_key(TraceTermination::StepExhaustion, words),
                Err(SampleInspectionError::UnknownPolarSide(value)) if value == unknown_side
            ));
        }
    }
}

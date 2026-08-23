use std::{
    num::NonZeroU64,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, TryRecvError},
    },
};

use gravlume_domain::ImageSample;
use num_traits::ToPrimitive as _;

use super::{TracePipeline, TracePlan, shader, size_of};
use crate::{
    extent::RenderExtent,
    scientific_capture::{ScientificChannelModel, ScientificTexel},
};

const INSPECTION_REQUEST_BYTES: u64 = size_of::<GpuInspectionRequest>();
const INSPECTION_RECORD_BYTES: u64 = size_of::<GpuInspectionRecord>();
const PUBLISHED_TEXEL_OFFSET: u64 = 256;
const PUBLISHED_TEXEL_BYTES: u64 = 8;
const INSPECTION_READBACK_BYTES: u64 = PUBLISHED_TEXEL_OFFSET + PUBLISHED_TEXEL_BYTES;
const INSPECTION_LOGICAL_BUFFER_BYTES: u64 =
    INSPECTION_REQUEST_BYTES + INSPECTION_RECORD_BYTES + INSPECTION_READBACK_BYTES;
const _: () = {
    assert!(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 256);
    assert!(INSPECTION_RECORD_BYTES <= PUBLISHED_TEXEL_OFFSET);
    assert!(PUBLISHED_TEXEL_BYTES == size_of::<[u16; 4]>());
};
const KNOWN_NUMERICAL_FLAGS: u32 = 1 | 2 | 4;
const KNOWN_EVENT_CANDIDATES: u32 = 1 | 2 | 4 | 8;
static NEXT_OBSERVATION_ID: AtomicU64 = AtomicU64::new(1);

const HORIZON_TAG: u32 = 0.0_f32.to_bits();
const ANALYTIC_ESCAPE_TAG: u32 = 1.0_f32.to_bits();
const SURFACE_RADIANCE_TAG: u32 = 2.0_f32.to_bits();
const SINGULARITY_FAILURE_TAG: u32 = (-3.0_f32).to_bits();
const STEP_EXHAUSTION_FAILURE_TAG: u32 = (-4.0_f32).to_bits();
const NUMERICAL_FAILURE_TAG: u32 = (-5.0_f32).to_bits();
const UNCERTAIN_FAILURE_TAG: u32 = (-6.0_f32).to_bits();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
#[non_exhaustive]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown GPU trace termination discriminant {0}")]
pub struct UnknownTraceTermination(u32);

impl UnknownTraceTermination {
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl TryFrom<u32> for TraceTermination {
    type Error = UnknownTraceTermination;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::HorizonCrossing),
            2 => Ok(Self::Escape),
            3 => Ok(Self::SingularityGuard),
            4 => Ok(Self::StepExhaustion),
            5 => Ok(Self::NumericalFailure),
            6 => Ok(Self::Uncertain),
            7 => Ok(Self::EquatorialSurface),
            unknown => Err(UnknownTraceTermination(unknown)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Opaque identity for one renderer's immutable observation.
///
/// This value is process-local and must not be used as a persisted artifact identity.
pub struct SampleObservationId(NonZeroU64);

impl SampleObservationId {
    pub(crate) fn allocate() -> Option<Self> {
        let raw = NEXT_OBSERVATION_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .ok()?;
        NonZeroU64::new(raw).map(Self)
    }

    #[cfg(test)]
    const fn for_test(raw: u64) -> Self {
        Self(NonZeroU64::new(raw).expect("test observation identity is nonzero"))
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Opaque identity for one request within a renderer instance.
pub struct SampleInspectionRequestId(NonZeroU64);

impl SampleInspectionRequestId {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SampleInspectionProfile {
    GpuKerrSchildRk4V1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SampleInspectionProducer {
    FullKerrSchildRetrace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SampleArithmeticDomain {
    WgslBinary32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Host-owned identity captured when one inspection request is admitted.
///
/// The observation and request identifiers are process-local. Persisted scientific artifacts need
/// their own canonical observation, producer revision, and backend identity.
pub struct SampleInspectionIdentity {
    request_id: SampleInspectionRequestId,
    observation_id: SampleObservationId,
    generation: u64,
    extent: [u32; 2],
    sample: ImageSample,
}

impl SampleInspectionIdentity {
    const fn new(
        request_id: SampleInspectionRequestId,
        observation_id: SampleObservationId,
        generation: u64,
        extent: RenderExtent,
        sample: ImageSample,
    ) -> Self {
        Self {
            request_id,
            observation_id,
            generation,
            extent: [extent.width(), extent.height()],
            sample,
        }
    }

    #[must_use]
    pub const fn request_id(self) -> SampleInspectionRequestId {
        self.request_id
    }

    #[must_use]
    pub const fn observation_id(self) -> SampleObservationId {
        self.observation_id
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

    #[must_use]
    pub const fn profile(self) -> SampleInspectionProfile {
        SampleInspectionProfile::GpuKerrSchildRk4V1
    }

    #[must_use]
    pub const fn producer(self) -> SampleInspectionProducer {
        SampleInspectionProducer::FullKerrSchildRetrace
    }

    #[must_use]
    pub const fn arithmetic_domain(self) -> SampleArithmeticDomain {
        SampleArithmeticDomain::WgslBinary32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleInspectionLimits {
    _private: (),
}

impl SampleInspectionLimits {
    #[must_use]
    pub(crate) const fn production() -> Self {
        Self { _private: () }
    }

    #[must_use]
    pub const fn maximum_pending_requests(self) -> u32 {
        1
    }

    #[must_use]
    pub const fn request_buffer_bytes(self) -> u64 {
        INSPECTION_REQUEST_BYTES
    }

    #[must_use]
    pub const fn record_buffer_bytes(self) -> u64 {
        INSPECTION_RECORD_BYTES
    }

    #[must_use]
    pub const fn readback_buffer_bytes(self) -> u64 {
        INSPECTION_READBACK_BYTES
    }

    #[must_use]
    pub const fn maximum_logical_buffer_bytes(self) -> u64 {
        INSPECTION_LOGICAL_BUFFER_BYTES
    }

    #[must_use]
    pub const fn readback_range_bytes(self) -> u64 {
        INSPECTION_READBACK_BYTES
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
#[non_exhaustive]
pub enum SampleInspectionSource {
    None,
    /// A normalized orientation used by the analytic sky preview, not a physical spectrum.
    AnalyticEscape {
        unit_direction: [f32; 3],
    },
    EquatorialSurface {
        radius_over_m: f32,
        azimuth_radians: f32,
        frequency_ratio: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SampleSceneValue {
    Horizon,
    /// Scene-linear orientation preview; these channels are not spectral radiance.
    AnalyticEscapePreview([f32; 3]),
    SurfaceRadiance([f32; 3]),
    TraceFailure {
        termination: TraceTermination,
        visible_rgb: [f32; 3],
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleInspection {
    identity: SampleInspectionIdentity,
    published_texel: ScientificTexel,
    pub(crate) channel_model: Option<ScientificChannelModel>,
    pub(crate) termination: TraceTermination,
    pub(crate) source: SampleInspectionSource,
    pub(crate) scene_value: SampleSceneValue,
    pub(crate) branch_key: Option<SampleBranchKey>,
    pub(crate) travel_time_over_m: f32,
    pub(crate) event_candidates: u32,
    pub(crate) event_residual: f32,
    pub(crate) steps: u32,
    pub(crate) numerical_flags: u32,
    pub(crate) maximum_invariant_drift: [f32; 4],
}

impl SampleInspection {
    #[must_use]
    pub const fn identity(&self) -> SampleInspectionIdentity {
        self.identity
    }

    /// Returns the exact `Rgba16Float` texel copied from the bound published generation.
    #[must_use]
    pub const fn published_texel(&self) -> ScientificTexel {
        self.published_texel
    }

    #[must_use]
    pub const fn channel_model(&self) -> Option<ScientificChannelModel> {
        self.channel_model
    }

    #[must_use]
    pub const fn termination(&self) -> TraceTermination {
        self.termination
    }

    #[must_use]
    pub const fn source(&self) -> SampleInspectionSource {
        self.source
    }

    /// Returns the f32 result of the fresh full Kerr-Schild retrace.
    ///
    /// This is deliberately separate from [`Self::published_texel`], which may include a
    /// conservative accelerator, shadow refinement, and `Rgba16Float` rounding.
    #[must_use]
    pub const fn evaluated_scene_value(&self) -> SampleSceneValue {
        self.scene_value
    }

    #[must_use]
    pub const fn branch_key(&self) -> Option<SampleBranchKey> {
        self.branch_key
    }

    #[must_use]
    pub const fn travel_time_over_m(&self) -> f32 {
        self.travel_time_over_m
    }

    #[must_use]
    pub const fn event_candidate_bits(&self) -> u32 {
        self.event_candidates
    }

    #[must_use]
    pub const fn event_residual(&self) -> f32 {
        self.event_residual
    }

    #[must_use]
    pub const fn steps(&self) -> u32 {
        self.steps
    }

    #[must_use]
    pub const fn numerical_flag_bits(&self) -> u32 {
        self.numerical_flags
    }

    #[must_use]
    pub const fn maximum_invariant_drift(&self) -> [f32; 4] {
        self.maximum_invariant_drift
    }
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SampleInspectionError {
    #[error(transparent)]
    UnknownTermination(#[from] UnknownTraceTermination),
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
#[non_exhaustive]
pub enum SampleInspectionRequestError {
    #[error("no complete scene generation has been published")]
    NoPublishedScene,
    #[error(
        "published scene generation {published} is not the current renderer generation {current}"
    )]
    PublishedGenerationStale { published: u64, current: u64 },
    #[error("sample pixel {pixel:?} lies outside the published extent {extent:?}")]
    SampleOutsideExtent { pixel: [u32; 2], extent: [u32; 2] },
    #[error("sample inspection request {active:?} is still in flight")]
    Busy { active: SampleInspectionRequestId },
    #[error("validated sample subpixel coordinate {field} cannot enter WGSL binary32")]
    SubpixelNotRepresentable { field: &'static str },
    #[error("sample inspection request identity space is exhausted")]
    RequestIdentityExhausted,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum SampleInspectionEvent {
    Completed(SampleInspection),
    Cancelled(SampleInspectionIdentity),
    Superseded(SampleInspectionIdentity),
    Failed {
        identity: SampleInspectionIdentity,
        error: SampleInspectionError,
    },
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
struct GpuInspectionRequest {
    pixel_extent: [u32; 4],
    subpixel: [f32; 4],
}

impl GpuInspectionRequest {
    fn new(
        sample: ImageSample,
        extent: RenderExtent,
    ) -> Result<Self, SampleInspectionRequestError> {
        let [pixel_x, pixel_y] = sample.pixel();
        let [subpixel_x, subpixel_y] = sample.subpixel();
        let subpixel_x =
            subpixel_x
                .to_f32()
                .ok_or(SampleInspectionRequestError::SubpixelNotRepresentable {
                    field: "subpixel_x",
                })?;
        let subpixel_y =
            subpixel_y
                .to_f32()
                .ok_or(SampleInspectionRequestError::SubpixelNotRepresentable {
                    field: "subpixel_y",
                })?;
        Ok(Self {
            pixel_extent: [pixel_x, pixel_y, extent.width(), extent.height()],
            subpixel: [subpixel_x, subpixel_y, 0.0, 0.0],
        })
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
    identity: SampleInspectionIdentity,
    cancelled: bool,
}

pub struct SampleInspector {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    request: wgpu::Buffer,
    record: wgpu::Buffer,
    readback: wgpu::Buffer,
    observation_id: SampleObservationId,
    channel_model: Option<ScientificChannelModel>,
    next_request_id: Option<NonZeroU64>,
    pending: Option<PendingInspection>,
}

impl SampleInspector {
    pub(crate) fn new(
        device: &wgpu::Device,
        trace: &TracePipeline,
        observation_id: SampleObservationId,
    ) -> Self {
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
            observation_id,
            channel_model: trace
                .scientific_capture_metadata()
                .map(crate::scientific_capture::ScientificCaptureMetadata::channels),
            next_request_id: Some(NonZeroU64::MIN),
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
    ) -> Result<SampleInspectionRequestId, SampleInspectionRequestError> {
        if let Some(pending) = &self.pending {
            return Err(SampleInspectionRequestError::Busy {
                active: pending.identity.request_id(),
            });
        }
        let pixel = sample.pixel();
        let extent_array = [extent.width(), extent.height()];
        if pixel[0] >= extent.width() || pixel[1] >= extent.height() {
            return Err(SampleInspectionRequestError::SampleOutsideExtent {
                pixel,
                extent: extent_array,
            });
        }
        let request_id = self
            .next_request_id
            .map(SampleInspectionRequestId)
            .ok_or(SampleInspectionRequestError::RequestIdentityExhausted)?;
        let request = GpuInspectionRequest::new(sample, extent)?;
        let identity = SampleInspectionIdentity::new(
            request_id,
            self.observation_id,
            generation,
            extent,
            sample,
        );
        self.next_request_id = request_id.get().checked_add(1).and_then(NonZeroU64::new);
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
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: Some(1),
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
            identity,
            cancelled: false,
        });
        Ok(request_id)
    }

    pub(crate) fn cancel(&mut self, request_id: SampleInspectionRequestId) -> bool {
        let Some(pending) = self.pending.as_mut() else {
            return false;
        };
        if pending.identity.request_id() != request_id {
            return false;
        }
        pending.cancelled = true;
        true
    }

    pub(crate) const fn cancel_active(&mut self) {
        if let Some(pending) = self.pending.as_mut() {
            pending.cancelled = true;
        }
    }

    pub(crate) fn poll(
        &mut self,
        accepted_generation: Option<u64>,
    ) -> Option<SampleInspectionEvent> {
        let pending = self.pending.take()?;
        let map_result = match pending.receiver.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => {
                self.pending = Some(pending);
                return None;
            }
            Err(TryRecvError::Disconnected) => {
                self.readback.unmap();
                return Some(Self::disposition_or_failure(
                    &pending,
                    accepted_generation,
                    SampleInspectionError::CallbackDisconnected,
                ));
            }
        };

        let disposition = if pending.cancelled {
            Some(SampleInspectionEvent::Cancelled(pending.identity))
        } else if accepted_generation != Some(pending.identity.generation()) {
            Some(SampleInspectionEvent::Superseded(pending.identity))
        } else {
            None
        };
        if let Some(event) = disposition {
            self.readback.unmap();
            return Some(event);
        }
        if let Err(error) = map_result {
            self.readback.unmap();
            return Some(SampleInspectionEvent::Failed {
                identity: pending.identity,
                error: error.into(),
            });
        }

        let result = self.read_inspection(pending.identity);
        self.readback.unmap();
        Some(match result {
            Ok(inspection) => SampleInspectionEvent::Completed(inspection),
            Err(error) => SampleInspectionEvent::Failed {
                identity: pending.identity,
                error,
            },
        })
    }

    fn disposition_or_failure(
        pending: &PendingInspection,
        accepted_generation: Option<u64>,
        error: SampleInspectionError,
    ) -> SampleInspectionEvent {
        if pending.cancelled {
            SampleInspectionEvent::Cancelled(pending.identity)
        } else if accepted_generation != Some(pending.identity.generation()) {
            SampleInspectionEvent::Superseded(pending.identity)
        } else {
            SampleInspectionEvent::Failed {
                identity: pending.identity,
                error,
            }
        }
    }

    fn read_inspection(
        &self,
        identity: SampleInspectionIdentity,
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
            identity,
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
        let mut inspector =
            SampleInspector::new(&gpu.device, self, SampleObservationId::for_test(1));
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
        match inspector.poll(Some(1)) {
            Some(SampleInspectionEvent::Completed(inspection)) => inspection,
            event => panic!("test inspection must complete, got {event:?}"),
        }
    }
}

fn decode_record(
    raw: GpuInspectionRecord,
    channel_model: Option<ScientificChannelModel>,
    identity: SampleInspectionIdentity,
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

    let termination = TraceTermination::try_from(raw.metadata[0])?;
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

    Ok(SampleInspection {
        identity,
        published_texel,
        channel_model,
        termination,
        source: decode_source(termination, raw.source_time)?,
        scene_value: decode_scene_value(termination, raw.scene_value)?,
        branch_key: decode_branch_key(termination, raw.branch_key)?,
        travel_time_over_m: raw.source_time[3],
        event_candidates,
        event_residual: raw.event_diagnostics[0],
        steps: raw.metadata[2],
        numerical_flags,
        maximum_invariant_drift: raw.maximum_invariant_drift,
    })
}

fn decode_source(
    termination: TraceTermination,
    source_time: [f32; 4],
) -> Result<SampleInspectionSource, SampleInspectionError> {
    let [source_x, source_y, source_z, _] = source_time;
    match termination {
        TraceTermination::Escape => Ok(SampleInspectionSource::AnalyticEscape {
            unit_direction: [source_x, source_y, source_z],
        }),
        TraceTermination::EquatorialSurface => {
            if source_x <= 0.0 || source_z <= 0.0 {
                return Err(SampleInspectionError::InvalidRecord {
                    field: "surface source",
                });
            }
            Ok(SampleInspectionSource::EquatorialSurface {
                radius_over_m: source_x,
                azimuth_radians: source_y,
                frequency_ratio: source_z,
            })
        }
        TraceTermination::HorizonCrossing
        | TraceTermination::SingularityGuard
        | TraceTermination::StepExhaustion
        | TraceTermination::NumericalFailure
        | TraceTermination::Uncertain => Ok(SampleInspectionSource::None),
    }
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

fn decode_scene_value(
    termination: TraceTermination,
    value: [f32; 4],
) -> Result<SampleSceneValue, SampleInspectionError> {
    let [red, green, blue, alpha] = value;
    let rgb = [red, green, blue];
    let tag = alpha.to_bits();

    if termination == TraceTermination::HorizonCrossing && tag == HORIZON_TAG {
        if rgb.map(f32::to_bits) != [0; 3] {
            return Err(SampleInspectionError::InvalidRecord {
                field: "horizon scene value",
            });
        }
        return Ok(SampleSceneValue::Horizon);
    }
    if termination == TraceTermination::Escape && tag == ANALYTIC_ESCAPE_TAG {
        return Ok(SampleSceneValue::AnalyticEscapePreview(rgb));
    }
    if termination == TraceTermination::EquatorialSurface && tag == SURFACE_RADIANCE_TAG {
        if rgb.into_iter().any(|channel| channel < 0.0) {
            return Err(SampleInspectionError::InvalidRecord {
                field: "surface radiance",
            });
        }
        return Ok(SampleSceneValue::SurfaceRadiance(rgb));
    }

    let visible_termination = match tag {
        SINGULARITY_FAILURE_TAG => TraceTermination::SingularityGuard,
        STEP_EXHAUSTION_FAILURE_TAG => TraceTermination::StepExhaustion,
        NUMERICAL_FAILURE_TAG => TraceTermination::NumericalFailure,
        UNCERTAIN_FAILURE_TAG => TraceTermination::Uncertain,
        _ => {
            return Err(SampleInspectionError::InvalidRecord { field: "scene tag" });
        }
    };
    let valid_failure = match termination {
        TraceTermination::EquatorialSurface => {
            visible_termination == TraceTermination::NumericalFailure
        }
        TraceTermination::SingularityGuard
        | TraceTermination::StepExhaustion
        | TraceTermination::NumericalFailure
        | TraceTermination::Uncertain => visible_termination == termination,
        TraceTermination::HorizonCrossing | TraceTermination::Escape => false,
    };
    if !valid_failure {
        return Err(SampleInspectionError::InvalidRecord {
            field: "termination/scene tag",
        });
    }
    Ok(SampleSceneValue::TraceFailure {
        termination: visible_termination,
        visible_rgb: rgb,
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{
        SampleArithmeticDomain, SampleBranchKey, SampleInspectionError, SampleInspectionEvent,
        SampleInspectionLimits, SampleInspectionProducer, SampleInspectionProfile,
        SampleInspectionRequestError, SampleInspector, SampleObservationId, SamplePolarSide,
        decode_branch_key,
    };
    use crate::{
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

    #[test]
    fn numerical_failure_uses_an_explicit_zero_branch_sentinel() {
        assert_eq!(
            decode_branch_key(TraceTermination::NumericalFailure, [0; 4])
                .expect("zero failure sentinel decodes"),
            None
        );
    }

    #[test]
    fn production_limits_keep_inspection_constant_and_single_flight() {
        let limits = SampleInspectionLimits::production();

        assert_eq!(limits.maximum_pending_requests(), 1);
        assert_eq!(limits.request_buffer_bytes(), 32);
        assert_eq!(limits.record_buffer_bytes(), 96);
        assert_eq!(limits.readback_buffer_bytes(), 264);
        assert_eq!(limits.maximum_logical_buffer_bytes(), 392);
        assert_eq!(limits.readback_range_bytes(), 264);
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
        let mut inspector =
            SampleInspector::new(&gpu.device, &trace, SampleObservationId::for_test(41));

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
            Err(SampleInspectionRequestError::Busy { active }) if active == request
        ));
        assert!(inspector.cancel(request));
        assert!(inspector.has_pending_request());

        let fence = gpu.queue.submit([]);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(fence),
                timeout: None,
            })
            .expect("cancelled inspection submission drains");
        let event = inspector
            .poll(Some(7))
            .expect("drained cancellation produces one event");
        assert!(matches!(
            event,
            SampleInspectionEvent::Cancelled(identity)
                if identity.request_id() == request && identity.generation() == 7
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
    fn completion_binds_observation_generation_profile_and_producer() {
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
        let observation_id = SampleObservationId::for_test(73);
        let mut inspector = SampleInspector::new(&gpu.device, &trace, observation_id);

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
        let SampleInspectionEvent::Completed(inspection) = inspector
            .poll(Some(11))
            .expect("completion produces one event")
        else {
            panic!("completed GPU work must decode as a completed inspection");
        };
        let identity = inspection.identity();

        assert_eq!(identity.request_id(), request);
        assert_eq!(identity.observation_id(), observation_id);
        assert_eq!(identity.generation(), 11);
        assert_eq!(identity.extent(), [extent.width(), extent.height()]);
        assert_eq!(identity.sample(), fixture.sample());
        assert_eq!(
            identity.profile(),
            SampleInspectionProfile::GpuKerrSchildRk4V1
        );
        assert_eq!(
            identity.producer(),
            SampleInspectionProducer::FullKerrSchildRetrace
        );
        assert_eq!(
            identity.arithmetic_domain(),
            SampleArithmeticDomain::WgslBinary32
        );
        assert_eq!(
            inspection.termination(),
            TraceTermination::EquatorialSurface
        );
        assert!(inspection.branch_key().is_some());
        assert_eq!(
            inspection.published_texel().kind(),
            crate::ScientificPixelKind::Horizon,
            "the zero-initialized published texel remains distinct from the fresh retrace"
        );
    }

    #[test]
    fn publication_change_supersedes_the_result_once_and_releases_the_slot() {
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
        let mut inspector =
            SampleInspector::new(&gpu.device, &trace, SampleObservationId::for_test(89));

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

        assert!(matches!(
            inspector.poll(Some(18)),
            Some(SampleInspectionEvent::Superseded(identity))
                if identity.request_id() == request && identity.generation() == 17
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

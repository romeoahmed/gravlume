use std::sync::mpsc::{self, Receiver, TryRecvError};

use num_traits::ToPrimitive as _;
use wgpu::util::DeviceExt as _;

use super::{TracePipeline, TracePlan, TraceUniforms, shader, size_of};
use crate::{extent::RenderExtent, scientific_capture::ScientificChannelModel};

const INSPECTION_ABI_VERSION: u32 = 1;
const INSPECTION_PRODUCER_TAG: u32 = 1;
const INSPECTION_DOMAIN_TAG: u32 = 1;
const INSPECTION_REQUEST_BYTES: u64 = size_of::<GpuInspectionRequest>();
const INSPECTION_RECORD_BYTES: u64 = size_of::<GpuInspectionRecord>();
#[cfg(test)]
const INSPECTION_LOGICAL_BUFFER_BYTES: u64 = INSPECTION_REQUEST_BYTES + 2 * INSPECTION_RECORD_BYTES;

/// One requested sample of an already-published scene generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleInspectionRequest {
    generation: u64,
    pixel: [u32; 2],
    subpixel: [f64; 2],
}

impl SampleInspectionRequest {
    #[must_use]
    pub const fn new(generation: u64, pixel: [u32; 2], subpixel: [f64; 2]) -> Self {
        Self {
            generation,
            pixel,
            subpixel,
        }
    }

    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }

    #[must_use]
    pub const fn pixel(self) -> [u32; 2] {
        self.pixel
    }

    #[must_use]
    pub const fn subpixel(self) -> [f64; 2] {
        self.subpixel
    }
}

/// Renderer-local identity for one inspection attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SampleInspectionRequestId(u64);

impl SampleInspectionRequestId {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Exact binary32 input words consumed by the GPU trace profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleObservationIdentity {
    words: [u32; 44],
}

impl SampleObservationIdentity {
    pub(super) fn from_uniforms(uniforms: TraceUniforms) -> Self {
        Self {
            words: bytemuck::cast(uniforms),
        }
    }

    #[must_use]
    pub const fn words(&self) -> &[u32; 44] {
        &self.words
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleInspectionProfile {
    GpuKsRk4V1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleProducer {
    /// A fresh single-invocation trace through the production full Kerr-Schild solver.
    OnDemandFullKerrSchildRetrace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleArithmeticDomain {
    WgslF32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SampleTermination {
    HorizonCrossing = 1,
    Escape = 2,
    SingularityGuard = 3,
    StepExhaustion = 4,
    NumericalFailure = 5,
    Uncertain = 6,
    EquatorialSurface = 7,
}

impl TryFrom<u32> for SampleTermination {
    type Error = SampleInspectionError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::HorizonCrossing),
            2 => Ok(Self::Escape),
            3 => Ok(Self::SingularityGuard),
            4 => Ok(Self::StepExhaustion),
            5 => Ok(Self::NumericalFailure),
            6 => Ok(Self::Uncertain),
            7 => Ok(Self::EquatorialSurface),
            unknown => Err(SampleInspectionError::UnknownTermination(unknown)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplePolarSide {
    Negative,
    Equatorial,
    Positive,
}

impl SamplePolarSide {
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::Negative => 0,
            Self::Equatorial => 1,
            Self::Positive => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleBranchKey {
    initial_polar_side: SamplePolarSide,
    radial_turnings: u32,
    equatorial_crossings: u32,
    azimuth_winding: i32,
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
pub enum SampleSceneValue {
    Horizon,
    /// Scene-linear orientation preview; these channels are not spectral radiance.
    AnalyticEscapePreview([f32; 3]),
    /// Scene-linear source output interpreted by [`SampleInspection::channel_model`].
    SurfaceRadiance([f32; 3]),
    TraceFailure {
        termination: SampleTermination,
        visible_rgb: [f32; 3],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleEventCandidates(u32);

impl SampleEventCandidates {
    const SINGULARITY: u32 = 1;
    const HORIZON: u32 = 1 << 1;
    const EQUATORIAL_SURFACE: u32 = 1 << 2;
    const ESCAPE: u32 = 1 << 3;
    const KNOWN: u32 = Self::SINGULARITY | Self::HORIZON | Self::EQUATORIAL_SURFACE | Self::ESCAPE;

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn singularity(self) -> bool {
        self.0 & Self::SINGULARITY != 0
    }

    #[must_use]
    pub const fn horizon(self) -> bool {
        self.0 & Self::HORIZON != 0
    }

    #[must_use]
    pub const fn equatorial_surface(self) -> bool {
        self.0 & Self::EQUATORIAL_SURFACE != 0
    }

    #[must_use]
    pub const fn escape(self) -> bool {
        self.0 & Self::ESCAPE != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleEventDiagnostics {
    candidates: SampleEventCandidates,
    residual: f32,
}

impl SampleEventDiagnostics {
    #[must_use]
    pub const fn candidates(self) -> SampleEventCandidates {
        self.candidates
    }

    #[must_use]
    pub const fn ambiguous(self) -> bool {
        self.candidates.bits().count_ones() > 1
    }

    #[must_use]
    pub const fn residual(self) -> f32 {
        self.residual
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleNumericalFlags(u32);

impl SampleNumericalFlags {
    const KNOWN: u32 = 1 | 2 | 4;

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn non_finite(self) -> bool {
        self.0 & 1 != 0
    }

    #[must_use]
    pub const fn invalid_radicand(self) -> bool {
        self.0 & 2 != 0
    }

    #[must_use]
    pub const fn invalid_denominator(self) -> bool {
        self.0 & 4 != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleInvariantDrift([f32; 4]);

impl SampleInvariantDrift {
    #[must_use]
    pub const fn normalized_null(self) -> f32 {
        self.0[0]
    }

    #[must_use]
    pub const fn energy(self) -> f32 {
        self.0[1]
    }

    #[must_use]
    pub const fn angular_momentum_z(self) -> f32 {
        self.0[2]
    }

    #[must_use]
    pub const fn carter(self) -> f32 {
        self.0[3]
    }

    #[must_use]
    pub const fn as_array(self) -> [f32; 4] {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleNumericalDiagnostics {
    steps: u32,
    flags: SampleNumericalFlags,
    maximum_invariant_drift: SampleInvariantDrift,
}

impl SampleNumericalDiagnostics {
    #[must_use]
    pub const fn steps(self) -> u32 {
        self.steps
    }

    #[must_use]
    pub const fn flags(self) -> SampleNumericalFlags {
        self.flags
    }

    #[must_use]
    pub const fn maximum_invariant_drift(self) -> SampleInvariantDrift {
        self.maximum_invariant_drift
    }
}

/// A bounded, scene-linear result from one fresh production-profile GPU trace.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleInspection {
    request_id: SampleInspectionRequestId,
    request: SampleInspectionRequest,
    extent: [u32; 2],
    observation_identity: SampleObservationIdentity,
    channel_model: Option<ScientificChannelModel>,
    termination: SampleTermination,
    source: SampleInspectionSource,
    scene_value: SampleSceneValue,
    branch_key: SampleBranchKey,
    travel_time_over_m: f32,
    event_diagnostics: SampleEventDiagnostics,
    numerical_diagnostics: SampleNumericalDiagnostics,
}

impl SampleInspection {
    #[must_use]
    pub const fn request_id(&self) -> SampleInspectionRequestId {
        self.request_id
    }

    #[must_use]
    pub const fn request(&self) -> SampleInspectionRequest {
        self.request
    }

    #[must_use]
    pub const fn extent(&self) -> [u32; 2] {
        self.extent
    }

    #[must_use]
    pub const fn observation_identity(&self) -> &SampleObservationIdentity {
        &self.observation_identity
    }

    #[must_use]
    pub const fn profile(&self) -> SampleInspectionProfile {
        SampleInspectionProfile::GpuKsRk4V1
    }

    #[must_use]
    pub const fn producer(&self) -> SampleProducer {
        SampleProducer::OnDemandFullKerrSchildRetrace
    }

    #[must_use]
    pub const fn arithmetic_domain(&self) -> SampleArithmeticDomain {
        SampleArithmeticDomain::WgslF32
    }

    /// Returns the physical interpretation of surface RGB, or `None` for analytic-sky scenes.
    #[must_use]
    pub const fn channel_model(&self) -> Option<ScientificChannelModel> {
        self.channel_model
    }

    #[must_use]
    pub const fn termination(&self) -> SampleTermination {
        self.termination
    }

    #[must_use]
    pub const fn source(&self) -> SampleInspectionSource {
        self.source
    }

    #[must_use]
    pub const fn scene_value(&self) -> SampleSceneValue {
        self.scene_value
    }

    #[must_use]
    pub const fn branch_key(&self) -> SampleBranchKey {
        self.branch_key
    }

    #[must_use]
    pub const fn travel_time_over_m(&self) -> f32 {
        self.travel_time_over_m
    }

    #[must_use]
    pub const fn event_diagnostics(&self) -> SampleEventDiagnostics {
        self.event_diagnostics
    }

    #[must_use]
    pub const fn numerical_diagnostics(&self) -> SampleNumericalDiagnostics {
        self.numerical_diagnostics
    }
}

/// Terminal host state for a submitted inspection request.
#[derive(Debug)]
pub enum SampleInspectionOutcome {
    Completed(Box<SampleInspection>),
    Cancelled {
        request_id: SampleInspectionRequestId,
    },
    Superseded {
        request_id: SampleInspectionRequestId,
        requested_generation: u64,
        published_generation: Option<u64>,
    },
    Failed {
        request_id: SampleInspectionRequestId,
        error: SampleInspectionError,
    },
}

impl SampleInspectionOutcome {
    #[must_use]
    pub const fn request_id(&self) -> SampleInspectionRequestId {
        match self {
            Self::Completed(inspection) => inspection.request_id(),
            Self::Cancelled { request_id }
            | Self::Superseded { request_id, .. }
            | Self::Failed { request_id, .. } => *request_id,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SampleInspectionError {
    #[error("no complete scene generation has been published")]
    NoPublishedScene,
    #[error(
        "sample inspection requested generation {requested}, but generation {published} is published"
    )]
    GenerationMismatch { requested: u64, published: u64 },
    #[error("sample inspection request {active:?} is still in flight")]
    Busy { active: SampleInspectionRequestId },
    #[error("there is no in-flight sample inspection request")]
    NoActiveRequest,
    #[error("sample inspection request {requested:?} does not match active request {active:?}")]
    RequestMismatch {
        requested: SampleInspectionRequestId,
        active: SampleInspectionRequestId,
    },
    #[error("sample inspection request identity space is exhausted")]
    RequestIdentityExhausted,
    #[error("sample pixel {pixel:?} is outside the published extent {extent:?}")]
    PixelOutsideExtent { pixel: [u32; 2], extent: [u32; 2] },
    #[error("sample subpixel coordinate {coordinate} is not finite")]
    NonFiniteSubpixel { coordinate: usize },
    #[error("sample subpixel coordinate {coordinate} must lie in [0, 1], got {value}")]
    SubpixelOutsideRange { coordinate: usize, value: f64 },
    #[error("sample subpixel coordinate {coordinate} cannot be represented as binary32")]
    SubpixelNotRepresentable { coordinate: usize },
    #[error("sample inspection buffer mapping failed: {0}")]
    Map(#[from] wgpu::BufferAsyncError),
    #[error("sample inspection mapped-range access failed: {0}")]
    BufferAccess(#[from] wgpu::MapRangeError),
    #[error("sample inspection map callback was dropped")]
    MapCallbackDropped,
    #[error("GPU sample inspection returned ABI version {actual}, expected {expected}")]
    UnsupportedRecordVersion { actual: u32, expected: u32 },
    #[error("GPU sample inspection returned producer tag {0}")]
    UnknownProducer(u32),
    #[error("GPU sample inspection returned arithmetic-domain tag {0}")]
    UnknownArithmeticDomain(u32),
    #[error("GPU sample inspection returned output-kind tag {0}")]
    UnknownOutputKind(u32),
    #[error("GPU sample inspection returned an identity echo for another request")]
    IdentityMismatch,
    #[error("unknown GPU sample termination discriminant {0}")]
    UnknownTermination(u32),
    #[error("unknown GPU sample initial polar-side discriminant {0}")]
    UnknownPolarSide(u32),
    #[error("GPU sample inspection returned an invalid {field} record")]
    InvalidRecord { field: &'static str },
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
struct GpuInspectionRequest {
    pixel_extent: [u32; 4],
    subpixel: [f32; 4],
    identity: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuInspectionRequest>() == 48);
const _: () = assert!(std::mem::align_of::<GpuInspectionRequest>() == 16);

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
struct GpuInspectionRecord {
    identity: [u32; 4],
    protocol: [u32; 4],
    metadata: [u32; 4],
    branch_key: [u32; 4],
    source_time: [f32; 4],
    scene_event: [f32; 4],
    maximum_invariant_drift: [f32; 4],
}

// Seven vec4 lanes give the storage record an exact, padding-free 16-byte stride contract.
// Source: https://www.w3.org/TR/WGSL/#alignment-and-size
const _: () = assert!(std::mem::size_of::<GpuInspectionRecord>() == 112);
const _: () = assert!(std::mem::align_of::<GpuInspectionRecord>() == 16);

#[derive(Clone, Copy, Debug)]
struct InspectionContext {
    request_id: SampleInspectionRequestId,
    request: SampleInspectionRequest,
    extent: RenderExtent,
}

struct PendingInspection {
    context: InspectionContext,
    receiver: Receiver<Result<(), wgpu::BufferAsyncError>>,
    cancelled: bool,
}

#[cfg(test)]
struct SubmittedInspection {
    request_id: SampleInspectionRequestId,
    submission: wgpu::SubmissionIndex,
}

pub struct SampleInspectionPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    request: wgpu::Buffer,
    record: wgpu::Buffer,
    readback: wgpu::Buffer,
    observation_identity: SampleObservationIdentity,
    channel_model: Option<ScientificChannelModel>,
    next_request_id: Option<u64>,
    pending: Option<PendingInspection>,
}

struct InspectionBuffers {
    request: wgpu::Buffer,
    record: wgpu::Buffer,
    readback: wgpu::Buffer,
}

fn create_inspection_bind_group_layout(
    device: &wgpu::Device,
    has_blackbody_lut: bool,
) -> wgpu::BindGroupLayout {
    let buffer_entry = |binding, ty, minimum_size| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(minimum_size),
        },
        count: None,
    };
    let mut entries = vec![
        buffer_entry(
            0,
            wgpu::BufferBindingType::Uniform,
            size_of::<TraceUniforms>(),
        ),
        buffer_entry(
            3,
            wgpu::BufferBindingType::Storage { read_only: false },
            INSPECTION_RECORD_BYTES,
        ),
    ];
    if has_blackbody_lut {
        entries.push(buffer_entry(
            8,
            wgpu::BufferBindingType::Storage { read_only: true },
            crate::spectral_lut::BLACKBODY_LUT_BYTE_SIZE,
        ));
    }
    entries.push(buffer_entry(
        9,
        wgpu::BufferBindingType::Uniform,
        INSPECTION_REQUEST_BYTES,
    ));
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sample inspection bind group layout"),
        entries: &entries,
    })
}

fn create_inspection_compute_pipeline(
    device: &wgpu::Device,
    trace: &TracePipeline,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let (shader_source, entry_point) = match trace.plan {
        TracePlan::AcceleratedSky => (
            shader::analytic_sample_inspection(),
            "inspect_analytic_sample",
        ),
        TracePlan::EquatorialBolometricSurface => (
            shader::bolometric_sample_inspection(),
            "inspect_surface_sample",
        ),
        TracePlan::EquatorialBlackbodySurface => (
            shader::blackbody_sample_inspection(),
            "inspect_surface_sample",
        ),
    };
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sample inspection pipeline layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bounded sample inspection shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline_constants = [(
        "SURFACE_EVENTS_ENABLED",
        trace.plan.surface_events_enabled(),
    )];
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &pipeline_constants,
            ..Default::default()
        },
        cache: None,
    })
}

fn create_inspection_buffers(device: &wgpu::Device) -> InspectionBuffers {
    let request = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sample inspection request"),
        contents: bytemuck::bytes_of(&GpuInspectionRequest {
            pixel_extent: [0; 4],
            subpixel: [0.5, 0.5, 0.0, 0.0],
            identity: [0; 4],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let record = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sample inspection record"),
        size: INSPECTION_RECORD_BYTES,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sample inspection readback"),
        size: INSPECTION_RECORD_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    InspectionBuffers {
        request,
        record,
        readback,
    }
}

fn create_inspection_bind_group(
    device: &wgpu::Device,
    trace: &TracePipeline,
    layout: &wgpu::BindGroupLayout,
    buffers: &InspectionBuffers,
) -> wgpu::BindGroup {
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: trace.uniforms.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: buffers.record.as_entire_binding(),
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
        resource: buffers.request.as_entire_binding(),
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sample inspection bind group"),
        layout,
        entries: &entries,
    })
}

impl SampleInspectionPipeline {
    pub(super) fn new(
        device: &wgpu::Device,
        trace: &TracePipeline,
        observation_identity: SampleObservationIdentity,
    ) -> Self {
        let bind_group_layout =
            create_inspection_bind_group_layout(device, trace.blackbody_lut.is_some());
        let pipeline = create_inspection_compute_pipeline(device, trace, &bind_group_layout);
        let buffers = create_inspection_buffers(device);
        let bind_group = create_inspection_bind_group(device, trace, &bind_group_layout, &buffers);
        Self {
            pipeline,
            bind_group,
            request: buffers.request,
            record: buffers.record,
            readback: buffers.readback,
            observation_identity,
            channel_model: trace
                .scientific_capture_metadata()
                .map(crate::scientific_capture::ScientificCaptureMetadata::channels),
            next_request_id: Some(1),
            pending: None,
        }
    }

    pub fn submit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        published_generation: Option<u64>,
        extent: RenderExtent,
        request: SampleInspectionRequest,
    ) -> Result<SampleInspectionRequestId, SampleInspectionError> {
        self.submit_inner(device, queue, published_generation, extent, request)
            .map(|(request_id, _)| request_id)
    }

    #[cfg(test)]
    fn submit_for_test(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        published_generation: Option<u64>,
        extent: RenderExtent,
        request: SampleInspectionRequest,
    ) -> Result<SubmittedInspection, SampleInspectionError> {
        self.submit_inner(device, queue, published_generation, extent, request)
            .map(|(request_id, submission)| SubmittedInspection {
                request_id,
                submission,
            })
    }

    fn submit_inner(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        published_generation: Option<u64>,
        extent: RenderExtent,
        request: SampleInspectionRequest,
    ) -> Result<(SampleInspectionRequestId, wgpu::SubmissionIndex), SampleInspectionError> {
        if let Some(pending) = &self.pending {
            return Err(SampleInspectionError::Busy {
                active: pending.context.request_id,
            });
        }
        let published = published_generation.ok_or(SampleInspectionError::NoPublishedScene)?;
        if request.generation() != published {
            return Err(SampleInspectionError::GenerationMismatch {
                requested: request.generation(),
                published,
            });
        }
        let request_id = SampleInspectionRequestId(
            self.next_request_id
                .ok_or(SampleInspectionError::RequestIdentityExhausted)?,
        );
        let gpu_request = validate_request(request_id, request, extent)?;
        self.next_request_id = request_id.get().checked_add(1);

        // This fixed 48-byte write is ordered before the following submission.
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html#method.write_buffer
        queue.write_buffer(&self.request, 0, bytemuck::bytes_of(&gpu_request));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sample inspection encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sample inspection pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // The shader retains the correctness-approved presentation workgroup shape, but 63
            // lanes return before the sequential solver. Exactly one requested ray is traced.
            // Source: https://www.w3.org/TR/WGSL/#compute-shader-workgroups
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.record, 0, &self.readback, 0, INSPECTION_RECORD_BYTES);
        let (sender, receiver) = mpsc::channel();
        // Mapping is part of the producing encoder, so it follows the result-to-readback copy.
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit
        encoder.map_buffer_on_submit(&self.readback, wgpu::MapMode::Read, .., move |result| {
            if sender.send(result).is_err() {
                tracing::debug!("sample inspection map receiver dropped");
            }
        });
        let submission = queue.submit([encoder.finish()]);
        self.pending = Some(PendingInspection {
            context: InspectionContext {
                request_id,
                request,
                extent,
            },
            receiver,
            cancelled: false,
        });
        Ok((request_id, submission))
    }

    pub fn cancel(
        &mut self,
        request_id: SampleInspectionRequestId,
    ) -> Result<(), SampleInspectionError> {
        let pending = self
            .pending
            .as_mut()
            .ok_or(SampleInspectionError::NoActiveRequest)?;
        if pending.context.request_id != request_id {
            return Err(SampleInspectionError::RequestMismatch {
                requested: request_id,
                active: pending.context.request_id,
            });
        }
        pending.cancelled = true;
        Ok(())
    }

    pub fn poll(&mut self, published_generation: Option<u64>) -> Option<SampleInspectionOutcome> {
        let pending = self.pending.take()?;
        match pending.receiver.try_recv() {
            Ok(Ok(())) => {
                let decoded = self
                    .read_record()
                    .and_then(|raw| self.decode_record(pending.context, raw));
                self.readback.unmap();
                Some(match decoded {
                    Err(error) => SampleInspectionOutcome::Failed {
                        request_id: pending.context.request_id,
                        error,
                    },
                    Ok(_) if pending.cancelled => SampleInspectionOutcome::Cancelled {
                        request_id: pending.context.request_id,
                    },
                    Ok(_) if published_generation != Some(pending.context.request.generation()) => {
                        SampleInspectionOutcome::Superseded {
                            request_id: pending.context.request_id,
                            requested_generation: pending.context.request.generation(),
                            published_generation,
                        }
                    }
                    Ok(inspection) => SampleInspectionOutcome::Completed(Box::new(inspection)),
                })
            }
            Ok(Err(error)) => {
                self.readback.unmap();
                Some(SampleInspectionOutcome::Failed {
                    request_id: pending.context.request_id,
                    error: error.into(),
                })
            }
            Err(TryRecvError::Empty) => {
                self.pending = Some(pending);
                None
            }
            Err(TryRecvError::Disconnected) => {
                self.readback.unmap();
                Some(SampleInspectionOutcome::Failed {
                    request_id: pending.context.request_id,
                    error: SampleInspectionError::MapCallbackDropped,
                })
            }
        }
    }

    pub const fn has_pending_readback(&self) -> bool {
        self.pending.is_some()
    }

    fn read_record(&self) -> Result<GpuInspectionRecord, SampleInspectionError> {
        let mapped = self.readback.get_mapped_range(..)?;
        let raw = bytemuck::pod_read_unaligned(&mapped);
        drop(mapped);
        Ok(raw)
    }

    fn decode_record(
        &self,
        context: InspectionContext,
        raw: GpuInspectionRecord,
    ) -> Result<SampleInspection, SampleInspectionError> {
        let expected_identity = request_identity(context.request_id, context.request.generation());
        if raw.identity != expected_identity {
            return Err(SampleInspectionError::IdentityMismatch);
        }
        if raw.protocol[0] != INSPECTION_ABI_VERSION {
            return Err(SampleInspectionError::UnsupportedRecordVersion {
                actual: raw.protocol[0],
                expected: INSPECTION_ABI_VERSION,
            });
        }
        if raw.protocol[1] != INSPECTION_PRODUCER_TAG {
            return Err(SampleInspectionError::UnknownProducer(raw.protocol[1]));
        }
        if raw.protocol[2] != INSPECTION_DOMAIN_TAG {
            return Err(SampleInspectionError::UnknownArithmeticDomain(
                raw.protocol[2],
            ));
        }
        let output_kind = SampleOutputKind::try_from(raw.protocol[3])?;
        if !raw.source_time.into_iter().all(f32::is_finite)
            || !raw.scene_event.into_iter().all(f32::is_finite)
            || !raw.maximum_invariant_drift.into_iter().all(f32::is_finite)
        {
            return Err(SampleInspectionError::InvalidRecord {
                field: "non-finite value",
            });
        }
        let termination = SampleTermination::try_from(raw.metadata[0])?;
        let flags = SampleNumericalFlags(raw.metadata[1]);
        if flags.bits() & !SampleNumericalFlags::KNOWN != 0 {
            return Err(SampleInspectionError::InvalidRecord {
                field: "failure flags",
            });
        }
        let candidates = SampleEventCandidates(raw.metadata[3]);
        if candidates.bits() & !SampleEventCandidates::KNOWN != 0 {
            return Err(SampleInspectionError::InvalidRecord {
                field: "event candidates",
            });
        }
        let initial_polar_side = match raw.branch_key[3] {
            0 => SamplePolarSide::Negative,
            1 => SamplePolarSide::Equatorial,
            2 => SamplePolarSide::Positive,
            unknown => return Err(SampleInspectionError::UnknownPolarSide(unknown)),
        };
        let source = decode_source(termination, raw.source_time)?;
        let rgb =
            raw.scene_event[..3]
                .try_into()
                .map_err(|_| SampleInspectionError::InvalidRecord {
                    field: "scene value",
                })?;
        let scene_value = decode_scene_value(output_kind, termination, rgb)?;
        Ok(SampleInspection {
            request_id: context.request_id,
            request: context.request,
            extent: [context.extent.width(), context.extent.height()],
            observation_identity: self.observation_identity.clone(),
            channel_model: self.channel_model,
            termination,
            source,
            scene_value,
            branch_key: SampleBranchKey {
                initial_polar_side,
                radial_turnings: raw.branch_key[0],
                equatorial_crossings: raw.branch_key[1],
                azimuth_winding: i32::from_ne_bytes(raw.branch_key[2].to_ne_bytes()),
            },
            travel_time_over_m: raw.source_time[3],
            event_diagnostics: SampleEventDiagnostics {
                candidates,
                residual: raw.scene_event[3],
            },
            numerical_diagnostics: SampleNumericalDiagnostics {
                steps: raw.metadata[2],
                flags,
                maximum_invariant_drift: SampleInvariantDrift(raw.maximum_invariant_drift),
            },
        })
    }

    #[cfg(test)]
    pub fn inspect_blocking(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        extent: RenderExtent,
        request: SampleInspectionRequest,
    ) -> Result<SampleInspection, SampleInspectionError> {
        let submitted =
            self.submit_for_test(device, queue, Some(request.generation()), extent, request)?;
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submitted.submission),
                timeout: None,
            })
            .expect("sample inspection submission completes");
        match self.poll(Some(request.generation())) {
            Some(SampleInspectionOutcome::Completed(inspection)) => Ok(*inspection),
            Some(SampleInspectionOutcome::Failed { error, .. }) => Err(error),
            Some(
                SampleInspectionOutcome::Cancelled { .. }
                | SampleInspectionOutcome::Superseded { .. },
            )
            | None => Err(SampleInspectionError::InvalidRecord {
                field: "blocking completion state",
            }),
        }
    }
}

fn validate_request(
    request_id: SampleInspectionRequestId,
    request: SampleInspectionRequest,
    extent: RenderExtent,
) -> Result<GpuInspectionRequest, SampleInspectionError> {
    let pixel = request.pixel();
    let extent_array = [extent.width(), extent.height()];
    if pixel[0] >= extent.width() || pixel[1] >= extent.height() {
        return Err(SampleInspectionError::PixelOutsideExtent {
            pixel,
            extent: extent_array,
        });
    }
    let mut packed_subpixel = [0.0; 2];
    for (coordinate, value) in request.subpixel().into_iter().enumerate() {
        if !value.is_finite() {
            return Err(SampleInspectionError::NonFiniteSubpixel { coordinate });
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(SampleInspectionError::SubpixelOutsideRange { coordinate, value });
        }
        packed_subpixel[coordinate] = value
            .to_f32()
            .ok_or(SampleInspectionError::SubpixelNotRepresentable { coordinate })?;
    }
    Ok(GpuInspectionRequest {
        pixel_extent: [pixel[0], pixel[1], extent.width(), extent.height()],
        subpixel: [packed_subpixel[0], packed_subpixel[1], 0.0, 0.0],
        identity: request_identity(request_id, request.generation()),
    })
}

const fn request_identity(request_id: SampleInspectionRequestId, generation: u64) -> [u32; 4] {
    let request_words = u64_words(request_id.get());
    let generation_words = u64_words(generation);
    [
        request_words[0],
        request_words[1],
        generation_words[0],
        generation_words[1],
    ]
}

const fn u64_words(value: u64) -> [u32; 2] {
    let bytes = value.to_le_bytes();
    [
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    ]
}

fn decode_source(
    termination: SampleTermination,
    source_time: [f32; 4],
) -> Result<SampleInspectionSource, SampleInspectionError> {
    match termination {
        SampleTermination::Escape => Ok(SampleInspectionSource::AnalyticEscape {
            unit_direction: source_time[..3].try_into().map_err(|_| {
                SampleInspectionError::InvalidRecord {
                    field: "escape source",
                }
            })?,
        }),
        SampleTermination::EquatorialSurface => {
            if source_time[0] <= 0.0 || source_time[2] <= 0.0 {
                return Err(SampleInspectionError::InvalidRecord {
                    field: "surface source",
                });
            }
            Ok(SampleInspectionSource::EquatorialSurface {
                radius_over_m: source_time[0],
                azimuth_radians: source_time[1],
                frequency_ratio: source_time[2],
            })
        }
        SampleTermination::HorizonCrossing
        | SampleTermination::SingularityGuard
        | SampleTermination::StepExhaustion
        | SampleTermination::NumericalFailure
        | SampleTermination::Uncertain => Ok(SampleInspectionSource::None),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleOutputKind {
    Horizon,
    AnalyticEscapePreview,
    SurfaceRadiance,
    TraceFailure,
}

impl TryFrom<u32> for SampleOutputKind {
    type Error = SampleInspectionError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Horizon),
            2 => Ok(Self::AnalyticEscapePreview),
            3 => Ok(Self::SurfaceRadiance),
            4 => Ok(Self::TraceFailure),
            unknown => Err(SampleInspectionError::UnknownOutputKind(unknown)),
        }
    }
}

fn decode_scene_value(
    output_kind: SampleOutputKind,
    termination: SampleTermination,
    rgb: [f32; 3],
) -> Result<SampleSceneValue, SampleInspectionError> {
    match (output_kind, termination) {
        (SampleOutputKind::Horizon, SampleTermination::HorizonCrossing) => {
            Ok(SampleSceneValue::Horizon)
        }
        (SampleOutputKind::AnalyticEscapePreview, SampleTermination::Escape) => {
            Ok(SampleSceneValue::AnalyticEscapePreview(rgb))
        }
        (SampleOutputKind::SurfaceRadiance, SampleTermination::EquatorialSurface) => {
            Ok(SampleSceneValue::SurfaceRadiance(rgb))
        }
        (
            SampleOutputKind::TraceFailure,
            SampleTermination::HorizonCrossing | SampleTermination::Escape,
        ) => Err(SampleInspectionError::InvalidRecord {
            field: "termination/output kind",
        }),
        (SampleOutputKind::TraceFailure, termination) => {
            let visible_termination = if termination == SampleTermination::EquatorialSurface {
                SampleTermination::NumericalFailure
            } else {
                termination
            };
            Ok(SampleSceneValue::TraceFailure {
                termination: visible_termination,
                visible_rgb: rgb,
            })
        }
        _ => Err(SampleInspectionError::InvalidRecord {
            field: "termination/output kind",
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of};

    use gravlume_domain::{ImageSample, Observation};
    use gravlume_reference::FixtureDocument;

    use super::{
        GpuInspectionRecord, GpuInspectionRequest, INSPECTION_ABI_VERSION, INSPECTION_DOMAIN_TAG,
        INSPECTION_LOGICAL_BUFFER_BYTES, INSPECTION_PRODUCER_TAG, INSPECTION_RECORD_BYTES,
        INSPECTION_REQUEST_BYTES, InspectionContext, SampleInspectionError,
        SampleInspectionOutcome, SampleInspectionRequest, SampleInspectionRequestId,
        SampleTermination, request_identity, validate_request,
    };
    use crate::{extent::RenderExtent, trace::TracePipeline};

    const SURFACE_OBSERVABLE: &str =
        include_str!("../../../gravlume-reference/fixtures/v2/kerr-surface-observable.toml");

    #[test]
    fn host_inspection_abi_matches_seven_aligned_vec4_lanes() {
        assert_eq!(INSPECTION_REQUEST_BYTES, 48);
        assert_eq!(INSPECTION_RECORD_BYTES, 112);
        assert_eq!(INSPECTION_LOGICAL_BUFFER_BYTES, 272);
        assert_eq!(align_of::<GpuInspectionRequest>(), 16);
        assert_eq!(align_of::<GpuInspectionRecord>(), 16);
        assert_eq!(offset_of!(GpuInspectionRequest, pixel_extent), 0);
        assert_eq!(offset_of!(GpuInspectionRequest, subpixel), 16);
        assert_eq!(offset_of!(GpuInspectionRequest, identity), 32);
        assert_eq!(offset_of!(GpuInspectionRecord, identity), 0);
        assert_eq!(offset_of!(GpuInspectionRecord, protocol), 16);
        assert_eq!(offset_of!(GpuInspectionRecord, metadata), 32);
        assert_eq!(offset_of!(GpuInspectionRecord, branch_key), 48);
        assert_eq!(offset_of!(GpuInspectionRecord, source_time), 64);
        assert_eq!(offset_of!(GpuInspectionRecord, scene_event), 80);
        assert_eq!(offset_of!(GpuInspectionRecord, maximum_invariant_drift), 96);
    }

    #[test]
    fn request_validation_preserves_attempt_and_generation_identity() {
        let extent = RenderExtent::new(128, 72).expect("test extent is nonzero");
        let request = SampleInspectionRequest::new(0x8877_6655_4433_2211, [127, 71], [0.25, 0.75]);
        let packed = validate_request(
            SampleInspectionRequestId(0xffee_ddcc_bbaa_9988),
            request,
            extent,
        )
        .expect("valid request packs");

        assert_eq!(packed.pixel_extent, [127, 71, 128, 72]);
        assert_eq!(
            packed.subpixel.map(f32::to_bits),
            [0.25_f32, 0.75, 0.0, 0.0].map(f32::to_bits)
        );
        assert_eq!(
            packed.identity,
            [0xbbaa_9988, 0xffee_ddcc, 0x4433_2211, 0x8877_6655]
        );
    }

    #[test]
    fn request_validation_rejects_extent_and_subpixel_boundaries() {
        let extent = RenderExtent::new(128, 72).expect("test extent is nonzero");
        let request_id = SampleInspectionRequestId(1);
        assert!(matches!(
            validate_request(
                request_id,
                SampleInspectionRequest::new(3, [128, 0], [0.5, 0.5]),
                extent,
            ),
            Err(SampleInspectionError::PixelOutsideExtent { .. })
        ));
        assert!(matches!(
            validate_request(
                request_id,
                SampleInspectionRequest::new(3, [0, 0], [f64::NAN, 0.5]),
                extent,
            ),
            Err(SampleInspectionError::NonFiniteSubpixel { coordinate: 0 })
        ));
        assert!(matches!(
            validate_request(
                request_id,
                SampleInspectionRequest::new(3, [0, 0], [0.5, 1.01]),
                extent,
            ),
            Err(SampleInspectionError::SubpixelOutsideRange { coordinate: 1, .. })
        ));
    }

    #[test]
    fn one_in_flight_slot_cancels_cleans_up_and_can_be_reused() {
        let (observation, sample) = canonical_surface_case();
        let extent = observation_extent(&observation);
        let gpu = crate::test_device::native_gpu();
        let (_trace, mut inspector) = TracePipeline::new_with_inspection(&gpu.device, &observation)
            .expect("canonical observation packs for inspection");
        let request = SampleInspectionRequest::new(17, sample.pixel(), sample.subpixel());

        assert!(matches!(
            inspector.submit(&gpu.device, &gpu.queue, None, extent, request),
            Err(SampleInspectionError::NoPublishedScene)
        ));
        assert!(matches!(
            inspector.submit(&gpu.device, &gpu.queue, Some(16), extent, request),
            Err(SampleInspectionError::GenerationMismatch {
                requested: 17,
                published: 16,
            })
        ));

        let first = inspector
            .submit_for_test(&gpu.device, &gpu.queue, Some(17), extent, request)
            .expect("first request occupies the bounded slot");
        let busy = inspector.submit(&gpu.device, &gpu.queue, Some(17), extent, request);
        assert!(matches!(
            busy,
            Err(SampleInspectionError::Busy { active }) if active == first.request_id
        ));
        assert!(matches!(
            inspector.cancel(SampleInspectionRequestId(first.request_id.get() + 1)),
            Err(SampleInspectionError::RequestMismatch { .. })
        ));
        inspector
            .cancel(first.request_id)
            .expect("the active request can be marked cancelled");
        wait_for(gpu, first.submission);
        assert!(matches!(
            inspector.poll(Some(17)),
            Some(SampleInspectionOutcome::Cancelled { request_id })
                if request_id == first.request_id
        ));
        assert!(!inspector.has_pending_readback());

        let second = inspector
            .submit_for_test(&gpu.device, &gpu.queue, Some(17), extent, request)
            .expect("the cleaned slot accepts a new attempt");
        assert_ne!(second.request_id, first.request_id);
        wait_for(gpu, second.submission);
        assert!(matches!(
            inspector.poll(Some(17)),
            Some(SampleInspectionOutcome::Completed(inspection))
                if inspection.request_id() == second.request_id
        ));
    }

    #[test]
    fn completed_record_is_superseded_by_a_new_published_generation() {
        let (observation, sample) = canonical_surface_case();
        let extent = observation_extent(&observation);
        let gpu = crate::test_device::native_gpu();
        let (_trace, mut inspector) = TracePipeline::new_with_inspection(&gpu.device, &observation)
            .expect("canonical observation packs for inspection");
        let request = SampleInspectionRequest::new(17, sample.pixel(), sample.subpixel());
        let submitted = inspector
            .submit_for_test(&gpu.device, &gpu.queue, Some(17), extent, request)
            .expect("request is submitted against generation 17");

        wait_for(gpu, submitted.submission);
        assert!(matches!(
            inspector.poll(Some(18)),
            Some(SampleInspectionOutcome::Superseded {
                request_id,
                requested_generation: 17,
                published_generation: Some(18),
            }) if request_id == submitted.request_id
        ));
    }

    #[test]
    fn record_decoder_preserves_branch_counts_beyond_test_capture_packing() {
        let (observation, _) = canonical_surface_case();
        let gpu = crate::test_device::native_gpu();
        let (_trace, inspector) = TracePipeline::new_with_inspection(&gpu.device, &observation)
            .expect("canonical observation packs for inspection");
        let extent = observation_extent(&observation);
        let request_id = SampleInspectionRequestId(0x1_0000_0001);
        let request = SampleInspectionRequest::new(17, [0, 0], [0.5, 0.5]);
        let winding = -123_456_i32;
        let raw = GpuInspectionRecord {
            identity: request_identity(request_id, 17),
            protocol: [
                INSPECTION_ABI_VERSION,
                INSPECTION_PRODUCER_TAG,
                INSPECTION_DOMAIN_TAG,
                4,
            ],
            metadata: [SampleTermination::StepExhaustion as u32, 0, 2_048, 0],
            branch_key: [
                0x1_0000,
                u32::MAX,
                u32::from_ne_bytes(winding.to_ne_bytes()),
                2,
            ],
            source_time: [0.0, 0.0, 0.0, 12.0],
            scene_event: [1.0, 0.25, 0.0, 0.0],
            maximum_invariant_drift: [0.01, 0.02, 0.03, 0.04],
        };

        let inspection = inspector
            .decode_record(
                InspectionContext {
                    request_id,
                    request,
                    extent,
                },
                raw,
            )
            .expect("known protocol record decodes");
        assert_eq!(inspection.branch_key().radial_turnings(), 0x1_0000);
        assert_eq!(inspection.branch_key().equatorial_crossings(), u32::MAX);
        assert_eq!(inspection.branch_key().azimuth_winding(), winding);
        assert_eq!(inspection.branch_key().initial_polar_side().code(), 2);
    }

    fn canonical_surface_case() -> (Observation, ImageSample) {
        let fixture = FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
            .expect("repository surface fixture parses")
            .into_surface_observation()
            .expect("fixture is a surface observation");
        (fixture.observation().clone(), fixture.sample())
    }

    fn observation_extent(observation: &Observation) -> RenderExtent {
        RenderExtent::new(
            observation.view().width().get(),
            observation.view().height().get(),
        )
        .expect("validated observation extent is nonzero")
    }

    fn wait_for(gpu: &crate::test_device::TestGpu, submission: wgpu::SubmissionIndex) {
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("inspection submission completes");
    }
}

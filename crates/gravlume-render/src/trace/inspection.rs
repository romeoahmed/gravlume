use gravlume_domain::ImageSample;

use crate::{
    extent::RenderExtent,
    scientific_capture::{ScientificChannelModel, ScientificTexel},
};

#[cfg(test)]
mod corpus;
mod kernel;
mod protocol;
mod slot;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub use protocol::TraceTermination;
pub use slot::SampleInspectionSlot;

#[derive(Clone, Copy, Debug, PartialEq)]
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
    StepExhaustion {
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
    /// Identifies the full Kerr-Schild RK4 retrace and its WGSL binary32 arithmetic domain.
    pub const METHOD_ID: &str = "gpu-ks-rk4-v2/full-kerr-schild-retrace/wgsl-binary32";

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
    pub const fn published_texel(self) -> ScientificTexel {
        self.published_texel
    }

    /// Returns the binary32 evidence from the fresh full Kerr-Schild retrace.
    ///
    /// This is deliberately separate from [`Self::published_texel`], which may include a
    /// shadow refinement and `Rgba16Float` rounding.
    #[must_use]
    pub const fn fresh_retrace(self) -> SampleRetrace {
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
pub enum SampleInspectionCompletion {
    Completed {
        ticket: SampleInspectionTicket,
        inspection: SampleInspection,
    },
    Cancelled {
        ticket: SampleInspectionTicket,
    },
    Failed {
        ticket: SampleInspectionTicket,
        error: SampleInspectionError,
    },
}

impl SampleInspectionCompletion {
    #[must_use]
    pub const fn ticket(&self) -> SampleInspectionTicket {
        match self {
            Self::Completed { ticket, .. }
            | Self::Cancelled { ticket }
            | Self::Failed { ticket, .. } => *ticket,
        }
    }
}

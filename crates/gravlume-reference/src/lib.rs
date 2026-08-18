#![forbid(unsafe_code)]

//! Deterministic CPU `f64` reference tracing for Gravlume.

mod batch;
mod comparison;
mod events;
mod fixture;
mod footprint;
mod geodesic;
mod integrator;
mod observation;
mod outcome;
mod policy;
mod radiation;
mod surface;

pub use batch::{GeodesicBatch, GeodesicBatchError};
pub use comparison::{ComparisonError, ComparisonIssue, ReferenceComparison};
pub use events::{EventConfiguration, EventConfigurationError, EventKind};
pub use fixture::{
    ExpectedOutcome, FixtureDocument, FixtureError, GeodesicFixture, ObservationFixture,
    SurfaceObservationFixture,
};
pub use footprint::{
    SurfaceFootprint, SurfaceFootprintError, SurfaceFootprintEstimate, SurfaceParity,
};
pub use geodesic::{GeodesicConfigurationError, GeodesicTracer};
pub use gravlume_domain::{SpectralBand, VISIBLE_BOXCAR_BANDS_V1};
pub use observation::{ObservationTrace, ObservationTraceError, ObservationTracer};
pub use outcome::{
    AffineDirection, EscapeDirection, GeodesicTrace, LocalizedEvent, NumericalFailure, PolarSide,
    ReferenceOutcome, ReferenceTerminal, Termination, TraceBranchKey, TraceDiagnostics,
    TraceInputId,
};
pub use policy::ReferencePolicy;
pub use surface::{FrequencyRatio, SourceAnchor, SurfaceObservable, SurfaceObservableError};

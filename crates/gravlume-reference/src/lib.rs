#![forbid(unsafe_code)]

//! Deterministic CPU `f64` reference tracing for Gravlume.

mod batch;
mod comparison;
mod events;
mod fixture;
mod geodesic;
mod integrator;
mod observation;
mod outcome;
mod policy;
mod surface;

pub use batch::{GeodesicBatch, GeodesicBatchError};
pub use comparison::{ComparisonError, ComparisonIssue, ReferenceComparison};
pub use events::{EventConfiguration, EventConfigurationError, EventKind};
pub use fixture::{
    ExpectedOutcome, FixtureDocument, FixtureError, GeodesicFixture, ObservationFixture,
    SurfaceObservationFixture,
};
pub use geodesic::{GeodesicConfigurationError, GeodesicTracer};
pub use observation::{ObservationTrace, ObservationTraceError, ObservationTracer};
pub use outcome::{
    AffineDirection, GeodesicTrace, LocalizedEvent, NumericalFailure, ReferenceOutcome,
    Termination, TraceDiagnostics, TraceInputId,
};
pub use policy::ReferencePolicy;
pub use surface::{FrequencyRatio, SourceAnchor, SurfaceObservable, SurfaceObservableError};

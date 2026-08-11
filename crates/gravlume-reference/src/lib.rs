#![forbid(unsafe_code)]

//! Deterministic CPU `f64` reference tracing for Gravlume.

mod batch;
mod comparison;
mod events;
mod fixture;
mod instrument;
mod integrator;
mod outcome;
mod policy;
mod tracer;

pub use batch::{ReferenceBatch, ReferenceBatchError};
pub use comparison::{ComparisonError, ComparisonIssue, ReferenceComparison};
pub use events::{EventConfiguration, EventConfigurationError, EventKind};
pub use fixture::{
    ExpectedOutcome, FixtureDocument, FixtureError, GeodesicFixture, ObservationFixture,
};
pub use instrument::{ReferenceInstrument, ReferenceRequest, ReferenceRuntimeError};
pub use outcome::{
    AffineDirection, LocalizedEvent, NumericalFailure, ReferenceOutcome, Termination,
    TraceDiagnostics, TraceInputId, TraceRequest,
};
pub use policy::ReferencePolicy;
pub use tracer::{ReferenceConfigurationError, ReferenceTracer};

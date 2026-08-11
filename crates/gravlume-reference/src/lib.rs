#![forbid(unsafe_code)]

//! Deterministic CPU `f64` reference tracing for Gravlume.

mod events;
mod integrator;
mod outcome;
mod policy;
mod tracer;

pub use events::{EventConfiguration, EventConfigurationError, EventKind};
pub use outcome::{
    AffineDirection, LocalizedEvent, NumericalFailure, ReferenceOutcome, Termination,
    TraceDiagnostics, TraceRequest,
};
pub use policy::ReferencePolicy;
pub use tracer::{ReferenceConfigurationError, ReferenceTracer};

#![forbid(unsafe_code)]

//! Validated scientific domain types for Gravlume.

mod math;
mod observer;
mod projection;
mod scene;
mod spacetime;
mod validation;

pub use math::{GeodesicState, SpacetimeEvent};
pub use observer::{ObserverFrame, StationaryObserverDraft};
pub use projection::{Angle, ViewportProjection, ViewportSample};
pub use scene::{InitialViewRay, Observation, PhysicalScene, PhysicalSceneDraft};
pub use spacetime::{
    GeodesicInvariants, GeometryError, KerrNewmanSpacetime, KerrSchildCoordinates, ParameterState,
};
pub use validation::{ValidationIssue, ValidationIssueCode, ValidationReport};

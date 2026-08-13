#![forbid(unsafe_code)]

//! Validated scientific domain types for Gravlume.

mod math;
mod observer;
mod scene;
mod spacetime;
mod validation;
mod view;

pub use math::{GeodesicState, SpacetimeEvent};
pub use observer::{ObserverFrame, StationaryObserverInput};
pub use scene::{InitialViewRay, Observation, PhysicalScene, PhysicalSceneInput};
pub use spacetime::{
    Extremality, GeodesicInvariants, GeometryError, KerrNewmanSpacetime, KerrSchildChart,
};
pub use validation::{ValidationIssue, ValidationIssueCode, ValidationReport};
pub use view::{Angle, ImageSample, PerspectiveView};

#![forbid(unsafe_code)]

//! Validated scientific domain types for Gravlume.

mod numerics;
mod observer;
mod scene;
mod spacetime;
mod state;
mod validation;
mod view;

pub use observer::{ObserverFrame, StationaryObserverInput};
pub use scene::{InitialViewRay, Observation, PhysicalScene, PhysicalSceneInput};
pub use spacetime::{
    Extremality, GeodesicInvariants, GeometryError, KerrNewmanSpacetime, KerrSchildChart,
};
pub use state::{GeodesicState, SpacetimeEvent};
pub use validation::{ValidationIssue, ValidationIssueCode, ValidationReport};
pub use view::{Angle, ImageSample, PerspectiveView};

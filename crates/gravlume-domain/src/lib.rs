#![forbid(unsafe_code)]

//! Validated scientific domain types for Gravlume.

mod emission;
mod numerics;
mod observer;
mod radiation;
mod scene;
mod spacetime;
mod state;
mod validation;
mod view;

pub use emission::{EquatorialCircularEmitter, EquatorialEmissionModel};
pub use observer::{ObserverFrame, StationaryObserverInput};
pub use radiation::{HomogeneousScalarSlab, SpectralBand, VISIBLE_BOXCAR_BANDS_V1};
pub use scene::{InitialViewRay, Observation, PhysicalScene, PhysicalSceneInput};
pub use spacetime::{
    Extremality, GeodesicInvariants, GeometryError, KerrNewmanSpacetime, KerrSchildChart,
};
pub use state::{GeodesicState, SpacetimeEvent};
pub use validation::{ValidationIssue, ValidationIssueCode, ValidationReport};
pub use view::{Angle, ImageSample, PerspectiveView};

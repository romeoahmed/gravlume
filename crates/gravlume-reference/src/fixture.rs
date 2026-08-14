mod v1;
mod v2;

use gravlume_domain::{
    GeodesicState, ImageSample, KerrNewmanSpacetime, Observation, ValidationReport,
};
use serde::Deserialize;

use crate::{
    AffineDirection, EquatorialCircularSurface, EventConfiguration, GeodesicTrace,
    ObservationTrace, ReferenceOutcome, ReferencePolicy, Termination, TraceInputId,
};

const MAX_FIXTURE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug)]
pub enum FixtureDocument {
    Observation(ObservationFixture),
    Geodesic(GeodesicFixture),
    SurfaceObservation(SurfaceObservationFixture),
}

impl FixtureDocument {
    /// Parses one strict, versioned TOML fixture document.
    ///
    /// Decimal inputs remain strings at the serialization seam and are converted explicitly.
    ///
    /// # Errors
    ///
    /// Rejects unknown fields/enums, unsupported envelopes, invalid decimal strings, and invalid
    /// physical initial data.
    pub fn parse_toml(source: &str) -> Result<Self, FixtureError> {
        if source.len() > MAX_FIXTURE_BYTES {
            return Err(FixtureError::TooLarge {
                actual_bytes: source.len(),
                maximum_bytes: MAX_FIXTURE_BYTES,
            });
        }
        let version: FixtureVersion = toml::from_str(source)?;
        match version.schema_version {
            1 => v1::parse(source),
            2 => v2::parse(source),
            unsupported => Err(FixtureError::UnsupportedSchema(unsupported)),
        }
    }

    #[must_use]
    pub fn into_geodesic(self) -> Option<GeodesicFixture> {
        match self {
            Self::Geodesic(fixture) => Some(fixture),
            Self::Observation(_) | Self::SurfaceObservation(_) => None,
        }
    }

    #[must_use]
    pub fn into_observation(self) -> Option<ObservationFixture> {
        match self {
            Self::Observation(fixture) => Some(fixture),
            Self::Geodesic(_) | Self::SurfaceObservation(_) => None,
        }
    }

    #[must_use]
    pub fn into_surface_observation(self) -> Option<SurfaceObservationFixture> {
        match self {
            Self::SurfaceObservation(fixture) => Some(fixture),
            Self::Observation(_) | Self::Geodesic(_) => None,
        }
    }
}

#[derive(Deserialize)]
struct FixtureVersion {
    schema_version: u32,
}

#[derive(Clone, Debug)]
pub struct ObservationFixture {
    input_id: TraceInputId,
    observation: Observation,
}

impl ObservationFixture {
    #[must_use]
    pub const fn input_id(&self) -> &TraceInputId {
        &self.input_id
    }

    #[must_use]
    pub const fn observation(&self) -> &Observation {
        &self.observation
    }
}

#[derive(Clone, Debug)]
pub struct SurfaceObservationFixture {
    input_id: TraceInputId,
    observation: Observation,
    sample: ImageSample,
    surface: EquatorialCircularSurface,
    expected: ExpectedSurfaceOutcome,
}

impl SurfaceObservationFixture {
    #[must_use]
    pub const fn input_id(&self) -> &TraceInputId {
        &self.input_id
    }

    /// Builds the same physical trace under the selected numerical policy.
    ///
    /// # Errors
    ///
    /// Returns a validation report if the stored sample no longer belongs to its observation.
    pub fn trace_request(
        &self,
        policy: ReferencePolicy,
    ) -> Result<ObservationTrace, ValidationReport> {
        ObservationTrace::new(
            self.input_id.clone(),
            &self.observation,
            self.sample,
            policy,
        )
        .map(|request| request.with_equatorial_circular_surface(self.surface))
    }

    #[must_use]
    pub const fn expected(&self) -> &ExpectedSurfaceOutcome {
        &self.expected
    }
}

#[derive(Clone, Debug)]
pub struct GeodesicFixture {
    input_id: TraceInputId,
    spacetime: KerrNewmanSpacetime,
    initial_state: GeodesicState,
    affine_direction: AffineDirection,
    events: EventConfiguration,
    expected: ExpectedOutcome,
}

impl GeodesicFixture {
    #[must_use]
    pub const fn input_id(&self) -> &TraceInputId {
        &self.input_id
    }

    #[must_use]
    pub const fn spacetime(&self) -> KerrNewmanSpacetime {
        self.spacetime
    }

    pub(crate) const fn event_configuration(&self) -> EventConfiguration {
        self.events
    }

    #[must_use]
    pub fn trace_request(&self) -> GeodesicTrace {
        GeodesicTrace::new(
            self.input_id.clone(),
            self.initial_state,
            self.affine_direction,
        )
    }

    #[must_use]
    pub const fn expected(&self) -> &ExpectedOutcome {
        &self.expected
    }
}

#[derive(Clone, Debug)]
pub struct ExpectedOutcome {
    input_id: TraceInputId,
    termination: Termination,
    event_radius_m: Option<f64>,
    turning_radius_m: Option<f64>,
    azimuth_advance_rad: f64,
    event_radius_absolute_tolerance_m: Option<f64>,
    turning_radius_absolute_tolerance_m: Option<f64>,
    azimuth_advance_absolute_tolerance_rad: f64,
    spacetime: KerrNewmanSpacetime,
}

impl ExpectedOutcome {
    #[must_use]
    pub fn accepts(&self, outcome: &ReferenceOutcome) -> bool {
        if outcome.input_id() != &self.input_id
            || outcome.termination() != self.termination
            || (outcome.azimuth_advance_rad() - self.azimuth_advance_rad).abs()
                > self.azimuth_advance_absolute_tolerance_rad
        {
            return false;
        }
        match (
            self.turning_radius_m,
            self.turning_radius_absolute_tolerance_m,
            outcome.turning_radius_m(),
        ) {
            (None, None, None) => {}
            (Some(expected), Some(tolerance), Some(actual))
                if (actual - expected).abs() <= tolerance => {}
            _ => return false,
        }
        if let (Some(expected), Some(tolerance)) =
            (self.event_radius_m, self.event_radius_absolute_tolerance_m)
        {
            let Ok(actual) = self.spacetime.radius(outcome.state().event()) else {
                return false;
            };
            if (actual - expected).abs() > tolerance {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Debug)]
pub struct ExpectedSurfaceOutcome {
    input_id: TraceInputId,
    termination: Termination,
    source_radius_m: f64,
    source_azimuth_rad: f64,
    frequency_ratio: f64,
    travel_time_m: f64,
    emitted_bolometric_intensity: f64,
    observed_bolometric_intensity: f64,
    intensity_at_six_m: f64,
    source_coordinate_absolute_tolerance_m: f64,
    source_azimuth_absolute_tolerance_rad: f64,
    frequency_ratio_relative_tolerance: f64,
    travel_time_absolute_tolerance_m: f64,
    bolometric_intensity_absolute_tolerance: f64,
}

impl ExpectedSurfaceOutcome {
    #[must_use]
    pub fn accepts(&self, outcome: &ReferenceOutcome) -> bool {
        if outcome.input_id() != &self.input_id
            || outcome.termination() != self.termination
            || (outcome.travel_time_m() - self.travel_time_m).abs()
                > self.travel_time_absolute_tolerance_m
        {
            return false;
        }
        let Some(observable) = outcome.surface_observable() else {
            return false;
        };
        let Some(anchor) = observable.source_anchor().as_equatorial_surface() else {
            return false;
        };
        let azimuth_error =
            wrapped_angle_difference(anchor.azimuth_rad(), self.source_azimuth_rad).abs();
        let ratio = observable.frequency_ratio().value();
        let ratio_relative_error = (ratio - self.frequency_ratio).abs() / self.frequency_ratio;
        if (anchor.radius_m() - self.source_radius_m).abs()
            > self.source_coordinate_absolute_tolerance_m
            || azimuth_error > self.source_azimuth_absolute_tolerance_rad
            || ratio_relative_error > self.frequency_ratio_relative_tolerance
        {
            return false;
        }
        let emitted = self.intensity_at_six_m * (anchor.radius_m() / 6.0).powi(-3);
        let Ok(observed) = observable.vacuum_observed_bolometric_intensity(emitted) else {
            return false;
        };
        (emitted - self.emitted_bolometric_intensity).abs()
            <= self.bolometric_intensity_absolute_tolerance
            && (observed - self.observed_bolometric_intensity).abs()
                <= self.bolometric_intensity_absolute_tolerance
    }
}

fn wrapped_angle_difference(left: f64, right: f64) -> f64 {
    use std::f64::consts::PI;

    (left - right + PI).rem_euclid(2.0 * PI) - PI
}

#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error("fixture has {actual_bytes} bytes; the maximum is {maximum_bytes}")]
    TooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("fixture TOML does not match its strict schema: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("unsupported fixture schema version {0}")]
    UnsupportedSchema(u32),
    #[error("unsupported validation profile {0}")]
    UnsupportedProfile(String),
    #[error("fixture physical data is invalid: {0}")]
    InvalidPhysicalData(String),
    #[error("fixture event envelope is internally inconsistent")]
    InconsistentEventEnvelope,
    #[error("fixture expected observables are internally inconsistent")]
    InconsistentExpectedObservables,
    #[error("fixture applicability is internally inconsistent")]
    InconsistentApplicability,
    #[error("fixture field {field} does not match its versioned preset")]
    PresetMismatch { field: &'static str },
}

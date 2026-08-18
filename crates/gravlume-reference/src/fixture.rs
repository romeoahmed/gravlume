mod v1;
mod v2;
mod v3;

use std::fmt;

use gravlume_domain::{GeodesicState, KerrNewmanSpacetime, Observation, ValidationReport};
use serde::{Deserialize, Deserializer, de};

use crate::{
    AffineDirection, EventConfiguration, GeodesicTrace, ObservationTrace, ReferenceOutcome,
    ReferencePolicy, Termination, TraceInputId, surface::wrapped_angle_difference,
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
            3 => v3::parse(source),
            unsupported => Err(FixtureError::UnsupportedSchema(unsupported)),
        }
    }

    #[must_use]
    pub fn into_geodesic(self) -> Option<GeodesicFixture> {
        match self {
            Self::Geodesic(fixture) => Some(fixture),
            _ => None,
        }
    }

    #[must_use]
    pub fn into_observation(self) -> Option<ObservationFixture> {
        match self {
            Self::Observation(fixture) => Some(fixture),
            _ => None,
        }
    }

    #[must_use]
    pub fn into_surface_observation(self) -> Option<SurfaceObservationFixture> {
        match self {
            Self::SurfaceObservation(fixture) => Some(fixture),
            _ => None,
        }
    }
}

#[derive(Deserialize)]
struct FixtureVersion {
    schema_version: u32,
}

#[derive(PartialEq)]
struct DecimalString {
    source: String,
    value: f64,
}

impl<'de> Deserialize<'de> for DecimalString {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let source = String::deserialize(deserializer)?;
        let value = source
            .parse::<f64>()
            .map_err(|_| de::Error::custom("decimal string is not representable as f64"))?;
        if !value.is_finite() || (value == 0.0 && source.trim_start().starts_with('-')) {
            return Err(de::Error::custom(
                "decimal string must be finite and must not encode negative zero",
            ));
        }
        Ok(Self { source, value })
    }
}

fn invalid_physical_data(error: impl fmt::Display) -> FixtureError {
    FixtureError::InvalidPhysicalData(error.to_string())
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
    sample: gravlume_domain::ImageSample,
    expected: ExpectedSurfaceOutcome,
}

impl SurfaceObservationFixture {
    #[must_use]
    pub const fn input_id(&self) -> &TraceInputId {
        &self.input_id
    }

    /// Builds the same validated physical trace under the selected numerical policy.
    /// Resolves the versioned sample through the same source-bearing Observation seam as callers.
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
    }

    #[must_use]
    pub const fn observation(&self) -> &Observation {
        &self.observation
    }

    #[must_use]
    pub const fn sample(&self) -> gravlume_domain::ImageSample {
        self.sample
    }

    #[must_use]
    pub fn accepts(&self, outcome: &ReferenceOutcome) -> bool {
        outcome.input_id() == &self.input_id
            && outcome.termination() == Termination::EquatorialSurface
            && self.expected.accepts(outcome)
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
struct ExpectedSurfaceOutcome {
    source_radius_m: f64,
    source_azimuth_rad: f64,
    frequency_ratio: f64,
    travel_time_m: f64,
    emitted_bolometric_intensity: f64,
    observed_bolometric_intensity: f64,
    source_coordinate_absolute_tolerance_m: f64,
    source_azimuth_absolute_tolerance_rad: f64,
    frequency_ratio_relative_tolerance: f64,
    travel_time_absolute_tolerance_m: f64,
    bolometric_intensity_absolute_tolerance: f64,
    transport: Option<ExpectedSurfaceTransport>,
}

#[derive(Clone, Debug)]
struct ExpectedSurfaceTransport {
    vacuum_observed_bolometric_intensity: f64,
    optical_depth: f64,
    emitted_temperature_kelvin: f64,
    vacuum_observed_temperature_kelvin: f64,
    observed_spectral_band_intensities: [f64; 3],
    scalar_absolute_tolerance: f64,
    temperature_relative_tolerance: f64,
    spectral_intensity_absolute_tolerance: f64,
}

impl ExpectedSurfaceOutcome {
    fn accepts(&self, outcome: &ReferenceOutcome) -> bool {
        if (outcome.travel_time_m() - self.travel_time_m).abs()
            > self.travel_time_absolute_tolerance_m
        {
            return false;
        }
        let Some(observable) = outcome.terminal().surface_observable() else {
            return false;
        };
        let anchor = observable.source_anchor();
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
        let bolometric_matches = (observable.emitted_bolometric_intensity()
            - self.emitted_bolometric_intensity)
            .abs()
            <= self.bolometric_intensity_absolute_tolerance
            && (observable.observed_bolometric_intensity() - self.observed_bolometric_intensity)
                .abs()
                <= self.bolometric_intensity_absolute_tolerance;
        bolometric_matches
            && self
                .transport
                .as_ref()
                .is_none_or(|expected| expected.accepts(observable))
    }
}

impl ExpectedSurfaceTransport {
    fn accepts(&self, observable: crate::SurfaceObservable) -> bool {
        let Some(emitted_temperature) = observable.emitted_temperature_kelvin() else {
            return false;
        };
        let Some(observed_temperature) = observable.vacuum_observed_temperature_kelvin() else {
            return false;
        };
        let Some(observed_bands) = observable.observed_spectral_band_intensities() else {
            return false;
        };
        let emitted_temperature_relative_error =
            (emitted_temperature - self.emitted_temperature_kelvin).abs()
                / self.emitted_temperature_kelvin;
        let observed_temperature_relative_error =
            (observed_temperature - self.vacuum_observed_temperature_kelvin).abs()
                / self.vacuum_observed_temperature_kelvin;

        (observable.vacuum_observed_bolometric_intensity()
            - self.vacuum_observed_bolometric_intensity)
            .abs()
            <= self.scalar_absolute_tolerance
            && (observable.optical_depth() - self.optical_depth).abs()
                <= self.scalar_absolute_tolerance
            && emitted_temperature_relative_error <= self.temperature_relative_tolerance
            && observed_temperature_relative_error <= self.temperature_relative_tolerance
            && observed_bands
                .into_iter()
                .zip(self.observed_spectral_band_intensities)
                .all(|(actual, expected)| {
                    (actual - expected).abs() <= self.spectral_intensity_absolute_tolerance
                })
    }
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

use std::fmt;

use serde::{Deserialize, Deserializer, de};

use crate::{
    AffineDirection, EquatorialCircularSurface, EventConfiguration, Termination, TraceInputId,
};

use super::{ExpectedSurfaceOutcome, FixtureDocument, FixtureError, SurfaceObservationFixture, v1};

const BASE_OBSERVATION: &str = include_str!("../../fixtures/v1/default-kerr-observation.toml");
const SURFACE_OBSERVABLE: &str = include_str!("../../fixtures/v2/kerr-surface-observable.toml");

pub fn parse(source: &str) -> Result<FixtureDocument, FixtureError> {
    let raw: RawSurfaceFixture = toml::from_str(source)?;
    let canonical: RawSurfaceFixture = toml::from_str(SURFACE_OBSERVABLE)?;
    if raw != canonical {
        return Err(FixtureError::PresetMismatch {
            field: "surface artifact",
        });
    }
    validate_envelope(&raw)?;

    let FixtureDocument::Observation(base) = v1::parse(BASE_OBSERVATION)? else {
        return Err(FixtureError::PresetMismatch {
            field: "base observation artifact",
        });
    };
    if raw.base_observation_id != base.input_id.as_str() {
        return Err(FixtureError::InconsistentApplicability);
    }
    let observation = base.observation;
    let sample = observation
        .view()
        .sample(
            raw.sample.x,
            raw.sample.y,
            raw.sample.subpixel[0].value,
            raw.sample.subpixel[1].value,
        )
        .map_err(invalid_physical_data)?;
    let surface = EquatorialCircularSurface::new(
        raw.surface.inner_radius_m.value,
        raw.surface.outer_radius_m.value,
    )
    .map_err(invalid_physical_data)?;
    validate_expected(&raw, surface)?;

    let initial_ray = observation
        .initial_ray(sample)
        .map_err(invalid_physical_data)?;
    let events = EventConfiguration::observation_baseline_v1()
        .with_equatorial_surface(surface.inner_radius_m(), surface.outer_radius_m())
        .map_err(invalid_physical_data)?;
    let input_id = TraceInputId::new(raw.id).bind(
        *observation.scene().spacetime(),
        initial_ray.state(),
        AffineDirection::Negative,
        events,
    );
    let expected = ExpectedSurfaceOutcome {
        input_id: input_id.clone(),
        termination: Termination::EquatorialSurface,
        source_radius_m: raw.expected.source_radius_m.value,
        source_azimuth_rad: raw.expected.source_azimuth_rad.value,
        frequency_ratio: raw.expected.frequency_ratio.value,
        travel_time_m: raw.expected.travel_time_m.value,
        emitted_bolometric_intensity: raw.expected.emitted_bolometric_intensity.value,
        observed_bolometric_intensity: raw.expected.observed_bolometric_intensity.value,
        intensity_at_six_m: raw.emission.intensity_at_6m.value,
        source_coordinate_absolute_tolerance_m: raw.tolerance.source_coordinate_abs_m.value,
        source_azimuth_absolute_tolerance_rad: raw.tolerance.source_azimuth_abs_rad.value,
        frequency_ratio_relative_tolerance: raw.tolerance.frequency_ratio_rel.value,
        travel_time_absolute_tolerance_m: raw.tolerance.travel_time_abs_m.value,
        bolometric_intensity_absolute_tolerance: raw.tolerance.bolometric_intensity_abs.value,
    };
    Ok(FixtureDocument::SurfaceObservation(
        SurfaceObservationFixture {
            input_id,
            observation,
            sample,
            surface,
            expected,
        },
    ))
}

fn validate_envelope(raw: &RawSurfaceFixture) -> Result<(), FixtureError> {
    if raw.schema_version != 2 {
        return Err(FixtureError::UnsupportedSchema(raw.schema_version));
    }
    if raw.profile != "surface-observable-v1" {
        return Err(FixtureError::UnsupportedProfile(raw.profile.clone()));
    }
    if !matches!(raw.kind, FixtureKind::SurfaceObservation)
        || !matches!(raw.evidence, Evidence::ReferenceConvergence)
        || raw.producer.precision_bits != 53
        || raw.producer.method.trim().is_empty()
        || !matches!(raw.surface.rotation, Rotation::ProgradeCircular)
        || !matches!(raw.emission.model, EmissionModel::InverseCubeBolometricV1)
        || !matches!(
            raw.expected.termination,
            ExpectedTermination::EquatorialSurface
        )
    {
        return Err(FixtureError::InconsistentApplicability);
    }
    Ok(())
}

fn validate_expected(
    raw: &RawSurfaceFixture,
    surface: EquatorialCircularSurface,
) -> Result<(), FixtureError> {
    let expected = &raw.expected;
    let tolerance = &raw.tolerance;
    let values_are_positive = raw.emission.intensity_at_6m.value > 0.0
        && expected.frequency_ratio.value > 0.0
        && expected.travel_time_m.value > 0.0
        && expected.emitted_bolometric_intensity.value >= 0.0
        && expected.observed_bolometric_intensity.value >= 0.0
        && tolerance.source_coordinate_abs_m.value > 0.0
        && tolerance.source_azimuth_abs_rad.value > 0.0
        && tolerance.frequency_ratio_rel.value > 0.0
        && tolerance.travel_time_abs_m.value > 0.0
        && tolerance.bolometric_intensity_abs.value > 0.0;
    if !values_are_positive
        || !(surface.inner_radius_m()..=surface.outer_radius_m())
            .contains(&expected.source_radius_m.value)
    {
        return Err(FixtureError::InconsistentExpectedObservables);
    }
    let emitted =
        raw.emission.intensity_at_6m.value * (expected.source_radius_m.value / 6.0).powi(-3);
    let observed = expected.frequency_ratio.value.powi(4) * emitted;
    if (emitted - expected.emitted_bolometric_intensity.value).abs()
        > tolerance.bolometric_intensity_abs.value
        || (observed - expected.observed_bolometric_intensity.value).abs()
            > tolerance.bolometric_intensity_abs.value
    {
        return Err(FixtureError::InconsistentExpectedObservables);
    }
    Ok(())
}

fn invalid_physical_data(error: impl fmt::Display) -> FixtureError {
    FixtureError::InvalidPhysicalData(error.to_string())
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawSurfaceFixture {
    schema_version: u32,
    profile: String,
    id: String,
    kind: FixtureKind,
    evidence: Evidence,
    base_observation_id: String,
    producer: RawProducer,
    sample: RawSample,
    surface: RawSurface,
    emission: RawEmission,
    expected: RawExpected,
    tolerance: RawTolerance,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawProducer {
    method: String,
    precision_bits: u32,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawSample {
    x: u32,
    y: u32,
    subpixel: [DecimalString; 2],
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawSurface {
    inner_radius_m: DecimalString,
    outer_radius_m: DecimalString,
    rotation: Rotation,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawEmission {
    model: EmissionModel,
    intensity_at_6m: DecimalString,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawExpected {
    termination: ExpectedTermination,
    source_radius_m: DecimalString,
    source_azimuth_rad: DecimalString,
    frequency_ratio: DecimalString,
    travel_time_m: DecimalString,
    emitted_bolometric_intensity: DecimalString,
    observed_bolometric_intensity: DecimalString,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawTolerance {
    source_coordinate_abs_m: DecimalString,
    source_azimuth_abs_rad: DecimalString,
    frequency_ratio_rel: DecimalString,
    travel_time_abs_m: DecimalString,
    bolometric_intensity_abs: DecimalString,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum FixtureKind {
    SurfaceObservation,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Evidence {
    ReferenceConvergence,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Rotation {
    ProgradeCircular,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum EmissionModel {
    InverseCubeBolometricV1,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum ExpectedTermination {
    EquatorialSurface,
}

#[derive(Clone, PartialEq)]
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

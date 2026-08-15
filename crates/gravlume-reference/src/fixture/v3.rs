use gravlume_domain::{
    EquatorialCircularEmitter, HomogeneousScalarSlab, Observation, PhysicalScene,
};
use serde::Deserialize;

use crate::{
    AffineDirection, EventConfiguration, TraceInputId,
    radiation::{
        blackbody_band_intensities, transport_blackbody_bands, transport_bolometric_intensity,
    },
    surface::{
        emitted_blackbody_temperature, emitted_bolometric_intensity,
        vacuum_observed_bolometric_intensity,
    },
};

use super::{
    DecimalString, ExpectedSurfaceOutcome, ExpectedSurfaceTransport, FixtureDocument, FixtureError,
    SurfaceObservationFixture, invalid_physical_data, v1,
};

const BASE_OBSERVATION: &str = include_str!("../../fixtures/v1/default-kerr-observation.toml");
const BLACKBODY_VACUUM: &str = include_str!("../../fixtures/v3/kerr-blackbody-vacuum.toml");
const PURE_ABSORPTION: &str = include_str!("../../fixtures/v3/kerr-blackbody-pure-absorption.toml");
const CONSTANT_BLACKBODY: &str =
    include_str!("../../fixtures/v3/kerr-blackbody-constant-slab.toml");
const PURE_EMISSION: &str = include_str!("../../fixtures/v3/kerr-blackbody-pure-emission.toml");

pub fn parse(source: &str) -> Result<FixtureDocument, FixtureError> {
    let raw: RawSurfaceTransportFixture = toml::from_str(source)?;
    validate_envelope(&raw)?;
    let canonical_source = match raw.id.as_str() {
        "kerr-blackbody-vacuum-v1" => BLACKBODY_VACUUM,
        "kerr-blackbody-pure-absorption-v1" => PURE_ABSORPTION,
        "kerr-blackbody-constant-slab-v1" => CONSTANT_BLACKBODY,
        "kerr-blackbody-pure-emission-v1" => PURE_EMISSION,
        _ => return Err(FixtureError::PresetMismatch { field: "id" }),
    };
    let canonical: RawSurfaceTransportFixture = toml::from_str(canonical_source)?;
    if raw != canonical {
        return Err(FixtureError::PresetMismatch {
            field: "surface transport artifact",
        });
    }

    let FixtureDocument::Observation(base) = v1::parse(BASE_OBSERVATION)? else {
        return Err(FixtureError::PresetMismatch {
            field: "base observation artifact",
        });
    };
    if raw.base_observation_id != base.input_id.as_str() {
        return Err(FixtureError::InconsistentApplicability);
    }
    let emitter = EquatorialCircularEmitter::inverse_cube_blackbody_v1(
        raw.surface.inner_radius_m.value,
        raw.surface.outer_radius_m.value,
        raw.emission.intensity_at_6m.value,
        raw.emission.temperature_at_6m_kelvin.value,
    )
    .map_err(invalid_physical_data)?;
    let slab = build_slab(&raw.transport)?;
    let scene = base
        .observation
        .scene()
        .clone()
        .with_equatorial_circular_emitter(emitter);
    let scene = install_slab(scene, slab);
    let observation = Observation::new(scene, *base.observation.view());
    let sample = observation
        .view()
        .sample(
            raw.sample.x,
            raw.sample.y,
            raw.sample.subpixel[0].value,
            raw.sample.subpixel[1].value,
        )
        .map_err(invalid_physical_data)?;
    let initial_ray = observation
        .initial_ray(sample)
        .map_err(invalid_physical_data)?;
    let spacetime = *observation.scene().spacetime();
    validate_expected(&raw, emitter, slab, spacetime.mass_m())?;
    let events = EventConfiguration::observation_baseline_v1()
        .with_equatorial_surface(emitter.inner_radius_m(), emitter.outer_radius_m())
        .map_err(invalid_physical_data)?;
    let input_id = TraceInputId::new(raw.id).bind(
        spacetime,
        initial_ray.state(),
        AffineDirection::Negative,
        events,
    );
    let expected = ExpectedSurfaceOutcome {
        source_radius_m: raw.expected.source_radius_m.value,
        source_azimuth_rad: raw.expected.source_azimuth_rad.value,
        frequency_ratio: raw.expected.frequency_ratio.value,
        travel_time_m: raw.expected.travel_time_m.value,
        emitted_bolometric_intensity: raw.expected.emitted_bolometric_intensity.value,
        observed_bolometric_intensity: raw.expected.observed_bolometric_intensity.value,
        source_coordinate_absolute_tolerance_m: raw.tolerance.source_coordinate_abs_m.value,
        source_azimuth_absolute_tolerance_rad: raw.tolerance.source_azimuth_abs_rad.value,
        frequency_ratio_relative_tolerance: raw.tolerance.frequency_ratio_rel.value,
        travel_time_absolute_tolerance_m: raw.tolerance.travel_time_abs_m.value,
        bolometric_intensity_absolute_tolerance: raw.tolerance.bolometric_intensity_abs.value,
        transport: Some(ExpectedSurfaceTransport {
            vacuum_observed_bolometric_intensity: raw
                .expected
                .vacuum_observed_bolometric_intensity
                .value,
            optical_depth: raw.expected.optical_depth.value,
            emitted_temperature_kelvin: raw.expected.emitted_temperature_kelvin.value,
            vacuum_observed_temperature_kelvin: raw
                .expected
                .vacuum_observed_temperature_kelvin
                .value,
            observed_spectral_band_intensities: decimal_array(
                &raw.expected.observed_spectral_band_intensities,
            ),
            scalar_absolute_tolerance: raw.tolerance.bolometric_intensity_abs.value,
            temperature_relative_tolerance: raw.tolerance.temperature_rel.value,
            spectral_intensity_absolute_tolerance: raw.tolerance.spectral_intensity_abs.value,
        }),
    };
    Ok(FixtureDocument::SurfaceObservation(
        SurfaceObservationFixture {
            input_id,
            observation,
            sample,
            expected,
        },
    ))
}

const fn install_slab(scene: PhysicalScene, slab: Option<HomogeneousScalarSlab>) -> PhysicalScene {
    match slab {
        Some(slab) => scene.with_homogeneous_scalar_slab(slab),
        None => scene,
    }
}

fn build_slab(transport: &RawTransport) -> Result<Option<HomogeneousScalarSlab>, FixtureError> {
    let slab = match transport {
        RawTransport::Vacuum {} => return Ok(None),
        RawTransport::PureAbsorption { optical_depth } => {
            HomogeneousScalarSlab::pure_absorption_v1(optical_depth.value)
        }
        RawTransport::ConstantBlackbody {
            optical_depth,
            source_bolometric_intensity,
            source_temperature_kelvin,
        } => HomogeneousScalarSlab::constant_blackbody_v1(
            optical_depth.value,
            source_bolometric_intensity.value,
            source_temperature_kelvin.value,
        ),
        RawTransport::PureEmissionBlackbody {
            integrated_bolometric_emission,
            emission_temperature_kelvin,
        } => HomogeneousScalarSlab::pure_emission_blackbody_v1(
            integrated_bolometric_emission.value,
            emission_temperature_kelvin.value,
        ),
    }
    .map_err(invalid_physical_data)?;
    Ok(Some(slab))
}

fn validate_envelope(raw: &RawSurfaceTransportFixture) -> Result<(), FixtureError> {
    if raw.schema_version != 3 {
        return Err(FixtureError::UnsupportedSchema(raw.schema_version));
    }
    if raw.profile != "surface-transport-v1" {
        return Err(FixtureError::UnsupportedProfile(raw.profile.clone()));
    }
    if raw.producer.precision_digits != 80 || raw.producer.method.trim().is_empty() {
        return Err(FixtureError::InconsistentApplicability);
    }
    Ok(())
}

fn validate_expected(
    raw: &RawSurfaceTransportFixture,
    emitter: EquatorialCircularEmitter,
    slab: Option<HomogeneousScalarSlab>,
    mass_m: f64,
) -> Result<(), FixtureError> {
    let expected = &raw.expected;
    let tolerance = &raw.tolerance;
    if expected.frequency_ratio.value <= 0.0
        || expected.travel_time_m.value <= 0.0
        || tolerance.source_coordinate_abs_m.value <= 0.0
        || tolerance.source_azimuth_abs_rad.value <= 0.0
        || tolerance.frequency_ratio_rel.value <= 0.0
        || tolerance.travel_time_abs_m.value <= 0.0
        || tolerance.bolometric_intensity_abs.value <= 0.0
        || tolerance.temperature_rel.value <= 0.0
        || tolerance.spectral_intensity_abs.value <= 0.0
        || !(emitter.inner_radius_m()..=emitter.outer_radius_m())
            .contains(&expected.source_radius_m.value)
    {
        return Err(FixtureError::InconsistentExpectedObservables);
    }
    let emitted_intensity =
        emitted_bolometric_intensity(emitter, mass_m, expected.source_radius_m.value)
            .map_err(|_| FixtureError::InconsistentExpectedObservables)?;
    let vacuum =
        vacuum_observed_bolometric_intensity(emitted_intensity, expected.frequency_ratio.value)
            .map_err(|_| FixtureError::InconsistentExpectedObservables)?;
    let emitted_temperature =
        emitted_blackbody_temperature(emitter, expected.source_radius_m.value / mass_m)
            .map_err(|_| FixtureError::InconsistentExpectedObservables)?
            .ok_or(FixtureError::InconsistentExpectedObservables)?;
    let observed_temperature = emitted_temperature * expected.frequency_ratio.value;
    let vacuum_bands = blackbody_band_intensities(vacuum, observed_temperature)
        .ok_or(FixtureError::InconsistentExpectedObservables)?;
    let transported = transport_bolometric_intensity(vacuum, slab)
        .ok_or(FixtureError::InconsistentExpectedObservables)?;
    let transported_bands = transport_blackbody_bands(vacuum_bands, slab)
        .map_err(|_| FixtureError::InconsistentExpectedObservables)?;
    let expected_bands = decimal_array(&expected.observed_spectral_band_intensities);
    let temperature_matches = relative_error(
        emitted_temperature,
        expected.emitted_temperature_kelvin.value,
    ) <= tolerance.temperature_rel.value
        && relative_error(
            observed_temperature,
            expected.vacuum_observed_temperature_kelvin.value,
        ) <= tolerance.temperature_rel.value;
    let scalar_matches = (emitted_intensity - expected.emitted_bolometric_intensity.value).abs()
        <= tolerance.bolometric_intensity_abs.value
        && (vacuum - expected.vacuum_observed_bolometric_intensity.value).abs()
            <= tolerance.bolometric_intensity_abs.value
        && (transported.0 - expected.observed_bolometric_intensity.value).abs()
            <= tolerance.bolometric_intensity_abs.value
        && (transported.1 - expected.optical_depth.value).abs()
            <= tolerance.bolometric_intensity_abs.value;
    let bands_match =
        transported_bands
            .into_iter()
            .zip(expected_bands)
            .all(|(actual, expected)| {
                (actual - expected).abs() <= tolerance.spectral_intensity_abs.value
            });
    if temperature_matches && scalar_matches && bands_match {
        Ok(())
    } else {
        Err(FixtureError::InconsistentExpectedObservables)
    }
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected
}

fn decimal_array<const N: usize>(values: &[DecimalString; N]) -> [f64; N] {
    std::array::from_fn(|index| values[index].value)
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawSurfaceTransportFixture {
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
    transport: RawTransport,
    expected: RawExpected,
    tolerance: RawTolerance,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawProducer {
    method: String,
    precision_digits: u32,
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
    temperature_at_6m_kelvin: DecimalString,
}

#[derive(Deserialize, PartialEq)]
#[serde(tag = "model", rename_all = "kebab-case", deny_unknown_fields)]
enum RawTransport {
    Vacuum {},
    PureAbsorption {
        optical_depth: DecimalString,
    },
    ConstantBlackbody {
        optical_depth: DecimalString,
        source_bolometric_intensity: DecimalString,
        source_temperature_kelvin: DecimalString,
    },
    PureEmissionBlackbody {
        integrated_bolometric_emission: DecimalString,
        emission_temperature_kelvin: DecimalString,
    },
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
    vacuum_observed_bolometric_intensity: DecimalString,
    observed_bolometric_intensity: DecimalString,
    optical_depth: DecimalString,
    emitted_temperature_kelvin: DecimalString,
    vacuum_observed_temperature_kelvin: DecimalString,
    observed_spectral_band_intensities: [DecimalString; 3],
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawTolerance {
    source_coordinate_abs_m: DecimalString,
    source_azimuth_abs_rad: DecimalString,
    frequency_ratio_rel: DecimalString,
    travel_time_abs_m: DecimalString,
    bolometric_intensity_abs: DecimalString,
    temperature_rel: DecimalString,
    spectral_intensity_abs: DecimalString,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum FixtureKind {
    SurfaceObservation,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Evidence {
    AlgebraicHighPrecisionAndReferenceConvergence,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Rotation {
    ProgradeCircular,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum EmissionModel {
    InverseCubeBlackbodyV1,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum ExpectedTermination {
    EquatorialSurface,
}

use std::num::NonZeroU32;

use gravlume_domain::{
    Angle, Extremality, GeodesicState, KerrNewmanSpacetime, KerrSchildChart, Observation,
    PerspectiveView, PhysicalScene, PhysicalSceneInput, StationaryObserverInput,
};
use serde::Deserialize;

use crate::{
    AffineDirection, EventConfiguration, ReferencePolicy, Termination, TraceInputId,
    events::escape_event_is_armed,
};

use super::{
    DecimalString, ExpectedOutcome, FixtureDocument, FixtureError, GeodesicFixture,
    ObservationFixture, invalid_physical_data,
};

const PRODUCER_PRECISION_DIGITS: u32 = 80;
const GEODESIC_INITIAL_NULL_ABS_MAX: f64 = 1.0e-80;
const OBSERVATION: &str = include_str!("../../fixtures/v1/default-kerr-observation.toml");
const SCATTER_B6: &str = include_str!("../../fixtures/v1/schwarzschild-scatter-b6.toml");
const SCATTER_NEAR_CRITICAL: &str =
    include_str!("../../fixtures/v1/schwarzschild-scatter-near-critical.toml");
const CAPTURE_NEAR_CRITICAL: &str =
    include_str!("../../fixtures/v1/schwarzschild-capture-near-critical.toml");

pub fn parse(source: &str) -> Result<FixtureDocument, FixtureError> {
    let raw: RawFixtureDocument = toml::from_str(source)?;
    match raw {
        RawFixtureDocument::Observation(raw) => raw.try_into(),
        RawFixtureDocument::Geodesic(raw) => raw.try_into(),
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum RawFixtureDocument {
    Observation(RawObservationFixture),
    Geodesic(RawGeodesicFixture),
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawObservationFixture {
    schema_version: u32,
    profile: String,
    id: String,
    evidence: Evidence,
    producer: RawProducer,
    spacetime: RawSpacetime,
    observer: RawObserver,
    viewport: RawViewport,
    events: RawObservationEvents,
    expected: RawObservationExpected,
    tolerance: RawObservationTolerance,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawGeodesicFixture {
    schema_version: u32,
    profile: String,
    id: String,
    evidence: Evidence,
    producer: RawProducer,
    spacetime: RawSpacetime,
    initial: RawInitial,
    events: RawGeodesicEvents,
    applicability: RawApplicability,
    expected: RawGeodesicExpected,
    tolerance: RawGeodesicTolerance,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawProducer {
    method: String,
    precision_digits: u32,
    cross_form_azimuth_disagreement: Option<DecimalString>,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawSpacetime {
    #[serde(rename = "family")]
    _family: SpacetimeFamily,
    #[serde(rename = "chart")]
    _chart: CoordinateChart,
    #[serde(rename = "signature")]
    _signature: MetricSignature,
    #[serde(rename = "component_order")]
    _component_order: ComponentOrder,
    mass_m: DecimalString,
    spin_m: DecimalString,
    charge_m: DecimalString,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawObserver {
    coordinate_time_m: DecimalString,
    oblate_radius_m: DecimalString,
    polar_angle_rad: DecimalString,
    azimuth_rad: DecimalString,
    #[serde(rename = "state")]
    _state: ObserverState,
    target_txyz_m: [DecimalString; 4],
    #[serde(rename = "up_hint")]
    _up_hint: UpHint,
    measured_frequency: DecimalString,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawViewport {
    projection: Projection,
    width: u32,
    height: u32,
    vertical_fov_rad: DecimalString,
    #[serde(rename = "origin")]
    _origin: ViewportOrigin,
    default_subpixel: [DecimalString; 2],
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawObservationEvents {
    escape_radius_m: DecimalString,
    singularity_guard_d_over_m4: DecimalString,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawObservationExpected {
    parameter_state: ExtremalityName,
    outer_horizon_radius_m: DecimalString,
    observer_cartesian_xyz_m: [DecimalString; 3],
    observer_g_tt: DecimalString,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawObservationTolerance {
    radius_abs_m: DecimalString,
    frame_gram_max_abs: DecimalString,
    initial_null_normalized_abs: DecimalString,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawInitial {
    position_txyz_m: [DecimalString; 4],
    momentum_covariant: [DecimalString; 4],
    affine_direction: AffineDirectionName,
    energy_at_infinity: DecimalString,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawGeodesicEvents {
    escape_radius_m: Option<DecimalString>,
    escape_event_initially_armed: Option<bool>,
    outer_horizon_radius_m: Option<DecimalString>,
}

#[derive(Deserialize, PartialEq)]
#[serde(tag = "class", rename_all = "kebab-case", deny_unknown_fields)]
enum RawApplicability {
    Regular {
        orbit: RawOrbit,
    },
    NearCritical {
        orbit: RawOrbit,
        critical_impact_parameter_m: DecimalString,
        impact_parameter_offset_m: DecimalString,
        side: RawCriticalSide,
    },
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawGeodesicExpected {
    termination: ExpectedTermination,
    turning_radius_m: Option<DecimalString>,
    event_radius_m: Option<DecimalString>,
    azimuth_advance_rad: DecimalString,
    initial_null_abs: DecimalString,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RawGeodesicTolerance {
    turning_radius_abs_m: Option<DecimalString>,
    event_radius_abs_m: Option<DecimalString>,
    azimuth_advance_abs_rad: DecimalString,
}

macro_rules! fixture_enum {
    ($name:ident { $($variant:ident => $wire:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
        enum $name {
            $(#[serde(rename = $wire)] $variant),+
        }
    };
}

fixture_enum!(SpacetimeFamily { KerrNewman => "kerr-newman" });
fixture_enum!(Evidence { Numeric => "numeric", AlgebraicAndNumeric => "algebraic-and-numeric" });
fixture_enum!(CoordinateChart { IngoingCartesianKerrSchild => "ingoing-cartesian-kerr-schild" });
fixture_enum!(MetricSignature { MostlyPlus => "mostly-plus" });
fixture_enum!(ComponentOrder { Txyz => "txyz" });
fixture_enum!(ObserverState { Stationary => "stationary" });
fixture_enum!(UpHint { SpinPositiveZ => "spin-positive-z" });
fixture_enum!(Projection { Perspective => "perspective" });
fixture_enum!(ViewportOrigin { TopLeft => "top-left" });
fixture_enum!(ExtremalityName { Subextremal => "subextremal" });
fixture_enum!(AffineDirectionName { Positive => "positive", Negative => "negative" });
fixture_enum!(RawOrbit {
    EquatorialSingleTurnScatter => "equatorial-single-turn-scatter",
    EquatorialCapture => "equatorial-capture"
});
fixture_enum!(RawCriticalSide { Escape => "escape", Capture => "capture" });
fixture_enum!(ExpectedTermination { Escape => "escape", HorizonCrossing => "horizon-crossing" });

impl TryFrom<RawObservationFixture> for FixtureDocument {
    type Error = FixtureError;

    fn try_from(raw: RawObservationFixture) -> Result<Self, Self::Error> {
        validate_envelope(raw.schema_version, &raw.profile)?;
        validate_observation_profile(&raw)?;
        let observation = build_observation(&raw)?;
        validate_observation_expected(&raw, &observation)?;
        Ok(Self::Observation(ObservationFixture {
            input_id: TraceInputId::new(raw.id),
            observation,
        }))
    }
}

impl TryFrom<RawGeodesicFixture> for FixtureDocument {
    type Error = FixtureError;

    fn try_from(raw: RawGeodesicFixture) -> Result<Self, Self::Error> {
        validate_envelope(raw.schema_version, &raw.profile)?;
        require_profile(raw.spacetime.mass_m.source == "1", "spacetime.mass_m")?;
        validate_geodesic_evidence(&raw)?;
        let spacetime = KerrNewmanSpacetime::new(
            raw.spacetime.mass_m.value,
            raw.spacetime.spin_m.value,
            raw.spacetime.charge_m.value,
            KerrSchildChart::Ingoing,
        )
        .map_err(invalid_physical_data)?;
        let initial_state = GeodesicState::new(
            decimal_array(&raw.initial.position_txyz_m),
            decimal_array(&raw.initial.momentum_covariant),
        )
        .map_err(invalid_physical_data)?;
        let events = validate_geodesic_events(&raw, spacetime, initial_state)?;
        let termination = validate_geodesic_expected(&raw)?;
        let initial_invariants = spacetime
            .invariants(initial_state)
            .map_err(invalid_physical_data)?;
        if raw.initial.energy_at_infinity.source != "1"
            || initial_invariants.energy().to_bits() != 1.0_f64.to_bits()
            || initial_invariants.normalized_null_residual() > 2.0e-12
            || raw.expected.initial_null_abs.value < 0.0
            || raw.expected.initial_null_abs.value > GEODESIC_INITIAL_NULL_ABS_MAX
        {
            return Err(FixtureError::InvalidPhysicalData(
                "initial energy/null contract is inconsistent".to_owned(),
            ));
        }
        validate_geodesic_applicability(
            &raw.applicability,
            termination,
            initial_invariants.energy(),
            initial_invariants.angular_momentum_z(),
        )?;
        validate_geodesic_profile(&raw)?;
        let affine_direction = match raw.initial.affine_direction {
            AffineDirectionName::Positive => AffineDirection::Positive,
            AffineDirectionName::Negative => AffineDirection::Negative,
        };
        let input_id =
            TraceInputId::new(raw.id).bind(spacetime, initial_state, affine_direction, events);
        let expected = ExpectedOutcome {
            input_id: input_id.clone(),
            termination,
            event_radius_m: raw.expected.event_radius_m.map(|value| value.value),
            turning_radius_m: raw.expected.turning_radius_m.map(|value| value.value),
            azimuth_advance_rad: raw.expected.azimuth_advance_rad.value,
            event_radius_absolute_tolerance_m: raw
                .tolerance
                .event_radius_abs_m
                .map(|value| value.value),
            turning_radius_absolute_tolerance_m: raw
                .tolerance
                .turning_radius_abs_m
                .map(|value| value.value),
            azimuth_advance_absolute_tolerance_rad: raw.tolerance.azimuth_advance_abs_rad.value,
            spacetime,
        };
        Ok(Self::Geodesic(GeodesicFixture {
            input_id,
            spacetime,
            initial_state,
            affine_direction,
            events,
            expected,
        }))
    }
}

fn validate_geodesic_profile(raw: &RawGeodesicFixture) -> Result<(), FixtureError> {
    let canonical_source = match raw.id.as_str() {
        "schwarzschild-scatter-b6-v1" => SCATTER_B6,
        "schwarzschild-scatter-near-critical-v1" => SCATTER_NEAR_CRITICAL,
        "schwarzschild-capture-near-critical-v1" => CAPTURE_NEAR_CRITICAL,
        _ => return Err(FixtureError::PresetMismatch { field: "id" }),
    };
    let canonical: RawFixtureDocument = toml::from_str(canonical_source)?;
    let RawFixtureDocument::Geodesic(canonical) = canonical else {
        return Err(FixtureError::PresetMismatch {
            field: "geodesic artifact",
        });
    };
    require_profile(raw == &canonical, "geodesic artifact")
}

fn validate_observation_profile(raw: &RawObservationFixture) -> Result<(), FixtureError> {
    let canonical: RawFixtureDocument = toml::from_str(OBSERVATION)?;
    let RawFixtureDocument::Observation(canonical) = canonical else {
        return Err(FixtureError::PresetMismatch {
            field: "observation artifact",
        });
    };
    require_profile(raw == &canonical, "observation artifact")
}

fn validate_geodesic_evidence(raw: &RawGeodesicFixture) -> Result<(), FixtureError> {
    let cross_form_is_valid = raw
        .producer
        .cross_form_azimuth_disagreement
        .as_ref()
        .is_some_and(|value| {
            value.value >= 0.0 && value.value <= raw.tolerance.azimuth_advance_abs_rad.value
        });
    if !matches!(raw.evidence, Evidence::Numeric)
        || raw.producer.precision_digits != PRODUCER_PRECISION_DIGITS
        || raw.producer.method.trim().is_empty()
        || !cross_form_is_valid
    {
        return Err(FixtureError::InvalidPhysicalData(
            "geodesic producer evidence is inconsistent with baseline-v1".to_owned(),
        ));
    }
    Ok(())
}

const fn require_profile(matches: bool, field: &'static str) -> Result<(), FixtureError> {
    if matches {
        Ok(())
    } else {
        Err(FixtureError::PresetMismatch { field })
    }
}

fn validate_geodesic_applicability(
    raw: &RawApplicability,
    termination: Termination,
    energy: f64,
    angular_momentum_z: f64,
) -> Result<(), FixtureError> {
    match raw {
        RawApplicability::Regular { orbit } => {
            if !orbit_matches_termination(*orbit, termination) {
                return Err(FixtureError::InconsistentApplicability);
            }
            Ok(())
        }
        RawApplicability::NearCritical {
            orbit,
            critical_impact_parameter_m,
            impact_parameter_offset_m,
            side,
        } => {
            let critical_impact_parameter_m = critical_impact_parameter_m.value;
            let impact_parameter_offset_m = impact_parameter_offset_m.value;
            let labeled_impact_parameter_m =
                critical_impact_parameter_m + impact_parameter_offset_m;
            let actual_impact_parameter_m = angular_momentum_z / energy;
            let side_matches_offset = match side {
                RawCriticalSide::Escape => impact_parameter_offset_m > 0.0,
                RawCriticalSide::Capture => impact_parameter_offset_m < 0.0,
            };
            let side_matches_orbit = matches!(
                (side, orbit),
                (
                    RawCriticalSide::Escape,
                    RawOrbit::EquatorialSingleTurnScatter
                ) | (RawCriticalSide::Capture, RawOrbit::EquatorialCapture)
            );
            if critical_impact_parameter_m <= 0.0
                || !side_matches_offset
                || !side_matches_orbit
                || !orbit_matches_termination(*orbit, termination)
                || !approximately_equal(actual_impact_parameter_m, labeled_impact_parameter_m)
            {
                return Err(FixtureError::InconsistentApplicability);
            }
            Ok(())
        }
    }
}

const fn orbit_matches_termination(orbit: RawOrbit, termination: Termination) -> bool {
    matches!(
        (orbit, termination),
        (RawOrbit::EquatorialSingleTurnScatter, Termination::Escape)
            | (RawOrbit::EquatorialCapture, Termination::HorizonCrossing)
    )
}

fn approximately_equal(left: f64, right: f64) -> bool {
    if !left.is_finite() || !right.is_finite() {
        return false;
    }
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= 64.0 * f64::EPSILON * scale
}

fn validate_geodesic_events(
    raw: &RawGeodesicFixture,
    spacetime: KerrNewmanSpacetime,
    initial_state: GeodesicState,
) -> Result<EventConfiguration, FixtureError> {
    let events = match (
        raw.events.escape_radius_m.as_ref(),
        raw.events.escape_event_initially_armed,
    ) {
        (None, None) => EventConfiguration::horizon_only(),
        (Some(escape_radius), Some(declared_armed)) => {
            let events = EventConfiguration::with_escape_radius(escape_radius.value)
                .map_err(|_| FixtureError::InconsistentEventEnvelope)?;
            let initial_radius = spacetime
                .radius(initial_state.event())
                .map_err(invalid_physical_data)?;
            let event_value = initial_radius - escape_radius.value;
            let actual_armed = escape_event_is_armed(
                event_value,
                ReferencePolicy::regular_v1().event_arming_band_m(),
            );
            if declared_armed != actual_armed {
                return Err(FixtureError::InconsistentEventEnvelope);
            }
            events
        }
        _ => return Err(FixtureError::InconsistentEventEnvelope),
    };
    if let Some(stored_horizon) = raw.events.outer_horizon_radius_m.as_ref() {
        let Some(actual_horizon) = spacetime.outer_horizon_radius() else {
            return Err(FixtureError::InconsistentEventEnvelope);
        };
        if (actual_horizon - stored_horizon.value).abs() > 8.0 * f64::EPSILON {
            return Err(FixtureError::InconsistentEventEnvelope);
        }
    }
    Ok(events)
}

fn validate_geodesic_expected(raw: &RawGeodesicFixture) -> Result<Termination, FixtureError> {
    let termination = match raw.expected.termination {
        ExpectedTermination::Escape => Termination::Escape,
        ExpectedTermination::HorizonCrossing => Termination::HorizonCrossing,
    };
    let expected_shape_is_valid = match termination {
        Termination::Escape => {
            raw.events.escape_radius_m.is_some()
                && raw.expected.turning_radius_m.is_some()
                && raw.expected.event_radius_m.is_none()
                && raw.tolerance.turning_radius_abs_m.is_some()
                && raw.tolerance.event_radius_abs_m.is_none()
        }
        Termination::HorizonCrossing => {
            raw.events.outer_horizon_radius_m.is_some()
                && raw.expected.event_radius_m.is_some()
                && raw.expected.turning_radius_m.is_none()
                && raw.tolerance.event_radius_abs_m.is_some()
                && raw.tolerance.turning_radius_abs_m.is_none()
        }
        _ => false,
    };
    if !expected_shape_is_valid {
        return Err(FixtureError::InconsistentEventEnvelope);
    }
    let tolerance_is_invalid = raw.tolerance.azimuth_advance_abs_rad.value <= 0.0
        || raw
            .tolerance
            .event_radius_abs_m
            .as_ref()
            .is_some_and(|value| value.value <= 0.0)
        || raw
            .tolerance
            .turning_radius_abs_m
            .as_ref()
            .is_some_and(|value| value.value <= 0.0);
    if tolerance_is_invalid {
        return Err(FixtureError::InvalidPhysicalData(
            "fixture tolerances must be positive".to_owned(),
        ));
    }
    Ok(termination)
}

fn validate_envelope(schema_version: u32, profile: &str) -> Result<(), FixtureError> {
    if schema_version != 1 {
        return Err(FixtureError::UnsupportedSchema(schema_version));
    }
    if profile != "baseline-v1" {
        return Err(FixtureError::UnsupportedProfile(profile.to_owned()));
    }
    Ok(())
}

fn build_observation(raw: &RawObservationFixture) -> Result<Observation, FixtureError> {
    let spacetime = KerrNewmanSpacetime::new(
        raw.spacetime.mass_m.value,
        raw.spacetime.spin_m.value,
        raw.spacetime.charge_m.value,
        KerrSchildChart::Ingoing,
    )
    .map_err(invalid_physical_data)?;
    let observer_xyz = spacetime.oblate_to_cartesian(
        raw.observer.oblate_radius_m.value,
        raw.observer.polar_angle_rad.value,
        raw.observer.azimuth_rad.value,
    );
    let observer_input = StationaryObserverInput::new(
        [
            raw.observer.coordinate_time_m.value,
            observer_xyz[0],
            observer_xyz[1],
            observer_xyz[2],
        ],
        decimal_array(&raw.observer.target_txyz_m),
        [0.0, 0.0, 1.0],
        raw.observer.measured_frequency.value,
    );
    let scene = PhysicalScene::new(PhysicalSceneInput::new(
        raw.spacetime.mass_m.value,
        raw.spacetime.spin_m.value,
        raw.spacetime.charge_m.value,
        KerrSchildChart::Ingoing,
        observer_input,
    ))
    .map_err(invalid_physical_data)?;
    let width = NonZeroU32::new(raw.viewport.width)
        .ok_or_else(|| FixtureError::InvalidPhysicalData("viewport width is zero".to_owned()))?;
    let height = NonZeroU32::new(raw.viewport.height)
        .ok_or_else(|| FixtureError::InvalidPhysicalData("viewport height is zero".to_owned()))?;
    let vertical_fov =
        Angle::from_radians(raw.viewport.vertical_fov_rad.value).map_err(invalid_physical_data)?;
    let view = PerspectiveView::new(width, height, vertical_fov).map_err(invalid_physical_data)?;
    let subpixel = decimal_array(&raw.viewport.default_subpixel);
    view.sample(0, 0, subpixel[0], subpixel[1])
        .map_err(invalid_physical_data)?;
    Ok(Observation::new(scene, view))
}

fn validate_observation_expected(
    raw: &RawObservationFixture,
    observation: &Observation,
) -> Result<(), FixtureError> {
    let scene = observation.scene();
    let event = scene.observer_event();
    let actual_xyz = event.to_txyz();
    let expected_xyz = decimal_array(&raw.expected.observer_cartesian_xyz_m);
    let radius = scene
        .spacetime()
        .radius(event)
        .map_err(invalid_physical_data)?;
    let radius_tolerance = raw.tolerance.radius_abs_m.value;
    let expected_extremality = match raw.expected.parameter_state {
        ExtremalityName::Subextremal => Extremality::Subextremal,
    };
    let expected_horizon = raw.expected.outer_horizon_radius_m.value;
    let horizon_matches = scene
        .spacetime()
        .outer_horizon_radius()
        .is_some_and(|actual| (actual - expected_horizon).abs() <= radius_tolerance);
    let position_matches = actual_xyz[1..]
        .iter()
        .zip(expected_xyz)
        .all(|(actual, expected)| (*actual - expected).abs() <= radius_tolerance);
    let frame_matches =
        scene.observer_frame().gram_residual() <= raw.tolerance.frame_gram_max_abs.value;
    let metric_matches =
        (scene.observer_metric_g_tt() - raw.expected.observer_g_tt.value).abs() <= radius_tolerance;
    let radius_matches = (radius - raw.observer.oblate_radius_m.value).abs() <= radius_tolerance;
    let center_sample = observation
        .view()
        .sample(raw.viewport.width / 2, raw.viewport.height / 2, 0.5, 0.5)
        .map_err(invalid_physical_data)?;
    let null_matches = observation
        .initial_ray(center_sample)
        .map_err(invalid_physical_data)?
        .normalized_null_residual()
        <= raw.tolerance.initial_null_normalized_abs.value;
    if scene.extremality() != expected_extremality
        || !horizon_matches
        || !position_matches
        || !frame_matches
        || !metric_matches
        || !radius_matches
        || !null_matches
    {
        return Err(FixtureError::InconsistentExpectedObservables);
    }
    Ok(())
}

fn decimal_array<const N: usize>(values: &[DecimalString; N]) -> [f64; N] {
    std::array::from_fn(|index| values[index].value)
}

use std::{fmt, num::NonZeroU32};

use gravlume_domain::{
    Angle, GeodesicState, KerrNewmanSpacetime, Observation, ParameterState, PhysicalScene,
    PhysicalSceneDraft, StationaryObserverDraft, ViewportProjection,
};
use serde::{Deserialize, Deserializer, de};

use crate::{
    AffineDirection, EventConfiguration, EventConfigurationError, ReferenceOutcome,
    ReferencePolicy, Termination, TraceInputId, TraceRequest,
    events::OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_M,
};

const MAX_FIXTURE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug)]
pub enum FixtureDocument {
    Observation(ObservationFixture),
    Geodesic(GeodesicFixture),
}

impl FixtureDocument {
    /// Parses one strict v1 TOML fixture document.
    ///
    /// Decimal inputs remain strings at the serialization seam and are converted explicitly.
    ///
    /// # Errors
    ///
    /// Rejects unknown fields/enums, non-v1 envelopes, invalid decimal strings, and invalid
    /// physical initial data.
    pub fn parse_toml(source: &str) -> Result<Self, FixtureError> {
        if source.len() > MAX_FIXTURE_BYTES {
            return Err(FixtureError::TooLarge {
                actual_bytes: source.len(),
                maximum_bytes: MAX_FIXTURE_BYTES,
            });
        }
        let raw: RawFixtureDocument = toml::from_str(source)?;
        match raw {
            RawFixtureDocument::Observation(raw) => raw.try_into(),
            RawFixtureDocument::Geodesic(raw) => raw.try_into(),
        }
    }

    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        match self {
            Self::Observation(fixture) => fixture.schema_version,
            Self::Geodesic(fixture) => fixture.schema_version,
        }
    }

    #[must_use]
    pub fn profile(&self) -> &str {
        match self {
            Self::Observation(fixture) => &fixture.profile,
            Self::Geodesic(fixture) => &fixture.profile,
        }
    }

    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Observation(fixture) => fixture.input_id.as_str(),
            Self::Geodesic(fixture) => fixture.input_id.as_str(),
        }
    }

    #[must_use]
    pub fn into_geodesic(self) -> Option<GeodesicFixture> {
        match self {
            Self::Geodesic(fixture) => Some(fixture),
            Self::Observation(_) => None,
        }
    }

    #[must_use]
    pub fn into_observation(self) -> Option<ObservationFixture> {
        match self {
            Self::Observation(fixture) => Some(fixture),
            Self::Geodesic(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ObservationFixture {
    schema_version: u32,
    profile: String,
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
pub struct GeodesicFixture {
    schema_version: u32,
    profile: String,
    input_id: TraceInputId,
    spacetime: KerrNewmanSpacetime,
    initial_state: GeodesicState,
    affine_direction: AffineDirection,
    escape_radius_m: Option<f64>,
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

    /// Resolves the versioned event configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if a stored escape radius violates the runtime event seam.
    pub fn event_configuration(&self) -> Result<EventConfiguration, EventConfigurationError> {
        self.escape_radius_m.map_or_else(
            || Ok(EventConfiguration::horizon_only()),
            EventConfiguration::with_escape_radius,
        )
    }

    #[must_use]
    pub fn trace_request(&self) -> TraceRequest {
        TraceRequest::new(
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
        if outcome.termination() != self.termination
            || (outcome.azimuth_advance_rad() - self.azimuth_advance_rad).abs()
                > self.azimuth_advance_absolute_tolerance_rad
        {
            return false;
        }
        if let (Some(expected), Some(tolerance)) = (
            self.turning_radius_m,
            self.turning_radius_absolute_tolerance_m,
        ) && outcome
            .turning_radius_m()
            .is_none_or(|actual| (actual - expected).abs() > tolerance)
        {
            return false;
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

#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error("fixture has {actual_bytes} bytes; the maximum is {maximum_bytes}")]
    TooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("fixture TOML does not match the strict v1 schema: {0}")]
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
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum RawFixtureDocument {
    Observation(RawObservationFixture),
    Geodesic(RawGeodesicFixture),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawObservationFixture {
    schema_version: u32,
    profile: String,
    id: String,
    #[serde(rename = "evidence")]
    _evidence: Evidence,
    #[serde(rename = "producer")]
    _producer: RawProducer,
    spacetime: RawSpacetime,
    observer: RawObserver,
    viewport: RawViewport,
    events: RawObservationEvents,
    expected: RawObservationExpected,
    tolerance: RawObservationTolerance,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGeodesicFixture {
    schema_version: u32,
    profile: String,
    id: String,
    #[serde(rename = "evidence")]
    _evidence: Evidence,
    #[serde(rename = "producer")]
    _producer: RawProducer,
    spacetime: RawSpacetime,
    initial: RawInitial,
    events: RawGeodesicEvents,
    #[serde(rename = "applicability")]
    _applicability: RawApplicability,
    expected: RawGeodesicExpected,
    tolerance: RawGeodesicTolerance,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProducer {
    #[serde(rename = "method")]
    _method: String,
    #[serde(rename = "precision_digits")]
    _precision_digits: u32,
    #[serde(rename = "cross_form_azimuth_disagreement")]
    _cross_form_azimuth_disagreement: Option<DecimalString>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSpacetime {
    family: SpacetimeFamily,
    chart: CoordinateChart,
    signature: MetricSignature,
    component_order: ComponentOrder,
    mass_m: DecimalString,
    spin_m: DecimalString,
    charge_m: DecimalString,
}

#[derive(Deserialize)]
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawViewport {
    #[serde(rename = "projection")]
    _projection: Projection,
    width: u32,
    height: u32,
    vertical_fov_rad: DecimalString,
    #[serde(rename = "origin")]
    _origin: ViewportOrigin,
    default_subpixel: [DecimalString; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawObservationEvents {
    escape_radius_m: DecimalString,
    singularity_guard_d_over_m4: DecimalString,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawObservationExpected {
    parameter_state: ParameterStateName,
    outer_horizon_radius_m: DecimalString,
    observer_cartesian_xyz_m: [DecimalString; 3],
    observer_g_tt: DecimalString,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawObservationTolerance {
    radius_abs_m: DecimalString,
    frame_gram_max_abs: DecimalString,
    initial_null_normalized_abs: DecimalString,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInitial {
    position_txyz_m: [DecimalString; 4],
    momentum_covariant: [DecimalString; 4],
    affine_direction: AffineDirectionName,
    energy_at_infinity: DecimalString,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGeodesicEvents {
    escape_radius_m: Option<DecimalString>,
    escape_event_initially_armed: Option<bool>,
    outer_horizon_radius_m: Option<DecimalString>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawApplicability {
    #[serde(rename = "class")]
    _class: ApplicabilityClass,
    #[serde(rename = "orbit")]
    _orbit: Orbit,
    #[serde(rename = "critical_impact_parameter_m")]
    _critical_impact_parameter_m: Option<DecimalString>,
    #[serde(rename = "impact_parameter_offset_m")]
    _impact_parameter_offset_m: Option<DecimalString>,
    #[serde(rename = "side")]
    _side: Option<CriticalSide>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGeodesicExpected {
    termination: ExpectedTermination,
    turning_radius_m: Option<DecimalString>,
    event_radius_m: Option<DecimalString>,
    azimuth_advance_rad: DecimalString,
    initial_null_abs: DecimalString,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGeodesicTolerance {
    turning_radius_abs_m: Option<DecimalString>,
    event_radius_abs_m: Option<DecimalString>,
    azimuth_advance_abs_rad: DecimalString,
}

#[derive(Clone)]
struct DecimalString {
    source: String,
    value: f64,
}

impl fmt::Debug for DecimalString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("DecimalString")
            .field(&self.source)
            .finish()
    }
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

macro_rules! fixture_enum {
    ($name:ident { $($variant:ident => $wire:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Deserialize)]
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
fixture_enum!(ParameterStateName { Subextremal => "subextremal" });
fixture_enum!(AffineDirectionName { Positive => "positive", Negative => "negative" });
fixture_enum!(ApplicabilityClass { Regular => "regular", NearCritical => "near-critical" });
fixture_enum!(Orbit {
    EquatorialSingleTurnScatter => "equatorial-single-turn-scatter",
    EquatorialCapture => "equatorial-capture"
});
fixture_enum!(CriticalSide { Escape => "escape", Capture => "capture" });
fixture_enum!(ExpectedTermination { Escape => "escape", HorizonCrossing => "horizon-crossing" });

impl TryFrom<RawObservationFixture> for FixtureDocument {
    type Error = FixtureError;

    fn try_from(raw: RawObservationFixture) -> Result<Self, Self::Error> {
        validate_envelope(raw.schema_version, &raw.profile)?;
        validate_conventions(&raw.spacetime);
        let observation = build_observation(&raw)?;
        validate_observation_expected(&raw, &observation)?;
        Ok(Self::Observation(ObservationFixture {
            schema_version: raw.schema_version,
            profile: raw.profile,
            input_id: TraceInputId::new(raw.id),
            observation,
        }))
    }
}

impl TryFrom<RawGeodesicFixture> for FixtureDocument {
    type Error = FixtureError;

    fn try_from(raw: RawGeodesicFixture) -> Result<Self, Self::Error> {
        validate_envelope(raw.schema_version, &raw.profile)?;
        validate_conventions(&raw.spacetime);
        let spacetime = KerrNewmanSpacetime::new(
            raw.spacetime.mass_m.value,
            raw.spacetime.spin_m.value,
            raw.spacetime.charge_m.value,
        )
        .map_err(|error| FixtureError::InvalidPhysicalData(error.to_string()))?;
        validate_geodesic_events(&raw, spacetime)?;
        let termination = validate_geodesic_expected(&raw)?;
        let initial_state = GeodesicState::new(
            raw.initial.position_txyz_m.map(|value| value.value),
            raw.initial.momentum_covariant.map(|value| value.value),
        )
        .map_err(|error| FixtureError::InvalidPhysicalData(error.to_string()))?;
        let initial_invariants = spacetime
            .invariants(initial_state)
            .map_err(invalid_physical_data)?;
        if (initial_invariants.energy() - raw.initial.energy_at_infinity.value).abs()
            > 32.0 * f64::EPSILON
            || initial_invariants.normalized_null_residual() > 2.0e-12
            || raw.expected.initial_null_abs.value < 0.0
        {
            return Err(FixtureError::InvalidPhysicalData(
                "initial energy/null contract is inconsistent".to_owned(),
            ));
        }
        let affine_direction = match raw.initial.affine_direction {
            AffineDirectionName::Positive => AffineDirection::Positive,
            AffineDirectionName::Negative => AffineDirection::Negative,
        };
        let expected = ExpectedOutcome {
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
            schema_version: raw.schema_version,
            profile: raw.profile,
            input_id: TraceInputId::new(raw.id),
            spacetime,
            initial_state,
            affine_direction,
            escape_radius_m: raw.events.escape_radius_m.map(|value| value.value),
            expected,
        }))
    }
}

fn validate_geodesic_events(
    raw: &RawGeodesicFixture,
    spacetime: KerrNewmanSpacetime,
) -> Result<(), FixtureError> {
    if raw.events.escape_radius_m.is_some()
        && raw.events.escape_event_initially_armed != Some(false)
    {
        return Err(FixtureError::InconsistentEventEnvelope);
    }
    if let Some(stored_horizon) = raw.events.outer_horizon_radius_m.as_ref() {
        let Some(actual_horizon) = spacetime.outer_horizon_radius() else {
            return Err(FixtureError::InconsistentEventEnvelope);
        };
        if (actual_horizon - stored_horizon.value).abs() > 8.0 * f64::EPSILON {
            return Err(FixtureError::InconsistentEventEnvelope);
        }
    }
    Ok(())
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
    )
    .map_err(invalid_physical_data)?;
    let observer_xyz = spacetime.oblate_to_cartesian(
        raw.observer.oblate_radius_m.value,
        raw.observer.polar_angle_rad.value,
        raw.observer.azimuth_rad.value,
    );
    let observer_draft = StationaryObserverDraft::new(
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
    let scene = PhysicalScene::commit(PhysicalSceneDraft::new(
        raw.spacetime.mass_m.value,
        raw.spacetime.spin_m.value,
        raw.spacetime.charge_m.value,
        observer_draft,
    ))
    .map_err(invalid_physical_data)?;
    let width = NonZeroU32::new(raw.viewport.width)
        .ok_or_else(|| FixtureError::InvalidPhysicalData("viewport width is zero".to_owned()))?;
    let height = NonZeroU32::new(raw.viewport.height)
        .ok_or_else(|| FixtureError::InvalidPhysicalData("viewport height is zero".to_owned()))?;
    let vertical_fov =
        Angle::from_radians(raw.viewport.vertical_fov_rad.value).map_err(invalid_physical_data)?;
    let projection = ViewportProjection::perspective(width, height, vertical_fov)
        .map_err(invalid_physical_data)?;
    let subpixel = decimal_array(&raw.viewport.default_subpixel);
    projection
        .sample(0, 0, subpixel[0], subpixel[1])
        .map_err(invalid_physical_data)?;
    if raw.events.escape_radius_m.value != OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_M
        || (raw.events.singularity_guard_d_over_m4.value
            - ReferencePolicy::regular_v1().singularity_guard_d_over_m4())
        .abs()
            > f64::EPSILON
    {
        return Err(FixtureError::InconsistentEventEnvelope);
    }
    Observation::new(scene, projection).map_err(invalid_physical_data)
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
    let expected_parameter_state = match raw.expected.parameter_state {
        ParameterStateName::Subextremal => ParameterState::Subextremal,
    };
    let expected_horizon = raw.expected.outer_horizon_radius_m.value;
    let horizon_matches = scene
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
        .projection()
        .sample(raw.viewport.width / 2, raw.viewport.height / 2, 0.5, 0.5)
        .map_err(invalid_physical_data)?;
    let null_matches = observation
        .initial_ray(center_sample)
        .map_err(invalid_physical_data)?
        .normalized_null_residual()
        <= raw.tolerance.initial_null_normalized_abs.value;
    if scene.parameter_state() != expected_parameter_state
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

fn invalid_physical_data(error: impl fmt::Display) -> FixtureError {
    FixtureError::InvalidPhysicalData(error.to_string())
}

const fn validate_conventions(spacetime: &RawSpacetime) {
    match (
        spacetime.family,
        spacetime.chart,
        spacetime.signature,
        spacetime.component_order,
    ) {
        (
            SpacetimeFamily::KerrNewman,
            CoordinateChart::IngoingCartesianKerrSchild,
            MetricSignature::MostlyPlus,
            ComponentOrder::Txyz,
        ) => {}
    }
}

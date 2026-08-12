use std::num::NonZeroUsize;

use gravlume_reference::{
    AffineDirection, ComparisonError, CriticalSide, FixtureDocument, FixtureError,
    GeodesicApplicability, GeodesicFixture, GeodesicOrbit, ObservationFixture, ReferenceBatch,
    ReferenceComparison, ReferenceInstrument, ReferenceOutcome, ReferencePolicy, ReferenceRequest,
    ReferenceTracer, Termination, TraceInputId, TraceRequest,
};

const SCATTER_B6: &str = include_str!("../../../tests/fixtures/v1/schwarzschild-scatter-b6.toml");
const SCATTER_NEAR_CRITICAL: &str =
    include_str!("../../../tests/fixtures/v1/schwarzschild-scatter-near-critical.toml");
const CAPTURE_NEAR_CRITICAL: &str =
    include_str!("../../../tests/fixtures/v1/schwarzschild-capture-near-critical.toml");
const DEFAULT_OBSERVATION: &str =
    include_str!("../../../tests/fixtures/v1/default-kerr-observation.toml");

fn edit_fixture(source: &str, edit: impl FnOnce(&mut toml::Value)) -> String {
    let mut document = toml::from_str(source).expect("repository fixture is valid TOML");
    edit(&mut document);
    toml::to_string(&document).expect("edited fixture remains representable as TOML")
}

fn table_at_mut<'a>(mut value: &'a mut toml::Value, path: &[&str]) -> &'a mut toml::Table {
    for segment in path {
        value = value
            .get_mut(*segment)
            .unwrap_or_else(|| panic!("fixture path segment {segment:?} exists"));
    }
    value.as_table_mut().expect("fixture path names a table")
}

fn set_field(source: &str, path: &[&str], value: toml::Value) -> String {
    edit_fixture(source, |document| {
        let (field, parent) = path.split_last().expect("field path is nonempty");
        table_at_mut(document, parent).insert((*field).to_owned(), value);
    })
}

fn decimal(value: &str) -> toml::Value {
    toml::Value::String(value.to_owned())
}

#[test]
fn fixture_envelope_is_strict_and_size_bounded() {
    let with_unknown = set_field(SCATTER_B6, &["undeclared"], toml::Value::Boolean(true));
    let oversized = "x".repeat(1024 * 1024 + 1);

    assert!(matches!(
        FixtureDocument::parse_toml(&with_unknown),
        Err(FixtureError::Toml(_))
    ));
    assert!(matches!(
        FixtureDocument::parse_toml(&oversized),
        Err(FixtureError::TooLarge { .. })
    ));
}

#[test]
fn observation_fixture_requires_the_exact_canonical_artifact() {
    let mutations = [
        (&["spacetime", "mass_m"][..], decimal("1.00000000000000001")),
        (
            &["expected", "observer_g_tt"][..],
            decimal("-0.9333451830784638"),
        ),
    ];

    for (path, value) in mutations {
        let mutated = set_field(DEFAULT_OBSERVATION, path, value);
        assert!(matches!(
            FixtureDocument::parse_toml(&mutated),
            Err(FixtureError::PresetMismatch { .. })
        ));
    }
}

#[test]
fn geodesic_fixture_rejects_versioned_oracle_drift() {
    for (path, value) in [
        (&["expected", "azimuth_advance_rad"][..], decimal("5.5")),
        (&["tolerance", "azimuth_advance_abs_rad"][..], decimal("1")),
    ] {
        let mutated = set_field(SCATTER_B6, path, value);
        assert!(matches!(
            FixtureDocument::parse_toml(&mutated),
            Err(FixtureError::PresetMismatch { .. })
        ));
    }
}

#[test]
fn geodesic_fixture_validates_declared_initial_escape_arming() {
    let initially_inside_escape_surface =
        set_field(SCATTER_B6, &["events", "escape_radius_m"], decimal("100"));
    let arming_without_escape_event = set_field(
        CAPTURE_NEAR_CRITICAL,
        &["events", "escape_event_initially_armed"],
        toml::Value::Boolean(false),
    );

    for source in [
        &initially_inside_escape_surface,
        &arming_without_escape_event,
    ] {
        assert!(matches!(
            FixtureDocument::parse_toml(source),
            Err(FixtureError::InconsistentEventEnvelope)
        ));
    }
}

#[test]
fn near_critical_fixture_rejects_incomplete_or_contradictory_applicability() {
    let wrong_side = set_field(
        CAPTURE_NEAR_CRITICAL,
        &["applicability", "side"],
        toml::Value::String("escape".to_owned()),
    );
    let wrong_offset = set_field(
        CAPTURE_NEAR_CRITICAL,
        &["applicability", "impact_parameter_offset_m"],
        decimal("0.001"),
    );
    for source in [&wrong_side, &wrong_offset] {
        assert!(matches!(
            FixtureDocument::parse_toml(source),
            Err(FixtureError::InconsistentApplicability)
        ));
    }
}

#[test]
fn geodesic_fixture_retains_validated_near_critical_applicability() {
    let fixture = geodesic_fixture(CAPTURE_NEAR_CRITICAL);

    assert_eq!(
        fixture.applicability(),
        GeodesicApplicability::NearCritical {
            orbit: GeodesicOrbit::EquatorialCapture,
            critical_impact_parameter_m: 5.196_152_422_706_632,
            impact_parameter_offset_m: -0.001,
            side: CriticalSide::Capture,
        }
    );
}

#[test]
fn regular_schwarzschild_fixture_matches_the_independent_observables() {
    let fixture = geodesic_fixture(SCATTER_B6);
    let tracer = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1());
    let outcome = tracer.trace(fixture.trace_request());

    assert_eq!(outcome.termination(), Termination::Escape);
    assert!(outcome.event().is_some_and(|event| {
        event.bracket_width_m() <= ReferencePolicy::regular_v1().event_affine_tolerance_m()
            && event.normalized_residual() <= 5.0e-12
            && !event.is_ambiguous()
    }));
    assert!(fixture.expected().accepts(&outcome));
    assert!(outcome.diagnostics().maximum_null_residual() < 5.0e-9);
}

#[test]
fn fixture_oracle_rejects_an_outcome_with_a_different_input_identity() {
    let fixture = geodesic_fixture(SCATTER_B6);
    let request = fixture.trace_request();
    let outcome = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1()).trace(
        TraceRequest::new(
            TraceInputId::new("different-input"),
            request.initial_state(),
            request.affine_direction(),
        ),
    );

    assert!(!fixture.expected().accepts(&outcome));
}

#[test]
fn regular_and_strict_outcomes_produce_a_passing_named_comparison_report() {
    let fixture = geodesic_fixture(SCATTER_B6);
    let regular = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1())
        .trace(fixture.trace_request());
    let strict = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::strict_v1())
        .trace(fixture.trace_request());
    let comparison = ReferenceComparison::baseline_v1(&regular, &strict)
        .expect("policy roles and input identity match");

    assert!(comparison.is_accepted(), "{:?}", comparison.issues());
}

#[test]
fn same_fixture_label_cannot_alias_different_canonical_inputs() {
    let fixture = geodesic_fixture(SCATTER_B6);
    let request = fixture.trace_request();
    let tracer = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1());
    let regular = tracer.trace(request.clone());
    let mut shifted_components = request.initial_state().components();
    shifted_components[0] = 1.0;
    let shifted_state = gravlume_domain::GeodesicState::from_components(shifted_components)
        .expect("shifted state is finite");
    let strict = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::strict_v1()).trace(
        TraceRequest::new(
            TraceInputId::new("schwarzschild-scatter-b6-v1"),
            shifted_state,
            request.affine_direction(),
        ),
    );

    assert!(matches!(
        ReferenceComparison::baseline_v1(&regular, &strict),
        Err(ComparisonError::InputIdentityCollision { input_id })
            if input_id == TraceInputId::new("schwarzschild-scatter-b6-v1")
    ));
}

#[test]
fn baseline_comparison_rejects_wrong_policy_roles() {
    let fixture = geodesic_fixture(SCATTER_B6);
    let strict = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::strict_v1())
        .trace(fixture.trace_request());

    let regular = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1())
        .trace(fixture.trace_request());

    assert!(matches!(
        ReferenceComparison::baseline_v1(&strict, &strict),
        Err(ComparisonError::UnexpectedBaselinePolicy { .. })
    ));
    assert!(matches!(
        ReferenceComparison::baseline_v1(&regular, &regular),
        Err(ComparisonError::UnexpectedCandidatePolicy { .. })
    ));
}

#[test]
fn baseline_comparison_rejects_different_input_ids() {
    let fixture = geodesic_fixture(SCATTER_B6);
    let request = fixture.trace_request();
    let regular = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1()).trace(
        TraceRequest::new(
            TraceInputId::new("regular-input"),
            request.initial_state(),
            request.affine_direction(),
        ),
    );
    let strict = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::strict_v1()).trace(
        TraceRequest::new(
            TraceInputId::new("strict-input"),
            request.initial_state(),
            request.affine_direction(),
        ),
    );

    assert_eq!(
        ReferenceComparison::baseline_v1(&regular, &strict),
        Err(ComparisonError::InputMismatch {
            baseline_input_id: TraceInputId::new("regular-input"),
            candidate_input_id: TraceInputId::new("strict-input"),
        })
    );
}

#[test]
fn turning_radius_is_dense_localized_for_negative_affine_traversal() {
    let fixture = geodesic_fixture(SCATTER_B6);
    let tracer = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::strict_v1());
    let forward = tracer.trace(fixture.trace_request());
    let backward = tracer.trace(TraceRequest::new(
        TraceInputId::new("schwarzschild-scatter-b6-v1-reverse"),
        forward.state(),
        AffineDirection::Negative,
    ));

    assert_eq!(backward.termination(), Termination::Escape);
    let forward_turning_radius = forward.turning_radius_m().expect("forward ray turns");
    let backward_turning_radius = backward.turning_radius_m().expect("backward ray turns");
    assert!((backward_turning_radius - forward_turning_radius).abs() <= 5.0e-11);
}

#[test]
fn travel_time_is_independent_of_the_coordinate_time_origin() {
    let fixture = geodesic_fixture(SCATTER_B6);
    let tracer = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1());
    let request = fixture.trace_request();
    let baseline = tracer.trace(request.clone());
    let mut shifted_components = request.initial_state().components();
    shifted_components[0] = 1.0e300;
    let shifted_state = gravlume_domain::GeodesicState::from_components(shifted_components)
        .expect("shifted state is finite");
    let shifted = tracer.trace(TraceRequest::new(
        TraceInputId::new("schwarzschild-scatter-b6-shifted-time"),
        shifted_state,
        request.affine_direction(),
    ));

    assert_eq!(shifted.termination(), baseline.termination());
    assert!(shifted.travel_time_m() > 0.0);
    assert!((shifted.travel_time_m() - baseline.travel_time_m()).abs() <= 2.0e-8);
}

#[test]
fn near_critical_pair_preserves_the_independent_discrete_classification() {
    let escape = run_fixture(SCATTER_NEAR_CRITICAL, ReferencePolicy::strict_v1());
    let capture = run_fixture(CAPTURE_NEAR_CRITICAL, ReferencePolicy::strict_v1());

    assert_eq!(escape.termination(), Termination::Escape);
    assert_eq!(capture.termination(), Termination::HorizonCrossing);
    assert_eq!(capture.turning_radius_m(), None);
}

#[test]
fn dedicated_rayon_pool_preserves_input_order() {
    let fixture = geodesic_fixture(SCATTER_B6);
    let tracer = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1());
    let request = fixture.trace_request();
    let inputs = [
        TraceRequest::new(
            TraceInputId::new("input-7"),
            request.initial_state(),
            request.affine_direction(),
        ),
        TraceRequest::new(
            TraceInputId::new("input-3"),
            request.initial_state(),
            request.affine_direction(),
        ),
    ];
    let pool =
        ReferenceBatch::new(NonZeroUsize::new(2).expect("two is nonzero")).expect("pool builds");
    let outcomes = pool.trace_ordered(&tracer, &inputs);

    assert_eq!(outcomes[0].input_id().as_str(), "input-7");
    assert_eq!(outcomes[1].input_id().as_str(), "input-3");
}

#[test]
fn observation_interface_traces_backward_without_flipping_photon_time_orientation() {
    let fixture = observation_fixture(DEFAULT_OBSERVATION);
    let input_id = fixture.input_id().clone();
    let observation = fixture.observation();
    let sample = observation
        .projection()
        .sample(0, 0, 0.5, 0.5)
        .expect("corner sample is valid");
    let regular_request = ReferenceRequest::new(
        input_id.clone(),
        observation,
        sample,
        ReferencePolicy::regular_v1(),
    )
    .expect("sample is valid for the request observation");
    let regular = ReferenceInstrument::baseline_v1()
        .trace(regular_request)
        .expect("validated observation preserves internal invariants");

    assert_eq!(regular.termination(), Termination::Escape, "{regular:?}");
    assert!(regular.affine_parameter_m() < 0.0);
    assert!(regular.travel_time_m() > 0.0);
    assert_eq!(regular.input_id().as_str(), fixture.input_id().as_str());
    let escape_direction = regular
        .escape_direction_xyz()
        .expect("escape has a traversal direction");
    let terminal = regular.state().components();
    let direction_norm = escape_direction[0].mul_add(
        escape_direction[0],
        escape_direction[1].mul_add(
            escape_direction[1],
            escape_direction[2] * escape_direction[2],
        ),
    );
    let radial_dot = terminal[1].mul_add(
        escape_direction[0],
        terminal[2].mul_add(escape_direction[1], terminal[3] * escape_direction[2]),
    );
    assert!((direction_norm - 1.0).abs() <= 8.0 * f64::EPSILON);
    assert!(radial_dot > 0.0, "escape traversal must point outward");

    let strict_request =
        ReferenceRequest::new(input_id, observation, sample, ReferencePolicy::strict_v1())
            .expect("sample is valid for the request observation");
    let strict = ReferenceInstrument::baseline_v1()
        .trace(strict_request)
        .expect("validated observation preserves internal invariants");
    let comparison = ReferenceComparison::baseline_v1(&regular, &strict)
        .expect("policy roles and input identity match");
    assert!(comparison.is_accepted(), "{comparison:?}");
}

fn geodesic_fixture(source: &str) -> GeodesicFixture {
    FixtureDocument::parse_toml(source)
        .expect("fixture parses")
        .into_geodesic()
        .expect("fixture is geodesic")
}

fn observation_fixture(source: &str) -> ObservationFixture {
    FixtureDocument::parse_toml(source)
        .expect("fixture parses")
        .into_observation()
        .expect("fixture is an observation")
}

fn run_fixture(source: &str, policy: ReferencePolicy) -> ReferenceOutcome {
    let fixture = geodesic_fixture(source);
    let tracer = ReferenceTracer::from_fixture(&fixture, policy);
    let outcome = tracer.trace(fixture.trace_request());
    assert!(fixture.expected().accepts(&outcome));
    outcome
}

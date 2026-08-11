use std::{num::NonZeroUsize, sync::Arc};

use gravlume_reference::{
    AffineDirection, ComparisonError, FixtureDocument, FixtureError, ReferenceBatch,
    ReferenceComparison, ReferenceInstrument, ReferencePolicy, ReferenceRequest, ReferenceTracer,
    Termination, TraceInputId, TraceRequest,
};

const SCATTER_B6: &str = include_str!("../../../tests/fixtures/v1/schwarzschild-scatter-b6.toml");
const SCATTER_NEAR_CRITICAL: &str =
    include_str!("../../../tests/fixtures/v1/schwarzschild-scatter-near-critical.toml");
const CAPTURE_NEAR_CRITICAL: &str =
    include_str!("../../../tests/fixtures/v1/schwarzschild-capture-near-critical.toml");
const DEFAULT_OBSERVATION: &str =
    include_str!("../../../tests/fixtures/v1/default-kerr-observation.toml");

#[test]
fn v1_fixture_seam_accepts_the_repository_documents_and_rejects_unknown_fields() {
    for source in [
        SCATTER_B6,
        SCATTER_NEAR_CRITICAL,
        CAPTURE_NEAR_CRITICAL,
        DEFAULT_OBSERVATION,
    ] {
        let fixture = FixtureDocument::parse_toml(source).expect("v1 fixture is valid");
        assert_eq!(fixture.schema_version(), 1);
        assert_eq!(fixture.profile(), "baseline-v1");
    }

    let with_unknown = SCATTER_B6.replace(
        "schema_version = 1",
        "schema_version = 1\nundeclared = true",
    );
    assert!(FixtureDocument::parse_toml(&with_unknown).is_err());

    let oversized = "x".repeat(1024 * 1024 + 1);
    assert!(FixtureDocument::parse_toml(&oversized).is_err());
}

#[test]
fn observation_fixture_rejects_an_escape_radius_outside_the_v1_profile() {
    let wrong_escape_radius =
        DEFAULT_OBSERVATION.replace("escape_radius_m = \"200\"", "escape_radius_m = \"123\"");

    assert!(matches!(
        FixtureDocument::parse_toml(&wrong_escape_radius),
        Err(FixtureError::InconsistentEventEnvelope)
    ));
}

#[test]
fn regular_schwarzschild_fixture_matches_the_independent_observables() {
    let fixture = FixtureDocument::parse_toml(SCATTER_B6)
        .expect("fixture parses")
        .into_geodesic()
        .expect("fixture is geodesic");
    let tracer = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1())
        .expect("fixture config is valid");
    let outcome = tracer.trace(fixture.trace_request());

    assert_eq!(outcome.termination(), Termination::Escape);
    assert!(outcome.event().is_some_and(|event| {
        event.bracket_width_m() <= ReferencePolicy::regular_v1().event_affine_tolerance_m()
            && event.normalized_residual() <= 5.0e-12
            && !event.is_ambiguous()
    }));
    assert!(fixture.expected().accepts(&outcome));
    assert!(outcome.diagnostics().accepted_steps() > 0);
    assert_eq!(
        outcome.diagnostics().rhs_evaluations(),
        2 + 6 * (outcome.diagnostics().accepted_steps() + outcome.diagnostics().rejected_steps())
    );
    assert!(outcome.diagnostics().maximum_null_residual() < 5.0e-9);
}

#[test]
fn regular_and_strict_outcomes_produce_a_passing_named_comparison_report() {
    let fixture = FixtureDocument::parse_toml(SCATTER_B6)
        .expect("fixture parses")
        .into_geodesic()
        .expect("fixture is geodesic");
    let regular = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1())
        .expect("fixture config is valid")
        .trace(fixture.trace_request());
    let strict = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::strict_v1())
        .expect("fixture config is valid")
        .trace(fixture.trace_request());
    let comparison = ReferenceComparison::baseline_v1(&regular, &strict)
        .expect("policy roles and input identity match");

    assert!(comparison.is_accepted(), "{:?}", comparison.issues());
    assert_eq!(comparison.baseline_policy(), "reference-regular-v1");
    assert_eq!(comparison.candidate_policy(), "reference-strict-v1");
}

#[test]
fn baseline_comparison_rejects_wrong_policy_roles() {
    let fixture = FixtureDocument::parse_toml(SCATTER_B6)
        .expect("fixture parses")
        .into_geodesic()
        .expect("fixture is geodesic");
    let strict = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::strict_v1())
        .expect("fixture config is valid")
        .trace(fixture.trace_request());

    let regular = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1())
        .expect("fixture config is valid")
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
    let fixture = FixtureDocument::parse_toml(SCATTER_B6)
        .expect("fixture parses")
        .into_geodesic()
        .expect("fixture is geodesic");
    let request = fixture.trace_request();
    let regular = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1())
        .expect("fixture config is valid")
        .trace(TraceRequest::new(
            TraceInputId::new("regular-input"),
            request.initial_state(),
            request.affine_direction(),
        ));
    let strict = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::strict_v1())
        .expect("fixture config is valid")
        .trace(TraceRequest::new(
            TraceInputId::new("strict-input"),
            request.initial_state(),
            request.affine_direction(),
        ));

    assert_eq!(
        ReferenceComparison::baseline_v1(&regular, &strict),
        Err(ComparisonError::InputMismatch {
            baseline_input_id: TraceInputId::new("regular-input"),
            candidate_input_id: TraceInputId::new("strict-input"),
        })
    );
}

#[test]
fn fixture_requests_preserve_their_logical_input_identity() {
    let scatter = FixtureDocument::parse_toml(SCATTER_B6)
        .expect("fixture parses")
        .into_geodesic()
        .expect("fixture is geodesic");
    let capture = FixtureDocument::parse_toml(CAPTURE_NEAR_CRITICAL)
        .expect("fixture parses")
        .into_geodesic()
        .expect("fixture is geodesic");

    assert_ne!(
        scatter.trace_request().input_id(),
        capture.trace_request().input_id()
    );
}

#[test]
fn turning_radius_is_dense_localized_for_negative_affine_traversal() {
    let fixture = FixtureDocument::parse_toml(SCATTER_B6)
        .expect("fixture parses")
        .into_geodesic()
        .expect("fixture is geodesic");
    let tracer = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::strict_v1())
        .expect("fixture config is valid");
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
fn near_critical_pair_preserves_the_independent_discrete_classification() {
    let escape = run_fixture(SCATTER_NEAR_CRITICAL, ReferencePolicy::strict_v1());
    let capture = run_fixture(CAPTURE_NEAR_CRITICAL, ReferencePolicy::strict_v1());

    assert_eq!(escape.termination(), Termination::Escape);
    assert_eq!(capture.termination(), Termination::HorizonCrossing);
    assert_eq!(capture.turning_radius_m(), None);
}

#[test]
fn dedicated_rayon_pool_preserves_input_order() {
    let fixture = FixtureDocument::parse_toml(SCATTER_B6)
        .expect("fixture parses")
        .into_geodesic()
        .expect("fixture is geodesic");
    let tracer = ReferenceTracer::from_fixture(&fixture, ReferencePolicy::regular_v1())
        .expect("fixture config is valid");
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
    let fixture = FixtureDocument::parse_toml(DEFAULT_OBSERVATION)
        .expect("fixture parses")
        .into_observation()
        .expect("fixture is an observation");
    let input_id = fixture.input_id().clone();
    let observation = Arc::new(fixture.observation().clone());
    let sample = observation
        .projection()
        .sample(0, 0, 0.5, 0.5)
        .expect("corner sample is valid");
    let regular_request = ReferenceRequest::new(
        input_id.clone(),
        Arc::clone(&observation),
        sample,
        ReferencePolicy::regular_v1(),
    )
    .expect("sample is valid for the request observation");
    let regular = ReferenceInstrument::baseline_v1()
        .trace(regular_request)
        .expect("validated observation preserves internal invariants");

    assert_eq!(regular.termination(), Termination::Escape, "{regular:?}");
    assert!(regular.affine_parameter_m() < 0.0);
    assert_eq!(regular.input_id(), fixture.input_id());
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

fn run_fixture(source: &str, policy: ReferencePolicy) -> gravlume_reference::ReferenceOutcome {
    let fixture = FixtureDocument::parse_toml(source)
        .expect("fixture parses")
        .into_geodesic()
        .expect("fixture is geodesic");
    let tracer = ReferenceTracer::from_fixture(&fixture, policy).expect("fixture config is valid");
    let outcome = tracer.trace(fixture.trace_request());
    assert!(fixture.expected().accepts(&outcome));
    outcome
}

use std::{num::NonZeroUsize, sync::Arc};

use gravlume_reference::{
    FixtureDocument, ReferenceBatch, ReferenceComparison, ReferenceInstrument, ReferencePolicy,
    ReferenceRequest, ReferenceTracer, Termination, TraceRequest,
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
        1 + 6 * (outcome.diagnostics().accepted_steps() + outcome.diagnostics().rejected_steps())
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
    let comparison = ReferenceComparison::baseline_v1(&regular, &strict);

    assert!(comparison.is_accepted(), "{:?}", comparison.issues());
    assert_eq!(comparison.baseline_policy(), "reference-regular-v1");
    assert_eq!(comparison.candidate_policy(), "reference-strict-v1");
}

#[test]
fn near_critical_pair_preserves_the_independent_discrete_classification() {
    let escape = run_fixture(SCATTER_NEAR_CRITICAL, ReferencePolicy::strict_v1());
    let capture = run_fixture(CAPTURE_NEAR_CRITICAL, ReferencePolicy::strict_v1());

    assert_eq!(escape.termination(), Termination::Escape);
    assert_eq!(capture.termination(), Termination::HorizonCrossing);
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
        TraceRequest::new(7, request.initial_state(), request.affine_direction()),
        TraceRequest::new(3, request.initial_state(), request.affine_direction()),
    ];
    let pool =
        ReferenceBatch::new(NonZeroUsize::new(2).expect("two is nonzero")).expect("pool builds");
    let outcomes = pool.trace_ordered(&tracer, &inputs);

    assert_eq!(outcomes[0].input_id(), 7);
    assert_eq!(outcomes[1].input_id(), 3);
}

#[test]
fn observation_interface_traces_backward_without_flipping_photon_time_orientation() {
    let fixture = FixtureDocument::parse_toml(DEFAULT_OBSERVATION)
        .expect("fixture parses")
        .into_observation()
        .expect("fixture is an observation");
    let observation = Arc::new(fixture.observation().clone());
    let sample = observation
        .projection()
        .sample(0, 0, 0.5, 0.5)
        .expect("corner sample is valid");
    let regular_request = ReferenceRequest::new(
        Arc::clone(&observation),
        sample,
        ReferencePolicy::regular_v1(),
    );
    let regular = ReferenceInstrument::baseline_v1()
        .trace(regular_request)
        .expect("validated observation preserves internal invariants");

    assert_eq!(regular.termination(), Termination::Escape, "{regular:?}");
    assert!(regular.affine_parameter_m() < 0.0);

    let strict_request = ReferenceRequest::new(observation, sample, ReferencePolicy::strict_v1());
    let strict = ReferenceInstrument::baseline_v1()
        .trace(strict_request)
        .expect("validated observation preserves internal invariants");
    let comparison = ReferenceComparison::baseline_v1(&regular, &strict);
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

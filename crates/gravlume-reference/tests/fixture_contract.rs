use std::{num::NonZeroUsize, sync::Arc};

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

#[test]
fn repository_v1_fixture_documents_parse() {
    for source in [
        SCATTER_B6,
        SCATTER_NEAR_CRITICAL,
        CAPTURE_NEAR_CRITICAL,
        DEFAULT_OBSERVATION,
    ] {
        FixtureDocument::parse_toml(source).expect("v1 fixture is valid");
    }
}

#[test]
fn fixture_envelope_is_strict_and_size_bounded() {
    let with_unknown = SCATTER_B6.replace(
        "schema_version = 1",
        "schema_version = 1\nundeclared = true",
    );
    let oversized = "x".repeat(1024 * 1024 + 1);

    for source in [&with_unknown, &oversized] {
        assert!(FixtureDocument::parse_toml(source).is_err());
    }
}

#[test]
fn observation_fixture_rejects_events_outside_the_v1_profile() {
    for source in [
        DEFAULT_OBSERVATION.replace("escape_radius_m = \"200\"", "escape_radius_m = \"123\""),
        DEFAULT_OBSERVATION.replace(
            "singularity_guard_d_over_m4 = \"9.094947017729282379150390625e-13\"",
            "singularity_guard_d_over_m4 = \"1e-12\"",
        ),
    ] {
        assert!(matches!(
            FixtureDocument::parse_toml(&source),
            Err(FixtureError::InconsistentEventEnvelope)
        ));
    }
}

#[test]
fn observation_fixture_rejects_fixed_v1_profile_drift() {
    let mutations = [
        (
            "id = \"kerr-exterior-observation-v1\"",
            "id = \"other\"",
            "id",
        ),
        (
            "evidence = \"algebraic-and-numeric\"",
            "evidence = \"numeric\"",
            "evidence",
        ),
        ("precision_digits = 80", "precision_digits = 16", "producer"),
        ("mass_m = \"1\"", "mass_m = \"2\"", "spacetime.mass_m"),
        ("spin_m = \"0.8\"", "spin_m = \"0.7\"", "spacetime.spin_m"),
        (
            "charge_m = \"0\"",
            "charge_m = \"0.1\"",
            "spacetime.charge_m",
        ),
        (
            "coordinate_time_m = \"0\"",
            "coordinate_time_m = \"1\"",
            "observer.coordinate_time_m",
        ),
        (
            "oblate_radius_m = \"30\"",
            "oblate_radius_m = \"31\"",
            "observer.oblate_radius_m",
        ),
        (
            "polar_angle_rad = \"1.0471975511965977461542144610931676280657231331250352736583148641026054687620697\"",
            "polar_angle_rad = \"1\"",
            "observer.polar_angle_rad",
        ),
        (
            "azimuth_rad = \"0\"",
            "azimuth_rad = \"0.1\"",
            "observer.azimuth_rad",
        ),
        (
            "target_txyz_m = [\"0\", \"0\", \"0\", \"0\"]",
            "target_txyz_m = [\"0\", \"1\", \"0\", \"0\"]",
            "observer.target_txyz_m",
        ),
        (
            "measured_frequency = \"1\"",
            "measured_frequency = \"2\"",
            "observer.measured_frequency",
        ),
        ("width = 1280", "width = 640", "viewport.width"),
        ("height = 720", "height = 360", "viewport.height"),
        (
            "vertical_fov_rad = \"0.78539816339744830961566084581987572104929234984377645524373614807695410157155225\"",
            "vertical_fov_rad = \"1\"",
            "viewport.vertical_fov_rad",
        ),
        (
            "default_subpixel = [\"0.5\", \"0.5\"]",
            "default_subpixel = [\"0.25\", \"0.5\"]",
            "viewport.default_subpixel",
        ),
        (
            "radius_abs_m = \"2e-13\"",
            "radius_abs_m = \"1\"",
            "tolerance.radius_abs_m",
        ),
        (
            "frame_gram_max_abs = \"2e-12\"",
            "frame_gram_max_abs = \"1\"",
            "tolerance.frame_gram_max_abs",
        ),
        (
            "initial_null_normalized_abs = \"2e-12\"",
            "initial_null_normalized_abs = \"1\"",
            "tolerance.initial_null_normalized_abs",
        ),
        (
            "mass_m = \"1\"",
            "mass_m = \"1.00000000000000001\"",
            "spacetime.mass_m",
        ),
    ];

    for (original, replacement, expected_field) in mutations {
        let mutated = DEFAULT_OBSERVATION.replace(original, replacement);
        assert_ne!(mutated, DEFAULT_OBSERVATION, "mutation target must exist");
        assert!(matches!(
            FixtureDocument::parse_toml(&mutated),
            Err(FixtureError::PresetMismatch { field }) if field == expected_field
        ));
    }
}

#[test]
fn geodesic_fixture_rejects_inconsistent_high_precision_null_evidence() {
    let invalid_oracle = SCATTER_B6.replace(
        "initial_null_abs = \"4.91454214841681154173676126201e-81\"",
        "initial_null_abs = \"1\"",
    );
    let wrong_precision = SCATTER_B6.replace("precision_digits = 80", "precision_digits = 16");

    for source in [&invalid_oracle, &wrong_precision] {
        assert!(matches!(
            FixtureDocument::parse_toml(source),
            Err(FixtureError::InvalidPhysicalData(_))
        ));
    }
}

#[test]
fn geodesic_fixture_rejects_a_self_consistent_non_unit_energy_scale() {
    let scaled = SCATTER_B6
        .replace(
            concat!(
                "momentum_covariant = [\n",
                "  \"-1\",\n",
                "  \"-0.99277494330677424337355254840577841602316427996638407416694633896574636965694494\",\n",
                "  \"0.12\",\n",
                "  \"0\",\n",
                "]",
            ),
            concat!(
                "momentum_covariant = [\n",
                "  \"-2\",\n",
                "  \"-1.98554988661354848674710509681155683204632855993276814833389267793149273931388988\",\n",
                "  \"0.24\",\n",
                "  \"0\",\n",
                "]",
            ),
        )
        .replace("energy_at_infinity = \"1\"", "energy_at_infinity = \"2\"");

    assert!(matches!(
        FixtureDocument::parse_toml(&scaled),
        Err(FixtureError::InvalidPhysicalData(_))
    ));
}

#[test]
fn geodesic_fixture_rejects_noncanonical_unit_mass_text() {
    let mutated = SCATTER_B6.replace("mass_m = \"1\"", "mass_m = \"1.000000000000001\"");

    assert!(matches!(
        FixtureDocument::parse_toml(&mutated),
        Err(FixtureError::PresetMismatch {
            field: "spacetime.mass_m"
        })
    ));
}

#[test]
fn geodesic_fixture_validates_declared_initial_escape_arming() {
    let initially_inside_escape_surface =
        SCATTER_B6.replace("escape_radius_m = \"50\"", "escape_radius_m = \"100\"");
    let arming_without_escape_event = CAPTURE_NEAR_CRITICAL.replace(
        "outer_horizon_radius_m = \"2\"",
        "outer_horizon_radius_m = \"2\"\nescape_event_initially_armed = false",
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
    let wrong_side = CAPTURE_NEAR_CRITICAL.replace("side = \"capture\"", "side = \"escape\"");
    let wrong_offset = CAPTURE_NEAR_CRITICAL.replace(
        "impact_parameter_offset_m = \"-0.001\"",
        "impact_parameter_offset_m = \"0.001\"",
    );
    let missing_distance = CAPTURE_NEAR_CRITICAL.replace(
        "critical_impact_parameter_m = \"5.196152422706631880582339024517617100829\"\n",
        "",
    );
    let overflowing_distance = SCATTER_NEAR_CRITICAL
        .replace(
            "critical_impact_parameter_m = \"5.196152422706631880582339024517617100829\"",
            "critical_impact_parameter_m = \"1.7976931348623157e308\"",
        )
        .replace(
            "impact_parameter_offset_m = \"0.001\"",
            "impact_parameter_offset_m = \"1.7976931348623157e308\"",
        );
    let regular_with_critical_fields = SCATTER_B6.replace(
        "orbit = \"equatorial-single-turn-scatter\"",
        concat!(
            "orbit = \"equatorial-single-turn-scatter\"\n",
            "critical_impact_parameter_m = \"5.196152422706632\"\n",
            "impact_parameter_offset_m = \"0.803847577293368\"\n",
            "side = \"escape\"",
        ),
    );

    for source in [&wrong_side, &wrong_offset, &overflowing_distance] {
        assert!(matches!(
            FixtureDocument::parse_toml(source),
            Err(FixtureError::InconsistentApplicability)
        ));
    }
    for source in [&missing_distance, &regular_with_critical_fields] {
        assert!(matches!(
            FixtureDocument::parse_toml(source),
            Err(FixtureError::Toml(_))
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

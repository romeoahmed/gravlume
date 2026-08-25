use std::f64::consts::{PI, TAU};

use approx::assert_abs_diff_eq;
use gravlume_reference::{
    FixtureDocument, ObservationTrace, ObservationTracer, PolarSide, ReferenceComparison,
    ReferenceOutcome, ReferencePolicy, SurfaceFootprintError, SurfaceFootprintEstimate,
    SurfaceParity, Termination, TraceInputId,
};
use proptest::prelude::*;

const SURFACE_OBSERVABLE: &str = include_str!("../fixtures/v2/kerr-surface-observable.toml");
const TRANSPORT_FIXTURES: [&str; 4] = [
    include_str!("../fixtures/v3/kerr-blackbody-vacuum.toml"),
    include_str!("../fixtures/v3/kerr-blackbody-pure-absorption.toml"),
    include_str!("../fixtures/v3/kerr-blackbody-constant-slab.toml"),
    include_str!("../fixtures/v3/kerr-blackbody-pure-emission.toml"),
];

#[test]
fn regular_and_strict_surface_observables_close_the_vacuum_radiance_chain() {
    let fixture = FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let regular = ObservationTracer::baseline_v1()
        .trace(
            fixture
                .trace_request(ReferencePolicy::regular_v1())
                .expect("fixture sample resolves through its observation"),
        )
        .expect("the regular source model is valid at the localized hit");
    let strict = ObservationTracer::baseline_v1()
        .trace(
            fixture
                .trace_request(ReferencePolicy::strict_v1())
                .expect("fixture sample resolves through its observation"),
        )
        .expect("the strict source model is valid at the localized hit");
    let comparison = ReferenceComparison::baseline_v1(&regular, &strict)
        .expect("policy roles and surface input identity match");

    assert!(fixture.accepts(&regular));
    assert!(fixture.accepts(&strict));
    assert!(comparison.is_accepted(), "{:?}", comparison.issues());
}

#[test]
fn canonical_surface_matches_the_independent_bl_mino_witness() {
    // Generated at 120/180 decimal digits by the independent separated-chart
    // witness in docs/research/scripts/verify_bl_mino_surface_witness.py.
    const SOURCE_RADIUS_M: f64 = 19.650_678_984_603_292;
    const SOURCE_AZIMUTH_RAD: f64 = 3.087_156_262_423_669;
    const FREQUENCY_RATIO: f64 = 0.953_264_138_194_622_9;
    const TRAVEL_TIME_M: f64 = 54.902_474_247_630_05;
    const EMITTED_INTENSITY: f64 = 0.028_465_647_567_239_85;
    const OBSERVED_INTENSITY: f64 = 0.023_505_748_696_197_13;

    let fixture = FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");

    for policy in [ReferencePolicy::regular_v1(), ReferencePolicy::strict_v1()] {
        let outcome = ObservationTracer::baseline_v1()
            .trace(
                fixture
                    .trace_request(policy)
                    .expect("fixture sample resolves through its observation"),
            )
            .expect("canonical surface source is valid");
        assert_eq!(outcome.termination(), Termination::EquatorialSurface);
        let branch = outcome.branch_key();
        assert_eq!(branch.initial_polar_side(), PolarSide::Positive);
        assert_eq!(branch.radial_turnings(), 1);
        assert_eq!(branch.equatorial_crossings(), 0);
        assert_eq!(branch.azimuth_winding(), 0);

        let observable = outcome
            .terminal()
            .surface_observable()
            .expect("canonical trace carries its surface observable");
        let anchor = observable.source_anchor();
        let radial_difference = anchor.radius_m() - SOURCE_RADIUS_M;
        let azimuth_difference =
            (anchor.azimuth_rad() - SOURCE_AZIMUTH_RAD + PI).rem_euclid(TAU) - PI;
        let mean_radius = anchor.radius_m().midpoint(SOURCE_RADIUS_M);
        let source_anchor_distance_m = radial_difference.hypot(mean_radius * azimuth_difference);
        assert_abs_diff_eq!(source_anchor_distance_m, 0.0, epsilon = 2.0e-9);
        assert_abs_diff_eq!(
            observable.frequency_ratio().value(),
            FREQUENCY_RATIO,
            epsilon = 2.0e-9 * FREQUENCY_RATIO
        );
        assert_abs_diff_eq!(outcome.travel_time_m(), TRAVEL_TIME_M, epsilon = 2.0e-8);
        assert_abs_diff_eq!(
            observable.emitted_bolometric_intensity(),
            EMITTED_INTENSITY,
            epsilon = 1.0e-12
        );
        assert_abs_diff_eq!(
            observable.observed_bolometric_intensity(),
            OBSERVED_INTENSITY,
            epsilon = 1.0e-12
        );
    }
}

#[test]
fn source_edge_pair_matches_the_independent_bl_mino_witness() {
    // Generated at 120/180 decimal digits by the independent separated-chart
    // witness in docs/research/scripts/verify_bl_mino_surface_witness.py.
    let fixture = FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let observation = fixture.observation();
    let outside_sample = observation
        .view()
        .sample(640, 13, 0.5, 0.5)
        .expect("outside source-edge sample belongs to the canonical view");
    let inside_sample = observation
        .view()
        .sample(640, 14, 0.5, 0.5)
        .expect("inside source-edge sample belongs to the canonical view");
    let oracle = ObservationTracer::baseline_v1();

    for policy in [ReferencePolicy::regular_v1(), ReferencePolicy::strict_v1()] {
        let outside = oracle
            .trace(
                ObservationTrace::new(
                    TraceInputId::new(format!("source-edge-outside-{}", policy.id())),
                    observation,
                    outside_sample,
                    policy,
                )
                .expect("outside source-edge trace request resolves"),
            )
            .expect("outside source-edge trace succeeds");
        assert_source_edge_escape(&outside);

        let inside = oracle
            .trace(
                ObservationTrace::new(
                    TraceInputId::new(format!("source-edge-inside-{}", policy.id())),
                    observation,
                    inside_sample,
                    policy,
                )
                .expect("inside source-edge trace request resolves"),
            )
            .expect("inside source-edge trace succeeds");
        assert_source_edge_surface(&inside);
    }
}

fn assert_source_edge_escape(outcome: &ReferenceOutcome) {
    const POSITION_XYZ_M: [f64; 3] = [
        -170.447_402_756_461_1,
        1.369_244_882_783_220_7,
        -104.624_437_497_465_03,
    ];
    const DIRECTION_XYZ: [f64; 3] = [
        -0.820_715_680_321_071_6,
        0.006_023_904_198_189_695,
        -0.571_305_071_440_234_7,
    ];
    const TRAVEL_TIME_M: f64 = 238.438_694_378_676_36;

    assert_eq!(outcome.termination(), Termination::Escape);
    let branch = outcome.branch_key();
    assert_eq!(branch.initial_polar_side(), PolarSide::Positive);
    assert_eq!(branch.radial_turnings(), 1);
    assert_eq!(branch.equatorial_crossings(), 1);
    assert_eq!(branch.azimuth_winding(), 0);

    let components = outcome.state().components();
    let position_error = components[1..4]
        .iter()
        .zip(POSITION_XYZ_M)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt();
    assert_abs_diff_eq!(position_error, 0.0, epsilon = 2.0e-9);

    let direction = outcome
        .terminal()
        .escape_direction()
        .and_then(gravlume_reference::EscapeDirection::xyz)
        .expect("escape direction is available");
    let expected_norm = DIRECTION_XYZ
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    let expected = DIRECTION_XYZ.map(|component| component / expected_norm);
    let dot = direction
        .into_iter()
        .zip(expected)
        .map(|(actual, expected)| actual * expected)
        .sum::<f64>()
        .clamp(-1.0, 1.0);
    let cross = [
        direction[1].mul_add(expected[2], -direction[2] * expected[1]),
        direction[2].mul_add(expected[0], -direction[0] * expected[2]),
        direction[0].mul_add(expected[1], -direction[1] * expected[0]),
    ];
    let cross_norm = cross
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    assert_abs_diff_eq!(cross_norm.atan2(dot), 0.0, epsilon = 2.0e-9);
    assert_abs_diff_eq!(outcome.travel_time_m(), TRAVEL_TIME_M, epsilon = 2.0e-8);
}

fn assert_source_edge_surface(outcome: &ReferenceOutcome) {
    const SOURCE_RADIUS_M: f64 = 19.906_414_902_636_66;
    const SOURCE_AZIMUTH_RAD: f64 = 3.088_172_652_067_336_7;
    const FREQUENCY_RATIO: f64 = 0.954_336_623_855_338_7;
    const TRAVEL_TIME_M: f64 = 55.111_445_736_567_96;
    const EMITTED_INTENSITY: f64 = 0.027_382_594_561_449_43;
    const OBSERVED_INTENSITY: f64 = 0.022_713_337_755_283_02;

    assert_eq!(outcome.termination(), Termination::EquatorialSurface);
    let branch = outcome.branch_key();
    assert_eq!(branch.initial_polar_side(), PolarSide::Positive);
    assert_eq!(branch.radial_turnings(), 1);
    assert_eq!(branch.equatorial_crossings(), 0);
    assert_eq!(branch.azimuth_winding(), 0);

    let observable = outcome
        .terminal()
        .surface_observable()
        .expect("inside source-edge trace carries its surface observable");
    let anchor = observable.source_anchor();
    let radial_difference = anchor.radius_m() - SOURCE_RADIUS_M;
    let azimuth_difference = (anchor.azimuth_rad() - SOURCE_AZIMUTH_RAD + PI).rem_euclid(TAU) - PI;
    let mean_radius = anchor.radius_m().midpoint(SOURCE_RADIUS_M);
    let anchor_distance_m = radial_difference.hypot(mean_radius * azimuth_difference);
    assert_abs_diff_eq!(anchor_distance_m, 0.0, epsilon = 2.0e-9);
    assert_abs_diff_eq!(
        observable.frequency_ratio().value(),
        FREQUENCY_RATIO,
        epsilon = 2.0e-9 * FREQUENCY_RATIO
    );
    assert_abs_diff_eq!(outcome.travel_time_m(), TRAVEL_TIME_M, epsilon = 2.0e-8);
    assert_abs_diff_eq!(
        observable.emitted_bolometric_intensity(),
        EMITTED_INTENSITY,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        observable.observed_bolometric_intensity(),
        OBSERVED_INTENSITY,
        epsilon = 1.0e-12
    );
}

#[test]
fn versioned_transport_fixtures_close_under_regular_and_strict_geometry() {
    for source in TRANSPORT_FIXTURES {
        let fixture = FixtureDocument::parse_toml(source)
            .expect("repository transport fixture parses")
            .into_surface_observation()
            .expect("transport fixture is a surface observation");
        let trace = |policy| {
            ObservationTracer::baseline_v1()
                .trace(
                    fixture
                        .trace_request(policy)
                        .expect("fixture sample resolves through its observation"),
                )
                .expect("fixture transport model is valid at the localized hit")
        };
        let regular = trace(ReferencePolicy::regular_v1());
        let strict = trace(ReferencePolicy::strict_v1());
        let comparison = ReferenceComparison::baseline_v1(&regular, &strict)
            .expect("policy roles and transport input identity match");

        assert!(
            fixture.accepts(&regular),
            "rejected regular fixture {}",
            fixture.input_id()
        );
        assert!(
            fixture.accepts(&strict),
            "rejected strict fixture {}",
            fixture.input_id()
        );
        assert!(comparison.is_accepted(), "{:?}", comparison.issues());
    }
}

#[test]
fn canonical_surface_footprint_is_resolved_only_with_one_exact_branch_key() {
    let fixture = FixtureDocument::parse_toml(TRANSPORT_FIXTURES[0])
        .expect("repository transport fixture parses")
        .into_surface_observation()
        .expect("transport fixture is a surface observation");
    let estimate = ObservationTracer::baseline_v1()
        .surface_footprint_v1(
            fixture.observation(),
            fixture.sample(),
            ReferencePolicy::regular_v1(),
        )
        .expect("the centered quarter-pixel neighborhood traces successfully");
    let SurfaceFootprintEstimate::Resolved(footprint) = estimate else {
        panic!("canonical regular source neighborhood must be branch-continuous");
    };

    assert!(
        footprint
            .jacobian_source_m_per_pixel()
            .into_iter()
            .flatten()
            .all(f64::is_finite)
    );
    let [major, minor] = footprint.singular_values_m_per_pixel();
    assert!(major >= minor && minor > 0.0);
    assert_ne!(footprint.parity(), SurfaceParity::Degenerate);
    assert_eq!(
        footprint.branch_key().equatorial_crossings(),
        0,
        "the terminal surface crossing is excluded from the exact branch key",
    );
}

fn unsupported_footprint_coordinate() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.25_f64.next_down()),
        Just(0.75_f64.next_up()),
        (0_u16..=249).prop_map(|value| f64::from(value) / 1000.0),
        (751_u16..=1000).prop_map(|value| f64::from(value) / 1000.0),
    ]
}

fn unsupported_footprint_subpixel() -> impl Strategy<Value = [f64; 2]> {
    (
        any::<bool>(),
        unsupported_footprint_coordinate(),
        0_u16..=1024,
    )
        .prop_map(|(unsupported_x, outside, other)| {
            let other = f64::from(other) / 1024.0;
            if unsupported_x {
                [outside, other]
            } else {
                [other, outside]
            }
        })
}

#[test]
fn surface_footprint_rejects_every_unsupported_subpixel_neighborhood() {
    let fixture = FixtureDocument::parse_toml(TRANSPORT_FIXTURES[0])
        .expect("repository transport fixture parses")
        .into_surface_observation()
        .expect("transport fixture is a surface observation");

    proptest!(|([subpixel_x, subpixel_y] in unsupported_footprint_subpixel(),)| {
        let [pixel_x, pixel_y] = fixture.sample().pixel();
        let sample = fixture
            .observation()
            .view()
            .sample(pixel_x, pixel_y, subpixel_x, subpixel_y)
            .expect("generated subpixel belongs to the fixture view");

        prop_assert!(matches!(
            ObservationTracer::baseline_v1().surface_footprint_v1(
                fixture.observation(),
                sample,
                ReferencePolicy::regular_v1(),
            ),
            Err(SurfaceFootprintError::NeighborhoodOutsidePixel)
        ));
    });
}

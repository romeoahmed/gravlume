use std::f64::consts::{PI, TAU};

use approx::assert_abs_diff_eq;
use gravlume_reference::{
    FixtureDocument, ObservationTracer, PolarSide, ReferenceComparison, ReferencePolicy,
    SurfaceFootprintError, SurfaceFootprintEstimate, SurfaceParity, Termination,
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
        assert!(
            source_anchor_distance_m <= 2.0e-9,
            "source anchor distance {source_anchor_distance_m:e} M exceeds the witness budget"
        );
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

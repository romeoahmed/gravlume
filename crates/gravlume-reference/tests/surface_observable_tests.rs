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
        assert_independent_surface_witness(&outcome, CANONICAL_SURFACE_WITNESS);
    }
}

#[derive(Clone, Copy)]
struct IndependentEscapeWitness {
    position_xyz_m: [f64; 3],
    direction_xyz: [f64; 3],
    travel_time_m: f64,
}

#[derive(Clone, Copy)]
struct IndependentSurfaceWitness {
    source_radius_m: f64,
    source_azimuth_rad: f64,
    frequency_ratio: f64,
    travel_time_m: f64,
    emitted_intensity: f64,
    observed_intensity: f64,
}

// Generated at 120/180 decimal digits by the independent separated-chart
// witness documented in docs/research/high-precision-bl-mino-witness.md.
const CANONICAL_SURFACE_WITNESS: IndependentSurfaceWitness = IndependentSurfaceWitness {
    source_radius_m: 19.650_678_984_603_292,
    source_azimuth_rad: 3.087_156_262_423_669_3,
    frequency_ratio: 0.953_264_138_194_622_9,
    travel_time_m: 54.902_474_247_630_05,
    emitted_intensity: 0.028_465_647_567_239_85,
    observed_intensity: 0.023_505_748_696_197_128,
};

#[derive(Clone, Copy)]
enum IndependentSourceEdgeWitness {
    Escape(IndependentEscapeWitness),
    Surface(IndependentSurfaceWitness),
}

#[derive(Clone, Copy)]
struct IndependentSourceEdgeCase {
    pixel_y: u32,
    witness: IndependentSourceEdgeWitness,
}

const INDEPENDENT_SOURCE_EDGE_CORPUS: [IndependentSourceEdgeCase; 9] = [
    IndependentSourceEdgeCase {
        pixel_y: 12,
        witness: IndependentSourceEdgeWitness::Escape(IndependentEscapeWitness {
            position_xyz_m: [
                -170.743_537_420_571_3,
                1.337_744_245_785_816_9,
                -104.140_872_592_375_22,
            ],
            direction_xyz: [
                -0.822_251_516_308_215_4,
                0.005_874_201_091_296_951,
                -0.569_093_962_092_710_7,
            ],
            travel_time_m: 238.406_047_718_117,
        }),
    },
    IndependentSourceEdgeCase {
        pixel_y: 13,
        witness: IndependentSourceEdgeWitness::Escape(IndependentEscapeWitness {
            position_xyz_m: [
                -170.447_402_756_461_1,
                1.369_244_882_783_220_7,
                -104.624_437_497_465_04,
            ],
            direction_xyz: [
                -0.820_715_680_321_071_6,
                0.006_023_904_198_189_695,
                -0.571_305_071_440_234_7,
            ],
            travel_time_m: 238.438_694_378_676_36,
        }),
    },
    IndependentSourceEdgeCase {
        pixel_y: 14,
        witness: IndependentSourceEdgeWitness::Surface(IndependentSurfaceWitness {
            source_radius_m: 19.906_414_902_636_66,
            source_azimuth_rad: 3.088_172_652_067_336_7,
            frequency_ratio: 0.954_336_623_855_338_8,
            travel_time_m: 55.111_445_736_567_96,
            emitted_intensity: 0.027_382_594_561_449_43,
            observed_intensity: 0.022_713_337_755_283_02,
        }),
    },
    IndependentSourceEdgeCase {
        pixel_y: 15,
        witness: IndependentSourceEdgeWitness::Surface(IndependentSurfaceWitness {
            source_radius_m: 19.778_228_798_382_42,
            source_azimuth_rad: 3.087_666_778_945_728,
            frequency_ratio: 0.953_802_558_846_997_9,
            travel_time_m: 55.006_599_298_066_206,
            emitted_intensity: 0.027_918_466_604_424_055,
            observed_intensity: 0.023_106_038_586_166_094,
        }),
    },
    IndependentSourceEdgeCase {
        pixel_y: 16,
        witness: IndependentSourceEdgeWitness::Surface(CANONICAL_SURFACE_WITNESS),
    },
    IndependentSourceEdgeCase {
        pixel_y: 17,
        witness: IndependentSourceEdgeWitness::Surface(IndependentSurfaceWitness {
            source_radius_m: 19.523_761_223_635_148,
            source_azimuth_rad: 3.086_641_041_843_746_4,
            frequency_ratio: 0.952_721_311_199_448_8,
            travel_time_m: 54.799_067_017_429_486,
            emitted_intensity: 0.029_024_402_523_646_662,
            observed_intensity: 0.023_912_600_497_345_077,
        }),
    },
    IndependentSourceEdgeCase {
        pixel_y: 18,
        witness: IndependentSourceEdgeWitness::Surface(IndependentSurfaceWitness {
            source_radius_m: 19.397_471_319_572_137,
            source_azimuth_rad: 3.086_121_055_522_226_7,
            frequency_ratio: 0.952_174_026_402_790_6,
            travel_time_m: 54.696_374_090_040_83,
            emitted_intensity: 0.029_595_003_517_611_927,
            observed_intensity: 0.024_326_729_051_185_183,
        }),
    },
    IndependentSourceEdgeCase {
        pixel_y: 19,
        witness: IndependentSourceEdgeWitness::Surface(IndependentSurfaceWitness {
            source_radius_m: 19.271_805_117_804_51,
            source_azimuth_rad: 3.085_596_240_727_706,
            frequency_ratio: 0.951_622_231_572_182_2,
            travel_time_m: 54.594_391_998_135_35,
            emitted_intensity: 0.030_177_729_768_427_544,
            observed_intensity: 0.024_748_272_123_592_492,
        }),
    },
    IndependentSourceEdgeCase {
        pixel_y: 20,
        witness: IndependentSourceEdgeWitness::Surface(IndependentSurfaceWitness {
            source_radius_m: 19.146_758_504_563_227,
            source_azimuth_rad: 3.085_066_533_659_228_7,
            frequency_ratio: 0.951_065_873_687_221_2,
            travel_time_m: 54.493_117_324_177_98,
            emitted_intensity: 0.030_772_867_882_523_796,
            observed_intensity: 0.025_177_370_240_523_023,
        }),
    },
];

#[test]
fn source_edge_corpus_matches_the_independent_bl_mino_witness() {
    // Generated at 120/180 decimal digits by the independent separated-chart
    // witness documented in docs/research/high-precision-bl-mino-witness.md.
    let fixture = FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let observation = fixture.observation();
    let oracle = ObservationTracer::baseline_v1();

    for case in INDEPENDENT_SOURCE_EDGE_CORPUS {
        let image_sample = observation
            .view()
            .sample(640, case.pixel_y, 0.5, 0.5)
            .expect("certified source-edge sample belongs to the canonical view");
        for policy in [ReferencePolicy::regular_v1(), ReferencePolicy::strict_v1()] {
            let outcome = oracle
                .trace(
                    ObservationTrace::new(
                        TraceInputId::new(format!("source-edge-{}-{}", case.pixel_y, policy.id())),
                        observation,
                        image_sample,
                        policy,
                    )
                    .expect("source-edge trace request resolves"),
                )
                .expect("source-edge trace succeeds");
            match case.witness {
                IndependentSourceEdgeWitness::Escape(witness) => {
                    assert_independent_escape_witness(&outcome, witness);
                }
                IndependentSourceEdgeWitness::Surface(witness) => {
                    assert_independent_surface_witness(&outcome, witness);
                }
            }
        }
    }
}

fn assert_independent_escape_witness(
    outcome: &ReferenceOutcome,
    witness: IndependentEscapeWitness,
) {
    assert_eq!(outcome.termination(), Termination::Escape);
    let branch = outcome.branch_key();
    assert_eq!(branch.initial_polar_side(), PolarSide::Positive);
    assert_eq!(branch.radial_turnings(), 1);
    assert_eq!(branch.equatorial_crossings(), 1);
    assert_eq!(branch.azimuth_winding(), 0);

    let components = outcome.state().components();
    let position_error = components[1..4]
        .iter()
        .zip(witness.position_xyz_m)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt();
    assert_abs_diff_eq!(position_error, 0.0, epsilon = 2.0e-9);

    let direction = outcome
        .terminal()
        .escape_direction()
        .and_then(gravlume_reference::EscapeDirection::xyz)
        .expect("escape direction is available");
    let expected_norm = witness
        .direction_xyz
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    let expected = witness
        .direction_xyz
        .map(|component| component / expected_norm);
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
    assert_abs_diff_eq!(
        outcome.travel_time_m(),
        witness.travel_time_m,
        epsilon = 2.0e-8
    );
}

fn assert_independent_surface_witness(
    outcome: &ReferenceOutcome,
    witness: IndependentSurfaceWitness,
) {
    assert_eq!(outcome.termination(), Termination::EquatorialSurface);
    let branch = outcome.branch_key();
    assert_eq!(branch.initial_polar_side(), PolarSide::Positive);
    assert_eq!(branch.radial_turnings(), 1);
    assert_eq!(branch.equatorial_crossings(), 0);
    assert_eq!(branch.azimuth_winding(), 0);

    let observable = outcome
        .terminal()
        .surface_observable()
        .expect("independently certified trace carries its surface observable");
    let anchor = observable.source_anchor();
    let radial_difference = anchor.radius_m() - witness.source_radius_m;
    let azimuth_difference =
        (anchor.azimuth_rad() - witness.source_azimuth_rad + PI).rem_euclid(TAU) - PI;
    let mean_radius = anchor.radius_m().midpoint(witness.source_radius_m);
    let anchor_distance_m = radial_difference.hypot(mean_radius * azimuth_difference);
    assert_abs_diff_eq!(anchor_distance_m, 0.0, epsilon = 2.0e-9);
    assert_abs_diff_eq!(
        observable.frequency_ratio().value(),
        witness.frequency_ratio,
        epsilon = 2.0e-9 * witness.frequency_ratio
    );
    assert_abs_diff_eq!(
        outcome.travel_time_m(),
        witness.travel_time_m,
        epsilon = 2.0e-8
    );
    assert_abs_diff_eq!(
        observable.emitted_bolometric_intensity(),
        witness.emitted_intensity,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        observable.observed_bolometric_intensity(),
        witness.observed_intensity,
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

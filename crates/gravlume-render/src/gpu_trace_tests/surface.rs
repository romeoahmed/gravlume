use approx::assert_abs_diff_eq;
use gravlume_domain::{
    EquatorialCircularEmitter, EquatorialSurface, HomogeneousScalarSlab, Observation,
    SurfaceTransport,
};
use gravlume_reference::{
    FixtureDocument, ObservationTrace, ObservationTracer, PolarSide, ReferencePolicy,
    SurfaceFootprintEstimate, SurfaceParity, Termination, TraceInputId,
};

use super::{decode_f16, default_observation, sample, transfer_profile_extent};
use crate::{
    GpuTraceInputError,
    gpu_capture::{
        capture_surface_footprint_sample, capture_surface_transport_case, capture_trace,
        capture_trace_sample,
    },
    trace::{TraceTermination, TraceUniforms},
};

const BLACKBODY_TRANSPORT_FIXTURES: [&str; 4] = [
    include_str!("../../../gravlume-reference/fixtures/v3/kerr-blackbody-vacuum.toml"),
    include_str!("../../../gravlume-reference/fixtures/v3/kerr-blackbody-pure-absorption.toml"),
    include_str!("../../../gravlume-reference/fixtures/v3/kerr-blackbody-constant-slab.toml"),
    include_str!("../../../gravlume-reference/fixtures/v3/kerr-blackbody-pure-emission.toml"),
];

#[test]
fn versioned_blackbody_fixtures_close_gpu_spectral_transport() {
    for source in BLACKBODY_TRANSPORT_FIXTURES {
        let fixture = FixtureDocument::parse_toml(source)
            .expect("repository transport fixture parses")
            .into_surface_observation()
            .expect("fixture is a surface observation");
        let observation = fixture.observation();
        let fixture_sample = fixture.sample();
        let capture = capture_trace_sample(observation, fixture_sample);
        let [pixel_x, pixel_y] = fixture_sample.pixel();
        let index = usize::try_from(pixel_y * observation.view().width().get() + pixel_x)
            .expect("fixture pixel index fits usize");
        let reference = ObservationTracer::baseline_v1()
            .trace(
                fixture
                    .trace_request(ReferencePolicy::regular_v1())
                    .expect("fixture sample resolves"),
            )
            .expect("reference spectral transport succeeds");
        let expected = reference
            .terminal()
            .surface_observable()
            .and_then(gravlume_reference::SurfaceObservable::observed_spectral_band_intensities)
            .expect("reference resolves the three instrument bands");
        let pixel = capture.hdr_pixel(index);

        assert_eq!(
            u16::from_le_bytes(pixel[6..].try_into().expect("alpha has two bytes")),
            0x4000,
            "spectral surface radiance alpha tag"
        );
        for (channel, expected) in pixel[..6].chunks_exact(2).zip(expected) {
            let actual = decode_f16(u16::from_le_bytes(
                channel.try_into().expect("half channel has two bytes"),
            ));
            assert_abs_diff_eq!(f64::from(actual), expected, epsilon = 4.0e-3 * expected);
        }
    }
}

#[test]
fn gpu_surface_footprint_matches_the_branch_checked_reference_jacobian() {
    let fixture = FixtureDocument::parse_toml(BLACKBODY_TRANSPORT_FIXTURES[0])
        .expect("repository transport fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let observation = fixture.observation();
    let fixture_sample = fixture.sample();
    let SurfaceFootprintEstimate::Resolved(reference) = ObservationTracer::baseline_v1()
        .surface_footprint_v1(observation, fixture_sample, ReferencePolicy::regular_v1())
        .expect("reference footprint traces")
    else {
        panic!("canonical footprint must be branch-continuous");
    };
    let capture = capture_surface_footprint_sample(observation, fixture_sample);
    let [pixel_x, pixel_y] = fixture_sample.pixel();
    let index = usize::try_from(pixel_y * observation.view().width().get() + pixel_x)
        .expect("fixture pixel index fits usize");
    let gpu = capture.records[index];

    assert_eq!(gpu.metadata[2], 1, "GPU footprint continuity flag");
    let expected = reference.jacobian_source_m_per_pixel();
    let actual = [
        [gpu.invariant_drift[0], gpu.invariant_drift[1]],
        [gpu.invariant_drift[2], gpu.invariant_drift[3]],
    ];
    let (maximum_error, maximum_reference_component) = actual
        .into_iter()
        .flatten()
        .zip(expected.into_iter().flatten())
        .fold(
            (0.0_f64, 0.0_f64),
            |(maximum_error, scale), (actual, expected)| {
                (
                    maximum_error.max((f64::from(actual) - expected).abs()),
                    scale.max(expected.abs()),
                )
            },
        );
    assert_abs_diff_eq!(
        maximum_error,
        0.0,
        epsilon = 3.0e-3 * maximum_reference_component
    );
    let expected_parity = match reference.parity() {
        SurfaceParity::Positive => 1,
        SurfaceParity::Negative => 2,
        SurfaceParity::Degenerate => 3,
    };
    assert_eq!(gpu.metadata[3], expected_parity);
    let branch = reference.branch_key();
    assert_eq!(gpu.event[0], branch.radial_turnings());
    assert_eq!(gpu.event[1], branch.equatorial_crossings());
    assert_eq!(
        i32::from_ne_bytes(gpu.event[2].to_ne_bytes()),
        branch.azimuth_winding()
    );
    let expected_initial_side = match branch.initial_polar_side() {
        PolarSide::Negative => 0,
        PolarSide::Equatorial => 1,
        PolarSide::Positive => 2,
    };
    assert_eq!(gpu.event[3], expected_initial_side);
}

const SURFACE_BRANCH_PROFILES: [SurfaceBranchProfile; 4] = [
    SurfaceBranchProfile {
        label: "schwarzschild",
        spin: 0.0,
        charge: 0.0,
        observer_radius: 30.0,
        vertical_fov: std::f64::consts::FRAC_PI_4,
        samples: [
            (23, 10),
            (16, 12),
            (9, 14),
            (58, 15),
            (20, 17),
            (1, 19),
            (43, 20),
            (61, 21),
            (12, 23),
            (16, 24),
            (20, 25),
            (25, 26),
        ],
    },
    SurfaceBranchProfile {
        label: "positive-spin-wide",
        spin: 0.8,
        charge: 0.0,
        observer_radius: 30.0,
        vertical_fov: std::f64::consts::FRAC_PI_2,
        samples: [
            (38, 15),
            (41, 16),
            (44, 17),
            (43, 18),
            (38, 19),
            (27, 20),
            (35, 21),
            (23, 22),
            (30, 23),
            (18, 24),
            (26, 25),
            (35, 26),
        ],
    },
    SurfaceBranchProfile {
        label: "negative-spin-near",
        spin: -0.8,
        charge: 0.0,
        observer_radius: 12.0,
        vertical_fov: std::f64::consts::FRAC_PI_4,
        samples: [
            (54, 6),
            (62, 7),
            (62, 9),
            (0, 12),
            (62, 14),
            (6, 18),
            (63, 21),
            (46, 24),
            (45, 26),
            (3, 28),
            (44, 29),
            (55, 30),
        ],
    },
    SurfaceBranchProfile {
        label: "kerr-newman",
        spin: 0.6,
        charge: 0.5,
        observer_radius: 30.0,
        vertical_fov: std::f64::consts::FRAC_PI_4,
        samples: [
            (23, 10),
            (18, 12),
            (10, 14),
            (58, 15),
            (20, 17),
            (1, 19),
            (41, 20),
            (60, 21),
            (10, 23),
            (14, 24),
            (19, 25),
            (23, 26),
        ],
    },
];

#[derive(Clone, Copy)]
struct SurfaceBranchProfile {
    label: &'static str,
    spin: f64,
    charge: f64,
    observer_radius: f64,
    vertical_fov: f64,
    samples: [(u32, u32); 12],
}

#[test]
fn gpu_surface_branches_match_fixed_reference_samples() {
    let oracle = ObservationTracer::baseline_v1();

    for profile in SURFACE_BRANCH_PROFILES {
        assert_surface_branch_profile(profile, oracle);
    }
}

fn assert_surface_branch_profile(profile: SurfaceBranchProfile, oracle: ObservationTracer) {
    // Keep inputs independent of GPU classification so a false-negative surface terminal cannot
    // remove itself from the comparison set.
    let base = transfer_profile_extent(
        profile.spin,
        profile.charge,
        profile.observer_radius,
        profile.vertical_fov,
        64,
        36,
    );
    let emitter = EquatorialCircularEmitter::inverse_cube_bolometric_v1(6.0, 20.0, 1.0)
        .expect("matrix surface is valid");
    let observation = Observation::new(
        base.scene().clone().with_equatorial_surface(
            EquatorialSurface::new(emitter, SurfaceTransport::Vacuum)
                .expect("matrix surface is compatible with vacuum"),
        ),
        *base.view(),
    );
    let capture = capture_trace(&observation);

    for (pixel_x, pixel_y) in profile.samples {
        let width = observation.view().width().get();
        let index = usize::try_from(pixel_y * width + pixel_x).expect("matrix index fits usize");
        let reference = oracle
            .trace(
                ObservationTrace::new(
                    TraceInputId::new(format!(
                        "surface-branch-{}-{pixel_x}-{pixel_y}",
                        profile.label
                    )),
                    &observation,
                    sample(&observation, pixel_x, pixel_y, 0.5, 0.5),
                    ReferencePolicy::regular_v1(),
                )
                .expect("matrix trace request resolves"),
            )
            .expect("matrix reference trace succeeds");
        assert_eq!(
            reference.termination(),
            Termination::EquatorialSurface,
            "{}: reference terminal at ({pixel_x}, {pixel_y})",
            profile.label
        );

        let gpu = capture.records[index];
        assert_eq!(
            TraceTermination::try_from(gpu.metadata[0]),
            Ok(TraceTermination::EquatorialSurface),
            "{}: GPU terminal at ({pixel_x}, {pixel_y})",
            profile.label
        );
        let branch = reference.branch_key();
        assert_eq!(
            gpu.event[2] & 0xffff,
            branch.radial_turnings(),
            "{}: radial branch at ({pixel_x}, {pixel_y})",
            profile.label
        );
        assert_eq!(
            gpu.event[2] >> 16,
            branch.equatorial_crossings(),
            "{}: equatorial branch at ({pixel_x}, {pixel_y})",
            profile.label
        );
        assert_eq!(
            i32::from_ne_bytes(gpu.event[3].to_ne_bytes()),
            branch.azimuth_winding(),
            "{}: winding branch at ({pixel_x}, {pixel_y})",
            profile.label
        );
        assert_eq!(
            branch.initial_polar_side(),
            PolarSide::Positive,
            "{}: observer is above the source plane",
            profile.label
        );
    }
}

#[test]
fn gpu_trace_rejects_transmittance_that_binary32_cannot_preserve_normally() {
    let base = default_observation(1, 1);
    let emitter = EquatorialCircularEmitter::inverse_cube_bolometric_v1(6.0, 20.0, 1.0e38)
        .expect("the high dynamic-range surface is intrinsically valid");

    for optical_depth in [92.0, 1_000.0] {
        let slab = HomogeneousScalarSlab::pure_absorption_v1(optical_depth)
            .expect("the high optical-depth slab is intrinsically valid");
        let observation = Observation::new(
            base.scene().clone().with_equatorial_surface(
                EquatorialSurface::new(emitter, SurfaceTransport::HomogeneousScalar(slab))
                    .expect("bolometric surface and slab are compatible"),
            ),
            *base.view(),
        );

        assert!(matches!(
            TraceUniforms::from_observation(&observation),
            Err(GpuTraceInputError::NotRepresentable {
                field: "surface_transport"
            })
        ));
    }
}

#[test]
fn high_absorption_keeps_a_representable_outgoing_surface_intensity() {
    let base = default_observation(1, 1);
    let emitted_intensity = f64::from(f32::MAX);
    let emitter =
        EquatorialCircularEmitter::inverse_cube_bolometric_v1(6.0, 20.0, emitted_intensity)
            .expect("the high dynamic-range surface is intrinsically valid");
    let slab = HomogeneousScalarSlab::pure_absorption_v1(79.0)
        .expect("the high optical-depth slab is intrinsically valid");
    let observation = Observation::new(
        base.scene().clone().with_equatorial_surface(
            EquatorialSurface::new(emitter, SurfaceTransport::HomogeneousScalar(slab))
                .expect("bolometric surface and slab are compatible"),
        ),
        *base.view(),
    );

    let capture = capture_surface_transport_case(&observation);
    let pixel = capture.hdr_pixel(0);
    assert_eq!(
        u16::from_le_bytes(pixel[6..].try_into().expect("alpha has two bytes")),
        0x4000,
        "representable surface radiance keeps its alpha tag",
    );
    let actual = decode_f16(u16::from_le_bytes(
        pixel[..2].try_into().expect("red channel has two bytes"),
    ));
    let expected = emitted_intensity * (-79.0_f64).exp() * 1.1_f64.powi(4);
    assert_abs_diff_eq!(f64::from(actual), expected, epsilon = 2.0e-3 * expected);
}

#[test]
fn low_temperature_diluted_spectrum_preserves_gpu_radiance() {
    let base = default_observation(1, 1);
    let emitter = EquatorialCircularEmitter::inverse_cube_blackbody_v1(6.0, 20.0, 1.0e38, 200.0)
        .expect("the diluted blackbody source is intrinsically valid");
    let observation = Observation::new(
        base.scene().clone().with_equatorial_surface(
            EquatorialSurface::new(emitter, SurfaceTransport::Vacuum)
                .expect("blackbody surface is compatible with vacuum"),
        ),
        *base.view(),
    );

    let capture = capture_surface_transport_case(&observation);
    let pixel = capture.hdr_pixel(0);
    assert_eq!(
        u16::from_le_bytes(pixel[6..].try_into().expect("alpha has two bytes")),
        0x4000,
        "representable spectral radiance keeps its alpha tag",
    );
    let expected = [505.403_345_434_498_3, 1.380_353_634_125_891_2e-4];
    for (channel, expected) in pixel[..4].chunks_exact(2).zip(expected) {
        let actual = decode_f16(u16::from_le_bytes(
            channel.try_into().expect("half channel has two bytes"),
        ));
        assert_abs_diff_eq!(f64::from(actual), expected, epsilon = 4.0e-3 * expected);
    }
}

#[test]
fn gpu_trace_rejects_a_blackbody_profile_outside_the_spectral_lut() {
    let base = default_observation(1, 1);
    let emitter = EquatorialCircularEmitter::inverse_cube_blackbody_v1(1.0e-12, 20.0, 1.0, 6_000.0)
        .expect("the intrinsic source profile is valid independently of the GPU LUT");
    let observation = Observation::new(
        base.scene().clone().with_equatorial_surface(
            EquatorialSurface::new(emitter, SurfaceTransport::Vacuum)
                .expect("blackbody surface is compatible with vacuum"),
        ),
        *base.view(),
    );

    assert!(matches!(
        TraceUniforms::from_observation(&observation),
        Err(GpuTraceInputError::TemperatureOutsideSpectralLut {
            field: "equatorial_circular_emitter.inner_temperature_kelvin",
            ..
        })
    ));
}

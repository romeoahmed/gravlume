use std::num::NonZeroU32;

use gravlume_domain::{
    Angle, EquatorialCircularEmitter, HomogeneousScalarSlab, ImageSample, KerrNewmanSpacetime,
    KerrSchildChart, Observation, PerspectiveView, PhysicalScene, PhysicalSceneInput,
    StationaryObserverInput,
};
use gravlume_reference::{
    ObservationTrace, ObservationTracer, PolarSide, ReferencePolicy, SurfaceFootprintEstimate,
    SurfaceParity, Termination, TraceInputId,
};
use num_traits::ToPrimitive as _;
use proptest::prelude::*;

use crate::{
    gpu_capture::{
        capture_accelerated_trace, capture_accelerated_trace_in_batches,
        capture_event_policy_cases, capture_initial_rays, capture_invariant_gate_cases,
        capture_refined_edge_count, capture_refined_trace, capture_surface_footprint_sample,
        capture_trace, capture_trace_sample,
    },
    ray_tracer::{INVARIANT_DRIFT_LIMIT, TraceTermination, TraceUniforms, UnknownTraceTermination},
};

const EVENT_CANDIDATE_HORIZON: u32 = 1 << 1;
const EVENT_CANDIDATE_SURFACE: u32 = 1 << 2;
const EVENT_CANDIDATE_ESCAPE: u32 = 1 << 3;

#[test]
fn trace_termination_discriminants_are_stable_and_checked() {
    let cases = [
        (1, TraceTermination::HorizonCrossing),
        (2, TraceTermination::Escape),
        (3, TraceTermination::SingularityGuard),
        (4, TraceTermination::StepExhaustion),
        (5, TraceTermination::NumericalFailure),
        (6, TraceTermination::Uncertain),
        (7, TraceTermination::EquatorialSurface),
    ];

    for (raw, expected) in cases {
        assert_eq!(u32::from(expected), raw);
        assert_eq!(TraceTermination::try_from(raw), Ok(expected));
    }

    for raw in [0, 8, u32::MAX] {
        assert_eq!(
            TraceTermination::try_from(raw),
            Err(UnknownTraceTermination(raw)),
        );
    }
}

#[test]
fn canonical_surface_sample_closes_gpu_geometry_frequency_and_radiance() {
    const SURFACE_OBSERVABLE: &str =
        include_str!("../../gravlume-reference/fixtures/v2/kerr-surface-observable.toml");
    let fixture = gravlume_reference::FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let observation = fixture.observation();
    let sample = fixture.sample();
    let capture = capture_trace_sample(observation, sample);
    let [pixel_x, pixel_y] = sample.pixel();
    let index = usize::try_from(pixel_y * observation.view().width().get() + pixel_x)
        .expect("fixture pixel index fits usize");
    let gpu = capture.records[index];
    let reference = ObservationTracer::baseline_v1()
        .trace(
            fixture
                .trace_request(ReferencePolicy::regular_v1())
                .expect("fixture sample resolves through its observation"),
        )
        .expect("canonical surface source is valid");
    let reference_observable = reference
        .surface_observable()
        .expect("canonical trace terminates on the surface");
    let reference_anchor = reference_observable.source_anchor();

    assert_eq!(
        TraceTermination::try_from(gpu.metadata[0]),
        Ok(TraceTermination::EquatorialSurface)
    );
    assert_eq!(gpu.metadata[1], 0, "surface trace failure flags");
    assert_eq!(gpu.event[..2], [EVENT_CANDIDATE_SURFACE, 0]);
    let reference_branch = reference.branch_key();
    assert_eq!(gpu.event[2] & 0xffff, reference_branch.radial_turnings());
    assert_eq!(gpu.event[2] >> 16, reference_branch.equatorial_crossings());
    assert_eq!(
        i32::from_ne_bytes(gpu.event[3].to_ne_bytes()),
        reference_branch.azimuth_winding()
    );
    assert!((f64::from(gpu.source_time[0]) - reference_anchor.radius_m()).abs() <= 5.0e-3);
    let azimuth_error = (f64::from(gpu.source_time[1]) - reference_anchor.azimuth_rad())
        .sin()
        .atan2((f64::from(gpu.source_time[1]) - reference_anchor.azimuth_rad()).cos())
        .abs();
    assert!(
        azimuth_error <= 3.0e-4,
        "source azimuth error {azimuth_error:e}"
    );
    let reference_ratio = reference_observable.frequency_ratio().value();
    let ratio_relative_error =
        (f64::from(gpu.source_time[2]) - reference_ratio).abs() / reference_ratio;
    assert!(
        ratio_relative_error <= 2.0e-3,
        "frequency-ratio error {ratio_relative_error:e}"
    );
    assert!((f64::from(gpu.source_time[3]) - reference.travel_time_m()).abs() <= 1.0e-3);

    let expected = reference_observable.observed_bolometric_intensity();
    let pixel = capture.hdr_pixel(index);
    assert_eq!(
        u16::from_le_bytes(pixel[6..].try_into().expect("alpha has two bytes")),
        0x4000,
        "surface radiance alpha tag"
    );
    for channel in pixel[..6].chunks_exact(2) {
        let actual = decode_f16(u16::from_le_bytes(
            channel.try_into().expect("half channel has two bytes"),
        ));
        assert!(
            (f64::from(actual) - expected).abs() / expected <= 2.0e-3,
            "surface radiance {actual:e}, expected {expected:e}"
        );
    }
}

#[test]
fn versioned_blackbody_transport_fixtures_close_gpu_spectral_transport() {
    const SURFACE_TRANSPORT: [&str; 4] = [
        include_str!("../../gravlume-reference/fixtures/v3/kerr-blackbody-vacuum.toml"),
        include_str!("../../gravlume-reference/fixtures/v3/kerr-blackbody-pure-absorption.toml"),
        include_str!("../../gravlume-reference/fixtures/v3/kerr-blackbody-constant-slab.toml"),
        include_str!("../../gravlume-reference/fixtures/v3/kerr-blackbody-pure-emission.toml"),
    ];
    for source in SURFACE_TRANSPORT {
        let fixture = gravlume_reference::FixtureDocument::parse_toml(source)
            .expect("repository transport fixture parses")
            .into_surface_observation()
            .expect("fixture is a surface observation");
        let observation = fixture.observation();
        let sample = fixture.sample();
        let capture = capture_trace_sample(observation, sample);
        let [pixel_x, pixel_y] = sample.pixel();
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
            assert!(
                (f64::from(actual) - expected).abs() / expected <= 4.0e-3,
                "spectral surface radiance {actual:e}, expected {expected:e}"
            );
        }
    }
}

#[test]
fn gpu_surface_footprint_matches_the_branch_checked_reference_jacobian() {
    const SURFACE_TRANSPORT: &str =
        include_str!("../../gravlume-reference/fixtures/v3/kerr-blackbody-vacuum.toml");
    let fixture = gravlume_reference::FixtureDocument::parse_toml(SURFACE_TRANSPORT)
        .expect("repository transport fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let observation = fixture.observation();
    let sample = fixture.sample();
    let SurfaceFootprintEstimate::Resolved(reference) = ObservationTracer::baseline_v1()
        .surface_footprint_v1(observation, sample, ReferencePolicy::regular_v1())
        .expect("reference footprint traces")
    else {
        panic!("canonical footprint must be branch-continuous");
    };
    let capture = capture_surface_footprint_sample(observation, sample);
    let [pixel_x, pixel_y] = sample.pixel();
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
    assert!(
        maximum_error / maximum_reference_component <= 3.0e-3,
        "GPU footprint max-norm error {maximum_error:e} against scale \
         {maximum_reference_component:e}"
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

proptest! {
    #[test]
    fn non_boundary_unknown_trace_termination_discriminants_are_rejected(
        raw in 8_u32..u32::MAX,
    ) {
        prop_assert_eq!(
            TraceTermination::try_from(raw),
            Err(UnknownTraceTermination(raw)),
        );
    }
}

#[test]
fn gpu_trace_rejects_non_normalized_observer_frequency() {
    let observation = observation_with(1.0, 0.8, 0.0, [30.0, 0.0, 0.0], 1.0e-6, 1, 1);

    assert!(matches!(
        TraceUniforms::from_observation(&observation),
        Err(crate::GpuTraceInputError::NonNormalizedObserverFrequency { .. })
    ));
}

#[test]
fn gpu_trace_rejects_extremality_changed_by_f32_packing() {
    let observation = observation_with(
        1.0,
        0.157_132_806_437_842_44,
        0.987_577_480_983_596_3,
        [30.0, 0.0, 0.0],
        1.0,
        1,
        1,
    );

    assert!(matches!(
        TraceUniforms::from_observation(&observation),
        Err(crate::GpuTraceInputError::ExtremalityChangedByPacking { .. })
    ));
}

#[test]
fn gpu_trace_rejects_surface_profiles_not_representable_after_f32_packing() {
    let base = default_observation(1, 1);
    let collapsed_interval = EquatorialCircularEmitter::inverse_cube_bolometric_v1(
        6.0,
        f64::from_bits(6.0_f64.to_bits() + 1),
        1.0,
    )
    .expect("binary64 interval is nonempty");
    let underflowed_intensity =
        EquatorialCircularEmitter::inverse_cube_bolometric_v1(6.0, 20.0, f64::MIN_POSITIVE)
            .expect("binary64 intensity is positive");

    for emitter in [collapsed_interval, underflowed_intensity] {
        let observation = Observation::new(
            base.scene()
                .clone()
                .with_equatorial_circular_emitter(emitter),
            *base.view(),
        );
        assert!(matches!(
            TraceUniforms::from_observation(&observation),
            Err(crate::GpuTraceInputError::NotRepresentable {
                field: "surface_emitter"
            })
        ));
    }
}

#[test]
fn gpu_trace_rejects_surface_profiles_reaching_packed_escape_boundary() {
    let base = default_observation(1, 1);

    for outer_radius_m in [199.999_999, 200.0, 250.0] {
        let emitter =
            EquatorialCircularEmitter::inverse_cube_bolometric_v1(6.0, outer_radius_m, 1.0)
                .expect("physical emitter is valid independently of the GPU profile");
        let observation = Observation::new(
            base.scene()
                .clone()
                .with_equatorial_circular_emitter(emitter),
            *base.view(),
        );

        assert!(
            matches!(
                TraceUniforms::from_observation(&observation),
                Err(crate::GpuTraceInputError::SurfaceOutsideEscapeBoundary { .. })
            ),
            "surface ending at {outer_radius_m} M escaped the GPU profile applicability check"
        );
    }
}

#[test]
fn gpu_trace_rejects_transport_models_without_a_resolvable_spectral_contract() {
    let base = default_observation(1, 1);
    let slab = HomogeneousScalarSlab::constant_bolometric_v1(0.5, 0.1).expect("test slab is valid");
    let slab_without_surface = Observation::new(
        base.scene().clone().with_homogeneous_scalar_slab(slab),
        *base.view(),
    );
    assert!(matches!(
        TraceUniforms::from_observation(&slab_without_surface),
        Err(crate::GpuTraceInputError::ScalarSlabRequiresSurface)
    ));

    let blackbody = EquatorialCircularEmitter::inverse_cube_blackbody_v1(6.0, 20.0, 1.0, 6_000.0)
        .expect("test blackbody is valid");
    let unresolved_spectrum = Observation::new(
        base.scene()
            .clone()
            .with_equatorial_circular_emitter(blackbody)
            .with_homogeneous_scalar_slab(slab),
        *base.view(),
    );
    assert!(matches!(
        TraceUniforms::from_observation(&unresolved_spectrum),
        Err(crate::GpuTraceInputError::UnresolvedSlabSourceSpectrum)
    ));
}

#[test]
fn gpu_trace_rejects_a_blackbody_profile_that_leaves_the_spectral_lut() {
    let base = default_observation(1, 1);
    let emitter = EquatorialCircularEmitter::inverse_cube_blackbody_v1(1.0e-12, 20.0, 1.0, 6_000.0)
        .expect("the intrinsic source profile is valid independently of the GPU LUT");
    let observation = Observation::new(
        base.scene()
            .clone()
            .with_equatorial_circular_emitter(emitter),
        *base.view(),
    );

    assert!(matches!(
        TraceUniforms::from_observation(&observation),
        Err(crate::GpuTraceInputError::TemperatureOutsideSpectralLut {
            field: "equatorial_circular_emitter.inner_temperature_kelvin",
            ..
        })
    ));
}

#[test]
fn gpu_event_ties_use_affine_distance_and_stable_protocol_order() {
    let capture = capture_event_policy_cases(&default_observation(8, 1));
    let cases = [
        (
            TraceTermination::Escape,
            EVENT_CANDIDATE_SURFACE | EVENT_CANDIDATE_ESCAPE,
        ),
        (
            TraceTermination::Escape,
            EVENT_CANDIDATE_SURFACE | EVENT_CANDIDATE_ESCAPE,
        ),
        (
            TraceTermination::EquatorialSurface,
            EVENT_CANDIDATE_SURFACE | EVENT_CANDIDATE_ESCAPE,
        ),
        (
            TraceTermination::EquatorialSurface,
            EVENT_CANDIDATE_HORIZON | EVENT_CANDIDATE_SURFACE,
        ),
    ];

    for (case_index, (record, (termination, candidates))) in
        capture.records[..cases.len()].iter().zip(cases).enumerate()
    {
        assert_eq!(
            record.metadata[0],
            u32::from(termination),
            "event-policy case {case_index}"
        );
        assert_eq!(
            record.event[..2],
            [candidates, 1],
            "event-policy case {case_index}"
        );
    }
}

#[test]
fn gpu_surface_event_arming_requires_leaving_the_profile_band() {
    let base = default_observation(8, 1);
    let emitter = EquatorialCircularEmitter::inverse_cube_bolometric_v1(6.0, 20.0, 1.0)
        .expect("surface profile is valid");
    let observation = Observation::new(
        base.scene()
            .clone()
            .with_equatorial_circular_emitter(emitter),
        *base.view(),
    );
    let capture = capture_event_policy_cases(&observation);
    let armed = capture.records[4..]
        .iter()
        .map(|record| record.metadata[0])
        .collect::<Vec<_>>();

    assert_eq!(
        armed,
        [0, 0, 1, 1],
        "surface, in-band, outside-band, and sticky-armed cases"
    );
}

#[test]
fn mass_scale_does_not_change_the_dimensionless_trace_result() {
    let unit = capture_trace(&observation_at_scale(7, 5, 1.0));
    let scaled = capture_trace(&observation_at_scale(7, 5, 8.0));

    assert_eq!(unit.records.len(), scaled.records.len());
    for (unit, scaled) in unit.records.iter().zip(&scaled.records) {
        assert_same_bits(unit.source_time, scaled.source_time);
        assert_same_bits(unit.invariant_drift, scaled.invariant_drift);
        assert_eq!(unit.metadata, scaled.metadata);
        assert_eq!(unit.event, scaled.event);
    }
}

#[test]
fn coordinate_time_origin_does_not_change_gpu_trace_observables() {
    let origin = capture_trace(&observation_at_coordinate_time(7, 5, 0.0));
    let translated = capture_trace(&observation_at_coordinate_time(7, 5, 1.0e8));
    let origin = origin.records[0];
    let translated = translated.records[0];

    let origin_termination =
        TraceTermination::try_from(origin.metadata[0]).expect("trace writes a typed termination");
    assert_eq!(origin_termination, TraceTermination::Escape);
    assert_eq!(
        Ok(origin_termination),
        TraceTermination::try_from(translated.metadata[0])
    );
    assert_eq!(origin.event, translated.event);
    let origin_direction: [f32; 3] = origin.source_time[..3]
        .try_into()
        .expect("trace direction contains three components");
    let translated_direction: [f32; 3] = translated.source_time[..3]
        .try_into()
        .expect("trace direction contains three components");
    let angular_difference = angle_between(
        origin_direction.map(f64::from),
        translated_direction.map(f64::from),
    );
    assert!(
        angular_difference <= 2.0e-6,
        "time translation changed the trace direction by {angular_difference:e} rad"
    );
    let travel_time_difference = (origin.source_time[3] - translated.source_time[3]).abs();
    assert!(
        travel_time_difference <= 1.0e-3,
        "time translation changed travel time from {} to {}",
        origin.source_time[3],
        translated.source_time[3]
    );
}

#[test]
fn far_field_geometry_does_not_fail_when_the_guard_observable_exceeds_f32() {
    let observation = observation_with(1.0, 0.8, 0.0, [1.0e10, 0.0, 0.0], 1.0, 1, 1);
    let capture = capture_initial_rays(&observation, [0.5, 0.5]);

    assert_eq!(capture.records[0].metadata, [0; 4]);
}

#[test]
fn default_view_produces_only_determinate_rays() {
    let capture = capture_trace(&default_observation(80, 45));
    let mut horizon = 0;
    let mut escape = 0;

    for record in capture.records {
        match TraceTermination::try_from(record.metadata[0]) {
            Ok(TraceTermination::HorizonCrossing) => horizon += 1,
            Ok(TraceTermination::Escape) => escape += 1,
            termination => panic!(
                "default view contains {termination:?} after {} steps with drift {:?}",
                record.metadata[2], record.invariant_drift
            ),
        }
    }

    assert!(
        horizon > 0,
        "default view does not contain the black-hole shadow"
    );
    assert!(escape > 0, "default view does not contain the lensed sky");
}

#[test]
fn every_recorded_invariant_can_make_a_terminal_uncertain() {
    let observation = default_observation(4, 1);
    let capture = capture_invariant_gate_cases(&observation);

    for (component, record) in capture.records.iter().enumerate() {
        assert_eq!(
            TraceTermination::try_from(record.metadata[0]),
            Ok(TraceTermination::Uncertain),
            "invariant component {component} did not gate the terminal"
        );
        assert!(record.invariant_drift[component] > INVARIANT_DRIFT_LIMIT);
    }
}

#[test]
fn trace_dispatch_writes_refinable_branch_coverage_across_workgroup_boundaries() {
    let observation = default_observation(9, 9);
    let capture = capture_trace(&observation);

    assert_eq!(capture.records.len(), 81);
    for (index, record) in capture.records.iter().enumerate() {
        let termination = TraceTermination::try_from(record.metadata[0])
            .unwrap_or_else(|error| panic!("pixel {index} has no typed termination: {error}"));
        let expected_coverage = match termination {
            TraceTermination::HorizonCrossing => 0,
            TraceTermination::Escape => f16_one_bits(),
            other => panic!("default view pixel {index} has non-refinable termination {other:?}"),
        };
        assert_eq!(
            u16::from_le_bytes(
                capture.hdr_pixel(index)[6..]
                    .try_into()
                    .expect("RGBA16F alpha occupies two bytes")
            ),
            expected_coverage,
            "pixel {index} does not expose its Horizon/Escape coverage class"
        );
    }
}

#[test]
fn selective_refinement_changes_only_the_horizon_escape_boundary() {
    let observation = default_observation(64, 36);
    let base = capture_trace(&observation);
    let refined = capture_refined_trace(&observation);
    let mut boundary_pixels = 0;
    let mut fractional_coverage_pixels = 0;

    for y in 1..35 {
        for x in 1..63 {
            let index = usize::try_from(y * 64 + x).expect("test index fits usize");
            let Some(center_is_escape) = refinable_branch(base.records[index].metadata[0]) else {
                continue;
            };
            let is_boundary = [index - 1, index + 1, index - 64, index + 64]
                .into_iter()
                .filter_map(|neighbor| refinable_branch(base.records[neighbor].metadata[0]))
                .any(|neighbor_is_escape| neighbor_is_escape != center_is_escape);

            if !is_boundary {
                assert_eq!(
                    refined.hdr_pixel(index),
                    base.hdr_pixel(index),
                    "non-boundary pixel ({x}, {y}) was needlessly retraced"
                );
                continue;
            }

            boundary_pixels += 1;
            let alpha = u16::from_le_bytes(
                refined.hdr_pixel(index)[6..]
                    .try_into()
                    .expect("RGBA16F alpha occupies two bytes"),
            );
            assert!(
                [0, 0x3400, 0x3800, 0x3a00, f16_one_bits()].contains(&alpha),
                "boundary pixel ({x}, {y}) has non-four-sample coverage {alpha:#06x}"
            );
            fractional_coverage_pixels += usize::from(![0, f16_one_bits()].contains(&alpha));
        }
    }

    assert!(
        boundary_pixels > 0,
        "fixture does not cross the shadow boundary"
    );
    assert!(
        fractional_coverage_pixels > 0,
        "selective refinement left the curved shadow boundary binary"
    );
}

#[test]
fn repeated_refinement_in_one_submission_does_not_accumulate_edges() {
    let observation = default_observation(64, 36);
    let once = capture_refined_edge_count(&observation, 1);
    let repeated = capture_refined_edge_count(&observation, 2);

    assert!(once > 0, "fixture does not cross the shadow boundary");
    assert_eq!(repeated, once);
}

#[test]
fn accelerated_trace_batches_reuse_the_same_packed_stencil_contract() {
    let observation = default_observation(17, 9);
    let single = capture_accelerated_trace(&observation);
    let progressive = capture_accelerated_trace_in_batches(&observation, 2);

    assert_eq!(single.records.len(), progressive.records.len());
    for (pixel, (single, progressive)) in
        single.records.iter().zip(&progressive.records).enumerate()
    {
        assert_same_bits(single.source_time, progressive.source_time);
        assert_eq!(single.metadata[0], progressive.metadata[0], "pixel {pixel}");
        assert_eq!(single.event, progressive.event, "pixel {pixel}");
    }
}

#[test]
fn wgsl_initial_rays_match_cpu_center_corners_and_jitter() {
    let observation = default_observation(7, 5);
    let samples = [
        sample(&observation, 3, 2, 0.5, 0.5),
        sample(&observation, 0, 0, 0.0, 0.0),
        sample(&observation, 6, 0, 1.0, 0.0),
        sample(&observation, 0, 4, 0.0, 1.0),
        sample(&observation, 6, 4, 1.0, 1.0),
        sample(&observation, 4, 1, 0.125, 0.875),
    ];

    for image_sample in samples {
        let subpixel = image_sample.subpixel().map(|value| {
            value
                .to_f32()
                .expect("normalized subpixel coordinate is representable as f32")
        });
        let capture = capture_initial_rays(&observation, subpixel);
        let [pixel_x, pixel_y] = image_sample.pixel();
        let index = usize::try_from(pixel_y * 7 + pixel_x).expect("test index fits usize");
        let gpu = capture.records[index];
        let cpu = observation
            .initial_ray(image_sample)
            .expect("validated sample resolves");
        let cpu_sight = cpu.sight_direction_txyz();
        let cpu_direction = [cpu_sight[1], cpu_sight[2], cpu_sight[3]];
        let gpu_direction = [
            f64::from(gpu.source_time[0]),
            f64::from(gpu.source_time[1]),
            f64::from(gpu.source_time[2]),
        ];
        let angle = angle_between(cpu_direction, gpu_direction);

        assert!(
            angle <= 2.0e-6,
            "{image_sample:?}: {angle:e} rad, CPU {cpu_direction:?}, GPU {gpu_direction:?}"
        );
        assert!(
            gpu.invariant_drift[0] <= 8.0e-5,
            "{image_sample:?}: initial null residual {}",
            gpu.invariant_drift[0]
        );
    }
}

#[test]
fn headless_gpu_regular_matrix_matches_reference_termination_and_escape_direction() {
    let observation = default_observation(7, 5);
    let capture = capture_trace(&observation);
    let samples = [
        sample(&observation, 0, 0, 0.5, 0.5),
        sample(&observation, 6, 0, 0.5, 0.5),
        sample(&observation, 0, 4, 0.5, 0.5),
        sample(&observation, 6, 4, 0.5, 0.5),
        sample(&observation, 3, 0, 0.5, 0.5),
        sample(&observation, 3, 2, 0.5, 0.5),
    ];
    let oracle = ObservationTracer::baseline_v1();

    for image_sample in samples {
        let [pixel_x, pixel_y] = image_sample.pixel();
        let index = usize::try_from(pixel_y * 7 + pixel_x).expect("test index fits usize");
        let gpu = capture.records[index];
        let gpu_termination =
            TraceTermination::try_from(gpu.metadata[0]).expect("GPU writes a typed termination");
        let reference = oracle
            .trace(
                ObservationTrace::new(
                    TraceInputId::new(format!("gpu-{pixel_x}-{pixel_y}")),
                    &observation,
                    image_sample,
                    ReferencePolicy::regular_v1(),
                )
                .expect("reference request resolves"),
            )
            .expect("default observation is normalized");
        let expected_termination = match reference.termination() {
            Termination::HorizonCrossing => TraceTermination::HorizonCrossing,
            Termination::Escape => TraceTermination::Escape,
            other => panic!("regular matrix produced unsupported reference terminal {other:?}"),
        };

        assert_eq!(gpu_termination, expected_termination, "{image_sample:?}");
        assert_eq!(gpu.metadata[1], 0, "{image_sample:?}: failure flags");
        assert!(gpu.metadata[2] > 0, "{image_sample:?}: step counter");

        if let Some(reference_direction) = reference.escape_direction_xyz() {
            let gpu_direction = [
                f64::from(gpu.source_time[0]),
                f64::from(gpu.source_time[1]),
                f64::from(gpu.source_time[2]),
            ];
            let angular_error = angle_between(reference_direction, gpu_direction);
            assert!(
                angular_error <= 3.82e-4,
                "{image_sample:?}: escape error {angular_error:e} rad"
            );
        }

        let event_residual = f32::from_bits(gpu.metadata[3]).abs();
        assert!(
            event_residual <= 5.0e-3,
            "{image_sample:?}: event residual {event_residual:e}"
        );

        let travel_time_error = (f64::from(gpu.source_time[3]) - reference.travel_time_m()).abs();
        assert!(
            travel_time_error <= 1.0e-3,
            "{image_sample:?}: travel-time error {travel_time_error:e}; GPU {}, reference {}; GPU drift {:?}, reference drift {:?}",
            gpu.source_time[3],
            reference.travel_time_m(),
            gpu.invariant_drift,
            reference.diagnostics(),
        );
        for (invariant, drift) in ["null", "energy", "angular momentum", "Carter"]
            .into_iter()
            .zip(gpu.invariant_drift)
        {
            assert!(
                (0.0..=INVARIANT_DRIFT_LIMIT).contains(&drift),
                "{image_sample:?}: {invariant} drift {drift:e}"
            );
        }
    }
}

#[test]
fn accelerated_trace_preserves_full_trace_terminals_and_escape_directions() {
    let profiles = [
        ("default", 0.8, 0.0, 30.0, std::f64::consts::FRAC_PI_4),
        (
            "near-extreme-wide",
            0.99,
            0.0,
            30.0,
            std::f64::consts::FRAC_PI_2,
        ),
        ("negative-near", -0.8, 0.0, 5.0, std::f64::consts::FRAC_PI_4),
        ("kerr-newman", 0.6, 0.5, 30.0, std::f64::consts::FRAC_PI_4),
    ];

    for (label, spin, charge, observer_radius, vertical_fov) in profiles {
        let observation = transfer_profile(spin, charge, observer_radius, vertical_fov);
        let full = capture_trace(&observation);
        let transferred = capture_accelerated_trace(&observation);
        let mut reconstructed_escape_pixels = 0;

        assert_eq!(transferred.records.len(), full.records.len());
        for (index, (expected, actual)) in full.records.iter().zip(&transferred.records).enumerate()
        {
            assert_eq!(
                actual.metadata[0], expected.metadata[0],
                "{label}: transfer changed the terminal branch at pixel {index}"
            );
            // Escape-map reconstruction promises terminal/event ambiguity and direction. It does
            // not manufacture the surface-only source branch key stored in the remaining lanes.
            assert_eq!(
                actual.event[..2],
                expected.event[..2],
                "{label}: transfer changed event diagnostics at pixel {index}"
            );
            if TraceTermination::try_from(expected.metadata[0]) != Ok(TraceTermination::Escape) {
                continue;
            }

            let expected_direction = &expected.source_time[..3];
            let actual_direction = &actual.source_time[..3];
            let chord_squared = expected_direction
                .iter()
                .zip(actual_direction)
                .map(|(lhs, rhs)| {
                    let difference = f64::from(*lhs) - f64::from(*rhs);
                    difference * difference
                })
                .sum::<f64>();
            assert!(
                chord_squared <= (3.82e-4_f64).powi(2),
                "{label}: transfer direction exceeded the angular budget at pixel {index}: chord²={chord_squared:e}"
            );
            reconstructed_escape_pixels += usize::from(
                expected_direction
                    .iter()
                    .zip(actual_direction)
                    .any(|(lhs, rhs)| lhs.to_bits() != rhs.to_bits()),
            );
        }

        if observer_radius >= 30.0 {
            assert!(
                reconstructed_escape_pixels > 0,
                "{label}: far-field fixture did not exercise the cooperative escape-direction map"
            );
        }
    }
}

#[test]
fn an_adjacent_shadow_edge_pair_matches_reference_classification() {
    let observation = default_observation(160, 90);
    let capture = capture_trace(&observation);
    let oracle = ObservationTracer::baseline_v1();
    let edge_pair = (0..90)
        .find_map(|pixel_y| {
            (0..159).find_map(|pixel_x| {
                let left_index =
                    usize::try_from(pixel_y * 160 + pixel_x).expect("test index fits usize");
                let right_index = left_index + 1;
                let left = refinable_branch(capture.records[left_index].metadata[0])?;
                let right = refinable_branch(capture.records[right_index].metadata[0])?;
                (left != right).then_some([(pixel_x, pixel_y), (pixel_x + 1, pixel_y)])
            })
        })
        .expect("default view contains a horizontal Horizon/Escape boundary");
    let mut terminations = Vec::new();

    for (pixel_x, pixel_y) in edge_pair {
        let image_sample = sample(&observation, pixel_x, pixel_y, 0.5, 0.5);
        let [pixel_x, pixel_y] = image_sample.pixel();
        let index = usize::try_from(pixel_y * 160 + pixel_x).expect("test index fits usize");
        let gpu = TraceTermination::try_from(capture.records[index].metadata[0])
            .expect("GPU writes a typed termination");
        let reference = oracle
            .trace(
                ObservationTrace::new(
                    TraceInputId::new(format!("shadow-edge-{pixel_x}-{pixel_y}")),
                    &observation,
                    image_sample,
                    ReferencePolicy::regular_v1(),
                )
                .expect("reference request resolves"),
            )
            .expect("default observation is normalized");
        let expected = match reference.termination() {
            Termination::HorizonCrossing => TraceTermination::HorizonCrossing,
            Termination::Escape => TraceTermination::Escape,
            other => panic!("shadow-edge reference produced {other:?}"),
        };

        assert_eq!(gpu, expected, "{image_sample:?}");
        terminations.push(gpu);
    }

    terminations.sort_unstable_by_key(|termination| u32::from(*termination));
    assert_eq!(
        terminations,
        [TraceTermination::HorizonCrossing, TraceTermination::Escape]
    );
}

const fn f16_one_bits() -> u16 {
    0x3c00
}

fn refinable_branch(raw: u32) -> Option<bool> {
    match TraceTermination::try_from(raw).ok()? {
        TraceTermination::HorizonCrossing => Some(false),
        TraceTermination::Escape => Some(true),
        TraceTermination::SingularityGuard
        | TraceTermination::StepExhaustion
        | TraceTermination::NumericalFailure
        | TraceTermination::Uncertain
        | TraceTermination::EquatorialSurface => None,
    }
}

fn assert_same_bits(left: [f32; 4], right: [f32; 4]) {
    assert_eq!(left.map(f32::to_bits), right.map(f32::to_bits));
}

fn decode_f16(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let significand = bits & 0x03ff;
    match exponent {
        0 => sign * f32::from(significand) * 2.0_f32.powi(-24),
        31 if significand == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + f32::from(significand) / 1024.0) * 2.0_f32.powi(exponent - 15),
    }
}

pub fn default_observation(width: u32, height: u32) -> Observation {
    observation_at_scale(width, height, 1.0)
}

fn transfer_profile(
    spin: f64,
    charge: f64,
    observer_radius: f64,
    vertical_fov: f64,
) -> Observation {
    transfer_profile_extent(spin, charge, observer_radius, vertical_fov, 1_280, 720)
}

fn transfer_profile_extent(
    spin: f64,
    charge: f64,
    observer_radius: f64,
    vertical_fov: f64,
    width: u32,
    height: u32,
) -> Observation {
    let spacetime = KerrNewmanSpacetime::new(1.0, spin, charge, KerrSchildChart::Outgoing)
        .expect("transfer profile spacetime is valid");
    let observer_xyz =
        spacetime.oblate_to_cartesian(observer_radius, std::f64::consts::FRAC_PI_3, 0.0);
    let observer = StationaryObserverInput::new(
        [0.0, observer_xyz[0], observer_xyz[1], observer_xyz[2]],
        [0.0; 4],
        [0.0, 0.0, 1.0],
        1.0,
    );
    let scene = PhysicalScene::new(PhysicalSceneInput::new(
        1.0,
        spin,
        charge,
        KerrSchildChart::Outgoing,
        observer,
    ))
    .expect("transfer profile scene is valid");
    let view = PerspectiveView::new(
        NonZeroU32::new(width).expect("profile width is nonzero"),
        NonZeroU32::new(height).expect("profile height is nonzero"),
        Angle::from_radians(vertical_fov).expect("profile FOV is finite"),
    )
    .expect("transfer profile view is valid");
    Observation::new(scene, view)
}

fn observation_at_coordinate_time(width: u32, height: u32, coordinate_time_m: f64) -> Observation {
    let mass = 1.0;
    let spin = 0.8;
    let spacetime = KerrNewmanSpacetime::new(mass, spin, 0.0, KerrSchildChart::Outgoing)
        .expect("fixture spacetime is valid");
    let observer_xyz = spacetime.oblate_to_cartesian(30.0, std::f64::consts::FRAC_PI_3, 0.0);
    observation_with_time(
        mass,
        spin,
        0.0,
        observer_xyz,
        1.0,
        coordinate_time_m,
        [width, height],
    )
}

fn observation_at_scale(width: u32, height: u32, mass: f64) -> Observation {
    let spin = 0.8 * mass;
    let spacetime = KerrNewmanSpacetime::new(mass, spin, 0.0, KerrSchildChart::Outgoing)
        .expect("fixture spacetime is valid");
    let observer_xyz = spacetime.oblate_to_cartesian(30.0 * mass, std::f64::consts::FRAC_PI_3, 0.0);
    observation_with(mass, spin, 0.0, observer_xyz, 1.0, width, height)
}

fn observation_with(
    mass: f64,
    spin: f64,
    charge: f64,
    observer_xyz: [f64; 3],
    observer_frequency: f64,
    width: u32,
    height: u32,
) -> Observation {
    observation_with_time(
        mass,
        spin,
        charge,
        observer_xyz,
        observer_frequency,
        0.0,
        [width, height],
    )
}

fn observation_with_time(
    mass: f64,
    spin: f64,
    charge: f64,
    observer_xyz: [f64; 3],
    observer_frequency: f64,
    coordinate_time_m: f64,
    extent: [u32; 2],
) -> Observation {
    let observer = StationaryObserverInput::new(
        [
            coordinate_time_m,
            observer_xyz[0],
            observer_xyz[1],
            observer_xyz[2],
        ],
        [coordinate_time_m, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        observer_frequency,
    );
    let scene = PhysicalScene::new(PhysicalSceneInput::new(
        mass,
        spin,
        charge,
        KerrSchildChart::Outgoing,
        observer,
    ))
    .expect("fixture scene is valid");
    let view = PerspectiveView::new(
        NonZeroU32::new(extent[0]).expect("test width is nonzero"),
        NonZeroU32::new(extent[1]).expect("test height is nonzero"),
        Angle::from_radians(std::f64::consts::FRAC_PI_4).expect("fixture FOV is finite"),
    )
    .expect("fixture view is valid");
    Observation::new(scene, view)
}

fn sample(
    observation: &Observation,
    pixel_x: u32,
    pixel_y: u32,
    subpixel_x: f64,
    subpixel_y: f64,
) -> ImageSample {
    observation
        .view()
        .sample(pixel_x, pixel_y, subpixel_x, subpixel_y)
        .expect("test sample is in range")
}

fn angle_between(left: [f64; 3], right: [f64; 3]) -> f64 {
    let dot = left
        .into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let cross = [
        left[2].mul_add(-right[1], left[1] * right[2]),
        left[0].mul_add(-right[2], left[2] * right[0]),
        left[1].mul_add(-right[0], left[0] * right[1]),
    ];
    let cross_length = cross
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    cross_length.atan2(dot)
}

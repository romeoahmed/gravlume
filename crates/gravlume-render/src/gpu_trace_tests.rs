use std::{mem::size_of, num::NonZeroU32};

use approx::{abs_diff_eq, assert_abs_diff_eq};
use gravlume_domain::{
    Angle, EquatorialCircularEmitter, EquatorialSurface, ImageSample, KerrNewmanSpacetime,
    KerrSchildChart, Observation, PerspectiveView, PhysicalScene, PhysicalSceneInput,
    StationaryObserverInput, SurfaceTransport,
};
use gravlume_reference::{
    ObservationTrace, ObservationTracer, PolarSide, ReferenceOutcome, ReferencePolicy, Termination,
    TraceBranchKey, TraceInputId,
};
use num_traits::ToPrimitive as _;

use crate::{
    gpu_capture::{
        TraceCapture, capture_event_policy_cases, capture_initial_rays,
        capture_invariant_gate_cases, capture_refined_edge_count, capture_refined_trace,
        capture_sample_corpus, capture_trace, capture_trace_in_batches, capture_trace_sample,
        inspect_sample,
    },
    scientific_capture::{GPU_BOLOMETRIC_RADIANCE_RELATIVE_BUDGET, INVARIANT_RELATIVE_DRIFT_LIMIT},
    trace::{
        SampleBranchKey, SamplePolarSide, SampleRetrace, SampleSurfaceEvaluation,
        SampleTraceDiagnostics, SampleTraceOutcome, TraceTermination,
    },
};

const EVENT_CANDIDATE_HORIZON: u32 = 1 << 1;
const EVENT_CANDIDATE_SURFACE: u32 = 1 << 2;
const EVENT_CANDIDATE_ESCAPE: u32 = 1 << 3;
const GPU_INITIAL_RAY_ANGULAR_BUDGET_RAD: f64 = 2.0e-6;
const GPU_INITIAL_NULL_RESIDUAL_ABSOLUTE_BUDGET: f32 = 8.0e-5;
const GPU_REGULAR_ANGULAR_BUDGET_RAD: f64 = 3.82e-4;
const GPU_EVENT_RESIDUAL_ABSOLUTE_BUDGET: f32 = 5.0e-3;
const GPU_TRAVEL_TIME_ABSOLUTE_BUDGET_M: f64 = 1.0e-3;
const GPU_SURFACE_POSITION_ABSOLUTE_BUDGET_M: f64 = 5.0e-3;
const GPU_FREQUENCY_RATIO_RELATIVE_BUDGET: f64 = 2.0e-3;

mod surface;

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
    let gpu = capture.record;
    let reference = ObservationTracer::baseline_v1()
        .trace(
            fixture
                .trace_request(ReferencePolicy::regular_v1())
                .expect("fixture sample resolves through its observation"),
        )
        .expect("canonical surface source is valid");
    let reference_observable = reference
        .terminal()
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
    assert_abs_diff_eq!(
        f64::from(gpu.source_time[0]),
        reference_anchor.radius_m(),
        epsilon = GPU_SURFACE_POSITION_ABSOLUTE_BUDGET_M
    );
    let azimuth_error = (f64::from(gpu.source_time[1]) - reference_anchor.azimuth_rad())
        .sin()
        .atan2((f64::from(gpu.source_time[1]) - reference_anchor.azimuth_rad()).cos());
    assert_abs_diff_eq!(azimuth_error, 0.0, epsilon = GPU_REGULAR_ANGULAR_BUDGET_RAD);
    let reference_ratio = reference_observable.frequency_ratio().value();
    assert_abs_diff_eq!(
        f64::from(gpu.source_time[2]),
        reference_ratio,
        epsilon = GPU_FREQUENCY_RATIO_RELATIVE_BUDGET * reference_ratio
    );
    assert_abs_diff_eq!(
        f64::from(gpu.source_time[3]),
        reference.travel_time_m(),
        epsilon = GPU_TRAVEL_TIME_ABSOLUTE_BUDGET_M
    );

    let expected = reference_observable.observed_bolometric_intensity();
    let pixel = capture.hdr_pixel;
    assert_eq!(
        u16::from_le_bytes(pixel[6..].try_into().expect("alpha has two bytes")),
        0x4000,
        "surface radiance alpha tag"
    );
    for channel in pixel[..6].as_chunks::<2>().0 {
        let actual = decode_f16(u16::from_le_bytes(*channel));
        assert_abs_diff_eq!(
            f64::from(actual),
            expected,
            epsilon = GPU_BOLOMETRIC_RADIANCE_RELATIVE_BUDGET * expected
        );
    }
}

#[test]
fn bounded_sample_inspection_exposes_the_canonical_surface_observables() {
    const SURFACE_OBSERVABLE: &str =
        include_str!("../../gravlume-reference/fixtures/v2/kerr-surface-observable.toml");
    let fixture = gravlume_reference::FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let observation = fixture.observation();
    let sample = fixture.sample();
    let inspection = inspect_sample(observation, sample);
    let reference = ObservationTracer::baseline_v1()
        .trace(
            fixture
                .trace_request(ReferencePolicy::regular_v1())
                .expect("fixture sample resolves through its observation"),
        )
        .expect("canonical surface source is valid");
    let retrace = inspection.fresh_retrace();
    assert_retrace_matches_reference(sample, &reference, retrace);
    let SampleTraceOutcome::EquatorialSurface { channels, .. } = retrace.outcome() else {
        panic!("canonical inspection must expose an equatorial-surface outcome");
    };
    assert_eq!(channels, crate::ScientificChannelModel::BolometricRepeated);

    let diagnostics = retrace.diagnostics();
    assert_eq!(diagnostics.event_candidate_bits(), EVENT_CANDIDATE_SURFACE);
}

#[test]
fn bounded_sample_inspection_keeps_analytic_escape_semantics_non_spectral() {
    let observation = default_observation(64, 32);
    let sample = observation
        .view()
        .sample(0, 0, 0.25, 0.75)
        .expect("analytic sample belongs to the view");
    let inspection = inspect_sample(&observation, sample);
    let reference = ObservationTracer::baseline_v1()
        .trace(
            ObservationTrace::new(
                TraceInputId::new("analytic-inspection"),
                &observation,
                sample,
                ReferencePolicy::regular_v1(),
            )
            .expect("reference request resolves"),
        )
        .expect("default observation is normalized");
    let retrace = inspection.fresh_retrace();

    assert_retrace_matches_reference(sample, &reference, retrace);

    assert!(matches!(
        retrace.outcome(),
        SampleTraceOutcome::Escape {
            preview_rgb,
            ..
        } if preview_rgb.into_iter().all(f32::is_finite)
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
        base.scene().clone().with_equatorial_surface(
            EquatorialSurface::new(emitter, SurfaceTransport::Vacuum)
                .expect("test surface is compatible with vacuum"),
        ),
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

    assert_trace_captures_equivalent(&unit, &scaled);
}

#[test]
fn coordinate_time_origin_does_not_change_gpu_trace_observables() {
    let origin = capture_trace(&observation_at_coordinate_time(7, 5, 0.0));
    let translated = capture_trace(&observation_at_coordinate_time(7, 5, 1.0e8));

    assert_trace_captures_equivalent(&origin, &translated);
}

#[test]
fn production_trace_is_invariant_to_cross_submission_batch_partitioning() {
    let observation = default_observation(17, 9);
    let single_submission = capture_refined_trace(&observation);
    let batched = capture_trace_in_batches(&observation, 2);

    assert_trace_captures_equivalent(&single_submission, &batched);
}

#[test]
fn far_field_geometry_does_not_fail_when_the_guard_observable_exceeds_f32() {
    let observation = observation_with(1.0, 0.8, 0.0, [1.0e10, 0.0, 0.0], 1.0, 1, 1);
    let capture = capture_initial_rays(&observation, [0.5, 0.5]);
    let record = capture.records[0];

    assert_eq!(record.metadata[0], 0, "initial-ray construction failed");
    assert!(record.source_time.into_iter().all(f32::is_finite));
    assert_abs_diff_eq!(
        record.invariant_drift[0],
        0.0,
        epsilon = GPU_INITIAL_NULL_RESIDUAL_ABSOLUTE_BUDGET
    );
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
        assert!(record.invariant_drift[component] > INVARIANT_RELATIVE_DRIFT_LIMIT);
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
        let actual_coverage = u16::from_le_bytes(
            capture.hdr_pixel(index)[6..]
                .try_into()
                .expect("RGBA16F alpha occupies two bytes"),
        );
        match termination {
            TraceTermination::HorizonCrossing => assert_eq!(
                actual_coverage & 0x7fff,
                0,
                "pixel {index} does not expose Horizon coverage"
            ),
            TraceTermination::Escape => assert_eq!(
                actual_coverage,
                f16_one_bits(),
                "pixel {index} does not expose Escape coverage"
            ),
            other => panic!("default view pixel {index} has non-refinable termination {other:?}"),
        }
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
                    canonical_half_texel(refined.hdr_pixel(index)),
                    canonical_half_texel(base.hdr_pixel(index)),
                    "non-boundary pixel ({x}, {y}) was needlessly retraced"
                );
                continue;
            }

            boundary_pixels += 1;
            let alpha = u16::from_le_bytes(
                refined.hdr_pixel(index)[6..]
                    .try_into()
                    .expect("RGBA16F alpha occupies two bytes"),
            ) & 0x7fff;
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
            abs_diff_eq!(angle, 0.0, epsilon = GPU_INITIAL_RAY_ANGULAR_BUDGET_RAD),
            "{image_sample:?}: {angle:e} rad, CPU {cpu_direction:?}, GPU {gpu_direction:?}"
        );
        assert!(
            abs_diff_eq!(
                gpu.invariant_drift[0],
                0.0,
                epsilon = GPU_INITIAL_NULL_RESIDUAL_ABSOLUTE_BUDGET
            ),
            "{image_sample:?}: initial null residual {}",
            gpu.invariant_drift[0]
        );
    }
}

#[test]
fn batched_gpu_regular_corpus_matches_reference_observables_in_request_order() {
    let observation = default_observation(7, 5);
    let samples = [
        sample(&observation, 0, 0, 0.5, 0.5),
        sample(&observation, 6, 0, 0.5, 0.5),
        sample(&observation, 0, 4, 0.5, 0.5),
        sample(&observation, 6, 4, 0.5, 0.5),
        sample(&observation, 3, 0, 0.5, 0.5),
        sample(&observation, 3, 2, 0.5, 0.5),
    ];
    let retraces = capture_sample_corpus(&observation, &samples);
    let oracle = ObservationTracer::baseline_v1();

    assert_eq!(retraces.len(), samples.len());
    for (image_sample, gpu) in samples.into_iter().zip(retraces) {
        let [pixel_x, pixel_y] = image_sample.pixel();
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
        assert_retrace_matches_reference(image_sample, &reference, gpu);
    }
}

#[test]
fn sample_corpus_crosses_a_partial_workgroup_without_reordering_records() {
    let observation = default_observation(13, 5);
    let mut samples = Vec::with_capacity(65);
    for pixel_y in 0..5 {
        for pixel_x in 0..13 {
            samples.push(sample(&observation, pixel_x, pixel_y, 0.25, 0.75));
        }
    }
    samples.reverse();

    let retraces = capture_sample_corpus(&observation, &samples);

    assert_eq!(retraces.len(), 65);
    for index in [0, 63, 64] {
        let single = inspect_sample(&observation, samples[index]).fresh_retrace();
        assert_eq!(retraces[index], single, "request index {index}");
    }
    let repeated_samples = [samples[64], samples[0], samples[64]];
    assert_eq!(
        capture_sample_corpus(&observation, &repeated_samples),
        vec![retraces[64], retraces[0], retraces[64]],
        "duplicate requests preserve multiplicity and order"
    );
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

fn assert_trace_captures_equivalent(left: &TraceCapture, right: &TraceCapture) {
    assert_eq!(left.records.len(), right.records.len());
    for (index, (left_record, right_record)) in left.records.iter().zip(&right.records).enumerate()
    {
        assert_same_bits(left_record.source_time, right_record.source_time);
        assert_same_bits(left_record.invariant_drift, right_record.invariant_drift);
        assert_eq!(left_record.metadata, right_record.metadata, "pixel {index}");
        assert_eq!(left_record.event, right_record.event, "pixel {index}");
        assert_eq!(
            canonical_half_texel(left.hdr_pixel(index)),
            canonical_half_texel(right.hdr_pixel(index)),
            "pixel {index} final HDR texel",
        );
    }
}

fn canonical_half_texel(bytes: [u8; 8]) -> [u16; 4] {
    std::array::from_fn(|channel| {
        let offset = channel * size_of::<u16>();
        let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        if matches!(bits, 0 | 0x8000) { 0 } else { bits }
    })
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

fn assert_retrace_matches_reference(
    sample: ImageSample,
    reference: &ReferenceOutcome,
    gpu: SampleRetrace,
) {
    let gpu_branch = match (reference.termination(), gpu.outcome()) {
        (Termination::HorizonCrossing, SampleTraceOutcome::Horizon { branch }) => branch,
        (
            Termination::Escape,
            SampleTraceOutcome::Escape {
                branch,
                unit_direction,
                ..
            },
        ) => {
            let reference_direction = reference
                .terminal()
                .escape_direction()
                .and_then(gravlume_reference::EscapeDirection::xyz)
                .expect("reference escape direction resolves");
            let angular_error = angle_between(reference_direction, unit_direction.map(f64::from));
            assert!(
                abs_diff_eq!(angular_error, 0.0, epsilon = GPU_REGULAR_ANGULAR_BUDGET_RAD),
                "{sample:?}: escape direction error {angular_error:e} rad"
            );
            branch
        }
        (
            Termination::EquatorialSurface,
            SampleTraceOutcome::EquatorialSurface {
                branch,
                radius_over_m,
                azimuth_radians,
                frequency_ratio,
                evaluation,
                ..
            },
        ) => {
            let observable = reference
                .terminal()
                .surface_observable()
                .expect("reference surface observable resolves");
            let anchor = observable.source_anchor();
            assert!(
                abs_diff_eq!(
                    f64::from(radius_over_m),
                    anchor.radius_m(),
                    epsilon = GPU_SURFACE_POSITION_ABSOLUTE_BUDGET_M
                ),
                "{sample:?}: surface radius disagrees with the reference"
            );
            let azimuth_delta = f64::from(azimuth_radians) - anchor.azimuth_rad();
            let azimuth_error = azimuth_delta.sin().atan2(azimuth_delta.cos());
            assert!(
                abs_diff_eq!(azimuth_error, 0.0, epsilon = GPU_REGULAR_ANGULAR_BUDGET_RAD),
                "{sample:?}: surface azimuth error {azimuth_error:e} rad"
            );
            let expected_ratio = observable.frequency_ratio().value();
            assert!(
                abs_diff_eq!(
                    f64::from(frequency_ratio),
                    expected_ratio,
                    epsilon = GPU_FREQUENCY_RATIO_RELATIVE_BUDGET * expected_ratio
                ),
                "{sample:?}: frequency ratio disagrees with the reference"
            );
            let SampleSurfaceEvaluation::Radiance(actual) = evaluation else {
                panic!("{sample:?}: surface radiance must be finite");
            };
            let expected_intensity = observable.observed_bolometric_intensity();
            for channel in actual {
                assert!(
                    abs_diff_eq!(
                        f64::from(channel),
                        expected_intensity,
                        epsilon = GPU_BOLOMETRIC_RADIANCE_RELATIVE_BUDGET * expected_intensity
                    ),
                    "{sample:?}: surface radiance disagrees with the reference"
                );
            }
            branch
        }
        (expected, actual) => panic!("{sample:?}: CPU {expected:?} disagrees with GPU {actual:?}"),
    };

    assert_branch_matches(sample, gpu_branch, reference.branch_key());
    assert_diagnostics_match(sample, gpu.diagnostics(), reference);
}

fn assert_branch_matches(sample: ImageSample, gpu: SampleBranchKey, reference: TraceBranchKey) {
    let reference_polar_side = match reference.initial_polar_side() {
        PolarSide::Negative => SamplePolarSide::Negative,
        PolarSide::Equatorial => SamplePolarSide::Equatorial,
        PolarSide::Positive => SamplePolarSide::Positive,
    };
    assert_eq!(gpu.initial_polar_side(), reference_polar_side, "{sample:?}");
    assert_eq!(
        gpu.radial_turnings(),
        reference.radial_turnings(),
        "{sample:?}"
    );
    assert_eq!(
        gpu.equatorial_crossings(),
        reference.equatorial_crossings(),
        "{sample:?}"
    );
    assert_eq!(
        gpu.azimuth_winding(),
        reference.azimuth_winding(),
        "{sample:?}"
    );
}

fn assert_diagnostics_match(
    sample: ImageSample,
    gpu: SampleTraceDiagnostics,
    reference: &ReferenceOutcome,
) {
    assert_eq!(gpu.numerical_flag_bits(), 0, "{sample:?}: failure flags");
    assert!(gpu.steps() > 0, "{sample:?}: step counter");
    assert!(
        abs_diff_eq!(
            gpu.event_residual(),
            0.0,
            epsilon = GPU_EVENT_RESIDUAL_ABSOLUTE_BUDGET
        ),
        "{sample:?}: event residual {:e}",
        gpu.event_residual()
    );

    let gpu_travel_time = gpu.coordinate_time_delta_over_m();
    assert!(
        abs_diff_eq!(
            f64::from(gpu_travel_time),
            reference.travel_time_m(),
            epsilon = GPU_TRAVEL_TIME_ABSOLUTE_BUDGET_M
        ),
        "{sample:?}: GPU travel time {gpu_travel_time}, reference {}; GPU drift {:?}, reference drift {:?}",
        reference.travel_time_m(),
        gpu.maximum_invariant_drift(),
        reference.diagnostics(),
    );
    for (invariant, drift) in ["null", "energy", "angular momentum", "Carter"]
        .into_iter()
        .zip(gpu.maximum_invariant_drift())
    {
        assert!(
            (0.0..=INVARIANT_RELATIVE_DRIFT_LIMIT).contains(&drift),
            "{sample:?}: {invariant} drift {drift:e}"
        );
    }
}

use std::{
    mem::{offset_of, size_of},
    num::NonZeroU32,
};

use gravlume_domain::{
    Angle, KerrNewmanSpacetime, KerrSchildCoordinates, Observation, PhysicalScene,
    PhysicalSceneDraft, StationaryObserverDraft, ViewportProjection, ViewportSample,
};
use gravlume_reference::{
    ReferenceInstrument, ReferencePolicy, ReferenceRequest, Termination, TraceInputId,
};
use num_traits::ToPrimitive as _;

use crate::{
    trace::{
        INVARIANT_DRIFT_LIMIT, TraceTermination, TraceUniforms, UnknownTraceTermination,
        production_shader_source,
    },
    trace_test_support::{
        capture_initial_rays, capture_invariant_gate_cases, capture_trace, capture_trace_in_batches,
    },
};

#[test]
fn trace_termination_discriminants_are_stable_and_checked() {
    let cases = [
        (1, TraceTermination::HorizonCrossing),
        (2, TraceTermination::Escape),
        (3, TraceTermination::SingularityGuard),
        (4, TraceTermination::StepExhaustion),
        (5, TraceTermination::NumericalFailure),
        (6, TraceTermination::Uncertain),
    ];

    for (raw, expected) in cases {
        assert_eq!(u32::from(expected), raw);
        assert_eq!(TraceTermination::try_from(raw), Ok(expected));
    }
    assert_eq!(
        TraceTermination::try_from(0),
        Err(UnknownTraceTermination(0))
    );
    assert_eq!(
        TraceTermination::try_from(u32::MAX),
        Err(UnknownTraceTermination(u32::MAX))
    );
}

#[test]
fn trace_uniform_layout_matches_the_shader_abi() {
    assert_eq!(size_of::<TraceUniforms>(), 144);
    assert_eq!(offset_of!(TraceUniforms, spacetime), 0);
    assert_eq!(offset_of!(TraceUniforms, observer_event), 16);
    assert_eq!(offset_of!(TraceUniforms, observer_velocity), 32);
    assert_eq!(offset_of!(TraceUniforms, image_right), 48);
    assert_eq!(offset_of!(TraceUniforms, image_up), 64);
    assert_eq!(offset_of!(TraceUniforms, arrival), 80);
    assert_eq!(offset_of!(TraceUniforms, projection), 96);
    assert_eq!(offset_of!(TraceUniforms, event_surfaces), 112);
    assert_eq!(offset_of!(TraceUniforms, step_policy), 128);
}

#[test]
fn interactive_trace_rejects_non_normalized_observer_frequency() {
    let observation = observation_with(1.0, 0.8, 0.0, [30.0, 0.0, 0.0], 1.0e-6, 1, 1);

    assert!(matches!(
        TraceUniforms::from_observation(&observation),
        Err(crate::TraceInputError::NonNormalizedObserverFrequency { .. })
    ));
}

#[test]
fn interactive_trace_rejects_parameter_state_changed_by_f32_packing() {
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
        Err(crate::TraceInputError::ParameterStateChangedByPacking { .. })
    ));
}

#[test]
fn mass_scale_does_not_change_the_dimensionless_trace_result() {
    let unit = capture_trace(&observation_at_scale(7, 5, 1.0));
    let scaled = capture_trace(&observation_at_scale(7, 5, 8.0));

    assert_eq!(unit.records.len(), scaled.records.len());
    for (unit, scaled) in unit.records.iter().zip(&scaled.records) {
        assert_same_bits(unit.direction_time, scaled.direction_time);
        assert_same_bits(unit.invariant_drift, scaled.invariant_drift);
        assert_eq!(unit.metadata, scaled.metadata);
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
    let origin_direction: [f32; 3] = origin.direction_time[..3]
        .try_into()
        .expect("trace direction contains three components");
    let translated_direction: [f32; 3] = translated.direction_time[..3]
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
    let travel_time_difference = (origin.direction_time[3] - translated.direction_time[3]).abs();
    assert!(
        travel_time_difference <= 1.0e-3,
        "time translation changed travel time from {} to {}",
        origin.direction_time[3],
        translated.direction_time[3]
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
fn production_shader_parses_validates_and_keeps_its_resource_contract() {
    let module = naga::front::wgsl::parse_str(production_shader_source())
        .expect("production trace WGSL parses");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("production trace WGSL validates");

    let mut entry_points = module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect::<Vec<_>>();
    entry_points.sort_unstable();
    assert_eq!(entry_points, ["trace_scene"]);

    let mut bindings = module
        .global_variables
        .iter()
        .filter_map(|(_, variable)| variable.binding.as_ref())
        .map(|binding| (binding.group, binding.binding))
        .collect::<Vec<_>>();
    bindings.sort_unstable();
    assert_eq!(bindings, [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]);
}

#[test]
fn trace_dispatch_writes_every_pixel_across_workgroup_boundaries() {
    let observation = default_observation(9, 9);
    let capture = capture_trace(&observation);

    assert_eq!(capture.records.len(), 81);
    for (index, record) in capture.records.iter().enumerate() {
        TraceTermination::try_from(record.metadata[0])
            .unwrap_or_else(|error| panic!("pixel {index} has no typed termination: {error}"));
        assert_eq!(
            &capture.hdr_pixel(index)[6..],
            &f16_one_bits().to_le_bytes(),
            "pixel {index} was not written"
        );
    }
}

#[test]
fn progressive_dispatch_matches_single_dispatch_across_batch_boundaries() {
    let observation = default_observation(17, 9);
    let single = capture_trace(&observation);
    let progressive = capture_trace_in_batches(&observation, 64);

    assert_eq!(single.records.len(), progressive.records.len());
    for (pixel, (single, progressive)) in
        single.records.iter().zip(&progressive.records).enumerate()
    {
        assert_same_bits(single.direction_time, progressive.direction_time);
        assert_same_bits(single.invariant_drift, progressive.invariant_drift);
        assert_eq!(single.metadata, progressive.metadata, "pixel {pixel}");
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

    for viewport_sample in samples {
        let subpixel = viewport_sample.subpixel().map(|value| {
            value
                .to_f32()
                .expect("normalized subpixel coordinate is representable as f32")
        });
        let capture = capture_initial_rays(&observation, subpixel);
        let [pixel_x, pixel_y] = viewport_sample.pixel();
        let index = usize::try_from(pixel_y * 7 + pixel_x).expect("test index fits usize");
        let gpu = capture.records[index];
        let cpu = observation
            .initial_ray(viewport_sample)
            .expect("validated sample resolves");
        let cpu_sight = cpu.sight_direction_txyz();
        let cpu_direction = [cpu_sight[1], cpu_sight[2], cpu_sight[3]];
        let gpu_direction = [
            f64::from(gpu.direction_time[0]),
            f64::from(gpu.direction_time[1]),
            f64::from(gpu.direction_time[2]),
        ];
        let angle = angle_between(cpu_direction, gpu_direction);

        assert!(
            angle <= 2.0e-6,
            "{viewport_sample:?}: {angle:e} rad, CPU {cpu_direction:?}, GPU {gpu_direction:?}"
        );
        assert!(
            gpu.invariant_drift[0] <= 8.0e-5,
            "{viewport_sample:?}: initial null residual {}",
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
    let instrument = ReferenceInstrument::baseline_v1();

    for viewport_sample in samples {
        let [pixel_x, pixel_y] = viewport_sample.pixel();
        let index = usize::try_from(pixel_y * 7 + pixel_x).expect("test index fits usize");
        let gpu = capture.records[index];
        let gpu_termination =
            TraceTermination::try_from(gpu.metadata[0]).expect("GPU writes a typed termination");
        let reference = instrument
            .trace(
                ReferenceRequest::new(
                    TraceInputId::new(format!("interactive-{pixel_x}-{pixel_y}")),
                    &observation,
                    viewport_sample,
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

        assert_eq!(gpu_termination, expected_termination, "{viewport_sample:?}");
        assert_eq!(gpu.metadata[1], 0, "{viewport_sample:?}: failure flags");
        assert!(gpu.metadata[2] > 0, "{viewport_sample:?}: step counter");

        if let Some(reference_direction) = reference.escape_direction_xyz() {
            let gpu_direction = [
                f64::from(gpu.direction_time[0]),
                f64::from(gpu.direction_time[1]),
                f64::from(gpu.direction_time[2]),
            ];
            let angular_error = angle_between(reference_direction, gpu_direction);
            assert!(
                angular_error <= 3.82e-4,
                "{viewport_sample:?}: escape error {angular_error:e} rad"
            );
        }

        let event_residual = f32::from_bits(gpu.metadata[3]).abs();
        assert!(
            event_residual <= 5.0e-3,
            "{viewport_sample:?}: event residual {event_residual:e}"
        );

        let travel_time_error =
            (f64::from(gpu.direction_time[3]) - reference.travel_time_m()).abs();
        assert!(
            travel_time_error <= 1.0e-3,
            "{viewport_sample:?}: travel-time error {travel_time_error:e}; GPU {}, reference {}; GPU drift {:?}, reference drift {:?}",
            gpu.direction_time[3],
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
                "{viewport_sample:?}: {invariant} drift {drift:e}"
            );
        }
    }
}

#[test]
fn shadow_edge_pair_matches_reference_classification() {
    let observation = default_observation(160, 90);
    let capture = capture_trace(&observation);
    let instrument = ReferenceInstrument::baseline_v1();
    let mut terminations = Vec::new();

    for viewport_sample in [
        sample(&observation, 69, 27, 0.5, 0.5),
        sample(&observation, 70, 27, 0.5, 0.5),
    ] {
        let [pixel_x, pixel_y] = viewport_sample.pixel();
        let index = usize::try_from(pixel_y * 160 + pixel_x).expect("test index fits usize");
        let gpu = TraceTermination::try_from(capture.records[index].metadata[0])
            .expect("GPU writes a typed termination");
        let reference = instrument
            .trace(
                ReferenceRequest::new(
                    TraceInputId::new(format!("shadow-edge-{pixel_x}-{pixel_y}")),
                    &observation,
                    viewport_sample,
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

        assert_eq!(gpu, expected, "{viewport_sample:?}");
        terminations.push(gpu);
    }

    assert_eq!(
        terminations,
        [TraceTermination::Escape, TraceTermination::HorizonCrossing]
    );
}

const fn f16_one_bits() -> u16 {
    0x3c00
}

fn assert_same_bits(left: [f32; 4], right: [f32; 4]) {
    assert_eq!(left.map(f32::to_bits), right.map(f32::to_bits));
}

fn default_observation(width: u32, height: u32) -> Observation {
    observation_at_scale(width, height, 1.0)
}

fn observation_at_coordinate_time(width: u32, height: u32, coordinate_time_m: f64) -> Observation {
    let mass = 1.0;
    let spin = 0.8;
    let spacetime = KerrNewmanSpacetime::new(mass, spin, 0.0, KerrSchildCoordinates::Outgoing)
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
    let spacetime = KerrNewmanSpacetime::new(mass, spin, 0.0, KerrSchildCoordinates::Outgoing)
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
    let observer = StationaryObserverDraft::new(
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
    let scene = PhysicalScene::commit(PhysicalSceneDraft::new(
        mass,
        spin,
        charge,
        KerrSchildCoordinates::Outgoing,
        observer,
    ))
    .expect("fixture scene is valid");
    let projection = ViewportProjection::perspective(
        NonZeroU32::new(extent[0]).expect("test width is nonzero"),
        NonZeroU32::new(extent[1]).expect("test height is nonzero"),
        Angle::from_radians(std::f64::consts::FRAC_PI_4).expect("fixture FOV is finite"),
    )
    .expect("fixture projection is valid");
    Observation::new(scene, projection)
}

fn sample(
    observation: &Observation,
    pixel_x: u32,
    pixel_y: u32,
    subpixel_x: f64,
    subpixel_y: f64,
) -> ViewportSample {
    observation
        .projection()
        .sample(pixel_x, pixel_y, subpixel_x, subpixel_y)
        .expect("test sample is in range")
}

fn angle_between(left: [f64; 3], right: [f64; 3]) -> f64 {
    let left_length = left
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let right_length = right
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let cosine = left
        .into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>()
        / (left_length * right_length);
    cosine.clamp(-1.0, 1.0).acos()
}

use std::{mem::offset_of, num::NonZeroU32};

use gravlume_domain::{
    Angle, KerrNewmanSpacetime, Observation, PhysicalScene, PhysicalSceneDraft,
    StationaryObserverDraft, ViewportProjection, ViewportSample,
};
use gravlume_reference::{
    ReferenceInstrument, ReferencePolicy, ReferenceRequest, Termination, TraceInputId,
};
use num_traits::ToPrimitive as _;

use crate::{
    trace::{
        INVARIANT_DRIFT_LIMIT, TraceTermination, TraceUniforms, UnknownTraceTermination,
        trace_shader_source,
    },
    trace_test_support::{TraceEntryPoint, TraceRecord, render_trace_for_test},
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
fn host_uniform_and_capture_layouts_match_the_gpu_abi() {
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

    assert_eq!(size_of::<TraceRecord>(), 48);
    assert_eq!(offset_of!(TraceRecord, direction_time), 0);
    assert_eq!(offset_of!(TraceRecord, invariant_drift), 16);
    assert_eq!(offset_of!(TraceRecord, metadata), 32);

    assert_pod::<TraceUniforms>();
    assert_pod::<TraceRecord>();
}

#[test]
fn observation_pack_is_dimensionless_and_separates_trace_policies() {
    let observation = default_observation(7, 5);
    let packed = TraceUniforms::from_observation(&observation).expect("default scene packs");

    assert_eq!(
        packed.spacetime.map(f32::to_bits),
        [1.0_f32, 0.8, 0.0, 0.0].map(f32::to_bits)
    );
    assert_eq!(packed.projection[1].to_bits(), 1.0_f32.to_bits());
    assert_eq!(packed.event_surfaces[0].to_bits(), 200.0_f32.to_bits());
    assert_eq!(packed.event_surfaces[1].to_bits(), 0x2b80_0000);
    assert_eq!(packed.event_surfaces[2].to_bits(), 1.6_f32.to_bits());
    let [_, _, sample_x, sample_y] = packed.projection;
    assert_eq!(
        [sample_x.to_bits(), sample_y.to_bits()],
        [0.5_f32; 2].map(f32::to_bits)
    );
    assert_eq!(
        packed.step_policy[3].to_bits(),
        INVARIANT_DRIFT_LIMIT.to_bits()
    );
}

#[test]
fn shader_parses_validates_and_keeps_its_resource_contract() {
    let shader = trace_shader_source();
    let module = naga::front::wgsl::parse_str(&shader).expect("interactive WGSL parses");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("interactive WGSL validates");

    let mut entry_points = module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect::<Vec<_>>();
    entry_points.sort_unstable();
    assert_eq!(entry_points, ["trace_scene", "write_initial_rays"]);

    let mut bindings = module
        .global_variables
        .iter()
        .filter_map(|(_, variable)| variable.binding.as_ref())
        .map(|binding| (binding.group, binding.binding))
        .collect::<Vec<_>>();
    bindings.sort_unstable();
    assert_eq!(bindings, [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]);
}

#[test]
fn trace_dispatch_writes_every_pixel_across_workgroup_boundaries() {
    let observation = default_observation(9, 9);
    let capture = render_trace_for_test(&observation, TraceEntryPoint::Trace);

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
        let capture = render_trace_for_test(&observation, TraceEntryPoint::InitialRay { subpixel });
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
    let capture = render_trace_for_test(&observation, TraceEntryPoint::Trace);
    let samples = [
        sample(&observation, 0, 0, 0.5, 0.5),
        sample(&observation, 6, 0, 0.5, 0.5),
        sample(&observation, 0, 4, 0.5, 0.5),
        sample(&observation, 6, 4, 0.5, 0.5),
        sample(&observation, 3, 0, 0.5, 0.5),
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
fn sky_horizon_and_failure_states_have_distinct_visible_outputs() {
    let observation = default_observation(7, 5);
    let capture = render_trace_for_test(&observation, TraceEntryPoint::Trace);
    let mut horizon = None;
    let mut escape = None;
    let mut diagnostic = None;

    for (index, record) in capture.records.iter().enumerate() {
        match TraceTermination::try_from(record.metadata[0]).expect("typed GPU terminal") {
            TraceTermination::HorizonCrossing => horizon = Some(index),
            TraceTermination::Escape => escape = Some(index),
            TraceTermination::SingularityGuard
            | TraceTermination::StepExhaustion
            | TraceTermination::NumericalFailure
            | TraceTermination::Uncertain => diagnostic = Some(index),
        }
    }

    let terminations = capture
        .records
        .iter()
        .map(|record| {
            (
                record.metadata[0],
                record.metadata[1],
                record.invariant_drift,
            )
        })
        .collect::<Vec<_>>();
    let horizon = capture.hdr_pixel(
        horizon.unwrap_or_else(|| panic!("matrix contains a horizon ray: {terminations:?}")),
    );
    let escape = capture.hdr_pixel(escape.expect("matrix contains an escape ray"));
    let diagnostic =
        capture.hdr_pixel(diagnostic.expect("matrix contains a visible diagnostic ray"));
    assert_eq!(&horizon[..6], &[0; 6], "horizon is physically black");
    assert_ne!(&escape[..6], &[0; 6], "analytic sky remains visible");
    assert_ne!(&diagnostic[..6], &[0; 6], "failure is not silent black");
    assert_ne!(diagnostic, escape, "failure and analytic sky stay distinct");
}

const fn size_of<T>() -> usize {
    std::mem::size_of::<T>()
}

fn assert_pod<T: bytemuck::Pod + bytemuck::Zeroable>() {}

const fn f16_one_bits() -> u16 {
    0x3c00
}

fn default_observation(width: u32, height: u32) -> Observation {
    let spacetime = KerrNewmanSpacetime::new(1.0, 0.8, 0.0).expect("fixture spacetime is valid");
    let observer_xyz = spacetime.oblate_to_cartesian(30.0, std::f64::consts::FRAC_PI_3, 0.0);
    let observer = StationaryObserverDraft::new(
        [0.0, observer_xyz[0], observer_xyz[1], observer_xyz[2]],
        [0.0; 4],
        [0.0, 0.0, 1.0],
        1.0,
    );
    let scene = PhysicalScene::commit(PhysicalSceneDraft::new(1.0, 0.8, 0.0, observer))
        .expect("fixture scene is valid");
    let projection = ViewportProjection::perspective(
        NonZeroU32::new(width).expect("test width is nonzero"),
        NonZeroU32::new(height).expect("test height is nonzero"),
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

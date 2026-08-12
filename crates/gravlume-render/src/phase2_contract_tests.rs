use std::{mem::offset_of, num::NonZeroU32};

use gravlume_domain::{
    Angle, KerrNewmanSpacetime, Observation, PhysicalScene, PhysicalSceneDraft,
    StationaryObserverDraft, ViewportProjection,
};

use crate::trace::{
    TRACE_RECORD_SIZE, TRACE_SHADER, TraceRecord, TraceTermination, TraceUniforms,
    UnknownTraceTermination,
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
fn host_struct_offsets_match_the_wgsl_abi() {
    assert_eq!(size_of::<TraceUniforms>(), 144);
    assert_eq!(offset_of!(TraceUniforms, spacetime), 0);
    assert_eq!(offset_of!(TraceUniforms, observer_event), 16);
    assert_eq!(offset_of!(TraceUniforms, observer_velocity), 32);
    assert_eq!(offset_of!(TraceUniforms, image_right), 48);
    assert_eq!(offset_of!(TraceUniforms, image_up), 64);
    assert_eq!(offset_of!(TraceUniforms, arrival), 80);
    assert_eq!(offset_of!(TraceUniforms, projection_policy), 96);
    assert_eq!(offset_of!(TraceUniforms, integration), 112);
    assert_eq!(offset_of!(TraceUniforms, viewport), 128);

    assert_eq!(size_of::<TraceRecord>(), TRACE_RECORD_SIZE as usize);
    assert_eq!(offset_of!(TraceRecord, direction_time), 0);
    assert_eq!(offset_of!(TraceRecord, invariant_drift), 16);
    assert_eq!(offset_of!(TraceRecord, metadata), 32);

    fn assert_pod<T: bytemuck::Pod + bytemuck::Zeroable>() {}
    assert_pod::<TraceUniforms>();
    assert_pod::<TraceRecord>();
}

#[test]
fn observation_pack_is_dimensionless_and_retains_the_viewport_contract() {
    let observation = default_observation(7, 5);
    let packed = TraceUniforms::from_observation(&observation).expect("default scene packs");

    assert_eq!(packed.spacetime, [1.0, 0.8, 0.0, 1.6]);
    assert_eq!(packed.viewport[..2], [7, 5]);
    assert_eq!(packed.projection_policy[1], 200.0);
    assert_eq!(packed.projection_policy[2], f32::from_bits(0x2b80_0000));
}

#[test]
fn phase2_shader_parses_validates_and_keeps_its_resource_contract() {
    let module = naga::front::wgsl::parse_str(TRACE_SHADER).expect("Phase 2 WGSL parses");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("Phase 2 WGSL validates");

    let mut entry_points = module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect::<Vec<_>>();
    entry_points.sort_unstable();
    assert_eq!(entry_points, ["initial_ray_contract", "trace_scene"]);

    let mut bindings = module
        .global_variables
        .iter()
        .filter_map(|(_, variable)| variable.binding.as_ref())
        .map(|binding| (binding.group, binding.binding))
        .collect::<Vec<_>>();
    bindings.sort_unstable();
    assert_eq!(bindings, [(0, 0), (0, 1), (0, 2)]);
}

const fn size_of<T>() -> usize {
    std::mem::size_of::<T>()
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

// Bounded on-demand single-sample evidence. Every host-shared lane is one aligned vec4; no vec3,
// bool, implicit padding, trajectory, or extent-scaled record survives the invocation.

struct SampleInspectionRequest {
    pixel_extent: vec4<u32>,
    subpixel: vec4<f32>,
}

struct SampleInspectionRecord {
    // (termination, failure flags, steps, event candidates)
    metadata: vec4<u32>,
    // Exact only for a determinate, non-numerical-failure termination. All zeroes mean
    // unavailable for a numerical failure; Uncertain may carry provisional counters that must
    // not be decoded as an exact branch.
    branch_key: vec4<u32>,
    source_time: vec4<f32>,
    // Full scene-linear RGBA retains the production output tag.
    scene_value: vec4<f32>,
    // (event residual, reserved, reserved, reserved)
    event_diagnostics: vec4<f32>,
    maximum_invariant_drift: vec4<f32>,
}

@group(0) @binding(3)
var<storage, read_write> inspected_sample: SampleInspectionRecord;

@group(0) @binding(9)
var<uniform> inspection_request: SampleInspectionRequest;

fn store_inspected_sample(sample: GeometricSample, value: vec4<f32>) {
    inspected_sample.metadata = vec4<u32>(
        sample.termination,
        sample.flags,
        sample.steps,
        sample.event_candidates,
    );
    inspected_sample.branch_key = sample.branch_key;
    inspected_sample.source_time = vec4<f32>(
        sample.source_coordinates,
        sample.travel_time,
    );
    inspected_sample.scene_value = value;
    inspected_sample.event_diagnostics = vec4<f32>(sample.event_residual, 0.0, 0.0, 0.0);
    inspected_sample.maximum_invariant_drift = sample.maximum_drift;
}

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn inspect_sample(@builtin(local_invocation_index) local_index: u32) {
    // A 1x1 workgroup produced incomplete records on the accepted Metal evidence platform. Keep
    // the proven specialization, but return every inactive lane before constructing ray state.
    if local_index != 0u {
        return;
    }
    let pixel = inspection_request.pixel_extent.xy;
    let extent = inspection_request.pixel_extent.zw;
    let sample = trace_pixel_at(pixel, extent, inspection_request.subpixel.xy);
    store_inspected_sample(sample, inspected_scene_value(sample));
}

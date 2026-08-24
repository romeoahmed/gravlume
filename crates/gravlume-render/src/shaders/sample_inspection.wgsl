// Bounded ordered sample evidence. Runtime-sized storage arrays let production bind one request
// and tests bind a sparse corpus without a second protocol or shader entry point. Every active
// invocation owns one request/record pair, so no synchronization is required.
// Source: https://www.w3.org/TR/WGSL/#buffer-binding-determines-runtime-sized-array-element-count

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
var<storage, read_write> inspection_records: array<SampleInspectionRecord>;

@group(0) @binding(9)
var<storage, read> inspection_requests: array<SampleInspectionRequest>;

fn encode_inspected_sample(sample: GeometricSample, value: vec4<f32>) -> SampleInspectionRecord {
    return SampleInspectionRecord(
        vec4<u32>(
            sample.termination,
            sample.flags,
            sample.steps,
            sample.event_candidates,
        ),
        sample.branch_key,
        vec4<f32>(sample.source_coordinates, sample.travel_time),
        value,
        vec4<f32>(sample.event_residual, 0.0, 0.0, 0.0),
        sample.maximum_drift,
    );
}

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn inspect_samples(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let invocations_per_workgroup = TRACE_WORKGROUP_AXIS * TRACE_WORKGROUP_AXIS;
    let index = workgroup_id.x * invocations_per_workgroup + local_index;
    if index >= arrayLength(&inspection_requests) {
        return;
    }

    let request = inspection_requests[index];
    let sample = trace_pixel_at(
        request.pixel_extent.xy,
        request.pixel_extent.zw,
        request.subpixel.xy,
    );
    inspection_records[index] = encode_inspected_sample(sample, inspected_scene_value(sample));
}

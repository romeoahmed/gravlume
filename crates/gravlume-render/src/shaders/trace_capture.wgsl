// Test-only per-pixel scientific capture. Production stores only the scene-linear preview.

@group(0) @binding(3)
var<storage, read_write> trace_source_time: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> trace_invariant_drift: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read_write> trace_metadata: array<vec4<u32>>;

@group(0) @binding(6)
var<storage, read_write> trace_event: array<vec4<u32>>;

fn store_trace_record(index: u32, result: GeometricSample) {
    trace_source_time[index] = vec4<f32>(result.source_coordinates, result.travel_time);
    trace_invariant_drift[index] = result.maximum_drift;
    trace_metadata[index] = vec4<u32>(
        result.termination,
        result.flags,
        result.steps,
        bitcast<u32>(result.event_residual),
    );
    let event_ambiguous = select(0u, 1u, countOneBits(result.event_candidates) > 1u);
    let packed_branch_counts = min(result.branch_key.x, 0xffffu)
        | (min(result.branch_key.y, 0xffffu) << 16u);
    trace_event[index] = vec4<u32>(
        result.event_candidates,
        event_ambiguous,
        packed_branch_counts,
        result.branch_key.z,
    );
}

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn capture_trace_scene(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_origin * TRACE_WORKGROUP_AXIS + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    let index = pixel.y * extent.x + pixel.x;
    let result = trace_pixel(pixel, extent);
    store_trace_record(index, result);
    store_scene_result(pixel, result.termination, result.source_coordinates);
}

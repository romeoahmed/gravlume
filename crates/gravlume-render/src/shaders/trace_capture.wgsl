// Test-only per-pixel scientific capture. Production stores only the scene-linear preview.

@group(0) @binding(3)
var<storage, read_write> trace_direction_time: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> trace_invariant_drift: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read_write> trace_metadata: array<vec4<u32>>;

fn trace_pixel_at(
    pixel: vec2<u32>,
    extent: vec2<u32>,
    subpixel: vec2<f32>,
) -> TraceResult {
    let initial = initial_state_at(pixel, extent, subpixel);
    if initial.rhs.flags != 0u {
        return failure_result(initial.rhs.flags, 0u, 0.0, vec4<f32>(0.0));
    }
    let initial_invariants = invariants_from_geometry_rhs(
        initial.state,
        initial.energy,
        initial.geometry,
        initial.rhs,
    );
    if initial_invariants.flags != 0u {
        return failure_result(initial_invariants.flags, 0u, 0.0, vec4<f32>(0.0));
    }
    return trace_initialized(initial, initial_invariants);
}

fn trace_pixel(pixel: vec2<u32>, extent: vec2<u32>) -> TraceResult {
    return trace_pixel_at(pixel, extent, trace_uniforms.camera.zw);
}

fn store_trace_record(index: u32, result: TraceResult) {
    trace_direction_time[index] = vec4<f32>(result.direction, result.travel_time);
    trace_invariant_drift[index] = result.maximum_drift;
    trace_metadata[index] = vec4<u32>(
        result.termination,
        result.flags,
        result.steps,
        bitcast<u32>(result.event_residual),
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
    store_scene_result(pixel, result.termination, result.direction);
}

@group(0) @binding(3)
var<storage, read_write> trace_direction_time: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> trace_invariant_drift: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read_write> trace_metadata: array<vec4<u32>>;

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

@compute @workgroup_size(8, 8, 1)
fn capture_trace_scene(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_region.xy * vec2<u32>(8u, 8u) + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    let index = pixel.y * extent.x + pixel.x;
    let result = trace_pixel(pixel, extent);
    store_trace_record(index, result);
    store_scene_result(pixel, result.termination, result.direction);
}

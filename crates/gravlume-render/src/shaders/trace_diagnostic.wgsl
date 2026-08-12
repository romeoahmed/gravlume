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
    let local_index = global_id.y * 8u + global_id.x;
    let index = trace_dispatch.pixels.x + local_index;
    if index >= extent.x * extent.y {
        return;
    }
    let pixel = vec2<u32>(index % extent.x, index / extent.x);
    let result = trace_pixel(pixel, extent);
    store_trace_record(index, result);
    store_scene_result(pixel, result.termination, result.direction);
}

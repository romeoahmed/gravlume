// Test-only serialization of the logical surface sample.

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn capture_surface_trace_scene(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_origin * TRACE_WORKGROUP_AXIS + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    let sample = trace_pixel(pixel, extent);
    store_trace_record(pixel.y * extent.x + pixel.x, sample);
    store_surface_scene_result(pixel, sample);
}

// Test-only capture of the conservative accelerator and its exact Kerr-Schild fallbacks.

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn capture_accelerated_trace_scene(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let extent = textureDimensions(scene_hdr);
    let tile = trace_dispatch.tile_origin + workgroup_id.xy;
    let pixel = tile * ESCAPE_MAP_TILE_AXIS + local_id.xy;
    let result = escape_map_result(local_id.xy, workgroup_id.xy, extent);
    if any(pixel >= extent) {
        return;
    }
    store_trace_record(pixel.y * extent.x + pixel.x, result);
    store_scene_result(pixel, result.termination, result.source_coordinates);
}

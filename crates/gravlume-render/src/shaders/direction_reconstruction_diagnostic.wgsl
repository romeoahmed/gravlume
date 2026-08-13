@compute @workgroup_size(8, 8, 1)
fn capture_direction_reconstruction_trace_scene(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let extent = textureDimensions(scene_hdr);
    let tile = trace_dispatch.tile_region.xy + workgroup_id.xy;
    let pixel = tile * DIRECTION_RECONSTRUCTION_TILE_AXIS + local_id.xy;
    let result = direction_reconstruction_result(local_id.xy, workgroup_id.xy, extent);
    if any(pixel >= extent) {
        return;
    }
    store_trace_record(pixel.y * extent.x + pixel.x, result);
    store_scene_result(pixel, result.termination, result.direction);
}

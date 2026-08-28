// Full Kerr--Schild trace for the analytic-sky presentation plan.

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn trace_analytic_sky_scene(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_origin * TRACE_WORKGROUP_AXIS + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    let result = trace_pixel(pixel, extent);
    store_scene_result(pixel, result.termination, result.source_coordinates);
}

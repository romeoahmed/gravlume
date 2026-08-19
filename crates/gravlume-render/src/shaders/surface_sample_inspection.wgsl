@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn inspect_surface_sample(@builtin(local_invocation_index) local_index: u32) {
    // Match the correctness-approved presentation specialization. Only this lane enters the
    // sequential solver; every other lane returns before creating ray state.
    if local_index != 0u {
        return;
    }
    let pixel = inspection_request.pixel_extent.xy;
    let extent = inspection_request.pixel_extent.zw;
    let sample = trace_pixel_at(pixel, extent, inspection_request.subpixel.xy);
    store_inspected_geometry(sample);
    let value = surface_scene_value(sample);
    store_inspected_scene(value);
}

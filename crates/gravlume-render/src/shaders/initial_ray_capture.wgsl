@compute @workgroup_size(8, 8, 1)
fn write_initial_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let local_index = global_id.y * 8u + global_id.x;
    let index = trace_dispatch.pixels.x + local_index;
    if index >= extent.x * extent.y {
        return;
    }
    let pixel = vec2<u32>(index % extent.x, index / extent.x);

    let initial = initial_state_at(pixel, extent, trace_uniforms.projection.zw);
    if initial.flags != 0u {
        store_failure(index, pixel, initial.flags);
        return;
    }
    trace_direction_time[index] = vec4<f32>(
        initial.sight.yzw,
        trace_uniforms.projection.y,
    );
    trace_invariant_drift[index] = vec4<f32>(initial.null_residual, 0.0, 0.0, 0.0);
    trace_metadata[index] = vec4<u32>(0u);
    textureStore(scene_hdr, vec2<i32>(pixel), vec4<f32>(0.0, 0.0, 0.0, 1.0));
}

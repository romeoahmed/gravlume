@compute @workgroup_size(8, 8, 1)
fn write_initial_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    if !inside_extent(global_id.xy, extent) {
        return;
    }

    let index = record_index(global_id.xy, extent);
    let initial = initial_state_at(global_id.xy, extent, trace_uniforms.projection.zw);
    if initial.flags != 0u {
        store_failure(index, global_id.xy, initial.flags);
        return;
    }
    trace_direction_time[index] = vec4<f32>(
        initial.sight.yzw,
        trace_uniforms.projection.y,
    );
    trace_invariant_drift[index] = vec4<f32>(initial.null_residual, 0.0, 0.0, 0.0);
    trace_metadata[index] = vec4<u32>(0u);
    textureStore(scene_hdr, vec2<i32>(global_id.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
}

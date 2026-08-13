@compute @workgroup_size(8, 8, 1)
fn write_initial_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_region.xy * vec2<u32>(8u, 8u) + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    let index = pixel.y * extent.x + pixel.x;

    let initial = initial_state_at(pixel, extent, trace_uniforms.view.zw);
    if initial.rhs.flags != 0u {
        store_trace_record(index, failure_result(initial.rhs.flags, 0u, 0.0, vec4<f32>(0.0)));
        store_scene_result(pixel, TERMINATION_NUMERICAL_FAILURE, vec3<f32>(0.0));
        return;
    }
    let sight = sight_direction(pixel, extent, trace_uniforms.view.zw);
    let arrival = -sight;
    let momentum_contravariant = trace_uniforms.view.y
        * (trace_uniforms.observer_velocity + arrival);
    let null_residual = normalized_null_residual(initial.state.momentum, momentum_contravariant);
    trace_direction_time[index] = vec4<f32>(
        sight.yzw,
        trace_uniforms.view.y,
    );
    trace_invariant_drift[index] = vec4<f32>(null_residual, 0.0, 0.0, 0.0);
    trace_metadata[index] = vec4<u32>(0u);
    textureStore(scene_hdr, vec2<i32>(pixel), vec4<f32>(0.0, 0.0, 0.0, 1.0));
}

fn normalized_null_residual(
    momentum_covariant: vec4<f32>,
    momentum_contravariant: vec4<f32>,
) -> f32 {
    let contraction = dot(momentum_covariant, momentum_contravariant);
    let term_norm = dot(abs(momentum_covariant), abs(momentum_contravariant));
    return abs(contraction) / max(1.0, term_norm);
}

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn write_initial_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_origin * TRACE_WORKGROUP_AXIS + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    let index = pixel.y * extent.x + pixel.x;

    let initial = initial_state_at(pixel, extent, trace_uniforms.camera.zw);
    if initial.rhs.flags != 0u {
        store_trace_record(index, failure_result(initial.rhs.flags, 0u, 0.0, vec4<f32>(0.0)));
        store_scene_result(pixel, TERMINATION_NUMERICAL_FAILURE, vec3<f32>(0.0));
        return;
    }
    let sight = sight_direction(pixel, extent, trace_uniforms.camera.zw);
    let arrival = -sight;
    let momentum_contravariant = trace_uniforms.camera.y
        * (trace_uniforms.observer_velocity + arrival);
    let momentum_covariant = vec4<f32>(-initial.energy, initial.state.momentum);
    let null_residual = normalized_null_residual(momentum_covariant, momentum_contravariant);
    trace_direction_time[index] = vec4<f32>(
        sight.yzw,
        trace_uniforms.camera.y,
    );
    trace_invariant_drift[index] = vec4<f32>(null_residual, 0.0, 0.0, 0.0);
    trace_metadata[index] = vec4<u32>(0u);
    textureStore(scene_hdr, vec2<i32>(pixel), vec4<f32>(0.0, 0.0, 0.0, 1.0));
}

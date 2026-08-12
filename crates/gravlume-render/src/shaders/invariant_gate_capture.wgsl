@compute @workgroup_size(8, 8, 1)
fn write_invariant_gate_cases(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    if !inside_extent(global_id.xy, extent) {
        return;
    }

    let index = record_index(global_id.xy, extent);
    var drift = vec4<f32>(0.0);
    drift[global_id.x] = trace_uniforms.step_policy.w + 0.01;
    let termination = select(
        TERMINATION_ESCAPE,
        TERMINATION_UNCERTAIN,
        invariant_budget_exceeded(drift),
    );
    store_trace_result(
        index,
        global_id.xy,
        termination,
        0u,
        1u,
        0.0,
        vec3<f32>(1.0, 0.0, 0.0),
        1.0,
        drift,
    );
}

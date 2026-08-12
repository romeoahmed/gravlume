@compute @workgroup_size(8, 8, 1)
fn write_invariant_gate_cases(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let local_index = global_id.y * 8u + global_id.x;
    let index = trace_dispatch.pixels.x + local_index;
    if index >= extent.x * extent.y {
        return;
    }
    let pixel = vec2<u32>(index % extent.x, index / extent.x);

    var drift = vec4<f32>(0.0);
    drift[pixel.x] = trace_uniforms.step_policy.w + 0.01;
    let termination = select(
        TERMINATION_ESCAPE,
        TERMINATION_UNCERTAIN,
        invariant_budget_exceeded(drift),
    );
    store_trace_result(
        index,
        pixel,
        termination,
        0u,
        1u,
        0.0,
        vec3<f32>(1.0, 0.0, 0.0),
        1.0,
        drift,
    );
}

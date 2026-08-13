@compute @workgroup_size(8, 8, 1)
fn write_invariant_gate_cases(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_region.xy * vec2<u32>(8u, 8u) + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    let index = pixel.y * extent.x + pixel.x;

    var drift = vec4<f32>(0.0);
    drift[pixel.x] = trace_uniforms.step_policy.w + 0.01;
    let termination = select(
        TERMINATION_ESCAPE,
        TERMINATION_UNCERTAIN,
        invariant_budget_exceeded(drift),
    );
    let result = TraceResult(
        termination,
        0u,
        1u,
        0.0,
        vec3<f32>(1.0, 0.0, 0.0),
        1.0,
        drift,
    );
    store_trace_record(index, result);
    store_scene_result(pixel, result.termination, result.direction);
}

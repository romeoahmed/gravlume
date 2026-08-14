// Test-only synthetic cases for the affine event-selection contract.

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn write_event_policy_cases(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_origin * TRACE_WORKGROUP_AXIS + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    let index = pixel.y * extent.x + pixel.x;
    if pixel.x == 3u {
        let step_magnitude = 1.0;
        let offset = (0.75 * trace_uniforms.event_surfaces.w) / step_magnitude;
        var events = no_event_candidates();
        events[EVENT_INDEX_HORIZON] = 0.5;
        events[EVENT_INDEX_SURFACE] = 0.5 - offset;
        events[EVENT_INDEX_ESCAPE] = 0.5 + offset;
        let selected = select_earliest_event(events, step_magnitude);
        trace_metadata[index] = vec4<u32>(selected.termination, 0u, 0u, 0u);
        trace_event[index] = vec4<u32>(selected.candidates, selected.ambiguous, 0u, 0u);
        textureStore(scene_hdr, vec2<i32>(pixel), vec4<f32>(0.0));
        return;
    }
    if pixel.x >= 4u {
        var armed = pixel.x == 7u;
        var surface_value = 0.0;
        if pixel.x == 5u {
            surface_value = 0.5 * trace_uniforms.surface_emitter.w;
        } else if pixel.x == 6u {
            surface_value = 2.0 * trace_uniforms.surface_emitter.w;
        }
        armed = update_surface_event_arming(armed, surface_value);
        trace_metadata[index] = vec4<u32>(select(0u, 1u, armed), 0u, 0u, 0u);
        trace_event[index] = vec4<u32>(0u);
        textureStore(scene_hdr, vec2<i32>(pixel), vec4<f32>(0.0));
        return;
    }
    var surface_fraction = 0.5;
    if pixel.x < 2u {
        let step_magnitude = select(
            trace_uniforms.step_policy.z,
            trace_uniforms.step_policy.y,
            pixel.x == 0u,
        );
        surface_fraction += (0.5 * trace_uniforms.event_surfaces.w)
            / step_magnitude;
    }
    let step_magnitude = select(
        trace_uniforms.step_policy.z,
        trace_uniforms.step_policy.y,
        pixel.x == 0u,
    );
    var events = no_event_candidates();
    events[EVENT_INDEX_SURFACE] = surface_fraction;
    events[EVENT_INDEX_ESCAPE] = 0.5;
    let selected = select_earliest_event(events, step_magnitude);
    trace_metadata[index] = vec4<u32>(selected.termination, 0u, 0u, 0u);
    trace_event[index] = vec4<u32>(selected.candidates, selected.ambiguous, 0u, 0u);
    textureStore(scene_hdr, vec2<i32>(pixel), vec4<f32>(0.0));
}

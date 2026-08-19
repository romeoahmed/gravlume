// Immediate bolometric transport for the neutral equatorial source. GeometricSample remains
// invocation-local; production persists only the scene-linear RGBA16F target.

fn store_surface_scene_result(pixel: vec2<u32>, sample: GeometricSample) {
    if sample.termination != TERMINATION_EQUATORIAL_SURFACE {
        store_scene_result(pixel, sample.termination, sample.source_coordinates);
        return;
    }
    let intensity = transported_surface_intensity(sample);
    if intensity < 0.0 {
        textureStore(
            scene_hdr,
            vec2<i32>(pixel),
            vec4<f32>(
                visible_failure_color(TERMINATION_NUMERICAL_FAILURE),
                -f32(TERMINATION_NUMERICAL_FAILURE),
            ),
        );
        return;
    }
    textureStore(
        scene_hdr,
        vec2<i32>(pixel),
        vec4<f32>(vec3<f32>(intensity), SURFACE_RADIANCE_TAG),
    );
}

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn trace_bolometric_surface_scene(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_origin * TRACE_WORKGROUP_AXIS + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    store_surface_scene_result(pixel, trace_pixel(pixel, extent));
}

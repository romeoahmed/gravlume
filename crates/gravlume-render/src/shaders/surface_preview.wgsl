// Immediate bolometric transport for the baseline thin equatorial source. GeometricSample remains
// invocation-local; production persists only the existing scene-linear RGBA16F target.

const MAXIMUM_RGBA16_FLOAT: f32 = 65504.0;

fn transported_surface_intensity(sample: GeometricSample) -> f32 {
    let radius_ratio = sample.source_coordinates.x / 6.0;
    let frequency_ratio = sample.source_coordinates.z;
    if radius_ratio <= 0.0 || frequency_ratio <= 0.0 {
        return -1.0;
    }
    var intensity = trace_uniforms.surface_emitter.z;
    for (var division = 0u; division < 3u; division += 1u) {
        if radius_ratio < 1.0 && intensity > MAXIMUM_FINITE_F32 * radius_ratio {
            return -1.0;
        }
        intensity /= radius_ratio;
    }
    for (var transport = 0u; transport < 4u; transport += 1u) {
        if frequency_ratio > 1.0 {
            if intensity > MAXIMUM_RGBA16_FLOAT / frequency_ratio {
                return -1.0;
            }
        }
        intensity *= frequency_ratio;
    }
    if intensity < 0.0 || intensity > MAXIMUM_RGBA16_FLOAT || !finite_scalar(intensity) {
        return -1.0;
    }
    return intensity;
}

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
    textureStore(scene_hdr, vec2<i32>(pixel), vec4<f32>(vec3<f32>(intensity), 1.0));
}

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn trace_surface_scene(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_origin * TRACE_WORKGROUP_AXIS + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    store_surface_scene_result(pixel, trace_pixel(pixel, extent));
}

// Versioned three-band blackbody transport. Log2 fractions preserve scale until the complete
// radiance product is known. The vec4 storage stride is uniformly 16 bytes on every backend.

const BLACKBODY_LUT_LAST_INDEX: u32 = 4096u;
const BLACKBODY_LUT_INTERVALS_PER_OCTAVE: f32 = 128.0;
const BLACKBODY_LUT_MIN_LOG2_TEMPERATURE: f32 = -8.0;

@group(0) @binding(8)
var<storage, read> blackbody_log2_fraction_lut: array<vec4<f32>, 4097>;

fn blackbody_band_log2_fractions(temperature_kelvin: f32) -> vec3<f32> {
    if temperature_kelvin <= 0.0 || !finite_scalar(temperature_kelvin) {
        return vec3<f32>(1.0);
    }
    let coordinate = (log2(temperature_kelvin) - BLACKBODY_LUT_MIN_LOG2_TEMPERATURE)
        * BLACKBODY_LUT_INTERVALS_PER_OCTAVE;
    if coordinate < 0.0 || coordinate > f32(BLACKBODY_LUT_LAST_INDEX) {
        return vec3<f32>(1.0);
    }
    let lower_index = min(u32(floor(coordinate)), BLACKBODY_LUT_LAST_INDEX - 1u);
    let weight = coordinate - f32(lower_index);
    let lower = blackbody_log2_fraction_lut[lower_index];
    let upper = blackbody_log2_fraction_lut[lower_index + 1u];
    let log2_fractions = mix(lower, upper, vec4<f32>(weight)).xyz;
    if any(log2_fractions > vec3<f32>(0.0))
        || !finite_vec3(log2_fractions)
    {
        return vec3<f32>(1.0);
    }
    return log2_fractions;
}

fn transported_surface_bands(sample: GeometricSample) -> vec3<f32> {
    let radius_ratio = sample.source_coordinates.x / 6.0;
    let frequency_ratio = sample.source_coordinates.z;
    if radius_ratio <= 0.0 || frequency_ratio <= 0.0 {
        return vec3<f32>(-1.0);
    }
    let emitted_temperature = trace_uniforms.surface_transport.x
        / sqrt(radius_ratio * sqrt(radius_ratio));
    let observed_temperature = emitted_temperature * frequency_ratio;
    let incoming_log2_fractions = blackbody_band_log2_fractions(observed_temperature);
    if any(incoming_log2_fractions > vec3<f32>(0.0)) {
        return vec3<f32>(-1.0);
    }
    var source_bands = vec3<f32>(0.0);
    if trace_uniforms.surface_transport.w > 0.0 {
        let source_log2_fractions = blackbody_band_log2_fractions(
            trace_uniforms.surface_transport.y,
        );
        if any(source_log2_fractions > vec3<f32>(0.0)) {
            return vec3<f32>(-1.0);
        }
        source_bands = scaled_spectral_intensities(
            trace_uniforms.surface_transport.w,
            source_log2_fractions,
        );
        if any(source_bands < vec3<f32>(0.0)) {
            return vec3<f32>(-1.0);
        }
    }
    let incoming = attenuated_surface_spectrum(
        radius_ratio,
        frequency_ratio,
        incoming_log2_fractions,
    );
    if any(incoming < vec3<f32>(0.0)) {
        return vec3<f32>(-1.0);
    }
    return bounded_surface_sums(incoming, source_bands);
}

fn surface_scene_value(sample: GeometricSample) -> vec4<f32> {
    if sample.termination != TERMINATION_EQUATORIAL_SURFACE {
        return scene_value(sample.termination, sample.source_coordinates);
    }
    let bands = transported_surface_bands(sample);
    if any(bands < vec3<f32>(0.0)) {
        return vec4<f32>(
            visible_failure_color(TERMINATION_NUMERICAL_FAILURE),
            -f32(TERMINATION_NUMERICAL_FAILURE),
        );
    }
    return vec4<f32>(bands, SURFACE_RADIANCE_TAG);
}

fn store_surface_scene_result(pixel: vec2<u32>, sample: GeometricSample) {
    textureStore(scene_hdr, vec2<i32>(pixel), surface_scene_value(sample));
}

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn trace_blackbody_surface_scene(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_origin * TRACE_WORKGROUP_AXIS + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    store_surface_scene_result(pixel, trace_pixel(pixel, extent));
}

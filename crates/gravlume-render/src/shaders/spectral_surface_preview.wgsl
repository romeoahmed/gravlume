// Versioned three-band blackbody transport. The read-only LUT is a fixed array of vec4<f32>, whose
// uniform 16-byte stride follows the WGSL storage-layout contract on every backend.

const BLACKBODY_LUT_LAST_INDEX: u32 = 4096u;
const BLACKBODY_LUT_INTERVALS_PER_OCTAVE: f32 = 128.0;
const BLACKBODY_LUT_MIN_LOG2_TEMPERATURE: f32 = -8.0;

@group(0) @binding(8)
var<storage, read> blackbody_spectral_lut: array<vec4<f32>, 4097>;

fn blackbody_band_fractions(temperature_kelvin: f32) -> vec3<f32> {
    if temperature_kelvin <= 0.0 || !finite_scalar(temperature_kelvin) {
        return vec3<f32>(-1.0);
    }
    let coordinate = (log2(temperature_kelvin) - BLACKBODY_LUT_MIN_LOG2_TEMPERATURE)
        * BLACKBODY_LUT_INTERVALS_PER_OCTAVE;
    if coordinate < 0.0 || coordinate > f32(BLACKBODY_LUT_LAST_INDEX) {
        return vec3<f32>(-1.0);
    }
    let lower_index = min(u32(floor(coordinate)), BLACKBODY_LUT_LAST_INDEX - 1u);
    let fraction = coordinate - f32(lower_index);
    let lower = blackbody_spectral_lut[lower_index];
    let upper = blackbody_spectral_lut[lower_index + 1u];
    let bands = mix(lower, upper, vec4<f32>(fraction)).xyz;
    if any(bands < vec3<f32>(0.0))
        || any(bands > vec3<f32>(1.0))
        || !finite_vec3(bands)
    {
        return vec3<f32>(-1.0);
    }
    return bands;
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
    let incoming_fractions = blackbody_band_fractions(observed_temperature);
    if any(incoming_fractions < vec3<f32>(0.0)) {
        return vec3<f32>(-1.0);
    }
    var source_bands = vec3<f32>(0.0);
    if trace_uniforms.surface_transport.w > 0.0 {
        let source_fractions = blackbody_band_fractions(trace_uniforms.surface_transport.y);
        if any(source_fractions < vec3<f32>(0.0)) {
            return vec3<f32>(-1.0);
        }
        source_bands = source_fractions * trace_uniforms.surface_transport.w;
    }
    let incoming = vec3<f32>(
        attenuated_surface_intensity(sample, incoming_fractions.x),
        attenuated_surface_intensity(sample, incoming_fractions.y),
        attenuated_surface_intensity(sample, incoming_fractions.z),
    );
    if any(incoming < vec3<f32>(0.0)) {
        return vec3<f32>(-1.0);
    }
    return vec3<f32>(
        bounded_surface_sum(incoming.x, source_bands.x),
        bounded_surface_sum(incoming.y, source_bands.y),
        bounded_surface_sum(incoming.z, source_bands.z),
    );
}

fn store_surface_scene_result(pixel: vec2<u32>, sample: GeometricSample) {
    if sample.termination != TERMINATION_EQUATORIAL_SURFACE {
        store_scene_result(pixel, sample.termination, sample.source_coordinates);
        return;
    }
    let bands = transported_surface_bands(sample);
    if any(bands < vec3<f32>(0.0)) {
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
        vec4<f32>(bands, SURFACE_RADIANCE_TAG),
    );
}

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn trace_spectral_surface_scene(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_origin * TRACE_WORKGROUP_AXIS + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    store_surface_scene_result(pixel, trace_pixel(pixel, extent));
}

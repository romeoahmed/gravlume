// Shared inverse-cube and homogeneous-slab transport. Emission-model selection remains a
// pipeline-level decision, so this module contains no per-invocation spectral-mode branch.

const MAXIMUM_RGBA16_FLOAT: f32 = 65504.0;
// Surface pipelines do not use alpha for shadow coverage. A value distinct from the analytic
// escape preview's 1.0 lets scientific readback identify which RGB texels are physical radiance.
const SURFACE_RADIANCE_TAG: f32 = 2.0;

fn vacuum_surface_intensity(sample: GeometricSample) -> f32 {
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
        if frequency_ratio > 1.0 && intensity > MAXIMUM_FINITE_F32 / frequency_ratio {
            return -1.0;
        }
        intensity *= frequency_ratio;
    }
    if intensity < 0.0 || !finite_scalar(intensity) {
        return -1.0;
    }
    return intensity;
}

fn transported_surface_intensity(sample: GeometricSample) -> f32 {
    let vacuum_intensity = vacuum_surface_intensity(sample);
    if vacuum_intensity < 0.0 {
        return -1.0;
    }
    let intensity = vacuum_intensity * trace_uniforms.surface_transport.z
        + trace_uniforms.surface_transport.w;
    if intensity < 0.0 || intensity > MAXIMUM_RGBA16_FLOAT || !finite_scalar(intensity) {
        return -1.0;
    }
    return intensity;
}

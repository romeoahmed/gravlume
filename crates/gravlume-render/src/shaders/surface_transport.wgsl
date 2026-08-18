// Shared inverse-cube and homogeneous-slab transport. Emission-model selection remains a
// pipeline-level decision, so this module contains no per-invocation spectral-mode branch.

const MAXIMUM_RGBA16_FLOAT: f32 = 65504.0;
const MAXIMUM_RGBA16_EXPONENT: i32 = 16i;
const MINIMUM_NORMAL_F32: f32 = 0x1p-126f;
const MINIMUM_NORMAL_F32_EXPONENT: i32 = -125i;
// Surface pipelines do not use alpha for shadow coverage. A value distinct from the analytic
// escape preview's 1.0 lets scientific readback identify which RGB texels are physical radiance.
const SURFACE_RADIANCE_TAG: f32 = 2.0;

fn resolve_positive_product(significand: f32, exponent: i32) -> f32 {
    let normalized = frexp(significand);
    let normalized_exponent = exponent + normalized.exp;
    if normalized_exponent > MAXIMUM_RGBA16_EXPONENT {
        return -1.0;
    }
    // The final value is far below the smallest RGBA16F texel. Resolve it portably instead of
    // depending on implementation-permitted flushing in numeric built-ins.
    if normalized_exponent < MINIMUM_NORMAL_F32_EXPONENT {
        return 0.0;
    }

    // Decomposition keeps every intermediate normal and in range. Runtime overflow would be
    // indeterminate under WGSL's finite-math assumption, so ldexp is called only after the final
    // exponent is proven safe.
    // Source: https://www.w3.org/TR/WGSL/#floating-point-evaluation
    // Source: https://www.w3.org/TR/WGSL/#frexp-builtin
    // Source: https://www.w3.org/TR/WGSL/#ldexp-builtin
    let value = ldexp(normalized.fract, normalized_exponent);
    if value > MAXIMUM_RGBA16_FLOAT || !finite_scalar(value) {
        return -1.0;
    }
    return value;
}

fn attenuated_surface_intensity(sample: GeometricSample, spectral_fraction: f32) -> f32 {
    let radius_ratio = sample.source_coordinates.x / 6.0;
    let frequency_ratio = sample.source_coordinates.z;
    let emitted_intensity = trace_uniforms.surface_emitter.z;
    let transmittance = trace_uniforms.surface_transport.z;
    if radius_ratio < MINIMUM_NORMAL_F32
        || frequency_ratio <= 0.0
        || emitted_intensity < 0.0
        || transmittance < 0.0
        || spectral_fraction < 0.0
        || spectral_fraction > 1.0
        || !finite_scalar(radius_ratio)
        || !finite_scalar(frequency_ratio)
        || !finite_scalar(emitted_intensity)
        || !finite_scalar(transmittance)
        || !finite_scalar(spectral_fraction)
    {
        return -1.0;
    }
    if emitted_intensity == 0.0 || transmittance == 0.0 || spectral_fraction == 0.0 {
        return 0.0;
    }
    // These two uniform factors are required to be normal at the host boundary.
    if emitted_intensity < MINIMUM_NORMAL_F32 || transmittance < MINIMUM_NORMAL_F32 {
        return -1.0;
    }
    // A subnormal frequency ratio raised to the fourth power is below binary32 even before the
    // bounded source profile is applied. Return its portable zero rather than passing it to frexp.
    if frequency_ratio < MINIMUM_NORMAL_F32 {
        return 0.0;
    }
    // Interpolated LUT fractions may enter the subnormal range, where WGSL numeric built-ins may
    // flush their inputs. The production LUT already resolves such entries to zero.
    if spectral_fraction < MINIMUM_NORMAL_F32 {
        return 0.0;
    }

    let inverse_radius_ratio = 1.0 / radius_ratio;
    if inverse_radius_ratio < MINIMUM_NORMAL_F32 || !finite_scalar(inverse_radius_ratio) {
        return -1.0;
    }
    let emitted_parts = frexp(emitted_intensity);
    let transmittance_parts = frexp(transmittance);
    let spectral_parts = frexp(spectral_fraction);
    let inverse_radius_parts = frexp(inverse_radius_ratio);
    let frequency_parts = frexp(frequency_ratio);
    let inverse_radius_squared = inverse_radius_parts.fract * inverse_radius_parts.fract;
    let frequency_squared = frequency_parts.fract * frequency_parts.fract;
    let significand = emitted_parts.fract
        * transmittance_parts.fract
        * spectral_parts.fract
        * inverse_radius_squared
        * inverse_radius_parts.fract
        * frequency_squared
        * frequency_squared;
    let exponent = emitted_parts.exp
        + transmittance_parts.exp
        + spectral_parts.exp
        + 3i * inverse_radius_parts.exp
        + 4i * frequency_parts.exp;
    return resolve_positive_product(significand, exponent);
}

fn bounded_surface_sum(incoming: f32, source: f32) -> f32 {
    if incoming < 0.0
        || source < 0.0
        || incoming > MAXIMUM_RGBA16_FLOAT
        || source > MAXIMUM_RGBA16_FLOAT - incoming
        || !finite_scalar(incoming)
        || !finite_scalar(source)
    {
        return -1.0;
    }
    return incoming + source;
}

fn transported_surface_intensity(sample: GeometricSample) -> f32 {
    let incoming = attenuated_surface_intensity(sample, 1.0);
    if incoming < 0.0 {
        return -1.0;
    }
    return bounded_surface_sum(incoming, trace_uniforms.surface_transport.w);
}

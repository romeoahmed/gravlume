// Shared inverse-cube and homogeneous-slab transport. Emission-model selection remains a
// pipeline-level decision, so this module contains no per-invocation spectral-mode branch.

const MAXIMUM_RGBA16_FLOAT: f32 = 65504.0;
const MAXIMUM_RGBA16_EXPONENT: i32 = 16i;
const MINIMUM_NORMAL_F32: f32 = 0x1p-126f;
const MINIMUM_NORMAL_F32_EXPONENT: i32 = -125i;
// Surface pipelines do not use alpha for shadow coverage. A value distinct from the analytic
// escape preview's 1.0 lets scientific readback identify which RGB texels are physical radiance.
const SURFACE_RADIANCE_TAG: f32 = 2.0;

struct PositiveLog2Scale3 {
    significand: vec3<f32>,
    exponent: vec3<i32>,
}

struct SurfaceAttenuationScale {
    // A negative significand reports invalid input; zero is an exact zero result.
    significand: f32,
    exponent: i32,
}

fn positive_log2_scale3(log2_scale: vec3<f32>) -> PositiveLog2Scale3 {
    // exp2 only sees [0, 1), so its result is normal and can safely enter frexp-based products.
    // Source: https://www.w3.org/TR/WGSL/#floating-point-evaluation
    // Source: https://www.w3.org/TR/WGSL/#exp2-builtin
    let exponent = vec3<i32>(floor(log2_scale));
    return PositiveLog2Scale3(
        exp2(log2_scale - vec3<f32>(exponent)),
        exponent,
    );
}

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

fn resolve_positive_products(significand: vec3<f32>, exponent: vec3<i32>) -> vec3<f32> {
    // frexp, ldexp, and exp2 have component-wise vector overloads. Keeping the three spectral
    // bands together avoids repeating the identical geometric decomposition for every channel.
    // Source: https://www.w3.org/TR/WGSL/#frexp-builtin
    // Source: https://www.w3.org/TR/WGSL/#ldexp-builtin
    let normalized = frexp(significand);
    let normalized_exponent = exponent + normalized.exp;
    if any(normalized_exponent > vec3<i32>(MAXIMUM_RGBA16_EXPONENT)) {
        return vec3<f32>(-1.0);
    }
    let underflow = normalized_exponent < vec3<i32>(MINIMUM_NORMAL_F32_EXPONENT);
    let safe_exponent = max(
        normalized_exponent,
        vec3<i32>(MINIMUM_NORMAL_F32_EXPONENT),
    );
    let value = ldexp(normalized.fract, safe_exponent);
    if any(value > vec3<f32>(MAXIMUM_RGBA16_FLOAT)) || !finite_vec3(value) {
        return vec3<f32>(-1.0);
    }
    return select(value, vec3<f32>(0.0), underflow);
}

fn scaled_spectral_intensities(
    intensity: f32,
    spectral_log2_fractions: vec3<f32>,
) -> vec3<f32> {
    if intensity < 0.0
        || any(spectral_log2_fractions > vec3<f32>(0.0))
        || !finite_scalar(intensity)
        || !finite_vec3(spectral_log2_fractions)
    {
        return vec3<f32>(-1.0);
    }
    if intensity == 0.0 {
        return vec3<f32>(0.0);
    }
    if intensity < MINIMUM_NORMAL_F32 {
        return vec3<f32>(-1.0);
    }

    let intensity_parts = frexp(intensity);
    let spectral_scale = positive_log2_scale3(spectral_log2_fractions);
    return resolve_positive_products(
        vec3<f32>(intensity_parts.fract) * spectral_scale.significand,
        vec3<i32>(intensity_parts.exp) + spectral_scale.exponent,
    );
}

fn surface_attenuation_scale(
    radius_ratio: f32,
    frequency_ratio: f32,
) -> SurfaceAttenuationScale {
    let emitted_intensity = trace_uniforms.surface_emitter.z;
    let transmittance = trace_uniforms.surface_transport.z;
    if radius_ratio < MINIMUM_NORMAL_F32
        || frequency_ratio <= 0.0
        || emitted_intensity < 0.0
        || transmittance < 0.0
        || !finite_scalar(radius_ratio)
        || !finite_scalar(frequency_ratio)
        || !finite_scalar(emitted_intensity)
        || !finite_scalar(transmittance)
    {
        return SurfaceAttenuationScale(-1.0, 0i);
    }
    if emitted_intensity == 0.0 || transmittance == 0.0 {
        return SurfaceAttenuationScale(0.0, 0i);
    }
    // These two uniform factors are required to be normal at the host boundary.
    if emitted_intensity < MINIMUM_NORMAL_F32 || transmittance < MINIMUM_NORMAL_F32 {
        return SurfaceAttenuationScale(-1.0, 0i);
    }
    // A subnormal frequency ratio raised to the fourth power is below binary32 even before the
    // bounded source profile is applied. Return its portable zero rather than passing it to frexp.
    if frequency_ratio < MINIMUM_NORMAL_F32 {
        return SurfaceAttenuationScale(0.0, 0i);
    }
    let inverse_radius_ratio = 1.0 / radius_ratio;
    if inverse_radius_ratio < MINIMUM_NORMAL_F32 || !finite_scalar(inverse_radius_ratio) {
        return SurfaceAttenuationScale(-1.0, 0i);
    }
    let emitted_parts = frexp(emitted_intensity);
    let transmittance_parts = frexp(transmittance);
    let inverse_radius_parts = frexp(inverse_radius_ratio);
    let frequency_parts = frexp(frequency_ratio);
    let inverse_radius_squared = inverse_radius_parts.fract * inverse_radius_parts.fract;
    let frequency_squared = frequency_parts.fract * frequency_parts.fract;
    let significand = emitted_parts.fract
        * transmittance_parts.fract
        * inverse_radius_squared
        * inverse_radius_parts.fract
        * frequency_squared
        * frequency_squared;
    let exponent = emitted_parts.exp
        + transmittance_parts.exp
        + 3i * inverse_radius_parts.exp
        + 4i * frequency_parts.exp;
    return SurfaceAttenuationScale(significand, exponent);
}

fn attenuated_bolometric_intensity(sample: GeometricSample) -> f32 {
    let attenuation = surface_attenuation_scale(
        sample.source_coordinates.x / 6.0,
        sample.source_coordinates.z,
    );
    if attenuation.significand <= 0.0 {
        return attenuation.significand;
    }
    return resolve_positive_product(attenuation.significand, attenuation.exponent);
}

fn attenuated_surface_spectrum(
    radius_ratio: f32,
    frequency_ratio: f32,
    spectral_log2_fractions: vec3<f32>,
) -> vec3<f32> {
    if any(spectral_log2_fractions > vec3<f32>(0.0))
        || !finite_vec3(spectral_log2_fractions)
    {
        return vec3<f32>(-1.0);
    }
    let attenuation = surface_attenuation_scale(radius_ratio, frequency_ratio);
    if attenuation.significand <= 0.0 {
        return vec3<f32>(attenuation.significand);
    }
    let spectral_scale = positive_log2_scale3(spectral_log2_fractions);
    return resolve_positive_products(
        vec3<f32>(attenuation.significand) * spectral_scale.significand,
        vec3<i32>(attenuation.exponent) + spectral_scale.exponent,
    );
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

fn bounded_surface_sums(incoming: vec3<f32>, source: vec3<f32>) -> vec3<f32> {
    if any(incoming < vec3<f32>(0.0))
        || any(source < vec3<f32>(0.0))
        || any(incoming > vec3<f32>(MAXIMUM_RGBA16_FLOAT))
        || any(source > vec3<f32>(MAXIMUM_RGBA16_FLOAT) - incoming)
        || !finite_vec3(incoming)
        || !finite_vec3(source)
    {
        return vec3<f32>(-1.0);
    }
    return incoming + source;
}

fn transported_surface_intensity(sample: GeometricSample) -> f32 {
    let incoming = attenuated_bolometric_intensity(sample);
    if incoming < 0.0 {
        return -1.0;
    }
    return bounded_surface_sum(incoming, trace_uniforms.surface_transport.w);
}

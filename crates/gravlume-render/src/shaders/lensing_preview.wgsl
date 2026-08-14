// Maps physical trace results to the scene-linear preview. The alpha channel is an internal
// branch/coverage tag consumed by selective shadow refinement, not display opacity.

@group(0) @binding(1)
var scene_hdr: texture_storage_2d<rgba16float, write>;

fn visible_failure_color(termination: u32) -> vec3<f32> {
    if termination == TERMINATION_SINGULARITY {
        return vec3<f32>(0.0, 1.0, 1.0);
    }
    if termination == TERMINATION_STEP_EXHAUSTION {
        return vec3<f32>(1.0, 0.25, 0.0);
    }
    if termination == TERMINATION_UNCERTAIN {
        return vec3<f32>(1.0, 1.0, 0.0);
    }
    return vec3<f32>(1.0, 0.0, 1.0);
}

fn analytic_sky(unit_direction: vec3<f32>) -> vec3<f32> {
    // Every Escape producer commits a normalized direction; preserving that interface avoids a
    // redundant inverse square root for most visible pixels.
    let encoded = 0.5 * (unit_direction + vec3<f32>(1.0));
    var sky = vec3<f32>(0.035, 0.045, 0.06)
        + vec3<f32>(0.22, 0.20, 0.24) * encoded;

    // Low-order spherical structure makes lensing visible without a seam or sub-pixel grid.
    let longitude_three = unit_direction.x
        * (unit_direction.x * unit_direction.x - 3.0 * unit_direction.y * unit_direction.y);
    let z_squared = unit_direction.z * unit_direction.z;
    let latitude_four = 8.0 * z_squared * z_squared - 8.0 * z_squared + 1.0;
    let bands = clamp(0.5 + 0.25 * longitude_three + 0.25 * latitude_four, 0.0, 1.0);
    sky *= 0.84 + 0.16 * bands;

    // The weights vanish to twelfth order at each sign plane, so the six direction markers have
    // no hemisphere seam.
    let squared = unit_direction * unit_direction;
    let fourth = squared * squared;
    let axis_weight = fourth * fourth * fourth;
    let weight_sum = axis_weight.x + axis_weight.y + axis_weight.z;
    let x_color = select(
        vec3<f32>(0.08, 0.45, 0.55),
        vec3<f32>(1.05, 0.16, 0.10),
        unit_direction.x >= 0.0,
    );
    let y_color = select(
        vec3<f32>(0.60, 0.12, 0.52),
        vec3<f32>(0.12, 0.65, 0.20),
        unit_direction.y >= 0.0,
    );
    let z_color = select(
        vec3<f32>(0.78, 0.50, 0.08),
        vec3<f32>(0.16, 0.28, 0.82),
        unit_direction.z >= 0.0,
    );
    let axis_color = (
        axis_weight.x * x_color
        + axis_weight.y * y_color
        + axis_weight.z * z_color
    ) / max(weight_sum, 1e-6);
    return mix(sky, axis_color, clamp(weight_sum, 0.0, 1.0));
}

fn store_scene_result(pixel: vec2<u32>, termination: u32, direction: vec3<f32>) {
    if termination == TERMINATION_HORIZON {
        textureStore(scene_hdr, vec2<i32>(pixel), vec4<f32>(0.0));
        return;
    }
    if termination == TERMINATION_ESCAPE {
        textureStore(
            scene_hdr,
            vec2<i32>(pixel),
            vec4<f32>(analytic_sky(direction), 1.0),
        );
        return;
    }
    textureStore(
        scene_hdr,
        vec2<i32>(pixel),
        vec4<f32>(visible_failure_color(termination), -f32(termination)),
    );
}

struct ShadowClassifyControl {
    count: atomic<u32>,
    capacity: u32,
    padding: vec2<u32>,
}

struct ShadowRefineControl {
    count: u32,
    capacity: u32,
    padding: vec2<u32>,
}

@group(1) @binding(0)
var shadow_source: texture_2d<f32>;

@group(1) @binding(1)
var<storage, read_write> classified_shadow_pixels: array<u32>;

@group(1) @binding(2)
var<storage, read_write> shadow_classify_control: ShadowClassifyControl;

@group(2) @binding(0)
var shadow_refined_scene: texture_storage_2d<rgba16float, write>;

@group(2) @binding(1)
var<storage, read> shadow_pixels_to_refine: array<u32>;

@group(2) @binding(2)
var<storage, read> shadow_refine_control: ShadowRefineControl;

fn shadow_branch_at(pixel: vec2<i32>) -> i32 {
    let coverage = textureLoad(shadow_source, pixel, 0).a;
    if coverage == 0.0 {
        return 0;
    }
    if coverage == 1.0 {
        return 1;
    }
    return -1;
}

@compute @workgroup_size(8, 8, 1)
fn classify_shadow_edges(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(shadow_source);
    let pixel = global_id.xy;
    if any(pixel >= extent) {
        return;
    }

    let pixel_i = vec2<i32>(pixel);
    let center = shadow_branch_at(pixel_i);
    if center < 0 {
        return;
    }

    var is_edge = false;
    if pixel.x > 0u {
        is_edge = is_edge || shadow_branch_at(pixel_i + vec2<i32>(-1, 0)) == 1 - center;
    }
    if pixel.x + 1u < extent.x {
        is_edge = is_edge || shadow_branch_at(pixel_i + vec2<i32>(1, 0)) == 1 - center;
    }
    if pixel.y > 0u {
        is_edge = is_edge || shadow_branch_at(pixel_i + vec2<i32>(0, -1)) == 1 - center;
    }
    if pixel.y + 1u < extent.y {
        is_edge = is_edge || shadow_branch_at(pixel_i + vec2<i32>(0, 1)) == 1 - center;
    }
    if !is_edge {
        return;
    }

    let slot = atomicAdd(&shadow_classify_control.count, 1u);
    if slot < shadow_classify_control.capacity {
        classified_shadow_pixels[slot] = pixel.y * extent.x + pixel.x;
    }
}

@compute @workgroup_size(64, 1, 1)
fn refine_shadow_edges(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let count = shadow_refine_control.count;
    let capacity = shadow_refine_control.capacity;
    if count > capacity || index >= count {
        return;
    }

    let extent = textureDimensions(shadow_refined_scene);
    let linear_pixel = shadow_pixels_to_refine[index];
    let pixel = vec2<u32>(linear_pixel % extent.x, linear_pixel / extent.x);
    let sample_offsets = array<vec2<f32>, 4>(
        vec2<f32>(0.375, 0.125),
        vec2<f32>(0.875, 0.375),
        vec2<f32>(0.625, 0.875),
        vec2<f32>(0.125, 0.625),
    );
    var scene_linear = vec3<f32>(0.0);
    var escape_samples = 0u;
    for (var sample_index = 0u; sample_index < 4u; sample_index += 1u) {
        let result = trace_pixel_with_kerr_capture_at(
            pixel,
            extent,
            sample_offsets[sample_index],
        );
        if result.termination == TERMINATION_ESCAPE {
            scene_linear += analytic_sky(result.direction);
            escape_samples += 1u;
        } else if result.termination != TERMINATION_HORIZON {
            return;
        }
    }

    textureStore(
        shadow_refined_scene,
        vec2<i32>(pixel),
        vec4<f32>(scene_linear * 0.25, f32(escape_samples) * 0.25),
    );
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@vertex
fn fullscreen_triangle(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    return VertexOutput(vec4<f32>(x, y, 0.0, 1.0));
}

fn is_non_finite(value: f32) -> bool {
    let exponent = bitcast<u32>(value) & 0x7f800000u;
    return exponent == 0x7f800000u;
}

fn invalid_scene_linear(scene_linear: vec3<f32>) -> bool {
    return any(scene_linear < vec3<f32>(0.0))
        || is_non_finite(scene_linear.x)
        || is_non_finite(scene_linear.y)
        || is_non_finite(scene_linear.z);
}

fn neutral_display(scene_linear: vec3<f32>) -> vec3<f32> {
    if invalid_scene_linear(scene_linear) {
        return vec3<f32>(1.0, 0.0, 1.0);
    }
    return scene_linear / (vec3<f32>(1.0) + scene_linear);
}

fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear <= vec3<f32>(0.0031308);
    let lower = linear * vec3<f32>(12.92);
    let upper = vec3<f32>(1.055) * pow(linear, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(upper, lower, cutoff);
}

fn srgb_to_linear(gamma: vec3<f32>) -> vec3<f32> {
    let cutoff = gamma <= vec3<f32>(0.04045);
    let lower = gamma / vec3<f32>(12.92);
    let upper = pow((gamma + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(upper, lower, cutoff);
}

fn display_linear_at(position: vec4<f32>) -> vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let scene_linear = textureLoad(input_texture, pixel, 0).rgb;
    return vec4<f32>(neutral_display(scene_linear), 1.0);
}

@fragment
fn display_to_gamma_target(in: VertexOutput) -> @location(0) vec4<f32> {
    let display_linear = display_linear_at(in.position);
    return vec4<f32>(linear_to_srgb(display_linear.rgb), display_linear.a);
}

@fragment
fn present_to_linear_target(in: VertexOutput) -> @location(0) vec4<f32> {
    let gamma = textureLoad(input_texture, vec2<i32>(in.position.xy), 0);
    return vec4<f32>(srgb_to_linear(gamma.rgb), gamma.a);
}

@fragment
fn present_to_gamma_target(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureLoad(input_texture, vec2<i32>(in.position.xy), 0);
}

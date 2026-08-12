struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct OutputUniforms {
    mapping: vec4<f32>,
}

@group(0) @binding(0)
var scene_texture: texture_2d<f32>;

@group(0) @binding(1)
var ui_texture: texture_2d<f32>;

@group(0) @binding(2)
var<uniform> output_uniforms: OutputUniforms;

@vertex
fn fullscreen_triangle(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    let uv = vec2<f32>(0.5 * (x + 1.0), 0.5 * (1.0 - y));
    return VertexOutput(vec4<f32>(x, y, 0.0, 1.0), uv);
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

fn texel_at(texture: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let dimensions = textureDimensions(texture);
    let maximum = vec2<i32>(dimensions) - vec2<i32>(1);
    let texel = min(vec2<i32>(uv * vec2<f32>(dimensions)), maximum);
    return textureLoad(texture, texel, 0);
}

fn aspect_fitted_scene_uv(uv: vec2<f32>) -> vec2<f32> {
    let scene_dimensions = vec2<f32>(textureDimensions(scene_texture));
    let output_dimensions = vec2<f32>(textureDimensions(ui_texture));
    let scene_aspect = scene_dimensions.x / scene_dimensions.y;
    let output_aspect = output_dimensions.x / output_dimensions.y;
    var fitted = uv;
    if output_aspect > scene_aspect {
        fitted.x = 0.5 + (uv.x - 0.5) * output_aspect / scene_aspect;
    } else {
        fitted.y = 0.5 + (uv.y - 0.5) * scene_aspect / output_aspect;
    }
    return fitted;
}

fn scene_at(uv: vec2<f32>) -> vec3<f32> {
    let fitted = aspect_fitted_scene_uv(uv);
    if any(fitted < vec2<f32>(0.0)) || any(fitted >= vec2<f32>(1.0)) {
        return vec3<f32>(0.0);
    }
    let scene_linear = texel_at(scene_texture, fitted).rgb;
    return select(scene_linear, vec3<f32>(1.0, 0.0, 1.0), invalid_scene_linear(scene_linear));
}

fn ui_at(uv: vec2<f32>) -> vec4<f32> {
    let gamma_premultiplied = texel_at(ui_texture, uv);
    let alpha = clamp(gamma_premultiplied.a, 0.0, 1.0);
    if alpha == 0.0 {
        return vec4<f32>(0.0);
    }
    let gamma_straight = clamp(gamma_premultiplied.rgb / alpha, vec3<f32>(0.0), vec3<f32>(1.0));
    let linear_premultiplied = srgb_to_linear(gamma_straight) * alpha;
    return vec4<f32>(linear_premultiplied, alpha);
}

fn sdr_map(scene_linear: vec3<f32>) -> vec3<f32> {
    return scene_linear / (vec3<f32>(1.0) + scene_linear);
}

fn hdr_map_channel(scene_linear: f32, headroom: f32) -> f32 {
    if headroom <= 1.0 {
        return min(scene_linear, 1.0);
    }
    if scene_linear <= 1.0 {
        return scene_linear;
    }
    let highlight_range = headroom - 1.0;
    return 1.0 + highlight_range * (1.0 - exp(-(scene_linear - 1.0) / highlight_range));
}

fn hdr_map(scene_linear: vec3<f32>) -> vec3<f32> {
    let headroom = max(output_uniforms.mapping.x, 1.0);
    return vec3<f32>(
        hdr_map_channel(scene_linear.x, headroom),
        hdr_map_channel(scene_linear.y, headroom),
        hdr_map_channel(scene_linear.z, headroom),
    );
}

fn composite(mapped_scene: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let ui = ui_at(uv);
    let reference_white_scale = max(output_uniforms.mapping.y, 0.0);
    return (mapped_scene * (1.0 - ui.a) + ui.rgb) * reference_white_scale;
}

@fragment
fn present_sdr_to_linear_target(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(composite(sdr_map(scene_at(in.uv)), in.uv), 1.0);
}

@fragment
fn present_sdr_to_gamma_target(in: VertexOutput) -> @location(0) vec4<f32> {
    let linear = composite(sdr_map(scene_at(in.uv)), in.uv);
    return vec4<f32>(linear_to_srgb(linear), 1.0);
}

@fragment
fn present_hdr_extended_linear(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(composite(hdr_map(scene_at(in.uv)), in.uv), 1.0);
}

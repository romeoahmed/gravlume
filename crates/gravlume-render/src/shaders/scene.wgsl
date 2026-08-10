@group(0) @binding(0)
var scene_hdr: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    if global_id.x >= extent.x || global_id.y >= extent.y {
        return;
    }

    let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) / vec2<f32>(extent);
    let scene_linear = vec3<f32>(
        4.0 * uv.x,
        0.1 + 0.7 * uv.y,
        2.0 * (1.0 - uv.x),
    );
    textureStore(scene_hdr, vec2<i32>(global_id.xy), vec4<f32>(scene_linear, 1.0));
}

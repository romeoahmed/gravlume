struct TraceUniforms {
    spacetime: vec4<f32>,
    observer_event: vec4<f32>,
    observer_velocity: vec4<f32>,
    image_right: vec4<f32>,
    image_up: vec4<f32>,
    arrival: vec4<f32>,
    projection_policy: vec4<f32>,
    integration: vec4<f32>,
    viewport: vec4<u32>,
}

struct TraceRecord {
    direction_time: vec4<f32>,
    invariant_drift: vec4<f32>,
    metadata: vec4<u32>,
}

@group(0) @binding(0)
var<uniform> trace_uniforms: TraceUniforms;

@group(0) @binding(1)
var scene_hdr: texture_storage_2d<rgba16float, write>;

@group(0) @binding(2)
var<storage, read_write> trace_records: array<TraceRecord>;

fn record_index(pixel: vec2<u32>, extent: vec2<u32>) -> u32 {
    return pixel.y * extent.x + pixel.x;
}

fn inside_extent(pixel: vec2<u32>, extent: vec2<u32>) -> bool {
    return pixel.x < extent.x && pixel.y < extent.y;
}

@compute @workgroup_size(8, 8, 1)
fn initial_ray_contract(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    if !inside_extent(global_id.xy, extent) {
        return;
    }

    let index = record_index(global_id.xy, extent);
    trace_records[index] = TraceRecord(
        vec4<f32>(trace_uniforms.arrival.xyz, 0.0),
        vec4<f32>(0.0),
        vec4<u32>(0u),
    );
    textureStore(scene_hdr, vec2<i32>(global_id.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn trace_scene(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    if !inside_extent(global_id.xy, extent) {
        return;
    }

    let index = record_index(global_id.xy, extent);
    trace_records[index] = TraceRecord(
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0),
        vec4<u32>(2u, 0u, 0u, 0u),
    );
    textureStore(scene_hdr, vec2<i32>(global_id.xy), vec4<f32>(0.1, 0.2, 0.4, 1.0));
}

// Test-only source-chart finite differences. Every neighbor is a real quarter-pixel trace; a
// mismatch in terminal semantics or the full branch key produces a discontinuity record.

const FOOTPRINT_OFFSET_PIXELS: f32 = 0.25;
const FOOTPRINT_DIFFERENCE_SPAN_PIXELS: f32 = 0.5;

fn footprint_surface_semantics_match(center: GeometricSample, neighbor: GeometricSample) -> bool {
    return center.termination == TERMINATION_EQUATORIAL_SURFACE
        && neighbor.termination == TERMINATION_EQUATORIAL_SURFACE
        && center.flags == 0u
        && neighbor.flags == 0u
        && center.event_candidates == EVENT_CANDIDATE_SURFACE
        && neighbor.event_candidates == EVENT_CANDIDATE_SURFACE
        && all(center.branch_key == neighbor.branch_key);
}

fn footprint_angle_difference(value: f32, center: f32) -> f32 {
    let pi = 3.141592653589793;
    let tau = 6.283185307179586;
    return value - center - tau * floor((value - center + pi) / tau);
}

fn store_footprint_record(
    index: u32,
    center: GeometricSample,
    left: GeometricSample,
    right: GeometricSample,
    up: GeometricSample,
    down: GeometricSample,
) {
    trace_source_time[index] = vec4<f32>(center.source_coordinates, center.travel_time);
    let continuous = footprint_surface_semantics_match(center, left)
        && footprint_surface_semantics_match(center, right)
        && footprint_surface_semantics_match(center, up)
        && footprint_surface_semantics_match(center, down);
    if !continuous {
        trace_invariant_drift[index] = vec4<f32>(0.0);
        trace_metadata[index] = vec4<u32>(center.termination, center.flags, 0u, 0u);
        trace_event[index] = center.branch_key;
        return;
    }

    let center_radius = center.source_coordinates.x;
    let left_arc = center_radius * footprint_angle_difference(
        left.source_coordinates.y,
        center.source_coordinates.y,
    );
    let right_arc = center_radius * footprint_angle_difference(
        right.source_coordinates.y,
        center.source_coordinates.y,
    );
    let up_arc = center_radius * footprint_angle_difference(
        up.source_coordinates.y,
        center.source_coordinates.y,
    );
    let down_arc = center_radius * footprint_angle_difference(
        down.source_coordinates.y,
        center.source_coordinates.y,
    );
    let jacobian = vec4<f32>(
        (right.source_coordinates.x - left.source_coordinates.x)
            / FOOTPRINT_DIFFERENCE_SPAN_PIXELS,
        (down.source_coordinates.x - up.source_coordinates.x)
            / FOOTPRINT_DIFFERENCE_SPAN_PIXELS,
        (right_arc - left_arc) / FOOTPRINT_DIFFERENCE_SPAN_PIXELS,
        (down_arc - up_arc) / FOOTPRINT_DIFFERENCE_SPAN_PIXELS,
    );
    let determinant = jacobian.x * jacobian.w - jacobian.y * jacobian.z;
    let squared_norm = dot(jacobian, jacobian);
    let determinant_floor = 64.0 * 0x1p-23f * squared_norm;
    var parity = 3u;
    if determinant > determinant_floor {
        parity = 1u;
    } else if determinant < -determinant_floor {
        parity = 2u;
    }
    let resolved = select(0u, 1u, finite_vec4(jacobian));
    trace_invariant_drift[index] = select(vec4<f32>(0.0), jacobian, resolved != 0u);
    trace_metadata[index] = vec4<u32>(center.termination, center.flags, resolved, parity);
    trace_event[index] = center.branch_key;
}

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn capture_surface_footprint(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let pixel = trace_dispatch.tile_origin * TRACE_WORKGROUP_AXIS + global_id.xy;
    if any(pixel >= extent) {
        return;
    }
    let subpixel = trace_uniforms.camera.zw;
    let center = trace_pixel_at(pixel, extent, subpixel);
    let left = trace_pixel_at(pixel, extent, subpixel + vec2<f32>(-FOOTPRINT_OFFSET_PIXELS, 0.0));
    let right = trace_pixel_at(pixel, extent, subpixel + vec2<f32>(FOOTPRINT_OFFSET_PIXELS, 0.0));
    let up = trace_pixel_at(pixel, extent, subpixel + vec2<f32>(0.0, -FOOTPRINT_OFFSET_PIXELS));
    let down = trace_pixel_at(pixel, extent, subpixel + vec2<f32>(0.0, FOOTPRINT_OFFSET_PIXELS));
    let index = pixel.y * extent.x + pixel.x;
    store_footprint_record(index, center, left, right, up, down);
    store_surface_scene_result(pixel, center);
}

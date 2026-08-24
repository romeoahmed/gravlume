// Conservative radial capture and a compact escape-direction map accelerate terminal tracing.
// Every inconclusive certificate or stencil falls back to the full Kerr-Schild integrator.

struct EscapeDirectionMap {
    nodes: array<u32>,
}

@group(0) @binding(7)
var<storage, read_write> escape_direction_map: EscapeDirectionMap;

const ESCAPE_MAP_TILE_AXIS = 8u;
const ESCAPE_MAP_NODE_STEP = 4u;
const ESCAPE_MAP_STENCIL_AXIS = 3u;
// Leave room inside the project-wide 3.82e-4 direction budget for octahedral quantization.
const ESCAPE_MAP_MAXIMUM_STENCIL_ERROR = 3.0e-4;
const ESCAPE_MAP_OCTAHEDRAL_MAXIMUM = 0x7fffu;
const ESCAPE_MAP_TAG_SHIFT = 30u;
const ESCAPE_MAP_TAG_ESCAPE = 1u;
const ESCAPE_MAP_TAG_HORIZON = 2u;
const RADIAL_CAPTURE_SEGMENTS = 12u;

const_assert ESCAPE_MAP_TILE_AXIS == 2u * ESCAPE_MAP_NODE_STEP;
const_assert ESCAPE_MAP_TILE_AXIS == TRACE_WORKGROUP_AXIS;
const_assert ESCAPE_MAP_STENCIL_AXIS == 3u;

alias IntervalF32 = vec2<f32>;

const INTERVAL_MINIMUM_NORMAL: f32 = 0x1p-126f;
const RADIAL_CAPTURE_INPUT_ENVELOPE: f32 = 0x1p-12f;

var<workgroup> escape_map_stencil: array<vec4<f32>, 9>;
var<workgroup> escape_map_accepted: u32;

fn interval_next_lower(value: f32) -> f32 {
    if abs(value) < INTERVAL_MINIMUM_NORMAL {
        return -INTERVAL_MINIMUM_NORMAL;
    }
    let bits = bitcast<u32>(value);
    return bitcast<f32>(select(bits + 1u, bits - 1u, value > 0.0));
}

fn interval_next_upper(value: f32) -> f32 {
    if abs(value) < INTERVAL_MINIMUM_NORMAL {
        return INTERVAL_MINIMUM_NORMAL;
    }
    let bits = bitcast<u32>(value);
    return bitcast<f32>(select(bits - 1u, bits + 1u, value > 0.0));
}

fn interval_point(value: f32) -> IntervalF32 {
    return vec2<f32>(value);
}

fn interval_one_ulp(value: f32) -> IntervalF32 {
    return vec2<f32>(interval_next_lower(value), interval_next_upper(value));
}

fn interval_add(left: IntervalF32, right: IntervalF32) -> IntervalF32 {
    let sum = left + right;
    return vec2<f32>(
        interval_next_lower(sum.x),
        interval_next_upper(sum.y),
    );
}

fn interval_subtract(left: IntervalF32, right: IntervalF32) -> IntervalF32 {
    let difference = left - right.yx;
    return vec2<f32>(
        interval_next_lower(difference.x),
        interval_next_upper(difference.y),
    );
}

fn interval_negate(value: IntervalF32) -> IntervalF32 {
    return -value.yx;
}

fn interval_multiply(left: IntervalF32, right: IntervalF32) -> IntervalF32 {
    let products = vec4<f32>(left.x * right, left.y * right);
    let lower = min(min(products.x, products.y), min(products.z, products.w));
    let upper = max(max(products.x, products.y), max(products.z, products.w));
    return vec2<f32>(interval_next_lower(lower), interval_next_upper(upper));
}

fn interval_square(value: IntervalF32) -> IntervalF32 {
    let squared = value * value;
    if value.x <= 0.0 && value.y >= 0.0 {
        return vec2<f32>(0.0, interval_next_upper(max(squared.x, squared.y)));
    }
    return vec2<f32>(
        interval_next_lower(min(squared.x, squared.y)),
        interval_next_upper(max(squared.x, squared.y)),
    );
}

fn interval_scale(value: IntervalF32, scalar: f32) -> IntervalF32 {
    return interval_multiply(value, interval_point(scalar));
}

fn interval_expand_input(value: f32) -> IntervalF32 {
    let scale = interval_add(interval_point(abs(value)), interval_point(1.0));
    let padding = interval_multiply(scale, interval_point(RADIAL_CAPTURE_INPUT_ENVELOPE)).y;
    return vec2<f32>(
        interval_next_lower(value - padding),
        interval_next_upper(value + padding),
    );
}

fn interval_is_finite(value: IntervalF32) -> bool {
    return all(value == value)
        && all(abs(value) <= vec2<f32>(MAXIMUM_FINITE_F32))
        && value.x <= value.y;
}

fn radial_capture_segment_is_positive(
    lower: IntervalF32,
    width: IntervalF32,
    leading: IntervalF32,
    quadratic: IntervalF32,
    linear: IntervalF32,
    constant: IntervalF32,
) -> bool {
    let lower_squared = interval_square(lower);
    let lower_cubed = interval_multiply(lower_squared, lower);
    let lower_fourth = interval_square(lower_squared);
    let width_squared = interval_square(width);
    let width_cubed = interval_multiply(width_squared, width);
    let power0 = interval_add(
        interval_add(
            interval_multiply(leading, lower_fourth),
            interval_multiply(quadratic, lower_squared),
        ),
        interval_add(interval_multiply(linear, lower), constant),
    );
    let power1 = interval_multiply(
        width,
        interval_add(
            interval_add(
                interval_scale(interval_multiply(leading, lower_cubed), 4.0),
                interval_scale(interval_multiply(quadratic, lower), 2.0),
            ),
            linear,
        ),
    );
    let power2 = interval_multiply(
        width_squared,
        interval_add(
            interval_scale(interval_multiply(leading, lower_squared), 6.0),
            quadratic,
        ),
    );
    let power3 = interval_scale(
        interval_multiply(interval_multiply(leading, lower), width_cubed),
        4.0,
    );
    let power4 = interval_multiply(leading, interval_square(width_squared));
    let one_sixth = vec2<f32>(
        bitcast<f32>(0x3e2aaaaau),
        bitcast<f32>(0x3e2aaaacu),
    );
    let coefficient0 = power0;
    let coefficient1 = interval_add(power0, interval_scale(power1, 0.25));
    let coefficient2 = interval_add(
        interval_add(power0, interval_scale(power1, 0.5)),
        interval_multiply(power2, one_sixth),
    );
    let coefficient3 = interval_add(
        interval_add(power0, interval_scale(power1, 0.75)),
        interval_add(
            interval_scale(power2, 0.5),
            interval_scale(power3, 0.25),
        ),
    );
    let coefficient4 = interval_add(
        interval_add(power0, power1),
        interval_add(interval_add(power2, power3), power4),
    );
    let coefficient_lowers = vec4<f32>(
        coefficient0.x,
        coefficient1.x,
        coefficient2.x,
        coefficient3.x,
    );
    let coefficient_uppers = vec4<f32>(
        coefficient0.y,
        coefficient1.y,
        coefficient2.y,
        coefficient3.y,
    );
    let finite = all(coefficient_lowers == coefficient_lowers)
        && all(coefficient_uppers == coefficient_uppers)
        && all(abs(coefficient_lowers) <= vec4<f32>(MAXIMUM_FINITE_F32))
        && all(abs(coefficient_uppers) <= vec4<f32>(MAXIMUM_FINITE_F32));
    return finite
        && all(coefficient_lowers <= coefficient_uppers)
        && all(coefficient_lowers > vec4<f32>(0.0))
        && interval_is_finite(coefficient4)
        && coefficient4.x > 0.0;
}

fn radial_capture_is_proven(
    initial: InitialState,
    invariants: Invariants,
) -> bool {
    let spin_value = trace_uniforms.spacetime.y;
    let charge_value = trace_uniforms.spacetime.z;
    let horizon_value = trace_uniforms.event_surfaces.z;
    let observer_radius_value = initial.geometry.radius;
    let observer_cylindrical_radius_squared = dot(
        initial.state.position.xy,
        initial.state.position.xy,
    );
    let axis_margin = 0.0625 * observer_radius_value;
    let future_radial_derivative = dot(
        initial.geometry.radius_gradient,
        initial.rhs.spacetime.yzw,
    );
    let extremality = spin_value * spin_value + charge_value * charge_value;
    if trace_uniforms.spacetime.x != 1.0
        || abs(spin_value) > 0.9
        || !finite_scalar(extremality)
        || extremality >= 1.0 - RADIAL_CAPTURE_INPUT_ENVELOPE
        || horizon_value <= 0.0
        || observer_radius_value <= horizon_value
        || observer_radius_value > 256.0
        || observer_cylindrical_radius_squared <= axis_margin * axis_margin
        || future_radial_derivative <= RADIAL_CAPTURE_INPUT_ENVELOPE
        || initial.rhs.flags != 0u
        || invariants.flags != 0u
        || any(abs(invariants.values.yzw) > vec3<f32>(4096.0))
    {
        return false;
    }

    let energy = interval_expand_input(invariants.values.y);
    let angular_momentum = interval_expand_input(invariants.values.z);
    let carter = interval_expand_input(invariants.values.w);
    let spin = interval_one_ulp(spin_value);
    let charge = interval_one_ulp(charge_value);
    let shifted = interval_subtract(angular_momentum, interval_multiply(spin, energy));
    let separation = interval_add(interval_square(shifted), carter);
    if !interval_is_finite(separation) || separation.x <= 0.0 {
        return false;
    }
    let leading = interval_square(energy);
    let quadratic = interval_subtract(
        interval_negate(interval_scale(interval_multiply(interval_multiply(energy, spin), shifted), 2.0)),
        separation,
    );
    let linear = interval_scale(separation, 2.0);
    let constant = interval_subtract(
        interval_negate(interval_multiply(interval_square(spin), carter)),
        interval_multiply(interval_square(charge), separation),
    );
    let horizon = interval_one_ulp(horizon_value);
    let observer_radius = interval_expand_input(observer_radius_value);
    let one_twelfth = vec2<f32>(
        bitcast<f32>(0x3daaaaaau),
        bitcast<f32>(0x3daaaaacu),
    );
    let width = interval_multiply(interval_subtract(observer_radius, horizon), one_twelfth);
    if !interval_is_finite(width) || width.x <= 0.0 {
        return false;
    }
    for (var segment = 0u; segment < RADIAL_CAPTURE_SEGMENTS; segment += 1u) {
        let lower = interval_add(horizon, interval_scale(width, f32(segment)));
        if !radial_capture_segment_is_positive(
            lower,
            width,
            leading,
            quadratic,
            linear,
            constant,
        ) {
            return false;
        }
    }
    return true;
}

fn trace_pixel_with_radial_capture_at(
    pixel: vec2<u32>,
    extent: vec2<u32>,
    subpixel: vec2<f32>,
) -> GeometricSample {
    let initial = initial_state_at(pixel, extent, subpixel);
    if initial.rhs.flags != 0u {
        return failure_result(initial.rhs.flags, 0u, 0.0, vec4<f32>(0.0));
    }
    let invariants = invariants_from_geometry_rhs(
        initial.state,
        initial.energy,
        initial.geometry,
        initial.rhs,
    );
    if invariants.flags != 0u {
        return failure_result(invariants.flags, 0u, 0.0, vec4<f32>(0.0));
    }
    if radial_capture_is_proven(initial, invariants) {
        return GeometricSample(
            TERMINATION_HORIZON,
            0u,
            EVENT_CANDIDATE_HORIZON,
            0u,
            0.0,
            vec3<f32>(0.0),
            0.0,
            vec4<f32>(invariants.values.x, 0.0, 0.0, 0.0),
            vec4<u32>(0u),
        );
    }
    return trace_initialized(initial, invariants, trace_uniforms.step_policy.xyz);
}

fn trace_pixel_with_radial_capture(
    pixel: vec2<u32>,
    extent: vec2<u32>,
) -> GeometricSample {
    return trace_pixel_with_radial_capture_at(
        pixel,
        extent,
        trace_uniforms.camera.zw,
    );
}

fn trace_accelerated_pixel(pixel: vec2<u32>, extent: vec2<u32>) -> GeometricSample {
    return trace_pixel_with_radial_capture(pixel, extent);
}

fn octahedral_nonzero_sign(value: f32) -> f32 {
    return select(-1.0, 1.0, value >= 0.0);
}

fn escape_map_pack_node(result: GeometricSample) -> u32 {
    if result.termination == TERMINATION_HORIZON {
        return ESCAPE_MAP_TAG_HORIZON << ESCAPE_MAP_TAG_SHIFT;
    }
    if result.termination != TERMINATION_ESCAPE || !finite_vec3(result.source_coordinates) {
        return 0u;
    }

    // Octahedral projection is homogeneous, so normalizing to L2 first would only add a square
    // root and another reciprocal.
    let l1_norm = dot(abs(result.source_coordinates), vec3<f32>(1.0));
    if l1_norm <= 0.0 || !finite_scalar(l1_norm) {
        return 0u;
    }
    var projected = result.source_coordinates.xy / l1_norm;
    if result.source_coordinates.z < 0.0 {
        projected = (vec2<f32>(1.0) - abs(projected.yx))
            * vec2<f32>(
                octahedral_nonzero_sign(projected.x),
                octahedral_nonzero_sign(projected.y),
            );
    }
    let encoded = clamp(0.5 * projected + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    let quantized = vec2<u32>(round(encoded * f32(ESCAPE_MAP_OCTAHEDRAL_MAXIMUM)));
    return quantized.x
        | (quantized.y << 15u)
        | (ESCAPE_MAP_TAG_ESCAPE << ESCAPE_MAP_TAG_SHIFT);
}

fn escape_map_unpack_node(packed: u32) -> vec4<f32> {
    let tag = packed >> ESCAPE_MAP_TAG_SHIFT;
    if tag == ESCAPE_MAP_TAG_HORIZON {
        return vec4<f32>(0.0, 0.0, 0.0, f32(TERMINATION_HORIZON));
    }
    if tag != ESCAPE_MAP_TAG_ESCAPE {
        return vec4<f32>(0.0);
    }

    let quantized = vec2<u32>(
        packed & ESCAPE_MAP_OCTAHEDRAL_MAXIMUM,
        (packed >> 15u) & ESCAPE_MAP_OCTAHEDRAL_MAXIMUM,
    );
    var projected = 2.0 * vec2<f32>(quantized)
        / f32(ESCAPE_MAP_OCTAHEDRAL_MAXIMUM)
        - vec2<f32>(1.0);
    let z = 1.0 - abs(projected.x) - abs(projected.y);
    if z < 0.0 {
        projected = (vec2<f32>(1.0) - abs(projected.yx))
            * vec2<f32>(
                octahedral_nonzero_sign(projected.x),
                octahedral_nonzero_sign(projected.y),
            );
    }
    return vec4<f32>(normalize(vec3<f32>(projected, z)), f32(TERMINATION_ESCAPE));
}

fn escape_map_grid_dimensions(extent: vec2<u32>) -> vec2<u32> {
    let tile_grid = (extent + vec2<u32>(ESCAPE_MAP_TILE_AXIS - 1u))
        / ESCAPE_MAP_TILE_AXIS;
    return tile_grid * 2u + vec2<u32>(1u);
}

fn escape_map_node_index(node: vec2<u32>, grid_width: u32) -> u32 {
    return node.y * grid_width + node.x;
}

fn escape_map_bilinear_direction(local_pixel: vec2<u32>) -> vec3<f32> {
    let coordinate = vec2<f32>(local_pixel) / f32(ESCAPE_MAP_TILE_AXIS);
    let upper = mix(
        escape_map_stencil[0].xyz,
        escape_map_stencil[2].xyz,
        coordinate.x,
    );
    let lower = mix(
        escape_map_stencil[6].xyz,
        escape_map_stencil[8].xyz,
        coordinate.x,
    );
    return normalize(mix(upper, lower, coordinate.y));
}

fn escape_map_interpolate_direction(local_pixel: vec2<u32>) -> vec3<f32> {
    let cell = min(local_pixel / ESCAPE_MAP_NODE_STEP, vec2<u32>(1u));
    let coordinate = vec2<f32>(local_pixel - cell * ESCAPE_MAP_NODE_STEP)
        / f32(ESCAPE_MAP_NODE_STEP);
    let first = cell.y * ESCAPE_MAP_STENCIL_AXIS + cell.x;
    let upper = mix(
        escape_map_stencil[first].xyz,
        escape_map_stencil[first + 1u].xyz,
        coordinate.x,
    );
    let lower = mix(
        escape_map_stencil[first + ESCAPE_MAP_STENCIL_AXIS].xyz,
        escape_map_stencil[first + ESCAPE_MAP_STENCIL_AXIS + 1u].xyz,
        coordinate.x,
    );
    return normalize(mix(upper, lower, coordinate.y));
}

fn escape_map_direction_matches(
    observed: vec3<f32>,
    local_pixel: vec2<u32>,
    maximum_error_squared: f32,
) -> bool {
    let predicted = escape_map_bilinear_direction(local_pixel);
    let difference = predicted - observed;
    return finite_vec3(predicted) && dot(difference, difference) <= maximum_error_squared;
}

fn escape_map_tile_is_stable(extent: vec2<u32>) -> bool {
    let escape_tag = f32(TERMINATION_ESCAPE);
    let tags0 = vec3<f32>(
        escape_map_stencil[0].w,
        escape_map_stencil[1].w,
        escape_map_stencil[2].w,
    );
    let tags1 = vec3<f32>(
        escape_map_stencil[3].w,
        escape_map_stencil[4].w,
        escape_map_stencil[5].w,
    );
    let tags2 = vec3<f32>(
        escape_map_stencil[6].w,
        escape_map_stencil[7].w,
        escape_map_stencil[8].w,
    );
    if !all(tags0 == vec3<f32>(escape_tag))
        || !all(tags1 == vec3<f32>(escape_tag))
        || !all(tags2 == vec3<f32>(escape_tag))
    {
        return false;
    }
    let half_pixel_angle = trace_uniforms.camera.x / f32(extent.y);
    let maximum_error = min(half_pixel_angle, ESCAPE_MAP_MAXIMUM_STENCIL_ERROR);
    let maximum_error_squared = maximum_error * maximum_error;
    return escape_map_direction_matches(
        escape_map_stencil[1].xyz,
        vec2<u32>(ESCAPE_MAP_NODE_STEP, 0u),
        maximum_error_squared,
    )
        && escape_map_direction_matches(
            escape_map_stencil[3].xyz,
            vec2<u32>(0u, ESCAPE_MAP_NODE_STEP),
            maximum_error_squared,
        )
        && escape_map_direction_matches(
            escape_map_stencil[4].xyz,
            vec2<u32>(ESCAPE_MAP_NODE_STEP),
            maximum_error_squared,
        )
        && escape_map_direction_matches(
            escape_map_stencil[5].xyz,
            vec2<u32>(ESCAPE_MAP_TILE_AXIS, ESCAPE_MAP_NODE_STEP),
            maximum_error_squared,
        )
        && escape_map_direction_matches(
            escape_map_stencil[7].xyz,
            vec2<u32>(ESCAPE_MAP_NODE_STEP, ESCAPE_MAP_TILE_AXIS),
            maximum_error_squared,
        );
}

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn trace_escape_map_nodes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let local_node = global_id.xy;
    let node_span = trace_dispatch.workgroup_count * 2u + vec2<u32>(1u);
    if any(local_node >= node_span) {
        return;
    }

    let extent = textureDimensions(scene_hdr);
    let grid = escape_map_grid_dimensions(extent);
    let node = trace_dispatch.tile_origin * 2u + local_node;
    if any(node >= grid) {
        return;
    }
    let pixel = node * ESCAPE_MAP_NODE_STEP;
    var packed = 0u;
    if all(pixel < extent) {
        packed = escape_map_pack_node(trace_accelerated_pixel(pixel, extent));
    }
    escape_direction_map.nodes[escape_map_node_index(node, grid.x)] = packed;
}

fn escape_map_result(
    local_id: vec2<u32>,
    workgroup_id: vec2<u32>,
    extent: vec2<u32>,
) -> GeometricSample {
    let tile = trace_dispatch.tile_origin + workgroup_id;
    let grid = escape_map_grid_dimensions(extent);
    if all(local_id == vec2<u32>(0u)) {
        escape_map_accepted = 0u;
    }
    if local_id.x < ESCAPE_MAP_STENCIL_AXIS
        && local_id.y < ESCAPE_MAP_STENCIL_AXIS
    {
        let stencil_coordinate = local_id;
        let stencil_index = stencil_coordinate.y * ESCAPE_MAP_STENCIL_AXIS
            + stencil_coordinate.x;
        let node = tile * 2u + stencil_coordinate;
        escape_map_stencil[stencil_index] = escape_map_unpack_node(
            escape_direction_map.nodes[escape_map_node_index(node, grid.x)]
        );
    }
    workgroupBarrier();
    if all(local_id == vec2<u32>(0u)) && escape_map_tile_is_stable(extent) {
        escape_map_accepted = 1u;
    }
    workgroupBarrier();

    let pixel = tile * ESCAPE_MAP_TILE_AXIS + local_id;
    if any(pixel >= extent) {
        return failure_result(FLAG_INVALID_DENOMINATOR, 0u, 0.0, vec4<f32>(0.0));
    }
    if escape_map_accepted != 0u {
        return GeometricSample(
            TERMINATION_ESCAPE,
            0u,
            EVENT_CANDIDATE_ESCAPE,
            0u,
            0.0,
            escape_map_interpolate_direction(local_id),
            0.0,
            vec4<f32>(0.0),
            vec4<u32>(0u),
        );
    }

    let reusable_x = local_id.x == 0u || local_id.x == ESCAPE_MAP_NODE_STEP;
    let reusable_y = local_id.y == 0u || local_id.y == ESCAPE_MAP_NODE_STEP;
    if reusable_x && reusable_y {
        let stencil = escape_map_stencil[
            (local_id.y / ESCAPE_MAP_NODE_STEP) * ESCAPE_MAP_STENCIL_AXIS
                + local_id.x / ESCAPE_MAP_NODE_STEP
        ];
        let termination = u32(stencil.w);
        if termination == TERMINATION_HORIZON || termination == TERMINATION_ESCAPE {
            return GeometricSample(
                termination,
                0u,
                event_candidate_mask(termination),
                0u,
                0.0,
                stencil.xyz,
                0.0,
                vec4<f32>(0.0),
                vec4<u32>(0u),
            );
        }
    }
    return trace_accelerated_pixel(pixel, extent);
}

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn trace_scene_accelerated(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let extent = textureDimensions(scene_hdr);
    let tile = trace_dispatch.tile_origin + workgroup_id.xy;
    let pixel = tile * ESCAPE_MAP_TILE_AXIS + local_id.xy;
    let result = escape_map_result(local_id.xy, workgroup_id.xy, extent);
    if any(pixel >= extent) {
        return;
    }
    store_scene_result(pixel, result.termination, result.source_coordinates);
}

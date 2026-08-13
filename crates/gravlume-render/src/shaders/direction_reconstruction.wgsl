// A compact direction reconstruction map shares the 4-pixel stencil nodes between neighboring 8x8 tiles.
// Each trace batch generates only the nodes it consumes, then either reconstructs a stable Escape
// direction or executes the exact Kerr-Schild fallback for that pixel.

struct DirectionReconstructionMap {
    nodes: array<u32>,
}

@group(0) @binding(6)
var<storage, read_write> direction_reconstruction_map: DirectionReconstructionMap;

const DIRECTION_RECONSTRUCTION_TILE_AXIS = 8u;
const DIRECTION_RECONSTRUCTION_NODE_STEP = 4u;
const DIRECTION_RECONSTRUCTION_STENCIL_AXIS = 3u;
// Leave room inside the project-wide 3.82e-4 direction budget for octahedral quantization.
const DIRECTION_RECONSTRUCTION_MAXIMUM_STENCIL_ERROR = 3.0e-4;
const DIRECTION_RECONSTRUCTION_OCTAHEDRAL_MAXIMUM = 0x7fffu;
const DIRECTION_RECONSTRUCTION_TAG_SHIFT = 30u;
const DIRECTION_RECONSTRUCTION_TAG_ESCAPE = 1u;
const DIRECTION_RECONSTRUCTION_TAG_HORIZON = 2u;
const KERR_CAPTURE_BERNSTEIN_SEGMENTS = 12u;

// The host enables this only for the direction reconstruction pipelines. Unsupported and ill-conditioned
// rays conservatively execute the full Kerr-Schild integrator.
override KERR_CAPTURE_FAST_PATH: bool = false;

struct IntervalF32 {
    lower: f32,
    upper: f32,
}

const INTERVAL_MINIMUM_NORMAL: f32 = 0x1p-126f;
const KERR_CAPTURE_INPUT_ENVELOPE: f32 = 0x1p-12f;

var<workgroup> direction_reconstruction_stencil: array<vec4<f32>, 9>;
var<workgroup> direction_reconstruction_accepted: u32;

fn kerr_capture_radius_gradient(initial: InitialState) -> vec3<f32> {
    let position = initial.state.position.yzw;
    let radius = initial.geometry.radius;
    if position.x == 0.0 && position.y == 0.0 {
        return vec3<f32>(0.0, 0.0, sign(position.z));
    }
    let spin = trace_uniforms.spacetime.y;
    let radius_squared = radius * radius;
    let sigma = radius_squared + spin * spin * position.z * position.z / radius_squared;
    return vec3<f32>(
        position.x * radius / sigma,
        position.y * radius / sigma,
        position.z * (radius_squared + spin * spin) / (radius * sigma),
    );
}

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
    return IntervalF32(value, value);
}

fn interval_one_ulp(value: f32) -> IntervalF32 {
    return IntervalF32(interval_next_lower(value), interval_next_upper(value));
}

fn interval_add(left: IntervalF32, right: IntervalF32) -> IntervalF32 {
    return IntervalF32(
        interval_next_lower(left.lower + right.lower),
        interval_next_upper(left.upper + right.upper),
    );
}

fn interval_subtract(left: IntervalF32, right: IntervalF32) -> IntervalF32 {
    return IntervalF32(
        interval_next_lower(left.lower - right.upper),
        interval_next_upper(left.upper - right.lower),
    );
}

fn interval_negate(value: IntervalF32) -> IntervalF32 {
    return IntervalF32(-value.upper, -value.lower);
}

fn interval_multiply(left: IntervalF32, right: IntervalF32) -> IntervalF32 {
    let p0 = left.lower * right.lower;
    let p1 = left.lower * right.upper;
    let p2 = left.upper * right.lower;
    let p3 = left.upper * right.upper;
    let lower = min(min(p0, p1), min(p2, p3));
    let upper = max(max(p0, p1), max(p2, p3));
    return IntervalF32(interval_next_lower(lower), interval_next_upper(upper));
}

fn interval_square(value: IntervalF32) -> IntervalF32 {
    let lower_squared = value.lower * value.lower;
    let upper_squared = value.upper * value.upper;
    if value.lower <= 0.0 && value.upper >= 0.0 {
        return IntervalF32(0.0, interval_next_upper(max(lower_squared, upper_squared)));
    }
    return IntervalF32(
        interval_next_lower(min(lower_squared, upper_squared)),
        interval_next_upper(max(lower_squared, upper_squared)),
    );
}

fn interval_scale(value: IntervalF32, scalar: f32) -> IntervalF32 {
    return interval_multiply(value, interval_point(scalar));
}

fn interval_expand_input(value: f32) -> IntervalF32 {
    let scale = interval_add(interval_point(abs(value)), interval_point(1.0));
    let padding = interval_multiply(scale, interval_point(KERR_CAPTURE_INPUT_ENVELOPE)).upper;
    return IntervalF32(
        interval_next_lower(value - padding),
        interval_next_upper(value + padding),
    );
}

fn interval_is_finite(value: IntervalF32) -> bool {
    return finite_scalar(value.lower)
        && finite_scalar(value.upper)
        && value.lower <= value.upper;
}

fn interval_kerr_capture_segment_is_positive(
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
    let one_sixth = IntervalF32(
        bitcast<f32>(0x3e2aaaaau),
        bitcast<f32>(0x3e2aaaacu),
    );
    let coefficients = array<IntervalF32, 5>(
        power0,
        interval_add(power0, interval_scale(power1, 0.25)),
        interval_add(
            interval_add(power0, interval_scale(power1, 0.5)),
            interval_multiply(power2, one_sixth),
        ),
        interval_add(
            interval_add(power0, interval_scale(power1, 0.75)),
            interval_add(
                interval_scale(power2, 0.5),
                interval_scale(power3, 0.25),
            ),
        ),
        interval_add(
            interval_add(power0, power1),
            interval_add(interval_add(power2, power3), power4),
        ),
    );
    for (var index = 0u; index < 5u; index += 1u) {
        if !interval_is_finite(coefficients[index]) || coefficients[index].lower <= 0.0 {
            return false;
        }
    }
    return true;
}

fn kerr_capture_certificate(
    initial: InitialState,
    invariants: Invariants,
) -> bool {
    let spin_value = trace_uniforms.spacetime.y;
    let horizon_value = trace_uniforms.event_surfaces.z;
    let observer_radius_value = initial.geometry.radius;
    let observer_cylindrical_radius = length(initial.state.position.yz);
    let future_radial_derivative = dot(
        kerr_capture_radius_gradient(initial),
        initial.rhs.position.yzw,
    );
    if trace_uniforms.spacetime.x != 1.0
        || trace_uniforms.spacetime.z != 0.0
        || abs(spin_value) > 0.9
        || horizon_value <= 0.0
        || observer_radius_value <= horizon_value
        || observer_radius_value > 256.0
        || observer_cylindrical_radius <= 0.0625 * observer_radius_value
        || future_radial_derivative <= KERR_CAPTURE_INPUT_ENVELOPE
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
    let shifted = interval_subtract(angular_momentum, interval_multiply(spin, energy));
    let separation = interval_add(interval_square(shifted), carter);
    if !interval_is_finite(separation) || separation.lower <= 0.0 {
        return false;
    }
    let leading = interval_square(energy);
    let quadratic = interval_subtract(
        interval_negate(interval_scale(interval_multiply(interval_multiply(energy, spin), shifted), 2.0)),
        separation,
    );
    let linear = interval_scale(separation, 2.0);
    let constant = interval_negate(interval_multiply(interval_square(spin), carter));
    let horizon = interval_one_ulp(horizon_value);
    let observer_radius = interval_expand_input(observer_radius_value);
    let one_twelfth = IntervalF32(
        bitcast<f32>(0x3daaaaaau),
        bitcast<f32>(0x3daaaaacu),
    );
    let width = interval_multiply(interval_subtract(observer_radius, horizon), one_twelfth);
    if !interval_is_finite(width) || width.lower <= 0.0 {
        return false;
    }
    for (var segment = 0u; segment < KERR_CAPTURE_BERNSTEIN_SEGMENTS; segment += 1u) {
        let lower = interval_add(horizon, interval_scale(width, f32(segment)));
        if !interval_kerr_capture_segment_is_positive(
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

fn trace_pixel_with_kerr_capture_at(
    pixel: vec2<u32>,
    extent: vec2<u32>,
    subpixel: vec2<f32>,
) -> TraceResult {
    let initial = initial_state_at(pixel, extent, subpixel);
    if initial.rhs.flags != 0u {
        return failure_result(initial.rhs.flags, 0u, 0.0, vec4<f32>(0.0));
    }
    let invariants = invariants_from_geometry_rhs(initial.state, initial.geometry, initial.rhs);
    if invariants.flags != 0u {
        return failure_result(invariants.flags, 0u, 0.0, vec4<f32>(0.0));
    }
    if kerr_capture_certificate(initial, invariants) {
        return TraceResult(
            TERMINATION_HORIZON,
            0u,
            0u,
            0.0,
            vec3<f32>(0.0),
            0.0,
            vec4<f32>(invariants.values.x, 0.0, 0.0, 0.0),
        );
    }
    return trace_initialized(initial, invariants);
}

fn trace_pixel_with_kerr_capture(
    pixel: vec2<u32>,
    extent: vec2<u32>,
) -> TraceResult {
    return trace_pixel_with_kerr_capture_at(
        pixel,
        extent,
        trace_uniforms.view.zw,
    );
}

fn direction_reconstruction_trace_pixel(pixel: vec2<u32>, extent: vec2<u32>) -> TraceResult {
    if KERR_CAPTURE_FAST_PATH {
        return trace_pixel_with_kerr_capture(pixel, extent);
    }
    return trace_pixel(pixel, extent);
}

fn direction_reconstruction_nonzero_sign(value: f32) -> f32 {
    return select(-1.0, 1.0, value >= 0.0);
}

fn direction_reconstruction_pack_node(result: TraceResult) -> u32 {
    if result.termination == TERMINATION_HORIZON {
        return DIRECTION_RECONSTRUCTION_TAG_HORIZON << DIRECTION_RECONSTRUCTION_TAG_SHIFT;
    }
    if result.termination != TERMINATION_ESCAPE || !finite_vec3(result.direction) {
        return 0u;
    }

    let unit = normalize(result.direction);
    let inverse_l1 = 1.0 / (abs(unit.x) + abs(unit.y) + abs(unit.z));
    var projected = unit.xy * inverse_l1;
    if unit.z < 0.0 {
        projected = (vec2<f32>(1.0) - abs(projected.yx))
            * vec2<f32>(
                direction_reconstruction_nonzero_sign(projected.x),
                direction_reconstruction_nonzero_sign(projected.y),
            );
    }
    let encoded = clamp(0.5 * projected + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    let quantized = vec2<u32>(round(encoded * f32(DIRECTION_RECONSTRUCTION_OCTAHEDRAL_MAXIMUM)));
    return quantized.x
        | (quantized.y << 15u)
        | (DIRECTION_RECONSTRUCTION_TAG_ESCAPE << DIRECTION_RECONSTRUCTION_TAG_SHIFT);
}

fn direction_reconstruction_unpack_node(packed: u32) -> vec4<f32> {
    let tag = packed >> DIRECTION_RECONSTRUCTION_TAG_SHIFT;
    if tag == DIRECTION_RECONSTRUCTION_TAG_HORIZON {
        return vec4<f32>(0.0, 0.0, 0.0, f32(TERMINATION_HORIZON));
    }
    if tag != DIRECTION_RECONSTRUCTION_TAG_ESCAPE {
        return vec4<f32>(0.0);
    }

    let quantized = vec2<u32>(
        packed & DIRECTION_RECONSTRUCTION_OCTAHEDRAL_MAXIMUM,
        (packed >> 15u) & DIRECTION_RECONSTRUCTION_OCTAHEDRAL_MAXIMUM,
    );
    var projected = 2.0 * vec2<f32>(quantized)
        / f32(DIRECTION_RECONSTRUCTION_OCTAHEDRAL_MAXIMUM)
        - vec2<f32>(1.0);
    let z = 1.0 - abs(projected.x) - abs(projected.y);
    if z < 0.0 {
        projected = (vec2<f32>(1.0) - abs(projected.yx))
            * vec2<f32>(
                direction_reconstruction_nonzero_sign(projected.x),
                direction_reconstruction_nonzero_sign(projected.y),
            );
    }
    return vec4<f32>(normalize(vec3<f32>(projected, z)), f32(TERMINATION_ESCAPE));
}

fn direction_reconstruction_grid_dimensions(extent: vec2<u32>) -> vec2<u32> {
    let tile_grid = (extent + vec2<u32>(DIRECTION_RECONSTRUCTION_TILE_AXIS - 1u))
        / DIRECTION_RECONSTRUCTION_TILE_AXIS;
    return tile_grid * 2u + vec2<u32>(1u);
}

fn direction_reconstruction_node_index(node: vec2<u32>, grid_width: u32) -> u32 {
    return node.y * grid_width + node.x;
}

fn direction_reconstruction_bilinear_direction(local_pixel: vec2<u32>) -> vec3<f32> {
    let coordinate = vec2<f32>(local_pixel) / f32(DIRECTION_RECONSTRUCTION_TILE_AXIS);
    let upper = mix(
        direction_reconstruction_stencil[0].xyz,
        direction_reconstruction_stencil[2].xyz,
        coordinate.x,
    );
    let lower = mix(
        direction_reconstruction_stencil[6].xyz,
        direction_reconstruction_stencil[8].xyz,
        coordinate.x,
    );
    return normalize(mix(upper, lower, coordinate.y));
}

fn direction_reconstruction_piecewise_direction(local_pixel: vec2<u32>) -> vec3<f32> {
    let cell = min(local_pixel / DIRECTION_RECONSTRUCTION_NODE_STEP, vec2<u32>(1u));
    let coordinate = vec2<f32>(local_pixel - cell * DIRECTION_RECONSTRUCTION_NODE_STEP)
        / f32(DIRECTION_RECONSTRUCTION_NODE_STEP);
    let first = cell.y * DIRECTION_RECONSTRUCTION_STENCIL_AXIS + cell.x;
    let upper = mix(
        direction_reconstruction_stencil[first].xyz,
        direction_reconstruction_stencil[first + 1u].xyz,
        coordinate.x,
    );
    let lower = mix(
        direction_reconstruction_stencil[first + DIRECTION_RECONSTRUCTION_STENCIL_AXIS].xyz,
        direction_reconstruction_stencil[first + DIRECTION_RECONSTRUCTION_STENCIL_AXIS + 1u].xyz,
        coordinate.x,
    );
    return normalize(mix(upper, lower, coordinate.y));
}

fn direction_reconstruction_interpolate_direction(local_pixel: vec2<u32>) -> vec3<f32> {
    return direction_reconstruction_piecewise_direction(local_pixel);
}

fn direction_reconstruction_direction_matches(
    stencil_index: u32,
    local_pixel: vec2<u32>,
    maximum_error_squared: f32,
) -> bool {
    let predicted = direction_reconstruction_bilinear_direction(local_pixel);
    let difference = predicted - direction_reconstruction_stencil[stencil_index].xyz;
    return finite_vec3(predicted) && dot(difference, difference) <= maximum_error_squared;
}

fn direction_reconstruction_tile_is_stable_escape(extent: vec2<u32>) -> bool {
    for (var index = 0u; index < 9u; index += 1u) {
        if u32(direction_reconstruction_stencil[index].w) != TERMINATION_ESCAPE
            || !finite_vec3(direction_reconstruction_stencil[index].xyz)
        {
            return false;
        }
    }
    let half_pixel_angle = trace_uniforms.view.x / f32(extent.y);
    let maximum_error = min(half_pixel_angle, DIRECTION_RECONSTRUCTION_MAXIMUM_STENCIL_ERROR);
    let maximum_error_squared = maximum_error * maximum_error;
    return direction_reconstruction_direction_matches(
        1u,
        vec2<u32>(DIRECTION_RECONSTRUCTION_NODE_STEP, 0u),
        maximum_error_squared,
    )
        && direction_reconstruction_direction_matches(
            3u,
            vec2<u32>(0u, DIRECTION_RECONSTRUCTION_NODE_STEP),
            maximum_error_squared,
        )
        && direction_reconstruction_direction_matches(
            4u,
            vec2<u32>(DIRECTION_RECONSTRUCTION_NODE_STEP),
            maximum_error_squared,
        )
        && direction_reconstruction_direction_matches(
            5u,
            vec2<u32>(DIRECTION_RECONSTRUCTION_TILE_AXIS, DIRECTION_RECONSTRUCTION_NODE_STEP),
            maximum_error_squared,
        )
        && direction_reconstruction_direction_matches(
            7u,
            vec2<u32>(DIRECTION_RECONSTRUCTION_NODE_STEP, DIRECTION_RECONSTRUCTION_TILE_AXIS),
            maximum_error_squared,
        );
}

@compute @workgroup_size(8, 8, 1)
fn trace_direction_reconstruction_nodes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let local_node = global_id.xy;
    let node_span = trace_dispatch.tile_region.zw * 2u + vec2<u32>(1u);
    if any(local_node >= node_span) {
        return;
    }

    let extent = textureDimensions(scene_hdr);
    let grid = direction_reconstruction_grid_dimensions(extent);
    let node = trace_dispatch.tile_region.xy * 2u + local_node;
    if any(node >= grid) {
        return;
    }
    let pixel = node * DIRECTION_RECONSTRUCTION_NODE_STEP;
    var packed = 0u;
    if all(pixel < extent) {
        packed = direction_reconstruction_pack_node(direction_reconstruction_trace_pixel(pixel, extent));
    }
    direction_reconstruction_map.nodes[direction_reconstruction_node_index(node, grid.x)] = packed;
}

fn direction_reconstruction_result(
    local_id: vec2<u32>,
    workgroup_id: vec2<u32>,
    extent: vec2<u32>,
) -> TraceResult {
    let tile = trace_dispatch.tile_region.xy + workgroup_id;
    let grid = direction_reconstruction_grid_dimensions(extent);
    if all(local_id == vec2<u32>(0u)) {
        direction_reconstruction_accepted = 0u;
    }
    if local_id.x < DIRECTION_RECONSTRUCTION_STENCIL_AXIS
        && local_id.y < DIRECTION_RECONSTRUCTION_STENCIL_AXIS
    {
        let stencil_coordinate = local_id;
        let stencil_index = stencil_coordinate.y * DIRECTION_RECONSTRUCTION_STENCIL_AXIS
            + stencil_coordinate.x;
        let node = tile * 2u + stencil_coordinate;
        direction_reconstruction_stencil[stencil_index] = direction_reconstruction_unpack_node(
            direction_reconstruction_map.nodes[direction_reconstruction_node_index(node, grid.x)]
        );
    }
    workgroupBarrier();
    if all(local_id == vec2<u32>(0u)) && direction_reconstruction_tile_is_stable_escape(extent) {
        direction_reconstruction_accepted = 1u;
    }
    workgroupBarrier();

    let pixel = tile * DIRECTION_RECONSTRUCTION_TILE_AXIS + local_id;
    if any(pixel >= extent) {
        return failure_result(FLAG_INVALID_DENOMINATOR, 0u, 0.0, vec4<f32>(0.0));
    }
    if direction_reconstruction_accepted != 0u {
        return TraceResult(
            TERMINATION_ESCAPE,
            0u,
            0u,
            0.0,
            direction_reconstruction_interpolate_direction(local_id),
            0.0,
            vec4<f32>(0.0),
        );
    }

    let reusable_x = local_id.x == 0u || local_id.x == DIRECTION_RECONSTRUCTION_NODE_STEP;
    let reusable_y = local_id.y == 0u || local_id.y == DIRECTION_RECONSTRUCTION_NODE_STEP;
    if reusable_x && reusable_y {
        let stencil = direction_reconstruction_stencil[
            (local_id.y / DIRECTION_RECONSTRUCTION_NODE_STEP) * DIRECTION_RECONSTRUCTION_STENCIL_AXIS
                + local_id.x / DIRECTION_RECONSTRUCTION_NODE_STEP
        ];
        let termination = u32(stencil.w);
        if termination == TERMINATION_HORIZON || termination == TERMINATION_ESCAPE {
            return TraceResult(
                termination,
                0u,
                0u,
                0.0,
                stencil.xyz,
                0.0,
                vec4<f32>(0.0),
            );
        }
    }
    return direction_reconstruction_trace_pixel(pixel, extent);
}

@compute @workgroup_size(8, 8, 1)
fn trace_scene_direction_reconstruction(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let extent = textureDimensions(scene_hdr);
    let tile = trace_dispatch.tile_region.xy + workgroup_id.xy;
    let pixel = tile * DIRECTION_RECONSTRUCTION_TILE_AXIS + local_id.xy;
    let result = direction_reconstruction_result(local_id.xy, workgroup_id.xy, extent);
    if any(pixel >= extent) {
        return;
    }
    store_scene_result(pixel, result.termination, result.direction);
}

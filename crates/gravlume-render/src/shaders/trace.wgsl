struct TraceUniforms {
    spacetime: vec4<f32>,
    observer_event: vec4<f32>,
    observer_velocity: vec4<f32>,
    image_right: vec4<f32>,
    image_up: vec4<f32>,
    arrival: vec4<f32>,
    projection: vec4<f32>,
    event_surfaces: vec4<f32>,
    step_policy: vec4<f32>,
}

struct TraceDispatch {
    pixels: vec4<u32>,
}

struct Geometry {
    radius: f32,
    radius_gradient: vec3<f32>,
    scalar_f: f32,
    scalar_f_gradient: vec3<f32>,
    null_covector: vec4<f32>,
    null_vector: vec4<f32>,
    null_dx: vec4<f32>,
    null_dy: vec4<f32>,
    null_dz: vec4<f32>,
    singularity_measure: f32,
    flags: u32,
}

struct TraceState {
    position: vec4<f32>,
    momentum: vec4<f32>,
}

struct RhsResult {
    position: vec4<f32>,
    momentum: vec4<f32>,
    flags: u32,
}

struct StepResult {
    state: TraceState,
    flags: u32,
}

struct InitialState {
    state: TraceState,
    sight: vec4<f32>,
    geometry: Geometry,
    rhs: RhsResult,
    null_residual: f32,
    flags: u32,
}

struct Invariants {
    values: vec4<f32>,
    flags: u32,
}

const TERMINATION_HORIZON: u32 = 1u;
const TERMINATION_ESCAPE: u32 = 2u;
const TERMINATION_SINGULARITY: u32 = 3u;
const TERMINATION_STEP_EXHAUSTION: u32 = 4u;
const TERMINATION_NUMERICAL_FAILURE: u32 = 5u;
const TERMINATION_UNCERTAIN: u32 = 6u;
const FLAG_NON_FINITE: u32 = 1u;
const FLAG_INVALID_RADICAND: u32 = 2u;
const FLAG_INVALID_DENOMINATOR: u32 = 4u;
const MAXIMUM_STEPS: u32 = 2048u;
const MAXIMUM_FINITE_F32: f32 = 0x1.fffffep+127f;

@group(0) @binding(0)
var<uniform> trace_uniforms: TraceUniforms;

@group(0) @binding(1)
var scene_hdr: texture_storage_2d<rgba16float, write>;

@group(0) @binding(2)
var<storage, read_write> trace_direction_time: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> trace_invariant_drift: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> trace_metadata: array<vec4<u32>>;

@group(0) @binding(5)
var<uniform> trace_dispatch: TraceDispatch;

fn finite_scalar(value: f32) -> bool {
    return value == value && abs(value) <= MAXIMUM_FINITE_F32;
}

fn finite_vec4(value: vec4<f32>) -> bool {
    return all(value == value) && all(abs(value) <= vec4<f32>(MAXIMUM_FINITE_F32));
}

fn finite_vec3(value: vec3<f32>) -> bool {
    return all(value == value) && all(abs(value) <= vec3<f32>(MAXIMUM_FINITE_F32));
}

// WGSL runtime overflow may produce an indeterminate value under the finite-math assumption, so
// guard the multiplication rather than trying to clamp its result.
// Source: https://www.w3.org/TR/WGSL/#floating-point-evaluation
fn saturating_positive_product(left: f32, right: f32) -> f32 {
    if left == 0.0 || right == 0.0 {
        return 0.0;
    }
    if left > 1.0 && right > MAXIMUM_FINITE_F32 / left {
        return MAXIMUM_FINITE_F32;
    }
    return left * right;
}

fn saturating_positive_sum(left: f32, right: f32) -> f32 {
    if left > MAXIMUM_FINITE_F32 - right {
        return MAXIMUM_FINITE_F32;
    }
    return left + right;
}

fn positive_square(value: f32) -> f32 {
    return saturating_positive_product(abs(value), abs(value));
}

fn singularity_measure(radius: f32, spin: f32, z: f32) -> f32 {
    let radius_squared = positive_square(radius);
    let radius_fourth = positive_square(radius_squared);
    let spin_z = saturating_positive_product(abs(spin), abs(z));
    return saturating_positive_sum(radius_fourth, positive_square(spin_z));
}

fn invalid_geometry(flags: u32) -> Geometry {
    return Geometry(
        0.0,
        vec3<f32>(0.0),
        0.0,
        vec3<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        0.0,
        flags,
    );
}

fn geometry_at(position: vec3<f32>) -> Geometry {
    let spin_physical = trace_uniforms.spacetime.y;
    let charge_physical = trace_uniforms.spacetime.z;
    let scale = max(
        max(max(abs(position.x), abs(position.y)), abs(position.z)),
        abs(spin_physical),
    );
    if !finite_scalar(scale) || scale == 0.0 {
        return invalid_geometry(FLAG_INVALID_DENOMINATOR);
    }

    let coordinates = position / scale;
    let x = coordinates.x;
    let y = coordinates.y;
    let z = coordinates.z;
    let spin = spin_physical / scale;
    let spin_squared = spin * spin;
    var radius_squared = 0.0;
    var radius = 0.0;
    var sigma = 0.0;

    if x == 0.0 && y == 0.0 {
        radius = abs(z);
        radius_squared = radius * radius;
        sigma = radius_squared + spin_squared;
        if radius <= 0.0 || sigma <= 0.0 || !finite_scalar(sigma) {
            return invalid_geometry(FLAG_INVALID_DENOMINATOR);
        }
        let axis_sign = sign(z);
        let radius_gradient = vec3<f32>(0.0, 0.0, axis_sign);
        let mass = trace_uniforms.spacetime.x / scale;
        let charge = charge_physical / scale;
        let numerator = 2.0 * mass * radius - charge * charge;
        let scalar_f = numerator / sigma;
        let sigma_gradient = vec3<f32>(0.0, 0.0, 2.0 * radius * axis_sign);
        let numerator_gradient = 2.0 * mass * radius_gradient;
        let scalar_f_gradient =
            (numerator_gradient * sigma - sigma_gradient * numerator) / (sigma * sigma * scale);
        let null_covector = vec4<f32>(1.0, 0.0, 0.0, axis_sign);
        let null_vector = vec4<f32>(-1.0, 0.0, 0.0, axis_sign);
        let inverse_denominator_scale = 1.0 / (sigma * scale);
        let null_dx = vec4<f32>(
            0.0,
            radius * inverse_denominator_scale,
            -spin * inverse_denominator_scale,
            0.0,
        );
        let null_dy = vec4<f32>(
            0.0,
            spin * inverse_denominator_scale,
            radius * inverse_denominator_scale,
            0.0,
        );
        let physical_radius = scale * radius;
        let singularity = singularity_measure(physical_radius, spin_physical, position.z);
        if !finite_scalar(physical_radius)
            || !finite_scalar(scalar_f)
            || !finite_vec3(scalar_f_gradient)
            || !finite_scalar(singularity)
        {
            return invalid_geometry(FLAG_NON_FINITE);
        }
        return Geometry(
            physical_radius,
            radius_gradient,
            scalar_f,
            scalar_f_gradient,
            null_covector,
            null_vector,
            null_dx,
            null_dy,
            vec4<f32>(0.0),
            singularity,
            0u,
        );
    }

    let radius_squared_3d = dot(coordinates, coordinates);
    let b = radius_squared_3d - spin_squared;
    let radicand = b * b + 4.0 * spin_squared * z * z;
    if radicand < 0.0 || !finite_scalar(radicand) {
        return invalid_geometry(FLAG_INVALID_RADICAND);
    }
    let root = sqrt(radicand);
    if b >= 0.0 {
        radius_squared = 0.5 * (b + root);
    } else {
        let denominator = root - b;
        if denominator <= 0.0 || !finite_scalar(denominator) {
            return invalid_geometry(FLAG_INVALID_DENOMINATOR);
        }
        radius_squared = 2.0 * spin_squared * z * z / denominator;
    }
    if radius_squared <= 0.0 || !finite_scalar(radius_squared) {
        return invalid_geometry(FLAG_INVALID_DENOMINATOR);
    }
    radius = sqrt(radius_squared);
    sigma = radius_squared + spin_squared * z * z / radius_squared;
    if radius <= 0.0 || sigma <= 0.0 || !finite_scalar(sigma) {
        return invalid_geometry(FLAG_INVALID_DENOMINATOR);
    }

    let radius_gradient = vec3<f32>(
        x * radius / sigma,
        y * radius / sigma,
        z * (radius_squared + spin_squared) / (radius * sigma),
    );
    let mass = trace_uniforms.spacetime.x / scale;
    let charge = charge_physical / scale;
    let numerator = 2.0 * mass * radius - charge * charge;
    let scalar_f = numerator / sigma;
    let vertical_sigma = spin_squared * z * z / radius_squared;
    let sigma_radius_factor = 2.0 * (radius - vertical_sigma / radius);
    let sigma_gradient = vec3<f32>(
        sigma_radius_factor * radius_gradient.x,
        sigma_radius_factor * radius_gradient.y,
        sigma_radius_factor * radius_gradient.z + 2.0 * spin_squared * z / radius_squared,
    );
    let numerator_gradient = 2.0 * mass * radius_gradient;
    let scalar_f_gradient =
        (numerator_gradient * sigma - sigma_gradient * numerator) / (sigma * sigma * scale);
    let radial_denominator = radius_squared + spin_squared;
    if radial_denominator <= 0.0 || !finite_scalar(radial_denominator) {
        return invalid_geometry(FLAG_INVALID_DENOMINATOR);
    }
    let null_spatial = vec3<f32>(
        (radius * x + spin * y) / radial_denominator,
        (radius * y - spin * x) / radial_denominator,
        z / radius,
    );
    let null_covector = vec4<f32>(1.0, null_spatial);
    let null_vector = vec4<f32>(-1.0, null_spatial);
    var null_gradients = array<vec4<f32>, 3>();
    for (var index = 0u; index < 3u; index += 1u) {
        let radius_i = radius_gradient[index];
        let delta_x = select(0.0, 1.0, index == 0u);
        let delta_y = select(0.0, 1.0, index == 1u);
        let delta_z = select(0.0, 1.0, index == 2u);
        let numerator_x = radius_i * x + radius * delta_x + spin * delta_y;
        let numerator_y = radius_i * y + radius * delta_y - spin * delta_x;
        let radial_derivative = 2.0 * radius * radius_i;
        let derivative_x =
            (numerator_x / radial_denominator - null_spatial.x * radial_derivative / radial_denominator)
            / scale;
        let derivative_y =
            (numerator_y / radial_denominator - null_spatial.y * radial_derivative / radial_denominator)
            / scale;
        let derivative_z = (delta_z / radius - coordinates.z * radius_i / radius_squared) / scale;
        null_gradients[index] = vec4<f32>(0.0, derivative_x, derivative_y, derivative_z);
    }
    let physical_radius = scale * radius;
    let singularity = singularity_measure(physical_radius, spin_physical, position.z);
    if !finite_scalar(physical_radius)
        || !finite_scalar(scalar_f)
        || !finite_vec3(scalar_f_gradient)
        || !finite_vec4(null_covector)
        || !finite_vec4(null_gradients[0])
        || !finite_vec4(null_gradients[1])
        || !finite_vec4(null_gradients[2])
        || !finite_scalar(singularity)
    {
        return invalid_geometry(FLAG_NON_FINITE);
    }
    return Geometry(
        physical_radius,
        radius_gradient,
        scalar_f,
        scalar_f_gradient,
        null_covector,
        null_vector,
        null_gradients[0],
        null_gradients[1],
        null_gradients[2],
        singularity,
        0u,
    );
}

fn sight_direction(pixel: vec2<u32>, extent: vec2<u32>, subpixel: vec2<f32>) -> vec4<f32> {
    let dimensions = vec2<f32>(extent);
    let normalized = vec2<f32>(
        2.0 * (f32(pixel.x) + subpixel.x) / dimensions.x - 1.0,
        1.0 - 2.0 * (f32(pixel.y) + subpixel.y) / dimensions.y,
    );
    let tangent_half_fov = trace_uniforms.projection.x;
    let sight_plane = vec2<f32>(
        dimensions.x / dimensions.y * tangent_half_fov * normalized.x,
        tangent_half_fov * normalized.y,
    );
    let inverse_length = inverseSqrt(1.0 + dot(sight_plane, sight_plane));
    return (
        sight_plane.x * trace_uniforms.image_right
        + sight_plane.y * trace_uniforms.image_up
        - trace_uniforms.arrival
    ) * inverse_length;
}

fn lower_momentum(momentum_contravariant: vec4<f32>, geometry: Geometry) -> vec4<f32> {
    let minkowski_lowered = vec4<f32>(
        -momentum_contravariant.x,
        momentum_contravariant.yzw,
    );
    let contraction = dot(geometry.null_covector, momentum_contravariant);
    return minkowski_lowered
        + geometry.scalar_f * contraction * geometry.null_covector;
}

fn normalized_null_residual(
    momentum_covariant: vec4<f32>,
    momentum_contravariant: vec4<f32>,
) -> f32 {
    let contraction = dot(momentum_covariant, momentum_contravariant);
    let term_norm = dot(abs(momentum_covariant), abs(momentum_contravariant));
    return abs(contraction) / max(1.0, term_norm);
}

fn invalid_rhs(flags: u32) -> RhsResult {
    return RhsResult(vec4<f32>(0.0), vec4<f32>(0.0), flags);
}

fn hamilton_rhs_from_geometry(state: TraceState, geometry: Geometry) -> RhsResult {
    if geometry.flags != 0u {
        return invalid_rhs(geometry.flags);
    }
    let contraction = dot(state.momentum, geometry.null_vector);
    let minkowski_raised = vec4<f32>(
        -state.momentum.x,
        state.momentum.yzw,
    );
    let position_derivative = minkowski_raised
        - geometry.scalar_f * contraction * geometry.null_vector;
    let null_derivative_contraction = vec3<f32>(
        dot(state.momentum, geometry.null_dx),
        dot(state.momentum, geometry.null_dy),
        dot(state.momentum, geometry.null_dz),
    );
    let momentum_spatial = geometry.scalar_f * contraction * null_derivative_contraction
        + 0.5 * contraction * contraction * geometry.scalar_f_gradient;
    let momentum_derivative = vec4<f32>(0.0, momentum_spatial);
    if !finite_vec4(position_derivative) || !finite_vec4(momentum_derivative) {
        return invalid_rhs(FLAG_NON_FINITE);
    }
    return RhsResult(position_derivative, momentum_derivative, 0u);
}

fn hamilton_rhs(state: TraceState) -> RhsResult {
    return hamilton_rhs_from_geometry(state, geometry_at(state.position.yzw));
}

fn state_add(state: TraceState, derivative: RhsResult, factor: f32) -> TraceState {
    return TraceState(
        state.position + factor * derivative.position,
        state.momentum + factor * derivative.momentum,
    );
}

fn invalid_step(state: TraceState, flags: u32) -> StepResult {
    return StepResult(state, flags);
}

fn rk4_step(state: TraceState, k1: RhsResult, step: f32) -> StepResult {
    if k1.flags != 0u {
        return invalid_step(state, k1.flags);
    }
    let k2 = hamilton_rhs(state_add(state, k1, 0.5 * step));
    if k2.flags != 0u {
        return invalid_step(state, k2.flags);
    }
    let k3 = hamilton_rhs(state_add(state, k2, 0.5 * step));
    if k3.flags != 0u {
        return invalid_step(state, k3.flags);
    }
    let k4 = hamilton_rhs(state_add(state, k3, step));
    if k4.flags != 0u {
        return invalid_step(state, k4.flags);
    }
    let next = TraceState(
        state.position
            + step / 6.0 * (k1.position + 2.0 * k2.position + 2.0 * k3.position + k4.position),
        state.momentum
            + step / 6.0 * (k1.momentum + 2.0 * k2.momentum + 2.0 * k3.momentum + k4.momentum),
    );
    if !finite_vec4(next.position) || !finite_vec4(next.momentum) {
        return invalid_step(state, FLAG_NON_FINITE);
    }
    return StepResult(next, 0u);
}

fn dense_state_at(
    start: TraceState,
    end: TraceState,
    start_derivative: RhsResult,
    end_derivative: RhsResult,
    step: f32,
    step_fraction: f32,
) -> TraceState {
    let fraction_squared = step_fraction * step_fraction;
    let fraction_cubed = fraction_squared * step_fraction;
    let start_weight = 2.0 * fraction_cubed - 3.0 * fraction_squared + 1.0;
    let start_derivative_weight = fraction_cubed - 2.0 * fraction_squared + step_fraction;
    let end_weight = -2.0 * fraction_cubed + 3.0 * fraction_squared;
    let end_derivative_weight = fraction_cubed - fraction_squared;
    return TraceState(
        start_weight * start.position
            + step * start_derivative_weight * start_derivative.position
            + end_weight * end.position
            + step * end_derivative_weight * end_derivative.position,
        start_weight * start.momentum
            + step * start_derivative_weight * start_derivative.momentum
            + end_weight * end.momentum
            + step * end_derivative_weight * end_derivative.momentum,
    );
}

fn initial_state_at(pixel: vec2<u32>, extent: vec2<u32>, subpixel: vec2<f32>) -> InitialState {
    let sight = sight_direction(pixel, extent, subpixel);
    let arrival = -sight;
    let momentum_contravariant = trace_uniforms.projection.y
        * (trace_uniforms.observer_velocity + arrival);
    let geometry = geometry_at(trace_uniforms.observer_event.yzw);
    if geometry.flags != 0u || !finite_vec4(momentum_contravariant) {
        return InitialState(
            TraceState(trace_uniforms.observer_event, vec4<f32>(0.0)),
            sight,
            geometry,
            invalid_rhs(geometry.flags | FLAG_NON_FINITE),
            0.0,
            geometry.flags | FLAG_NON_FINITE,
        );
    }
    let momentum_covariant = lower_momentum(momentum_contravariant, geometry);
    let null_residual = normalized_null_residual(momentum_covariant, momentum_contravariant);
    if !finite_vec4(momentum_covariant) || !finite_scalar(null_residual) {
        return InitialState(
            TraceState(trace_uniforms.observer_event, vec4<f32>(0.0)),
            sight,
            geometry,
            invalid_rhs(FLAG_NON_FINITE),
            0.0,
            FLAG_NON_FINITE,
        );
    }
    let state = TraceState(trace_uniforms.observer_event, momentum_covariant);
    let rhs = hamilton_rhs_from_geometry(state, geometry);
    return InitialState(
        state,
        sight,
        geometry,
        rhs,
        null_residual,
        rhs.flags,
    );
}

fn invariants_from_geometry_rhs(
    state: TraceState,
    geometry: Geometry,
    rhs: RhsResult,
) -> Invariants {
    if geometry.flags != 0u || rhs.flags != 0u {
        return Invariants(vec4<f32>(0.0), geometry.flags | rhs.flags);
    }
    let energy = -state.momentum.x;
    let x = state.position.y;
    let y = state.position.z;
    let z = state.position.w;
    let angular_momentum_z = x * state.momentum.z - y * state.momentum.y;
    let spin = trace_uniforms.spacetime.y;
    let spin_squared = spin * spin;
    let radius = geometry.radius;
    let rho = length(vec2<f32>(x, y));
    var carter = 0.0;
    if rho == 0.0 {
        let transverse_momentum_squared = dot(state.momentum.yz, state.momentum.yz);
        carter = (radius * radius + spin_squared) * transverse_momentum_squared
            - spin_squared * energy * energy;
    } else {
        let cos_theta = z / radius;
        let sin_theta = rho / sqrt(radius * radius + spin_squared);
        if sin_theta == 0.0 || !finite_scalar(sin_theta) {
            return Invariants(vec4<f32>(0.0), FLAG_INVALID_DENOMINATOR);
        }
        let projected_momentum = x * state.momentum.y + y * state.momentum.z;
        let p_theta = -radius * sin_theta * state.momentum.w
            + cos_theta / sin_theta * projected_momentum;
        let lz_over_sin = angular_momentum_z / sin_theta;
        carter = p_theta * p_theta
            + cos_theta * cos_theta
                * (lz_over_sin * lz_over_sin - spin_squared * energy * energy);
    }
    let null_residual = normalized_null_residual(state.momentum, rhs.position);
    let values = vec4<f32>(null_residual, energy, angular_momentum_z, carter);
    if !finite_vec4(values) {
        return Invariants(vec4<f32>(0.0), FLAG_NON_FINITE);
    }
    return Invariants(values, 0u);
}

fn invariants(state: TraceState) -> Invariants {
    let geometry = geometry_at(state.position.yzw);
    return invariants_from_geometry_rhs(
        state,
        geometry,
        hamilton_rhs_from_geometry(state, geometry),
    );
}

fn invariant_drift(initial: vec4<f32>, current: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        current.x,
        abs(current.y - initial.y) / max(1.0, abs(initial.y)),
        abs(current.z - initial.z) / max(1.0, abs(initial.z)),
        abs(current.w - initial.w) / max(1.0, abs(initial.w)),
    );
}

fn invariant_budget_exceeded(maximum_drift: vec4<f32>) -> bool {
    return any(maximum_drift > vec4<f32>(trace_uniforms.step_policy.w));
}

fn visible_failure_color(termination: u32) -> vec4<f32> {
    if termination == TERMINATION_SINGULARITY {
        return vec4<f32>(0.0, 1.0, 1.0, 1.0);
    }
    if termination == TERMINATION_STEP_EXHAUSTION {
        return vec4<f32>(1.0, 0.25, 0.0, 1.0);
    }
    if termination == TERMINATION_UNCERTAIN {
        return vec4<f32>(1.0, 1.0, 0.0, 1.0);
    }
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}

fn analytic_sky(direction: vec3<f32>) -> vec3<f32> {
    let unit = normalize(direction);
    let longitude = atan2(unit.y, unit.x);
    let latitude = asin(clamp(unit.z, -1.0, 1.0));
    let longitude_grid = 1.0 - smoothstep(0.46, 0.5, abs(fract(longitude * 4.0 / 3.14159265) - 0.5));
    let latitude_grid = 1.0 - smoothstep(0.45, 0.5, abs(fract(latitude * 6.0 / 3.14159265) - 0.5));
    let grid = max(longitude_grid, latitude_grid);
    let axes = vec3<f32>(
        select(0.15, 1.4, unit.x >= 0.0),
        select(0.15, 1.2, unit.y >= 0.0),
        select(0.15, 1.6, unit.z >= 0.0),
    );
    return axes * (0.35 + 0.65 * abs(unit)) + vec3<f32>(2.0 * grid);
}

fn store_trace_result(
    index: u32,
    pixel: vec2<u32>,
    termination: u32,
    flags: u32,
    steps: u32,
    event_residual: f32,
    direction: vec3<f32>,
    travel_time: f32,
    maximum_drift: vec4<f32>,
) {
    trace_direction_time[index] = vec4<f32>(direction, travel_time);
    trace_invariant_drift[index] = maximum_drift;
    trace_metadata[index] = vec4<u32>(termination, flags, steps, bitcast<u32>(event_residual));
    var scene_linear = visible_failure_color(termination);
    if termination == TERMINATION_HORIZON {
        scene_linear = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    } else if termination == TERMINATION_ESCAPE {
        scene_linear = vec4<f32>(analytic_sky(direction), 1.0);
    }
    textureStore(scene_hdr, vec2<i32>(pixel), scene_linear);
}

fn store_failure(index: u32, pixel: vec2<u32>, flags: u32) {
    trace_direction_time[index] = vec4<f32>(0.0);
    trace_invariant_drift[index] = vec4<f32>(0.0);
    trace_metadata[index] = vec4<u32>(TERMINATION_NUMERICAL_FAILURE, flags, 0u, 0u);
    textureStore(scene_hdr, vec2<i32>(pixel), vec4<f32>(1.0, 0.0, 1.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn trace_scene(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extent = textureDimensions(scene_hdr);
    let local_index = global_id.y * 8u + global_id.x;
    let index = trace_dispatch.pixels.x + local_index;
    if index >= extent.x * extent.y {
        return;
    }
    let pixel = vec2<u32>(index % extent.x, index / extent.x);

    let initial = initial_state_at(pixel, extent, trace_uniforms.projection.zw);
    if initial.flags != 0u {
        store_failure(index, pixel, initial.flags);
        return;
    }
    let initial_invariants = invariants_from_geometry_rhs(
        initial.state,
        initial.geometry,
        initial.rhs,
    );
    if initial_invariants.flags != 0u {
        store_failure(index, pixel, initial_invariants.flags);
        return;
    }

    var state = initial.state;
    var state_geometry = initial.geometry;
    var state_rhs = initial.rhs;
    var maximum_drift = vec4<f32>(initial_invariants.values.x, 0.0, 0.0, 0.0);
    var travel_time = 0.0;
    let escape_radius = trace_uniforms.event_surfaces.x;
    let singularity_guard = trace_uniforms.event_surfaces.y;
    let horizon_radius = trace_uniforms.event_surfaces.z;

    for (var step_index = 0u; step_index < MAXIMUM_STEPS; step_index += 1u) {
        let start_geometry = state_geometry;
        if start_geometry.flags != 0u {
            store_trace_result(
                index,
                pixel,
                TERMINATION_NUMERICAL_FAILURE,
                start_geometry.flags,
                step_index,
                0.0,
                vec3<f32>(0.0),
                travel_time,
                maximum_drift,
            );
            return;
        }
        if start_geometry.singularity_measure <= singularity_guard {
            let residual = (start_geometry.singularity_measure - singularity_guard)
                / max(1.0, singularity_guard);
            store_trace_result(
                index,
                pixel,
                TERMINATION_SINGULARITY,
                0u,
                step_index,
                residual,
                vec3<f32>(0.0),
                travel_time,
                maximum_drift,
            );
            return;
        }

        let strong_field_scale = select(0.1, 1.0, start_geometry.radius > 6.0);
        let step_magnitude = clamp(
            strong_field_scale * trace_uniforms.step_policy.x * start_geometry.radius,
            strong_field_scale * trace_uniforms.step_policy.y,
            trace_uniforms.step_policy.z,
        );
        let stepped = rk4_step(state, state_rhs, -step_magnitude);
        if stepped.flags != 0u {
            store_trace_result(
                index,
                pixel,
                TERMINATION_NUMERICAL_FAILURE,
                stepped.flags,
                step_index,
                0.0,
                vec3<f32>(0.0),
                travel_time,
                maximum_drift,
            );
            return;
        }
        let next_geometry = geometry_at(stepped.state.position.yzw);
        if next_geometry.flags != 0u {
            store_trace_result(
                index,
                pixel,
                TERMINATION_NUMERICAL_FAILURE,
                next_geometry.flags,
                step_index + 1u,
                0.0,
                vec3<f32>(0.0),
                travel_time,
                maximum_drift,
            );
            return;
        }
        let next_rhs = hamilton_rhs_from_geometry(stepped.state, next_geometry);
        if next_rhs.flags != 0u {
            store_trace_result(
                index,
                pixel,
                TERMINATION_NUMERICAL_FAILURE,
                next_rhs.flags,
                step_index + 1u,
                0.0,
                vec3<f32>(0.0),
                travel_time,
                maximum_drift,
            );
            return;
        }
        var termination = 0u;
        var event_surface = 0.0;
        var start_value = 0.0;
        var end_value = 0.0;
        if next_geometry.singularity_measure <= singularity_guard {
            termination = TERMINATION_SINGULARITY;
            event_surface = singularity_guard;
            start_value = start_geometry.singularity_measure;
            end_value = next_geometry.singularity_measure;
        } else if horizon_radius > 0.0
            && start_geometry.radius > horizon_radius
            && next_geometry.radius <= horizon_radius
        {
            termination = TERMINATION_HORIZON;
            event_surface = horizon_radius;
            start_value = start_geometry.radius;
            end_value = next_geometry.radius;
        } else if start_geometry.radius < escape_radius
            && next_geometry.radius >= escape_radius
        {
            termination = TERMINATION_ESCAPE;
            event_surface = escape_radius;
            start_value = start_geometry.radius;
            end_value = next_geometry.radius;
        }

        if termination != 0u {
            let denominator = end_value - start_value;
            if denominator == 0.0 || !finite_scalar(denominator) {
                store_trace_result(
                    index,
                    pixel,
                    TERMINATION_NUMERICAL_FAILURE,
                    FLAG_INVALID_DENOMINATOR,
                    step_index + 1u,
                    0.0,
                    vec3<f32>(0.0),
                    travel_time,
                    maximum_drift,
                );
                return;
            }
            let event_fraction = clamp((event_surface - start_value) / denominator, 0.0, 1.0);
            let localized = dense_state_at(
                state,
                stepped.state,
                state_rhs,
                next_rhs,
                -step_magnitude,
                event_fraction,
            );
            let localized_geometry = geometry_at(localized.position.yzw);
            let localized_rhs = hamilton_rhs_from_geometry(localized, localized_geometry);
            let localized_invariants = invariants_from_geometry_rhs(
                localized,
                localized_geometry,
                localized_rhs,
            );
            let localized_flags = localized_geometry.flags
                | localized_rhs.flags
                | localized_invariants.flags;
            if localized_flags != 0u {
                store_trace_result(
                    index,
                    pixel,
                    TERMINATION_NUMERICAL_FAILURE,
                    localized_flags,
                    step_index + 1u,
                    0.0,
                    vec3<f32>(0.0),
                    travel_time,
                    maximum_drift,
                );
                return;
            }
            let committed_travel_time = travel_time
                + abs(localized.position.x - state.position.x);
            let committed_maximum_drift = max(
                maximum_drift,
                invariant_drift(initial_invariants.values, localized_invariants.values),
            );
            var direction = vec3<f32>(0.0);
            if termination == TERMINATION_ESCAPE {
                direction = normalize(-localized_rhs.position.yzw);
            }
            let localized_value = select(
                localized_geometry.radius,
                localized_geometry.singularity_measure,
                termination == TERMINATION_SINGULARITY,
            );
            let event_residual = (localized_value - event_surface) / max(1.0, abs(event_surface));
            if invariant_budget_exceeded(committed_maximum_drift) {
                termination = TERMINATION_UNCERTAIN;
            }
            store_trace_result(
                index,
                pixel,
                termination,
                0u,
                step_index + 1u,
                event_residual,
                direction,
                committed_travel_time,
                committed_maximum_drift,
            );
            return;
        }
        let current_invariants = invariants_from_geometry_rhs(
            stepped.state,
            next_geometry,
            next_rhs,
        );
        if current_invariants.flags != 0u {
            store_trace_result(
                index,
                pixel,
                TERMINATION_NUMERICAL_FAILURE,
                current_invariants.flags,
                step_index + 1u,
                0.0,
                vec3<f32>(0.0),
                travel_time,
                maximum_drift,
            );
            return;
        }
        travel_time += abs(stepped.state.position.x - state.position.x);
        maximum_drift = max(
            maximum_drift,
            invariant_drift(initial_invariants.values, current_invariants.values),
        );
        state = stepped.state;
        state_geometry = next_geometry;
        state_rhs = next_rhs;
    }

    store_trace_result(
        index,
        pixel,
        TERMINATION_STEP_EXHAUSTION,
        0u,
        MAXIMUM_STEPS,
        0.0,
        vec3<f32>(0.0),
        travel_time,
        maximum_drift,
    );
}

// Exact Cartesian Kerr-Schild null-geodesic solver. Appearance and conservative accelerators are
// composed as separate WGSL modules so every rejected fast path returns here unchanged.

struct TraceUniforms {
    // (mass, spin, charge, Kerr-Schild branch sign)
    spacetime: vec4<f32>,
    observer_event: vec4<f32>,
    observer_velocity: vec4<f32>,
    image_right: vec4<f32>,
    image_up: vec4<f32>,
    arrival: vec4<f32>,
    // (tan(vertical FOV / 2), observer frequency, subpixel x, subpixel y)
    camera: vec4<f32>,
    // (escape radius, singularity guard, horizon radius, event tie tolerance in M)
    event_surfaces: vec4<f32>,
    // (surface inner radius, outer radius, intensity at 6 M, arming band in M)
    surface_emitter: vec4<f32>,
    // (radial step scale, minimum step, maximum step, invariant drift limit)
    step_policy: vec4<f32>,
}

struct TraceDispatch {
    tile_origin: vec2<u32>,
    tile_count: vec2<u32>,
}

struct Geometry {
    radius: f32,
    ks_profile: f32,
    ks_profile_gradient: vec3<f32>,
    principal_spatial: vec3<f32>,
    radius_gradient: vec3<f32>,
    inverse_scale: f32,
    inverse_radius: f32,
    inverse_oblate_factor: f32,
    singularity_measure: f32,
    flags: u32,
}

struct TraceState {
    position: vec3<f32>,
    momentum: vec3<f32>,
}

struct RhsResult {
    spacetime: vec4<f32>,
    momentum: vec3<f32>,
    flags: u32,
}

struct StepResult {
    state: TraceState,
    coordinate_time_increment: f32,
    flags: u32,
}

struct InitialState {
    state: TraceState,
    energy: f32,
    geometry: Geometry,
    rhs: RhsResult,
}

struct Invariants {
    values: vec4<f32>,
    flags: u32,
}

struct GeometricSample {
    termination: u32,
    flags: u32,
    event_candidates: u32,
    steps: u32,
    event_residual: f32,
    source_coordinates: vec3<f32>,
    travel_time: f32,
    maximum_drift: vec4<f32>,
}

const TERMINATION_HORIZON: u32 = 1u;
const TERMINATION_ESCAPE: u32 = 2u;
const TERMINATION_SINGULARITY: u32 = 3u;
const TERMINATION_STEP_EXHAUSTION: u32 = 4u;
const TERMINATION_NUMERICAL_FAILURE: u32 = 5u;
const TERMINATION_UNCERTAIN: u32 = 6u;
const TERMINATION_EQUATORIAL_SURFACE: u32 = 7u;
const FLAG_NON_FINITE: u32 = 1u;
const FLAG_INVALID_RADICAND: u32 = 2u;
const FLAG_INVALID_DENOMINATOR: u32 = 4u;
const EVENT_CANDIDATE_SINGULARITY: u32 = 1u;
const EVENT_CANDIDATE_HORIZON: u32 = 2u;
const EVENT_CANDIDATE_SURFACE: u32 = 4u;
const EVENT_CANDIDATE_ESCAPE: u32 = 8u;
const EVENT_INDEX_SINGULARITY: u32 = 0u;
const EVENT_INDEX_HORIZON: u32 = 1u;
const EVENT_INDEX_SURFACE: u32 = 2u;
const EVENT_INDEX_ESCAPE: u32 = 3u;
const TRACE_WORKGROUP_AXIS: u32 = 8u;
const MAXIMUM_STEPS: u32 = 2048u;
const MAXIMUM_FINITE_F32: f32 = 0x1.fffffep+127f;
const EVENT_REFINEMENT_ITERATIONS: u32 = 6u;
const EVENT_DERIVATIVE_RELATIVE_FLOOR: f32 = 0x1p-11f;
override SURFACE_EVENTS_ENABLED: u32 = 0u;

alias EventCandidates = vec4<f32>;

struct EventSelection {
    termination: u32,
    fraction: f32,
    candidates: u32,
    ambiguous: u32,
}

struct SurfaceSource {
    coordinates: vec3<f32>,
    flags: u32,
}

@group(0) @binding(0)
var<uniform> trace_uniforms: TraceUniforms;

@group(0) @binding(2)
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
        0.0,
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        0.0,
        0.0,
        0.0,
        0.0,
        flags,
    );
}

fn metric_geometry_at(position: vec3<f32>) -> Geometry {
    let spin_physical = trace_uniforms.spacetime.y;
    let charge_physical = trace_uniforms.spacetime.z;
    let chart_sign = trace_uniforms.spacetime.w;
    let scale = max(
        max(max(abs(position.x), abs(position.y)), abs(position.z)),
        abs(spin_physical),
    );
    if !finite_scalar(scale) || scale == 0.0 {
        return invalid_geometry(FLAG_INVALID_DENOMINATOR);
    }

    let inverse_scale = 1.0 / scale;
    let coordinates = position * inverse_scale;
    let x = coordinates.x;
    let y = coordinates.y;
    let z = coordinates.z;
    let spin = spin_physical * inverse_scale;
    let spin_squared = spin * spin;

    // The analytic axis form avoids a rounded oblate root while retaining the transverse
    // principal-vector derivative used by contract_principal_gradient.
    if x == 0.0 && y == 0.0 {
        let radius = abs(z);
        let radius_squared = radius * radius;
        let sigma = radius_squared + spin_squared;
        if radius <= 0.0 || sigma <= 0.0 || !finite_scalar(sigma) {
            return invalid_geometry(FLAG_INVALID_DENOMINATOR);
        }
        let axis_sign = sign(z);
        let radius_gradient = vec3<f32>(0.0, 0.0, axis_sign);
        let mass = trace_uniforms.spacetime.x * inverse_scale;
        let charge = charge_physical * inverse_scale;
        let numerator = 2.0 * mass * radius - charge * charge;
        let inverse_sigma = 1.0 / sigma;
        let ks_profile = numerator * inverse_sigma;
        let sigma_gradient = vec3<f32>(0.0, 0.0, 2.0 * radius * axis_sign);
        let numerator_gradient = 2.0 * mass * radius_gradient;
        let ks_profile_gradient = (numerator_gradient - ks_profile * sigma_gradient)
            * (inverse_sigma * inverse_scale);
        let principal_spatial = vec3<f32>(0.0, 0.0, chart_sign * axis_sign);
        let inverse_radius = 1.0 / radius;
        let inverse_oblate_factor = inverse_sigma;
        let physical_radius = scale * radius;
        if !finite_scalar(physical_radius)
            || !finite_scalar(ks_profile)
            || !finite_vec3(ks_profile_gradient)
            || !finite_scalar(inverse_radius)
            || !finite_scalar(inverse_oblate_factor)
        {
            return invalid_geometry(FLAG_NON_FINITE);
        }
        return Geometry(
            physical_radius,
            ks_profile,
            ks_profile_gradient,
            principal_spatial,
            radius_gradient,
            inverse_scale,
            inverse_radius,
            inverse_oblate_factor,
            0.0,
            0u,
        );
    }

    let radius_squared_3d = dot(coordinates, coordinates);
    let oblate_offset = radius_squared_3d - spin_squared;
    let radicand = oblate_offset * oblate_offset + 4.0 * spin_squared * z * z;
    if radicand < 0.0 || !finite_scalar(radicand) {
        return invalid_geometry(FLAG_INVALID_RADICAND);
    }
    let sigma = sqrt(radicand);
    var radius_squared = 0.0;
    if oblate_offset >= 0.0 {
        radius_squared = 0.5 * (oblate_offset + sigma);
    } else {
        // Use the conjugate root when direct addition would cancel.
        let denominator = sigma - oblate_offset;
        if denominator <= 0.0 || !finite_scalar(denominator) {
            return invalid_geometry(FLAG_INVALID_DENOMINATOR);
        }
        radius_squared = 2.0 * spin_squared * z * z / denominator;
    }
    if radius_squared <= 0.0 || !finite_scalar(radius_squared) {
        return invalid_geometry(FLAG_INVALID_DENOMINATOR);
    }
    let radius = sqrt(radius_squared);
    if radius <= 0.0 || sigma <= 0.0 || !finite_scalar(sigma) {
        return invalid_geometry(FLAG_INVALID_DENOMINATOR);
    }

    let inverse_radius = 1.0 / radius;
    let inverse_sigma = 1.0 / sigma;
    let radius_gradient = vec3<f32>(
        coordinates.xy * (radius * inverse_sigma),
        z * (radius_squared + spin_squared) * inverse_radius * inverse_sigma,
    );
    let mass = trace_uniforms.spacetime.x * inverse_scale;
    let charge = charge_physical * inverse_scale;
    let numerator = 2.0 * mass * radius - charge * charge;
    let ks_profile = numerator * inverse_sigma;
    let sigma_gradient = 2.0 * inverse_sigma * vec3<f32>(
        oblate_offset * x,
        oblate_offset * y,
        z * (radius_squared_3d + spin_squared),
    );
    let numerator_gradient = 2.0 * mass * radius_gradient;
    let ks_profile_gradient = (numerator_gradient - ks_profile * sigma_gradient)
        * (inverse_sigma * inverse_scale);
    let oblate_factor = radius_squared + spin_squared;
    if oblate_factor <= 0.0 || !finite_scalar(oblate_factor) {
        return invalid_geometry(FLAG_INVALID_DENOMINATOR);
    }
    let inverse_oblate_factor = 1.0 / oblate_factor;
    let chart_spin = chart_sign * spin;
    let principal_spatial = chart_sign * vec3<f32>(
        (radius * coordinates.xy + chart_spin * vec2<f32>(y, -x))
            * inverse_oblate_factor,
        z * inverse_radius,
    );
    let physical_radius = scale * radius;
    if !finite_scalar(physical_radius)
        || !finite_scalar(ks_profile)
        || !finite_vec3(ks_profile_gradient)
        || !finite_vec3(principal_spatial)
        || !finite_vec3(radius_gradient)
        || !finite_scalar(inverse_scale)
        || !finite_scalar(inverse_radius)
        || !finite_scalar(inverse_oblate_factor)
    {
        return invalid_geometry(FLAG_NON_FINITE);
    }
    return Geometry(
        physical_radius,
        ks_profile,
        ks_profile_gradient,
        principal_spatial,
        radius_gradient,
        inverse_scale,
        inverse_radius,
        inverse_oblate_factor,
        0.0,
        0u,
    );
}

fn event_geometry_at(position: vec3<f32>) -> Geometry {
    var geometry = metric_geometry_at(position);
    if geometry.flags == 0u {
        geometry.singularity_measure = singularity_measure(
            geometry.radius,
            trace_uniforms.spacetime.y,
            position.z,
        );
    }
    return geometry;
}

fn sight_direction(pixel: vec2<u32>, extent: vec2<u32>, subpixel: vec2<f32>) -> vec4<f32> {
    let dimensions = vec2<f32>(extent);
    let normalized = vec2<f32>(
        2.0 * (f32(pixel.x) + subpixel.x) / dimensions.x - 1.0,
        1.0 - 2.0 * (f32(pixel.y) + subpixel.y) / dimensions.y,
    );
    let tangent_half_fov = trace_uniforms.camera.x;
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
    let principal_covector = vec4<f32>(1.0, geometry.principal_spatial);
    let contraction = dot(principal_covector, momentum_contravariant);
    return minkowski_lowered
        + geometry.ks_profile * contraction * principal_covector;
}

fn invalid_rhs(flags: u32) -> RhsResult {
    return RhsResult(vec4<f32>(0.0), vec3<f32>(0.0), flags);
}

fn contract_principal_gradient(state: TraceState, geometry: Geometry) -> vec3<f32> {
    let inverse_scale = geometry.inverse_scale;
    let coordinates = state.position * inverse_scale;
    let radius = geometry.radius * inverse_scale;
    let spin = trace_uniforms.spacetime.y * inverse_scale;
    let chart_sign = trace_uniforms.spacetime.w;
    let chart_spin = chart_sign * spin;
    let inverse_radius = geometry.inverse_radius;
    let inverse_radius_squared = inverse_radius * inverse_radius;
    let inverse_oblate_factor = geometry.inverse_oblate_factor;
    let momentum = state.momentum;
    let unoriented_principal_spatial = chart_sign * geometry.principal_spatial;
    let radial_coefficient = (
        dot(coordinates.xy, momentum.xy)
        - 2.0 * radius * dot(unoriented_principal_spatial.xy, momentum.xy)
    ) * inverse_oblate_factor
        - coordinates.z * momentum.z * inverse_radius_squared;
    let direct = vec3<f32>(
        (radius * momentum.xy + chart_spin * vec2<f32>(-momentum.y, momentum.x))
            * inverse_oblate_factor,
        momentum.z * inverse_radius,
    );
    return chart_sign * inverse_scale
        * (direct + radial_coefficient * geometry.radius_gradient);
}

fn hamilton_rhs_from_geometry(state: TraceState, energy: f32, geometry: Geometry) -> RhsResult {
    if geometry.flags != 0u {
        return invalid_rhs(geometry.flags);
    }
    let contraction = energy + dot(state.momentum, geometry.principal_spatial);
    let profile_contraction = geometry.ks_profile * contraction;
    let position_derivative = state.momentum
        - profile_contraction * geometry.principal_spatial;
    let coordinate_time_derivative = energy + profile_contraction;
    let principal_gradient_contraction = contract_principal_gradient(state, geometry);
    let momentum_derivative = profile_contraction * principal_gradient_contraction
        + 0.5 * contraction * contraction * geometry.ks_profile_gradient;
    if !finite_vec3(position_derivative)
        || !finite_vec3(momentum_derivative)
        || !finite_scalar(coordinate_time_derivative)
    {
        return invalid_rhs(FLAG_NON_FINITE);
    }
    return RhsResult(
        vec4<f32>(coordinate_time_derivative, position_derivative),
        momentum_derivative,
        0u,
    );
}

fn hamilton_rhs(state: TraceState, energy: f32) -> RhsResult {
    return hamilton_rhs_from_geometry(
        state,
        energy,
        // Intermediate RK stages do not evaluate event-only singularity guards.
        metric_geometry_at(state.position),
    );
}

fn state_add(state: TraceState, derivative: RhsResult, factor: f32) -> TraceState {
    return TraceState(
        state.position + factor * derivative.spacetime.yzw,
        state.momentum + factor * derivative.momentum,
    );
}

fn invalid_step(state: TraceState, flags: u32) -> StepResult {
    return StepResult(state, 0.0, flags);
}

fn rk4_step(state: TraceState, energy: f32, k1: RhsResult, step: f32) -> StepResult {
    if k1.flags != 0u {
        return invalid_step(state, k1.flags);
    }
    var stage = hamilton_rhs(state_add(state, k1, 0.5 * step), energy);
    if stage.flags != 0u {
        return invalid_step(state, stage.flags);
    }
    var weighted_spacetime = k1.spacetime + 2.0 * stage.spacetime;
    var weighted_momentum = k1.momentum + 2.0 * stage.momentum;
    stage = hamilton_rhs(state_add(state, stage, 0.5 * step), energy);
    if stage.flags != 0u {
        return invalid_step(state, stage.flags);
    }
    weighted_spacetime += 2.0 * stage.spacetime;
    weighted_momentum += 2.0 * stage.momentum;
    stage = hamilton_rhs(state_add(state, stage, step), energy);
    if stage.flags != 0u {
        return invalid_step(state, stage.flags);
    }
    weighted_spacetime += stage.spacetime;
    weighted_momentum += stage.momentum;
    let step_weight = step / 6.0;
    let next = TraceState(
        state.position + step_weight * weighted_spacetime.yzw,
        state.momentum + step_weight * weighted_momentum,
    );
    let coordinate_time_increment = step_weight * weighted_spacetime.x;
    if !finite_vec3(next.position)
        || !finite_vec3(next.momentum)
        || !finite_scalar(coordinate_time_increment)
    {
        return invalid_step(state, FLAG_NON_FINITE);
    }
    return StepResult(next, coordinate_time_increment, 0u);
}

fn cubic_dense_weights(
    step: f32,
    step_fraction: f32,
) -> vec4<f32> {
    let fraction_squared = step_fraction * step_fraction;
    let fraction_cubed = fraction_squared * step_fraction;
    return vec4<f32>(
        2.0 * fraction_cubed - 3.0 * fraction_squared + 1.0,
        step * (fraction_cubed - 2.0 * fraction_squared + step_fraction),
        -2.0 * fraction_cubed + 3.0 * fraction_squared,
        step * (fraction_cubed - fraction_squared),
    );
}

fn dense_state_at(
    start: TraceState,
    end: TraceState,
    start_derivative: RhsResult,
    end_derivative: RhsResult,
    weights: vec4<f32>,
) -> TraceState {
    return TraceState(
        weights.x * start.position
            + weights.y * start_derivative.spacetime.yzw
            + weights.z * end.position
            + weights.w * end_derivative.spacetime.yzw,
        weights.x * start.momentum
            + weights.y * start_derivative.momentum
            + weights.z * end.momentum
            + weights.w * end_derivative.momentum,
    );
}

fn dense_coordinate_time_increment(
    step_increment: f32,
    start_derivative: RhsResult,
    end_derivative: RhsResult,
    weights: vec4<f32>,
) -> f32 {
    return weights.y * start_derivative.spacetime.x
        + weights.z * step_increment
        + weights.w * end_derivative.spacetime.x;
}

fn event_guard_derivative(
    termination: u32,
    state: TraceState,
    geometry: Geometry,
    rhs: RhsResult,
) -> f32 {
    if termination == TERMINATION_EQUATORIAL_SURFACE {
        return rhs.spacetime.w;
    }
    let radial_velocity = dot(geometry.radius_gradient, rhs.spacetime.yzw);
    if termination != TERMINATION_SINGULARITY {
        return radial_velocity;
    }
    let radius = geometry.radius;
    let spin = trace_uniforms.spacetime.y;
    return 4.0 * radius * radius * radius * radial_velocity
        + 2.0 * spin * spin * state.position.z * rhs.spacetime.w;
}

fn cubic_hermite_coefficients(
    start_value: f32,
    end_value: f32,
    start_slope: f32,
    end_slope: f32,
) -> vec4<f32> {
    return vec4<f32>(
        2.0 * (start_value - end_value) + start_slope + end_slope,
        3.0 * (end_value - start_value) - 2.0 * start_slope - end_slope,
        start_slope,
        start_value,
    );
}

fn cubic_value_and_derivative(
    fraction: f32,
    coefficients: vec4<f32>,
) -> vec2<f32> {
    return vec2<f32>(
        ((coefficients.x * fraction + coefficients.y) * fraction + coefficients.z)
            * fraction
            + coefficients.w,
        (3.0 * coefficients.x * fraction + 2.0 * coefficients.y) * fraction
            + coefficients.z,
    );
}

fn localized_event_fraction(
    start_residual: f32,
    end_residual: f32,
    start_slope: f32,
    end_slope: f32,
) -> f32 {
    let chord_denominator = end_residual - start_residual;
    let chord = clamp(-start_residual / chord_denominator, 0.0, 1.0);
    if !finite_scalar(start_slope) || !finite_scalar(end_slope) {
        return chord;
    }
    let middle_slope = 3.0 * chord_denominator - start_slope - end_slope;
    if !finite_scalar(middle_slope) {
        return chord;
    }
    let slope_controls = sign(chord_denominator) * vec3<f32>(
        start_slope,
        middle_slope,
        end_slope,
    );
    if any(slope_controls < vec3<f32>(0.0)) {
        return chord;
    }
    let coefficients = cubic_hermite_coefficients(
        start_residual,
        end_residual,
        start_slope,
        end_slope,
    );
    let derivative_scale = max(
        abs(chord_denominator),
        max(
            max(slope_controls.x, slope_controls.y),
            slope_controls.z,
        ),
    );
    // A smaller derivative makes the Hermite root ill-conditioned in binary32. Falling back to
    // the original chord preserves the established near-tangent event semantics.
    let derivative_floor = EVENT_DERIVATIVE_RELATIVE_FLOOR * derivative_scale;

    var lower = 0.0;
    var upper = 1.0;
    var fraction = chord;
    let start_is_positive = start_residual > 0.0;
    for (var iteration = 0u; iteration < EVENT_REFINEMENT_ITERATIONS; iteration += 1u) {
        let current = fraction;
        let evaluation = cubic_value_and_derivative(current, coefficients);
        if !finite_scalar(evaluation.x) {
            return chord;
        }
        if evaluation.x == 0.0 {
            return current;
        }
        let on_start_side = (evaluation.x > 0.0) == start_is_positive;
        lower = select(lower, current, on_start_side);
        upper = select(current, upper, on_start_side);
        let derivative = evaluation.y;
        if !finite_scalar(derivative) || abs(derivative) <= derivative_floor {
            return chord;
        }
        let midpoint = 0.5 * (lower + upper);
        fraction = midpoint;
        let newton = current - evaluation.x / derivative;
        if finite_scalar(newton) && newton > lower && newton < upper {
            fraction = newton;
        }
    }
    return fraction;
}

fn no_event_candidates() -> EventCandidates {
    return EventCandidates(2.0);
}

fn event_candidate_mask(termination: u32) -> u32 {
    switch termination {
        case TERMINATION_SINGULARITY: { return EVENT_CANDIDATE_SINGULARITY; }
        case TERMINATION_HORIZON: { return EVENT_CANDIDATE_HORIZON; }
        case TERMINATION_EQUATORIAL_SURFACE: { return EVENT_CANDIDATE_SURFACE; }
        case TERMINATION_ESCAPE: { return EVENT_CANDIDATE_ESCAPE; }
        default: { return 0u; }
    }
}

fn event_fraction(
    start_residual: f32,
    end_residual: f32,
    start_slope: f32,
    end_slope: f32,
) -> f32 {
    return localized_event_fraction(
        start_residual,
        end_residual,
        start_slope,
        end_slope,
    );
}

fn event_termination(index: u32) -> u32 {
    switch index {
        case EVENT_INDEX_SINGULARITY: { return TERMINATION_SINGULARITY; }
        case EVENT_INDEX_HORIZON: { return TERMINATION_HORIZON; }
        case EVENT_INDEX_SURFACE: { return TERMINATION_EQUATORIAL_SURFACE; }
        default: { return TERMINATION_ESCAPE; }
    }
}

fn select_earliest_event(
    events: EventCandidates,
    step_magnitude: f32,
) -> EventSelection {
    // Strict replacement preserves protocol order for exactly equal fractions.
    var earliest_index = EVENT_INDEX_SINGULARITY;
    var earliest_fraction = events.x;
    if events.y < earliest_fraction {
        earliest_index = EVENT_INDEX_HORIZON;
        earliest_fraction = events.y;
    }
    if events.z < earliest_fraction {
        earliest_index = EVENT_INDEX_SURFACE;
        earliest_fraction = events.z;
    }
    if events.w < earliest_fraction {
        earliest_index = EVENT_INDEX_ESCAPE;
        earliest_fraction = events.w;
    }
    if earliest_fraction > 1.0 {
        return EventSelection(0u, earliest_fraction, 0u, 0u);
    }

    // Tie proximity is non-transitive, so classify every slot against the global earliest.
    let affine_separation = abs(events - EventCandidates(earliest_fraction)) * step_magnitude;
    let tied_bits = select(
        vec4<u32>(0u),
        vec4<u32>(
            EVENT_CANDIDATE_SINGULARITY,
            EVENT_CANDIDATE_HORIZON,
            EVENT_CANDIDATE_SURFACE,
            EVENT_CANDIDATE_ESCAPE,
        ),
        affine_separation <= EventCandidates(trace_uniforms.event_surfaces.w),
    );
    let candidate_bits = select(
        vec4<u32>(0u),
        tied_bits,
        events <= EventCandidates(1.0),
    );
    let candidates = candidate_bits.x | candidate_bits.y | candidate_bits.z | candidate_bits.w;
    let ambiguous = select(0u, 1u, countOneBits(candidates) > 1u);
    return EventSelection(
        event_termination(earliest_index),
        earliest_fraction,
        candidates,
        ambiguous,
    );
}

fn update_surface_event_arming(armed: bool, surface_value: f32) -> bool {
    return armed || abs(surface_value) > trace_uniforms.surface_emitter.w;
}

fn surface_source_at(
    state: TraceState,
    geometry: Geometry,
    invariants: Invariants,
) -> SurfaceSource {
    let radius = geometry.radius;
    let spin = trace_uniforms.spacetime.y;
    let charge = trace_uniforms.spacetime.z;
    let charge_squared = charge * charge;
    let circular_root_squared = radius - charge_squared;
    if circular_root_squared < 0.0 || !finite_scalar(circular_root_squared) {
        return SurfaceSource(vec3<f32>(0.0), FLAG_INVALID_RADICAND);
    }
    let circular_root = sqrt(circular_root_squared);
    let branch_sign = select(-1.0, 1.0, spin >= 0.0);
    let radius_squared = radius * radius;
    let angular_velocity_denominator = radius_squared + branch_sign * spin * circular_root;
    if angular_velocity_denominator <= 0.0 || !finite_scalar(angular_velocity_denominator) {
        return SurfaceSource(vec3<f32>(0.0), FLAG_INVALID_DENOMINATOR);
    }
    let angular_velocity = branch_sign * circular_root / angular_velocity_denominator;
    let radial_numerator = 2.0 * radius - charge_squared;
    let spin_squared = spin * spin;
    let timelike_discriminant = radius_squared - 3.0 * radius
        + 2.0 * charge_squared
        + 2.0 * branch_sign * spin * circular_root;
    if timelike_discriminant <= 0.0 || !finite_scalar(timelike_discriminant) {
        return SurfaceSource(vec3<f32>(0.0), FLAG_INVALID_DENOMINATOR);
    }
    let delta = radius_squared - 2.0 * radius + spin_squared + charge_squared;
    let g_tt = -1.0 + radial_numerator / radius_squared;
    let g_t_phi = -radial_numerator * spin / radius_squared;
    let radial_factor = radius_squared + spin_squared;
    let g_phi_phi = (radial_factor * radial_factor - spin_squared * delta) / radius_squared;
    let circular_norm_squared = g_tt
        + 2.0 * g_t_phi * angular_velocity
        + g_phi_phi * angular_velocity * angular_velocity;
    if circular_norm_squared >= 0.0 || !finite_scalar(circular_norm_squared) {
        return SurfaceSource(vec3<f32>(0.0), FLAG_INVALID_DENOMINATOR);
    }
    let time_component = inverseSqrt(-circular_norm_squared);
    let emitter_frequency = time_component
        * (invariants.values.y - angular_velocity * invariants.values.z);
    if emitter_frequency <= 0.0 || !finite_scalar(emitter_frequency) {
        return SurfaceSource(vec3<f32>(0.0), FLAG_INVALID_DENOMINATOR);
    }
    let frequency_ratio = trace_uniforms.camera.y / emitter_frequency;
    let chart_spin = trace_uniforms.spacetime.w * spin;
    let raw_azimuth = atan2(state.position.y, state.position.x) - atan2(chart_spin, radius);
    let pi = 3.141592653589793;
    let tau = 6.283185307179586;
    let source_azimuth = raw_azimuth - tau * floor((raw_azimuth + pi) / tau);
    let coordinates = vec3<f32>(radius, source_azimuth, frequency_ratio);
    if frequency_ratio <= 0.0 || !finite_vec3(coordinates) {
        return SurfaceSource(vec3<f32>(0.0), FLAG_NON_FINITE);
    }
    return SurfaceSource(coordinates, 0u);
}

fn initial_state_at(pixel: vec2<u32>, extent: vec2<u32>, subpixel: vec2<f32>) -> InitialState {
    let sight = sight_direction(pixel, extent, subpixel);
    let arrival = -sight;
    let momentum_contravariant = trace_uniforms.camera.y
        * (trace_uniforms.observer_velocity + arrival);
    let geometry = event_geometry_at(trace_uniforms.observer_event.yzw);
    var initial_flags = geometry.flags;
    if !finite_vec4(momentum_contravariant) {
        initial_flags |= FLAG_NON_FINITE;
    }
    if initial_flags != 0u {
        return InitialState(
            TraceState(trace_uniforms.observer_event.yzw, vec3<f32>(0.0)),
            0.0,
            geometry,
            invalid_rhs(initial_flags),
        );
    }
    let momentum_covariant = lower_momentum(momentum_contravariant, geometry);
    if !finite_vec4(momentum_covariant) {
        return InitialState(
            TraceState(trace_uniforms.observer_event.yzw, vec3<f32>(0.0)),
            0.0,
            geometry,
            invalid_rhs(FLAG_NON_FINITE),
        );
    }
    let energy = -momentum_covariant.x;
    let state = TraceState(trace_uniforms.observer_event.yzw, momentum_covariant.yzw);
    let rhs = hamilton_rhs_from_geometry(state, energy, geometry);
    return InitialState(state, energy, geometry, rhs);
}

fn invariants_from_geometry_rhs(
    state: TraceState,
    energy: f32,
    geometry: Geometry,
    rhs: RhsResult,
) -> Invariants {
    if geometry.flags != 0u || rhs.flags != 0u {
        return Invariants(vec4<f32>(0.0), geometry.flags | rhs.flags);
    }
    let x = state.position.x;
    let y = state.position.y;
    let z = state.position.z;
    let angular_momentum_z = x * state.momentum.y - y * state.momentum.x;
    let spin = trace_uniforms.spacetime.y;
    let spin_squared = spin * spin;
    let radius = geometry.radius;
    let radius_squared = radius * radius;
    let oblate_factor = radius_squared + spin_squared;
    if oblate_factor <= 0.0 || !finite_scalar(oblate_factor) {
        return Invariants(vec4<f32>(0.0), FLAG_INVALID_DENOMINATOR);
    }
    let transverse_position = state.position.xy;
    let transverse_momentum = state.momentum.xy;
    let projected_momentum = dot(transverse_position, transverse_momentum);
    let transverse_momentum_squared = dot(transverse_momentum, transverse_momentum);
    let scaled_position = state.position * geometry.inverse_scale;
    let scaled_cosine = scaled_position.z * geometry.inverse_radius;
    let polar_weights = vec2<f32>(
        scaled_cosine * scaled_cosine,
        dot(scaled_position.xy, scaled_position.xy) * geometry.inverse_oblate_factor,
    );
    let carter = polar_weights.x
        * (oblate_factor * transverse_momentum_squared - spin_squared * energy * energy)
        - 2.0 * z * projected_momentum * state.momentum.z
        + radius_squared * polar_weights.y * state.momentum.z * state.momentum.z;
    let null_contraction = -energy * rhs.spacetime.x
        + dot(state.momentum, rhs.spacetime.yzw);
    let null_term_norm = abs(energy * rhs.spacetime.x)
        + dot(abs(state.momentum), abs(rhs.spacetime.yzw));
    let null_residual = abs(null_contraction) / max(1.0, null_term_norm);
    let values = vec4<f32>(null_residual, energy, angular_momentum_z, carter);
    if !finite_vec4(values) {
        return Invariants(vec4<f32>(0.0), FLAG_NON_FINITE);
    }
    return Invariants(values, 0u);
}

fn invariant_drift(initial: vec4<f32>, current: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        current.x,
        0.0,
        abs(current.z - initial.z) / max(1.0, abs(initial.z)),
        abs(current.w - initial.w) / max(1.0, abs(initial.w)),
    );
}

fn invariant_budget_exceeded(maximum_drift: vec4<f32>) -> bool {
    return any(maximum_drift > vec4<f32>(trace_uniforms.step_policy.w));
}

fn failure_result(flags: u32, steps: u32, travel_time: f32, maximum_drift: vec4<f32>) -> GeometricSample {
    return GeometricSample(
        TERMINATION_NUMERICAL_FAILURE,
        flags,
        0u,
        steps,
        0.0,
        vec3<f32>(0.0),
        travel_time,
        maximum_drift,
    );
}

fn trace_initialized(initial: InitialState, initial_invariants: Invariants) -> GeometricSample {
    var state = initial.state;
    let energy = initial.energy;
    var state_geometry = initial.geometry;
    var state_rhs = initial.rhs;
    var maximum_drift = vec4<f32>(initial_invariants.values.x, 0.0, 0.0, 0.0);
    var travel_time = 0.0;
    let escape_radius = trace_uniforms.event_surfaces.x;
    let singularity_guard = trace_uniforms.event_surfaces.y;
    let horizon_radius = trace_uniforms.event_surfaces.z;
    var surface_armed = SURFACE_EVENTS_ENABLED != 0u
        && update_surface_event_arming(false, state.position.z);

    for (var step_index = 0u; step_index < MAXIMUM_STEPS; step_index += 1u) {
        let start_geometry = state_geometry;
        if start_geometry.flags != 0u {
            return failure_result(
                start_geometry.flags,
                step_index,
                travel_time,
                maximum_drift,
            );
        }
        if start_geometry.singularity_measure <= singularity_guard {
            let residual = (start_geometry.singularity_measure - singularity_guard)
                / max(1.0, singularity_guard);
            return GeometricSample(
                TERMINATION_SINGULARITY,
                0u,
                EVENT_CANDIDATE_SINGULARITY,
                step_index,
                residual,
                vec3<f32>(0.0),
                travel_time,
                maximum_drift,
            );
        }

        let step_magnitude = clamp(
            trace_uniforms.step_policy.x * start_geometry.radius,
            trace_uniforms.step_policy.y,
            trace_uniforms.step_policy.z,
        );
        let stepped = rk4_step(state, energy, state_rhs, -step_magnitude);
        if stepped.flags != 0u {
            return failure_result(stepped.flags, step_index, travel_time, maximum_drift);
        }
        let next_geometry = event_geometry_at(stepped.state.position);
        if next_geometry.flags != 0u {
            return failure_result(
                next_geometry.flags,
                step_index + 1u,
                travel_time,
                maximum_drift,
            );
        }
        let next_rhs = hamilton_rhs_from_geometry(stepped.state, energy, next_geometry);
        if next_rhs.flags != 0u {
            return failure_result(
                next_rhs.flags,
                step_index + 1u,
                travel_time,
                maximum_drift,
            );
        }
        let signed_step = -step_magnitude;
        var event_candidates = no_event_candidates();
        if next_geometry.singularity_measure <= singularity_guard {
            event_candidates[EVENT_INDEX_SINGULARITY] = event_fraction(
                start_geometry.singularity_measure - singularity_guard,
                next_geometry.singularity_measure - singularity_guard,
                signed_step * event_guard_derivative(
                    TERMINATION_SINGULARITY,
                    state,
                    start_geometry,
                    state_rhs,
                ),
                signed_step * event_guard_derivative(
                    TERMINATION_SINGULARITY,
                    stepped.state,
                    next_geometry,
                    next_rhs,
                ),
            );
        }
        if horizon_radius > 0.0
            && start_geometry.radius > horizon_radius
            && next_geometry.radius <= horizon_radius
        {
            event_candidates[EVENT_INDEX_HORIZON] = event_fraction(
                start_geometry.radius - horizon_radius,
                next_geometry.radius - horizon_radius,
                signed_step * event_guard_derivative(
                    TERMINATION_HORIZON,
                    state,
                    start_geometry,
                    state_rhs,
                ),
                signed_step * event_guard_derivative(
                    TERMINATION_HORIZON,
                    stepped.state,
                    next_geometry,
                    next_rhs,
                ),
            );
        }
        if start_geometry.radius < escape_radius
            && next_geometry.radius >= escape_radius
        {
            event_candidates[EVENT_INDEX_ESCAPE] = event_fraction(
                start_geometry.radius - escape_radius,
                next_geometry.radius - escape_radius,
                signed_step * event_guard_derivative(
                    TERMINATION_ESCAPE,
                    state,
                    start_geometry,
                    state_rhs,
                ),
                signed_step * event_guard_derivative(
                    TERMINATION_ESCAPE,
                    stepped.state,
                    next_geometry,
                    next_rhs,
                ),
            );
        }
        let crosses_equatorial_plane = SURFACE_EVENTS_ENABLED != 0u
            && surface_armed
            && ((state.position.z > 0.0 && stepped.state.position.z <= 0.0)
                || (state.position.z < 0.0 && stepped.state.position.z >= 0.0));
        if crosses_equatorial_plane {
            let surface_fraction = event_fraction(
                state.position.z,
                stepped.state.position.z,
                signed_step * state_rhs.spacetime.w,
                signed_step * next_rhs.spacetime.w,
            );
            let surface_weights = cubic_dense_weights(signed_step, surface_fraction);
            let surface_state = dense_state_at(
                state,
                stepped.state,
                state_rhs,
                next_rhs,
                surface_weights,
            );
            let surface_geometry = event_geometry_at(surface_state.position);
            if surface_geometry.flags != 0u {
                return failure_result(
                    surface_geometry.flags,
                    step_index + 1u,
                    travel_time,
                    maximum_drift,
                );
            }
            if surface_geometry.radius >= trace_uniforms.surface_emitter.x
                && surface_geometry.radius <= trace_uniforms.surface_emitter.y
            {
                event_candidates[EVENT_INDEX_SURFACE] = surface_fraction;
            }
        }

        let selection = select_earliest_event(event_candidates, step_magnitude);
        if selection.termination != 0u {
            let dense_weights = cubic_dense_weights(signed_step, selection.fraction);
            let localized = dense_state_at(
                state,
                stepped.state,
                state_rhs,
                next_rhs,
                dense_weights,
            );
            let localized_time_increment = dense_coordinate_time_increment(
                stepped.coordinate_time_increment,
                state_rhs,
                next_rhs,
                dense_weights,
            );
            if !finite_scalar(localized_time_increment) {
                return failure_result(
                    FLAG_NON_FINITE,
                    step_index + 1u,
                    travel_time,
                    maximum_drift,
                );
            }
            let localized_geometry = event_geometry_at(localized.position);
            let localized_rhs = hamilton_rhs_from_geometry(localized, energy, localized_geometry);
            let localized_invariants = invariants_from_geometry_rhs(
                localized,
                energy,
                localized_geometry,
                localized_rhs,
            );
            let localized_flags = localized_geometry.flags
                | localized_rhs.flags
                | localized_invariants.flags;
            if localized_flags != 0u {
                return failure_result(
                    localized_flags,
                    step_index + 1u,
                    travel_time,
                    maximum_drift,
                );
            }
            let committed_travel_time = travel_time + abs(localized_time_increment);
            let committed_maximum_drift = max(
                maximum_drift,
                invariant_drift(initial_invariants.values, localized_invariants.values),
            );
            var source_coordinates = vec3<f32>(0.0);
            if selection.ambiguous == 0u && selection.termination == TERMINATION_ESCAPE {
                source_coordinates = normalize(-localized_rhs.spacetime.yzw);
            } else if selection.ambiguous == 0u
                && selection.termination == TERMINATION_EQUATORIAL_SURFACE
            {
                let source = surface_source_at(
                    localized,
                    localized_geometry,
                    localized_invariants,
                );
                if source.flags != 0u {
                    return failure_result(
                        source.flags,
                        step_index + 1u,
                        travel_time,
                        maximum_drift,
                    );
                }
                source_coordinates = source.coordinates;
            }
            let is_singularity = selection.termination == TERMINATION_SINGULARITY;
            let is_horizon = selection.termination == TERMINATION_HORIZON;
            let is_surface = selection.termination == TERMINATION_EQUATORIAL_SURFACE;
            let localized_value = select(
                select(
                    localized_geometry.radius,
                    localized_geometry.singularity_measure,
                    is_singularity,
                ),
                localized.position.z,
                is_surface,
            );
            let selected_event_surface = select(
                select(
                    select(escape_radius, horizon_radius, is_horizon),
                    singularity_guard,
                    is_singularity,
                ),
                0.0,
                is_surface,
            );
            let event_residual = (localized_value - selected_event_surface)
                / max(1.0, abs(selected_event_surface));
            var termination = selection.termination;
            if selection.ambiguous != 0u || invariant_budget_exceeded(committed_maximum_drift) {
                termination = TERMINATION_UNCERTAIN;
            }
            return GeometricSample(
                termination,
                0u,
                selection.candidates,
                step_index + 1u,
                event_residual,
                source_coordinates,
                committed_travel_time,
                committed_maximum_drift,
            );
        }
        let current_invariants = invariants_from_geometry_rhs(
            stepped.state,
            energy,
            next_geometry,
            next_rhs,
        );
        if current_invariants.flags != 0u {
            return failure_result(
                current_invariants.flags,
                step_index + 1u,
                travel_time,
                maximum_drift,
            );
        }
        travel_time += abs(stepped.coordinate_time_increment);
        maximum_drift = max(
            maximum_drift,
            invariant_drift(initial_invariants.values, current_invariants.values),
        );
        if SURFACE_EVENTS_ENABLED != 0u {
            surface_armed = update_surface_event_arming(surface_armed, stepped.state.position.z);
        }
        state = stepped.state;
        state_geometry = next_geometry;
        state_rhs = next_rhs;
    }

    return GeometricSample(
        TERMINATION_STEP_EXHAUSTION,
        0u,
        0u,
        MAXIMUM_STEPS,
        0.0,
        vec3<f32>(0.0),
        travel_time,
        maximum_drift,
    );
}

fn trace_pixel_at(
    pixel: vec2<u32>,
    extent: vec2<u32>,
    subpixel: vec2<f32>,
) -> GeometricSample {
    let initial = initial_state_at(pixel, extent, subpixel);
    if initial.rhs.flags != 0u {
        return failure_result(initial.rhs.flags, 0u, 0.0, vec4<f32>(0.0));
    }
    let initial_invariants = invariants_from_geometry_rhs(
        initial.state,
        initial.energy,
        initial.geometry,
        initial.rhs,
    );
    if initial_invariants.flags != 0u {
        return failure_result(initial_invariants.flags, 0u, 0.0, vec4<f32>(0.0));
    }
    return trace_initialized(initial, initial_invariants);
}

fn trace_pixel(pixel: vec2<u32>, extent: vec2<u32>) -> GeometricSample {
    return trace_pixel_at(pixel, extent, trace_uniforms.camera.zw);
}

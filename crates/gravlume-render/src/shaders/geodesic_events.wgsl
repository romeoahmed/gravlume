// Dense event localization, source observables, invariants, and exact branch evidence.

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
    let geometry = event_geometry_at(trace_uniforms.observer.yzw);
    var initial_flags = geometry.flags;
    if !finite_vec4(momentum_contravariant) {
        initial_flags |= FLAG_NON_FINITE;
    }
    if initial_flags != 0u {
        return InitialState(
            TraceState(trace_uniforms.observer.yzw, vec3<f32>(0.0)),
            0.0,
            geometry,
            invalid_rhs(initial_flags),
        );
    }
    let momentum_covariant = lower_momentum(momentum_contravariant, geometry);
    if !finite_vec4(momentum_covariant) {
        return InitialState(
            TraceState(trace_uniforms.observer.yzw, vec3<f32>(0.0)),
            0.0,
            geometry,
            invalid_rhs(FLAG_NON_FINITE),
        );
    }
    let energy = -momentum_covariant.x;
    let state = TraceState(trace_uniforms.observer.yzw, momentum_covariant.yzw);
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

fn increment_branch_count(value: u32) -> u32 {
    if value == 0xffffffffu {
        return value;
    }
    return value + 1u;
}

fn radial_traversal_velocity(geometry: Geometry, rhs: RhsResult) -> f32 {
    return -dot(geometry.radius_gradient, rhs.spacetime.yzw);
}

fn dense_radial_traversal_velocity(
    start: TraceState,
    end: TraceState,
    start_rhs: RhsResult,
    end_rhs: RhsResult,
    energy: f32,
    signed_step: f32,
    fraction: f32,
) -> RadialVelocitySample {
    let state = dense_state_at(
        start,
        end,
        start_rhs,
        end_rhs,
        cubic_dense_weights(signed_step, fraction),
    );
    if !finite_vec3(state.position) || !finite_vec3(state.momentum) {
        return RadialVelocitySample(0.0, FLAG_NON_FINITE);
    }
    let geometry = event_geometry_at(state.position);
    let rhs = hamilton_rhs_from_geometry(state, energy, geometry);
    let flags = geometry.flags | rhs.flags;
    if flags != 0u {
        return RadialVelocitySample(0.0, flags);
    }
    let velocity = radial_traversal_velocity(geometry, rhs);
    if !finite_scalar(velocity) {
        return RadialVelocitySample(0.0, FLAG_NON_FINITE);
    }
    return RadialVelocitySample(velocity, 0u);
}

fn localize_radial_turning(
    start: TraceState,
    end: TraceState,
    start_rhs: RhsResult,
    end_rhs: RhsResult,
    energy: f32,
    signed_step: f32,
    start_velocity: f32,
) -> RadialTurningBracket {
    var lower = 0.0;
    var upper = 1.0;
    var lower_velocity = start_velocity;
    for (var iteration = 0u; iteration < TURNING_REFINEMENT_ITERATIONS; iteration += 1u) {
        let middle = 0.5 * (lower + upper);
        if middle == lower || middle == upper {
            break;
        }
        let sample = dense_radial_traversal_velocity(
            start,
            end,
            start_rhs,
            end_rhs,
            energy,
            signed_step,
            middle,
        );
        if sample.flags != 0u {
            return RadialTurningBracket(vec2<f32>(0.0), 0u);
        }
        if sample.value == 0.0 {
            return RadialTurningBracket(vec2<f32>(middle), 1u);
        }
        let on_start_side = (sample.value < 0.0) == (lower_velocity < 0.0);
        if on_start_side {
            lower = middle;
            lower_velocity = sample.value;
        } else {
            upper = middle;
        }
    }
    return RadialTurningBracket(vec2<f32>(lower, upper), 1u);
}

fn update_azimuth_winding(previous: f32, position: vec3<f32>, winding: i32) -> i32 {
    let difference = atan2(position.y, position.x) - previous;
    if difference > 3.141592653589793 {
        return winding - 1i;
    }
    if difference < -3.141592653589793 {
        return winding + 1i;
    }
    return winding;
}

fn trace_branch_key(
    radial_turnings: u32,
    equatorial_crossings: u32,
    azimuth_winding: i32,
    initial_polar_side: u32,
) -> vec4<u32> {
    return vec4<u32>(
        radial_turnings,
        equatorial_crossings,
        bitcast<u32>(azimuth_winding),
        initial_polar_side,
    );
}

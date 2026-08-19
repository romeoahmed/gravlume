// Kerr-Schild metric geometry, Hamiltonian dynamics, and one classical RK4 step.

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

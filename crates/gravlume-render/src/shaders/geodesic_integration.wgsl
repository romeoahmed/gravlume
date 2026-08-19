// Per-ray integration loop and image-space trace entry points.

fn failure_result(
    flags: u32,
    steps: u32,
    travel_time: f32,
    maximum_drift: vec4<f32>,
) -> GeometricSample {
    return GeometricSample(
        TERMINATION_NUMERICAL_FAILURE,
        flags,
        0u,
        steps,
        0.0,
        vec3<f32>(0.0),
        travel_time,
        maximum_drift,
        vec4<u32>(0u),
    );
}

fn trace_initialized(initial: InitialState, initial_invariants: Invariants) -> GeometricSample {
    var state = initial.state;
    let energy = initial.energy;
    var state_geometry = initial.geometry;
    var state_rhs = initial.rhs;
    var maximum_drift = vec4<f32>(initial_invariants.values.x, 0.0, 0.0, 0.0);
    var travel_time = 0.0;
    var radial_turnings = 0u;
    var equatorial_crossings = 0u;
    var azimuth_winding = 0i;
    var previous_azimuth = atan2(state.position.y, state.position.x);
    let initial_polar_side = u32(trace_uniforms.observer.x);
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
                trace_branch_key(
                    radial_turnings,
                    equatorial_crossings,
                    azimuth_winding,
                    initial_polar_side,
                ),
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
        let start_radial_velocity = radial_traversal_velocity(start_geometry, state_rhs);
        let end_radial_velocity = radial_traversal_velocity(next_geometry, next_rhs);
        let crosses_radial_turning = (start_radial_velocity < 0.0
                && end_radial_velocity >= 0.0)
            || (start_radial_velocity > 0.0 && end_radial_velocity <= 0.0);
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
        // Arming controls whether a crossing may terminate on the configured surface. The branch
        // key still commits every accepted plane crossing, including one inside the initial band.
        let crosses_equatorial_plane = SURFACE_EVENTS_ENABLED != 0u
            && ((state.position.z > 0.0 && stepped.state.position.z <= 0.0)
                || (state.position.z < 0.0 && stepped.state.position.z >= 0.0));
        var equatorial_crossing_fraction = 2.0;
        if crosses_equatorial_plane {
            let surface_fraction = event_fraction(
                state.position.z,
                stepped.state.position.z,
                signed_step * state_rhs.spacetime.w,
                signed_step * next_rhs.spacetime.w,
            );
            equatorial_crossing_fraction = surface_fraction;
            if surface_armed {
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
            var committed_radial_turnings = radial_turnings;
            var radial_turning_order_uncertain = false;
            if crosses_radial_turning {
                let turning = localize_radial_turning(
                    state,
                    stepped.state,
                    state_rhs,
                    next_rhs,
                    energy,
                    signed_step,
                    start_radial_velocity,
                );
                if turning.valid == 0u {
                    radial_turning_order_uncertain = true;
                } else if turning.fractions.y < selection.fraction {
                    committed_radial_turnings = increment_branch_count(committed_radial_turnings);
                } else if turning.fractions.x < selection.fraction {
                    // The terminal lies inside the root bracket, so the discrete order is not
                    // justified at binary32 precision.
                    radial_turning_order_uncertain = true;
                }
            }
            var committed_equatorial_crossings = equatorial_crossings;
            if equatorial_crossing_fraction < selection.fraction {
                committed_equatorial_crossings = increment_branch_count(
                    committed_equatorial_crossings,
                );
            }
            var termination = selection.termination;
            if selection.ambiguous != 0u
                || radial_turning_order_uncertain
                || invariant_budget_exceeded(committed_maximum_drift)
            {
                termination = TERMINATION_UNCERTAIN;
            }
            let committed_azimuth_winding = update_azimuth_winding(
                previous_azimuth,
                localized.position,
                azimuth_winding,
            );
            return GeometricSample(
                termination,
                0u,
                selection.candidates,
                step_index + 1u,
                event_residual,
                source_coordinates,
                committed_travel_time,
                committed_maximum_drift,
                trace_branch_key(
                    committed_radial_turnings,
                    committed_equatorial_crossings,
                    committed_azimuth_winding,
                    initial_polar_side,
                ),
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
        if crosses_radial_turning {
            radial_turnings = increment_branch_count(radial_turnings);
        }
        if equatorial_crossing_fraction <= 1.0 {
            equatorial_crossings = increment_branch_count(equatorial_crossings);
        }
        azimuth_winding = update_azimuth_winding(
            previous_azimuth,
            stepped.state.position,
            azimuth_winding,
        );
        previous_azimuth = atan2(stepped.state.position.y, stepped.state.position.x);
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
        trace_branch_key(
            radial_turnings,
            equatorial_crossings,
            azimuth_winding,
            initial_polar_side,
        ),
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

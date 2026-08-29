"""Build and validate the named negative-spin surface proof."""

import mpmath as mp

from ._geometry import (
    _azimuth_winding,
    _canonical_initial_ray,
    _negative_spin_geometry,
    _separated_initial_state,
    _wrap_angle,
)
from ._model import (
    _NEGATIVE_SPIN_EMITTER_BRANCH_SIGN,
    _NEGATIVE_SPIN_PIXEL,
    _NEGATIVE_SPIN_ROOT_CLASS,
    _OUTGOING_CHART_SIGN,
    _RESIDUAL_GUARD_DIGITS,
    _SOURCE_EDGE_INITIAL_POLAR_SIDE,
    _SURFACE_INNER_RADIUS_M,
    _SURFACE_OUTER_RADIUS_M,
    _SURFACE_TERMINAL,
    _NegativeSpinSurfaceWitness,
    _SeparatedState,
    _UnsupportedWitnessError,
    _validate_precision_digits,
)
from ._motion import (
    _build_polar_motion,
    _build_radial_motion,
    _circular_emitter_angular_velocity,
    _classify_radial_barrier,
    _integrate_path_observables,
    _solve_source_radius,
    _surface_transfer_observables,
    _unit_integrand,
)


def _negative_spin_surface_witness(
    *,
    precision_digits: int,
) -> _NegativeSpinSurfaceWitness:
    """Recompute one pre-registered negative-spin surface ray from inputs."""

    _validate_precision_digits(precision_digits)
    with mp.workdps(precision_digits):
        geometry = _negative_spin_geometry()
        initial_ray = _canonical_initial_ray(geometry, *_NEGATIVE_SPIN_PIXEL)
        separated = _separated_initial_state(geometry, initial_ray)
        if separated.radial_velocity <= 0 or separated.polar_velocity >= 0:
            raise _UnsupportedWitnessError(
                "negative-spin surface witness requires the named outgoing polar branch"
            )
        classification = _classify_radial_barrier(geometry, separated)
        if classification.margin >= 0 or len(classification.exterior_roots) != 2:
            raise _UnsupportedWitnessError(
                "negative-spin surface witness lacks the two-root scattering topology"
            )

        initial_mu = mp.cos(geometry.theta)
        polar = _build_polar_motion(
            geometry.spin,
            separated.impact,
            separated.carter,
            initial_mu,
        )
        initial_to_polar_turn = polar.integrate_to_turn(
            _unit_integrand,
            initial_mu,
        )
        equator_to_polar_turn = polar.integrate_to_turn(
            _unit_integrand,
            mp.mpf(0),
        )
        source_mino_duration = initial_to_polar_turn + equator_to_polar_turn

        radial = _build_radial_motion(geometry, separated, precision_digits)
        radial_turn_mino_duration = radial.integrate_from_turn(
            _unit_integrand,
            geometry.radius,
        )
        source_radius = _solve_source_radius(
            radial,
            geometry.radius,
            source_mino_duration,
            precision_digits,
        )
        path = _integrate_path_observables(
            geometry,
            polar,
            radial,
            initial_mu,
            source_radius,
        )
        transfer = _surface_transfer_observables(
            geometry,
            initial_ray,
            separated,
            radial,
            source_radius,
            emitter_branch_sign=_NEGATIVE_SPIN_EMITTER_BRANCH_SIGN,
        )
        horizon = geometry.mass + mp.sqrt(geometry.mass**2 - geometry.spin**2)
        return _NegativeSpinSurfaceWitness(
            pixel=_NEGATIVE_SPIN_PIXEL,
            precision_digits=precision_digits,
            physical_spin_m=geometry.spin,
            chart_sign=geometry.chart_sign,
            emitter_branch_sign=_NEGATIVE_SPIN_EMITTER_BRANCH_SIGN,
            terminal=_SURFACE_TERMINAL,
            initial_polar_side=_SOURCE_EDGE_INITIAL_POLAR_SIDE,
            radial_root_class=_NEGATIVE_SPIN_ROOT_CLASS,
            exterior_radial_root_count=len(classification.exterior_roots),
            radial_turnings=1,
            polar_turnings=1,
            equatorial_crossings_before_terminal=0,
            azimuth_winding=path.azimuth_winding,
            source_mino_duration=source_mino_duration,
            radial_turn_mino_duration=radial_turn_mino_duration,
            source_after_radial_turn_mino_margin=(
                source_mino_duration - radial_turn_mino_duration
            ),
            next_crossing_after_source_mino_margin=2 * equator_to_polar_turn,
            radial_stationary_radius_m=classification.stationary_radius,
            radial_classification_margin=classification.margin,
            radial_turning_radius_m=radial.turning,
            radial_turning_above_horizon_margin_m=radial.turning - horizon,
            source_radius_m=source_radius,
            source_inner_margin_m=source_radius - _SURFACE_INNER_RADIUS_M,
            source_outer_margin_m=_SURFACE_OUTER_RADIUS_M - source_radius,
            source_azimuth_unwrapped_rad=path.terminal_azimuth_unwrapped,
            source_azimuth_rad=path.terminal_azimuth,
            emitter_angular_velocity_per_m=_circular_emitter_angular_velocity(
                geometry,
                source_radius,
                _NEGATIVE_SPIN_EMITTER_BRANCH_SIGN,
            ),
            frequency_ratio=transfer.frequency_ratio,
            travel_time_m=path.travel_time,
            emitted_bolometric_intensity=transfer.emitted_intensity,
            observed_bolometric_intensity=transfer.observed_intensity,
            energy=separated.energy,
            impact_parameter=separated.impact,
            carter_parameter=separated.carter,
            radial_turning_derivative=radial.turning_derivative,
            polar_turning_derivative=polar.turning_derivative,
            initial_null_residual=initial_ray.initial_null_residual,
            mino_constraint_residual=separated.constraint_residual,
            chart_primitive_residual=path.chart_primitive_residual,
        )


def _validate_negative_spin_surface_witness(
    witness: _NegativeSpinSurfaceWitness,
) -> None:
    _validate_precision_digits(witness.precision_digits)
    if witness.pixel != _NEGATIVE_SPIN_PIXEL:
        raise _UnsupportedWitnessError(
            "negative-spin witness does not use its pre-registered pixel"
        )
    discrete_identity = (
        (witness.chart_sign, _OUTGOING_CHART_SIGN),
        (witness.emitter_branch_sign, _NEGATIVE_SPIN_EMITTER_BRANCH_SIGN),
        (witness.terminal, _SURFACE_TERMINAL),
        (witness.initial_polar_side, _SOURCE_EDGE_INITIAL_POLAR_SIDE),
        (witness.radial_root_class, _NEGATIVE_SPIN_ROOT_CLASS),
        (witness.exterior_radial_root_count, 2),
        (witness.radial_turnings, 1),
        (witness.polar_turnings, 1),
        (witness.equatorial_crossings_before_terminal, 0),
        (witness.azimuth_winding, 0),
    )
    if any(
        type(actual) is not type(expected) or actual != expected
        for actual, expected in discrete_identity
    ):
        raise _UnsupportedWitnessError(
            "negative-spin witness does not match its named path identity"
        )

    continuous_fields = (
        witness.physical_spin_m,
        witness.source_mino_duration,
        witness.radial_turn_mino_duration,
        witness.source_after_radial_turn_mino_margin,
        witness.next_crossing_after_source_mino_margin,
        witness.radial_stationary_radius_m,
        witness.radial_classification_margin,
        witness.radial_turning_radius_m,
        witness.radial_turning_above_horizon_margin_m,
        witness.source_radius_m,
        witness.source_inner_margin_m,
        witness.source_outer_margin_m,
        witness.source_azimuth_unwrapped_rad,
        witness.source_azimuth_rad,
        witness.emitter_angular_velocity_per_m,
        witness.frequency_ratio,
        witness.travel_time_m,
        witness.emitted_bolometric_intensity,
        witness.observed_bolometric_intensity,
        witness.energy,
        witness.impact_parameter,
        witness.carter_parameter,
        witness.radial_turning_derivative,
        witness.polar_turning_derivative,
        witness.initial_null_residual,
        witness.mino_constraint_residual,
        witness.chart_primitive_residual,
    )
    if not all(
        isinstance(value, mp.mpf) and mp.isfinite(value) for value in continuous_fields
    ):
        raise _UnsupportedWitnessError(
            "negative-spin witness contains a non-real or non-finite value"
        )

    positive_fields = (
        witness.source_mino_duration,
        witness.radial_turn_mino_duration,
        witness.source_after_radial_turn_mino_margin,
        witness.next_crossing_after_source_mino_margin,
        witness.radial_stationary_radius_m,
        witness.radial_turning_radius_m,
        witness.radial_turning_above_horizon_margin_m,
        witness.source_radius_m,
        witness.source_inner_margin_m,
        witness.source_outer_margin_m,
        witness.frequency_ratio,
        witness.travel_time_m,
        witness.emitted_bolometric_intensity,
        witness.observed_bolometric_intensity,
        witness.energy,
        witness.carter_parameter,
        witness.radial_turning_derivative,
        witness.polar_turning_derivative,
    )
    if any(value <= 0 for value in positive_fields):
        raise _UnsupportedWitnessError(
            "negative-spin witness contains a non-positive physical margin"
        )
    if witness.radial_classification_margin >= 0:
        raise _UnsupportedWitnessError(
            "negative-spin witness lacks a certified scattering barrier"
        )
    if witness.emitter_angular_velocity_per_m >= 0:
        raise _UnsupportedWitnessError(
            "negative-spin witness does not use the negative emitter branch"
        )

    with mp.workdps(witness.precision_digits):
        geometry = _negative_spin_geometry()
        if witness.physical_spin_m != geometry.spin:
            raise _UnsupportedWitnessError(
                "negative-spin witness changed the physical Kerr spin"
            )
        separated = _SeparatedState(
            energy=witness.energy,
            impact=witness.impact_parameter,
            carter=witness.carter_parameter,
            radial_velocity=mp.mpf(0),
            polar_velocity=mp.mpf(0),
            constraint_residual=witness.mino_constraint_residual,
        )
        classification = _classify_radial_barrier(geometry, separated)
        if len(classification.exterior_roots) != witness.exterior_radial_root_count:
            raise _UnsupportedWitnessError(
                "negative-spin witness changed the exterior radial-root count"
            )
        initial_mu = mp.cos(geometry.theta)
        polar = _build_polar_motion(
            geometry.spin,
            witness.impact_parameter,
            witness.carter_parameter,
            initial_mu,
        )
        initial_to_polar_turn = polar.integrate_to_turn(
            _unit_integrand,
            initial_mu,
        )
        equator_to_polar_turn = polar.integrate_to_turn(
            _unit_integrand,
            mp.mpf(0),
        )
        expected_source_mino_duration = initial_to_polar_turn + equator_to_polar_turn
        radial = _build_radial_motion(
            geometry,
            separated,
            witness.precision_digits,
        )
        expected_radial_turn_mino_duration = radial.integrate_from_turn(
            _unit_integrand,
            geometry.radius,
        )
        expected_angular_velocity = _circular_emitter_angular_velocity(
            geometry,
            witness.source_radius_m,
            witness.emitter_branch_sign,
        )
        g_tt = -1 + 2 * geometry.mass / witness.source_radius_m
        g_t_phi = -2 * geometry.mass * geometry.spin / witness.source_radius_m
        delta = (
            witness.source_radius_m**2
            - 2 * geometry.mass * witness.source_radius_m
            + geometry.spin**2
        )
        g_phi_phi = (
            (witness.source_radius_m**2 + geometry.spin**2) ** 2
            - geometry.spin**2 * delta
        ) / witness.source_radius_m**2
        emitter_time_component = 1 / mp.sqrt(
            -(
                g_tt
                + 2 * expected_angular_velocity * g_t_phi
                + expected_angular_velocity**2 * g_phi_phi
            )
        )
        expected_frequency_ratio = 1 / (
            emitter_time_component
            * witness.energy
            * (1 - expected_angular_velocity * witness.impact_parameter)
        )
        expected_emitted_intensity = (
            witness.source_radius_m / _SURFACE_INNER_RADIUS_M
        ) ** -3
        expected_observed_intensity = (
            expected_emitted_intensity * expected_frequency_ratio**4
        )
        horizon = geometry.mass + mp.sqrt(geometry.mass**2 - geometry.spin**2)
        expected_values = (
            (witness.source_mino_duration, expected_source_mino_duration),
            (
                witness.radial_turn_mino_duration,
                expected_radial_turn_mino_duration,
            ),
            (
                witness.source_after_radial_turn_mino_margin,
                expected_source_mino_duration - expected_radial_turn_mino_duration,
            ),
            (
                witness.next_crossing_after_source_mino_margin,
                2 * equator_to_polar_turn,
            ),
            (
                witness.radial_stationary_radius_m,
                classification.stationary_radius,
            ),
            (witness.radial_classification_margin, classification.margin),
            (witness.radial_turning_radius_m, radial.turning),
            (radial.turning, max(classification.exterior_roots)),
            (
                witness.radial_turning_above_horizon_margin_m,
                radial.turning - horizon,
            ),
            (
                witness.source_inner_margin_m,
                witness.source_radius_m - _SURFACE_INNER_RADIUS_M,
            ),
            (
                witness.source_outer_margin_m,
                _SURFACE_OUTER_RADIUS_M - witness.source_radius_m,
            ),
            (
                witness.source_azimuth_rad,
                _wrap_angle(witness.source_azimuth_unwrapped_rad),
            ),
            (witness.emitter_angular_velocity_per_m, expected_angular_velocity),
            (witness.frequency_ratio, expected_frequency_ratio),
            (witness.emitted_bolometric_intensity, expected_emitted_intensity),
            (witness.observed_bolometric_intensity, expected_observed_intensity),
            (witness.radial_turning_derivative, radial.turning_derivative),
            (witness.polar_turning_derivative, polar.turning_derivative),
        )
        if (
            _azimuth_winding(
                geometry,
                witness.source_radius_m,
                witness.source_azimuth_unwrapped_rad,
            )
            != witness.azimuth_winding
        ):
            raise _UnsupportedWitnessError(
                "negative-spin witness changed the signed azimuth winding"
            )
        residual_limit = mp.power(
            10,
            _RESIDUAL_GUARD_DIGITS - witness.precision_digits,
        )
        relation_residuals = tuple(
            abs(actual - expected) / max(mp.mpf(1), abs(expected))
            for actual, expected in expected_values
        )
        equation_residuals = (
            witness.initial_null_residual,
            witness.mino_constraint_residual,
            witness.chart_primitive_residual,
        )
        if any(
            residual < 0 or residual >= residual_limit
            for residual in (*relation_residuals, *equation_residuals)
        ):
            raise _UnsupportedWitnessError(
                "negative-spin relation or equation residual lost certified digits"
            )

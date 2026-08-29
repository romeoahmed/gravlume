"""Solve separated radial/polar motion and observable quadratures.

The equations follow Carter's separated Hamilton--Jacobi system and Mino's
affine reparameterization.  Endpoint quadratures remain private to the named
proofs and are not a general Kerr solver.
"""

import mpmath as mp

from ._geometry import (
    _azimuth_winding,
    _canonical_initial_ray,
    _chart_primitives,
    _real_polynomial_roots,
    _separated_initial_state,
    _wrap_angle,
)
from ._model import (
    _CRITICAL_CAPTURE_PIXEL,
    _CRITICAL_SURFACE_PIXEL,
    _OUTGOING_CHART_SIGN,
    _SURFACE_INNER_RADIUS_M,
    _SURFACE_OUTER_RADIUS_M,
    _CaptureRadialMotion,
    _CriticalPoint,
    _InitialRay,
    _ObservationGeometry,
    _PathObservables,
    _PolarMotion,
    _RadialClassification,
    _RadialMotion,
    _SeparatedState,
    _TransferObservables,
    _UnsupportedWitnessError,
)


def _unit_integrand(_: mp.mpf) -> mp.mpf:
    return mp.mpf(1)


def _build_polar_motion(
    spin: mp.mpf,
    impact: mp.mpf,
    carter: mp.mpf,
    initial_mu: mp.mpf,
) -> _PolarMotion:
    quadratic_coefficient = spin**2 - carter - impact**2
    discriminant = mp.sqrt(quadratic_coefficient**2 + 4 * spin**2 * carter)
    turning_squared = (quadratic_coefficient + discriminant) / (2 * spin**2)
    negative_turning_squared = (quadratic_coefficient - discriminant) / (2 * spin**2)
    turning = mp.sqrt(turning_squared)
    if not initial_mu < turning < 1 or negative_turning_squared >= 0:
        raise _UnsupportedWitnessError(
            "polar potential does not have the required simple turning"
        )
    return _PolarMotion(
        spin=spin,
        turning=turning,
        turning_squared=turning_squared,
        negative_turning_squared=negative_turning_squared,
        quadratic_coefficient=quadratic_coefficient,
    )


def _build_radial_motion(
    geometry: _ObservationGeometry,
    separated: _SeparatedState,
    precision_digits: int,
) -> _RadialMotion:
    mass = geometry.mass
    spin = geometry.spin
    impact = separated.impact
    separation = (impact - spin) ** 2 + separated.carter
    radial_constant = spin**2 - spin * impact
    quadratic_coefficient = 2 * radial_constant - separation
    linear_coefficient = 2 * mass * separation

    def delta(radius: mp.mpf) -> mp.mpf:
        return radius**2 - 2 * mass * radius + spin**2

    def factor(radius: mp.mpf) -> mp.mpf:
        return radius**2 + spin**2 - spin * impact

    def potential(radius: mp.mpf) -> mp.mpf:
        return factor(radius) ** 2 - delta(radius) * separation

    roots = _real_polynomial_roots(
        (
            mp.mpf(1),
            mp.mpf(0),
            quadratic_coefficient,
            linear_coefficient,
            radial_constant**2 - spin**2 * separation,
        )
    )
    horizon_radius = mass + mp.sqrt(mass**2 - spin**2)
    turning_candidates = tuple(
        root for root in roots if horizon_radius < root < geometry.radius
    )
    if not turning_candidates:
        raise _UnsupportedWitnessError(
            "radial root topology has no exterior turning point"
        )
    turning = max(turning_candidates)
    if turning >= _SURFACE_OUTER_RADIUS_M:
        raise _UnsupportedWitnessError(
            "radial turning precedes the canonical source edge"
        )
    turning = mp.findroot(
        potential,
        turning,
        tol=mp.power(10, -(precision_digits - 20)),
        verify=True,
    )
    turning_derivative = (
        4 * turning * factor(turning) - 2 * (turning - mass) * separation
    )
    if turning_derivative <= 0:
        raise _UnsupportedWitnessError(
            "radial turning root is not simple and outward-facing"
        )
    return _RadialMotion(
        mass=mass,
        spin=spin,
        impact=impact,
        turning=turning,
        turning_derivative=turning_derivative,
        quadratic_coefficient=quadratic_coefficient,
        linear_coefficient=linear_coefficient,
    )


def _radial_polynomial_coefficients(
    geometry: _ObservationGeometry,
    separated: _SeparatedState,
) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    spin = geometry.spin
    impact = separated.impact
    separation = (impact - spin) ** 2 + separated.carter
    radial_constant = spin**2 - spin * impact
    return (
        2 * radial_constant - separation,
        2 * geometry.mass * separation,
        radial_constant**2 - spin**2 * separation,
    )


def _evaluate_radial_polynomial(
    coefficients: tuple[mp.mpf, mp.mpf, mp.mpf],
    radius: mp.mpf,
) -> mp.mpf:
    quadratic, linear, constant = coefficients
    return radius**4 + quadratic * radius**2 + linear * radius + constant


def _stationary_radial_barrier(
    geometry: _ObservationGeometry,
    coefficients: tuple[mp.mpf, mp.mpf, mp.mpf],
) -> tuple[mp.mpf, mp.mpf]:
    quadratic, linear, _ = coefficients
    horizon = geometry.mass + mp.sqrt(geometry.mass**2 - geometry.spin**2)
    stationary = _real_polynomial_roots((mp.mpf(4), mp.mpf(0), 2 * quadratic, linear))
    minima = tuple(
        radius
        for radius in stationary
        if horizon < radius < geometry.radius and 12 * radius**2 + 2 * quadratic > 0
    )
    if not minima:
        raise _UnsupportedWitnessError(
            "radial topology has no exterior potential minimum"
        )
    stationary_radius = min(
        minima,
        key=lambda radius: _evaluate_radial_polynomial(coefficients, radius),
    )
    return (
        stationary_radius,
        _evaluate_radial_polynomial(coefficients, stationary_radius),
    )


def _radial_barrier_at_sample_y(
    geometry: _ObservationGeometry,
    sample_y: mp.mpf,
) -> tuple[_SeparatedState, tuple[mp.mpf, mp.mpf, mp.mpf], mp.mpf, mp.mpf]:
    sample_x = mp.mpf(_CRITICAL_SURFACE_PIXEL[0]) + mp.mpf(1) / 2
    initial_ray = _canonical_initial_ray(
        geometry,
        sample_x,
        sample_y,
        coordinates_are_centers=True,
    )
    separated = _separated_initial_state(geometry, initial_ray)
    coefficients = _radial_polynomial_coefficients(geometry, separated)
    stationary_radius, margin = _stationary_radial_barrier(
        geometry,
        coefficients,
    )
    return separated, coefficients, stationary_radius, margin


def _solve_critical_point(
    geometry: _ObservationGeometry,
    precision_digits: int,
) -> _CriticalPoint:
    lower_sample_y = mp.mpf(_CRITICAL_SURFACE_PIXEL[1]) + mp.mpf(1) / 2
    upper_sample_y = mp.mpf(_CRITICAL_CAPTURE_PIXEL[1]) + mp.mpf(1) / 2

    def barrier_margin(sample_y: mp.mpf) -> mp.mpf:
        return _radial_barrier_at_sample_y(geometry, sample_y)[3]

    lower_margin = barrier_margin(lower_sample_y)
    upper_margin = barrier_margin(upper_sample_y)
    if not lower_margin < 0 < upper_margin:
        raise _UnsupportedWitnessError(
            "named sample centers do not bracket the radial separatrix"
        )
    sample_y = mp.findroot(
        barrier_margin,
        (lower_sample_y, upper_sample_y),
        solver="anderson",
        tol=mp.power(10, -(precision_digits - 25)),
        verify=True,
    )
    _, coefficients, radius, potential = _radial_barrier_at_sample_y(
        geometry,
        sample_y,
    )
    quadratic, linear, constant = coefficients
    derivative = 4 * radius**3 + 2 * quadratic * radius + linear
    second_derivative = 12 * radius**2 + 2 * quadratic
    potential_scale = max(
        mp.mpf(1),
        abs(radius**4),
        abs(quadratic * radius**2),
        abs(linear * radius),
        abs(constant),
    )
    derivative_scale = max(
        mp.mpf(1),
        abs(4 * radius**3),
        abs(2 * quadratic * radius),
        abs(linear),
    )
    return _CriticalPoint(
        sample_y=sample_y,
        radius=radius,
        potential_residual=abs(potential) / potential_scale,
        derivative_residual=abs(derivative) / derivative_scale,
        second_derivative=second_derivative,
    )


def _classify_radial_barrier(
    geometry: _ObservationGeometry,
    separated: _SeparatedState,
) -> _RadialClassification:
    """Classify the relevant exterior minimum without consuming a trace."""

    coefficients = _radial_polynomial_coefficients(geometry, separated)
    quadratic_coefficient, linear_coefficient, constant = coefficients
    horizon = geometry.mass + mp.sqrt(geometry.mass**2 - geometry.spin**2)
    roots = _real_polynomial_roots(
        (mp.mpf(1), mp.mpf(0), quadratic_coefficient, linear_coefficient, constant)
    )
    exterior_roots = tuple(root for root in roots if horizon < root < geometry.radius)
    stationary_radius, margin = _stationary_radial_barrier(
        geometry,
        coefficients,
    )
    return _RadialClassification(
        margin=margin,
        stationary_radius=stationary_radius,
        exterior_roots=exterior_roots,
    )


def _build_capture_radial_motion(
    geometry: _ObservationGeometry,
    separated: _SeparatedState,
    classification: _RadialClassification,
) -> _CaptureRadialMotion:
    if geometry.chart_sign != _OUTGOING_CHART_SIGN:
        raise _UnsupportedWitnessError(
            "horizon capture witness requires the outgoing Kerr--Schild chart"
        )
    if classification.margin <= 0 or classification.exterior_roots:
        raise _UnsupportedWitnessError(
            "capture witness does not have a strictly positive exterior barrier"
        )
    separation = (separated.impact - geometry.spin) ** 2 + separated.carter
    horizon = geometry.mass + mp.sqrt(geometry.mass**2 - geometry.spin**2)
    return _CaptureRadialMotion(
        mass=geometry.mass,
        spin=geometry.spin,
        impact=separated.impact,
        separation=separation,
        horizon=horizon,
        observer_radius=geometry.radius,
    )


def _solve_equatorial_crossing_radius(
    radial: _RadialMotion,
    observer_radius: mp.mpf,
    polar_mino_duration: mp.mpf,
    precision_digits: int,
    lower_radius: mp.mpf,
    upper_radius: mp.mpf,
) -> mp.mpf:
    observer_duration = radial.integrate_from_turn(
        _unit_integrand,
        observer_radius,
    )

    def crossing_equation(crossing_radius: mp.mpf) -> mp.mpf:
        return (
            observer_duration
            + radial.integrate_from_turn(_unit_integrand, crossing_radius)
            - polar_mino_duration
        )

    lower_value = crossing_equation(lower_radius)
    upper_value = crossing_equation(upper_radius)
    if not (lower_value < 0 and upper_value > 0):
        raise _UnsupportedWitnessError(
            "first equatorial crossing is not bracketed by the named interval: "
            f"lower={mp.nstr(lower_value, 12)}, "
            f"upper={mp.nstr(upper_value, 12)}, "
            f"polar={mp.nstr(polar_mino_duration, 12)}"
        )
    return mp.findroot(
        crossing_equation,
        (lower_radius, upper_radius),
        solver="anderson",
        tol=mp.power(10, -(precision_digits - 25)),
        verify=True,
    )


def _solve_inbound_radius(
    radial: _RadialMotion,
    observer_radius: mp.mpf,
    target_mino_duration: mp.mpf,
    precision_digits: int,
) -> mp.mpf:
    """Invert the observer-to-turn branch before the radial turning event."""

    observer_to_turn = radial.integrate_from_turn(
        _unit_integrand,
        observer_radius,
    )
    if not 0 < target_mino_duration < observer_to_turn:
        raise _UnsupportedWitnessError(
            "requested event is not on the inbound radial branch"
        )

    def event_equation(radius: mp.mpf) -> mp.mpf:
        return (
            target_mino_duration
            - observer_to_turn
            + radial.integrate_from_turn(_unit_integrand, radius)
        )

    return mp.findroot(
        event_equation,
        (radial.turning, observer_radius),
        solver="anderson",
        tol=mp.power(10, -(precision_digits - 25)),
        verify=True,
    )


def _solve_capture_radius(
    radial: _CaptureRadialMotion,
    target_mino_duration: mp.mpf,
    precision_digits: int,
) -> mp.mpf:
    """Invert one monotonic observer-to-horizon capture segment."""

    total_duration = radial.mino_duration()
    if not 0 < target_mino_duration < total_duration:
        raise _UnsupportedWitnessError(
            "requested event is not on the capture radial segment"
        )

    def event_equation(radius: mp.mpf) -> mp.mpf:
        return target_mino_duration - radial.mino_duration_to(radius)

    return mp.findroot(
        event_equation,
        (radial.horizon, radial.observer_radius),
        solver="anderson",
        tol=mp.power(10, -(precision_digits - 25)),
        verify=True,
    )


def _solve_source_radius(
    radial: _RadialMotion,
    observer_radius: mp.mpf,
    polar_mino_duration: mp.mpf,
    precision_digits: int,
) -> mp.mpf:
    return _solve_equatorial_crossing_radius(
        radial,
        observer_radius,
        polar_mino_duration,
        precision_digits,
        radial.turning,
        mp.mpf(_SURFACE_OUTER_RADIUS_M),
    )


def _integrate_polar_observables(
    geometry: _ObservationGeometry,
    polar: _PolarMotion,
    impact: mp.mpf,
    initial_mu: mp.mpf,
    terminal_mu_magnitude: mp.mpf,
    completed_polar_oscillations: int,
) -> tuple[mp.mpf, mp.mpf]:
    """Integrate even polar primitives along one explicitly counted branch."""

    if (
        type(completed_polar_oscillations) is not int
        or completed_polar_oscillations < 0
    ):
        raise _UnsupportedWitnessError(
            "completed polar oscillations must be a non-negative integer"
        )
    spin = geometry.spin

    def polar_time_numerator(mu: mp.mpf) -> mp.mpf:
        return spin * (impact - spin) + spin**2 * mu**2

    def polar_azimuth_numerator(mu: mp.mpf) -> mp.mpf:
        return impact / (1 - mu**2) - spin

    quarter_multiplier = 2 * completed_polar_oscillations + 1
    polar_time = polar.integrate_to_turn(polar_time_numerator, initial_mu)
    polar_time += quarter_multiplier * polar.integrate_to_turn(
        polar_time_numerator,
        mp.mpf(0),
    )
    polar_time += polar.integrate_from_equator(
        polar_time_numerator,
        terminal_mu_magnitude,
    )

    polar_azimuth = polar.integrate_to_turn(
        polar_azimuth_numerator,
        initial_mu,
    )
    polar_azimuth += quarter_multiplier * polar.integrate_to_turn(
        polar_azimuth_numerator,
        mp.mpf(0),
    )
    polar_azimuth += polar.integrate_from_equator(
        polar_azimuth_numerator,
        terminal_mu_magnitude,
    )
    return polar_time, polar_azimuth


def _integrate_path_observables(
    geometry: _ObservationGeometry,
    polar: _PolarMotion,
    radial: _RadialMotion,
    initial_mu: mp.mpf,
    terminal_radius: mp.mpf,
    terminal_mu_magnitude: mp.mpf | None = None,
    completed_polar_oscillations: int = 0,
) -> _PathObservables:
    terminal_mu_magnitude = (
        mp.mpf(0) if terminal_mu_magnitude is None else terminal_mu_magnitude
    )
    spin = geometry.spin
    impact = radial.impact

    def radial_time_numerator(radius: mp.mpf) -> mp.mpf:
        return (radius**2 + spin**2) * radial.factor(radius) / radial.delta(radius)

    radial_time = radial.integrate_from_turn(radial_time_numerator, terminal_radius)
    radial_time += radial.integrate_from_turn(radial_time_numerator, geometry.radius)

    polar_time, polar_azimuth = _integrate_polar_observables(
        geometry,
        polar,
        impact,
        initial_mu,
        terminal_mu_magnitude,
        completed_polar_oscillations,
    )

    def radial_azimuth_numerator(radius: mp.mpf) -> mp.mpf:
        return spin * radial.factor(radius) / radial.delta(radius)

    radial_azimuth = radial.integrate_from_turn(
        radial_azimuth_numerator,
        terminal_radius,
    )
    radial_azimuth += radial.integrate_from_turn(
        radial_azimuth_numerator,
        geometry.radius,
    )

    chart_time_quad = mp.quad(
        lambda radius: 2 * geometry.mass * radius / radial.delta(radius),
        [terminal_radius, geometry.radius],
    )
    chart_azimuth_quad = mp.quad(
        lambda radius: spin / radial.delta(radius),
        [terminal_radius, geometry.radius],
    )
    chart_time, chart_azimuth = _chart_primitives(
        terminal_radius,
        geometry.radius,
        geometry.mass,
        spin,
    )
    chart_residual = max(
        abs(chart_time_quad - chart_time) / max(mp.mpf(1), abs(chart_time)),
        abs(chart_azimuth_quad - chart_azimuth) / max(mp.mpf(1), abs(chart_azimuth)),
    )
    signed_chart_time = geometry.chart_sign * chart_time
    signed_chart_azimuth = geometry.chart_sign * chart_azimuth
    source_azimuth_unwrapped = geometry.chart_azimuth - (
        radial_azimuth + polar_azimuth + signed_chart_azimuth
    )
    return _PathObservables(
        terminal_azimuth_unwrapped=source_azimuth_unwrapped,
        terminal_azimuth=_wrap_angle(source_azimuth_unwrapped),
        travel_time=radial_time + polar_time + signed_chart_time,
        azimuth_winding=_azimuth_winding(
            geometry,
            terminal_radius,
            source_azimuth_unwrapped,
        ),
        chart_primitive_residual=chart_residual,
    )


def _solve_escape_polar_endpoint(
    polar: _PolarMotion,
    radial: _RadialMotion,
    observer_radius: mp.mpf,
    escape_radius: mp.mpf,
    initial_mu: mp.mpf,
    precision_digits: int,
) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Resolve the negative-polar endpoint and the next-crossing event margin."""

    initial_to_turn = polar.integrate_to_turn(_unit_integrand, initial_mu)
    equator_to_turn = polar.integrate_to_turn(_unit_integrand, mp.mpf(0))
    first_crossing_duration = initial_to_turn + equator_to_turn
    escape_duration = radial.integrate_from_turn(
        _unit_integrand,
        observer_radius,
    ) + radial.integrate_from_turn(_unit_integrand, escape_radius)
    after_first_crossing = escape_duration - first_crossing_duration
    if not 0 < after_first_crossing < equator_to_turn:
        raise _UnsupportedWitnessError(
            "escape is not ordered between the first crossing and southern turning"
        )
    target_to_turn = equator_to_turn - after_first_crossing
    terminal_mu_magnitude = mp.findroot(
        lambda mu: polar.integrate_to_turn(_unit_integrand, mu) - target_to_turn,
        (mp.mpf(0), polar.turning),
        solver="anderson",
        tol=mp.power(10, -(precision_digits - 25)),
        verify=True,
    )
    next_crossing_duration = first_crossing_duration + 2 * equator_to_turn
    return (
        terminal_mu_magnitude,
        first_crossing_duration,
        next_crossing_duration - escape_duration,
    )


def _solve_capture_polar_endpoint(
    polar: _PolarMotion,
    capture_mino_duration: mp.mpf,
    initial_mu: mp.mpf,
    precision_digits: int,
) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Resolve a horizon endpoint after crossing but before the southern turn."""

    initial_to_turn = polar.integrate_to_turn(_unit_integrand, initial_mu)
    equator_to_turn = polar.integrate_to_turn(_unit_integrand, mp.mpf(0))
    first_crossing_duration = initial_to_turn + equator_to_turn
    after_first_crossing = capture_mino_duration - first_crossing_duration
    if not 0 < after_first_crossing < equator_to_turn:
        raise _UnsupportedWitnessError(
            "horizon is not ordered between the first crossing and southern turning"
        )
    terminal_mu_magnitude = mp.findroot(
        lambda mu: (
            polar.integrate_from_equator(_unit_integrand, mu) - after_first_crossing
        ),
        (mp.mpf(0), polar.turning),
        solver="anderson",
        tol=mp.power(10, -(precision_digits - 25)),
        verify=True,
    )
    return (
        terminal_mu_magnitude,
        first_crossing_duration,
        after_first_crossing,
    )


def _escape_position_and_direction(
    geometry: _ObservationGeometry,
    radial: _RadialMotion,
    polar: _PolarMotion,
    terminal_radius: mp.mpf,
    terminal_mu_magnitude: mp.mpf,
    terminal_azimuth: mp.mpf,
) -> tuple[tuple[mp.mpf, mp.mpf, mp.mpf], tuple[mp.mpf, mp.mpf, mp.mpf]]:
    """Map the separated endpoint to ingoing Cartesian traversal observables."""

    spin = geometry.spin
    terminal_mu = -terminal_mu_magnitude
    sin_theta = mp.sqrt(1 - terminal_mu**2)
    theta = mp.acos(terminal_mu)
    sin_azimuth = mp.sin(terminal_azimuth)
    cos_azimuth = mp.cos(terminal_azimuth)
    position = (
        (terminal_radius * cos_azimuth - spin * sin_azimuth) * sin_theta,
        (terminal_radius * sin_azimuth + spin * cos_azimuth) * sin_theta,
        terminal_radius * terminal_mu,
    )

    radial_velocity = -mp.sqrt(radial.potential(terminal_radius))
    polar_potential = (
        spin**2
        * (polar.turning_squared - terminal_mu**2)
        * (terminal_mu**2 - polar.negative_turning_squared)
    )
    mu_velocity = mp.sqrt(polar_potential)
    theta_velocity = -mu_velocity / sin_theta
    bl_azimuth_velocity = (
        spin * radial.factor(terminal_radius) / radial.delta(terminal_radius)
        + radial.impact / (1 - terminal_mu**2)
        - spin
    )
    ks_azimuth_velocity = (
        bl_azimuth_velocity + spin / radial.delta(terminal_radius) * radial_velocity
    )

    x, y, _ = position
    radial_basis = (
        sin_theta * cos_azimuth,
        sin_theta * sin_azimuth,
        terminal_mu,
    )
    polar_basis = (
        mp.cot(theta) * x,
        mp.cot(theta) * y,
        -terminal_radius * sin_theta,
    )
    azimuth_basis = (-y, x, mp.mpf(0))
    physical_tangent = tuple(
        radial_basis[index] * radial_velocity
        + polar_basis[index] * theta_velocity
        + azimuth_basis[index] * ks_azimuth_velocity
        for index in range(3)
    )
    tangent_norm = mp.sqrt(mp.fsum(component**2 for component in physical_tangent))
    if tangent_norm <= 0 or not mp.isfinite(tangent_norm):
        raise _UnsupportedWitnessError("escape traversal direction is unavailable")
    traversal_direction = tuple(
        -component / tangent_norm for component in physical_tangent
    )
    return position, traversal_direction


def _circular_emitter_angular_velocity(
    geometry: _ObservationGeometry,
    source_radius: mp.mpf,
    branch_sign: int,
) -> mp.mpf:
    """Return the signed pure-Kerr equatorial circular-orbit frequency."""

    if type(branch_sign) is not int or branch_sign not in (-1, 1):
        raise _UnsupportedWitnessError("emitter branch sign must be -1 or 1")
    circular_root = mp.sqrt(geometry.mass * source_radius)
    denominator = source_radius**2 + branch_sign * geometry.spin * circular_root
    if not mp.isfinite(denominator) or denominator <= 0:
        raise _UnsupportedWitnessError("circular emitter branch is unavailable")
    return branch_sign * circular_root / denominator


def _surface_transfer_observables(
    geometry: _ObservationGeometry,
    initial_ray: _InitialRay,
    separated: _SeparatedState,
    radial: _RadialMotion,
    source_radius: mp.mpf,
    *,
    emitter_branch_sign: int,
) -> _TransferObservables:
    spin = geometry.spin
    angular_velocity = _circular_emitter_angular_velocity(
        geometry,
        source_radius,
        emitter_branch_sign,
    )
    g_tt = -1 + 2 * geometry.mass / source_radius
    g_t_phi = -2 * geometry.mass * spin / source_radius
    g_phi_phi = (
        (source_radius**2 + spin**2) ** 2 - spin**2 * radial.delta(source_radius)
    ) / source_radius**2
    emitter_time_component = 1 / mp.sqrt(
        -(g_tt + 2 * angular_velocity * g_t_phi + angular_velocity**2 * g_phi_phi)
    )
    emitter_frequency = (
        emitter_time_component
        * separated.energy
        * (1 - angular_velocity * separated.impact)
    )
    frequency_ratio = initial_ray.observer_frequency / emitter_frequency
    emitted_intensity = (source_radius / _SURFACE_INNER_RADIUS_M) ** -3
    return _TransferObservables(
        frequency_ratio=frequency_ratio,
        emitted_intensity=emitted_intensity,
        observed_intensity=emitted_intensity * frequency_ratio**4,
    )

"""Build an independent high-precision BL/Mino witness for one surface ray.

This research-only module reconstructs the canonical observation from decimal
physics inputs, maps its photon covector to Boyer--Lindquist constants, and
solves the first certified equatorial crossing with turning-segment Mino quadratures.
It does not import Gravlume, consume a Rust trace, or enter the Cargo runtime.

Primary mathematical sources:

* Carter's separated Hamilton--Jacobi equations:
  https://doi.org/10.1103/PhysRev.174.1559
* Manifestly real Kerr null-geodesic integrals and root classification:
  https://doi.org/10.1103/PhysRevD.101.044032
* Mino's affine reparameterization:
  https://doi.org/10.1103/PhysRevD.67.084027
* Arbitrary-precision quadrature behavior used here:
  https://mpmath.org/doc/current/calculus/integration.html
* Verified root finding and polynomial-root conditioning:
  https://mpmath.org/doc/current/calculus/optimization.html
  https://mpmath.org/doc/current/calculus/polynomials.html
* Precision contexts and finite-value classification:
  https://mpmath.org/doc/1.3.0/general.html
* Python NaN comparison semantics:
  https://docs.python.org/3/reference/expressions.html#value-comparisons
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import mpmath as mp

LOW_PRECISION_DIGITS = 120
HIGH_PRECISION_DIGITS = 180
REQUIRED_STABLE_DIGITS = 80
MINIMUM_WITNESS_DIGITS = 70
RESIDUAL_GUARD_DIGITS = 15
SURFACE_INNER_RADIUS_M = 6
SURFACE_OUTER_RADIUS_M = 20


class UnsupportedWitness(ValueError):
    """The requested case lies outside this research slice's certified domain."""


@dataclass(frozen=True)
class SurfaceWitness:
    """Certified independent observables and margins for one surface terminal."""

    precision_digits: int
    terminal: str
    initial_polar_side: str
    radial_turnings: int
    polar_turnings: int
    equatorial_crossings_before_terminal: int
    azimuth_winding: int
    source_radius_m: mp.mpf
    source_azimuth_rad: mp.mpf
    frequency_ratio: mp.mpf
    travel_time_m: mp.mpf
    emitted_bolometric_intensity: mp.mpf
    observed_bolometric_intensity: mp.mpf
    energy: mp.mpf
    impact_parameter: mp.mpf
    carter_parameter: mp.mpf
    radial_turning_derivative: mp.mpf
    polar_turning_derivative: mp.mpf
    initial_null_residual: mp.mpf
    mino_constraint_residual: mp.mpf
    chart_primitive_residual: mp.mpf

    def __post_init__(self) -> None:
        _validate_surface_witness(self)


@dataclass(frozen=True)
class PrecisionCertificate:
    """Precision-doubling evidence for the canonical witness."""

    low_precision_digits: int
    high_precision_digits: int
    required_stable_digits: int
    maximum_normalized_delta: mp.mpf
    witness: SurfaceWitness


@dataclass(frozen=True)
class _InitialRay:
    momentum_covariant: tuple[mp.mpf, ...]
    observer_frequency: mp.mpf
    initial_null_residual: mp.mpf


@dataclass(frozen=True)
class _SeparatedState:
    energy: mp.mpf
    impact: mp.mpf
    carter: mp.mpf
    radial_velocity: mp.mpf
    polar_velocity: mp.mpf
    constraint_residual: mp.mpf


def _validate_precision_digits(precision_digits: object) -> None:
    if not isinstance(precision_digits, int) or isinstance(precision_digits, bool):
        raise UnsupportedWitness("witness precision must be an integer")
    if precision_digits < MINIMUM_WITNESS_DIGITS:
        raise UnsupportedWitness(
            f"witness precision must be at least {MINIMUM_WITNESS_DIGITS} digits"
        )


def _validate_surface_witness(witness: SurfaceWitness) -> None:
    _validate_precision_digits(witness.precision_digits)
    continuous_fields = (
        witness.source_radius_m,
        witness.source_azimuth_rad,
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
        isinstance(value, mp.mpf) and mp.isfinite(value)
        for value in continuous_fields
    ):
        raise UnsupportedWitness("witness contains a non-real or non-finite value")
    if not (
        SURFACE_INNER_RADIUS_M
        <= witness.source_radius_m
        <= SURFACE_OUTER_RADIUS_M
    ):
        raise UnsupportedWitness("crossing lies outside the canonical surface")
    positive_fields = (
        witness.frequency_ratio,
        witness.travel_time_m,
        witness.emitted_bolometric_intensity,
        witness.observed_bolometric_intensity,
        witness.energy,
    )
    if any(value <= 0 for value in positive_fields):
        raise UnsupportedWitness("witness contains a non-positive physical value")
    if witness.radial_turning_derivative <= 0 or witness.polar_turning_derivative <= 0:
        raise UnsupportedWitness("separated turning root is not simple")

    residuals = (
        witness.initial_null_residual,
        witness.mino_constraint_residual,
        witness.chart_primitive_residual,
    )
    with mp.workdps(witness.precision_digits):
        residual_limit = mp.power(
            10,
            RESIDUAL_GUARD_DIGITS - witness.precision_digits,
        )
        if any(residual < 0 or residual >= residual_limit for residual in residuals):
            certified_digits = witness.precision_digits - RESIDUAL_GUARD_DIGITS
            raise UnsupportedWitness(
                "equation residual does not retain the required "
                f"{certified_digits} decimal digits"
            )


def _vector_add(left: Sequence[mp.mpf], right: Sequence[mp.mpf]) -> tuple[mp.mpf, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _vector_scale(vector: Sequence[mp.mpf], scalar: mp.mpf) -> tuple[mp.mpf, ...]:
    return tuple(scalar * component for component in vector)


def _metric_dot(
    metric: Sequence[Sequence[mp.mpf]],
    left: Sequence[mp.mpf],
    right: Sequence[mp.mpf],
) -> mp.mpf:
    return mp.fsum(
        metric[row][column] * left[row] * right[column]
        for row in range(4)
        for column in range(4)
    )


def _lower(
    metric: Sequence[Sequence[mp.mpf]], vector: Sequence[mp.mpf]
) -> tuple[mp.mpf, ...]:
    return tuple(
        mp.fsum(metric[row][column] * vector[column] for column in range(4))
        for row in range(4)
    )


def _project_and_normalize(
    metric: Sequence[Sequence[mp.mpf]],
    four_velocity: Sequence[mp.mpf],
    seed: Sequence[mp.mpf],
    orthogonal_to: Iterable[Sequence[mp.mpf]],
) -> tuple[mp.mpf, ...]:
    projected = _vector_add(
        seed,
        _vector_scale(four_velocity, _metric_dot(metric, four_velocity, seed)),
    )
    for basis in orthogonal_to:
        projected = _vector_add(
            projected,
            _vector_scale(basis, -_metric_dot(metric, projected, basis)),
        )
    norm_squared = _metric_dot(metric, projected, projected)
    if norm_squared <= 0:
        raise UnsupportedWitness("canonical observer frame seed is degenerate")
    return _vector_scale(projected, 1 / mp.sqrt(norm_squared))


def _orientation_determinant(columns: Sequence[Sequence[mp.mpf]]) -> mp.mpf:
    matrix = mp.matrix(
        [[columns[column][row] for column in range(4)] for row in range(4)]
    )
    return mp.det(matrix)


def _canonical_geometry() -> tuple[
    mp.mpf,
    mp.mpf,
    mp.mpf,
    mp.mpf,
    tuple[mp.mpf, mp.mpf, mp.mpf],
    tuple[tuple[mp.mpf, ...], ...],
]:
    mass = mp.mpf(1)
    spin = mp.mpf(4) / 5
    radius = mp.mpf(30)
    theta = mp.pi / 3
    sin_theta = mp.sin(theta)
    cos_theta = mp.cos(theta)
    x = radius * sin_theta
    y = spin * sin_theta
    z = radius * cos_theta
    sigma = radius**2 + spin**2 * cos_theta**2
    scalar_f = 2 * mass * radius / sigma
    principal = (
        mp.mpf(1),
        (radius * x + spin * y) / (radius**2 + spin**2),
        (radius * y - spin * x) / (radius**2 + spin**2),
        z / radius,
    )
    minkowski = (-1, 1, 1, 1)
    metric = tuple(
        tuple(
            (mp.mpf(minkowski[row]) if row == column else mp.mpf(0))
            + scalar_f * principal[row] * principal[column]
            for column in range(4)
        )
        for row in range(4)
    )
    return mass, spin, radius, theta, (x, y, z), metric


def _canonical_initial_ray(pixel_x: int, pixel_y: int) -> _InitialRay:
    _, _, _, _, (x, y, z), metric = _canonical_geometry()
    g_tt = metric[0][0]
    four_velocity = (1 / mp.sqrt(-g_tt), mp.mpf(0), mp.mpf(0), mp.mpf(0))
    sight = _project_and_normalize(
        metric,
        four_velocity,
        (mp.mpf(0), -x, -y, -z),
        (),
    )
    arrival = _vector_scale(sight, -1)
    up = _project_and_normalize(
        metric,
        four_velocity,
        (mp.mpf(0), mp.mpf(0), mp.mpf(0), mp.mpf(1)),
        (sight,),
    )
    right_candidates = []
    for axis in (
        (mp.mpf(0), mp.mpf(1), mp.mpf(0), mp.mpf(0)),
        (mp.mpf(0), mp.mpf(0), mp.mpf(1), mp.mpf(0)),
        (mp.mpf(0), mp.mpf(0), mp.mpf(0), mp.mpf(1)),
    ):
        try:
            right_candidates.append(
                _project_and_normalize(metric, four_velocity, axis, (sight, up))
            )
        except UnsupportedWitness:
            continue
    if not right_candidates:
        raise UnsupportedWitness("canonical observer frame has no image-right axis")
    right = max(
        right_candidates,
        key=lambda candidate: max(abs(component) for component in candidate[1:]),
    )
    if _orientation_determinant((four_velocity, right, up, arrival)) < 0:
        right = _vector_scale(right, -1)

    width = mp.mpf(1280)
    height = mp.mpf(720)
    half = mp.mpf(1) / 2
    normalized_x = 2 * (mp.mpf(pixel_x) + half) / width - 1
    normalized_y = 1 - 2 * (mp.mpf(pixel_y) + half) / height
    tangent_half_fov = mp.tan(mp.pi / 8)
    sight_x = width / height * tangent_half_fov * normalized_x
    sight_y = tangent_half_fov * normalized_y
    normalization = 1 / mp.sqrt(1 + sight_x**2 + sight_y**2)
    sight_direction = _vector_scale(
        _vector_add(
            _vector_add(_vector_scale(right, sight_x), _vector_scale(up, sight_y)),
            _vector_scale(arrival, -1),
        ),
        normalization,
    )
    photon_arrival = _vector_scale(sight_direction, -1)
    momentum_contravariant = _vector_add(four_velocity, photon_arrival)
    momentum_covariant = _lower(metric, momentum_contravariant)
    observer_frequency = -mp.fsum(
        covector * vector
        for covector, vector in zip(momentum_covariant, four_velocity, strict=True)
    )
    null_value = _metric_dot(metric, momentum_contravariant, momentum_contravariant)
    null_term_norm = mp.fsum(
        abs(metric[row][column] * momentum_contravariant[row] * momentum_contravariant[column])
        for row in range(4)
        for column in range(4)
    )
    initial_null_residual = abs(null_value) / max(mp.mpf(1), null_term_norm)
    return _InitialRay(
        momentum_covariant=momentum_covariant,
        observer_frequency=observer_frequency,
        initial_null_residual=initial_null_residual,
    )


def _separated_initial_state(initial_ray: _InitialRay) -> _SeparatedState:
    mass, spin, radius, theta, (x, y, _), _ = _canonical_geometry()
    sin_theta = mp.sin(theta)
    cos_theta = mp.cos(theta)
    p_t, p_x, p_y, p_z = initial_ray.momentum_covariant
    p_r_ks = sin_theta * p_x + cos_theta * p_z
    p_theta = mp.cos(theta) / sin_theta * (x * p_x + y * p_y) - radius * sin_theta * p_z
    p_phi = x * p_y - y * p_x
    delta = radius**2 - 2 * mass * radius + spin**2
    p_r_bl = p_r_ks + 2 * mass * radius / delta * p_t + spin / delta * p_phi
    energy = -p_t
    impact = p_phi / energy
    mu = cos_theta
    carter = (p_theta / energy) ** 2 + mu**2 * (
        impact**2 / (1 - mu**2) - spin**2
    )
    radial_velocity = delta * p_r_bl / energy
    polar_velocity = -sin_theta * p_theta / energy

    def radial_potential(value: mp.mpf) -> mp.mpf:
        radial_factor = value**2 + spin**2 - spin * impact
        separation = (impact - spin) ** 2 + carter
        return radial_factor**2 - (value**2 - 2 * mass * value + spin**2) * separation

    def polar_potential(value: mp.mpf) -> mp.mpf:
        return (
            carter
            + (spin**2 - carter - impact**2) * value**2
            - spin**2 * value**4
        )

    radial_residual = abs(radial_velocity**2 - radial_potential(radius)) / max(
        mp.mpf(1),
        abs(radial_velocity**2),
        abs(radial_potential(radius)),
    )
    polar_residual = abs(polar_velocity**2 - polar_potential(mu)) / max(
        mp.mpf(1),
        abs(polar_velocity**2),
        abs(polar_potential(mu)),
    )
    return _SeparatedState(
        energy=energy,
        impact=impact,
        carter=carter,
        radial_velocity=radial_velocity,
        polar_velocity=polar_velocity,
        constraint_residual=max(radial_residual, polar_residual),
    )


def _real_polynomial_roots(coefficients: Sequence[mp.mpf]) -> tuple[mp.mpf, ...]:
    roots = mp.polyroots(
        coefficients,
        maxsteps=400,
        cleanup=False,
        extraprec=80,
    )
    imaginary_tolerance = mp.power(10, -(mp.mp.dps - 30))
    return tuple(
        mp.re(root)
        for root in roots
        if abs(mp.im(root)) <= imaginary_tolerance
    )


def _chart_primitives(
    lower: mp.mpf, upper: mp.mpf, spin: mp.mpf
) -> tuple[mp.mpf, mp.mpf]:
    horizon_gap = mp.sqrt(1 - spin**2)
    outer = 1 + horizon_gap
    inner = 1 - horizon_gap
    denominator = outer - inner

    def time_primitive(radius: mp.mpf) -> mp.mpf:
        return (
            2 * outer / denominator * mp.log(radius - outer)
            - 2 * inner / denominator * mp.log(radius - inner)
        )

    def azimuth_primitive(radius: mp.mpf) -> mp.mpf:
        return spin / denominator * (
            mp.log(radius - outer) - mp.log(radius - inner)
        )

    return (
        time_primitive(upper) - time_primitive(lower),
        azimuth_primitive(upper) - azimuth_primitive(lower),
    )


def _wrap_angle(angle: mp.mpf) -> mp.mpf:
    return angle - 2 * mp.pi * mp.floor((angle + mp.pi) / (2 * mp.pi))


def _compute_canonical_surface_witness(
    pixel_x: int, pixel_y: int, precision_digits: int
) -> SurfaceWitness:
    mass, spin, observer_radius, theta, _, _ = _canonical_geometry()
    initial_ray = _canonical_initial_ray(pixel_x, pixel_y)
    separated = _separated_initial_state(initial_ray)
    if separated.radial_velocity <= 0 or separated.polar_velocity >= 0:
        raise UnsupportedWitness(
            "first slice requires a future-outgoing ray after one northern polar turning"
        )
    impact = separated.impact
    carter = separated.carter
    initial_mu = mp.cos(theta)
    separation = (impact - spin) ** 2 + carter

    def delta(radius: mp.mpf) -> mp.mpf:
        return radius**2 - 2 * mass * radius + spin**2

    def radial_factor(radius: mp.mpf) -> mp.mpf:
        return radius**2 + spin**2 - spin * impact

    def radial_potential(radius: mp.mpf) -> mp.mpf:
        return radial_factor(radius) ** 2 - delta(radius) * separation

    polar_coefficient = spin**2 - carter - impact**2
    polar_discriminant = mp.sqrt(
        polar_coefficient**2 + 4 * spin**2 * carter
    )
    positive_mu_squared = (
        polar_coefficient + polar_discriminant
    ) / (2 * spin**2)
    negative_mu_squared = (
        polar_coefficient - polar_discriminant
    ) / (2 * spin**2)
    polar_turning = mp.sqrt(positive_mu_squared)
    if not initial_mu < polar_turning < 1 or negative_mu_squared >= 0:
        raise UnsupportedWitness("polar potential does not have the required simple turning")

    def polar_to_turn(
        numerator: Callable[[mp.mpf], mp.mpf], lower_mu: mp.mpf
    ) -> mp.mpf:
        # U = a^2 (mu_turn^2 - mu^2) (mu^2 - mu_negative^2).
        # mu = mu_turn sin(angle) removes the simple-root endpoint singularity.
        lower_angle = mp.asin(lower_mu / polar_turning)
        return mp.quad(
            lambda angle: numerator(polar_turning * mp.sin(angle))
            / (
                spin
                * mp.sqrt(
                    positive_mu_squared * mp.sin(angle) ** 2
                    - negative_mu_squared
                )
            ),
            [lower_angle, mp.pi / 2],
        )

    polar_mino_duration = polar_to_turn(lambda _: mp.mpf(1), mp.mpf(0))
    polar_mino_duration += polar_to_turn(lambda _: mp.mpf(1), initial_mu)

    outer_radius = mp.mpf(SURFACE_OUTER_RADIUS_M)
    radial_constant = spin**2 - spin * impact
    radial_roots = _real_polynomial_roots(
        (
            mp.mpf(1),
            mp.mpf(0),
            2 * radial_constant - separation,
            2 * separation,
            radial_constant**2 - spin**2 * separation,
        )
    )
    horizon_radius = 1 + mp.sqrt(1 - spin**2)
    turning_candidates = tuple(
        root
        for root in radial_roots
        if horizon_radius < root < observer_radius
    )
    if not turning_candidates:
        raise UnsupportedWitness("radial root topology has no exterior turning point")
    radial_turning = max(turning_candidates)
    if radial_turning >= outer_radius:
        raise UnsupportedWitness("radial turning precedes the canonical source edge")
    radial_turning = mp.findroot(
        radial_potential,
        radial_turning,
        tol=mp.power(10, -(precision_digits - 20)),
        verify=True,
    )
    radial_turning_derivative = (
        4 * radial_turning * radial_factor(radial_turning)
        - 2 * (radial_turning - mass) * separation
    )
    if radial_turning_derivative <= 0:
        raise UnsupportedWitness("radial turning root is not simple and outward-facing")
    radial_quadratic_coefficient = 2 * radial_constant - separation
    radial_linear_coefficient = 2 * separation

    def radial_quotient(radius: mp.mpf) -> mp.mpf:
        # Synthetic division evaluates R / (r - r_turn) without subtracting
        # nearly equal arbitrary-precision values at the simple turning root.
        quadratic = radial_quadratic_coefficient + radial_turning**2
        constant = radial_linear_coefficient + radial_turning * quadratic
        return (
            radius**3
            + radial_turning * radius**2
            + quadratic * radius
            + constant
        )

    def radial_from_turn(
        numerator: Callable[[mp.mpf], mp.mpf], upper_radius: mp.mpf
    ) -> mp.mpf:
        if upper_radius < radial_turning:
            raise UnsupportedWitness("radial quadrature crossed its turning root")
        upper_coordinate = mp.sqrt(upper_radius - radial_turning)

        def regularized(coordinate: mp.mpf) -> mp.mpf:
            # r = r_turn + s^2 cancels the remaining square-root endpoint.
            radius = radial_turning + coordinate**2
            return 2 * numerator(radius) / mp.sqrt(radial_quotient(radius))

        return mp.quad(regularized, [mp.mpf(0), upper_coordinate])

    observer_radial_duration = radial_from_turn(
        lambda _: mp.mpf(1), observer_radius
    )

    def crossing_equation(source_radius: mp.mpf) -> mp.mpf:
        return (
            observer_radial_duration
            + radial_from_turn(lambda _: mp.mpf(1), source_radius)
            - polar_mino_duration
        )

    inner_radius = radial_turning
    inner_value = crossing_equation(inner_radius)
    outer_value = crossing_equation(outer_radius)
    if not (inner_value < 0 and outer_value > 0):
        raise UnsupportedWitness(
            "first equatorial crossing is not bracketed by the canonical surface: "
            f"inner={mp.nstr(inner_value, 12)}, "
            f"outer={mp.nstr(outer_value, 12)}, "
            f"polar={mp.nstr(polar_mino_duration, 12)}"
        )
    tolerance = mp.power(10, -(precision_digits - 25))
    source_radius = mp.findroot(
        crossing_equation,
        (inner_radius, outer_radius),
        solver="anderson",
        tol=tolerance,
        verify=True,
    )

    def radial_time_numerator(radius: mp.mpf) -> mp.mpf:
        return (radius**2 + spin**2) * radial_factor(radius) / delta(radius)

    radial_time_bl = radial_from_turn(radial_time_numerator, source_radius)
    radial_time_bl += radial_from_turn(radial_time_numerator, observer_radius)
    polar_time_bl = polar_to_turn(
        lambda mu: spin * (impact - spin) + spin**2 * mu**2,
        mp.mpf(0),
    )
    polar_time_bl += polar_to_turn(
        lambda mu: spin * (impact - spin) + spin**2 * mu**2,
        initial_mu,
    )
    def radial_azimuth_numerator(radius: mp.mpf) -> mp.mpf:
        return spin * radial_factor(radius) / delta(radius)

    radial_azimuth_bl = radial_from_turn(
        radial_azimuth_numerator, source_radius
    )
    radial_azimuth_bl += radial_from_turn(
        radial_azimuth_numerator, observer_radius
    )
    polar_azimuth_bl = polar_to_turn(
        lambda mu: impact / (1 - mu**2) - spin,
        mp.mpf(0),
    )
    polar_azimuth_bl += polar_to_turn(
        lambda mu: impact / (1 - mu**2) - spin,
        initial_mu,
    )
    chart_time_quad = mp.quad(
        lambda radius: 2 * mass * radius / delta(radius),
        [source_radius, observer_radius],
    )
    chart_azimuth_quad = mp.quad(
        lambda radius: spin / delta(radius),
        [source_radius, observer_radius],
    )
    chart_time_exact, chart_azimuth_exact = _chart_primitives(
        source_radius, observer_radius, spin
    )
    chart_primitive_residual = max(
        abs(chart_time_quad - chart_time_exact)
        / max(mp.mpf(1), abs(chart_time_exact)),
        abs(chart_azimuth_quad - chart_azimuth_exact)
        / max(mp.mpf(1), abs(chart_azimuth_exact)),
    )
    travel_time = radial_time_bl + polar_time_bl + chart_time_exact
    observer_minus_source_azimuth = (
        radial_azimuth_bl + polar_azimuth_bl + chart_azimuth_exact
    )
    source_azimuth_unwrapped = -observer_minus_source_azimuth
    source_azimuth = _wrap_angle(source_azimuth_unwrapped)
    observer_cartesian_azimuth = mp.atan2(spin, observer_radius)
    source_cartesian_azimuth = source_azimuth_unwrapped + mp.atan2(
        spin, source_radius
    )
    observer_azimuth_cycle = mp.floor(
        (observer_cartesian_azimuth + mp.pi) / (2 * mp.pi)
    )
    source_azimuth_cycle = mp.floor(
        (source_cartesian_azimuth + mp.pi) / (2 * mp.pi)
    )
    azimuth_winding = int(source_azimuth_cycle - observer_azimuth_cycle)

    angular_velocity = mp.sqrt(source_radius) / (
        source_radius**2 + spin * mp.sqrt(source_radius)
    )
    g_tt = -1 + 2 / source_radius
    g_t_phi = -2 * spin / source_radius
    source_delta = delta(source_radius)
    g_phi_phi = (
        (source_radius**2 + spin**2) ** 2 - spin**2 * source_delta
    ) / source_radius**2
    emitter_time_component = 1 / mp.sqrt(
        -(
            g_tt
            + 2 * angular_velocity * g_t_phi
            + angular_velocity**2 * g_phi_phi
        )
    )
    emitter_frequency = (
        emitter_time_component
        * separated.energy
        * (1 - angular_velocity * impact)
    )
    frequency_ratio = initial_ray.observer_frequency / emitter_frequency
    emitted_intensity = (source_radius / SURFACE_INNER_RADIUS_M) ** -3
    observed_intensity = emitted_intensity * frequency_ratio**4

    radial_conditioning = radial_turning_derivative
    polar_conditioning = abs(
        2 * polar_coefficient * polar_turning
        - 4 * spin**2 * polar_turning**3
    )
    return SurfaceWitness(
        precision_digits=precision_digits,
        terminal="equatorial-surface",
        initial_polar_side="positive",
        radial_turnings=1,
        polar_turnings=1,
        equatorial_crossings_before_terminal=0,
        azimuth_winding=azimuth_winding,
        source_radius_m=source_radius,
        source_azimuth_rad=source_azimuth,
        frequency_ratio=frequency_ratio,
        travel_time_m=travel_time,
        emitted_bolometric_intensity=emitted_intensity,
        observed_bolometric_intensity=observed_intensity,
        energy=separated.energy,
        impact_parameter=impact,
        carter_parameter=carter,
        radial_turning_derivative=radial_conditioning,
        polar_turning_derivative=polar_conditioning,
        initial_null_residual=initial_ray.initial_null_residual,
        mino_constraint_residual=separated.constraint_residual,
        chart_primitive_residual=chart_primitive_residual,
    )


def compute_canonical_surface_witness(
    *, pixel_x: int, pixel_y: int, precision_digits: int
) -> SurfaceWitness:
    """Compute the named ordinary-region surface witness.

    The external seam validates the fixed viewport and a precision large enough
    to distinguish this research tool from the binary64 reference.
    """

    if not 0 <= pixel_x < 1280 or not 0 <= pixel_y < 720:
        raise UnsupportedWitness("sample lies outside the canonical 1280x720 viewport")
    if (pixel_x, pixel_y) != (640, 16):
        raise UnsupportedWitness(
            "this first research slice certifies only canonical sample (640, 16)"
        )
    _validate_precision_digits(precision_digits)
    with mp.workdps(precision_digits):
        return _compute_canonical_surface_witness(
            pixel_x,
            pixel_y,
            precision_digits,
        )


def build_precision_certificate() -> PrecisionCertificate:
    """Recompute the canonical case at 120 and 180 digits and certify stability."""

    low = compute_canonical_surface_witness(
        pixel_x=640,
        pixel_y=16,
        precision_digits=LOW_PRECISION_DIGITS,
    )
    high = compute_canonical_surface_witness(
        pixel_x=640,
        pixel_y=16,
        precision_digits=HIGH_PRECISION_DIGITS,
    )
    fields = (
        "source_radius_m",
        "source_azimuth_rad",
        "frequency_ratio",
        "travel_time_m",
        "emitted_bolometric_intensity",
        "observed_bolometric_intensity",
        "energy",
        "impact_parameter",
        "carter_parameter",
        "radial_turning_derivative",
        "polar_turning_derivative",
    )
    discrete_fields = (
        "terminal",
        "initial_polar_side",
        "radial_turnings",
        "polar_turnings",
        "equatorial_crossings_before_terminal",
        "azimuth_winding",
    )
    for field in discrete_fields:
        if getattr(low, field) != getattr(high, field):
            raise AssertionError(f"precision doubling changed discrete field {field}")
    with mp.workdps(HIGH_PRECISION_DIGITS):
        normalized_deltas = tuple(
            abs(getattr(low, field) - getattr(high, field))
            / max(mp.mpf(1), abs(getattr(high, field)))
            for field in fields
        )
        if not all(mp.isfinite(delta) for delta in normalized_deltas):
            raise AssertionError("precision doubling produced a non-finite delta")
        maximum_delta = max(normalized_deltas)
        required = mp.power(10, -REQUIRED_STABLE_DIGITS)
        if maximum_delta >= required:
            raise AssertionError(
                f"precision doubling retained only {-mp.log10(maximum_delta)} digits"
            )
    return PrecisionCertificate(
        low_precision_digits=LOW_PRECISION_DIGITS,
        high_precision_digits=HIGH_PRECISION_DIGITS,
        required_stable_digits=REQUIRED_STABLE_DIGITS,
        maximum_normalized_delta=maximum_delta,
        witness=high,
    )


def _scientific(value: mp.mpf, digits: int = 110) -> str:
    return mp.nstr(value, digits, strip_zeros=False)


def main() -> None:
    certificate = build_precision_certificate()
    witness = certificate.witness
    print(f"python={platform.python_version()}")
    print(f"mpmath={mp.__version__}")
    print(
        "precision="
        f"{certificate.low_precision_digits},{certificate.high_precision_digits} "
        f"stable_digits>={certificate.required_stable_digits} "
        f"maximum_normalized_delta={_scientific(certificate.maximum_normalized_delta, 12)}"
    )
    print("case=kerr-exterior-observation-v1:640:16:0.5:0.5")
    print(f"terminal={witness.terminal}")
    print(
        "branch="
        f"initial_polar_side:{witness.initial_polar_side},"
        f"radial_turnings:{witness.radial_turnings},"
        f"polar_turnings:{witness.polar_turnings},"
        "equatorial_crossings_before_terminal:"
        f"{witness.equatorial_crossings_before_terminal},"
        f"azimuth_winding:{witness.azimuth_winding}"
    )
    print(f"source_radius_m={_scientific(witness.source_radius_m)}")
    print(f"source_azimuth_rad={_scientific(witness.source_azimuth_rad)}")
    print(f"frequency_ratio={_scientific(witness.frequency_ratio)}")
    print(f"travel_time_m={_scientific(witness.travel_time_m)}")
    print(
        "emitted_bolometric_intensity="
        f"{_scientific(witness.emitted_bolometric_intensity)}"
    )
    print(
        "observed_bolometric_intensity="
        f"{_scientific(witness.observed_bolometric_intensity)}"
    )
    print(f"energy={_scientific(witness.energy)}")
    print(f"impact_parameter={_scientific(witness.impact_parameter)}")
    print(f"carter_parameter={_scientific(witness.carter_parameter)}")
    print(
        "radial_turning_derivative="
        f"{_scientific(witness.radial_turning_derivative, 20)}"
    )
    print(
        "polar_turning_derivative="
        f"{_scientific(witness.polar_turning_derivative, 20)}"
    )
    print(f"initial_null_residual={_scientific(witness.initial_null_residual, 12)}")
    print(
        "mino_constraint_residual="
        f"{_scientific(witness.mino_constraint_residual, 12)}"
    )
    print(
        "chart_primitive_residual="
        f"{_scientific(witness.chart_primitive_residual, 12)}"
    )
    print("RESULT=PASS")


if __name__ == "__main__":
    main()

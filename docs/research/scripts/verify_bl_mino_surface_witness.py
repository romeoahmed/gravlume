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
  https://mpmath.org/doc/1.3.0/calculus/integration.html
* Verified root finding and polynomial-root conditioning:
  https://mpmath.org/doc/1.3.0/calculus/optimization.html
  https://mpmath.org/doc/1.3.0/calculus/polynomials.html
* Precision contexts and finite-value classification:
  https://mpmath.org/doc/1.3.0/general.html
* Python NaN comparison semantics:
  https://docs.python.org/3/reference/expressions.html#value-comparisons
* Dataclass post-init and replacement behavior:
  https://docs.python.org/3.10/library/dataclasses.html#post-init-processing
* Runtime relationship between bool and int:
  https://docs.python.org/3.10/library/stdtypes.html#boolean-type-bool
"""

from __future__ import annotations

import platform
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import mpmath as mp

LOW_PRECISION_DIGITS = 120
HIGH_PRECISION_DIGITS = 180
REQUIRED_STABLE_DIGITS = 80
MINIMUM_WITNESS_DIGITS = 70
RESIDUAL_GUARD_DIGITS = 15
SURFACE_INNER_RADIUS_M = 6
SURFACE_OUTER_RADIUS_M = 20
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720
CANONICAL_PIXEL = (640, 16)
_CANONICAL_TERMINAL = "equatorial-surface"
_CANONICAL_INITIAL_POLAR_SIDE = "positive"
_CANONICAL_RADIAL_TURNINGS = 1
_CANONICAL_POLAR_TURNINGS = 1
_CANONICAL_EQUATORIAL_CROSSINGS_BEFORE_TERMINAL = 0
_CANONICAL_AZIMUTH_WINDING = 0


class UnsupportedWitnessError(ValueError):
    """The requested case lies outside this research slice's certified domain."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfaceWitness:
    """Certified canonical identity, observables, and margins for one surface ray."""

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


@dataclass(frozen=True, slots=True, kw_only=True)
class PrecisionCertificate:
    """Precision-doubling evidence for the canonical witness."""

    low_precision_digits: int
    high_precision_digits: int
    required_stable_digits: int
    maximum_normalized_delta: mp.mpf
    witness: SurfaceWitness


@dataclass(frozen=True, slots=True, kw_only=True)
class _ObservationGeometry:
    mass: mp.mpf
    spin: mp.mpf
    radius: mp.mpf
    theta: mp.mpf
    position: tuple[mp.mpf, mp.mpf, mp.mpf]
    metric: tuple[tuple[mp.mpf, ...], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _InitialRay:
    momentum_covariant: tuple[mp.mpf, ...]
    observer_frequency: mp.mpf
    initial_null_residual: mp.mpf


@dataclass(frozen=True, slots=True, kw_only=True)
class _SeparatedState:
    energy: mp.mpf
    impact: mp.mpf
    carter: mp.mpf
    radial_velocity: mp.mpf
    polar_velocity: mp.mpf
    constraint_residual: mp.mpf


@dataclass(frozen=True, slots=True, kw_only=True)
class _PolarMotion:
    spin: mp.mpf
    turning: mp.mpf
    turning_squared: mp.mpf
    negative_turning_squared: mp.mpf
    quadratic_coefficient: mp.mpf

    def integrate_to_turn(
        self,
        numerator: Callable[[mp.mpf], mp.mpf],
        lower_mu: mp.mpf,
    ) -> mp.mpf:
        """Integrate through the regularized simple polar turning endpoint."""

        lower_angle = mp.asin(lower_mu / self.turning)
        return mp.quad(
            lambda angle: (
                numerator(self.turning * mp.sin(angle))
                / (
                    self.spin
                    * mp.sqrt(
                        self.turning_squared * mp.sin(angle) ** 2
                        - self.negative_turning_squared
                    )
                )
            ),
            [lower_angle, mp.pi / 2],
        )

    @property
    def turning_derivative(self) -> mp.mpf:
        return abs(
            2 * self.quadratic_coefficient * self.turning
            - 4 * self.spin**2 * self.turning**3
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _RadialMotion:
    mass: mp.mpf
    spin: mp.mpf
    impact: mp.mpf
    turning: mp.mpf
    turning_derivative: mp.mpf
    quadratic_coefficient: mp.mpf
    linear_coefficient: mp.mpf

    def delta(self, radius: mp.mpf) -> mp.mpf:
        return radius**2 - 2 * self.mass * radius + self.spin**2

    def factor(self, radius: mp.mpf) -> mp.mpf:
        return radius**2 + self.spin**2 - self.spin * self.impact

    def quotient(self, radius: mp.mpf) -> mp.mpf:
        """Evaluate R/(r-r_turn) by synthetic division near the simple root."""

        quadratic = self.quadratic_coefficient + self.turning**2
        constant = self.linear_coefficient + self.turning * quadratic
        return radius**3 + self.turning * radius**2 + quadratic * radius + constant

    def integrate_from_turn(
        self,
        numerator: Callable[[mp.mpf], mp.mpf],
        upper_radius: mp.mpf,
    ) -> mp.mpf:
        """Integrate through the regularized simple radial turning endpoint."""

        if upper_radius < self.turning:
            raise UnsupportedWitnessError("radial quadrature crossed its turning root")
        upper_coordinate = mp.sqrt(upper_radius - self.turning)

        def regularized(coordinate: mp.mpf) -> mp.mpf:
            radius = self.turning + coordinate**2
            return 2 * numerator(radius) / mp.sqrt(self.quotient(radius))

        return mp.quad(regularized, [mp.mpf(0), upper_coordinate])


@dataclass(frozen=True, slots=True, kw_only=True)
class _PathObservables:
    source_azimuth: mp.mpf
    travel_time: mp.mpf
    azimuth_winding: int
    chart_primitive_residual: mp.mpf


@dataclass(frozen=True, slots=True, kw_only=True)
class _TransferObservables:
    frequency_ratio: mp.mpf
    emitted_intensity: mp.mpf
    observed_intensity: mp.mpf


def _validate_precision_digits(precision_digits: object) -> None:
    if type(precision_digits) is not int:
        raise UnsupportedWitnessError("witness precision must be an integer")
    if precision_digits < MINIMUM_WITNESS_DIGITS:
        raise UnsupportedWitnessError(
            f"witness precision must be at least {MINIMUM_WITNESS_DIGITS} digits"
        )


def _validate_surface_witness(witness: SurfaceWitness) -> None:
    _validate_precision_digits(witness.precision_digits)
    discrete_identity = (
        (witness.terminal, _CANONICAL_TERMINAL),
        (witness.initial_polar_side, _CANONICAL_INITIAL_POLAR_SIDE),
        (witness.radial_turnings, _CANONICAL_RADIAL_TURNINGS),
        (witness.polar_turnings, _CANONICAL_POLAR_TURNINGS),
        (
            witness.equatorial_crossings_before_terminal,
            _CANONICAL_EQUATORIAL_CROSSINGS_BEFORE_TERMINAL,
        ),
        (witness.azimuth_winding, _CANONICAL_AZIMUTH_WINDING),
    )
    if any(
        type(actual) is not type(expected) or actual != expected
        for actual, expected in discrete_identity
    ):
        raise UnsupportedWitnessError(
            "witness does not match the canonical discrete path identity"
        )
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
        isinstance(value, mp.mpf) and mp.isfinite(value) for value in continuous_fields
    ):
        raise UnsupportedWitnessError("witness contains a non-real or non-finite value")
    if not (
        SURFACE_INNER_RADIUS_M <= witness.source_radius_m <= SURFACE_OUTER_RADIUS_M
    ):
        raise UnsupportedWitnessError("crossing lies outside the canonical surface")
    positive_fields = (
        witness.frequency_ratio,
        witness.travel_time_m,
        witness.emitted_bolometric_intensity,
        witness.observed_bolometric_intensity,
        witness.energy,
    )
    if any(value <= 0 for value in positive_fields):
        raise UnsupportedWitnessError("witness contains a non-positive physical value")
    if witness.radial_turning_derivative <= 0 or witness.polar_turning_derivative <= 0:
        raise UnsupportedWitnessError("separated turning root is not simple")

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
            raise UnsupportedWitnessError(
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
        raise UnsupportedWitnessError("canonical observer frame seed is degenerate")
    return _vector_scale(projected, 1 / mp.sqrt(norm_squared))


def _orientation_determinant(columns: Sequence[Sequence[mp.mpf]]) -> mp.mpf:
    matrix = mp.matrix(
        [[columns[column][row] for column in range(4)] for row in range(4)]
    )
    return mp.det(matrix)


def _canonical_geometry() -> _ObservationGeometry:
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
    return _ObservationGeometry(
        mass=mass,
        spin=spin,
        radius=radius,
        theta=theta,
        position=(x, y, z),
        metric=metric,
    )


def _try_image_right_axis(
    metric: Sequence[Sequence[mp.mpf]],
    four_velocity: Sequence[mp.mpf],
    sight: Sequence[mp.mpf],
    up: Sequence[mp.mpf],
    axis: Sequence[mp.mpf],
) -> tuple[mp.mpf, ...] | None:
    try:
        return _project_and_normalize(metric, four_velocity, axis, (sight, up))
    except UnsupportedWitnessError:
        return None


def _canonical_initial_ray(
    geometry: _ObservationGeometry,
    pixel_x: int,
    pixel_y: int,
) -> _InitialRay:
    x, y, z = geometry.position
    metric = geometry.metric
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
    right_candidates = tuple(
        candidate
        for candidate in (
            _try_image_right_axis(
                metric,
                four_velocity,
                sight,
                up,
                axis,
            )
            for axis in (
                (mp.mpf(0), mp.mpf(1), mp.mpf(0), mp.mpf(0)),
                (mp.mpf(0), mp.mpf(0), mp.mpf(1), mp.mpf(0)),
                (mp.mpf(0), mp.mpf(0), mp.mpf(0), mp.mpf(1)),
            )
        )
        if candidate is not None
    )
    if not right_candidates:
        raise UnsupportedWitnessError(
            "canonical observer frame has no image-right axis"
        )
    right = max(
        right_candidates,
        key=lambda candidate: max(abs(component) for component in candidate[1:]),
    )
    if _orientation_determinant((four_velocity, right, up, arrival)) < 0:
        right = _vector_scale(right, -1)

    width = mp.mpf(VIEWPORT_WIDTH)
    height = mp.mpf(VIEWPORT_HEIGHT)
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
        abs(
            metric[row][column]
            * momentum_contravariant[row]
            * momentum_contravariant[column]
        )
        for row in range(4)
        for column in range(4)
    )
    initial_null_residual = abs(null_value) / max(mp.mpf(1), null_term_norm)
    return _InitialRay(
        momentum_covariant=momentum_covariant,
        observer_frequency=observer_frequency,
        initial_null_residual=initial_null_residual,
    )


def _separated_initial_state(
    geometry: _ObservationGeometry,
    initial_ray: _InitialRay,
) -> _SeparatedState:
    mass = geometry.mass
    spin = geometry.spin
    radius = geometry.radius
    theta = geometry.theta
    x, y, _ = geometry.position
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
    carter = (p_theta / energy) ** 2 + mu**2 * (impact**2 / (1 - mu**2) - spin**2)
    radial_velocity = delta * p_r_bl / energy
    polar_velocity = -sin_theta * p_theta / energy

    def radial_potential(value: mp.mpf) -> mp.mpf:
        radial_factor = value**2 + spin**2 - spin * impact
        separation = (impact - spin) ** 2 + carter
        return radial_factor**2 - (value**2 - 2 * mass * value + spin**2) * separation

    def polar_potential(value: mp.mpf) -> mp.mpf:
        return carter + (spin**2 - carter - impact**2) * value**2 - spin**2 * value**4

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
        mp.re(root) for root in roots if abs(mp.im(root)) <= imaginary_tolerance
    )


def _chart_primitives(
    lower: mp.mpf,
    upper: mp.mpf,
    mass: mp.mpf,
    spin: mp.mpf,
) -> tuple[mp.mpf, mp.mpf]:
    horizon_gap = mp.sqrt(mass**2 - spin**2)
    outer = mass + horizon_gap
    inner = mass - horizon_gap
    denominator = outer - inner

    def time_primitive(radius: mp.mpf) -> mp.mpf:
        return 2 * mass * outer / denominator * mp.log(
            radius - outer
        ) - 2 * mass * inner / denominator * mp.log(radius - inner)

    def azimuth_primitive(radius: mp.mpf) -> mp.mpf:
        return spin / denominator * (mp.log(radius - outer) - mp.log(radius - inner))

    return (
        time_primitive(upper) - time_primitive(lower),
        azimuth_primitive(upper) - azimuth_primitive(lower),
    )


def _wrap_angle(angle: mp.mpf) -> mp.mpf:
    return angle - 2 * mp.pi * mp.floor((angle + mp.pi) / (2 * mp.pi))


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
        raise UnsupportedWitnessError(
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
        raise UnsupportedWitnessError(
            "radial root topology has no exterior turning point"
        )
    turning = max(turning_candidates)
    if turning >= SURFACE_OUTER_RADIUS_M:
        raise UnsupportedWitnessError(
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
        raise UnsupportedWitnessError(
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


def _solve_source_radius(
    radial: _RadialMotion,
    observer_radius: mp.mpf,
    polar_mino_duration: mp.mpf,
    precision_digits: int,
) -> mp.mpf:
    observer_duration = radial.integrate_from_turn(
        _unit_integrand,
        observer_radius,
    )

    def crossing_equation(source_radius: mp.mpf) -> mp.mpf:
        return (
            observer_duration
            + radial.integrate_from_turn(_unit_integrand, source_radius)
            - polar_mino_duration
        )

    outer_radius = mp.mpf(SURFACE_OUTER_RADIUS_M)
    inner_value = crossing_equation(radial.turning)
    outer_value = crossing_equation(outer_radius)
    if not (inner_value < 0 and outer_value > 0):
        raise UnsupportedWitnessError(
            "first equatorial crossing is not bracketed by the canonical surface: "
            f"inner={mp.nstr(inner_value, 12)}, "
            f"outer={mp.nstr(outer_value, 12)}, "
            f"polar={mp.nstr(polar_mino_duration, 12)}"
        )
    return mp.findroot(
        crossing_equation,
        (radial.turning, outer_radius),
        solver="anderson",
        tol=mp.power(10, -(precision_digits - 25)),
        verify=True,
    )


def _integrate_path_observables(
    geometry: _ObservationGeometry,
    polar: _PolarMotion,
    radial: _RadialMotion,
    initial_mu: mp.mpf,
    source_radius: mp.mpf,
) -> _PathObservables:
    spin = geometry.spin
    impact = radial.impact

    def radial_time_numerator(radius: mp.mpf) -> mp.mpf:
        return (radius**2 + spin**2) * radial.factor(radius) / radial.delta(radius)

    radial_time = radial.integrate_from_turn(radial_time_numerator, source_radius)
    radial_time += radial.integrate_from_turn(radial_time_numerator, geometry.radius)

    def polar_time_numerator(mu: mp.mpf) -> mp.mpf:
        return spin * (impact - spin) + spin**2 * mu**2

    polar_time = polar.integrate_to_turn(polar_time_numerator, mp.mpf(0))
    polar_time += polar.integrate_to_turn(polar_time_numerator, initial_mu)

    def radial_azimuth_numerator(radius: mp.mpf) -> mp.mpf:
        return spin * radial.factor(radius) / radial.delta(radius)

    radial_azimuth = radial.integrate_from_turn(
        radial_azimuth_numerator,
        source_radius,
    )
    radial_azimuth += radial.integrate_from_turn(
        radial_azimuth_numerator,
        geometry.radius,
    )

    def polar_azimuth_numerator(mu: mp.mpf) -> mp.mpf:
        return impact / (1 - mu**2) - spin

    polar_azimuth = polar.integrate_to_turn(
        polar_azimuth_numerator,
        mp.mpf(0),
    )
    polar_azimuth += polar.integrate_to_turn(
        polar_azimuth_numerator,
        initial_mu,
    )

    chart_time_quad = mp.quad(
        lambda radius: 2 * geometry.mass * radius / radial.delta(radius),
        [source_radius, geometry.radius],
    )
    chart_azimuth_quad = mp.quad(
        lambda radius: spin / radial.delta(radius),
        [source_radius, geometry.radius],
    )
    chart_time, chart_azimuth = _chart_primitives(
        source_radius,
        geometry.radius,
        geometry.mass,
        spin,
    )
    chart_residual = max(
        abs(chart_time_quad - chart_time) / max(mp.mpf(1), abs(chart_time)),
        abs(chart_azimuth_quad - chart_azimuth) / max(mp.mpf(1), abs(chart_azimuth)),
    )
    source_azimuth_unwrapped = -(radial_azimuth + polar_azimuth + chart_azimuth)
    observer_cartesian_azimuth = mp.atan2(spin, geometry.radius)
    source_cartesian_azimuth = source_azimuth_unwrapped + mp.atan2(
        spin,
        source_radius,
    )
    observer_azimuth_cycle = mp.floor(
        (observer_cartesian_azimuth + mp.pi) / (2 * mp.pi)
    )
    source_azimuth_cycle = mp.floor((source_cartesian_azimuth + mp.pi) / (2 * mp.pi))
    return _PathObservables(
        source_azimuth=_wrap_angle(source_azimuth_unwrapped),
        travel_time=radial_time + polar_time + chart_time,
        azimuth_winding=int(source_azimuth_cycle - observer_azimuth_cycle),
        chart_primitive_residual=chart_residual,
    )


def _surface_transfer_observables(
    geometry: _ObservationGeometry,
    initial_ray: _InitialRay,
    separated: _SeparatedState,
    radial: _RadialMotion,
    source_radius: mp.mpf,
) -> _TransferObservables:
    spin = geometry.spin
    sqrt_mass = mp.sqrt(geometry.mass)
    angular_velocity = sqrt_mass / (source_radius ** mp.mpf("1.5") + spin * sqrt_mass)
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
    emitted_intensity = (source_radius / SURFACE_INNER_RADIUS_M) ** -3
    return _TransferObservables(
        frequency_ratio=frequency_ratio,
        emitted_intensity=emitted_intensity,
        observed_intensity=emitted_intensity * frequency_ratio**4,
    )


def _compute_canonical_surface_witness(
    pixel_x: int, pixel_y: int, precision_digits: int
) -> SurfaceWitness:
    geometry = _canonical_geometry()
    initial_ray = _canonical_initial_ray(geometry, pixel_x, pixel_y)
    separated = _separated_initial_state(geometry, initial_ray)
    if separated.radial_velocity <= 0 or separated.polar_velocity >= 0:
        raise UnsupportedWitnessError(
            "first slice requires a future-outgoing ray after one northern "
            "polar turning"
        )
    initial_mu = mp.cos(geometry.theta)
    polar = _build_polar_motion(
        geometry.spin,
        separated.impact,
        separated.carter,
        initial_mu,
    )
    polar_mino_duration = polar.integrate_to_turn(_unit_integrand, mp.mpf(0))
    polar_mino_duration += polar.integrate_to_turn(_unit_integrand, initial_mu)

    radial = _build_radial_motion(geometry, separated, precision_digits)
    source_radius = _solve_source_radius(
        radial,
        geometry.radius,
        polar_mino_duration,
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
    )
    return SurfaceWitness(
        precision_digits=precision_digits,
        terminal=_CANONICAL_TERMINAL,
        initial_polar_side=_CANONICAL_INITIAL_POLAR_SIDE,
        radial_turnings=_CANONICAL_RADIAL_TURNINGS,
        polar_turnings=_CANONICAL_POLAR_TURNINGS,
        equatorial_crossings_before_terminal=(
            _CANONICAL_EQUATORIAL_CROSSINGS_BEFORE_TERMINAL
        ),
        azimuth_winding=path.azimuth_winding,
        source_radius_m=source_radius,
        source_azimuth_rad=path.source_azimuth,
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


def compute_canonical_surface_witness(
    *, pixel_x: int, pixel_y: int, precision_digits: int
) -> SurfaceWitness:
    """Compute the named ordinary-region surface witness.

    The external seam requires exact integer pixel coordinates, validates the
    fixed viewport, and requires enough precision to exceed binary64.
    """

    if type(pixel_x) is not int or type(pixel_y) is not int:
        raise UnsupportedWitnessError("sample coordinates must be integers")
    if not 0 <= pixel_x < VIEWPORT_WIDTH or not 0 <= pixel_y < VIEWPORT_HEIGHT:
        raise UnsupportedWitnessError(
            "sample lies outside the canonical "
            f"{VIEWPORT_WIDTH}x{VIEWPORT_HEIGHT} viewport"
        )
    if (pixel_x, pixel_y) != CANONICAL_PIXEL:
        raise UnsupportedWitnessError(
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

    pixel_x, pixel_y = CANONICAL_PIXEL
    low = compute_canonical_surface_witness(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        precision_digits=LOW_PRECISION_DIGITS,
    )
    high = compute_canonical_surface_witness(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
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
        "maximum_normalized_delta="
        f"{_scientific(certificate.maximum_normalized_delta, 12)}"
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
        f"polar_turning_derivative={_scientific(witness.polar_turning_derivative, 20)}"
    )
    print(f"initial_null_residual={_scientific(witness.initial_null_residual, 12)}")
    print(
        f"mino_constraint_residual={_scientific(witness.mino_constraint_residual, 12)}"
    )
    print(
        f"chart_primitive_residual={_scientific(witness.chart_primitive_residual, 12)}"
    )
    print("RESULT=PASS")


if __name__ == "__main__":
    main()

"""Build independent high-precision BL/Mino witnesses for named source-edge rays.

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
* Dataclass post-init behavior:
  https://docs.python.org/3.14/library/dataclasses.html#post-init-processing
* Runtime relationship between bool and int:
  https://docs.python.org/3.14/library/stdtypes.html#boolean-type-bool
"""

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
ESCAPE_RADIUS_M = 200
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720
CANONICAL_PIXEL = (640, 16)
SOURCE_EDGE_OUTSIDE_PIXEL = (640, 13)
SOURCE_EDGE_INSIDE_PIXEL = (640, 14)
_CANONICAL_TERMINAL = "equatorial-surface"
_CANONICAL_INITIAL_POLAR_SIDE = "positive"
_CANONICAL_RADIAL_TURNINGS = 1
_CANONICAL_POLAR_TURNINGS = 1
_CANONICAL_EQUATORIAL_CROSSINGS_BEFORE_TERMINAL = 0
_CANONICAL_AZIMUTH_WINDING = 0
_SOURCE_EDGE_ESCAPE_TERMINAL = "escape"
_SOURCE_EDGE_ESCAPE_EQUATORIAL_CROSSINGS = 1


class _UnsupportedWitnessError(ValueError):
    """The requested case lies outside this research slice's certified domain."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _SurfaceWitness:
    """Certified identity, observables, and margins for one named surface ray."""

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
class _EscapeWitness:
    """Certified identity, observables, and event margins for one escape ray."""

    precision_digits: int
    terminal: str
    initial_polar_side: str
    radial_turnings: int
    polar_turnings: int
    equatorial_crossings_before_terminal: int
    azimuth_winding: int
    first_equatorial_crossing_radius_m: mp.mpf
    escape_radius_m: mp.mpf
    escape_position_xyz_m: tuple[mp.mpf, mp.mpf, mp.mpf]
    escape_direction_xyz: tuple[mp.mpf, mp.mpf, mp.mpf]
    travel_time_m: mp.mpf
    escape_before_next_crossing_mino_margin: mp.mpf
    energy: mp.mpf
    impact_parameter: mp.mpf
    carter_parameter: mp.mpf
    radial_turning_derivative: mp.mpf
    polar_turning_derivative: mp.mpf
    initial_null_residual: mp.mpf
    mino_constraint_residual: mp.mpf
    chart_primitive_residual: mp.mpf

    def __post_init__(self) -> None:
        _validate_escape_witness(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class _SourceEdgePairWitness:
    """One fixed outside/inside pair that brackets the canonical outer source edge."""

    outside: _EscapeWitness
    inside: _SurfaceWitness

    def __post_init__(self) -> None:
        if self.outside.precision_digits != self.inside.precision_digits:
            raise _UnsupportedWitnessError("source-edge pair mixes working precisions")
        if not (
            self.outside.first_equatorial_crossing_radius_m
            > SURFACE_OUTER_RADIUS_M
            > self.inside.source_radius_m
        ):
            raise _UnsupportedWitnessError(
                "source-edge pair does not bracket the outer radial edge"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class _PrecisionCertificate[Witness]:
    """Precision-doubling evidence for one named witness."""

    low_precision_digits: int
    high_precision_digits: int
    required_stable_digits: int
    maximum_normalized_delta: mp.mpf
    witness: Witness


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

    def integrate_from_equator(
        self,
        numerator: Callable[[mp.mpf], mp.mpf],
        upper_mu: mp.mpf,
    ) -> mp.mpf:
        """Integrate from the equator without subtracting two endpoint integrals."""

        if not 0 <= upper_mu <= self.turning:
            raise _UnsupportedWitnessError(
                "polar quadrature crossed its simple turning root"
            )
        upper_angle = mp.asin(upper_mu / self.turning)
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
            [mp.mpf(0), upper_angle],
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

    def potential(self, radius: mp.mpf) -> mp.mpf:
        return (radius - self.turning) * self.quotient(radius)

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
            raise _UnsupportedWitnessError("radial quadrature crossed its turning root")
        upper_coordinate = mp.sqrt(upper_radius - self.turning)

        def regularized(coordinate: mp.mpf) -> mp.mpf:
            radius = self.turning + coordinate**2
            return 2 * numerator(radius) / mp.sqrt(self.quotient(radius))

        return mp.quad(regularized, [mp.mpf(0), upper_coordinate])


@dataclass(frozen=True, slots=True, kw_only=True)
class _PathObservables:
    terminal_azimuth: mp.mpf
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
        raise _UnsupportedWitnessError("witness precision must be an integer")
    if precision_digits < MINIMUM_WITNESS_DIGITS:
        raise _UnsupportedWitnessError(
            f"witness precision must be at least {MINIMUM_WITNESS_DIGITS} digits"
        )


def _validate_surface_witness(witness: _SurfaceWitness) -> None:
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
        raise _UnsupportedWitnessError(
            "surface witness does not match the named discrete path identity"
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
        raise _UnsupportedWitnessError(
            "witness contains a non-real or non-finite value"
        )
    if not (
        SURFACE_INNER_RADIUS_M <= witness.source_radius_m <= SURFACE_OUTER_RADIUS_M
    ):
        raise _UnsupportedWitnessError("crossing lies outside the canonical surface")
    positive_fields = (
        witness.frequency_ratio,
        witness.travel_time_m,
        witness.emitted_bolometric_intensity,
        witness.observed_bolometric_intensity,
        witness.energy,
    )
    if any(value <= 0 for value in positive_fields):
        raise _UnsupportedWitnessError("witness contains a non-positive physical value")
    if witness.radial_turning_derivative <= 0 or witness.polar_turning_derivative <= 0:
        raise _UnsupportedWitnessError("separated turning root is not simple")

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
            raise _UnsupportedWitnessError(
                "equation residual does not retain the required "
                f"{certified_digits} decimal digits"
            )


def _validate_escape_witness(witness: _EscapeWitness) -> None:
    _validate_precision_digits(witness.precision_digits)
    discrete_identity = (
        (witness.terminal, _SOURCE_EDGE_ESCAPE_TERMINAL),
        (witness.initial_polar_side, _CANONICAL_INITIAL_POLAR_SIDE),
        (witness.radial_turnings, _CANONICAL_RADIAL_TURNINGS),
        (witness.polar_turnings, _CANONICAL_POLAR_TURNINGS),
        (
            witness.equatorial_crossings_before_terminal,
            _SOURCE_EDGE_ESCAPE_EQUATORIAL_CROSSINGS,
        ),
        (witness.azimuth_winding, _CANONICAL_AZIMUTH_WINDING),
    )
    if any(
        type(actual) is not type(expected) or actual != expected
        for actual, expected in discrete_identity
    ):
        raise _UnsupportedWitnessError(
            "escape witness does not match the named discrete path identity"
        )
    if (
        len(witness.escape_position_xyz_m) != 3
        or len(witness.escape_direction_xyz) != 3
    ):
        raise _UnsupportedWitnessError(
            "escape vectors must contain exactly three lanes"
        )
    continuous_fields = (
        witness.first_equatorial_crossing_radius_m,
        witness.escape_radius_m,
        *witness.escape_position_xyz_m,
        *witness.escape_direction_xyz,
        witness.travel_time_m,
        witness.escape_before_next_crossing_mino_margin,
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
            "escape witness contains a non-real or non-finite value"
        )
    if witness.escape_radius_m != ESCAPE_RADIUS_M:
        raise _UnsupportedWitnessError("escape witness uses the wrong terminal radius")
    if not (
        SURFACE_OUTER_RADIUS_M
        < witness.first_equatorial_crossing_radius_m
        < witness.escape_radius_m
    ):
        raise _UnsupportedWitnessError(
            "escape witness first crossing is not ordered between the outer "
            "source edge and escape terminal"
        )
    positive_fields = (
        witness.travel_time_m,
        witness.escape_before_next_crossing_mino_margin,
        witness.energy,
        witness.radial_turning_derivative,
        witness.polar_turning_derivative,
    )
    if any(value <= 0 for value in positive_fields):
        raise _UnsupportedWitnessError(
            "escape witness contains a non-positive physical or event margin"
        )

    with mp.workdps(witness.precision_digits):
        residual_limit = mp.power(
            10,
            RESIDUAL_GUARD_DIGITS - witness.precision_digits,
        )
        direction_norm_squared = mp.fsum(
            component**2 for component in witness.escape_direction_xyz
        )
        if abs(direction_norm_squared - 1) >= residual_limit:
            raise _UnsupportedWitnessError(
                "escape traversal direction is not normalized"
            )
        if (
            mp.fsum(
                position * direction
                for position, direction in zip(
                    witness.escape_position_xyz_m,
                    witness.escape_direction_xyz,
                    strict=True,
                )
            )
            <= 0
        ):
            raise _UnsupportedWitnessError("escape traversal direction is not outward")

        x, y, z = witness.escape_position_xyz_m
        spin = mp.mpf(4) / 5
        cylindrical_squared = x**2 + y**2
        oblate_term = cylindrical_squared + z**2 - spin**2
        recovered_radius_squared = (
            oblate_term + mp.sqrt(oblate_term**2 + 4 * spin**2 * z**2)
        ) / 2
        radius_residual = (
            abs(mp.sqrt(recovered_radius_squared) - witness.escape_radius_m)
            / witness.escape_radius_m
        )
        if radius_residual >= residual_limit:
            raise _UnsupportedWitnessError(
                "escape position does not lie on the named oblate radius"
            )

        residuals = (
            witness.initial_null_residual,
            witness.mino_constraint_residual,
            witness.chart_primitive_residual,
        )
        if any(residual < 0 or residual >= residual_limit for residual in residuals):
            certified_digits = witness.precision_digits - RESIDUAL_GUARD_DIGITS
            raise _UnsupportedWitnessError(
                "escape equation residual does not retain the required "
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
        raise _UnsupportedWitnessError("canonical observer frame seed is degenerate")
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
    except _UnsupportedWitnessError:
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
        raise _UnsupportedWitnessError(
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
    if turning >= SURFACE_OUTER_RADIUS_M:
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
        mp.mpf(SURFACE_OUTER_RADIUS_M),
    )


def _integrate_path_observables(
    geometry: _ObservationGeometry,
    polar: _PolarMotion,
    radial: _RadialMotion,
    initial_mu: mp.mpf,
    terminal_radius: mp.mpf,
    terminal_mu_magnitude: mp.mpf | None = None,
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

    def polar_time_numerator(mu: mp.mpf) -> mp.mpf:
        return spin * (impact - spin) + spin**2 * mu**2

    polar_time = polar.integrate_to_turn(polar_time_numerator, mp.mpf(0))
    polar_time += polar.integrate_to_turn(polar_time_numerator, initial_mu)
    polar_time += polar.integrate_from_equator(
        polar_time_numerator,
        terminal_mu_magnitude,
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
    polar_azimuth += polar.integrate_from_equator(
        polar_azimuth_numerator,
        terminal_mu_magnitude,
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
    source_azimuth_unwrapped = -(radial_azimuth + polar_azimuth + chart_azimuth)
    observer_cartesian_azimuth = mp.atan2(spin, geometry.radius)
    source_cartesian_azimuth = source_azimuth_unwrapped + mp.atan2(
        spin,
        terminal_radius,
    )
    observer_azimuth_cycle = mp.floor(
        (observer_cartesian_azimuth + mp.pi) / (2 * mp.pi)
    )
    source_azimuth_cycle = mp.floor((source_cartesian_azimuth + mp.pi) / (2 * mp.pi))
    return _PathObservables(
        terminal_azimuth=_wrap_angle(source_azimuth_unwrapped),
        travel_time=radial_time + polar_time + chart_time,
        azimuth_winding=int(source_azimuth_cycle - observer_azimuth_cycle),
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


def _compute_source_edge_escape_witness(
    pixel_x: int,
    pixel_y: int,
    precision_digits: int,
) -> _EscapeWitness:
    geometry = _canonical_geometry()
    initial_ray = _canonical_initial_ray(geometry, pixel_x, pixel_y)
    separated = _separated_initial_state(geometry, initial_ray)
    if separated.radial_velocity <= 0 or separated.polar_velocity >= 0:
        raise _UnsupportedWitnessError(
            "source-edge escape requires a future-outgoing ray after one "
            "northern polar turning"
        )
    initial_mu = mp.cos(geometry.theta)
    polar = _build_polar_motion(
        geometry.spin,
        separated.impact,
        separated.carter,
        initial_mu,
    )
    radial = _build_radial_motion(geometry, separated, precision_digits)
    escape_radius = mp.mpf(ESCAPE_RADIUS_M)
    (
        terminal_mu_magnitude,
        first_crossing_duration,
        next_crossing_margin,
    ) = _solve_escape_polar_endpoint(
        polar,
        radial,
        geometry.radius,
        escape_radius,
        initial_mu,
        precision_digits,
    )
    first_crossing_radius = _solve_equatorial_crossing_radius(
        radial,
        geometry.radius,
        first_crossing_duration,
        precision_digits,
        mp.mpf(SURFACE_OUTER_RADIUS_M),
        geometry.radius,
    )
    path = _integrate_path_observables(
        geometry,
        polar,
        radial,
        initial_mu,
        escape_radius,
        terminal_mu_magnitude,
    )
    position, direction = _escape_position_and_direction(
        geometry,
        radial,
        polar,
        escape_radius,
        terminal_mu_magnitude,
        path.terminal_azimuth,
    )
    return _EscapeWitness(
        precision_digits=precision_digits,
        terminal=_SOURCE_EDGE_ESCAPE_TERMINAL,
        initial_polar_side=_CANONICAL_INITIAL_POLAR_SIDE,
        radial_turnings=_CANONICAL_RADIAL_TURNINGS,
        polar_turnings=_CANONICAL_POLAR_TURNINGS,
        equatorial_crossings_before_terminal=(_SOURCE_EDGE_ESCAPE_EQUATORIAL_CROSSINGS),
        azimuth_winding=path.azimuth_winding,
        first_equatorial_crossing_radius_m=first_crossing_radius,
        escape_radius_m=escape_radius,
        escape_position_xyz_m=position,
        escape_direction_xyz=direction,
        travel_time_m=path.travel_time,
        escape_before_next_crossing_mino_margin=next_crossing_margin,
        energy=separated.energy,
        impact_parameter=separated.impact,
        carter_parameter=separated.carter,
        radial_turning_derivative=radial.turning_derivative,
        polar_turning_derivative=polar.turning_derivative,
        initial_null_residual=initial_ray.initial_null_residual,
        mino_constraint_residual=separated.constraint_residual,
        chart_primitive_residual=path.chart_primitive_residual,
    )


def _compute_surface_witness(
    pixel_x: int, pixel_y: int, precision_digits: int
) -> _SurfaceWitness:
    geometry = _canonical_geometry()
    initial_ray = _canonical_initial_ray(geometry, pixel_x, pixel_y)
    separated = _separated_initial_state(geometry, initial_ray)
    if separated.radial_velocity <= 0 or separated.polar_velocity >= 0:
        raise _UnsupportedWitnessError(
            "named surface witness requires a future-outgoing ray after one northern "
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
    return _SurfaceWitness(
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
        source_azimuth_rad=path.terminal_azimuth,
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


def _canonical_surface_witness(*, precision_digits: int) -> _SurfaceWitness:
    """Compute the single named ordinary-region surface witness."""

    _validate_precision_digits(precision_digits)
    with mp.workdps(precision_digits):
        pixel_x, pixel_y = CANONICAL_PIXEL
        return _compute_surface_witness(
            pixel_x,
            pixel_y,
            precision_digits,
        )


def _source_edge_pair_witness(*, precision_digits: int) -> _SourceEdgePairWitness:
    """Compute the fixed adjacent outside/inside source-edge pair."""

    _validate_precision_digits(precision_digits)
    with mp.workdps(precision_digits):
        outside_x, outside_y = SOURCE_EDGE_OUTSIDE_PIXEL
        inside_x, inside_y = SOURCE_EDGE_INSIDE_PIXEL
        return _SourceEdgePairWitness(
            outside=_compute_source_edge_escape_witness(
                outside_x,
                outside_y,
                precision_digits,
            ),
            inside=_compute_surface_witness(
                inside_x,
                inside_y,
                precision_digits,
            ),
        )


_SURFACE_PRECISION_FIELDS = (
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
_ESCAPE_PRECISION_FIELDS = (
    "first_equatorial_crossing_radius_m",
    "escape_radius_m",
    "travel_time_m",
    "escape_before_next_crossing_mino_margin",
    "energy",
    "impact_parameter",
    "carter_parameter",
    "radial_turning_derivative",
    "polar_turning_derivative",
)


def _certify_precision_doubling(
    values: Iterable[tuple[mp.mpf, mp.mpf]],
) -> mp.mpf:
    normalized_deltas = tuple(
        abs(low - high) / max(mp.mpf(1), abs(high)) for low, high in values
    )
    if not normalized_deltas or not all(
        mp.isfinite(delta) for delta in normalized_deltas
    ):
        raise AssertionError("precision doubling produced an empty or non-finite delta")
    maximum_delta = max(normalized_deltas)
    required = mp.power(10, -REQUIRED_STABLE_DIGITS)
    if maximum_delta >= required:
        raise AssertionError(
            f"precision doubling retained only {-mp.log10(maximum_delta)} digits"
        )
    return maximum_delta


def _build_precision_certificate() -> _PrecisionCertificate[_SurfaceWitness]:
    """Recompute the canonical case at 120 and 180 digits and certify stability."""

    low = _canonical_surface_witness(precision_digits=LOW_PRECISION_DIGITS)
    high = _canonical_surface_witness(precision_digits=HIGH_PRECISION_DIGITS)
    with mp.workdps(HIGH_PRECISION_DIGITS):
        maximum_delta = _certify_precision_doubling(
            (getattr(low, field), getattr(high, field))
            for field in _SURFACE_PRECISION_FIELDS
        )
    return _PrecisionCertificate(
        low_precision_digits=LOW_PRECISION_DIGITS,
        high_precision_digits=HIGH_PRECISION_DIGITS,
        required_stable_digits=REQUIRED_STABLE_DIGITS,
        maximum_normalized_delta=maximum_delta,
        witness=high,
    )


def _build_source_edge_pair_precision_certificate() -> _PrecisionCertificate[
    _SourceEdgePairWitness
]:
    """Recompute both edge cases at 120/180 digits and certify every observable."""

    low = _source_edge_pair_witness(precision_digits=LOW_PRECISION_DIGITS)
    high = _source_edge_pair_witness(precision_digits=HIGH_PRECISION_DIGITS)
    with mp.workdps(HIGH_PRECISION_DIGITS):
        value_pairs = [
            (getattr(low.outside, field), getattr(high.outside, field))
            for field in _ESCAPE_PRECISION_FIELDS
        ]
        value_pairs.extend(
            zip(
                low.outside.escape_position_xyz_m,
                high.outside.escape_position_xyz_m,
                strict=True,
            )
        )
        value_pairs.extend(
            zip(
                low.outside.escape_direction_xyz,
                high.outside.escape_direction_xyz,
                strict=True,
            )
        )
        value_pairs.extend(
            (getattr(low.inside, field), getattr(high.inside, field))
            for field in _SURFACE_PRECISION_FIELDS
        )
        maximum_delta = _certify_precision_doubling(value_pairs)
    return _PrecisionCertificate(
        low_precision_digits=LOW_PRECISION_DIGITS,
        high_precision_digits=HIGH_PRECISION_DIGITS,
        required_stable_digits=REQUIRED_STABLE_DIGITS,
        maximum_normalized_delta=maximum_delta,
        witness=high,
    )


def _scientific(value: mp.mpf, digits: int = 110) -> str:
    return mp.nstr(value, digits, strip_zeros=False)


def _print_path_identity(witness: _SurfaceWitness | _EscapeWitness) -> None:
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


def _print_turning_and_residuals(witness: _SurfaceWitness | _EscapeWitness) -> None:
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


def _print_surface_observables(
    witness: _SurfaceWitness,
    *,
    outer_edge_signed_margin: mp.mpf | None = None,
) -> None:
    print(f"source_radius_m={_scientific(witness.source_radius_m)}")
    if outer_edge_signed_margin is not None:
        print(f"outer_edge_signed_margin_m={_scientific(outer_edge_signed_margin)}")
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


def run() -> None:
    certificate = _build_precision_certificate()
    witness = certificate.witness
    edge_certificate = _build_source_edge_pair_precision_certificate()
    edge = edge_certificate.witness
    with mp.workdps(edge_certificate.high_precision_digits):
        outside_edge_margin = (
            mp.mpf(SURFACE_OUTER_RADIUS_M)
            - edge.outside.first_equatorial_crossing_radius_m
        )
        inside_edge_margin = (
            mp.mpf(SURFACE_OUTER_RADIUS_M) - edge.inside.source_radius_m
        )
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
    _print_path_identity(witness)
    _print_surface_observables(witness)
    _print_turning_and_residuals(witness)
    print(
        "source_edge_precision="
        f"{edge_certificate.low_precision_digits},"
        f"{edge_certificate.high_precision_digits} "
        f"stable_digits>={edge_certificate.required_stable_digits} "
        "maximum_normalized_delta="
        f"{_scientific(edge_certificate.maximum_normalized_delta, 12)}"
    )
    outside = edge.outside
    print("case=kerr-exterior-observation-v1:640:13:0.5:0.5")
    _print_path_identity(outside)
    print(
        "first_equatorial_crossing_radius_m="
        f"{_scientific(outside.first_equatorial_crossing_radius_m)}"
    )
    print(f"outer_edge_signed_margin_m={_scientific(outside_edge_margin)}")
    print(
        "escape_position_xyz_m="
        + ",".join(_scientific(value) for value in outside.escape_position_xyz_m)
    )
    print(
        "escape_direction_xyz="
        + ",".join(_scientific(value) for value in outside.escape_direction_xyz)
    )
    print(f"travel_time_m={_scientific(outside.travel_time_m)}")
    print(
        "escape_before_next_crossing_mino_margin="
        f"{_scientific(outside.escape_before_next_crossing_mino_margin)}"
    )
    _print_turning_and_residuals(outside)
    inside = edge.inside
    print("case=kerr-exterior-observation-v1:640:14:0.5:0.5")
    _print_path_identity(inside)
    _print_surface_observables(
        inside,
        outer_edge_signed_margin=inside_edge_margin,
    )
    print("RESULT=PASS")

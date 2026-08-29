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
from itertools import pairwise

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
_SOURCE_EDGE_PIXELS = tuple((640, pixel_y) for pixel_y in range(12, 21))
_SOURCE_EDGE_ESCAPE_PIXELS = ((640, 12), (640, 13))
_SOURCE_EDGE_SURFACE_PIXELS = (
    (640, 14),
    (640, 15),
    (640, 16),
    (640, 17),
    (640, 18),
    (640, 19),
    (640, 20),
)
_SURFACE_TERMINAL = "equatorial-surface"
_SOURCE_EDGE_INITIAL_POLAR_SIDE = "positive"
_SOURCE_EDGE_RADIAL_TURNINGS = 1
_SOURCE_EDGE_POLAR_TURNINGS = 1
_SURFACE_EQUATORIAL_CROSSINGS_BEFORE_TERMINAL = 0
_SOURCE_EDGE_AZIMUTH_WINDING = 0
_SOURCE_EDGE_ESCAPE_TERMINAL = "escape"
_SOURCE_EDGE_ESCAPE_EQUATORIAL_CROSSINGS = 1
_HORIZON_TERMINAL = "horizon"
_CRITICAL_CURVE_PIXELS = ((33, 10), (33, 11))
_CRITICAL_SURFACE_PIXEL = (33, 10)
_CRITICAL_CAPTURE_PIXEL = (33, 11)
_CRITICAL_VIEWPORT_WIDTH = 64
_CRITICAL_VIEWPORT_HEIGHT = 36
_CRITICAL_SURFACE_POLAR_TURNINGS = 2
_CRITICAL_CAPTURE_POLAR_TURNINGS = 1
_CRITICAL_EQUATORIAL_CROSSINGS = 1
_CRITICAL_SURFACE_AZIMUTH_WINDING = 1
_CRITICAL_CAPTURE_AZIMUTH_WINDING = 0
_CRITICAL_ROOT_CLASS = "exterior-double-root"
_INGOING_CHART_SIGN = 1
_OUTGOING_CHART_SIGN = -1


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
class _SourceEdgeCaseWitness:
    """One fixed pixel and its independently reconstructed terminal witness."""

    pixel: tuple[int, int]
    witness: _SurfaceWitness | _EscapeWitness

    def __post_init__(self) -> None:
        if self.pixel in _SOURCE_EDGE_ESCAPE_PIXELS:
            expected_type = _EscapeWitness
        elif self.pixel in _SOURCE_EDGE_SURFACE_PIXELS:
            expected_type = _SurfaceWitness
        else:
            raise _UnsupportedWitnessError(
                "pixel is not in the named source-edge corpus"
            )
        if type(self.witness) is not expected_type:
            raise _UnsupportedWitnessError(
                "source-edge case does not match its certified terminal stratum"
            )

    @property
    def outer_edge_signed_margin_m(self) -> mp.mpf:
        with mp.workdps(self.witness.precision_digits):
            terminal_radius = (
                self.witness.first_equatorial_crossing_radius_m
                if isinstance(self.witness, _EscapeWitness)
                else self.witness.source_radius_m
            )
            return mp.mpf(SURFACE_OUTER_RADIUS_M) - terminal_radius


@dataclass(frozen=True, slots=True, kw_only=True)
class _SourceEdgeCorpusWitness:
    """The fixed ordered nine-pixel corpus crossing the canonical outer edge."""

    cases: tuple[_SourceEdgeCaseWitness, ...]

    def __post_init__(self) -> None:
        if tuple(case.pixel for case in self.cases) != _SOURCE_EDGE_PIXELS:
            raise _UnsupportedWitnessError(
                "source-edge corpus does not contain the named ordered pixels"
            )
        precision_digits = self.cases[0].witness.precision_digits
        if any(
            case.witness.precision_digits != precision_digits for case in self.cases
        ):
            raise _UnsupportedWitnessError(
                "source-edge corpus mixes working precisions"
            )
        margins = tuple(case.outer_edge_signed_margin_m for case in self.cases)
        if not all(left < right for left, right in pairwise(margins)):
            raise _UnsupportedWitnessError(
                "source-edge corpus margins are not strictly ordered"
            )
        if not margins[1] < 0 < margins[2]:
            raise _UnsupportedWitnessError(
                "source-edge corpus does not bracket the outer radial edge"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class _CriticalSurfaceWitness:
    """Independent higher-order surface result on the scattering side."""

    precision_digits: int
    terminal: str
    initial_polar_side: str
    radial_turnings: int
    polar_turnings: int
    equatorial_crossings_before_terminal: int
    azimuth_winding: int
    first_equatorial_crossing_mino_duration: mp.mpf
    terminal_equatorial_crossing_mino_duration: mp.mpf
    terminal_after_first_crossing_mino_margin: mp.mpf
    first_equatorial_crossing_radius_m: mp.mpf
    first_crossing_below_surface_margin_m: mp.mpf
    radial_turning_above_horizon_margin_m: mp.mpf
    source_radius_m: mp.mpf
    source_azimuth_unwrapped_rad: mp.mpf
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
        _validate_critical_surface_witness(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class _HorizonWitness:
    """Independent monotonic capture result in outgoing Kerr--Schild time."""

    precision_digits: int
    terminal: str
    initial_polar_side: str
    radial_turnings: int
    polar_turnings: int
    equatorial_crossings_before_terminal: int
    azimuth_winding: int
    first_equatorial_crossing_mino_duration: mp.mpf
    horizon_mino_duration: mp.mpf
    first_equatorial_crossing_radius_m: mp.mpf
    first_crossing_below_surface_margin_m: mp.mpf
    horizon_after_first_crossing_mino_margin: mp.mpf
    horizon_radius_m: mp.mpf
    horizon_mu: mp.mpf
    horizon_azimuth_unwrapped_rad: mp.mpf
    horizon_azimuth_rad: mp.mpf
    horizon_position_xyz_m: tuple[mp.mpf, mp.mpf, mp.mpf]
    travel_time_m: mp.mpf
    energy: mp.mpf
    impact_parameter: mp.mpf
    carter_parameter: mp.mpf
    polar_turning_derivative: mp.mpf
    initial_null_residual: mp.mpf
    mino_constraint_residual: mp.mpf
    horizon_cancellation_residual: mp.mpf

    def __post_init__(self) -> None:
        _validate_horizon_witness(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class _CriticalCurveCaseWitness:
    """One named pixel and its signed distance from the radial separatrix."""

    pixel: tuple[int, int]
    witness: _CriticalSurfaceWitness | _HorizonWitness
    exterior_radial_root_count: int
    signed_critical_distance_pixels: mp.mpf
    radial_classification_margin: mp.mpf

    def __post_init__(self) -> None:
        if self.pixel == _CRITICAL_SURFACE_PIXEL:
            expected_type = _CriticalSurfaceWitness
        elif self.pixel == _CRITICAL_CAPTURE_PIXEL:
            expected_type = _HorizonWitness
        else:
            raise _UnsupportedWitnessError(
                "pixel is not in the named critical-curve corpus"
            )
        if type(self.witness) is not expected_type:
            raise _UnsupportedWitnessError(
                "critical-curve case does not match its certified terminal stratum"
            )
        expected_root_count = 2 if self.pixel == _CRITICAL_SURFACE_PIXEL else 0
        if (
            type(self.exterior_radial_root_count) is not int
            or self.exterior_radial_root_count != expected_root_count
        ):
            raise _UnsupportedWitnessError(
                "critical-curve case has the wrong exterior radial root topology"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class _CriticalCurveCorpusWitness:
    """The adjacent higher-order surface/capture pair around one critical curve."""

    critical_root_class: str
    critical_sample_y: mp.mpf
    critical_radius_m: mp.mpf
    critical_potential_residual: mp.mpf
    critical_derivative_residual: mp.mpf
    critical_second_derivative: mp.mpf
    cases: tuple[_CriticalCurveCaseWitness, ...]

    def __post_init__(self) -> None:
        if self.critical_root_class != _CRITICAL_ROOT_CLASS:
            raise _UnsupportedWitnessError(
                "critical-curve corpus has the wrong root multiplicity class"
            )
        if tuple(case.pixel for case in self.cases) != _CRITICAL_CURVE_PIXELS:
            raise _UnsupportedWitnessError(
                "critical-curve corpus does not contain the named ordered pixels"
            )
        precision_digits = self.cases[0].witness.precision_digits
        if any(
            case.witness.precision_digits != precision_digits for case in self.cases
        ):
            raise _UnsupportedWitnessError(
                "critical-curve corpus mixes working precisions"
            )
        continuous_fields = (
            self.critical_sample_y,
            self.critical_radius_m,
            self.critical_potential_residual,
            self.critical_derivative_residual,
            self.critical_second_derivative,
            *(case.signed_critical_distance_pixels for case in self.cases),
            *(case.radial_classification_margin for case in self.cases),
        )
        if not all(
            isinstance(value, mp.mpf) and mp.isfinite(value)
            for value in continuous_fields
        ):
            raise _UnsupportedWitnessError(
                "critical-curve corpus contains a non-real or non-finite value"
            )
        if not self.cases[0].signed_critical_distance_pixels < 0:
            raise _UnsupportedWitnessError(
                "higher-order surface case is not below the critical sample"
            )
        if not self.cases[1].signed_critical_distance_pixels > 0:
            raise _UnsupportedWitnessError(
                "capture case is not above the critical sample"
            )
        if not self.cases[0].radial_classification_margin < 0:
            raise _UnsupportedWitnessError(
                "higher-order surface case lacks a certified radial turning barrier"
            )
        if not self.cases[1].radial_classification_margin > 0:
            raise _UnsupportedWitnessError(
                "capture case lacks a certified positive radial barrier margin"
            )
        if self.critical_second_derivative <= 0:
            raise _UnsupportedWitnessError(
                "critical radial root is not a local double-root barrier"
            )
        with mp.workdps(precision_digits):
            residual_limit = mp.power(
                10,
                RESIDUAL_GUARD_DIGITS - precision_digits,
            )
            distance_residual = max(
                abs(
                    case.signed_critical_distance_pixels
                    - (mp.mpf(case.pixel[1]) + mp.mpf(1) / 2 - self.critical_sample_y)
                )
                for case in self.cases
            )
            if (
                self.critical_potential_residual >= residual_limit
                or self.critical_derivative_residual >= residual_limit
                or distance_residual >= residual_limit
            ):
                raise _UnsupportedWitnessError(
                    "critical double-root residual does not retain the required digits"
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
    chart_sign: int
    radius: mp.mpf
    theta: mp.mpf
    chart_azimuth: mp.mpf
    position: tuple[mp.mpf, mp.mpf, mp.mpf]
    metric: tuple[tuple[mp.mpf, ...], ...]
    viewport_width: int
    viewport_height: int
    vertical_fov: mp.mpf


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
                    abs(self.spin)
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
                    abs(self.spin)
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
class _RadialClassification:
    """Signed exterior potential barrier and the root topology it certifies."""

    margin: mp.mpf
    stationary_radius: mp.mpf
    exterior_roots: tuple[mp.mpf, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _CriticalPoint:
    """Verified double-root location between two adjacent sample centers."""

    sample_y: mp.mpf
    radius: mp.mpf
    potential_residual: mp.mpf
    derivative_residual: mp.mpf
    second_derivative: mp.mpf


@dataclass(frozen=True, slots=True, kw_only=True)
class _CaptureRadialMotion:
    """Monotonic outgoing-chart capture with horizon-regular observables."""

    mass: mp.mpf
    spin: mp.mpf
    impact: mp.mpf
    separation: mp.mpf
    horizon: mp.mpf
    observer_radius: mp.mpf

    def delta(self, radius: mp.mpf) -> mp.mpf:
        return radius**2 - 2 * self.mass * radius + self.spin**2

    def factor(self, radius: mp.mpf) -> mp.mpf:
        return radius**2 + self.spin**2 - self.spin * self.impact

    def potential(self, radius: mp.mpf) -> mp.mpf:
        return self.factor(radius) ** 2 - self.delta(radius) * self.separation

    def root(self, radius: mp.mpf) -> mp.mpf:
        value = self.potential(radius)
        if value <= 0:
            raise _UnsupportedWitnessError(
                "capture radial potential is not strictly positive"
            )
        return mp.sqrt(value)

    def mino_duration_to(self, terminal_radius: mp.mpf) -> mp.mpf:
        if not self.horizon <= terminal_radius <= self.observer_radius:
            raise _UnsupportedWitnessError(
                "capture radial endpoint lies outside the exterior segment"
            )
        return mp.quad(
            lambda radius: 1 / self.root(radius),
            [terminal_radius, self.observer_radius],
        )

    def mino_duration(self) -> mp.mpf:
        return self.mino_duration_to(self.horizon)

    def stable_time_integrand(self, radius: mp.mpf) -> mp.mpf:
        radial_factor = self.factor(radius)
        radial_root = self.root(radius)
        if radial_factor + radial_root <= 0:
            raise _UnsupportedWitnessError(
                "capture time cancellation has an invalid radial factor"
            )
        radius_factor = radius**2 + self.spin**2
        return 1 + (
            radius_factor
            * self.separation
            / (radial_root * (radial_factor + radial_root))
        )

    def stable_azimuth_integrand(self, radius: mp.mpf) -> mp.mpf:
        radial_factor = self.factor(radius)
        radial_root = self.root(radius)
        if radial_factor + radial_root <= 0:
            raise _UnsupportedWitnessError(
                "capture azimuth cancellation has an invalid radial factor"
            )
        return (
            self.spin * self.separation / (radial_root * (radial_factor + radial_root))
        )

    def time_integral(self) -> mp.mpf:
        return mp.quad(
            self.stable_time_integrand,
            [self.horizon, self.observer_radius],
        )

    def azimuth_integral(self) -> mp.mpf:
        return mp.quad(
            self.stable_azimuth_integrand,
            [self.horizon, self.observer_radius],
        )

    def cancellation_residual(self) -> mp.mpf:
        probe = (self.horizon + self.observer_radius) / 2
        radial_factor = self.factor(probe)
        radial_root = self.root(probe)
        direct = (
            (probe**2 + self.spin**2) * radial_factor / radial_root
            - 2 * self.mass * probe
        ) / self.delta(probe)
        stable = self.stable_time_integrand(probe)
        return abs(direct - stable) / max(mp.mpf(1), abs(stable))


@dataclass(frozen=True, slots=True, kw_only=True)
class _PathObservables:
    terminal_azimuth_unwrapped: mp.mpf
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
        (witness.terminal, _SURFACE_TERMINAL),
        (witness.initial_polar_side, _SOURCE_EDGE_INITIAL_POLAR_SIDE),
        (witness.radial_turnings, _SOURCE_EDGE_RADIAL_TURNINGS),
        (witness.polar_turnings, _SOURCE_EDGE_POLAR_TURNINGS),
        (
            witness.equatorial_crossings_before_terminal,
            _SURFACE_EQUATORIAL_CROSSINGS_BEFORE_TERMINAL,
        ),
        (witness.azimuth_winding, _SOURCE_EDGE_AZIMUTH_WINDING),
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
        (witness.initial_polar_side, _SOURCE_EDGE_INITIAL_POLAR_SIDE),
        (witness.radial_turnings, _SOURCE_EDGE_RADIAL_TURNINGS),
        (witness.polar_turnings, _SOURCE_EDGE_POLAR_TURNINGS),
        (
            witness.equatorial_crossings_before_terminal,
            _SOURCE_EDGE_ESCAPE_EQUATORIAL_CROSSINGS,
        ),
        (witness.azimuth_winding, _SOURCE_EDGE_AZIMUTH_WINDING),
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


def _validate_critical_surface_witness(witness: _CriticalSurfaceWitness) -> None:
    _validate_precision_digits(witness.precision_digits)
    discrete_identity = (
        (witness.terminal, _SURFACE_TERMINAL),
        (witness.initial_polar_side, _SOURCE_EDGE_INITIAL_POLAR_SIDE),
        (witness.radial_turnings, _SOURCE_EDGE_RADIAL_TURNINGS),
        (witness.polar_turnings, _CRITICAL_SURFACE_POLAR_TURNINGS),
        (
            witness.equatorial_crossings_before_terminal,
            _CRITICAL_EQUATORIAL_CROSSINGS,
        ),
        (witness.azimuth_winding, _CRITICAL_SURFACE_AZIMUTH_WINDING),
    )
    if any(
        type(actual) is not type(expected) or actual != expected
        for actual, expected in discrete_identity
    ):
        raise _UnsupportedWitnessError(
            "critical surface witness does not match its discrete path identity"
        )
    continuous_fields = (
        witness.first_equatorial_crossing_mino_duration,
        witness.terminal_equatorial_crossing_mino_duration,
        witness.terminal_after_first_crossing_mino_margin,
        witness.first_equatorial_crossing_radius_m,
        witness.first_crossing_below_surface_margin_m,
        witness.radial_turning_above_horizon_margin_m,
        witness.source_radius_m,
        witness.source_azimuth_unwrapped_rad,
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
            "critical surface witness contains a non-real or non-finite value"
        )
    if not (
        witness.first_equatorial_crossing_radius_m < SURFACE_INNER_RADIUS_M
        and SURFACE_INNER_RADIUS_M <= witness.source_radius_m <= SURFACE_OUTER_RADIUS_M
    ):
        raise _UnsupportedWitnessError(
            "critical surface crossings do not certify the named event order"
        )
    positive_fields = (
        witness.first_equatorial_crossing_mino_duration,
        witness.terminal_equatorial_crossing_mino_duration,
        witness.terminal_after_first_crossing_mino_margin,
        witness.first_crossing_below_surface_margin_m,
        witness.radial_turning_above_horizon_margin_m,
        witness.frequency_ratio,
        witness.travel_time_m,
        witness.emitted_bolometric_intensity,
        witness.observed_bolometric_intensity,
        witness.energy,
        witness.radial_turning_derivative,
        witness.polar_turning_derivative,
    )
    if any(value <= 0 for value in positive_fields):
        raise _UnsupportedWitnessError(
            "critical surface witness lacks a positive physical or event margin"
        )

    with mp.workdps(witness.precision_digits):
        residual_limit = mp.power(
            10,
            RESIDUAL_GUARD_DIGITS - witness.precision_digits,
        )
        margin_residual = abs(
            witness.first_crossing_below_surface_margin_m
            - (
                mp.mpf(SURFACE_INNER_RADIUS_M)
                - witness.first_equatorial_crossing_radius_m
            )
        )
        event_residual = abs(
            witness.terminal_after_first_crossing_mino_margin
            - (
                witness.terminal_equatorial_crossing_mino_duration
                - witness.first_equatorial_crossing_mino_duration
            )
        )
        phase_residual = abs(
            _wrap_angle(witness.source_azimuth_unwrapped_rad)
            - witness.source_azimuth_rad
        )
        transfer_residual = abs(
            witness.observed_bolometric_intensity
            - witness.emitted_bolometric_intensity * witness.frequency_ratio**4
        ) / max(mp.mpf(1), abs(witness.observed_bolometric_intensity))
        residuals = (
            witness.initial_null_residual,
            witness.mino_constraint_residual,
            witness.chart_primitive_residual,
            margin_residual,
            event_residual,
            phase_residual,
            transfer_residual,
        )
        if any(residual < 0 or residual >= residual_limit for residual in residuals):
            raise _UnsupportedWitnessError(
                "critical surface equation or identity residual is too large"
            )
        if not -mp.pi <= witness.source_azimuth_rad < mp.pi:
            raise _UnsupportedWitnessError("critical surface phase is not canonical")
        geometry = _critical_curve_geometry()
        if (
            _azimuth_winding(
                geometry,
                witness.source_radius_m,
                witness.source_azimuth_unwrapped_rad,
            )
            != witness.azimuth_winding
        ):
            raise _UnsupportedWitnessError(
                "critical surface unwrapped phase disagrees with its winding"
            )


def _validate_horizon_witness(witness: _HorizonWitness) -> None:
    _validate_precision_digits(witness.precision_digits)
    discrete_identity = (
        (witness.terminal, _HORIZON_TERMINAL),
        (witness.initial_polar_side, _SOURCE_EDGE_INITIAL_POLAR_SIDE),
        (witness.radial_turnings, 0),
        (witness.polar_turnings, _CRITICAL_CAPTURE_POLAR_TURNINGS),
        (
            witness.equatorial_crossings_before_terminal,
            _CRITICAL_EQUATORIAL_CROSSINGS,
        ),
        (witness.azimuth_winding, _CRITICAL_CAPTURE_AZIMUTH_WINDING),
    )
    if any(
        type(actual) is not type(expected) or actual != expected
        for actual, expected in discrete_identity
    ):
        raise _UnsupportedWitnessError(
            "horizon witness does not match its discrete path identity"
        )
    if len(witness.horizon_position_xyz_m) != 3:
        raise _UnsupportedWitnessError(
            "horizon position must contain exactly three lanes"
        )
    continuous_fields = (
        witness.first_equatorial_crossing_mino_duration,
        witness.horizon_mino_duration,
        witness.first_equatorial_crossing_radius_m,
        witness.first_crossing_below_surface_margin_m,
        witness.horizon_after_first_crossing_mino_margin,
        witness.horizon_radius_m,
        witness.horizon_mu,
        witness.horizon_azimuth_unwrapped_rad,
        witness.horizon_azimuth_rad,
        *witness.horizon_position_xyz_m,
        witness.travel_time_m,
        witness.energy,
        witness.impact_parameter,
        witness.carter_parameter,
        witness.polar_turning_derivative,
        witness.initial_null_residual,
        witness.mino_constraint_residual,
        witness.horizon_cancellation_residual,
    )
    if not all(
        isinstance(value, mp.mpf) and mp.isfinite(value) for value in continuous_fields
    ):
        raise _UnsupportedWitnessError(
            "horizon witness contains a non-real or non-finite value"
        )
    if not (
        witness.horizon_radius_m
        < witness.first_equatorial_crossing_radius_m
        < SURFACE_INNER_RADIUS_M
    ):
        raise _UnsupportedWitnessError(
            "horizon witness does not certify the first non-surface crossing"
        )
    if not -1 < witness.horizon_mu < 0:
        raise _UnsupportedWitnessError(
            "horizon endpoint is not on the certified southern polar segment"
        )
    positive_fields = (
        witness.first_equatorial_crossing_mino_duration,
        witness.horizon_mino_duration,
        witness.first_crossing_below_surface_margin_m,
        witness.horizon_after_first_crossing_mino_margin,
        witness.travel_time_m,
        witness.energy,
        witness.polar_turning_derivative,
    )
    if any(value <= 0 for value in positive_fields):
        raise _UnsupportedWitnessError(
            "horizon witness lacks a positive physical or event margin"
        )

    with mp.workdps(witness.precision_digits):
        expected_horizon = mp.mpf(8) / 5
        residual_limit = mp.power(
            10,
            RESIDUAL_GUARD_DIGITS - witness.precision_digits,
        )
        horizon_residual = abs(witness.horizon_radius_m - expected_horizon)
        event_residual = abs(
            witness.horizon_after_first_crossing_mino_margin
            - (
                witness.horizon_mino_duration
                - witness.first_equatorial_crossing_mino_duration
            )
        )
        phase_residual = abs(
            _wrap_angle(witness.horizon_azimuth_unwrapped_rad)
            - witness.horizon_azimuth_rad
        )
        margin_residual = abs(
            witness.first_crossing_below_surface_margin_m
            - (
                mp.mpf(SURFACE_INNER_RADIUS_M)
                - witness.first_equatorial_crossing_radius_m
            )
        )
        sin_theta = mp.sqrt(1 - witness.horizon_mu**2)
        azimuth = witness.horizon_azimuth_rad
        spin = mp.mpf(4) / 5
        expected_position = (
            (witness.horizon_radius_m * mp.cos(azimuth) + spin * mp.sin(azimuth))
            * sin_theta,
            (witness.horizon_radius_m * mp.sin(azimuth) - spin * mp.cos(azimuth))
            * sin_theta,
            witness.horizon_radius_m * witness.horizon_mu,
        )
        position_residual = max(
            abs(actual - expected)
            for actual, expected in zip(
                witness.horizon_position_xyz_m,
                expected_position,
                strict=True,
            )
        ) / max(
            mp.mpf(1),
            *(abs(component) for component in expected_position),
        )
        residuals = (
            witness.initial_null_residual,
            witness.mino_constraint_residual,
            witness.horizon_cancellation_residual,
            horizon_residual,
            event_residual,
            phase_residual,
            margin_residual,
            position_residual,
        )
        if any(residual < 0 or residual >= residual_limit for residual in residuals):
            raise _UnsupportedWitnessError(
                "horizon equation or identity residual is too large"
            )
        if not -mp.pi <= witness.horizon_azimuth_rad < mp.pi:
            raise _UnsupportedWitnessError("horizon phase is not canonical")
        geometry = _critical_curve_geometry()
        if (
            _azimuth_winding(
                geometry,
                witness.horizon_radius_m,
                witness.horizon_azimuth_unwrapped_rad,
            )
            != witness.azimuth_winding
        ):
            raise _UnsupportedWitnessError(
                "horizon unwrapped phase disagrees with its winding"
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


def _build_observation_geometry(
    *,
    spin: mp.mpf,
    chart_sign: int,
    radius: mp.mpf,
    theta: mp.mpf,
    chart_azimuth: mp.mpf,
    viewport_width: int,
    viewport_height: int,
    vertical_fov: mp.mpf,
) -> _ObservationGeometry:
    """Reconstruct one pure-Kerr observation without importing Rust state."""

    if chart_sign not in (_INGOING_CHART_SIGN, _OUTGOING_CHART_SIGN):
        raise _UnsupportedWitnessError("Kerr--Schild chart sign must be +1 or -1")
    mass = mp.mpf(1)
    sin_theta = mp.sin(theta)
    cos_theta = mp.cos(theta)
    sin_azimuth = mp.sin(chart_azimuth)
    cos_azimuth = mp.cos(chart_azimuth)
    chart_spin = chart_sign * spin
    x = (radius * cos_azimuth - chart_spin * sin_azimuth) * sin_theta
    y = (radius * sin_azimuth + chart_spin * cos_azimuth) * sin_theta
    z = radius * cos_theta
    sigma = radius**2 + spin**2 * cos_theta**2
    scalar_f = 2 * mass * radius / sigma
    principal = (
        mp.mpf(1),
        (chart_sign * radius * x + spin * y) / (radius**2 + spin**2),
        (chart_sign * radius * y - spin * x) / (radius**2 + spin**2),
        chart_sign * z / radius,
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
        chart_sign=chart_sign,
        radius=radius,
        theta=theta,
        chart_azimuth=chart_azimuth,
        position=(x, y, z),
        metric=metric,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        vertical_fov=vertical_fov,
    )


def _canonical_geometry() -> _ObservationGeometry:
    return _build_observation_geometry(
        spin=mp.mpf(4) / 5,
        chart_sign=_INGOING_CHART_SIGN,
        radius=mp.mpf(30),
        theta=mp.pi / 3,
        chart_azimuth=mp.mpf(0),
        viewport_width=VIEWPORT_WIDTH,
        viewport_height=VIEWPORT_HEIGHT,
        vertical_fov=mp.pi / 4,
    )


def _critical_curve_geometry() -> _ObservationGeometry:
    return _build_observation_geometry(
        spin=mp.mpf(4) / 5,
        chart_sign=_OUTGOING_CHART_SIGN,
        radius=mp.mpf(30),
        theta=mp.pi / 3,
        chart_azimuth=mp.mpf(0),
        viewport_width=_CRITICAL_VIEWPORT_WIDTH,
        viewport_height=_CRITICAL_VIEWPORT_HEIGHT,
        vertical_fov=mp.pi / 4,
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
    pixel_x: int | mp.mpf,
    pixel_y: int | mp.mpf,
    *,
    coordinates_are_centers: bool = False,
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

    width = mp.mpf(geometry.viewport_width)
    height = mp.mpf(geometry.viewport_height)
    half = mp.mpf(1) / 2
    sample_x = mp.mpf(pixel_x) if coordinates_are_centers else mp.mpf(pixel_x) + half
    sample_y = mp.mpf(pixel_y) if coordinates_are_centers else mp.mpf(pixel_y) + half
    normalized_x = 2 * sample_x / width - 1
    normalized_y = 1 - 2 * sample_y / height
    tangent_half_fov = mp.tan(geometry.vertical_fov / 2)
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
    p_r_ks = (
        sin_theta * mp.cos(geometry.chart_azimuth) * p_x
        + sin_theta * mp.sin(geometry.chart_azimuth) * p_y
        + cos_theta * p_z
    )
    p_theta = mp.cos(theta) / sin_theta * (x * p_x + y * p_y) - radius * sin_theta * p_z
    p_phi = x * p_y - y * p_x
    delta = radius**2 - 2 * mass * radius + spin**2
    p_r_bl = p_r_ks + geometry.chart_sign * (
        2 * mass * radius / delta * p_t + spin / delta * p_phi
    )
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


def _oblate_position(
    geometry: _ObservationGeometry,
    radius: mp.mpf,
    mu: mp.mpf,
    chart_azimuth: mp.mpf,
) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Map one signed BL polar endpoint into the selected KS chart."""

    sin_theta = mp.sqrt(1 - mu**2)
    sin_azimuth = mp.sin(chart_azimuth)
    cos_azimuth = mp.cos(chart_azimuth)
    chart_spin = geometry.chart_sign * geometry.spin
    return (
        (radius * cos_azimuth - chart_spin * sin_azimuth) * sin_theta,
        (radius * sin_azimuth + chart_spin * cos_azimuth) * sin_theta,
        radius * mu,
    )


def _azimuth_winding(
    geometry: _ObservationGeometry,
    terminal_radius: mp.mpf,
    terminal_chart_azimuth_unwrapped: mp.mpf,
) -> int:
    observer_cartesian_azimuth = geometry.chart_azimuth + mp.atan2(
        geometry.chart_sign * geometry.spin,
        geometry.radius,
    )
    terminal_cartesian_azimuth = terminal_chart_azimuth_unwrapped + mp.atan2(
        geometry.chart_sign * geometry.spin,
        terminal_radius,
    )
    observer_cycle = mp.floor((observer_cartesian_azimuth + mp.pi) / (2 * mp.pi))
    terminal_cycle = mp.floor((terminal_cartesian_azimuth + mp.pi) / (2 * mp.pi))
    return int(terminal_cycle - observer_cycle)


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
        mp.mpf(SURFACE_OUTER_RADIUS_M),
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
        initial_polar_side=_SOURCE_EDGE_INITIAL_POLAR_SIDE,
        radial_turnings=_SOURCE_EDGE_RADIAL_TURNINGS,
        polar_turnings=_SOURCE_EDGE_POLAR_TURNINGS,
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
        terminal=_SURFACE_TERMINAL,
        initial_polar_side=_SOURCE_EDGE_INITIAL_POLAR_SIDE,
        radial_turnings=_SOURCE_EDGE_RADIAL_TURNINGS,
        polar_turnings=_SOURCE_EDGE_POLAR_TURNINGS,
        equatorial_crossings_before_terminal=(
            _SURFACE_EQUATORIAL_CROSSINGS_BEFORE_TERMINAL
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


def _compute_critical_surface_witness(
    geometry: _ObservationGeometry,
    initial_ray: _InitialRay,
    separated: _SeparatedState,
    classification: _RadialClassification,
    precision_digits: int,
) -> _CriticalSurfaceWitness:
    if separated.radial_velocity <= 0 or separated.polar_velocity >= 0:
        raise _UnsupportedWitnessError(
            "critical surface witness requires the named outgoing polar branch"
        )
    if classification.margin >= 0 or len(classification.exterior_roots) != 2:
        raise _UnsupportedWitnessError(
            "critical surface witness lacks the two-root scattering topology"
        )
    initial_mu = mp.cos(geometry.theta)
    polar = _build_polar_motion(
        geometry.spin,
        separated.impact,
        separated.carter,
        initial_mu,
    )
    initial_to_turn = polar.integrate_to_turn(_unit_integrand, initial_mu)
    equator_to_turn = polar.integrate_to_turn(_unit_integrand, mp.mpf(0))
    first_crossing_duration = initial_to_turn + equator_to_turn
    second_crossing_duration = initial_to_turn + 3 * equator_to_turn

    radial = _build_radial_motion(geometry, separated, precision_digits)
    first_crossing_radius = _solve_inbound_radius(
        radial,
        geometry.radius,
        first_crossing_duration,
        precision_digits,
    )
    source_radius = _solve_equatorial_crossing_radius(
        radial,
        geometry.radius,
        second_crossing_duration,
        precision_digits,
        mp.mpf(SURFACE_INNER_RADIUS_M),
        mp.mpf(SURFACE_OUTER_RADIUS_M),
    )
    path = _integrate_path_observables(
        geometry,
        polar,
        radial,
        initial_mu,
        source_radius,
        completed_polar_oscillations=1,
    )
    transfer = _surface_transfer_observables(
        geometry,
        initial_ray,
        separated,
        radial,
        source_radius,
    )
    horizon = geometry.mass + mp.sqrt(geometry.mass**2 - geometry.spin**2)
    return _CriticalSurfaceWitness(
        precision_digits=precision_digits,
        terminal=_SURFACE_TERMINAL,
        initial_polar_side=_SOURCE_EDGE_INITIAL_POLAR_SIDE,
        radial_turnings=_SOURCE_EDGE_RADIAL_TURNINGS,
        polar_turnings=_CRITICAL_SURFACE_POLAR_TURNINGS,
        equatorial_crossings_before_terminal=_CRITICAL_EQUATORIAL_CROSSINGS,
        azimuth_winding=path.azimuth_winding,
        first_equatorial_crossing_mino_duration=first_crossing_duration,
        terminal_equatorial_crossing_mino_duration=second_crossing_duration,
        terminal_after_first_crossing_mino_margin=(
            second_crossing_duration - first_crossing_duration
        ),
        first_equatorial_crossing_radius_m=first_crossing_radius,
        first_crossing_below_surface_margin_m=(
            mp.mpf(SURFACE_INNER_RADIUS_M) - first_crossing_radius
        ),
        radial_turning_above_horizon_margin_m=radial.turning - horizon,
        source_radius_m=source_radius,
        source_azimuth_unwrapped_rad=path.terminal_azimuth_unwrapped,
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


def _compute_horizon_witness(
    geometry: _ObservationGeometry,
    initial_ray: _InitialRay,
    separated: _SeparatedState,
    classification: _RadialClassification,
    precision_digits: int,
) -> _HorizonWitness:
    if separated.radial_velocity <= 0 or separated.polar_velocity >= 0:
        raise _UnsupportedWitnessError(
            "horizon witness requires the named outgoing polar branch"
        )
    polar = _build_polar_motion(
        geometry.spin,
        separated.impact,
        separated.carter,
        mp.cos(geometry.theta),
    )
    radial = _build_capture_radial_motion(
        geometry,
        separated,
        classification,
    )
    capture_mino_duration = radial.mino_duration()
    initial_mu = mp.cos(geometry.theta)
    (
        terminal_mu_magnitude,
        first_crossing_duration,
        horizon_after_first_crossing,
    ) = _solve_capture_polar_endpoint(
        polar,
        capture_mino_duration,
        initial_mu,
        precision_digits,
    )
    first_crossing_radius = _solve_capture_radius(
        radial,
        first_crossing_duration,
        precision_digits,
    )
    polar_time, polar_azimuth = _integrate_polar_observables(
        geometry,
        polar,
        separated.impact,
        initial_mu,
        terminal_mu_magnitude,
        0,
    )
    horizon_azimuth_unwrapped = geometry.chart_azimuth - (
        radial.azimuth_integral() + polar_azimuth
    )
    horizon_mu = -terminal_mu_magnitude
    horizon_position = _oblate_position(
        geometry,
        radial.horizon,
        horizon_mu,
        horizon_azimuth_unwrapped,
    )
    return _HorizonWitness(
        precision_digits=precision_digits,
        terminal=_HORIZON_TERMINAL,
        initial_polar_side=_SOURCE_EDGE_INITIAL_POLAR_SIDE,
        radial_turnings=0,
        polar_turnings=_CRITICAL_CAPTURE_POLAR_TURNINGS,
        equatorial_crossings_before_terminal=_CRITICAL_EQUATORIAL_CROSSINGS,
        azimuth_winding=_azimuth_winding(
            geometry,
            radial.horizon,
            horizon_azimuth_unwrapped,
        ),
        first_equatorial_crossing_mino_duration=first_crossing_duration,
        horizon_mino_duration=capture_mino_duration,
        first_equatorial_crossing_radius_m=first_crossing_radius,
        first_crossing_below_surface_margin_m=(
            mp.mpf(SURFACE_INNER_RADIUS_M) - first_crossing_radius
        ),
        horizon_after_first_crossing_mino_margin=horizon_after_first_crossing,
        horizon_radius_m=radial.horizon,
        horizon_mu=horizon_mu,
        horizon_azimuth_unwrapped_rad=horizon_azimuth_unwrapped,
        horizon_azimuth_rad=_wrap_angle(horizon_azimuth_unwrapped),
        horizon_position_xyz_m=horizon_position,
        travel_time_m=radial.time_integral() + polar_time,
        energy=separated.energy,
        impact_parameter=separated.impact,
        carter_parameter=separated.carter,
        polar_turning_derivative=polar.turning_derivative,
        initial_null_residual=initial_ray.initial_null_residual,
        mino_constraint_residual=separated.constraint_residual,
        horizon_cancellation_residual=radial.cancellation_residual(),
    )


def _critical_curve_corpus_witness(
    *,
    precision_digits: int,
) -> _CriticalCurveCorpusWitness:
    """Recompute the adjacent scattering/capture pair from canonical inputs."""

    _validate_precision_digits(precision_digits)
    with mp.workdps(precision_digits):
        geometry = _critical_curve_geometry()
        critical = _solve_critical_point(geometry, precision_digits)
        cases = []
        for pixel in _CRITICAL_CURVE_PIXELS:
            initial_ray = _canonical_initial_ray(geometry, *pixel)
            separated = _separated_initial_state(geometry, initial_ray)
            classification = _classify_radial_barrier(geometry, separated)
            if pixel == _CRITICAL_SURFACE_PIXEL:
                witness = _compute_critical_surface_witness(
                    geometry,
                    initial_ray,
                    separated,
                    classification,
                    precision_digits,
                )
            else:
                witness = _compute_horizon_witness(
                    geometry,
                    initial_ray,
                    separated,
                    classification,
                    precision_digits,
                )
            sample_y = mp.mpf(pixel[1]) + mp.mpf(1) / 2
            cases.append(
                _CriticalCurveCaseWitness(
                    pixel=pixel,
                    witness=witness,
                    exterior_radial_root_count=len(classification.exterior_roots),
                    signed_critical_distance_pixels=sample_y - critical.sample_y,
                    radial_classification_margin=classification.margin,
                )
            )
        return _CriticalCurveCorpusWitness(
            critical_root_class=_CRITICAL_ROOT_CLASS,
            critical_sample_y=critical.sample_y,
            critical_radius_m=critical.radius,
            critical_potential_residual=critical.potential_residual,
            critical_derivative_residual=critical.derivative_residual,
            critical_second_derivative=critical.second_derivative,
            cases=tuple(cases),
        )


def _source_edge_corpus_witness(*, precision_digits: int) -> _SourceEdgeCorpusWitness:
    """Compute the fixed ordered source-edge corpus from canonical inputs."""

    _validate_precision_digits(precision_digits)
    with mp.workdps(precision_digits):
        return _SourceEdgeCorpusWitness(
            cases=tuple(
                _SourceEdgeCaseWitness(
                    pixel=pixel,
                    witness=(
                        _compute_source_edge_escape_witness(
                            *pixel,
                            precision_digits,
                        )
                        if pixel in _SOURCE_EDGE_ESCAPE_PIXELS
                        else _compute_surface_witness(
                            *pixel,
                            precision_digits,
                        )
                    ),
                )
                for pixel in _SOURCE_EDGE_PIXELS
            )
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
_CRITICAL_SURFACE_PRECISION_FIELDS = (
    "first_equatorial_crossing_mino_duration",
    "terminal_equatorial_crossing_mino_duration",
    "terminal_after_first_crossing_mino_margin",
    "first_equatorial_crossing_radius_m",
    "first_crossing_below_surface_margin_m",
    "radial_turning_above_horizon_margin_m",
    "source_radius_m",
    "source_azimuth_unwrapped_rad",
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
_HORIZON_PRECISION_FIELDS = (
    "first_equatorial_crossing_mino_duration",
    "horizon_mino_duration",
    "first_equatorial_crossing_radius_m",
    "first_crossing_below_surface_margin_m",
    "horizon_after_first_crossing_mino_margin",
    "horizon_radius_m",
    "horizon_mu",
    "horizon_azimuth_unwrapped_rad",
    "horizon_azimuth_rad",
    "travel_time_m",
    "energy",
    "impact_parameter",
    "carter_parameter",
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


def _witness_precision_pairs(
    low: _SurfaceWitness | _EscapeWitness,
    high: _SurfaceWitness | _EscapeWitness,
) -> list[tuple[mp.mpf, mp.mpf]]:
    if isinstance(low, _SurfaceWitness) and isinstance(high, _SurfaceWitness):
        return [
            (getattr(low, field), getattr(high, field))
            for field in _SURFACE_PRECISION_FIELDS
        ]
    if isinstance(low, _EscapeWitness) and isinstance(high, _EscapeWitness):
        value_pairs = [
            (getattr(low, field), getattr(high, field))
            for field in _ESCAPE_PRECISION_FIELDS
        ]
        value_pairs.extend(
            zip(
                low.escape_position_xyz_m,
                high.escape_position_xyz_m,
                strict=True,
            )
        )
        value_pairs.extend(
            zip(
                low.escape_direction_xyz,
                high.escape_direction_xyz,
                strict=True,
            )
        )
        return value_pairs
    raise AssertionError("precision doubling changed a source-edge terminal stratum")


def _critical_witness_precision_pairs(
    low: _CriticalSurfaceWitness | _HorizonWitness,
    high: _CriticalSurfaceWitness | _HorizonWitness,
) -> list[tuple[mp.mpf, mp.mpf]]:
    if isinstance(low, _CriticalSurfaceWitness) and isinstance(
        high,
        _CriticalSurfaceWitness,
    ):
        return [
            (getattr(low, field), getattr(high, field))
            for field in _CRITICAL_SURFACE_PRECISION_FIELDS
        ]
    if isinstance(low, _HorizonWitness) and isinstance(high, _HorizonWitness):
        value_pairs = [
            (getattr(low, field), getattr(high, field))
            for field in _HORIZON_PRECISION_FIELDS
        ]
        value_pairs.extend(
            zip(
                low.horizon_position_xyz_m,
                high.horizon_position_xyz_m,
                strict=True,
            )
        )
        return value_pairs
    raise AssertionError("precision doubling changed a critical-curve terminal stratum")


def _build_source_edge_corpus_precision_certificate() -> _PrecisionCertificate[
    _SourceEdgeCorpusWitness
]:
    """Recompute all nine cases at 120/180 digits and certify every observable."""

    low = _source_edge_corpus_witness(precision_digits=LOW_PRECISION_DIGITS)
    high = _source_edge_corpus_witness(precision_digits=HIGH_PRECISION_DIGITS)
    with mp.workdps(HIGH_PRECISION_DIGITS):
        value_pairs = []
        for low_case, high_case in zip(low.cases, high.cases, strict=True):
            value_pairs.extend(
                _witness_precision_pairs(low_case.witness, high_case.witness)
            )
        maximum_delta = _certify_precision_doubling(value_pairs)
    return _PrecisionCertificate(
        low_precision_digits=LOW_PRECISION_DIGITS,
        high_precision_digits=HIGH_PRECISION_DIGITS,
        required_stable_digits=REQUIRED_STABLE_DIGITS,
        maximum_normalized_delta=maximum_delta,
        witness=high,
    )


def _build_critical_curve_precision_certificate() -> _PrecisionCertificate[
    _CriticalCurveCorpusWitness
]:
    """Rebuild both critical-curve cases at 120/180 digits."""

    low = _critical_curve_corpus_witness(precision_digits=LOW_PRECISION_DIGITS)
    high = _critical_curve_corpus_witness(precision_digits=HIGH_PRECISION_DIGITS)
    with mp.workdps(HIGH_PRECISION_DIGITS):
        value_pairs = [
            (low.critical_sample_y, high.critical_sample_y),
            (low.critical_radius_m, high.critical_radius_m),
            (low.critical_second_derivative, high.critical_second_derivative),
        ]
        for low_case, high_case in zip(low.cases, high.cases, strict=True):
            value_pairs.extend(
                (
                    (
                        low_case.signed_critical_distance_pixels,
                        high_case.signed_critical_distance_pixels,
                    ),
                    (
                        low_case.radial_classification_margin,
                        high_case.radial_classification_margin,
                    ),
                )
            )
            value_pairs.extend(
                _critical_witness_precision_pairs(
                    low_case.witness,
                    high_case.witness,
                )
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


def _print_path_identity(
    witness: _SurfaceWitness
    | _EscapeWitness
    | _CriticalSurfaceWitness
    | _HorizonWitness,
) -> None:
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


def _print_turning_and_residuals(
    witness: _SurfaceWitness | _EscapeWitness | _CriticalSurfaceWitness,
) -> None:
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
    witness: _SurfaceWitness | _CriticalSurfaceWitness,
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


def _print_escape_observables(
    witness: _EscapeWitness,
    *,
    outer_edge_signed_margin: mp.mpf,
) -> None:
    print(
        "first_equatorial_crossing_radius_m="
        f"{_scientific(witness.first_equatorial_crossing_radius_m)}"
    )
    print(f"outer_edge_signed_margin_m={_scientific(outer_edge_signed_margin)}")
    print(
        "escape_position_xyz_m="
        + ",".join(_scientific(value) for value in witness.escape_position_xyz_m)
    )
    print(
        "escape_direction_xyz="
        + ",".join(_scientific(value) for value in witness.escape_direction_xyz)
    )
    print(f"travel_time_m={_scientific(witness.travel_time_m)}")
    print(
        "escape_before_next_crossing_mino_margin="
        f"{_scientific(witness.escape_before_next_crossing_mino_margin)}"
    )


def _print_critical_surface_observables(witness: _CriticalSurfaceWitness) -> None:
    print(
        "first_equatorial_crossing_mino_duration="
        f"{_scientific(witness.first_equatorial_crossing_mino_duration)}"
    )
    print(
        "terminal_equatorial_crossing_mino_duration="
        f"{_scientific(witness.terminal_equatorial_crossing_mino_duration)}"
    )
    print(
        "terminal_after_first_crossing_mino_margin="
        f"{_scientific(witness.terminal_after_first_crossing_mino_margin)}"
    )
    print(
        "first_equatorial_crossing_radius_m="
        f"{_scientific(witness.first_equatorial_crossing_radius_m)}"
    )
    print(
        "first_crossing_below_surface_margin_m="
        f"{_scientific(witness.first_crossing_below_surface_margin_m)}"
    )
    print(
        "radial_turning_above_horizon_margin_m="
        f"{_scientific(witness.radial_turning_above_horizon_margin_m)}"
    )
    _print_surface_observables(witness)
    print(
        "source_azimuth_unwrapped_rad="
        f"{_scientific(witness.source_azimuth_unwrapped_rad)}"
    )


def _print_horizon_observables(witness: _HorizonWitness) -> None:
    print(
        "first_equatorial_crossing_mino_duration="
        f"{_scientific(witness.first_equatorial_crossing_mino_duration)}"
    )
    print(f"horizon_mino_duration={_scientific(witness.horizon_mino_duration)}")
    print(
        "first_equatorial_crossing_radius_m="
        f"{_scientific(witness.first_equatorial_crossing_radius_m)}"
    )
    print(
        "first_crossing_below_surface_margin_m="
        f"{_scientific(witness.first_crossing_below_surface_margin_m)}"
    )
    print(
        "horizon_after_first_crossing_mino_margin="
        f"{_scientific(witness.horizon_after_first_crossing_mino_margin)}"
    )
    print(f"horizon_radius_m={_scientific(witness.horizon_radius_m)}")
    print(f"horizon_mu={_scientific(witness.horizon_mu)}")
    print(
        "horizon_azimuth_unwrapped_rad="
        f"{_scientific(witness.horizon_azimuth_unwrapped_rad)}"
    )
    print(f"horizon_azimuth_rad={_scientific(witness.horizon_azimuth_rad)}")
    print(
        "horizon_position_xyz_m="
        + ",".join(_scientific(value) for value in witness.horizon_position_xyz_m)
    )
    print(f"travel_time_m={_scientific(witness.travel_time_m)}")
    print(f"energy={_scientific(witness.energy)}")
    print(f"impact_parameter={_scientific(witness.impact_parameter)}")
    print(f"carter_parameter={_scientific(witness.carter_parameter)}")
    print(
        f"polar_turning_derivative={_scientific(witness.polar_turning_derivative, 20)}"
    )
    print(f"initial_null_residual={_scientific(witness.initial_null_residual, 12)}")
    print(
        f"mino_constraint_residual={_scientific(witness.mino_constraint_residual, 12)}"
    )
    print(
        "horizon_cancellation_residual="
        f"{_scientific(witness.horizon_cancellation_residual, 12)}"
    )


def run() -> None:
    source_edge_certificate = _build_source_edge_corpus_precision_certificate()
    print(f"python={platform.python_version()}")
    print(f"mpmath={mp.__version__}")
    print(
        "source_edge_corpus_precision="
        f"{source_edge_certificate.low_precision_digits},"
        f"{source_edge_certificate.high_precision_digits} "
        f"stable_digits>={source_edge_certificate.required_stable_digits} "
        "maximum_normalized_delta="
        f"{_scientific(source_edge_certificate.maximum_normalized_delta, 12)}"
    )
    for case in source_edge_certificate.witness.cases:
        pixel_x, pixel_y = case.pixel
        print(f"case=kerr-exterior-observation-v1:{pixel_x}:{pixel_y}:0.5:0.5")
        _print_path_identity(case.witness)
        if isinstance(case.witness, _EscapeWitness):
            _print_escape_observables(
                case.witness,
                outer_edge_signed_margin=case.outer_edge_signed_margin_m,
            )
        else:
            _print_surface_observables(
                case.witness,
                outer_edge_signed_margin=case.outer_edge_signed_margin_m,
            )
        _print_turning_and_residuals(case.witness)

    critical_certificate = _build_critical_curve_precision_certificate()
    critical_corpus = critical_certificate.witness
    print(
        "critical_curve_corpus_precision="
        f"{critical_certificate.low_precision_digits},"
        f"{critical_certificate.high_precision_digits} "
        f"stable_digits>={critical_certificate.required_stable_digits} "
        "maximum_normalized_delta="
        f"{_scientific(critical_certificate.maximum_normalized_delta, 12)}"
    )
    print(f"critical_root_class={critical_corpus.critical_root_class}")
    print(f"critical_sample_y={_scientific(critical_corpus.critical_sample_y)}")
    print(f"critical_radius_m={_scientific(critical_corpus.critical_radius_m)}")
    print(
        "critical_potential_residual="
        f"{_scientific(critical_corpus.critical_potential_residual, 12)}"
    )
    print(
        "critical_derivative_residual="
        f"{_scientific(critical_corpus.critical_derivative_residual, 12)}"
    )
    print(
        "critical_second_derivative="
        f"{_scientific(critical_corpus.critical_second_derivative)}"
    )
    for case in critical_corpus.cases:
        pixel_x, pixel_y = case.pixel
        print(f"case=kerr-critical-outgoing-v1:{pixel_x}:{pixel_y}:0.5:0.5")
        print(
            "signed_critical_distance_pixels="
            f"{_scientific(case.signed_critical_distance_pixels)}"
        )
        print(f"exterior_radial_root_count={case.exterior_radial_root_count}")
        print(
            "radial_classification_margin="
            f"{_scientific(case.radial_classification_margin)}"
        )
        _print_path_identity(case.witness)
        if isinstance(case.witness, _CriticalSurfaceWitness):
            _print_critical_surface_observables(case.witness)
            _print_turning_and_residuals(case.witness)
        else:
            _print_horizon_observables(case.witness)
    print("RESULT=PASS")

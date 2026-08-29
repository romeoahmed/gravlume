"""Validated records and shared constants for private BL/Mino proofs.

Proof-specific validators are imported during ``__post_init__`` so record
mutation keeps its scientific checks without creating a cyclic top-level
module graph.
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import mpmath as mp

from ..._precision import _PrecisionEvidence

_MINIMUM_WITNESS_DIGITS = 70
_RESIDUAL_GUARD_DIGITS = 15
_SURFACE_INNER_RADIUS_M = 6
_SURFACE_OUTER_RADIUS_M = 20
_ESCAPE_RADIUS_M = 200
_VIEWPORT_WIDTH = 1280
_VIEWPORT_HEIGHT = 720
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
_NEGATIVE_SPIN_PIXEL = (62, 7)
_NEGATIVE_SPIN_VIEWPORT_WIDTH = 64
_NEGATIVE_SPIN_VIEWPORT_HEIGHT = 36
_NEGATIVE_SPIN_EMITTER_BRANCH_SIGN = -1
_NEGATIVE_SPIN_ROOT_CLASS = "two-exterior-simple-roots"
_INGOING_CHART_SIGN = 1
_OUTGOING_CHART_SIGN = -1


class _UnsupportedWitnessError(ValueError):
    """The requested case lies outside this research slice's certified domain."""


def _validate_precision_digits(precision_digits: object) -> None:
    if type(precision_digits) is not int:
        raise _UnsupportedWitnessError("witness precision must be an integer")
    if precision_digits < _MINIMUM_WITNESS_DIGITS:
        raise _UnsupportedWitnessError(
            f"witness precision must be at least {_MINIMUM_WITNESS_DIGITS} digits"
        )


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
        from ._source_edge import _validate_surface_witness

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
        from ._source_edge import _validate_escape_witness

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
            return mp.mpf(_SURFACE_OUTER_RADIUS_M) - terminal_radius


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
        from ._critical_curve import _validate_critical_surface_witness

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
        from ._critical_curve import _validate_horizon_witness

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
                _RESIDUAL_GUARD_DIGITS - precision_digits,
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
class _NegativeSpinSurfaceWitness:
    """Independent continuous-field certificate for one negative-spin ray."""

    pixel: tuple[int, int]
    precision_digits: int
    physical_spin_m: mp.mpf
    chart_sign: int
    emitter_branch_sign: int
    terminal: str
    initial_polar_side: str
    radial_root_class: str
    exterior_radial_root_count: int
    radial_turnings: int
    polar_turnings: int
    equatorial_crossings_before_terminal: int
    azimuth_winding: int
    source_mino_duration: mp.mpf
    radial_turn_mino_duration: mp.mpf
    source_after_radial_turn_mino_margin: mp.mpf
    next_crossing_after_source_mino_margin: mp.mpf
    radial_stationary_radius_m: mp.mpf
    radial_classification_margin: mp.mpf
    radial_turning_radius_m: mp.mpf
    radial_turning_above_horizon_margin_m: mp.mpf
    source_radius_m: mp.mpf
    source_inner_margin_m: mp.mpf
    source_outer_margin_m: mp.mpf
    source_azimuth_unwrapped_rad: mp.mpf
    source_azimuth_rad: mp.mpf
    emitter_angular_velocity_per_m: mp.mpf
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
        from ._negative_spin import _validate_negative_spin_surface_witness

        _validate_negative_spin_surface_witness(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class _PrecisionCertificate[Witness]:
    """Precision-doubling evidence for one named witness."""

    precision: _PrecisionEvidence
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

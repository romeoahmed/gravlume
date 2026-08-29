"""Certify a private Kerr-root and Carlson-oracle research slice.

The topology half isolates real roots of exact rational Kerr quartics with
SymPy.  It never promotes a numerically small imaginary part to a real root.
The Carlson half evaluates the positive-real defining integrals independently
of mpmath's special-function implementations, then checks identities and
precision doubling.  Neither half is a production solver or a WGSL accuracy
claim.

Primary sources:

* Kerr exterior root taxonomy: https://doi.org/10.1103/PhysRevD.101.044032
* Carlson definitions and algorithms: https://arxiv.org/abs/math/9409227
* DLMF definitions and duplication: https://dlmf.nist.gov/19.16
  and https://dlmf.nist.gov/19.26
"""

import platform
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import cache
from itertools import pairwise

import mpmath as mp
import sympy as sp

LOW_PRECISION_DIGITS = 120
HIGH_PRECISION_DIGITS = 180
REQUIRED_STABLE_DIGITS = 80

_ROOT_SEPARATION_LIMIT = "1e-40"
_CARLSON_CANCELLATION_LIMIT = "1e40"
_CARLSON_PARAMETER_LIMIT = "1e-40"


class _UnsupportedTopologyError(ValueError):
    """The radial case is inconsistent with this certified research slice."""


class _UnsupportedCarlsonError(ValueError):
    """The Carlson input is outside this oracle's positive-real domain."""


class _TopologyDecision(Enum):
    ELIGIBLE_FOR_FURTHER_ANALYSIS = "eligible-for-further-analysis"
    FALLBACK = "fallback"


class _CarlsonDecision(Enum):
    ELIGIBLE_FOR_FURTHER_ANALYSIS = "eligible-for-further-analysis"
    FALLBACK = "fallback"


class _CarlsonKind(Enum):
    RF = "RF"
    RC = "RC"
    RD = "RD"
    RJ = "RJ"


@dataclass(frozen=True, slots=True, kw_only=True)
class _IsolatedRoot:
    """One exact rational isolating interval and its numerical midpoint."""

    midpoint: mp.mpf
    lower: mp.mpf
    upper: mp.mpf
    multiplicity: int
    lower_sign: int
    upper_sign: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _TopologyCase:
    """One exact rational Kerr input and its expected conservative outcome."""

    name: str
    mass: Fraction
    spin: Fraction
    energy: Fraction
    impact: Fraction
    carter: Fraction
    initial_radius: Fraction
    initial_radial_sign: int
    polar_mu_squared: Fraction
    expected_topology: str
    expected_motion_branch: str | None
    expected_turn_sequence: tuple[str, ...]
    expected_decision: _TopologyDecision

    def __post_init__(self) -> None:
        if not self.name or self.mass <= 0:
            raise _UnsupportedTopologyError(
                f"{self.name!r} is outside the nonextremal separated null domain"
            )
        separation = (self.impact - self.spin) ** 2 + self.carter
        conditioning_limit = Fraction(1, 10**40)
        extremality_gap = (self.mass**2 - self.spin**2) / self.mass**2
        axis_denominator = 1 - self.polar_mu_squared
        if (
            self.energy <= 0
            or not 0 < abs(self.spin) < self.mass
            or self.initial_radius <= 0
            or self.initial_radial_sign not in (-1, 1)
            or not 0 <= self.polar_mu_squared < 1
            or separation <= 0
            or extremality_gap <= conditioning_limit
            or axis_denominator <= conditioning_limit
        ):
            raise _UnsupportedTopologyError(
                f"{self.name!r} is outside the nonextremal separated null domain"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class _TopologyReport:
    """One precision-specific exact-isolation topology report."""

    name: str
    precision_digits: int
    normalized_spin: Fraction
    normalized_impact: Fraction
    normalized_carter: Fraction
    polynomial_coefficients: tuple[Fraction, ...]
    horizon_radii: tuple[mp.mpf, mp.mpf]
    initial_radius: mp.mpf
    initial_radial_sign: int
    ordered_real_roots: tuple[mp.mpf, ...]
    ordered_real_root_brackets: tuple[tuple[mp.mpf, mp.mpf], ...]
    real_root_bracket_signs: tuple[tuple[int, int], ...]
    real_root_multiplicities: tuple[int, ...]
    complex_root_count: int
    stationary_points: tuple[mp.mpf, ...]
    stationary_values: tuple[mp.mpf, ...]
    topology_id: str
    motion_branch: str | None
    turn_sequence: tuple[str, ...]
    decision: _TopologyDecision
    fallback_reasons: tuple[str, ...]
    minimum_root_separation: mp.mpf
    required_root_separation: mp.mpf
    minimum_horizon_root_separation: mp.mpf
    minimum_initial_root_separation: mp.mpf
    minimum_root_derivative: mp.mpf
    maximum_root_residual: mp.mpf
    vieta_residual: mp.mpf | None
    stationary_real_root_count: int | None
    polar_margin: mp.mpf

    def __post_init__(self) -> None:
        with mp.workdps(self.precision_digits + 20):
            if (
                not self.name
                or self.precision_digits < REQUIRED_STABLE_DIGITS
                or len(self.polynomial_coefficients) != 5
                or self.polynomial_coefficients[0] != 1
                or len(self.ordered_real_roots) != len(self.real_root_multiplicities)
                or len(self.ordered_real_roots) != len(self.ordered_real_root_brackets)
                or len(self.ordered_real_roots) != len(self.real_root_bracket_signs)
                or self.initial_radial_sign not in (-1, 1)
                or self.polar_margin <= 0
                or sum(self.real_root_multiplicities) + self.complex_root_count != 4
                or self.complex_root_count < 0
                or self.complex_root_count % 2 != 0
            ):
                raise _UnsupportedTopologyError("invalid radial topology report")
            if any(
                not left < right for left, right in pairwise(self.ordered_real_roots)
            ):
                raise _UnsupportedTopologyError(
                    "real roots must remain in strict ascending order"
                )
            for root, bracket, signs, multiplicity in zip(
                self.ordered_real_roots,
                self.ordered_real_root_brackets,
                self.real_root_bracket_signs,
                self.real_root_multiplicities,
                strict=True,
            ):
                lower, upper = bracket
                if not lower <= root <= upper:
                    raise _UnsupportedTopologyError(
                        "root midpoint escaped its exact isolating interval"
                    )
                if multiplicity == 1 and lower < upper and signs[0] * signs[1] >= 0:
                    raise _UnsupportedTopologyError(
                        "simple-root bracket must retain an exact sign change"
                    )
            if self.topology_id == "I" and len(self.ordered_real_roots) != 4:
                raise _UnsupportedTopologyError("class I requires four real roots")
            if self.topology_id == "II" and len(self.ordered_real_roots) != 4:
                raise _UnsupportedTopologyError("class II requires four real roots")
            if self.topology_id == "III" and len(self.ordered_real_roots) != 2:
                raise _UnsupportedTopologyError("class III requires two real roots")
            if self.topology_id == "IV" and self.ordered_real_roots:
                raise _UnsupportedTopologyError("class IV has no real roots")
            expected_turns = (
                ("outer-radial-turn",)
                if self.motion_branch == "Ib" and self.initial_radial_sign < 0
                else ()
            )
            if self.turn_sequence != expected_turns:
                raise _UnsupportedTopologyError(
                    "turn sequence does not match the initial radial sign"
                )
            if self.decision is _TopologyDecision.ELIGIBLE_FOR_FURTHER_ANALYSIS:
                if (
                    self.topology_id != "I"
                    or self.motion_branch != "Ib"
                    or self.fallback_reasons
                    or self.minimum_root_separation <= self.required_root_separation
                    or self.minimum_horizon_root_separation
                    <= self.required_root_separation
                    or self.minimum_initial_root_separation
                    <= self.required_root_separation
                    or any(
                        lower_sign * upper_sign >= 0
                        for lower_sign, upper_sign in self.real_root_bracket_signs
                    )
                ):
                    raise _UnsupportedTopologyError(
                        "only a separated class-Ib report can be eligible"
                    )
            elif not self.fallback_reasons:
                raise _UnsupportedTopologyError(
                    "a fallback report must retain its reasons"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class _TopologyCertificate:
    """Named topology corpus reconstructed at two decimal precisions."""

    low_precision_digits: int
    high_precision_digits: int
    required_stable_digits: int
    maximum_normalized_delta: mp.mpf
    false_acceptance_count: int
    reports: tuple[_TopologyReport, ...]

    def report(self, name: str) -> _TopologyReport:
        for report in self.reports:
            if report.name == name:
                return report
        raise KeyError(f"unknown topology report {name!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class _CarlsonDifferenceReport:
    """Condition report for the cancellation-prone RC-difference identity."""

    precision_digits: int
    cancellation_ratio: mp.mpf
    normalized_identity_residual: mp.mpf
    decision: _CarlsonDecision

    def __post_init__(self) -> None:
        if self.precision_digits < REQUIRED_STABLE_DIGITS:
            raise _UnsupportedCarlsonError("insufficient working precision")
        if self.cancellation_ratio < 1 or self.normalized_identity_residual < 0:
            raise _UnsupportedCarlsonError("invalid cancellation report")
        limit = mp.mpf(_CARLSON_CANCELLATION_LIMIT)
        expected = (
            _CarlsonDecision.FALLBACK
            if self.cancellation_ratio > limit
            else _CarlsonDecision.ELIGIBLE_FOR_FURTHER_ANALYSIS
        )
        if self.decision is not expected:
            raise _UnsupportedCarlsonError("cancellation decision lost its guard")


@dataclass(frozen=True, slots=True, kw_only=True)
class _CarlsonCertificate:
    """Positive-real definition, identity, and Kerr-reduction certificate."""

    low_precision_digits: int
    high_precision_digits: int
    required_stable_digits: int
    maximum_definition_residual: mp.mpf
    maximum_identity_residual: mp.mpf
    maximum_normalized_delta: mp.mpf
    rd_xy_symmetry_residual: mp.mpf
    rd_full_permutation_delta: mp.mpf
    kerr_radial_reduction_residual: mp.mpf


def _named_topology_cases() -> tuple[_TopologyCase, ...]:
    one = Fraction(1)
    spin = Fraction(4, 5)
    class_i_impact = Fraction(-12)
    class_i_carter = Fraction(1, 100)
    fallback = _TopologyDecision.FALLBACK
    eligible = _TopologyDecision.ELIGIBLE_FOR_FURTHER_ANALYSIS
    return (
        _TopologyCase(
            name="class-i-ib-inward",
            mass=one,
            spin=spin,
            energy=one,
            impact=class_i_impact,
            carter=class_i_carter,
            initial_radius=Fraction(12),
            initial_radial_sign=-1,
            polar_mu_squared=Fraction(0),
            expected_topology="I",
            expected_motion_branch="Ib",
            expected_turn_sequence=("outer-radial-turn",),
            expected_decision=eligible,
        ),
        _TopologyCase(
            name="class-i-ib-outward",
            mass=one,
            spin=spin,
            energy=one,
            impact=class_i_impact,
            carter=class_i_carter,
            initial_radius=Fraction(12),
            initial_radial_sign=1,
            polar_mu_squared=Fraction(0),
            expected_topology="I",
            expected_motion_branch="Ib",
            expected_turn_sequence=(),
            expected_decision=eligible,
        ),
        _TopologyCase(
            name="class-i-ia",
            mass=one,
            spin=spin,
            energy=one,
            impact=class_i_impact,
            carter=class_i_carter,
            initial_radius=Fraction(2),
            initial_radial_sign=-1,
            polar_mu_squared=Fraction(0),
            expected_topology="I",
            expected_motion_branch="Ia",
            expected_turn_sequence=(),
            expected_decision=fallback,
        ),
        _TopologyCase(
            name="class-ii",
            mass=one,
            spin=spin,
            energy=one,
            impact=Fraction(469, 500),
            carter=Fraction(1, 60_000),
            initial_radius=Fraction(12),
            initial_radial_sign=-1,
            polar_mu_squared=Fraction(0),
            expected_topology="II",
            expected_motion_branch=None,
            expected_turn_sequence=(),
            expected_decision=fallback,
        ),
        _TopologyCase(
            name="class-iii",
            mass=one,
            spin=spin,
            energy=one,
            impact=Fraction(-13, 2),
            carter=Fraction(1, 100),
            initial_radius=Fraction(12),
            initial_radial_sign=-1,
            polar_mu_squared=Fraction(0),
            expected_topology="III",
            expected_motion_branch=None,
            expected_turn_sequence=(),
            expected_decision=fallback,
        ),
        _TopologyCase(
            name="class-iv",
            mass=one,
            spin=spin,
            energy=one,
            impact=Fraction(-3, 20),
            carter=Fraction(-2, 5),
            initial_radius=Fraction(12),
            initial_radial_sign=-1,
            polar_mu_squared=Fraction(407, 512),
            expected_topology="IV",
            expected_motion_branch=None,
            expected_turn_sequence=(),
            expected_decision=fallback,
        ),
        _TopologyCase(
            name="negative-spin-ib",
            mass=one,
            spin=-spin,
            energy=one,
            impact=-class_i_impact,
            carter=class_i_carter,
            initial_radius=Fraction(12),
            initial_radial_sign=-1,
            polar_mu_squared=Fraction(0),
            expected_topology="I",
            expected_motion_branch="Ib",
            expected_turn_sequence=("outer-radial-turn",),
            expected_decision=eligible,
        ),
        _TopologyCase(
            name="exact-double-boundary",
            mass=one,
            spin=spin,
            energy=one,
            impact=Fraction(59, 80),
            carter=Fraction(5375, 256),
            initial_radius=Fraction(12),
            initial_radial_sign=-1,
            polar_mu_squared=Fraction(0),
            expected_topology="boundary",
            expected_motion_branch=None,
            expected_turn_sequence=(),
            expected_decision=fallback,
        ),
        _TopologyCase(
            name="near-double-boundary",
            mass=one,
            spin=spin,
            energy=one,
            impact=Fraction(59, 80) + Fraction(1, 10**100),
            carter=Fraction(5375, 256),
            initial_radius=Fraction(12),
            initial_radial_sign=-1,
            polar_mu_squared=Fraction(0),
            expected_topology="I",
            expected_motion_branch="Ib",
            expected_turn_sequence=("outer-radial-turn",),
            expected_decision=fallback,
        ),
    )


def _canonical_parameters(case: _TopologyCase) -> tuple[Fraction, Fraction, Fraction]:
    impact = case.impact
    carter = case.carter
    spin = case.spin
    if spin < 0:
        spin = -spin
        impact = -impact
    return spin, impact, carter


def _radial_coefficients(case: _TopologyCase) -> tuple[Fraction, ...]:
    spin, impact, carter = _canonical_parameters(case)
    return (
        Fraction(1),
        Fraction(0),
        spin**2 - impact**2 - carter,
        2 * case.mass * ((impact - spin) ** 2 + carter),
        -(spin**2) * carter,
    )


def _sympy_fraction(value: Fraction) -> sp.Rational:
    return sp.Rational(value.numerator, value.denominator)


def _mp_fraction(value: Fraction | sp.Rational, precision_digits: int) -> mp.mpf:
    with mp.workdps(precision_digits):
        if isinstance(value, Fraction):
            return mp.mpf(value.numerator) / value.denominator
        return mp.mpf(int(value.p)) / int(value.q)


def _exact_polynomial(coefficients: tuple[Fraction, ...]) -> sp.Poly:
    radius = sp.Symbol("r", real=True)
    expression = sum(
        _sympy_fraction(coefficient) * radius ** (4 - index)
        for index, coefficient in enumerate(coefficients)
    )
    return sp.Poly(expression, radius, domain=sp.QQ)


def _isolated_real_roots(
    polynomial: sp.Poly,
    precision_digits: int,
) -> tuple[_IsolatedRoot, ...]:
    epsilon = sp.Rational(1, 10 ** (precision_digits + 30))
    intervals = polynomial.intervals(eps=epsilon)
    roots: list[_IsolatedRoot] = []
    for (lower, upper), multiplicity in intervals:
        midpoint = (lower + upper) / 2
        roots.append(
            _IsolatedRoot(
                midpoint=_mp_fraction(midpoint, precision_digits + 60),
                lower=_mp_fraction(lower, precision_digits + 60),
                upper=_mp_fraction(upper, precision_digits + 60),
                multiplicity=multiplicity,
                lower_sign=int(sp.sign(polynomial.eval(lower))),
                upper_sign=int(sp.sign(polynomial.eval(upper))),
            )
        )
    return tuple(roots)


def _evaluate_polynomial(coefficients: tuple[Fraction, ...], value: mp.mpf) -> mp.mpf:
    result = mp.mpf("0")
    for coefficient in coefficients:
        result = (
            result * value + mp.mpf(coefficient.numerator) / coefficient.denominator
        )
    return result


def _evaluate_radial_derivative(
    coefficients: tuple[Fraction, ...],
    value: mp.mpf,
) -> mp.mpf:
    _, _, quadratic, linear, _ = coefficients
    q = mp.mpf(quadratic.numerator) / quadratic.denominator
    ell = mp.mpf(linear.numerator) / linear.denominator
    return 4 * value**3 + 2 * q * value + ell


def _normalized_residual(value: mp.mpf, scale: mp.mpf) -> mp.mpf:
    return abs(value) / max(mp.mpf(1), abs(scale))


def _point_interval_distance(
    point: mp.mpf,
    interval: tuple[mp.mpf, mp.mpf],
) -> mp.mpf:
    lower, upper = interval
    if point < lower:
        return lower - point
    if point > upper:
        return point - upper
    return mp.mpf("0")


def _stationary_root_count(
    stationary_values: tuple[mp.mpf, ...],
    required_margin: mp.mpf,
) -> int | None:
    if any(abs(value) <= required_margin for value in stationary_values):
        return None
    signs = (1, *(1 if value > 0 else -1 for value in stationary_values), 1)
    return sum(left != right for left, right in pairwise(signs))


def _classify_topology(
    roots: tuple[mp.mpf, ...],
    multiplicities: tuple[int, ...],
    horizons: tuple[mp.mpf, mp.mpf],
) -> str:
    if any(multiplicity != 1 for multiplicity in multiplicities):
        return "boundary"
    inner_horizon, outer_horizon = horizons
    if len(roots) == 4:
        if roots[1] < inner_horizon < outer_horizon < roots[2]:
            return "I"
        if roots[3] < inner_horizon < outer_horizon:
            return "II"
    elif len(roots) == 2 and roots[1] < inner_horizon < outer_horizon:
        return "III"
    elif not roots:
        return "IV"
    return "boundary"


def _vieta_residual(
    roots: tuple[mp.mpf, ...],
    coefficients: tuple[Fraction, ...],
) -> mp.mpf | None:
    if len(roots) != 4:
        return None
    _, cubic, quadratic, linear, constant = coefficients
    expected = (
        -mp.mpf(cubic.numerator) / cubic.denominator,
        mp.mpf(quadratic.numerator) / quadratic.denominator,
        -mp.mpf(linear.numerator) / linear.denominator,
        mp.mpf(constant.numerator) / constant.denominator,
    )
    actual = (
        sum(roots, mp.mpf("0")),
        sum(
            roots[left] * roots[right]
            for left in range(4)
            for right in range(left + 1, 4)
        ),
        sum(
            roots[first] * roots[second] * roots[third]
            for first in range(4)
            for second in range(first + 1, 4)
            for third in range(second + 1, 4)
        ),
        mp.fprod(roots),
    )
    return max(
        _normalized_residual(found - wanted, wanted)
        for found, wanted in zip(actual, expected, strict=True)
    )


def _polar_margin(case: _TopologyCase, precision_digits: int) -> mp.mpf:
    spin, impact, carter = _canonical_parameters(case)
    mu_squared = case.polar_mu_squared
    exact = (
        carter + (spin**2 - carter - impact**2) * mu_squared - spin**2 * mu_squared**2
    )
    if exact <= 0:
        raise _UnsupportedTopologyError(
            f"{case.name}: initial polar point is not in the allowed region"
        )
    return _mp_fraction(exact, precision_digits + 20)


def _build_topology_report(
    case: _TopologyCase,
    precision_digits: int,
) -> _TopologyReport:
    with mp.workdps(precision_digits + 30):
        spin, impact, carter = _canonical_parameters(case)
        coefficients = _radial_coefficients(case)
        polynomial = _exact_polynomial(coefficients)
        isolated = _isolated_real_roots(polynomial, precision_digits)
        roots = tuple(root.midpoint for root in isolated)
        root_brackets = tuple((root.lower, root.upper) for root in isolated)
        root_bracket_signs = tuple(
            (root.lower_sign, root.upper_sign) for root in isolated
        )
        multiplicities = tuple(root.multiplicity for root in isolated)
        complex_root_count = 4 - sum(multiplicities)

        mass = _mp_fraction(case.mass, precision_digits + 20)
        spin_value = _mp_fraction(spin, precision_digits + 20)
        horizon_offset = mp.sqrt(mass**2 - spin_value**2)
        horizons = (mass - horizon_offset, mass + horizon_offset)
        topology_id = _classify_topology(roots, multiplicities, horizons)

        derivative_roots = _isolated_real_roots(polynomial.diff(), precision_digits)
        stationary_points = tuple(root.midpoint for root in derivative_roots)
        stationary_values = tuple(
            _evaluate_polynomial(coefficients, point) for point in stationary_points
        )

        initial_radius = _mp_fraction(case.initial_radius, precision_digits + 20)
        motion_branch: str | None = None
        if topology_id == "I":
            if initial_radius > roots[3]:
                motion_branch = "Ib"
            elif horizons[1] < initial_radius < roots[2]:
                motion_branch = "Ia"
        turn_sequence = (
            ("outer-radial-turn",)
            if motion_branch == "Ib" and case.initial_radial_sign < 0
            else ()
        )

        required_separation = mp.mpf(_ROOT_SEPARATION_LIMIT)
        if any(multiplicity > 1 for multiplicity in multiplicities):
            minimum_separation = mp.mpf("0")
        elif len(root_brackets) >= 2:
            minimum_separation = min(
                right[0] - left[1] for left, right in pairwise(root_brackets)
            )
        else:
            minimum_separation = mp.inf
        minimum_horizon_separation = min(
            (
                _point_interval_distance(horizon, bracket)
                for horizon in horizons
                for bracket in root_brackets
            ),
            default=mp.inf,
        )
        minimum_initial_separation = min(
            (
                _point_interval_distance(initial_radius, bracket)
                for bracket in root_brackets
            ),
            default=mp.inf,
        )
        stationary_real_root_count = _stationary_root_count(
            stationary_values,
            required_separation,
        )
        derivative_margins = tuple(
            abs(_evaluate_radial_derivative(coefficients, root)) for root in roots
        )
        minimum_derivative = min(derivative_margins, default=mp.inf)
        coefficient_scale = max(
            mp.mpf(1),
            *(
                abs(mp.mpf(value.numerator) / value.denominator)
                for value in coefficients
            ),
        )
        root_residuals = tuple(
            _normalized_residual(
                _evaluate_polynomial(coefficients, root),
                coefficient_scale * max(mp.mpf(1), abs(root) ** 4),
            )
            for root in roots
        )
        maximum_root_residual = max(root_residuals, default=mp.mpf("0"))
        vieta_residual = _vieta_residual(roots, coefficients)
        residual_limit = mp.power(10, -(precision_digits - 20))

        reasons: list[str] = []
        if any(multiplicity != 1 for multiplicity in multiplicities):
            reasons.append("repeated-root")
        if minimum_separation <= required_separation:
            reasons.append("root-separation")
        if minimum_horizon_separation <= required_separation:
            reasons.append("horizon-root-separation")
        if minimum_initial_separation <= required_separation:
            reasons.append("initial-root-separation")
        if topology_id == "boundary":
            reasons.append("unsupported-boundary-topology")
        elif topology_id != "I":
            reasons.append(f"unsupported-topology-{topology_id}")
        elif motion_branch != "Ib":
            reasons.append("unsupported-motion-branch")
        if roots and minimum_derivative <= required_separation:
            reasons.append("root-derivative-margin")
        if maximum_root_residual >= residual_limit:
            reasons.append("root-residual")
        if vieta_residual is not None and vieta_residual >= residual_limit:
            reasons.append("vieta-residual")
        if stationary_real_root_count is None:
            reasons.append("stationary-sign-margin")
        elif stationary_real_root_count != sum(multiplicities):
            reasons.append("stationary-root-count")

        decision = (
            _TopologyDecision.FALLBACK
            if reasons
            else _TopologyDecision.ELIGIBLE_FOR_FURTHER_ANALYSIS
        )
        return _TopologyReport(
            name=case.name,
            precision_digits=precision_digits,
            normalized_spin=spin,
            normalized_impact=impact,
            normalized_carter=carter,
            polynomial_coefficients=coefficients,
            horizon_radii=horizons,
            initial_radius=initial_radius,
            initial_radial_sign=case.initial_radial_sign,
            ordered_real_roots=roots,
            ordered_real_root_brackets=root_brackets,
            real_root_bracket_signs=root_bracket_signs,
            real_root_multiplicities=multiplicities,
            complex_root_count=complex_root_count,
            stationary_points=stationary_points,
            stationary_values=stationary_values,
            topology_id=topology_id,
            motion_branch=motion_branch,
            turn_sequence=turn_sequence,
            decision=decision,
            fallback_reasons=tuple(dict.fromkeys(reasons)),
            minimum_root_separation=minimum_separation,
            required_root_separation=required_separation,
            minimum_horizon_root_separation=minimum_horizon_separation,
            minimum_initial_root_separation=minimum_initial_separation,
            minimum_root_derivative=minimum_derivative,
            maximum_root_residual=maximum_root_residual,
            vieta_residual=vieta_residual,
            stationary_real_root_count=stationary_real_root_count,
            polar_margin=_polar_margin(case, precision_digits),
        )


def _topology_precision_delta(
    low_reports: tuple[_TopologyReport, ...],
    high_reports: tuple[_TopologyReport, ...],
) -> mp.mpf:
    deltas: list[mp.mpf] = []
    with mp.workdps(HIGH_PRECISION_DIGITS + 20):
        for low, high in zip(low_reports, high_reports, strict=True):
            discrete_low = (
                low.topology_id,
                low.motion_branch,
                low.turn_sequence,
                low.decision,
                low.fallback_reasons,
                low.real_root_multiplicities,
                low.real_root_bracket_signs,
                low.complex_root_count,
                low.stationary_real_root_count,
            )
            discrete_high = (
                high.topology_id,
                high.motion_branch,
                high.turn_sequence,
                high.decision,
                high.fallback_reasons,
                high.real_root_multiplicities,
                high.real_root_bracket_signs,
                high.complex_root_count,
                high.stationary_real_root_count,
            )
            if discrete_low != discrete_high:
                raise AssertionError(f"{low.name}: topology changed with precision")
            vectors = (
                (low.ordered_real_roots, high.ordered_real_roots),
                (low.stationary_points, high.stationary_points),
                (low.stationary_values, high.stationary_values),
                (low.horizon_radii, high.horizon_radii),
                (
                    tuple(
                        value
                        for bracket in low.ordered_real_root_brackets
                        for value in bracket
                    ),
                    tuple(
                        value
                        for bracket in high.ordered_real_root_brackets
                        for value in bracket
                    ),
                ),
            )
            for low_values, high_values in vectors:
                if len(low_values) != len(high_values):
                    raise AssertionError(f"{low.name}: continuous vector changed shape")
                deltas.extend(
                    _normalized_residual(low_value - high_value, high_value)
                    for low_value, high_value in zip(
                        low_values, high_values, strict=True
                    )
                )
            for low_value, high_value in (
                (
                    low.minimum_root_separation,
                    high.minimum_root_separation,
                ),
                (
                    low.minimum_horizon_root_separation,
                    high.minimum_horizon_root_separation,
                ),
                (
                    low.minimum_initial_root_separation,
                    high.minimum_initial_root_separation,
                ),
            ):
                if mp.isfinite(low_value) and mp.isfinite(high_value):
                    deltas.append(
                        _normalized_residual(low_value - high_value, high_value)
                    )
        return max(deltas, default=mp.mpf("0"))


@cache
def _build_topology_certificate() -> _TopologyCertificate:
    """Rebuild the exact named topology corpus at 120 and 180 digits."""

    cases = _named_topology_cases()
    low_reports = tuple(
        _build_topology_report(case, LOW_PRECISION_DIGITS) for case in cases
    )
    high_reports = tuple(
        _build_topology_report(case, HIGH_PRECISION_DIGITS) for case in cases
    )
    false_acceptance_count = 0
    for case, report in zip(cases, high_reports, strict=True):
        actual = (
            report.topology_id,
            report.motion_branch,
            report.turn_sequence,
            report.decision,
        )
        expected = (
            case.expected_topology,
            case.expected_motion_branch,
            case.expected_turn_sequence,
            case.expected_decision,
        )
        if actual != expected:
            raise AssertionError(f"{case.name}: expected {expected}, got {actual}")
        if (
            case.expected_decision is _TopologyDecision.FALLBACK
            and report.decision is _TopologyDecision.ELIGIBLE_FOR_FURTHER_ANALYSIS
        ):
            false_acceptance_count += 1
    maximum_delta = _topology_precision_delta(low_reports, high_reports)
    with mp.workdps(HIGH_PRECISION_DIGITS):
        if maximum_delta >= mp.power(10, -REQUIRED_STABLE_DIGITS):
            raise AssertionError("topology certificate lost its stable-digit budget")
    return _TopologyCertificate(
        low_precision_digits=LOW_PRECISION_DIGITS,
        high_precision_digits=HIGH_PRECISION_DIGITS,
        required_stable_digits=REQUIRED_STABLE_DIGITS,
        maximum_normalized_delta=maximum_delta,
        false_acceptance_count=false_acceptance_count,
        reports=high_reports,
    )


def _validate_carlson_arguments(
    kind: _CarlsonKind,
    arguments: tuple[mp.mpf, ...],
) -> None:
    expected_count = (
        2 if kind is _CarlsonKind.RC else (4 if kind is _CarlsonKind.RJ else 3)
    )
    if len(arguments) != expected_count or any(
        not mp.isfinite(value) for value in arguments
    ):
        raise _UnsupportedCarlsonError(f"{kind.value} requires finite real arguments")
    if kind is _CarlsonKind.RC:
        x, y = arguments
        if x < 0 or y <= 0:
            reason = "principal-value policy" if y < 0 else "positive-real domain"
            raise _UnsupportedCarlsonError(f"RC input requires a {reason}")
        return
    leading = arguments[:3]
    if any(value < 0 for value in leading) or sum(value == 0 for value in leading) > 1:
        raise _UnsupportedCarlsonError(
            f"{kind.value} is outside the one-zero positive-real domain"
        )
    if kind is _CarlsonKind.RD and arguments[2] <= 0:
        raise _UnsupportedCarlsonError("RD requires z > 0")
    if kind is _CarlsonKind.RJ:
        p = arguments[3]
        if p < 0:
            raise _UnsupportedCarlsonError(
                "negative p requires an explicit principal-value policy"
            )
        if p == 0:
            raise _UnsupportedCarlsonError("RJ requires p > 0")
        scale = max(arguments)
        if p / scale < mp.mpf(_CARLSON_PARAMETER_LIMIT):
            raise _UnsupportedCarlsonError("RJ p is below the conditioning limit")


def _parse_real_arguments(argument_strings: tuple[str, ...]) -> tuple[mp.mpf, ...]:
    try:
        return tuple(mp.mpf(value) for value in argument_strings)
    except (TypeError, ValueError) as error:
        raise _UnsupportedCarlsonError(
            "Carlson inputs must be canonical finite real values"
        ) from error


def _as_definition_arguments(
    kind: _CarlsonKind,
    arguments: tuple[mp.mpf, ...],
) -> tuple[tuple[mp.mpf, mp.mpf, mp.mpf], mp.mpf | None]:
    if kind is _CarlsonKind.RF:
        return (arguments[0], arguments[1], arguments[2]), None
    if kind is _CarlsonKind.RC:
        return (arguments[0], arguments[1], arguments[1]), None
    if kind is _CarlsonKind.RD:
        return (arguments[0], arguments[1], arguments[2]), arguments[2]
    return (arguments[0], arguments[1], arguments[2]), arguments[3]


def _definition_integrand(
    u: mp.mpf,
    leading: tuple[mp.mpf, mp.mpf, mp.mpf],
    pole: mp.mpf | None,
) -> mp.mpf:
    if u == 1:
        return mp.mpf(1) if pole is None else mp.mpf(0)
    zero_count = sum(value == 0 for value in leading)
    if u == 0:
        if zero_count == 0:
            return mp.mpf(0)
        nonzero_product = mp.fprod(value for value in leading if value != 0)
        if pole is None:
            return 1 / mp.sqrt(nonzero_product)
        return 3 / (pole * mp.sqrt(nonzero_product))
    one_minus_u = 1 - u
    t = (u / one_minus_u) ** 2
    derivative = 2 * u / one_minus_u**3
    radical = mp.sqrt(mp.fprod(t + value for value in leading))
    if pole is None:
        return derivative / (2 * radical)
    return 3 * derivative / (2 * (t + pole) * radical)


@cache
def _carlson_definition(
    kind: _CarlsonKind,
    argument_strings: tuple[str, ...],
    precision_digits: int,
) -> mp.mpf:
    """Evaluate one positive-real Carlson function from its defining integral."""

    with mp.workdps(precision_digits + 30):
        arguments = _parse_real_arguments(argument_strings)
        _validate_carlson_arguments(kind, arguments)
        scale = max(arguments)
        normalized = tuple(value / scale for value in arguments)
        leading, pole = _as_definition_arguments(kind, normalized)
        integral = mp.quad(
            lambda u: _definition_integrand(u, leading, pole),
            [0, mp.mpf("0.125"), mp.mpf("0.5"), mp.mpf("0.875"), 1],
        )
        degree = mp.mpf("-0.5") if pole is None else mp.mpf("-1.5")
        return integral * mp.power(scale, degree)


def _evaluate_library(
    kind: _CarlsonKind,
    arguments: tuple[mp.mpf, ...],
) -> mp.mpf:
    _validate_carlson_arguments(kind, arguments)
    if kind is _CarlsonKind.RF:
        return mp.elliprf(*arguments)
    if kind is _CarlsonKind.RC:
        return mp.elliprc(*arguments)
    if kind is _CarlsonKind.RD:
        return mp.elliprd(*arguments)
    return mp.elliprj(*arguments)


def _library_from_strings(
    kind: _CarlsonKind,
    argument_strings: tuple[str, ...],
    precision_digits: int,
) -> mp.mpf:
    with mp.workdps(precision_digits + 30):
        return _evaluate_library(kind, _parse_real_arguments(argument_strings))


def _homogeneity_residual(
    kind: _CarlsonKind,
    argument_strings: tuple[str, ...],
    *,
    scale: str,
    degree: str,
) -> mp.mpf:
    with mp.workdps(HIGH_PRECISION_DIGITS):
        arguments = _parse_real_arguments(argument_strings)
        scale_value = mp.mpf(scale)
        base = _evaluate_library(kind, arguments)
        scaled = _evaluate_library(
            kind,
            tuple(scale_value * value for value in arguments),
        )
        expected = mp.power(scale_value, mp.mpf(degree)) * base
        return _normalized_residual(scaled - expected, expected)


def _rf_duplication_residual() -> mp.mpf:
    with mp.workdps(HIGH_PRECISION_DIGITS):
        x, y, z = mp.mpf("0.5"), mp.mpf("1.25"), mp.mpf("2.75")
        root_x, root_y, root_z = mp.sqrt(x), mp.sqrt(y), mp.sqrt(z)
        duplication = root_x * root_y + root_y * root_z + root_z * root_x
        reduced = tuple((value + duplication) / 4 for value in (x, y, z))
        left = mp.elliprf(x, y, z)
        right = mp.elliprf(*reduced)
        return _normalized_residual(left - right, left)


def _rd_duplication_residual(*, include_additive_term: bool) -> mp.mpf:
    with mp.workdps(HIGH_PRECISION_DIGITS):
        x, y, z = mp.mpf("0.5"), mp.mpf("1.25"), mp.mpf("2.75")
        root_x, root_y, root_z = mp.sqrt(x), mp.sqrt(y), mp.sqrt(z)
        duplication = root_x * root_y + root_y * root_z + root_z * root_x
        reduced = tuple((value + duplication) / 4 for value in (x, y, z))
        left = mp.elliprd(x, y, z)
        right = mp.elliprd(*reduced) / 4
        if include_additive_term:
            right += 3 / (root_z * (z + duplication))
        return _normalized_residual(left - right, left)


def _rj_duplication_residual(*, include_rc_correction: bool) -> mp.mpf:
    with mp.workdps(HIGH_PRECISION_DIGITS):
        x, y, z, p = (
            mp.mpf("0.5"),
            mp.mpf("1.25"),
            mp.mpf("2.75"),
            mp.mpf("1.75"),
        )
        root_x, root_y, root_z = mp.sqrt(x), mp.sqrt(y), mp.sqrt(z)
        duplication = root_x * root_y + root_y * root_z + root_z * root_x
        reduced = tuple((value + duplication) / 4 for value in (x, y, z, p))
        left = mp.elliprj(x, y, z, p)
        right = mp.elliprj(*reduced) / 4
        if include_rc_correction:
            alpha = p * (root_x + root_y + root_z) + root_x * root_y * root_z
            beta = mp.sqrt(p) * (p + duplication)
            right += 3 * mp.elliprc(alpha**2, beta**2)
        return _normalized_residual(left - right, left)


def _rc_duplication_residual() -> mp.mpf:
    with mp.workdps(HIGH_PRECISION_DIGITS):
        x, y = mp.mpf("0.5"), mp.mpf("1.25")
        duplication = 2 * mp.sqrt(x * y) + y
        left = mp.elliprc(x, y)
        right = mp.elliprc((x + duplication) / 4, (y + duplication) / 4)
        return _normalized_residual(left - right, left)


def _carlson_difference_report(
    *,
    x: str,
    y: str,
    p: str,
    precision_digits: int,
) -> _CarlsonDifferenceReport:
    with mp.workdps(precision_digits + 30):
        x_value, y_value, p_value = _parse_real_arguments((x, y, p))
        if (
            any(not mp.isfinite(value) for value in (x_value, y_value, p_value))
            or x_value < 0
            or y_value <= 0
            or p_value <= 0
            or p_value == y_value
        ):
            raise _UnsupportedCarlsonError(
                "RC-difference identity requires x >= 0 and distinct y,p > 0"
            )
        first = mp.elliprc(x_value, y_value)
        second = mp.elliprc(x_value, p_value)
        difference = first - second
        cancellation_ratio = (abs(first) + abs(second)) / abs(difference)
        identity = 3 * difference / (p_value - y_value)
        direct = mp.elliprj(x_value, y_value, y_value, p_value)
        residual = _normalized_residual(identity - direct, direct)
        decision = (
            _CarlsonDecision.FALLBACK
            if cancellation_ratio > mp.mpf(_CARLSON_CANCELLATION_LIMIT)
            else _CarlsonDecision.ELIGIBLE_FOR_FURTHER_ANALYSIS
        )
        return _CarlsonDifferenceReport(
            precision_digits=precision_digits,
            cancellation_ratio=cancellation_ratio,
            normalized_identity_residual=residual,
            decision=decision,
        )


def _definition_cases() -> tuple[tuple[_CarlsonKind, tuple[str, ...]], ...]:
    return (
        (_CarlsonKind.RF, ("0.5", "1.25", "2.75")),
        (_CarlsonKind.RF, ("0", "0.25", "1")),
        (
            _CarlsonKind.RF,
            (
                "0.9999999999999999999999999999999999999999",
                "1",
                "1.0000000000000000000000000000000000000001",
            ),
        ),
        (_CarlsonKind.RF, ("5e-80", "1.25e-79", "2.75e-79")),
        (_CarlsonKind.RC, ("2.25", "2")),
        (_CarlsonKind.RC, ("0", "0.25")),
        (_CarlsonKind.RD, ("0.5", "1.25", "2.75")),
        (_CarlsonKind.RD, ("0", "0.25", "1")),
        (_CarlsonKind.RJ, ("0.5", "1.25", "2.75", "1.75")),
        (_CarlsonKind.RJ, ("0", "0.25", "1", "0.5")),
        (_CarlsonKind.RJ, ("0.5", "1.25", "2.75", "1e-20")),
    )


def _definition_vector(precision_digits: int) -> tuple[mp.mpf, ...]:
    values: list[mp.mpf] = []
    for kind, arguments in _definition_cases():
        values.append(_carlson_definition(kind, arguments, precision_digits))
        values.append(_library_from_strings(kind, arguments, precision_digits))
    return tuple(values)


def _carlson_precision_delta() -> mp.mpf:
    low = _definition_vector(LOW_PRECISION_DIGITS)
    high = _definition_vector(HIGH_PRECISION_DIGITS)
    with mp.workdps(HIGH_PRECISION_DIGITS + 20):
        return max(
            _normalized_residual(low_value - high_value, high_value)
            for low_value, high_value in zip(low, high, strict=True)
        )


def _maximum_definition_residual() -> mp.mpf:
    with mp.workdps(HIGH_PRECISION_DIGITS + 20):
        residuals = []
        for kind, arguments in _definition_cases():
            direct = _carlson_definition(kind, arguments, HIGH_PRECISION_DIGITS)
            library = _library_from_strings(kind, arguments, HIGH_PRECISION_DIGITS)
            residuals.append(_normalized_residual(direct - library, library))
        return max(residuals)


def _positive_diagonal_bound_residual(
    kind: _CarlsonKind,
    arguments: tuple[mp.mpf, ...],
) -> mp.mpf:
    """Check monotonic diagonal bounds implied by the positive definitions."""

    value = _evaluate_library(kind, arguments)
    degree = (
        mp.mpf("-0.5") if kind in (_CarlsonKind.RF, _CarlsonKind.RC) else mp.mpf("-1.5")
    )
    lower = mp.power(max(arguments), degree)
    upper = mp.power(min(arguments), degree)
    violation = max(lower - value, value - upper, mp.mpf("0"))
    return _normalized_residual(violation, value)


def _carlson_identity_residuals() -> tuple[mp.mpf, ...]:
    with mp.workdps(HIGH_PRECISION_DIGITS):
        x, y, z, p = (
            mp.mpf("0.5"),
            mp.mpf("1.25"),
            mp.mpf("2.75"),
            mp.mpf("1.75"),
        )
        diagonal = mp.mpf("2")
        rf = mp.elliprf(x, y, z)
        rj = mp.elliprj(x, y, z, p)
        residuals = [
            _homogeneity_residual(
                _CarlsonKind.RF,
                ("0.5", "1.25", "2.75"),
                scale="3.5",
                degree="-0.5",
            ),
            _homogeneity_residual(
                _CarlsonKind.RC,
                ("0.5", "1.25"),
                scale="3.5",
                degree="-0.5",
            ),
            _homogeneity_residual(
                _CarlsonKind.RD,
                ("0.5", "1.25", "2.75"),
                scale="3.5",
                degree="-1.5",
            ),
            _homogeneity_residual(
                _CarlsonKind.RJ,
                ("0.5", "1.25", "2.75", "1.75"),
                scale="3.5",
                degree="-1.5",
            ),
            _homogeneity_residual(
                _CarlsonKind.RF,
                ("0.5", "1.25", "2.75"),
                scale="1e-80",
                degree="-0.5",
            ),
            _homogeneity_residual(
                _CarlsonKind.RJ,
                ("0.5", "1.25", "2.75", "1.75"),
                scale="1e80",
                degree="-1.5",
            ),
            _rf_duplication_residual(),
            _rd_duplication_residual(include_additive_term=True),
            _rj_duplication_residual(include_rc_correction=True),
            _rc_duplication_residual(),
            _normalized_residual(mp.elliprf(x, y, y) - mp.elliprc(x, y), rf),
            _normalized_residual(mp.elliprj(x, y, z, z) - mp.elliprd(x, y, z), rj),
            _normalized_residual(
                mp.elliprf(diagonal, diagonal, diagonal) - 1 / mp.sqrt(diagonal),
                1 / mp.sqrt(diagonal),
            ),
            _normalized_residual(
                mp.elliprd(diagonal, diagonal, diagonal)
                - mp.power(diagonal, mp.mpf("-1.5")),
                mp.power(diagonal, mp.mpf("-1.5")),
            ),
            _normalized_residual(mp.elliprc(0, mp.mpf("0.25")) - mp.pi, mp.pi),
            _normalized_residual(mp.elliprc(mp.mpf("2.25"), 2) - mp.log(2), mp.log(2)),
            _normalized_residual(mp.elliprf(x, y, z) - mp.elliprf(z, x, y), rf),
            _normalized_residual(mp.elliprj(x, y, z, p) - mp.elliprj(z, x, y, p), rj),
            _positive_diagonal_bound_residual(_CarlsonKind.RF, (x, y, z)),
            _positive_diagonal_bound_residual(_CarlsonKind.RC, (x, y)),
            _positive_diagonal_bound_residual(_CarlsonKind.RD, (x, y, z)),
            _positive_diagonal_bound_residual(_CarlsonKind.RJ, (x, y, z, p)),
        ]
        cyclic = mp.elliprd(x, y, z) + mp.elliprd(y, z, x) + mp.elliprd(z, x, y)
        cyclic_expected = 3 / mp.sqrt(x * y * z)
        residuals.append(
            _normalized_residual(cyclic - cyclic_expected, cyclic_expected)
        )
        difference_report = _carlson_difference_report(
            x="0.5",
            y="1.25",
            p="1.75",
            precision_digits=HIGH_PRECISION_DIGITS,
        )
        residuals.append(difference_report.normalized_identity_residual)
        return tuple(residuals)


def _kerr_radial_reduction_residual() -> mp.mpf:
    report = _build_topology_certificate().report("class-i-ib-inward")
    root_1, _, _, root_4 = report.ordered_real_roots
    with mp.workdps(HIGH_PRECISION_DIGITS + 20):
        endpoint = report.initial_radius
        extent = endpoint - root_4
        gap_a = root_4 - root_1
        gap_b = root_4 - report.ordered_real_roots[1]
        gap_c = root_4 - report.ordered_real_roots[2]
        x = 1 / extent + 1 / gap_a
        y = 1 / extent + 1 / gap_b
        z = 1 / extent + 1 / gap_c
        rf = _carlson_definition(
            _CarlsonKind.RF,
            tuple(mp.nstr(value, HIGH_PRECISION_DIGITS) for value in (x, y, z)),
            HIGH_PRECISION_DIGITS,
        )
        reduced = 2 * rf / mp.sqrt(gap_a * gap_b * gap_c)

        def radial_integrand(parameter: mp.mpf) -> mp.mpf:
            u = extent * parameter**2
            return (
                2 * mp.sqrt(extent) / mp.sqrt((u + gap_a) * (u + gap_b) * (u + gap_c))
            )

        direct = mp.quad(radial_integrand, [0, mp.mpf("0.5"), 1])

        pole_gap = mp.mpf("2")
        pole_argument = 1 / extent + 1 / pole_gap
        rj = _carlson_definition(
            _CarlsonKind.RJ,
            tuple(
                mp.nstr(value, HIGH_PRECISION_DIGITS)
                for value in (x, y, z, pole_argument)
            ),
            HIGH_PRECISION_DIGITS,
        )
        reduced_third_kind = reduced / pole_gap - 2 * rj / (
            3 * pole_gap**2 * mp.sqrt(gap_a * gap_b * gap_c)
        )

        def third_kind_integrand(parameter: mp.mpf) -> mp.mpf:
            u = extent * parameter**2
            return radial_integrand(parameter) / (u + pole_gap)

        direct_third_kind = mp.quad(
            third_kind_integrand,
            [0, mp.mpf("0.5"), 1],
        )
        return max(
            _normalized_residual(reduced - direct, direct),
            _normalized_residual(
                reduced_third_kind - direct_third_kind,
                direct_third_kind,
            ),
        )


@cache
def _build_carlson_certificate() -> _CarlsonCertificate:
    """Build the positive-real definition and identity certificate."""

    with mp.workdps(HIGH_PRECISION_DIGITS + 20):
        maximum_definition = _maximum_definition_residual()
        identity_residuals = _carlson_identity_residuals()
        maximum_identity = max(identity_residuals)
        maximum_delta = _carlson_precision_delta()
        x, y, z = mp.mpf("0.5"), mp.mpf("1.25"), mp.mpf("2.75")
        rd_xy = _normalized_residual(
            mp.elliprd(x, y, z) - mp.elliprd(y, x, z),
            mp.elliprd(x, y, z),
        )
        rd_full = _normalized_residual(
            mp.elliprd(x, y, z) - mp.elliprd(z, y, x),
            mp.elliprd(x, y, z),
        )
        kerr_reduction = _kerr_radial_reduction_residual()
        guard = mp.power(10, -REQUIRED_STABLE_DIGITS)
        if maximum_delta >= guard:
            raise AssertionError("Carlson certificate lost its stable-digit budget")
        return _CarlsonCertificate(
            low_precision_digits=LOW_PRECISION_DIGITS,
            high_precision_digits=HIGH_PRECISION_DIGITS,
            required_stable_digits=REQUIRED_STABLE_DIGITS,
            maximum_definition_residual=maximum_definition,
            maximum_identity_residual=maximum_identity,
            maximum_normalized_delta=maximum_delta,
            rd_xy_symmetry_residual=rd_xy,
            rd_full_permutation_delta=rd_full,
            kerr_radial_reduction_residual=kerr_reduction,
        )


def _scientific(value: mp.mpf, digits: int = 12) -> str:
    with mp.workdps(HIGH_PRECISION_DIGITS):
        return mp.nstr(value, digits, strip_zeros=False)


def run() -> None:
    topology = _build_topology_certificate()
    carlson = _build_carlson_certificate()
    print(f"python={platform.python_version()}")
    print(
        "precision="
        f"{LOW_PRECISION_DIGITS},{HIGH_PRECISION_DIGITS} "
        f"stable_digits>={REQUIRED_STABLE_DIGITS}"
    )
    print(
        "topology.maximum_normalized_delta="
        f"{_scientific(topology.maximum_normalized_delta)}"
    )
    print(f"topology.false_acceptance_count={topology.false_acceptance_count}")
    for report in topology.reports:
        reasons = ",".join(report.fallback_reasons) or "none"
        print(
            f"topology.case={report.name} class={report.topology_id} "
            f"branch={report.motion_branch or 'none'} "
            f"decision={report.decision.value} reasons={reasons}"
        )
        vieta = (
            _scientific(report.vieta_residual)
            if report.vieta_residual is not None
            else "not-applicable"
        )
        sign_changes = sum(
            lower_sign * upper_sign < 0
            for lower_sign, upper_sign in report.real_root_bracket_signs
        )
        print(
            f"topology.metrics.{report.name}="
            f"real:{len(report.ordered_real_roots)} "
            f"complex:{report.complex_root_count} "
            f"sign_brackets:{sign_changes} "
            f"root_separation:{_scientific(report.minimum_root_separation)} "
            "horizon_separation:"
            f"{_scientific(report.minimum_horizon_root_separation)} "
            f"initial_separation:{_scientific(report.minimum_initial_root_separation)} "
            f"root_derivative:{_scientific(report.minimum_root_derivative)} "
            f"root_residual:{_scientific(report.maximum_root_residual)} "
            f"vieta:{vieta}"
        )
    print(
        "carlson.maximum_definition_residual="
        f"{_scientific(carlson.maximum_definition_residual)}"
    )
    print(
        "carlson.maximum_identity_residual="
        f"{_scientific(carlson.maximum_identity_residual)}"
    )
    print(
        "carlson.maximum_normalized_delta="
        f"{_scientific(carlson.maximum_normalized_delta)}"
    )
    print(
        "carlson.kerr_radial_reduction_residual="
        f"{_scientific(carlson.kerr_radial_reduction_residual)}"
    )
    print("RESULT=PASS")

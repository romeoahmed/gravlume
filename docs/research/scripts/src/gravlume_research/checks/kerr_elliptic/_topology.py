"""Certify exact rational Kerr radial-root topology and fallback decisions.

Real roots come only from SymPy exact isolation; a numerically small imaginary
part never becomes a real root.  This private proof module is neither a
production solver nor a WGSL accuracy claim.

Primary sources:

* Kerr exterior root taxonomy: https://doi.org/10.1103/PhysRevD.101.044032
"""

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import cache
from itertools import pairwise

import mpmath as mp
import sympy as sp

from ..._precision import _SCIENTIFIC_PRECISION, _PrecisionEvidence

_ROOT_SEPARATION_LIMIT = "1e-40"


class _UnsupportedTopologyError(ValueError):
    """The radial case is inconsistent with this certified research slice."""


class _TopologyDecision(Enum):
    ELIGIBLE_FOR_FURTHER_ANALYSIS = "eligible-for-further-analysis"
    FALLBACK = "fallback"


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
                or self.precision_digits < _SCIENTIFIC_PRECISION.required_digits
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

    precision: _PrecisionEvidence
    false_acceptance_count: int
    reports: tuple[_TopologyReport, ...]

    def report(self, name: str) -> _TopologyReport:
        for report in self.reports:
            if report.name == name:
                return report
        raise KeyError(f"unknown topology report {name!r}")


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
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits + 20):
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
        _build_topology_report(case, _SCIENTIFIC_PRECISION.low_digits) for case in cases
    )
    high_reports = tuple(
        _build_topology_report(case, _SCIENTIFIC_PRECISION.high_digits)
        for case in cases
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
    return _TopologyCertificate(
        precision=_SCIENTIFIC_PRECISION.certify_delta(
            maximum_delta,
            subject="topology certificate",
        ),
        false_acceptance_count=false_acceptance_count,
        reports=high_reports,
    )

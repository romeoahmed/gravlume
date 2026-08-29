"""Certify conservative fallback guards for a future Kerr accelerator.

The named corpus uses exact rational Kerr inputs and an outward-rounded,
explicitly sequenced binary32 interval graph.  It can prove that a margin is
too small for admission; it deliberately does not prove that an eligible case
is accurate in an arbitrary WGSL implementation or production solver.

Primary specification source:
https://www.w3.org/TR/WGSL/#floating-point-evaluation
"""

import platform
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

import mpmath as mp

from .._interval_f32 import IntervalF32, exact_fraction, interval_contains

LOW_PRECISION_DIGITS = 120
HIGH_PRECISION_DIGITS = 180
REQUIRED_STABLE_DIGITS = 80

_AXIS_CONDITION = "axis-chart-denominator"
_EXTREMALITY_CONDITION = "extremality-gap"
_HORIZON_ROOT_CONDITION = "horizon-root-separation-squared"
_HORIZON_POLYNOMIAL_CONDITION = "horizon-polynomial-separation"
_RADIAL_ROOT_CONDITION = "radial-root-separation"


class _SupportDecision(Enum):
    ELIGIBLE_FOR_FURTHER_ANALYSIS = "eligible-for-further-analysis"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True, kw_only=True)
class _ConditionGuard:
    """One positive exact margin and its strict-binary32 enclosure."""

    name: str
    margin: Fraction
    interval_lower: Fraction
    interval_upper: Fraction
    absolute_error_bound: Fraction

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a support condition must have a name")
        if self.margin <= 0:
            raise ValueError(f"{self.name}: exact margin must be positive")
        if not self.interval_lower <= self.margin <= self.interval_upper:
            raise ValueError(f"{self.name}: interval does not enclose its exact margin")
        required_error = max(
            abs(self.margin - self.interval_lower),
            abs(self.interval_upper - self.margin),
        )
        if self.absolute_error_bound < required_error:
            raise ValueError(f"{self.name}: error bound does not enclose its interval")

    @property
    def certified(self) -> bool:
        """Return whether the positive margin strictly exceeds all uncertainty."""

        return self.interval_lower > 0 and self.margin > self.absolute_error_bound


@dataclass(frozen=True, slots=True, kw_only=True)
class _SupportReport:
    """Conditioning-only decision for one named exact input."""

    name: str
    conditions: tuple[_ConditionGuard, ...]

    def __post_init__(self) -> None:
        names = tuple(condition.name for condition in self.conditions)
        if not self.name or not names or len(names) != len(set(names)):
            raise ValueError("a support report needs a name and unique conditions")

    def condition(self, name: str) -> _ConditionGuard:
        for condition in self.conditions:
            if condition.name == name:
                return condition
        raise KeyError(f"unknown condition {name!r} for case {self.name!r}")

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        return tuple(
            condition.name for condition in self.conditions if not condition.certified
        )

    @property
    def decision(self) -> _SupportDecision:
        return _classify_support(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class _SupportCase:
    """Pre-registered rational input and expected conservative decision."""

    name: str
    mass: Fraction
    spin: Fraction
    radius: Fraction
    sin_theta_squared: Fraction
    radial_roots: tuple[Fraction, Fraction]
    expected_failed_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        inner_root, outer_root = self.radial_roots
        if (
            not self.name
            or self.mass <= 0
            or abs(self.spin) >= self.mass
            or self.radius <= self.mass
            or not 0 < self.sin_theta_squared <= 1
            or inner_root <= 0
            or outer_root <= inner_root
        ):
            raise ValueError(f"{self.name!r} is not a valid subextremal exterior case")


@dataclass(frozen=True, slots=True, kw_only=True)
class _SupportCertificate:
    """Exact corpus decisions plus a 120/180-digit reproducibility witness."""

    low_precision_digits: int
    high_precision_digits: int
    required_stable_digits: int
    maximum_normalized_delta: mp.mpf
    false_acceptance_count: int
    reports: tuple[_SupportReport, ...]

    def report(self, name: str) -> _SupportReport:
        for report in self.reports:
            if report.name == name:
                return report
        raise KeyError(f"unknown support report {name!r}")


def _classify_support(report: _SupportReport) -> _SupportDecision:
    """Require every guard to be strictly outside its absolute error bound."""

    if report.failed_conditions:
        return _SupportDecision.FALLBACK
    return _SupportDecision.ELIGIBLE_FOR_FURTHER_ANALYSIS


def _named_support_cases() -> tuple[_SupportCase, ...]:
    one = Fraction(1)
    regular_spin = Fraction(3, 5)
    regular_radius = Fraction(5)
    regular_axis = Fraction(3, 4)
    regular_roots = (Fraction(3), Fraction(7))
    near_axis = Fraction(3, 2 * 10**70)
    near_horizon_radius = Fraction(9, 5) + Fraction(3, 2 * 10**60)
    near_extreme_spin = one - Fraction(3, 2 * 10**60)
    near_double_roots = (
        Fraction(4),
        Fraction(4) + Fraction(3, 2 * 10**30),
    )
    return (
        _SupportCase(
            name="regular-control",
            mass=one,
            spin=regular_spin,
            radius=regular_radius,
            sin_theta_squared=regular_axis,
            radial_roots=regular_roots,
            expected_failed_conditions=(),
        ),
        _SupportCase(
            name="near-axis",
            mass=one,
            spin=regular_spin,
            radius=regular_radius,
            sin_theta_squared=near_axis,
            radial_roots=regular_roots,
            expected_failed_conditions=(_AXIS_CONDITION,),
        ),
        _SupportCase(
            name="near-horizon",
            mass=one,
            spin=regular_spin,
            radius=near_horizon_radius,
            sin_theta_squared=regular_axis,
            radial_roots=regular_roots,
            expected_failed_conditions=(_HORIZON_POLYNOMIAL_CONDITION,),
        ),
        _SupportCase(
            name="near-extremality",
            mass=one,
            spin=near_extreme_spin,
            radius=Fraction(3),
            sin_theta_squared=regular_axis,
            radial_roots=regular_roots,
            expected_failed_conditions=(
                _EXTREMALITY_CONDITION,
                _HORIZON_ROOT_CONDITION,
            ),
        ),
        _SupportCase(
            name="near-radial-degeneracy",
            mass=one,
            spin=regular_spin,
            radius=regular_radius,
            sin_theta_squared=regular_axis,
            radial_roots=near_double_roots,
            expected_failed_conditions=(_RADIAL_ROOT_CONDITION,),
        ),
    )


def _interval(value: Fraction) -> IntervalF32:
    interval = IntervalF32.rational(value.numerator, value.denominator)
    if not interval_contains(interval, value):
        raise AssertionError(f"binary32 input interval lost exact value {value}")
    return interval


def _condition(
    name: str,
    margin: Fraction,
    interval: IntervalF32,
) -> _ConditionGuard:
    if not interval_contains(interval, margin):
        raise AssertionError(f"{name}: arithmetic interval lost exact margin")
    lower = exact_fraction(interval.lower)
    upper = exact_fraction(interval.upper)
    error_bound = max(abs(margin - lower), abs(upper - margin))
    return _ConditionGuard(
        name=name,
        margin=margin,
        interval_lower=lower,
        interval_upper=upper,
        absolute_error_bound=error_bound,
    )


def _build_support_report(case: _SupportCase) -> _SupportReport:
    mass = _interval(case.mass)
    spin = _interval(abs(case.spin))
    radius = _interval(case.radius)
    sin_theta_squared = _interval(case.sin_theta_squared)
    inner_root = _interval(case.radial_roots[0])
    outer_root = _interval(case.radial_roots[1])

    mass_squared = mass.square()
    spin_squared = spin.square()
    extremality_gap_interval = mass_squared.sub(spin_squared)
    horizon_root_separation_squared_interval = extremality_gap_interval.scale_rational(
        4, 1
    )
    horizon_polynomial_interval = (
        radius.sub(mass).square().sub(extremality_gap_interval)
    )
    radial_root_separation_interval = outer_root.sub(inner_root)

    extremality_gap = case.mass**2 - case.spin**2
    horizon_root_separation_squared = 4 * extremality_gap
    horizon_polynomial = (case.radius - case.mass) ** 2 - extremality_gap
    radial_root_separation = case.radial_roots[1] - case.radial_roots[0]
    conditions = (
        _condition(
            _AXIS_CONDITION,
            case.sin_theta_squared,
            sin_theta_squared,
        ),
        _condition(
            _EXTREMALITY_CONDITION,
            extremality_gap,
            extremality_gap_interval,
        ),
        _condition(
            _HORIZON_ROOT_CONDITION,
            horizon_root_separation_squared,
            horizon_root_separation_squared_interval,
        ),
        _condition(
            _HORIZON_POLYNOMIAL_CONDITION,
            horizon_polynomial,
            horizon_polynomial_interval,
        ),
        _condition(
            _RADIAL_ROOT_CONDITION,
            radial_root_separation,
            radial_root_separation_interval,
        ),
    )
    return _SupportReport(name=case.name, conditions=conditions)


def _mp_fraction(value: Fraction, precision_digits: int) -> mp.mpf:
    with mp.workdps(precision_digits):
        return mp.mpf(value.numerator) / value.denominator


def _precision_delta(reports: tuple[_SupportReport, ...]) -> mp.mpf:
    fractions = tuple(
        value
        for report in reports
        for condition in report.conditions
        for value in (
            condition.margin,
            condition.interval_lower,
            condition.interval_upper,
            condition.absolute_error_bound,
        )
    )
    low = tuple(_mp_fraction(value, LOW_PRECISION_DIGITS) for value in fractions)
    high = tuple(_mp_fraction(value, HIGH_PRECISION_DIGITS) for value in fractions)
    with mp.workdps(HIGH_PRECISION_DIGITS):
        return max(
            abs(low_value - high_value) / max(mp.mpf(1), abs(high_value))
            for low_value, high_value in zip(low, high, strict=True)
        )


def _build_support_certificate() -> _SupportCertificate:
    """Rebuild all exact cases and reject any unexpected admission."""

    cases = _named_support_cases()
    reports = tuple(_build_support_report(case) for case in cases)
    false_acceptance_count = 0
    for case, report in zip(cases, reports, strict=True):
        if report.failed_conditions != case.expected_failed_conditions:
            raise AssertionError(
                f"{case.name}: expected failures {case.expected_failed_conditions}, "
                f"got {report.failed_conditions}"
            )
        if case.expected_failed_conditions and report.decision is not (
            _SupportDecision.FALLBACK
        ):
            false_acceptance_count += 1
    maximum_delta = _precision_delta(reports)
    with mp.workdps(HIGH_PRECISION_DIGITS):
        if maximum_delta >= mp.power(10, -REQUIRED_STABLE_DIGITS):
            raise AssertionError(
                "support certificate lost its required precision-doubling digits"
            )
    return _SupportCertificate(
        low_precision_digits=LOW_PRECISION_DIGITS,
        high_precision_digits=HIGH_PRECISION_DIGITS,
        required_stable_digits=REQUIRED_STABLE_DIGITS,
        maximum_normalized_delta=maximum_delta,
        false_acceptance_count=false_acceptance_count,
        reports=reports,
    )


def _scientific(value: Fraction | mp.mpf, digits: int = 12) -> str:
    with mp.workdps(HIGH_PRECISION_DIGITS):
        converted = (
            mp.mpf(value.numerator) / value.denominator
            if isinstance(value, Fraction)
            else value
        )
        return mp.nstr(converted, digits, strip_zeros=False)


def run() -> None:
    certificate = _build_support_certificate()
    print(f"python={platform.python_version()}")
    print(
        "precision="
        f"{certificate.low_precision_digits},{certificate.high_precision_digits} "
        f"stable_digits>={certificate.required_stable_digits} "
        "maximum_normalized_delta="
        f"{_scientific(certificate.maximum_normalized_delta)}"
    )
    print(f"false_acceptance_count={certificate.false_acceptance_count}")
    for report in certificate.reports:
        failed = ",".join(report.failed_conditions) or "none"
        print(f"case={report.name} decision={report.decision.value} failed={failed}")
        for condition in report.conditions:
            print(
                f"condition.{report.name}.{condition.name}="
                f"margin:{_scientific(condition.margin)} "
                f"error:{_scientific(condition.absolute_error_bound)} "
                f"certified:{str(condition.certified).lower()}"
            )
    print("RESULT=PASS")

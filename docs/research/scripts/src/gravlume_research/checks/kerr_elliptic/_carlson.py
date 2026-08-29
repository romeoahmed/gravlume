"""Certify a positive-real Carlson oracle and one Kerr radial reduction.

Definitions are evaluated independently of mpmath's special-function
implementations, then checked with exact identities and precision doubling.
This private proof module is neither a production solver nor a WGSL accuracy
claim.

Primary sources:

* Carlson definitions and algorithms: https://arxiv.org/abs/math/9409227
* DLMF definitions and duplication: https://dlmf.nist.gov/19.16
  and https://dlmf.nist.gov/19.26
"""

from dataclasses import dataclass
from enum import Enum
from functools import cache

import mpmath as mp

from ..._precision import _SCIENTIFIC_PRECISION, _PrecisionEvidence
from ._topology import _build_topology_certificate

_CARLSON_CANCELLATION_LIMIT = "1e40"
_CARLSON_PARAMETER_LIMIT = "1e-40"


class _UnsupportedCarlsonError(ValueError):
    """The Carlson input is outside this oracle's positive-real domain."""


class _CarlsonDecision(Enum):
    ELIGIBLE_FOR_FURTHER_ANALYSIS = "eligible-for-further-analysis"
    FALLBACK = "fallback"


class _CarlsonKind(Enum):
    RF = "RF"
    RC = "RC"
    RD = "RD"
    RJ = "RJ"


@dataclass(frozen=True, slots=True, kw_only=True)
class _CarlsonDifferenceReport:
    """Condition report for the cancellation-prone RC-difference identity."""

    precision_digits: int
    cancellation_ratio: mp.mpf
    normalized_identity_residual: mp.mpf
    decision: _CarlsonDecision

    def __post_init__(self) -> None:
        if self.precision_digits < _SCIENTIFIC_PRECISION.required_digits:
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

    precision: _PrecisionEvidence
    maximum_definition_residual: mp.mpf
    maximum_identity_residual: mp.mpf
    rd_xy_symmetry_residual: mp.mpf
    rd_full_permutation_delta: mp.mpf
    kerr_radial_reduction_residual: mp.mpf

    def __post_init__(self) -> None:
        """Reject any record that could make the scientific witness false-pass."""

        if self.precision.policy is not _SCIENTIFIC_PRECISION:
            raise AssertionError("Carlson certificate precision contract changed")
        with mp.workdps(self.precision.policy.high_digits + 20):
            pass_metrics = (
                ("definition residual", self.maximum_definition_residual),
                ("identity residual", self.maximum_identity_residual),
                ("RD x-y symmetry residual", self.rd_xy_symmetry_residual),
                ("Kerr radial reduction residual", self.kerr_radial_reduction_residual),
            )
            for name, value in pass_metrics:
                self.precision.policy.require_metric(
                    value,
                    subject=f"Carlson certificate {name}",
                )

            # RD is symmetric only in x and y.  This named full permutation is a
            # negative control, not another identity residual.
            # Source: https://dlmf.nist.gov/19.21.ii
            minimum_negative_control = mp.mpf(1) / 100
            if (
                not mp.isfinite(self.rd_full_permutation_delta)
                or self.rd_full_permutation_delta <= minimum_negative_control
            ):
                raise AssertionError(
                    "Carlson certificate lost its RD non-symmetry negative control"
                )


def _normalized_residual(value: mp.mpf, scale: mp.mpf) -> mp.mpf:
    return abs(value) / max(mp.mpf(1), abs(scale))


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
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits):
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
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits):
        x, y, z = mp.mpf("0.5"), mp.mpf("1.25"), mp.mpf("2.75")
        root_x, root_y, root_z = mp.sqrt(x), mp.sqrt(y), mp.sqrt(z)
        duplication = root_x * root_y + root_y * root_z + root_z * root_x
        reduced = tuple((value + duplication) / 4 for value in (x, y, z))
        left = mp.elliprf(x, y, z)
        right = mp.elliprf(*reduced)
        return _normalized_residual(left - right, left)


def _rd_duplication_residual() -> mp.mpf:
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits):
        x, y, z = mp.mpf("0.5"), mp.mpf("1.25"), mp.mpf("2.75")
        root_x, root_y, root_z = mp.sqrt(x), mp.sqrt(y), mp.sqrt(z)
        duplication = root_x * root_y + root_y * root_z + root_z * root_x
        reduced = tuple((value + duplication) / 4 for value in (x, y, z))
        left = mp.elliprd(x, y, z)
        right = mp.elliprd(*reduced) / 4 + 3 / (root_z * (z + duplication))
        return _normalized_residual(left - right, left)


def _rj_duplication_residual() -> mp.mpf:
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits):
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
        alpha = p * (root_x + root_y + root_z) + root_x * root_y * root_z
        beta = mp.sqrt(p) * (p + duplication)
        right += 3 * mp.elliprc(alpha**2, beta**2)
        return _normalized_residual(left - right, left)


def _rc_duplication_residual() -> mp.mpf:
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits):
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


def _carlson_precision_evidence() -> _PrecisionEvidence:
    low = _definition_vector(_SCIENTIFIC_PRECISION.low_digits)
    high = _definition_vector(_SCIENTIFIC_PRECISION.high_digits)
    return _SCIENTIFIC_PRECISION.certify_pairs(
        zip(low, high, strict=True),
        subject="Carlson definition corpus",
    )


def _maximum_definition_residual() -> mp.mpf:
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits + 20):
        residuals = []
        for kind, arguments in _definition_cases():
            direct = _carlson_definition(
                kind,
                arguments,
                _SCIENTIFIC_PRECISION.high_digits,
            )
            library = _library_from_strings(
                kind,
                arguments,
                _SCIENTIFIC_PRECISION.high_digits,
            )
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
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits):
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
            _rd_duplication_residual(),
            _rj_duplication_residual(),
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
            precision_digits=_SCIENTIFIC_PRECISION.high_digits,
        )
        residuals.append(difference_report.normalized_identity_residual)
        return tuple(residuals)


def _kerr_radial_reduction_residual() -> mp.mpf:
    report = _build_topology_certificate().report("class-i-ib-inward")
    root_1, _, _, root_4 = report.ordered_real_roots
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits + 20):
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
            tuple(
                mp.nstr(value, _SCIENTIFIC_PRECISION.high_digits) for value in (x, y, z)
            ),
            _SCIENTIFIC_PRECISION.high_digits,
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
                mp.nstr(value, _SCIENTIFIC_PRECISION.high_digits)
                for value in (x, y, z, pole_argument)
            ),
            _SCIENTIFIC_PRECISION.high_digits,
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

    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits + 20):
        maximum_definition = _maximum_definition_residual()
        identity_residuals = _carlson_identity_residuals()
        maximum_identity = max(identity_residuals)
        precision = _carlson_precision_evidence()
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
        return _CarlsonCertificate(
            precision=precision,
            maximum_definition_residual=maximum_definition,
            maximum_identity_residual=maximum_identity,
            rd_xy_symmetry_residual=rd_xy,
            rd_full_permutation_delta=rd_full,
            kerr_radial_reduction_residual=kerr_reduction,
        )

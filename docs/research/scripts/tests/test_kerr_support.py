"""Conservative support-boundary checks for a future Kerr accelerator."""

from dataclasses import replace

import mpmath as mp

from gravlume_research.checks.kerr_support import (
    _build_support_certificate,
    _classify_support,
    _SupportDecision,
)


def test_support_classifier_rejects_conditioning_margin_inside_error_bound() -> None:
    certificate = _build_support_certificate()
    regular = certificate.report("regular-control")
    axis = regular.condition("axis-chart-denominator")
    ambiguous_axis = replace(axis, absolute_error_bound=axis.margin)
    mutated = replace(
        regular,
        conditions=tuple(
            ambiguous_axis if condition.name == ambiguous_axis.name else condition
            for condition in regular.conditions
        ),
    )

    assert _classify_support(mutated) is _SupportDecision.FALLBACK
    assert mutated.failed_conditions == ("axis-chart-denominator",)


def test_named_conditioning_corpus_has_zero_false_acceptance() -> None:
    certificate = _build_support_certificate()

    assert certificate.precision.maximum_normalized_delta < mp.power(
        10,
        -certificate.precision.policy.required_digits,
    )
    assert certificate.false_acceptance_count == 0
    assert certificate.report("regular-control").decision is (
        _SupportDecision.ELIGIBLE_FOR_FURTHER_ANALYSIS
    )
    expected_failures = {
        "near-axis": ("axis-chart-denominator",),
        "near-horizon": ("horizon-polynomial-separation",),
        "near-extremality": (
            "extremality-gap",
            "horizon-root-separation-squared",
        ),
        "near-radial-degeneracy": ("radial-root-separation",),
    }
    for name, failed_conditions in expected_failures.items():
        report = certificate.report(name)
        assert report.decision is _SupportDecision.FALLBACK
        assert report.failed_conditions == failed_conditions


def test_condition_error_bounds_enclose_every_exact_named_margin() -> None:
    certificate = _build_support_certificate()

    for report in certificate.reports:
        for condition in report.conditions:
            assert condition.interval_lower <= condition.margin
            assert condition.margin <= condition.interval_upper
            assert abs(condition.margin - condition.interval_lower) <= (
                condition.absolute_error_bound
            )
            assert abs(condition.interval_upper - condition.margin) <= (
                condition.absolute_error_bound
            )

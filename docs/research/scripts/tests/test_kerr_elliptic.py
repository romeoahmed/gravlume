"""High-precision Carlson and Kerr radial-topology research checks."""

from dataclasses import replace
from fractions import Fraction

import mpmath as mp
import pytest

from gravlume_research.checks.kerr_elliptic import (
    REQUIRED_STABLE_DIGITS,
    _build_carlson_certificate,
    _build_topology_certificate,
    _build_topology_report,
    _carlson_definition,
    _carlson_difference_report,
    _CarlsonDecision,
    _CarlsonKind,
    _homogeneity_residual,
    _rd_duplication_residual,
    _rj_duplication_residual,
    _TopologyCase,
    _TopologyDecision,
    _UnsupportedCarlsonError,
    _UnsupportedTopologyError,
)


def test_radial_topology_rejects_near_double_root_without_margin() -> None:
    report = _build_topology_certificate().report("near-double-boundary")

    assert report.decision is _TopologyDecision.FALLBACK
    assert "root-separation" in report.fallback_reasons
    assert report.minimum_root_separation < report.required_root_separation


def test_radial_topology_uses_initial_sign_to_select_ib_turn_sequence() -> None:
    certificate = _build_topology_certificate()
    inward = certificate.report("class-i-ib-inward")
    outward = certificate.report("class-i-ib-outward")

    assert inward.decision is _TopologyDecision.ELIGIBLE_FOR_FURTHER_ANALYSIS
    assert outward.decision is _TopologyDecision.ELIGIBLE_FOR_FURTHER_ANALYSIS
    assert inward.motion_branch == outward.motion_branch == "Ib"
    assert inward.turn_sequence == ("outer-radial-turn",)
    assert outward.turn_sequence == ()


def test_radial_topology_classifies_i_through_iv_before_fallback() -> None:
    certificate = _build_topology_certificate()

    assert certificate.maximum_normalized_delta < mp.power(
        10,
        -REQUIRED_STABLE_DIGITS,
    )
    assert certificate.false_acceptance_count == 0
    assert certificate.report("class-i-ib-inward").topology_id == "I"
    for name, topology_id in (
        ("class-ii", "II"),
        ("class-iii", "III"),
        ("class-iv", "IV"),
    ):
        report = certificate.report(name)
        assert report.topology_id == topology_id
        assert report.decision is _TopologyDecision.FALLBACK


def test_radial_topology_rejects_swapped_ordered_roots() -> None:
    report = _build_topology_certificate().report("class-i-ib-inward")
    first, second, *remaining = report.ordered_real_roots

    with pytest.raises(_UnsupportedTopologyError):
        replace(
            report,
            ordered_real_roots=(second, first, *remaining),
        )


def test_radial_topology_retains_sign_changing_root_brackets() -> None:
    report = _build_topology_certificate().report("class-i-ib-inward")

    for root, bracket, signs in zip(
        report.ordered_real_roots,
        report.ordered_real_root_brackets,
        report.real_root_bracket_signs,
        strict=True,
    ):
        assert bracket[0] < root < bracket[1]
        assert signs[0] * signs[1] < 0


def test_radial_topology_counts_nonreal_roots_without_imaginary_tolerance() -> None:
    certificate = _build_topology_certificate()

    assert certificate.report("class-iii").complex_root_count == 2
    assert certificate.report("class-iv").complex_root_count == 4
    with pytest.raises(_UnsupportedTopologyError):
        replace(certificate.report("class-iii"), complex_root_count=1)
    exact_double = certificate.report("exact-double-boundary")
    assert exact_double.minimum_root_separation == 0
    assert "root-separation" in exact_double.fallback_reasons


def test_negative_spin_canonicalization_preserves_radial_polynomial() -> None:
    certificate = _build_topology_certificate()
    positive = certificate.report("class-i-ib-inward")
    negative = certificate.report("negative-spin-ib")

    assert negative.normalized_spin == positive.normalized_spin
    assert negative.normalized_impact == positive.normalized_impact
    assert negative.polynomial_coefficients == positive.polynomial_coefficients
    assert negative.topology_id == positive.topology_id == "I"


def test_radial_topology_rejects_preclassification_domain_boundaries() -> None:
    baseline = _TopologyCase(
        name="domain-control",
        mass=Fraction(1),
        spin=Fraction(4, 5),
        energy=Fraction(1),
        impact=Fraction(-12),
        carter=Fraction(1, 100),
        initial_radius=Fraction(12),
        initial_radial_sign=-1,
        polar_mu_squared=Fraction(0),
        expected_topology="I",
        expected_motion_branch="Ib",
        expected_turn_sequence=("outer-radial-turn",),
        expected_decision=_TopologyDecision.ELIGIBLE_FOR_FURTHER_ANALYSIS,
    )

    invalid_changes = (
        {"energy": Fraction(0)},
        {"spin": Fraction(0)},
        {"impact": Fraction(4, 5), "carter": Fraction(0)},
        {"spin": Fraction(1) - Fraction(1, 10**100)},
        {"polar_mu_squared": Fraction(1) - Fraction(1, 10**100)},
    )
    for changes in invalid_changes:
        with pytest.raises(_UnsupportedTopologyError):
            replace(baseline, **changes)


def test_normalized_radial_polynomial_is_invariant_to_positive_energy_scale() -> None:
    baseline = _TopologyCase(
        name="energy-scale-control",
        mass=Fraction(1),
        spin=Fraction(4, 5),
        energy=Fraction(1),
        impact=Fraction(-12),
        carter=Fraction(1, 100),
        initial_radius=Fraction(12),
        initial_radial_sign=-1,
        polar_mu_squared=Fraction(0),
        expected_topology="I",
        expected_motion_branch="Ib",
        expected_turn_sequence=("outer-radial-turn",),
        expected_decision=_TopologyDecision.ELIGIBLE_FOR_FURTHER_ANALYSIS,
    )
    scaled = replace(baseline, energy=Fraction(7, 3))

    assert _build_topology_report(baseline, 120).polynomial_coefficients == (
        _build_topology_report(scaled, 120).polynomial_coefficients
    )


def test_carlson_oracle_rejects_wrong_homogeneity_degree() -> None:
    correct = _homogeneity_residual(
        _CarlsonKind.RF,
        ("0.5", "1.25", "2.75"),
        scale="3.5",
        degree="-0.5",
    )
    wrong = _homogeneity_residual(
        _CarlsonKind.RF,
        ("0.5", "1.25", "2.75"),
        scale="3.5",
        degree="-1.5",
    )

    assert correct < mp.mpf("1e-100")
    assert wrong > mp.mpf("0.1")


def test_carlson_oracle_detects_missing_rd_additive_term() -> None:
    assert _rd_duplication_residual(include_additive_term=True) < mp.mpf("1e-100")
    assert _rd_duplication_residual(include_additive_term=False) > mp.mpf("0.1")


def test_carlson_oracle_detects_missing_rj_rc_correction() -> None:
    assert _rj_duplication_residual(include_rc_correction=True) < mp.mpf("1e-100")
    assert _rj_duplication_residual(include_rc_correction=False) > mp.mpf("0.1")


def test_carlson_oracle_does_not_treat_rd_as_fully_symmetric() -> None:
    certificate = _build_carlson_certificate()

    assert certificate.rd_xy_symmetry_residual < mp.mpf("1e-100")
    assert certificate.rd_full_permutation_delta > mp.mpf("0.01")


def test_carlson_certificate_rejects_any_failed_pass_metric() -> None:
    certificate = _build_carlson_certificate()
    invalid_metrics = {
        "maximum_definition_residual": mp.mpf("1"),
        "maximum_identity_residual": mp.mpf("1"),
        "maximum_normalized_delta": mp.mpf("1"),
        "rd_xy_symmetry_residual": mp.mpf("1"),
        "rd_full_permutation_delta": mp.mpf("0"),
        "kerr_radial_reduction_residual": mp.mpf("1"),
    }

    for field, invalid_value in invalid_metrics.items():
        with pytest.raises(AssertionError, match="Carlson certificate"):
            replace(certificate, **{field: invalid_value})


def test_carlson_oracle_rejects_negative_p_without_principal_value_policy() -> None:
    with pytest.raises(_UnsupportedCarlsonError, match="principal-value"):
        _carlson_definition(
            _CarlsonKind.RJ,
            ("0.5", "1.25", "2.75", "-0.25"),
            precision_digits=120,
        )


def test_carlson_oracle_rejects_complex_and_ill_conditioned_arguments() -> None:
    with pytest.raises(_UnsupportedCarlsonError, match="finite real"):
        _carlson_definition(
            _CarlsonKind.RF,
            ("0.5j", "1.25", "2.75"),
            precision_digits=120,
        )
    with pytest.raises(_UnsupportedCarlsonError, match="conditioning limit"):
        _carlson_definition(
            _CarlsonKind.RJ,
            ("0.5", "1.25", "2.75", "1e-60"),
            precision_digits=120,
        )


def test_carlson_combination_reports_catastrophic_cancellation() -> None:
    report = _carlson_difference_report(
        x="1",
        y="2",
        p="2.00000000000000000000000000000000000000000000000001",
        precision_digits=120,
    )

    assert report.decision is _CarlsonDecision.FALLBACK
    assert report.cancellation_ratio > mp.mpf("1e40")


def test_carlson_certificate_matches_definitions_identities_and_kerr_segment() -> None:
    certificate = _build_carlson_certificate()

    assert certificate.maximum_definition_residual < mp.mpf("1e-100")
    assert certificate.maximum_identity_residual < mp.mpf("1e-100")
    assert certificate.maximum_normalized_delta < mp.power(
        10,
        -REQUIRED_STABLE_DIGITS,
    )
    assert certificate.kerr_radial_reduction_residual < mp.mpf("1e-100")

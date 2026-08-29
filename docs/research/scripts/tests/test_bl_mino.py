"""Deterministic checks for the named high-precision BL/Mino corpus."""

from dataclasses import replace
from itertools import pairwise

import mpmath as mp
import pytest

from gravlume_research.checks.bl_mino import (
    MINIMUM_WITNESS_DIGITS,
    REQUIRED_STABLE_DIGITS,
    _build_critical_curve_precision_certificate,
    _critical_curve_corpus_witness,
    _source_edge_corpus_witness,
    _UnsupportedWitnessError,
)


def test_source_edge_corpus_orders_cases_across_the_outer_edge() -> None:
    corpus = _source_edge_corpus_witness(
        precision_digits=MINIMUM_WITNESS_DIGITS,
    )

    assert tuple(case.pixel for case in corpus.cases) == tuple(
        (640, pixel_y) for pixel_y in range(12, 21)
    )
    assert tuple(case.witness.terminal for case in corpus.cases) == (
        "escape",
        "escape",
        "equatorial-surface",
        "equatorial-surface",
        "equatorial-surface",
        "equatorial-surface",
        "equatorial-surface",
        "equatorial-surface",
        "equatorial-surface",
    )

    margins = tuple(case.outer_edge_signed_margin_m for case in corpus.cases)
    assert all(left < right for left, right in pairwise(margins))
    assert margins[1] < 0 < margins[2]


def test_critical_curve_pair_brackets_capture_and_higher_order_surface() -> None:
    corpus = _critical_curve_corpus_witness(
        precision_digits=MINIMUM_WITNESS_DIGITS,
    )

    assert tuple(case.pixel for case in corpus.cases) == ((33, 10), (33, 11))
    assert corpus.critical_root_class == "exterior-double-root"
    assert tuple(case.exterior_radial_root_count for case in corpus.cases) == (2, 0)
    assert tuple(case.witness.terminal for case in corpus.cases) == (
        "equatorial-surface",
        "horizon",
    )

    surface = corpus.cases[0]
    capture = corpus.cases[1]
    assert surface.witness.radial_turnings == 1
    assert surface.witness.equatorial_crossings_before_terminal == 1
    assert surface.witness.azimuth_winding == 1
    assert capture.witness.radial_turnings == 0
    assert capture.witness.equatorial_crossings_before_terminal == 1
    assert capture.witness.azimuth_winding == 0
    assert surface.signed_critical_distance_pixels < 0
    assert capture.signed_critical_distance_pixels > 0
    assert surface.radial_classification_margin < 0
    assert capture.radial_classification_margin > 0


def test_critical_curve_certificate_rejects_stale_separatrix_distance() -> None:
    corpus = _critical_curve_corpus_witness(
        precision_digits=MINIMUM_WITNESS_DIGITS,
    )

    with pytest.raises(_UnsupportedWitnessError):
        replace(
            corpus,
            critical_sample_y=corpus.critical_sample_y + mp.mpf("0.125"),
        )


def test_surface_capture_certificate_rejects_stale_event_margin() -> None:
    corpus = _critical_curve_corpus_witness(
        precision_digits=MINIMUM_WITNESS_DIGITS,
    )
    capture = corpus.cases[1].witness

    with (
        mp.workdps(capture.precision_digits),
        pytest.raises(_UnsupportedWitnessError),
    ):
        replace(
            capture,
            horizon_after_first_crossing_mino_margin=(
                capture.horizon_after_first_crossing_mino_margin + mp.mpf("0.125")
            ),
        )


def test_higher_order_certificate_rejects_unwrapped_phase_mutation() -> None:
    corpus = _critical_curve_corpus_witness(
        precision_digits=MINIMUM_WITNESS_DIGITS,
    )
    surface = corpus.cases[0].witness

    with (
        mp.workdps(surface.precision_digits),
        pytest.raises(_UnsupportedWitnessError),
    ):
        replace(
            surface,
            source_azimuth_unwrapped_rad=(
                surface.source_azimuth_unwrapped_rad + 2 * mp.pi
            ),
        )


def test_critical_curve_precision_doubling_retains_required_digits() -> None:
    certificate = _build_critical_curve_precision_certificate()

    assert certificate.maximum_normalized_delta < mp.power(
        10,
        -REQUIRED_STABLE_DIGITS,
    )
    assert tuple(case.witness.terminal for case in certificate.witness.cases) == (
        "equatorial-surface",
        "horizon",
    )

"""Deterministic checks for the named high-precision BL/Mino corpus."""

from itertools import pairwise

from gravlume_research.checks.bl_mino import (
    MINIMUM_WITNESS_DIGITS,
    _source_edge_corpus_witness,
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

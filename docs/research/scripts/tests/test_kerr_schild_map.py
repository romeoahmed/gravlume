"""Exact identities for the Kerr--Schild/Boyer--Lindquist research seam."""

from gravlume_research.checks.kerr_schild_map import (
    _verify_outgoing_capture_regularization,
)


def test_outgoing_capture_primitives_are_algebraically_horizon_regular() -> None:
    _verify_outgoing_capture_regularization()

"""Behavior tests for the independent BL/Mino surface witness."""

from __future__ import annotations

import unittest
from dataclasses import replace

import mpmath as mp

from verify_bl_mino_surface_witness import (
    UnsupportedWitness,
    compute_canonical_surface_witness,
)


class CanonicalSurfaceWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.witness = compute_canonical_surface_witness(
            pixel_x=640,
            pixel_y=16,
            precision_digits=80,
        )

    def test_recovers_the_named_surface_terminal(self) -> None:
        self.assertEqual(self.witness.terminal, "equatorial-surface")
        self.assertLess(
            abs(
                self.witness.source_radius_m
                - mp.mpf("19.6506789846041094")
            ),
            mp.mpf("2e-9"),
        )
        self.assertLess(
            abs(
                self.witness.source_azimuth_rad
                - mp.mpf("3.08715626242367058")
            ),
            mp.mpf("1e-10"),
        )

    def test_preserves_the_discrete_path_identity(self) -> None:
        self.assertEqual(self.witness.initial_polar_side, "positive")
        self.assertEqual(self.witness.radial_turnings, 1)
        self.assertEqual(self.witness.polar_turnings, 1)
        self.assertEqual(self.witness.equatorial_crossings_before_terminal, 0)
        self.assertEqual(self.witness.azimuth_winding, 0)
        invalid_identity_fields = (
            ("terminal", "escape"),
            ("initial_polar_side", "negative"),
            ("radial_turnings", True),
            ("polar_turnings", 1.0),
            ("equatorial_crossings_before_terminal", -1),
            ("azimuth_winding", 1),
        )
        for field, invalid_value in invalid_identity_fields:
            with self.subTest(field=field, invalid_value=invalid_value):
                with self.assertRaises(UnsupportedWitness):
                    replace(self.witness, **{field: invalid_value})

    def test_recovers_transfer_and_phase_observables(self) -> None:
        self.assertLess(
            abs(
                self.witness.frequency_ratio
                - mp.mpf("0.953264138194626409")
            ),
            mp.mpf("2e-9"),
        )
        self.assertLess(
            abs(
                self.witness.travel_time_m
                - mp.mpf("54.9024742476313605")
            ),
            mp.mpf("2e-8"),
        )

    def test_retains_independent_equation_residuals(self) -> None:
        self.assertGreater(self.witness.radial_turning_derivative, 0)
        self.assertGreater(self.witness.polar_turning_derivative, 0)
        self.assertLess(self.witness.initial_null_residual, mp.mpf("1e-65"))
        self.assertLess(self.witness.mino_constraint_residual, mp.mpf("1e-65"))
        self.assertLess(self.witness.chart_primitive_residual, mp.mpf("1e-65"))
        for invalid_residual in (mp.mpf("1e-20"), mp.nan):
            with self.subTest(invalid_residual=invalid_residual):
                with self.assertRaises(UnsupportedWitness):
                    replace(
                        self.witness,
                        mino_constraint_residual=invalid_residual,
                    )

    def test_rejects_inputs_outside_the_named_observation(self) -> None:
        with self.assertRaises(UnsupportedWitness):
            compute_canonical_surface_witness(
                pixel_x=1280,
                pixel_y=16,
                precision_digits=80,
            )

    def test_rejects_uncertified_samples_inside_the_viewport(self) -> None:
        with self.assertRaises(UnsupportedWitness):
            compute_canonical_surface_witness(
                pixel_x=640,
                pixel_y=15,
                precision_digits=80,
            )


if __name__ == "__main__":
    unittest.main()

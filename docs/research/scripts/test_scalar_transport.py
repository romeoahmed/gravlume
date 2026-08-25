"""Regression tests for the arbitrary-precision scalar-transport oracle."""

from __future__ import annotations

import unittest

import mpmath as mp
from verify_scalar_transport import planck_band_fraction

ORACLE_PRECISION_DIGITS = 80


class ScalarTransportOracleTests(unittest.TestCase):
    def test_six_thousand_kelvin_red_band_retains_decimal_input_precision(
        self,
    ) -> None:
        with mp.workdps(ORACLE_PRECISION_DIGITS):
            actual = planck_band_fraction(mp.mpf(6000), mp.mpf(600), mp.mpf(700))
            expected = mp.mpf(
                "0.11240120367128770505189618323010601606160060957692600823721937343264961963013441"
            )
            self.assertTrue(
                mp.almosteq(
                    actual,
                    expected,
                    rel_eps=mp.mpf(0),
                    abs_eps=mp.mpf("1e-70"),
                ),
                f"red-band oracle lost decimal input precision: {actual - expected}",
            )


if __name__ == "__main__":
    unittest.main()

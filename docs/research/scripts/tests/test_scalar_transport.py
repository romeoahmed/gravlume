"""High-precision examples and generated properties for scalar transport."""

import mpmath as mp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gravlume_research.checks.scalar_transport import (
    planck_band_fraction,
    planck_band_fraction_fast,
)

ORACLE_PRECISION_DIGITS = 80


def test_six_thousand_kelvin_red_band_retains_decimal_precision() -> None:
    with mp.workdps(ORACLE_PRECISION_DIGITS):
        actual = planck_band_fraction(mp.mpf(6000), mp.mpf(600), mp.mpf(700))
        expected = mp.mpf(
            "0.11240120367128770505189618323010601606160060957692600823721937343264961963013441"
        )
        assert actual == pytest.approx(
            expected,
            rel=mp.mpf(0),
            abs=mp.mpf("1e-70"),
        )


@st.composite
def wavelength_partition(draw: st.DrawFn) -> tuple[int, int, int]:
    lower = draw(st.integers(min_value=50, max_value=4_000))
    first_width = draw(st.integers(min_value=1, max_value=500))
    second_width = draw(st.integers(min_value=1, max_value=500))
    middle = lower + first_width
    return lower, middle, middle + second_width


@settings(deadline=None, derandomize=True, max_examples=50)
@given(
    temperature=st.integers(min_value=100, max_value=20_000),
    wavelengths=wavelength_partition(),
)
def test_planck_band_fraction_is_bounded_and_additive(
    temperature: int,
    wavelengths: tuple[int, int, int],
) -> None:
    lower, middle, upper = wavelengths
    with mp.workdps(ORACLE_PRECISION_DIGITS):
        temperature_mpf = mp.mpf(temperature)
        whole = planck_band_fraction(temperature_mpf, lower, upper)
        closed_form = planck_band_fraction_fast(temperature_mpf, lower, upper)
        partitioned = planck_band_fraction(
            temperature_mpf, lower, middle
        ) + planck_band_fraction(temperature_mpf, middle, upper)

        assert mp.mpf(0) <= whole <= mp.mpf(1)
        assert closed_form == pytest.approx(
            whole,
            rel=mp.mpf("1e-65"),
            abs=mp.mpf("1e-70"),
        )
        assert partitioned == pytest.approx(
            whole,
            rel=mp.mpf("1e-65"),
            abs=mp.mpf("1e-70"),
        )

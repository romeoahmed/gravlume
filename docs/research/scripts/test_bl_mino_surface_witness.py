"""Behavior tests for the independent BL/Mino surface witness.

Property generation follows Hypothesis's ``@given``/unittest contract:
https://hypothesis.readthedocs.io/en/latest/settings.html#hypothesis.given
Named arbitrary-precision observables use explicit absolute ``almosteq`` budgets:
https://mpmath.org/doc/1.3.0/general.html#almosteq
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import mpmath as mp
from hypothesis import example, given, settings
from hypothesis import strategies as st

from verify_bl_mino_surface_witness import (
    UnsupportedWitness,
    compute_canonical_surface_witness,
)

CANONICAL_IDENTITY = {
    "terminal": "equatorial-surface",
    "initial_polar_side": "positive",
    "radial_turnings": 1,
    "polar_turnings": 1,
    "equatorial_crossings_before_terminal": 0,
    "azimuth_winding": 0,
}
TEST_PRECISION_DIGITS = 80
NON_INTEGER = st.one_of(
    st.none(),
    st.booleans(),
    st.floats(),
    st.text(),
)
UNSUPPORTED_SAMPLE = st.one_of(
    st.tuples(st.integers(), st.integers()).filter(
        lambda sample: sample != (640, 16)
    ),
    st.tuples(NON_INTEGER, st.just(16)),
    st.tuples(st.just(640), NON_INTEGER),
)
INVALID_PRECISION = st.one_of(
    st.integers(max_value=69),
    NON_INTEGER,
)
NON_MPF = st.one_of(
    NON_INTEGER,
    st.integers(),
)
RESIDUAL_FIELDS = (
    "initial_null_residual",
    "mino_constraint_residual",
    "chart_primitive_residual",
)
with mp.workdps(TEST_PRECISION_DIGITS):
    RESIDUAL_LIMIT = mp.power(10, -65)


INVALID_RESIDUAL = st.one_of(
    NON_MPF,
    st.just(mp.nan),
    st.just(mp.inf),
    st.just(RESIDUAL_LIMIT),
    st.integers(max_value=-1).map(mp.mpf),
    st.integers(min_value=1).map(mp.mpf),
)


@st.composite
def invalid_identity_mutation(draw: st.DrawFn) -> tuple[str, object]:
    field, expected = draw(st.sampled_from(tuple(CANONICAL_IDENTITY.items())))
    invalid_value = draw(
        NON_MPF.filter(
            lambda value: type(value) is not type(expected) or value != expected
        )
    )
    return field, invalid_value


class CanonicalSurfaceWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.witness = compute_canonical_surface_witness(
            pixel_x=640,
            pixel_y=16,
            precision_digits=TEST_PRECISION_DIGITS,
        )

    def test_recovers_the_named_surface_observables(self) -> None:
        expected_observables = (
            ("source_radius_m", "19.6506789846041094", "2e-9"),
            ("source_azimuth_rad", "3.08715626242367058", "1e-10"),
            ("frequency_ratio", "0.953264138194626409", "2e-9"),
            ("travel_time_m", "54.9024742476313605", "2e-8"),
        )
        for field, expected, tolerance in expected_observables:
            actual = getattr(self.witness, field)
            with self.subTest(field=field), mp.workdps(TEST_PRECISION_DIGITS):
                self.assertTrue(
                    mp.almosteq(
                        actual,
                        mp.mpf(expected),
                        rel_eps=mp.mpf(0),
                        abs_eps=mp.mpf(tolerance),
                    ),
                    (
                        f"{field}={actual} differs from {expected} "
                        f"by more than {tolerance}"
                    ),
                )

    def test_preserves_the_discrete_path_identity(self) -> None:
        for field, expected in CANONICAL_IDENTITY.items():
            with self.subTest(field=field):
                self.assertEqual(getattr(self.witness, field), expected)

    @settings(database=None, derandomize=True)
    @given(mutation=invalid_identity_mutation())
    def test_rejects_every_noncanonical_discrete_identity(
        self, mutation: tuple[str, object]
    ) -> None:
        field, invalid_value = mutation
        with self.assertRaises(UnsupportedWitness):
            replace(self.witness, **{field: invalid_value})

    @settings(database=None, derandomize=True)
    @given(invalid_residual=INVALID_RESIDUAL)
    def test_rejects_every_uncertified_equation_residual(
        self, invalid_residual: mp.mpf
    ) -> None:
        for field in RESIDUAL_FIELDS:
            with self.assertRaises(UnsupportedWitness, msg=field):
                replace(self.witness, **{field: invalid_residual})

    @settings(database=None, derandomize=True)
    @example(sample=(640.0, 16.0))
    @given(sample=UNSUPPORTED_SAMPLE)
    def test_rejects_every_sample_outside_the_integer_named_case(
        self, sample: tuple[object, object]
    ) -> None:
        pixel_x, pixel_y = sample
        with self.assertRaises(UnsupportedWitness):
            compute_canonical_surface_witness(
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                precision_digits=TEST_PRECISION_DIGITS,
            )

    @settings(database=None, derandomize=True)
    @given(precision_digits=INVALID_PRECISION)
    def test_rejects_every_invalid_working_precision(
        self, precision_digits: object
    ) -> None:
        with self.assertRaises(UnsupportedWitness):
            compute_canonical_surface_witness(
                pixel_x=640,
                pixel_y=16,
                precision_digits=precision_digits,
            )


if __name__ == "__main__":
    unittest.main()

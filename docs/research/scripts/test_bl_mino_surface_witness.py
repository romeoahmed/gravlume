"""Behavior tests for the independent BL/Mino surface witness.

Property generation follows Hypothesis's ``@given``/unittest contract:
https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.given
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
    UnsupportedWitnessError,
    compute_canonical_surface_witness,
    compute_source_edge_pair_witness,
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
    st.tuples(st.integers(), st.integers()).filter(lambda sample: sample != (640, 16)),
    st.tuples(NON_INTEGER, st.just(16)),
    st.tuples(st.just(640), NON_INTEGER),
)
INVALID_PRECISION = st.one_of(
    st.integers(max_value=69),
    NON_INTEGER,
)
NON_MPF_VALUE = st.one_of(
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
    NON_MPF_VALUE,
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
        NON_MPF_VALUE.filter(
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

    @settings(derandomize=True)
    @given(mutation=invalid_identity_mutation())
    def test_rejects_every_noncanonical_discrete_identity(
        self, mutation: tuple[str, object]
    ) -> None:
        field, invalid_value = mutation
        with self.assertRaises(UnsupportedWitnessError):
            replace(self.witness, **{field: invalid_value})

    @settings(derandomize=True)
    @given(invalid_residual=INVALID_RESIDUAL)
    def test_rejects_every_uncertified_equation_residual(
        self, invalid_residual: object
    ) -> None:
        for field in RESIDUAL_FIELDS:
            with self.assertRaises(UnsupportedWitnessError, msg=field):
                replace(self.witness, **{field: invalid_residual})

    @settings(derandomize=True)
    @example(sample=(640.0, 16.0))
    @given(sample=UNSUPPORTED_SAMPLE)
    def test_rejects_every_sample_outside_the_integer_named_case(
        self, sample: tuple[object, object]
    ) -> None:
        pixel_x, pixel_y = sample
        with self.assertRaises(UnsupportedWitnessError):
            compute_canonical_surface_witness(
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                precision_digits=TEST_PRECISION_DIGITS,
            )

    @settings(derandomize=True)
    @given(precision_digits=INVALID_PRECISION)
    def test_rejects_every_invalid_working_precision(
        self, precision_digits: object
    ) -> None:
        with self.assertRaises(UnsupportedWitnessError):
            compute_canonical_surface_witness(
                pixel_x=640,
                pixel_y=16,
                precision_digits=precision_digits,
            )


class SourceEdgePairWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pair = compute_source_edge_pair_witness(
            precision_digits=TEST_PRECISION_DIGITS
        )

    def test_brackets_the_outer_source_edge_with_exact_path_identity(self) -> None:
        outside = self.pair.outside
        inside = self.pair.inside

        self.assertEqual(outside.terminal, "escape")
        self.assertEqual(outside.initial_polar_side, "positive")
        self.assertEqual(outside.radial_turnings, 1)
        self.assertEqual(outside.polar_turnings, 1)
        self.assertEqual(outside.equatorial_crossings_before_terminal, 1)
        self.assertEqual(outside.azimuth_winding, 0)
        self.assertGreater(outside.first_equatorial_crossing_radius_m, mp.mpf(20))

        for field, expected in CANONICAL_IDENTITY.items():
            with self.subTest(field=field):
                self.assertEqual(getattr(inside, field), expected)
        self.assertLess(inside.source_radius_m, mp.mpf(20))

    def test_recovers_the_inside_surface_observables(self) -> None:
        expected_observables = (
            ("source_radius_m", "19.9064149026366577", "2e-9"),
            ("source_azimuth_rad", "3.08817265206733668", "1e-10"),
            ("frequency_ratio", "0.954336623855338749", "2e-9"),
            ("travel_time_m", "55.1114457365679603", "2e-8"),
        )
        for field, expected, tolerance in expected_observables:
            actual = getattr(self.pair.inside, field)
            with self.subTest(field=field), mp.workdps(TEST_PRECISION_DIGITS):
                self.assertTrue(
                    mp.almosteq(
                        actual,
                        mp.mpf(expected),
                        rel_eps=mp.mpf(0),
                        abs_eps=mp.mpf(tolerance),
                    )
                )

    def test_recovers_the_outside_escape_observables_and_event_order(self) -> None:
        outside = self.pair.outside
        expected_position = tuple(
            map(
                mp.mpf,
                (
                    "-170.447402756461085",
                    "1.36924488278322070",
                    "-104.624437497465033",
                ),
            )
        )
        expected_direction = tuple(
            map(
                mp.mpf,
                (
                    "-0.820715680321071555",
                    "0.00602390419818969499",
                    "-0.571305071440234735",
                ),
            )
        )
        with mp.workdps(TEST_PRECISION_DIGITS):
            position_error = mp.sqrt(
                mp.fsum(
                    (actual - expected) ** 2
                    for actual, expected in zip(
                        outside.escape_position_xyz_m,
                        expected_position,
                        strict=True,
                    )
                )
            )
            expected_direction_norm = mp.sqrt(
                mp.fsum(component**2 for component in expected_direction)
            )
            direction_dot = mp.fsum(
                actual * expected / expected_direction_norm
                for actual, expected in zip(
                    outside.escape_direction_xyz,
                    expected_direction,
                    strict=True,
                )
            )
            actual_x, actual_y, actual_z = outside.escape_direction_xyz
            expected_x, expected_y, expected_z = (
                component / expected_direction_norm for component in expected_direction
            )
            direction_cross_norm = mp.sqrt(
                (actual_y * expected_z - actual_z * expected_y) ** 2
                + (actual_z * expected_x - actual_x * expected_z) ** 2
                + (actual_x * expected_y - actual_y * expected_x) ** 2
            )
            direction_angle = mp.atan2(direction_cross_norm, direction_dot)

            self.assertLess(position_error, mp.mpf("2e-9"))
            self.assertLess(direction_angle, mp.mpf("2e-9"))
            self.assertTrue(
                mp.almosteq(
                    outside.travel_time_m,
                    mp.mpf("238.438694378676361"),
                    rel_eps=mp.mpf(0),
                    abs_eps=mp.mpf("2e-8"),
                )
            )
        self.assertGreater(outside.escape_before_next_crossing_mino_margin, 0)

    def test_rejects_uncertified_escape_identity_and_event_margins(self) -> None:
        outside = self.pair.outside
        invalid_mutations = (
            {"equatorial_crossings_before_terminal": 0},
            {"first_equatorial_crossing_radius_m": mp.mpf(20)},
            {"escape_before_next_crossing_mino_margin": mp.mpf(0)},
            {
                "escape_direction_xyz": (
                    mp.mpf(2),
                    mp.mpf(0),
                    mp.mpf(0),
                )
            },
        )
        for mutation in invalid_mutations:
            with self.subTest(mutation=mutation), self.assertRaises(
                UnsupportedWitnessError
            ):
                replace(outside, **mutation)

    @settings(derandomize=True)
    @given(precision_digits=INVALID_PRECISION)
    def test_rejects_every_invalid_pair_working_precision(
        self, precision_digits: object
    ) -> None:
        with self.assertRaises(UnsupportedWitnessError):
            compute_source_edge_pair_witness(precision_digits=precision_digits)


if __name__ == "__main__":
    unittest.main()

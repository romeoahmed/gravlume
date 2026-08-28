"""Behavior and validation properties for the independent BL/Mino witness."""

from dataclasses import replace
from functools import cache

import mpmath as mp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gravlume_research.checks.bl_mino import (
    UnsupportedWitnessError,
    canonical_surface_witness,
    source_edge_pair_witness,
)

TEST_PRECISION_DIGITS = 80
CANONICAL_IDENTITY = {
    "terminal": "equatorial-surface",
    "initial_polar_side": "positive",
    "radial_turnings": 1,
    "polar_turnings": 1,
    "equatorial_crossings_before_terminal": 0,
    "azimuth_winding": 0,
}
NON_INTEGER = st.one_of(st.none(), st.booleans(), st.floats(), st.text())
INVALID_PRECISION = st.one_of(st.integers(max_value=69), NON_INTEGER)
NON_MPF_VALUE = st.one_of(NON_INTEGER, st.integers())
RESIDUAL_FIELDS = (
    "initial_null_residual",
    "mino_constraint_residual",
    "chart_primitive_residual",
)
with mp.workdps(TEST_PRECISION_DIGITS):
    RESIDUAL_LIMIT = mp.power(10, -65)
INVALID_RESIDUAL = st.one_of(
    NON_MPF_VALUE,
    st.sampled_from((mp.nan, mp.inf, RESIDUAL_LIMIT)),
    st.integers().filter(bool).map(mp.mpf),
)


@st.composite
def invalid_identity_mutation(draw: st.DrawFn) -> tuple[str, object]:
    field, expected = draw(st.sampled_from(tuple(CANONICAL_IDENTITY.items())))
    invalid = draw(
        NON_MPF_VALUE.filter(
            lambda value: type(value) is not type(expected) or value != expected
        )
    )
    return field, invalid


@cache
def _canonical_witness():
    return canonical_surface_witness(precision_digits=TEST_PRECISION_DIGITS)


@cache
def _source_edge_pair():
    return source_edge_pair_witness(precision_digits=TEST_PRECISION_DIGITS)


@pytest.mark.parametrize(
    ("field", "expected", "absolute_tolerance"),
    (
        ("source_radius_m", "19.6506789846041094", "2e-9"),
        ("source_azimuth_rad", "3.08715626242367058", "1e-10"),
        ("frequency_ratio", "0.953264138194626409", "2e-9"),
        ("travel_time_m", "54.9024742476313605", "2e-8"),
    ),
)
def test_canonical_surface_observables(
    field: str, expected: str, absolute_tolerance: str
) -> None:
    with mp.workdps(TEST_PRECISION_DIGITS):
        assert getattr(_canonical_witness(), field) == pytest.approx(
            mp.mpf(expected),
            rel=mp.mpf(0),
            abs=mp.mpf(absolute_tolerance),
        )


@pytest.mark.parametrize("field", CANONICAL_IDENTITY)
def test_canonical_discrete_path_identity(field: str) -> None:
    assert getattr(_canonical_witness(), field) == CANONICAL_IDENTITY[field]


@settings(derandomize=True, deadline=None)
@given(mutation=invalid_identity_mutation())
def test_noncanonical_discrete_identity_is_rejected(
    mutation: tuple[str, object],
) -> None:
    field, invalid = mutation
    with pytest.raises(UnsupportedWitnessError):
        replace(_canonical_witness(), **{field: invalid})


@settings(derandomize=True, deadline=None)
@given(field=st.sampled_from(RESIDUAL_FIELDS), invalid=INVALID_RESIDUAL)
def test_uncertified_equation_residual_is_rejected(field: str, invalid: object) -> None:
    with pytest.raises(UnsupportedWitnessError):
        replace(_canonical_witness(), **{field: invalid})


@settings(derandomize=True, deadline=None)
@given(precision_digits=INVALID_PRECISION)
def test_invalid_canonical_precision_is_rejected(precision_digits: object) -> None:
    with pytest.raises(UnsupportedWitnessError):
        canonical_surface_witness(precision_digits=precision_digits)


def test_source_edge_pair_brackets_the_outer_edge() -> None:
    outside = _source_edge_pair().outside
    inside = _source_edge_pair().inside

    assert (
        outside.terminal,
        outside.initial_polar_side,
        outside.radial_turnings,
        outside.polar_turnings,
        outside.equatorial_crossings_before_terminal,
        outside.azimuth_winding,
    ) == ("escape", "positive", 1, 1, 1, 0)
    assert mp.mpf(20) < outside.first_equatorial_crossing_radius_m
    assert outside.first_equatorial_crossing_radius_m < outside.escape_radius_m
    assert inside.source_radius_m < mp.mpf(20)
    for field, expected in CANONICAL_IDENTITY.items():
        assert getattr(inside, field) == expected


@pytest.mark.parametrize(
    ("field", "expected", "absolute_tolerance"),
    (
        ("source_radius_m", "19.9064149026366577", "2e-9"),
        ("source_azimuth_rad", "3.08817265206733668", "1e-10"),
        ("frequency_ratio", "0.954336623855338749", "2e-9"),
        ("travel_time_m", "55.1114457365679603", "2e-8"),
    ),
)
def test_inside_surface_observables(
    field: str, expected: str, absolute_tolerance: str
) -> None:
    with mp.workdps(TEST_PRECISION_DIGITS):
        assert getattr(_source_edge_pair().inside, field) == pytest.approx(
            mp.mpf(expected),
            rel=mp.mpf(0),
            abs=mp.mpf(absolute_tolerance),
        )


def test_outside_escape_observables_and_event_order() -> None:
    outside = _source_edge_pair().outside
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
                    outside.escape_position_xyz_m, expected_position, strict=True
                )
            )
        )
        expected_norm = mp.sqrt(mp.fsum(value**2 for value in expected_direction))
        normalized_expected = tuple(
            value / expected_norm for value in expected_direction
        )
        actual_x, actual_y, actual_z = outside.escape_direction_xyz
        expected_x, expected_y, expected_z = normalized_expected
        direction_dot = mp.fsum(
            actual * expected
            for actual, expected in zip(
                outside.escape_direction_xyz, normalized_expected, strict=True
            )
        )
        direction_cross_norm = mp.sqrt(
            (actual_y * expected_z - actual_z * expected_y) ** 2
            + (actual_z * expected_x - actual_x * expected_z) ** 2
            + (actual_x * expected_y - actual_y * expected_x) ** 2
        )
        direction_angle = mp.atan2(direction_cross_norm, direction_dot)

        assert position_error == pytest.approx(mp.mpf(0), abs=mp.mpf("2e-9"))
        assert direction_angle == pytest.approx(mp.mpf(0), abs=mp.mpf("2e-9"))
        assert outside.travel_time_m == pytest.approx(
            mp.mpf("238.438694378676361"),
            rel=mp.mpf(0),
            abs=mp.mpf("2e-8"),
        )
    assert outside.escape_before_next_crossing_mino_margin > 0


@pytest.mark.parametrize(
    "mutation",
    (
        {"equatorial_crossings_before_terminal": 0},
        {"first_equatorial_crossing_radius_m": mp.mpf(20)},
        {"escape_before_next_crossing_mino_margin": mp.mpf(0)},
        {"escape_direction_xyz": (mp.mpf(2), mp.mpf(0), mp.mpf(0))},
    ),
)
def test_uncertified_escape_identity_is_rejected(
    mutation: dict[str, object],
) -> None:
    with pytest.raises(UnsupportedWitnessError):
        replace(_source_edge_pair().outside, **mutation)


def test_crossing_at_escape_terminal_is_rejected() -> None:
    outside = _source_edge_pair().outside
    with pytest.raises(UnsupportedWitnessError):
        replace(
            outside,
            first_equatorial_crossing_radius_m=outside.escape_radius_m,
        )


@settings(derandomize=True, deadline=None)
@given(precision_digits=INVALID_PRECISION)
def test_invalid_edge_pair_precision_is_rejected(precision_digits: object) -> None:
    with pytest.raises(UnsupportedWitnessError):
        source_edge_pair_witness(precision_digits=precision_digits)

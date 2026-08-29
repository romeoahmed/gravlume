"""Generated strict-binary32 enclosure checks for the Kerr capture model."""

from hypothesis import example, given, settings
from hypothesis import strategies as st

from gravlume_research._binary32 import (
    MAX_FINITE,
    MIN_NORMAL,
    MIN_SUBNORMAL,
    next_down,
)
from gravlume_research.checks.kerr_capture import (
    _BernsteinSample,
    _check_interval_bernstein,
    _check_interval_primitives,
)

FINITE_BINARY32 = st.floats(
    allow_infinity=False,
    allow_nan=False,
    allow_subnormal=True,
    width=32,
)
BOUNDED_BINARY32 = st.floats(
    min_value=-8.0,
    max_value=8.0,
    allow_infinity=False,
    allow_nan=False,
    allow_subnormal=True,
    width=32,
)
BERNSTEIN_SAMPLES = st.builds(
    _BernsteinSample,
    leading=BOUNDED_BINARY32,
    quadratic=BOUNDED_BINARY32,
    linear=BOUNDED_BINARY32,
    constant=BOUNDED_BINARY32,
    lower=BOUNDED_BINARY32,
    width=BOUNDED_BINARY32,
)


@settings(deadline=None, derandomize=True, max_examples=20_000)
@example(left=0.0, right=-0.0)
@example(left=MIN_SUBNORMAL, right=-MIN_SUBNORMAL)
@example(left=MIN_NORMAL, right=-next_down(MIN_NORMAL))
@example(left=MAX_FINITE, right=-MAX_FINITE)
@given(left=FINITE_BINARY32, right=FINITE_BINARY32)
def test_interval_primitives_enclose_exact_results(left: float, right: float) -> None:
    _check_interval_primitives(left, right)


@settings(deadline=None, derandomize=True, max_examples=5_000)
@example(
    sample=_BernsteinSample(
        leading=0.0,
        quadratic=MIN_SUBNORMAL,
        linear=-MIN_SUBNORMAL,
        constant=MIN_NORMAL,
        lower=0.0,
        width=0.0,
    )
)
@given(sample=BERNSTEIN_SAMPLES)
def test_interval_bernstein_coefficients_enclose_exact_results(
    sample: _BernsteinSample,
) -> None:
    _check_interval_bernstein(sample)

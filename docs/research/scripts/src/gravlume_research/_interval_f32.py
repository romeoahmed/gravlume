"""Strict binary32 interval primitives shared by research-only checks.

Every operation widens across adjacent binary32 values and the WGSL-permitted
flush-to-zero behavior before its result can feed another operation.  This is
an explicitly sequenced arithmetic model, not a promise that an arbitrary
shader expression has the same evaluation graph.

WGSL floating-point rules:
https://www.w3.org/TR/WGSL/#floating-point-evaluation
"""

import math
from dataclasses import dataclass
from fractions import Fraction

from ._binary32 import MIN_NORMAL, next_down, next_up, round_binary32


def ftz_safe_lower(value: float) -> float:
    """Widen a lower bound across implementations that may flush subnormals."""

    if not math.isfinite(value) or abs(value) >= MIN_NORMAL:
        return value
    return -MIN_NORMAL if value <= 0.0 else 0.0


def ftz_safe_upper(value: float) -> float:
    """Widen an upper bound across implementations that may flush subnormals."""

    if not math.isfinite(value) or abs(value) >= MIN_NORMAL:
        return value
    return MIN_NORMAL if value >= 0.0 else 0.0


def outward_lower(value: float) -> float:
    """Round and widen one binary32 value toward negative infinity."""

    return ftz_safe_lower(next_down(round_binary32(value)))


def outward_upper(value: float) -> float:
    """Round and widen one binary32 value toward positive infinity."""

    return ftz_safe_upper(next_up(round_binary32(value)))


@dataclass(frozen=True, slots=True)
class IntervalF32:
    """Outward-rounded interval for an explicitly sequenced binary32 graph."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise ValueError(f"invalid interval [{self.lower}, {self.upper}]")

    @classmethod
    def point(cls, value: float) -> IntervalF32:
        """Create a point interval from a value already rounded to binary32."""

        packed = round_binary32(value)
        if not math.isfinite(packed):
            raise ValueError("an interval input must be finite")
        return cls(packed, packed)

    @classmethod
    def rational(cls, numerator: int, denominator: int) -> IntervalF32:
        """Enclose an exact rational after binary32 input conversion."""

        exact = float(Fraction(numerator, denominator))
        return cls(outward_lower(exact), outward_upper(exact))

    def add(self, other: IntervalF32) -> IntervalF32:
        return IntervalF32(
            outward_lower(self.lower + other.lower),
            outward_upper(self.upper + other.upper),
        )

    def sub(self, other: IntervalF32) -> IntervalF32:
        return IntervalF32(
            outward_lower(self.lower - other.upper),
            outward_upper(self.upper - other.lower),
        )

    def mul(self, other: IntervalF32) -> IntervalF32:
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return IntervalF32(outward_lower(min(products)), outward_upper(max(products)))

    def square(self) -> IntervalF32:
        if self.lower <= 0.0 <= self.upper:
            maximum = max(self.lower * self.lower, self.upper * self.upper)
            return IntervalF32(0.0, outward_upper(maximum))
        products = (self.lower * self.lower, self.upper * self.upper)
        return IntervalF32(outward_lower(min(products)), outward_upper(max(products)))

    def scale_rational(self, numerator: int, denominator: int) -> IntervalF32:
        return self.mul(IntervalF32.rational(numerator, denominator))


def exact_fraction(value: float) -> Fraction:
    """Return the exact rational represented by a finite binary float."""

    if not math.isfinite(value):
        raise ValueError("infinite values have no finite rational representation")
    return Fraction(value)


def interval_contains(interval: IntervalF32, exact: Fraction) -> bool:
    """Return whether an interval contains an exact rational value."""

    lower_ok = interval.lower == -math.inf or exact_fraction(interval.lower) <= exact
    upper_ok = interval.upper == math.inf or exact <= exact_fraction(interval.upper)
    return lower_ok and upper_ok

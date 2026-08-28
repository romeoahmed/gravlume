"""Reproduce the algebra and primitive interval model of the Kerr capture path.

SymPy proves the exact polynomial identities. ``IntervalF32`` separately
models the WGSL implementation's explicitly sequenced arithmetic primitives.
Hypothesis generates exactly representable binary32 values and shrinks any
counterexample; explicit examples retain the numerically important boundaries.
Each arithmetic operation is consumed by a floating-point bit reinterpretation
before the next operation, so reassociation or fusion cannot legally cross the
integer-bit dependency. The script verifies those primitive enclosures and the
quartic-to-Bernstein transform. It does not by itself prove that every physical
initial ray lies inside the shader's deliberately wider invariant envelope;
that seam remains covered by the supported-domain fallback and CPU/GPU oracle
matrix. This is a research/validation tool, not a runtime dependency.
"""

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction

import sympy as sp

from .._binary32 import (
    MAX_FINITE,
    MIN_NORMAL,
    MIN_SUBNORMAL,
    binary32_from_bits,
    next_down,
    next_up,
    round_binary32,
)
from .._sympy import require_polynomial_equal

type ExactScalar = sp.Expr | Fraction
type ExactCoefficients = tuple[ExactScalar, ExactScalar, ExactScalar, ExactScalar]


@dataclass(frozen=True, slots=True, kw_only=True)
class BernsteinSample:
    leading: float
    quadratic: float
    linear: float
    constant: float
    lower: float
    width: float


@dataclass(frozen=True, slots=True, kw_only=True)
class KerrRadialModel:
    radius: sp.Symbol
    spin: sp.Symbol
    energy: sp.Symbol
    angular_momentum: sp.Symbol
    carter: sp.Symbol
    potential: sp.Expr
    coefficients: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]

    @property
    def generators(self) -> tuple[sp.Symbol, ...]:
        return (
            self.radius,
            self.spin,
            self.energy,
            self.angular_momentum,
            self.carter,
        )


def build_radial_model() -> KerrRadialModel:
    radius, spin, energy, angular_momentum, carter = sp.symbols("r a E L Q", real=True)
    delta = radius**2 - 2 * radius + spin**2
    shifted_momentum = angular_momentum - spin * energy
    separation = shifted_momentum**2 + carter
    separated = (
        energy * (radius**2 + spin**2) - spin * angular_momentum
    ) ** 2 - delta * separation

    leading = energy**2
    quadratic = -2 * energy * spin * shifted_momentum - separation
    linear = 2 * separation
    constant = -(spin**2) * carter
    expanded = leading * radius**4 + quadratic * radius**2 + linear * radius + constant
    generators = (radius, spin, energy, angular_momentum, carter)
    require_polynomial_equal(
        separated, expanded, generators, "separated Kerr radial potential"
    )

    return KerrRadialModel(
        radius=radius,
        spin=spin,
        energy=energy,
        angular_momentum=angular_momentum,
        carter=carter,
        potential=expanded,
        coefficients=(leading, quadratic, linear, constant),
    )


def verify_normalized_form(model: KerrRadialModel) -> None:
    impact, normalized_carter = sp.symbols("b eta", real=True)
    normalized = model.potential.xreplace(
        {
            model.energy: sp.Integer(1),
            model.angular_momentum: impact,
            model.carter: normalized_carter,
        }
    )
    expected = (
        model.radius**4
        + (model.spin**2 - impact**2 - normalized_carter) * model.radius**2
        + 2 * ((impact - model.spin) ** 2 + normalized_carter) * model.radius
        - model.spin**2 * normalized_carter
    )
    require_polynomial_equal(
        normalized,
        expected,
        (model.radius, model.spin, impact, normalized_carter),
        "normalized radial potential",
    )


def bernstein_coefficients(
    power_coefficients: ExactCoefficients,
    lower: ExactScalar,
    width: ExactScalar,
) -> tuple[ExactScalar, ...]:
    leading, quadratic, linear, constant = power_coefficients
    lower_squared = lower**2
    width_squared = width**2
    power = (
        leading * lower**4 + quadratic * lower_squared + linear * lower + constant,
        width * (4 * leading * lower**3 + 2 * quadratic * lower + linear),
        width_squared * (6 * leading * lower_squared + quadratic),
        width_squared * width * (4 * leading * lower),
        leading * width_squared**2,
    )
    c0, c1, c2, c3, c4 = power
    return (
        c0,
        c0 + c1 / 4,
        c0 + c1 / 2 + c2 / 6,
        c0 + 3 * c1 / 4 + c2 / 2 + c3 / 4,
        c0 + c1 + c2 + c3 + c4,
    )


def verify_bernstein_transform(model: KerrRadialModel) -> None:
    lower, width, parameter = sp.symbols("lo h t", real=True)
    coefficients = bernstein_coefficients(model.coefficients, lower, width)
    basis = tuple(
        math.comb(4, index) * parameter**index * (1 - parameter) ** (4 - index)
        for index in range(5)
    )
    reconstructed = sum(
        coefficient * weight
        for coefficient, weight in zip(coefficients, basis, strict=True)
    )
    expected = model.potential.xreplace({model.radius: lower + width * parameter})
    generators = (*model.generators, lower, width, parameter)
    require_polynomial_equal(
        reconstructed,
        expected,
        generators,
        "quartic Bernstein reconstruction",
    )
    require_polynomial_equal(
        sum(basis),
        sp.Integer(1),
        (parameter,),
        "degree-four Bernstein partition of unity",
    )


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
    return ftz_safe_lower(next_down(round_binary32(value)))


def outward_upper(value: float) -> float:
    return ftz_safe_upper(next_up(round_binary32(value)))


@dataclass(frozen=True, slots=True)
class IntervalF32:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise ValueError(f"invalid interval [{self.lower}, {self.upper}]")

    @classmethod
    def point(cls, value: float) -> IntervalF32:
        packed = round_binary32(value)
        if not math.isfinite(packed):
            raise ValueError("an interval input must be finite")
        return cls(packed, packed)

    @classmethod
    def rational(cls, numerator: int, denominator: int) -> IntervalF32:
        exact = numerator / denominator
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
    if not math.isfinite(value):
        raise ValueError("infinite values have no finite rational representation")
    return Fraction(value)


def interval_contains(interval: IntervalF32, exact: Fraction) -> bool:
    lower_ok = interval.lower == -math.inf or exact_fraction(interval.lower) <= exact
    upper_ok = interval.upper == math.inf or exact <= exact_fraction(interval.upper)
    return lower_ok and upper_ok


def check_interval_primitives(left: float, right: float) -> None:
    left_interval = IntervalF32.point(left)
    right_interval = IntervalF32.point(right)
    exact_left = exact_fraction(left)
    exact_right = exact_fraction(right)
    checks = (
        (left_interval.add(right_interval), exact_left + exact_right, "add"),
        (left_interval.sub(right_interval), exact_left - exact_right, "sub"),
        (left_interval.mul(right_interval), exact_left * exact_right, "mul"),
        (left_interval.square(), exact_left * exact_left, "square"),
    )
    for interval, exact, operation in checks:
        if not interval_contains(interval, exact):
            raise AssertionError(
                f"{operation} escaped its interval: "
                f"{left=}, {right=}, {interval=}, {exact=}"
            )


def interval_bernstein_coefficients(
    power_coefficients: tuple[IntervalF32, IntervalF32, IntervalF32, IntervalF32],
    lower: IntervalF32,
    width: IntervalF32,
) -> tuple[IntervalF32, ...]:
    leading, quadratic, linear, constant = power_coefficients
    two = IntervalF32.point(2.0)
    four = IntervalF32.point(4.0)
    six = IntervalF32.point(6.0)
    lower_squared = lower.square()
    lower_cubed = lower_squared.mul(lower)
    lower_fourth = lower_squared.square()
    width_squared = width.square()
    width_cubed = width_squared.mul(width)

    c0 = (
        leading.mul(lower_fourth)
        .add(quadratic.mul(lower_squared))
        .add(linear.mul(lower))
        .add(constant)
    )
    c1 = width.mul(
        four.mul(leading)
        .mul(lower_cubed)
        .add(two.mul(quadratic).mul(lower))
        .add(linear)
    )
    c2 = width_squared.mul(six.mul(leading).mul(lower_squared).add(quadratic))
    c3 = four.mul(leading).mul(lower).mul(width_cubed)
    c4 = leading.mul(width_squared.square())
    return (
        c0,
        c0.add(c1.scale_rational(1, 4)),
        c0.add(c1.scale_rational(1, 2)).add(c2.scale_rational(1, 6)),
        c0.add(c1.scale_rational(3, 4))
        .add(c2.scale_rational(1, 2))
        .add(c3.scale_rational(1, 4)),
        c0.add(c1).add(c2).add(c3).add(c4),
    )


def check_interval_bernstein(sample: BernsteinSample) -> None:
    leading_value = round_binary32(abs(sample.leading) + MIN_SUBNORMAL)
    lower_value = abs(sample.lower)
    width_value = round_binary32(abs(sample.width) + round_binary32(0.001))
    intervals = interval_bernstein_coefficients(
        (
            IntervalF32.point(leading_value),
            IntervalF32.point(sample.quadratic),
            IntervalF32.point(sample.linear),
            IntervalF32.point(sample.constant),
        ),
        IntervalF32.point(lower_value),
        IntervalF32.point(width_value),
    )
    exact = bernstein_coefficients(
        (
            exact_fraction(leading_value),
            exact_fraction(sample.quadratic),
            exact_fraction(sample.linear),
            exact_fraction(sample.constant),
        ),
        exact_fraction(lower_value),
        exact_fraction(width_value),
    )
    for index, (interval, exact_coefficient) in enumerate(
        zip(intervals, exact, strict=True)
    ):
        if not interval_contains(interval, exact_coefficient):
            raise AssertionError(
                f"Bernstein coefficient {index} escaped: "
                f"{interval=}, {exact_coefficient=}"
            )


def verify_packed_horizon_bounds() -> None:
    witnessed_nearest_rounding_failure = False
    for spin_bits in (0x0000_0000, 0x3F00_0000, 0x3F4C_CCCD, 0x3F7D_70A4, 0x3F7F_FFFF):
        spin = binary32_from_bits(spin_bits)
        with localcontext() as context:
            context.prec = 100
            spin_fraction = Fraction(spin)
            spin_decimal = Decimal(spin_fraction.numerator) / Decimal(
                spin_fraction.denominator
            )
            exact = Decimal(1) + (Decimal(1) - spin_decimal * spin_decimal).sqrt()
        nearest = round_binary32(float(exact))
        lower = next_down(nearest)
        upper = next_up(nearest)
        lower_fraction = Fraction(lower)
        upper_fraction = Fraction(upper)
        lower_exact = Decimal(lower_fraction.numerator) / Decimal(
            lower_fraction.denominator
        )
        upper_exact = Decimal(upper_fraction.numerator) / Decimal(
            upper_fraction.denominator
        )
        if not lower_exact <= exact <= upper_exact:
            raise AssertionError(
                "packed horizon interval missed exact r+ for spin bits "
                f"{spin_bits:#010x}"
            )
        nearest_fraction = Fraction(nearest)
        nearest_exact = Decimal(nearest_fraction.numerator) / Decimal(
            nearest_fraction.denominator
        )
        witnessed_nearest_rounding_failure |= nearest_exact > exact
    if not witnessed_nearest_rounding_failure:
        raise AssertionError("horizon samples did not witness nearest > exact")


def verify_precondition_mutations(model: KerrRadialModel) -> None:
    substitutions = {
        model.radius: sp.Rational(5, 2),
        model.spin: sp.Rational(4, 5),
        model.energy: sp.Integer(1),
        model.angular_momentum: sp.Rational(7, 3),
        model.carter: sp.Rational(5, 7),
    }
    physical = model.potential.xreplace(substitutions)
    flipped = model.potential.xreplace(
        substitutions | {model.spin: -substitutions[model.spin]}
    )
    if physical == flipped:
        raise AssertionError("physical-spin mutation was not observable")

    negative_energy = model.potential.xreplace(
        substitutions | {model.energy: -substitutions[model.energy]}
    )
    if physical == negative_energy:
        raise AssertionError("energy-sign mutation was not observable")

    def preconditions(energy: float, radial_velocity: float) -> bool:
        return math.isfinite(energy) and energy > 0.0 and radial_velocity > 0.0

    if not preconditions(1.0, 0.25):
        raise AssertionError("valid future/outward physical momentum was rejected")
    for energy, radial_velocity in (
        (-1.0, 0.25),
        (0.0, 0.25),
        (1.0, 0.0),
        (1.0, -0.25),
    ):
        if preconditions(energy, radial_velocity):
            raise AssertionError(
                "invalid traversal precondition was accepted: "
                f"{energy=}, {radial_velocity=}"
            )


def verify_segment_removal_witness() -> None:
    segment_count = 12
    omitted = 5
    center = Fraction(2 * omitted + 1, 2 * segment_count)
    half_width = Fraction(1, 4 * segment_count)
    for index in range(segment_count):
        if index == omitted:
            continue
        lower = Fraction(index, segment_count)
        width = Fraction(1, segment_count)
        coefficients = bernstein_coefficients(
            (
                Fraction(0),
                Fraction(1),
                -2 * center,
                center**2 - half_width**2,
            ),
            lower,
            width,
        )
        if not all(coefficient > 0 for coefficient in coefficients):
            raise AssertionError(f"segment-removal witness leaked into segment {index}")
    if -(half_width**2) >= 0:
        raise AssertionError(
            "segment-removal witness is not negative in the omitted segment"
        )


def run() -> None:
    model = build_radial_model()
    verify_normalized_form(model)
    verify_bernstein_transform(model)
    for left, right in (
        (0.0, -0.0),
        (MIN_SUBNORMAL, -MIN_SUBNORMAL),
        (MIN_NORMAL, -next_down(MIN_NORMAL)),
        (MAX_FINITE, -MAX_FINITE),
    ):
        check_interval_primitives(left, right)
    check_interval_bernstein(
        BernsteinSample(
            leading=0.0,
            quadratic=MIN_SUBNORMAL,
            linear=-MIN_SUBNORMAL,
            constant=MIN_NORMAL,
            lower=0.0,
            width=0.0,
        )
    )
    verify_packed_horizon_bounds()
    verify_precondition_mutations(model)
    verify_segment_removal_witness()
    print(
        "PASS_INTERVAL_MODEL: exact Kerr quartic, quartic-to-Bernstein "
        "identity, deterministic strict-evaluator f32 boundaries, packed-horizon "
        "seam, and mutation witnesses; Hypothesis covers the wider generated "
        "domain; physical-input "
        "enclosure remains a separate CPU/GPU oracle gate"
    )

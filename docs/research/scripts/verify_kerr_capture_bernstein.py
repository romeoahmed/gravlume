"""Reproduce the algebra and primitive interval model of the Kerr capture path.

The exact polynomial identities are durable. ``IntervalF32`` models the WGSL
implementation's explicitly sequenced arithmetic primitives: each arithmetic
operation is consumed by a floating-point bit reinterpretation before the next
operation, so reassociation or fusion cannot legally cross the integer-bit
dependency. The script verifies those primitive enclosures and the
quartic-to-Bernstein transform. It does not by itself prove that every physical
initial ray lies inside the shader's deliberately wider invariant envelope;
that seam remains covered by the supported-domain fallback and CPU/GPU oracle
matrix. This is a research/validation tool, not a runtime dependency.
"""

from __future__ import annotations

import math
import random
import struct
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction

SEED = 0x4B434243  # "KCBC"
RANDOM_CASES = 20_000
MIN_NORMAL_F32 = 2.0**-126
MAX_FINITE_F32 = float.fromhex("0x1.fffffep+127")


Monomial = tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Polynomial:
    """Tiny exact sparse polynomial algebra used to keep this tool dependency-free."""

    terms: dict[Monomial, Fraction]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terms",
            {
                monomial: coefficient
                for monomial, coefficient in self.terms.items()
                if coefficient
            },
        )

    @classmethod
    def scalar(cls, value: int | Fraction) -> Polynomial:
        coefficient = Fraction(value)
        return cls({(): coefficient} if coefficient else {})

    @classmethod
    def variable(cls, name: str) -> Polynomial:
        return cls({((name, 1),): Fraction(1)})

    @staticmethod
    def coerce(value: Polynomial | int | Fraction) -> Polynomial:
        return value if isinstance(value, Polynomial) else Polynomial.scalar(value)

    def __add__(self, other: Polynomial | int | Fraction) -> Polynomial:
        result = dict(self.terms)
        for monomial, coefficient in self.coerce(other).terms.items():
            result[monomial] = result.get(monomial, Fraction(0)) + coefficient
        return Polynomial(result)

    def __radd__(self, other: Polynomial | int | Fraction) -> Polynomial:
        return self + other

    def __neg__(self) -> Polynomial:
        return Polynomial(
            {monomial: -coefficient for monomial, coefficient in self.terms.items()}
        )

    def __sub__(self, other: Polynomial | int | Fraction) -> Polynomial:
        return self + -self.coerce(other)

    def __rsub__(self, other: Polynomial | int | Fraction) -> Polynomial:
        return self.coerce(other) - self

    def __mul__(self, other: Polynomial | int | Fraction) -> Polynomial:
        result: dict[Monomial, Fraction] = {}
        for left_monomial, left_coefficient in self.terms.items():
            for right_monomial, right_coefficient in self.coerce(other).terms.items():
                powers: dict[str, int] = {}
                for name, exponent in left_monomial + right_monomial:
                    powers[name] = powers.get(name, 0) + exponent
                monomial = tuple(sorted(powers.items()))
                result[monomial] = result.get(monomial, Fraction(0)) + (
                    left_coefficient * right_coefficient
                )
        return Polynomial(result)

    def __rmul__(self, other: Polynomial | int | Fraction) -> Polynomial:
        return self * other

    def __truediv__(self, divisor: int | Fraction) -> Polynomial:
        divisor = Fraction(divisor)
        if divisor == 0:
            raise ZeroDivisionError
        return Polynomial(
            {
                monomial: coefficient / divisor
                for monomial, coefficient in self.terms.items()
            }
        )

    def __pow__(self, exponent: int) -> Polynomial:
        if exponent < 0:
            raise ValueError("polynomial exponent must be non-negative")
        result = Polynomial.scalar(1)
        factor = self
        remaining = exponent
        while remaining:
            if remaining & 1:
                result *= factor
            factor *= factor
            remaining >>= 1
        return result

    def substitute(
        self, replacements: dict[str, Polynomial | int | Fraction]
    ) -> Polynomial:
        result = Polynomial.scalar(0)
        for monomial, coefficient in self.terms.items():
            term = Polynomial.scalar(coefficient)
            for name, exponent in monomial:
                replacement = self.coerce(
                    replacements.get(name, Polynomial.variable(name))
                )
                term *= replacement**exponent
            result += term
        return result

    def evaluate(self, replacements: dict[str, int | Fraction]) -> Fraction:
        reduced = self.substitute(replacements)
        if any(monomial for monomial in reduced.terms):
            raise ValueError(f"evaluation left free variables: {reduced}")
        return reduced.terms.get((), Fraction(0))


def require_zero(expression: Polynomial, label: str) -> None:
    if expression.terms:
        raise AssertionError(f"{label} is nonzero: {expression.terms}")


@dataclass(frozen=True)
class RadialPolynomial:
    radius: str
    spin: str
    energy: str
    angular_momentum: str
    carter: str
    potential: Polynomial
    coefficients: tuple[Polynomial, Polynomial, Polynomial, Polynomial]


def build_radial_polynomial() -> RadialPolynomial:
    radius_name, spin_name, energy_name, angular_momentum_name, carter_name = (
        "r",
        "a",
        "E",
        "L",
        "Q",
    )
    radius = Polynomial.variable(radius_name)
    spin = Polynomial.variable(spin_name)
    energy = Polynomial.variable(energy_name)
    angular_momentum = Polynomial.variable(angular_momentum_name)
    carter = Polynomial.variable(carter_name)
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
    require_zero(separated - expanded, "separated Kerr radial potential")

    return RadialPolynomial(
        radius=radius_name,
        spin=spin_name,
        energy=energy_name,
        angular_momentum=angular_momentum_name,
        carter=carter_name,
        potential=expanded,
        coefficients=(leading, quadratic, linear, constant),
    )


def verify_normalized_form(polynomial: RadialPolynomial) -> None:
    radius = Polynomial.variable(polynomial.radius)
    spin = Polynomial.variable(polynomial.spin)
    impact = Polynomial.variable("b")
    normalized_carter = Polynomial.variable("eta")
    normalized = polynomial.potential.substitute(
        {
            polynomial.energy: 1,
            polynomial.angular_momentum: impact,
            polynomial.carter: normalized_carter,
        }
    )
    expected = (
        radius**4
        + (spin**2 - impact**2 - normalized_carter) * radius**2
        + 2 * ((impact - spin) ** 2 + normalized_carter) * radius
        - spin**2 * normalized_carter
    )
    require_zero(normalized - expected, "normalized radial potential")


def bernstein_coefficients(
    leading: Polynomial | Fraction | int,
    quadratic: Polynomial | Fraction | int,
    linear: Polynomial | Fraction | int,
    constant: Polynomial | Fraction | int,
    lower: Polynomial | Fraction | int,
    width: Polynomial | Fraction | int,
) -> tuple[Polynomial | Fraction, ...]:
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


def verify_bernstein_transform(polynomial: RadialPolynomial) -> None:
    lower = Polynomial.variable("lo")
    width = Polynomial.variable("h")
    parameter = Polynomial.variable("t")
    leading, quadratic, linear, constant = polynomial.coefficients
    coefficients = bernstein_coefficients(
        leading, quadratic, linear, constant, lower, width
    )
    basis = tuple(
        math.comb(4, index) * parameter**index * (1 - parameter) ** (4 - index)
        for index in range(5)
    )
    reconstructed = sum(
        coefficient * weight
        for coefficient, weight in zip(coefficients, basis, strict=True)
    )
    expected = polynomial.potential.substitute(
        {polynomial.radius: lower + width * parameter}
    )
    require_zero(reconstructed - expected, "quartic Bernstein reconstruction")
    require_zero(sum(basis) - 1, "degree-four Bernstein partition of unity")


def f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def next_down_f32(value: float) -> float:
    value = f32(value)
    if math.isnan(value) or value == -math.inf:
        return value
    if value == math.inf:
        return MAX_FINITE_F32
    if value == 0.0:
        return -f32_from_bits(1)
    bits = f32_bits(value)
    return f32_from_bits(bits + 1 if value < 0.0 else bits - 1)


def next_up_f32(value: float) -> float:
    value = f32(value)
    if math.isnan(value) or value == math.inf:
        return value
    if value == -math.inf:
        return -MAX_FINITE_F32
    if value == 0.0:
        return f32_from_bits(1)
    bits = f32_bits(value)
    return f32_from_bits(bits - 1 if value < 0.0 else bits + 1)


def ftz_safe_lower(value: float) -> float:
    """Widen a lower bound across implementations that may flush subnormals."""

    if not math.isfinite(value) or abs(value) >= MIN_NORMAL_F32:
        return value
    return -MIN_NORMAL_F32 if value <= 0.0 else 0.0


def ftz_safe_upper(value: float) -> float:
    """Widen an upper bound across implementations that may flush subnormals."""

    if not math.isfinite(value) or abs(value) >= MIN_NORMAL_F32:
        return value
    return MIN_NORMAL_F32 if value >= 0.0 else 0.0


def outward_lower(value: float) -> float:
    return ftz_safe_lower(next_down_f32(f32(value)))


def outward_upper(value: float) -> float:
    return ftz_safe_upper(next_up_f32(f32(value)))


@dataclass(frozen=True)
class IntervalF32:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise ValueError(f"invalid interval [{self.lower}, {self.upper}]")

    @classmethod
    def point(cls, value: float) -> IntervalF32:
        packed = f32(value)
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


def random_finite_f32(randomizer: random.Random) -> float:
    while True:
        bits = randomizer.getrandbits(32)
        value = f32_from_bits(bits)
        if math.isfinite(value) and abs(value) <= 2.0**40:
            return value


def verify_interval_primitives() -> None:
    randomizer = random.Random(SEED)
    for case in range(RANDOM_CASES):
        left = random_finite_f32(randomizer)
        right = random_finite_f32(randomizer)
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
                    f"{operation} escaped its interval at case {case}: "
                    f"{left=}, {right=}, {interval=}, {exact=}"
                )


def interval_bernstein_coefficients(
    leading: IntervalF32,
    quadratic: IntervalF32,
    linear: IntervalF32,
    constant: IntervalF32,
    lower: IntervalF32,
    width: IntervalF32,
) -> tuple[IntervalF32, ...]:
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


def verify_interval_bernstein() -> None:
    randomizer = random.Random(SEED ^ 0xB37)
    for case in range(5_000):
        values = [f32(randomizer.uniform(-8.0, 8.0)) for _ in range(6)]
        leading_value = abs(values[0]) + f32_from_bits(1)
        quadratic_value, linear_value, constant_value = values[1:4]
        lower_value = abs(values[4])
        width_value = abs(values[5]) + 0.001
        intervals = interval_bernstein_coefficients(
            IntervalF32.point(leading_value),
            IntervalF32.point(quadratic_value),
            IntervalF32.point(linear_value),
            IntervalF32.point(constant_value),
            IntervalF32.point(lower_value),
            IntervalF32.point(width_value),
        )
        exact = bernstein_coefficients(
            exact_fraction(leading_value),
            exact_fraction(quadratic_value),
            exact_fraction(linear_value),
            exact_fraction(constant_value),
            exact_fraction(lower_value),
            exact_fraction(width_value),
        )
        for index, (interval, exact_coefficient) in enumerate(
            zip(intervals, exact, strict=True)
        ):
            if not interval_contains(interval, exact_coefficient):
                raise AssertionError(
                    f"Bernstein coefficient {index} escaped at case {case}: "
                    f"{interval=}, {exact_coefficient=}"
                )


def verify_packed_horizon_bounds(polynomial: RadialPolynomial) -> None:
    del polynomial
    witnessed_nearest_rounding_failure = False
    for spin_bits in (0x0000_0000, 0x3F00_0000, 0x3F4C_CCCD, 0x3F7D_70A4, 0x3F7F_FFFF):
        spin = f32_from_bits(spin_bits)
        with localcontext() as context:
            context.prec = 100
            spin_fraction = Fraction(spin)
            spin_decimal = Decimal(spin_fraction.numerator) / Decimal(
                spin_fraction.denominator
            )
            exact = Decimal(1) + (Decimal(1) - spin_decimal * spin_decimal).sqrt()
        nearest = f32(float(exact))
        lower = next_down_f32(nearest)
        upper = next_up_f32(nearest)
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
                f"packed horizon interval missed exact r+ for spin bits {spin_bits:#010x}"
            )
        nearest_fraction = Fraction(nearest)
        nearest_exact = Decimal(nearest_fraction.numerator) / Decimal(
            nearest_fraction.denominator
        )
        witnessed_nearest_rounding_failure |= nearest_exact > exact
    if not witnessed_nearest_rounding_failure:
        raise AssertionError("horizon samples did not witness nearest > exact")


def verify_precondition_mutations(polynomial: RadialPolynomial) -> None:
    substitutions = {
        polynomial.radius: Fraction(5, 2),
        polynomial.spin: Fraction(4, 5),
        polynomial.energy: Fraction(1),
        polynomial.angular_momentum: Fraction(7, 3),
        polynomial.carter: Fraction(5, 7),
    }
    physical = polynomial.potential.evaluate(substitutions)
    flipped = polynomial.potential.evaluate(
        substitutions | {polynomial.spin: -substitutions[polynomial.spin]}
    )
    if physical == flipped:
        raise AssertionError("physical-spin mutation was not observable")

    negative_energy = polynomial.potential.evaluate(
        substitutions | {polynomial.energy: -substitutions[polynomial.energy]}
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
                f"invalid traversal precondition was accepted: {energy=}, {radial_velocity=}"
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
            0,
            1,
            -2 * center,
            center**2 - half_width**2,
            lower,
            width,
        )
        if not all(coefficient > 0 for coefficient in coefficients):
            raise AssertionError(f"segment-removal witness leaked into segment {index}")
    if -(half_width**2) >= 0:
        raise AssertionError(
            "segment-removal witness is not negative in the omitted segment"
        )


def main() -> None:
    polynomial = build_radial_polynomial()
    verify_normalized_form(polynomial)
    verify_bernstein_transform(polynomial)
    verify_interval_primitives()
    verify_interval_bernstein()
    verify_packed_horizon_bounds(polynomial)
    verify_precondition_mutations(polynomial)
    verify_segment_removal_witness()
    print(
        "PASS_INTERVAL_MODEL: exact Kerr quartic, quartic-to-Bernstein "
        f"identity, strict-evaluator f32 intervals ({RANDOM_CASES} primitive "
        "cases), packed-horizon seam, and mutation witnesses; physical-input "
        "enclosure remains a separate CPU/GPU oracle gate"
    )


if __name__ == "__main__":
    main()

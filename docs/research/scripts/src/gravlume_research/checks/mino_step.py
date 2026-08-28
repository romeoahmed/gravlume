"""Verify the order and cost model used to assess restricted Mino stepping."""

from math import factorial

import sympy as sp

from .._sympy import require_equal, require_zero

BASELINE_STEP_FACTOR = sp.Rational(3, 4)
STEP_FACTORS = (
    sp.Rational(5, 8),
    BASELINE_STEP_FACTOR,
    sp.Rational(13, 16),
    sp.Rational(27, 32),
    sp.Rational(219, 256),
    sp.Rational(7, 8),
    sp.Rational(29, 32),
    sp.Rational(59, 64),
    sp.Rational(119, 128),
    sp.Rational(15, 16),
    sp.Integer(1),
)


def lie_derivative(
    vector_field: sp.MatrixBase,
    expression: sp.MatrixBase,
    variables: tuple[sp.Symbol, ...],
) -> sp.ImmutableMatrix:
    return sp.ImmutableMatrix(
        [
            sp.expand(
                sum(
                    sp.diff(component, variable) * velocity
                    for variable, velocity in zip(variables, vector_field, strict=True)
                )
            )
            for component in expression
        ]
    )


def substitute_state(
    vector_field: sp.MatrixBase,
    variables: tuple[sp.Symbol, ...],
    state: sp.MatrixBase,
) -> sp.ImmutableMatrix:
    replacements = dict(zip(variables, state, strict=True))
    return sp.ImmutableMatrix(
        [component.xreplace(replacements) for component in vector_field]
    )


def fifth_order_coefficient(expression: sp.Expr, step: sp.Symbol) -> sp.Expr:
    truncated = sp.series(expression, step, 0, 6).removeO()
    polynomial = sp.Poly(truncated, step)
    for order in range(5):
        require_zero(polynomial.nth(order), f"step^{order} defect coefficient")
    leading = sp.factor(polynomial.nth(5))
    if leading == 0:
        raise AssertionError("step^5 defect coefficient unexpectedly vanished")
    return leading


def verify_rk4_local_order() -> tuple[sp.Expr, ...]:
    # x'=v, v'=A*x+B*x^2+C*x^3 contains the radial subsystem. The polar
    # subsystem is its B=0 specialization, so one proof covers both cores.
    position, velocity = sp.symbols("x velocity", real=True)
    linear, quadratic, cubic, step = sp.symbols(
        "coefficient_1 coefficient_2 coefficient_3 step", real=True
    )
    variables = (position, velocity)
    state = sp.ImmutableMatrix([position, velocity])
    vector_field = sp.ImmutableMatrix(
        [velocity, linear * position + quadratic * position**2 + cubic * position**3]
    )

    exact = state
    derivative = vector_field
    for order in range(1, 6):
        exact += step**order * derivative / factorial(order)
        derivative = lie_derivative(vector_field, derivative, variables)

    first = vector_field
    second = substitute_state(vector_field, variables, state + step * first / 2)
    third = substitute_state(vector_field, variables, state + step * second / 2)
    fourth = substitute_state(vector_field, variables, state + step * third)
    rk4 = state + step * (first + 2 * second + 2 * third + fourth) / 6

    return tuple(fifth_order_coefficient(component, step) for component in exact - rk4)


def verify_cubic_hermite_order() -> sp.Expr:
    step, fraction = sp.symbols("step fraction", real=True)
    derivatives = sp.symbols("y0:6", real=True)
    exact = sum(
        derivatives[order] * (fraction * step) ** order / factorial(order)
        for order in range(6)
    )
    endpoint = sum(
        derivatives[order] * step**order / factorial(order) for order in range(6)
    )
    endpoint_derivative = sum(
        derivatives[order] * step ** (order - 1) / factorial(order - 1)
        for order in range(1, 6)
    )
    squared = fraction**2
    cubed = fraction**3
    hermite = (
        (2 * cubed - 3 * squared + 1) * derivatives[0]
        + (cubed - 2 * squared + fraction) * step * derivatives[1]
        + (-2 * cubed + 3 * squared) * endpoint
        + (cubed - squared) * step * endpoint_derivative
    )

    defect = sp.Poly(sp.expand(exact - hermite), step)
    for order in range(4):
        require_zero(defect.nth(order), f"Hermite step^{order} defect coefficient")
    leading = sp.factor(defect.nth(4))
    expected = derivatives[4] * fraction**2 * (fraction - 1) ** 2 / 24
    require_equal(leading, expected, "Hermite leading defect")
    return leading


def print_factor_tradeoffs() -> None:
    print("factor,ideal_work_vs_0.75,truncation_envelope_vs_0.75")
    for factor in STEP_FACTORS:
        work = (BASELINE_STEP_FACTOR / factor).evalf(12)
        truncation = ((factor / BASELINE_STEP_FACTOR) ** 4).evalf(12)
        print(f"{factor},{work},{truncation}")


def run() -> None:
    local_defect = verify_rk4_local_order()
    hermite_defect = verify_cubic_hermite_order()
    print("RK4 polynomial-core defect starts at step^5: PASS")
    print(f"radial/polar leading coefficients: {list(local_defect)}")
    print("Cubic Hermite interior defect starts at step^4: PASS")
    print(f"Hermite leading coefficient: {hermite_defect}")
    print(
        "Global smooth-trajectory envelope is O(factor^4); "
        "ideal work is Theta(1/factor)."
    )
    print_factor_tradeoffs()
    print("RESULT=PASS")

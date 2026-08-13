"""Verify the order and cost model used to select the restricted Mino step factor."""

from math import factorial

import sympy as sp


def lie_derivative(
    field: sp.Matrix, expression: sp.Matrix, variables: tuple[sp.Symbol, ...]
) -> sp.Matrix:
    return expression.applyfunc(
        lambda component: sp.expand(
            sum(
                derivative * velocity
                for derivative, velocity in zip(
                    (sp.diff(component, variable) for variable in variables),
                    field,
                    strict=True,
                )
            )
        )
    )


def substitute(
    vector: sp.Matrix, variables: tuple[sp.Symbol, ...], state: sp.Matrix
) -> sp.Matrix:
    replacements = dict(zip(variables, state, strict=True))
    return vector.applyfunc(
        lambda component: component.subs(replacements, simultaneous=True)
    )


def verify_rk4_local_order() -> list[sp.Expr]:
    # x'=v, v'=A*x+B*x^2+C*x^3 contains the Mino radial subsystem.  The polar
    # subsystem is the B=0 specialization, so one proof covers both polynomial cores.
    x, velocity, coefficient_1, coefficient_2, coefficient_3, step = sp.symbols(
        "x velocity coefficient_1 coefficient_2 coefficient_3 step"
    )
    variables = (x, velocity)
    state = sp.Matrix([x, velocity])
    field = sp.Matrix(
        [
            velocity,
            coefficient_1 * x + coefficient_2 * x**2 + coefficient_3 * x**3,
        ]
    )

    exact = state
    derivative = field
    for order in range(1, 6):
        exact += step**order * derivative / factorial(order)
        derivative = lie_derivative(field, derivative, variables)

    first = field
    second = substitute(field, variables, state + step * first / 2)
    third = substitute(field, variables, state + step * second / 2)
    fourth = substitute(field, variables, state + step * third)
    rk4 = state + step * (first + 2 * second + 2 * third + fourth) / 6
    defect = (exact - rk4).applyfunc(
        lambda component: sp.series(component, step, 0, 6).removeO().expand()
    )

    for component in defect:
        for order in range(5):
            assert sp.expand(component).coeff(step, order) == 0
        assert sp.expand(component).coeff(step, 5) != 0
    return [sp.factor(component.coeff(step, 5)) for component in defect]


def verify_cubic_hermite_order() -> sp.Expr:
    step, fraction = sp.symbols("step fraction")
    derivatives = sp.symbols("y0:6")
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
    defect = sp.series(exact - hermite, step, 0, 6).removeO().expand()
    for order in range(4):
        assert defect.coeff(step, order) == 0
    leading = sp.factor(defect.coeff(step, 4))
    assert (
        sp.simplify(leading - derivatives[4] * fraction**2 * (fraction - 1) ** 2 / 24)
        == 0
    )
    return leading


def print_factor_tradeoffs() -> None:
    baseline = sp.Rational(3, 4)
    factors = [
        sp.Rational(5, 8),
        baseline,
        sp.Rational(13, 16),
        sp.Rational(27, 32),
        sp.Rational(219, 256),
        sp.Rational(7, 8),
        sp.Rational(29, 32),
        sp.Rational(59, 64),
        sp.Rational(119, 128),
        sp.Rational(15, 16),
        sp.Integer(1),
    ]
    print("factor,ideal_work_vs_0.75,truncation_envelope_vs_0.75")
    for factor in factors:
        work = sp.N(baseline / factor, 12)
        truncation = sp.N((factor / baseline) ** 4, 12)
        print(f"{factor},{work},{truncation}")


def main() -> None:
    local_defect = verify_rk4_local_order()
    hermite_defect = verify_cubic_hermite_order()
    print("RK4 polynomial-core defect starts at step^5: PASS")
    print(f"radial/polar leading coefficients: {local_defect}")
    print("Cubic Hermite interior defect starts at step^4: PASS")
    print(f"Hermite leading coefficient: {hermite_defect}")
    print(
        "Global smooth-trajectory envelope is O(factor^4); ideal work is Theta(1/factor)."
    )
    print_factor_tradeoffs()
    print("RESULT=PASS")


if __name__ == "__main__":
    main()

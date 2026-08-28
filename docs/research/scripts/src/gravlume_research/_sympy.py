"""Explicit proof contracts shared by Gravlume's SymPy checks.

The reducers deliberately follow SymPy's programmatic best practices: exact
quantities stay symbolic, transformations are targeted, and numerical
substitutions are passed directly to ``evalf``.
"""

from collections.abc import Callable, Mapping, Sequence

import sympy as sp

type ExpressionReducer = Callable[[sp.Expr], sp.Expr]
type Substitutions = Mapping[sp.Symbol, sp.Expr]


def rational_form(expression: sp.Expr) -> sp.Expr:
    """Reduce a rational expression to a canonical numerator/denominator form."""

    return sp.cancel(expression)


def trigonometric_rational_form(expression: sp.Expr) -> sp.Expr:
    """Apply the trigonometric identities needed by the oblate-coordinate proof."""

    return sp.cancel(sp.trigsimp(expression))


def require_zero(
    expression: sp.Expr,
    label: str,
    *,
    reduce: ExpressionReducer = rational_form,
) -> None:
    reduced = reduce(expression)
    if reduced != 0:
        raise AssertionError(f"{label} is nonzero: {reduced}")


def require_equal(
    actual: sp.Expr,
    expected: sp.Expr,
    label: str,
    *,
    reduce: ExpressionReducer = rational_form,
) -> None:
    require_zero(actual - expected, label, reduce=reduce)


def require_matrix_equal(
    actual: sp.MatrixBase,
    expected: sp.MatrixBase,
    label: str,
    *,
    reduce: ExpressionReducer = rational_form,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label} has shape {actual.shape}, expected {expected.shape}"
        )
    for row in range(actual.rows):
        for column in range(actual.cols):
            require_equal(
                actual[row, column],
                expected[row, column],
                f"{label}[{row},{column}]",
                reduce=reduce,
            )


def require_polynomial_equal(
    actual: sp.Expr,
    expected: sp.Expr,
    generators: Sequence[sp.Symbol],
    label: str,
) -> None:
    difference = sp.Poly(actual - expected, *generators, domain=sp.QQ)
    if not difference.is_zero:
        raise AssertionError(f"{label} is nonzero: {difference.as_expr()}")


def evaluate_real(
    expression: sp.Expr,
    substitutions: Substitutions,
    precision_digits: int,
) -> sp.Expr:
    value = expression.evalf(
        precision_digits,
        subs=dict(substitutions),
        maxn=2 * precision_digits,
    )
    if value.is_real is not True or value.is_finite is not True:
        raise AssertionError(f"expression did not evaluate to a finite real: {value}")
    return value


def maximum_relative_residual(
    left: Sequence[sp.Expr] | sp.MatrixBase,
    right: Sequence[sp.Expr] | sp.MatrixBase,
    substitutions: Substitutions,
    precision_digits: int,
) -> sp.Float:
    """Evaluate independent expressions before subtracting to expose cancellation."""

    left_expressions = list(left)
    right_expressions = list(right)
    if len(left_expressions) != len(right_expressions):
        raise AssertionError("residual operands have different lengths")

    worst = sp.Float(0, precision_digits)
    one = sp.Float(1, precision_digits)
    for left_expression, right_expression in zip(
        left_expressions, right_expressions, strict=True
    ):
        left_value = evaluate_real(left_expression, substitutions, precision_digits)
        right_value = evaluate_real(right_expression, substitutions, precision_digits)
        scale = max(one, abs(left_value), abs(right_value))
        residual = abs(left_value - right_value) / scale
        worst = max(worst, residual)
    return worst

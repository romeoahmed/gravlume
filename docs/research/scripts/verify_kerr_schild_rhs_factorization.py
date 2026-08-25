"""Prove the exact algebra used by the reduced Kerr-Schild RK4 shader.

The checks cover the six-dimensional canonical Hamilton system, compact
``J_l.T * p`` contraction, geometry gradients, globally axis-regular Carter
invariant, and cubic Hermite event interpolation.  Every identity is exact;
the script does not model binary32 rounding or GPU execution time.
"""

from __future__ import annotations

import sympy as sp
from sympy_checks import require_equal, require_matrix_equal


def verify_sigma_identity() -> None:
    radial_square, oblate_offset = sp.symbols("u A", nonzero=True, real=True)
    spin_vertical_square = radial_square * (radial_square - oblate_offset)
    sigma_from_radius = radial_square + spin_vertical_square / radial_square
    sigma_from_root = 2 * radial_square - oblate_offset

    require_equal(sigma_from_radius, sigma_from_root, "Sigma reconstruction")
    require_equal(
        sigma_from_root**2,
        oblate_offset**2 + 4 * spin_vertical_square,
        "Sigma discriminant root",
    )


def verify_geometry_gradients() -> None:
    oblate_offset, sigma = sp.symbols("B Sigma", nonzero=True, real=True)
    spin, z = sp.symbols("a z", real=True)
    x, y = sp.symbols("x y", real=True)
    mass, numerator = sp.symbols("M N", real=True)
    radius_gradient = sp.ImmutableMatrix(sp.symbols("r_x r_y r_z", real=True))
    sigma_gradient = sp.ImmutableMatrix(
        [
            2 * oblate_offset * x / sigma,
            2 * oblate_offset * y / sigma,
            2 * z * (oblate_offset + 2 * spin**2) / sigma,
        ]
    )

    discriminant_gradients = sp.ImmutableMatrix(
        [
            4 * oblate_offset * x,
            4 * oblate_offset * y,
            4 * z * (oblate_offset + 2 * spin**2),
        ]
    )
    require_matrix_equal(
        sigma_gradient,
        discriminant_gradients / (2 * sigma),
        "Sigma discriminant gradient",
    )

    numerator_gradient = 2 * mass * radius_gradient
    quotient_gradient = (
        numerator_gradient * sigma - numerator * sigma_gradient
    ) / sigma**2
    scalar_f = numerator / sigma
    factorized_gradient = (numerator_gradient - scalar_f * sigma_gradient) / sigma
    require_matrix_equal(
        factorized_gradient,
        quotient_gradient,
        "factorized Kerr-Schild scalar gradient",
    )


def null_derivative_contractions(
    branch_sign: int,
) -> tuple[sp.ImmutableMatrix, sp.ImmutableMatrix]:
    radius, spin, scale = sp.symbols("r a scale", nonzero=True, real=True)
    x, y, z = sp.symbols("x y z", real=True)
    radius_x, radius_y, radius_z = sp.symbols("r_x r_y r_z", real=True)
    momentum_x, momentum_y, momentum_z = sp.symbols("p_x p_y p_z", real=True)

    coordinates = sp.ImmutableMatrix([x, y, z])
    radius_gradient = sp.ImmutableMatrix([radius_x, radius_y, radius_z])
    momentum = sp.ImmutableMatrix([momentum_x, momentum_y, momentum_z])
    radial_denominator = radius**2 + spin**2
    chart_spin = branch_sign * spin
    base_null = sp.ImmutableMatrix(
        [
            (radius * x + chart_spin * y) / radial_denominator,
            (radius * y - chart_spin * x) / radial_denominator,
            z / radius,
        ]
    )

    coordinate_gradients: list[sp.ImmutableMatrix] = []
    for index, radius_derivative in enumerate(radius_gradient):
        delta_x = sp.Integer(index == 0)
        delta_y = sp.Integer(index == 1)
        delta_z = sp.Integer(index == 2)
        radial_denominator_derivative = 2 * radius * radius_derivative
        numerator_x_derivative = (
            radius_derivative * x + radius * delta_x + chart_spin * delta_y
        )
        numerator_y_derivative = (
            radius_derivative * y + radius * delta_y - chart_spin * delta_x
        )
        coordinate_gradients.append(
            sp.ImmutableMatrix(
                [
                    branch_sign
                    * (
                        numerator_x_derivative
                        - base_null[0] * radial_denominator_derivative
                    )
                    / (radial_denominator * scale),
                    branch_sign
                    * (
                        numerator_y_derivative
                        - base_null[1] * radial_denominator_derivative
                    )
                    / (radial_denominator * scale),
                    branch_sign
                    * (delta_z / radius - z * radius_derivative / radius**2)
                    / scale,
                ]
            )
        )
    coordinate_wise = sp.ImmutableMatrix(
        [momentum.dot(gradient) for gradient in coordinate_gradients]
    )

    radial_coefficient = (
        coordinates[:2, :].dot(momentum[:2, :])
        - 2 * radius * base_null[:2, :].dot(momentum[:2, :])
    ) / radial_denominator - z * momentum_z / radius**2
    direct = sp.ImmutableMatrix(
        [
            (radius * momentum_x - chart_spin * momentum_y) / radial_denominator,
            (chart_spin * momentum_x + radius * momentum_y) / radial_denominator,
            momentum_z / radius,
        ]
    )
    factorized = branch_sign * (direct + radial_coefficient * radius_gradient) / scale
    return coordinate_wise, factorized


def verify_factorization() -> None:
    for branch_sign in (-1, 1):
        coordinate_wise, factorized = null_derivative_contractions(branch_sign)
        require_matrix_equal(
            coordinate_wise,
            factorized,
            f"branch {branch_sign} null derivative contraction",
        )


def verify_schwarzschild_limit() -> None:
    radius = sp.symbols("r", nonzero=True, real=True)
    x, y, z = sp.symbols("x y z", real=True)
    momentum_x, momentum_y, momentum_z = sp.symbols("p_x p_y p_z", real=True)
    position = sp.ImmutableMatrix([x, y, z])
    momentum = sp.ImmutableMatrix([momentum_x, momentum_y, momentum_z])
    radial_unit = position / radius
    expected = (momentum - radial_unit * radial_unit.dot(momentum)) / radius

    direct = momentum / radius
    radial_coefficient = -position.dot(momentum) / radius**2
    factorized = direct + radial_coefficient * radial_unit
    require_matrix_equal(factorized, expected, "Schwarzschild radial projector")


def verify_reduced_hamiltonian() -> None:
    coordinates = sp.symbols("x y z", real=True)
    momentum = sp.ImmutableMatrix(sp.symbols("p_x p_y p_z", real=True))
    energy = sp.symbols("E", real=True)
    scalar_f = sp.Function("f")(*coordinates)
    null_spatial = sp.ImmutableMatrix(
        [sp.Function(f"ell_{axis}")(*coordinates) for axis in "xyz"]
    )
    contraction = energy + null_spatial.dot(momentum)
    hamiltonian = (
        -(energy**2) + momentum.dot(momentum) - scalar_f * contraction**2
    ) / 2

    position_rhs = sp.ImmutableMatrix(
        [sp.diff(hamiltonian, component) for component in momentum]
    )
    expected_position_rhs = momentum - scalar_f * contraction * null_spatial
    require_matrix_equal(
        position_rhs,
        expected_position_rhs,
        "six-dimensional position Hamilton equation",
    )

    coordinate_time_rhs = -sp.diff(hamiltonian, energy)
    require_equal(
        coordinate_time_rhs,
        energy + scalar_f * contraction,
        "coordinate-time Hamilton equation",
    )

    momentum_rhs = sp.ImmutableMatrix(
        [-sp.diff(hamiltonian, coordinate) for coordinate in coordinates]
    )
    contracted_null_gradient = sp.ImmutableMatrix(
        [
            sum(
                momentum[component] * sp.diff(null_spatial[component], coordinate)
                for component in range(3)
            )
            for coordinate in coordinates
        ]
    )
    scalar_f_gradient = sp.ImmutableMatrix(
        [sp.diff(scalar_f, coordinate) for coordinate in coordinates]
    )
    expected_momentum_rhs = (
        scalar_f * contraction * contracted_null_gradient
        + contraction**2 * scalar_f_gradient / 2
    )
    require_matrix_equal(
        momentum_rhs,
        expected_momentum_rhs,
        "six-dimensional momentum Hamilton equation",
    )


def verify_axis_regular_carter_invariant() -> None:
    radius, spin, energy = sp.symbols("r a E", nonzero=True, real=True)
    x, y, z = sp.symbols("x y z", real=True)
    momentum_x, momentum_y, momentum_z = sp.symbols("p_x p_y p_z", real=True)
    radial_factor = radius**2 + spin**2
    rho_squared = x**2 + y**2
    projected_momentum = x * momentum_x + y * momentum_y
    angular_momentum = x * momentum_y - y * momentum_x
    transverse_momentum_squared = momentum_x**2 + momentum_y**2
    cos_theta = z / radius
    sin_theta_squared = rho_squared / radial_factor

    p_theta_squared = (
        radius**2 * sin_theta_squared * momentum_z**2
        - 2 * z * projected_momentum * momentum_z
        + cos_theta**2 * projected_momentum**2 / sin_theta_squared
    )
    trigonometric = p_theta_squared + cos_theta**2 * (
        angular_momentum**2 / sin_theta_squared - spin**2 * energy**2
    )
    axis_regular = (
        cos_theta**2
        * (radial_factor * transverse_momentum_squared - spin**2 * energy**2)
        - 2 * z * projected_momentum * momentum_z
        + radius**2 * sin_theta_squared * momentum_z**2
    )
    require_equal(
        projected_momentum**2 + angular_momentum**2,
        rho_squared * transverse_momentum_squared,
        "transverse dot/cross identity",
    )
    require_equal(axis_regular, trigonometric, "axis-regular Carter invariant")

    angular_momentum_x = y * momentum_z - z * momentum_y
    angular_momentum_y = z * momentum_x - x * momentum_z
    schwarzschild = sp.factor(
        axis_regular.subs(spin, 0).subs(radius**2, rho_squared + z**2)
    )
    require_equal(
        schwarzschild,
        angular_momentum_x**2 + angular_momentum_y**2,
        "Schwarzschild Carter limit",
    )

    axis_value = axis_regular.subs({x: 0, y: 0, z**2: radius**2})
    require_equal(
        axis_value,
        radial_factor * transverse_momentum_squared - spin**2 * energy**2,
        "direct Carter axis value",
    )


def verify_kerr_newman_radial_potential() -> None:
    radius, mass, spin, charge = sp.symbols("r M a q_e", real=True)
    energy, angular_momentum, carter = sp.symbols("E L_z Q", real=True)
    separation = (angular_momentum - spin * energy) ** 2 + carter
    delta = radius**2 - 2 * mass * radius + spin**2 + charge**2
    potential = (
        energy * (radius**2 + spin**2) - spin * angular_momentum
    ) ** 2 - delta * separation
    expanded = (
        energy**2 * radius**4
        + (-2 * energy * spin * (angular_momentum - spin * energy) - separation)
        * radius**2
        + 2 * mass * separation * radius
        - spin**2 * carter
        - charge**2 * separation
    )
    require_equal(potential, expanded, "Kerr-Newman radial quartic")


def verify_cubic_hermite_event_interpolant() -> None:
    theta, step = sp.symbols("theta h", nonzero=True, real=True)
    coordinate = sp.symbols("lambda", real=True)
    coefficients = sp.symbols("c0:5", real=True)
    polynomial = sum(
        coefficient * coordinate**degree
        for degree, coefficient in enumerate(coefficients)
    )
    start = polynomial.subs(coordinate, 0)
    end = polynomial.subs(coordinate, step)
    derivative = sp.diff(polynomial, coordinate)
    start_derivative = derivative.subs(coordinate, 0)
    end_derivative = derivative.subs(coordinate, step)
    hermite = (
        (2 * theta**3 - 3 * theta**2 + 1) * start
        + (theta**3 - 2 * theta**2 + theta) * step * start_derivative
        + (-2 * theta**3 + 3 * theta**2) * end
        + (theta**3 - theta**2) * step * end_derivative
    )
    exact = polynomial.subs(coordinate, theta * step)
    expanded_hermite = sp.expand(hermite.subs(coefficients[4], 0))
    expanded_exact = sp.expand(exact.subs(coefficients[4], 0))
    for degree in range(4):
        require_equal(
            expanded_hermite.coeff(coefficients[degree]),
            expanded_exact.coeff(coefficients[degree]),
            f"Hermite degree {degree}",
        )
    require_equal(
        hermite - exact,
        -coefficients[4] * step**4 * theta**2 * (theta - 1) ** 2,
        "Hermite quartic defect",
    )
    start_value, end_value, start_slope, end_slope = sp.symbols(
        "y0 y1 m0 m1", real=True
    )
    cubic = (
        (2 * theta**3 - 3 * theta**2 + 1) * start_value
        + (theta**3 - 2 * theta**2 + theta) * start_slope
        + (-2 * theta**3 + 3 * theta**2) * end_value
        + (theta**3 - theta**2) * end_slope
    )
    middle_slope = 3 * (end_value - start_value) - start_slope - end_slope
    bernstein_derivative = (
        (1 - theta) ** 2 * start_slope
        + 2 * theta * (1 - theta) * middle_slope
        + theta**2 * end_slope
    )
    require_equal(
        sp.diff(cubic, theta),
        bernstein_derivative,
        "Hermite monotonicity control polygon",
    )


def main() -> None:
    verify_sigma_identity()
    verify_geometry_gradients()
    verify_factorization()
    verify_schwarzschild_limit()
    verify_reduced_hamiltonian()
    verify_axis_regular_carter_invariant()
    verify_kerr_newman_radial_potential()
    verify_cubic_hermite_event_interpolant()
    print("Sigma discriminant identity: PASS")
    print("Sigma and Kerr-Schild scalar gradients: PASS")
    print("Both Kerr-Schild branch contractions: PASS")
    print("Schwarzschild projector limit: PASS")
    print("Six-dimensional Hamilton system: PASS")
    print("Axis-regular Carter invariant: PASS")
    print("Kerr-Newman radial quartic: PASS")
    print("Cubic Hermite event order: PASS")
    print("RESULT=PASS")


if __name__ == "__main__":
    main()

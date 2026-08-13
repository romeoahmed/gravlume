"""Reproduce the formal checks in docs/research/kerr-schild-mino-map.md.

This is deliberately a research tool rather than production code. It proves
both the legacy outgoing handedness defect and the corrected, physical-spin
zero-step seam. Every failed identity raises and makes the process exit nonzero.
"""

from __future__ import annotations

import platform
import random
from dataclasses import dataclass

import sympy as sp

SEED = 0x4B534D53  # ASCII-ish "KSMS"
PRECISION_DIGITS = 180
BOUNDARY_TOLERANCE = sp.Float("1e-80", PRECISION_DIGITS)


def canonical(expr: sp.Expr) -> sp.Expr:
    """Normalize the rational/trigonometric expressions used in this proof."""

    return sp.cancel(sp.trigsimp(sp.factor(expr)))


def require_zero(expr: sp.Expr, label: str) -> None:
    reduced = canonical(expr)
    if reduced != 0:
        raise AssertionError(f"{label} is nonzero: {reduced}")


def require_zero_matrix(matrix: sp.Matrix, label: str) -> None:
    for row in range(matrix.rows):
        for column in range(matrix.cols):
            require_zero(matrix[row, column], f"{label}[{row},{column}]")


def max_normalized_residual(
    left: sp.Matrix | list[sp.Expr],
    right: sp.Matrix | list[sp.Expr],
    substitutions: dict[sp.Symbol, sp.Expr],
) -> sp.Expr:
    """Evaluate two independently-built expressions before subtracting them."""

    left_values = list(left) if isinstance(left, sp.MatrixBase) else left
    right_values = list(right) if isinstance(right, sp.MatrixBase) else right
    if len(left_values) != len(right_values):
        raise AssertionError("residual operands have different lengths")

    worst = sp.Float(0, PRECISION_DIGITS)
    one = sp.Float(1, PRECISION_DIGITS)
    for lhs, rhs in zip(left_values, right_values, strict=True):
        lhs_value = sp.N(lhs.subs(substitutions), PRECISION_DIGITS)
        rhs_value = sp.N(rhs.subs(substitutions), PRECISION_DIGITS)
        scale = max(one, abs(lhs_value), abs(rhs_value))
        residual = sp.N(abs(lhs_value - rhs_value) / scale, PRECISION_DIGITS)
        worst = max(worst, residual)
    return worst


@dataclass(frozen=True)
class Geometry:
    mass: sp.Symbol
    radius: sp.Symbol
    spin: sp.Symbol
    sin_theta_squared: sp.Symbol
    sigma: sp.Expr
    delta: sp.Expr
    ks_metric: dict[int, sp.Matrix]
    bl_metric: dict[int, sp.Matrix]
    jacobian: dict[int, sp.Matrix]
    pullback: dict[int, sp.Matrix]
    legacy_outgoing_pullback: sp.Matrix


def build_geometry() -> Geometry:
    mass, radius, spin, sin_theta_squared = sp.symbols(
        "M r a u", real=True, nonzero=True
    )
    sigma = radius**2 + spin**2 * (1 - sin_theta_squared)
    delta = radius**2 - 2 * mass * radius + spin**2
    ks_metric: dict[int, sp.Matrix] = {}
    bl_metric: dict[int, sp.Matrix] = {}
    jacobian: dict[int, sp.Matrix] = {}
    pullback: dict[int, sp.Matrix] = {}
    for branch in (1, -1):
        # a is always the physical BL spin. The oblate spatial twist is s*a.
        flat_metric = sp.Matrix(
            [
                [-1, 0, 0, 0],
                [0, 1, 0, -branch * spin * sin_theta_squared],
                [0, 0, sigma, 0],
                [
                    0,
                    -branch * spin * sin_theta_squared,
                    0,
                    (radius**2 + spin**2) * sin_theta_squared,
                ],
            ]
        )
        principal_covector = sp.Matrix([1, branch, 0, -spin * sin_theta_squared])
        ks_metric[branch] = (
            flat_metric
            + (2 * mass * radius / sigma) * principal_covector * principal_covector.T
        )

        bl = sp.zeros(4)
        bl[0, 0] = -1 + 2 * mass * radius / sigma
        bl[0, 3] = bl[3, 0] = -2 * mass * radius * spin * sin_theta_squared / sigma
        bl[1, 1] = sigma / delta
        bl[2, 2] = sigma
        bl[3, 3] = (
            sin_theta_squared
            * ((radius**2 + spin**2) ** 2 - spin**2 * delta * sin_theta_squared)
            / sigma
        )
        bl_metric[branch] = bl

        # q_s = (t_s, r, theta, phi_s), q_B = (t_B, r, theta, phi_B).
        jacobian[branch] = sp.Matrix(
            [
                [1, branch * 2 * mass * radius / delta, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, branch * spin / delta, 0, 1],
            ]
        )
        pullback[branch] = jacobian[branch].T * ks_metric[branch] * jacobian[branch]

    # Legacy outgoing used the ingoing spatial twist (+a) while reversing the
    # whole principal spatial covector. Its pullback is kept only as a RED
    # witness against the physical-spin (+a) BL metric.
    legacy_flat_metric = sp.Matrix(
        [
            [-1, 0, 0, 0],
            [0, 1, 0, -spin * sin_theta_squared],
            [0, 0, sigma, 0],
            [
                0,
                -spin * sin_theta_squared,
                0,
                (radius**2 + spin**2) * sin_theta_squared,
            ],
        ]
    )
    legacy_principal = sp.Matrix([1, -1, 0, spin * sin_theta_squared])
    legacy_metric = (
        legacy_flat_metric
        + (2 * mass * radius / sigma) * legacy_principal * legacy_principal.T
    )
    legacy_jacobian = sp.Matrix(
        [
            [1, -2 * mass * radius / delta, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, spin / delta, 0, 1],
        ]
    )

    return Geometry(
        mass=mass,
        radius=radius,
        spin=spin,
        sin_theta_squared=sin_theta_squared,
        sigma=sigma,
        delta=delta,
        ks_metric=ks_metric,
        bl_metric=bl_metric,
        jacobian=jacobian,
        pullback=pullback,
        legacy_outgoing_pullback=(legacy_jacobian.T * legacy_metric * legacy_jacobian),
    )


def verify_metric_pullback(geometry: Geometry) -> None:
    for branch in (1, -1):
        require_zero_matrix(
            geometry.pullback[branch] - geometry.bl_metric[branch],
            f"metric pullback, branch={branch}",
        )


def verify_legacy_same_spin_outgoing_mismatch(geometry: Geometry) -> sp.Expr:
    mass = geometry.mass
    radius = geometry.radius
    spin = geometry.spin
    u = geometry.sin_theta_squared
    sigma = geometry.sigma
    mismatch = 4 * mass * radius * spin * u / sigma
    expected = sp.zeros(4)
    expected[0, 3] = expected[3, 0] = mismatch
    require_zero_matrix(
        geometry.legacy_outgoing_pullback - geometry.bl_metric[-1] - expected,
        "legacy same-spin outgoing mismatch support",
    )

    sample = canonical(
        mismatch.subs(
            {
                mass: sp.Integer(1),
                radius: sp.Integer(5),
                spin: sp.Rational(2, 3),
                u: sp.Rational(3, 7),
            }
        )
    )
    if sample != sp.Rational(360, 1591):
        raise AssertionError(f"unexpected mismatch sample: {sample}")
    return mismatch


def verify_cartesian_oblate_map(geometry: Geometry) -> None:
    radius, spin = geometry.radius, geometry.spin
    theta, azimuth = sp.symbols("theta phi_s", real=True)
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)
    vr, vtheta, vazimuth = sp.symbols("v_r v_theta v_phi", real=True)

    sin_theta, cos_theta = sp.sin(theta), sp.cos(theta)
    cos_phi, sin_phi = sp.cos(azimuth), sp.sin(azimuth)
    variables = (radius, theta, azimuth)
    sigma = radius**2 + spin**2 * cos_theta**2
    cartesian_covector = sp.Matrix([px, py, pz])
    spheroidal_tangent = sp.Matrix([vr, vtheta, vazimuth])

    for branch in (1, -1):
        chart_spin = branch * spin
        x = (radius * cos_phi - chart_spin * sin_phi) * sin_theta
        y = (radius * sin_phi + chart_spin * cos_phi) * sin_theta
        z = radius * cos_theta
        cartesian = sp.Matrix([x, y, z])
        basis = cartesian.jacobian(variables)

        expected_flat_spatial = sp.Matrix(
            [
                [1, 0, -chart_spin * sin_theta**2],
                [0, sigma, 0],
                [
                    -chart_spin * sin_theta**2,
                    0,
                    (radius**2 + spin**2) * sin_theta**2,
                ],
            ]
        )
        require_zero_matrix(
            basis.T * basis - expected_flat_spatial,
            f"Cartesian oblate flat metric, branch={branch}",
        )

        principal_cartesian = sp.Matrix(
            [
                (branch * radius * x + spin * y) / (radius**2 + spin**2),
                (branch * radius * y - spin * x) / (radius**2 + spin**2),
                branch * z / radius,
            ]
        )
        require_zero_matrix(
            basis.T * principal_cartesian
            - sp.Matrix([branch, 0, -spin * sin_theta**2]),
            f"Cartesian principal covector, branch={branch}",
        )

        spheroidal_covector = basis.T * cartesian_covector
        expected_covector = sp.Matrix(
            [
                sin_theta * (cos_phi * px + sin_phi * py) + cos_theta * pz,
                sp.cot(theta) * (x * px + y * py) - radius * sin_theta * pz,
                x * py - y * px,
            ]
        )
        require_zero_matrix(
            spheroidal_covector - expected_covector,
            f"Cartesian to oblate covector, branch={branch}",
        )

        rho_squared = x**2 + y**2
        grad_radius = sp.Matrix(
            [
                x * radius / sigma,
                y * radius / sigma,
                z * (radius**2 + spin**2) / (radius * sigma),
            ]
        )
        grad_theta = sp.Matrix(
            [
                cos_theta * x / (sin_theta * sigma),
                cos_theta * y / (sin_theta * sigma),
                -radius * sin_theta / sigma,
            ]
        )
        grad_azimuth = (
            sp.Matrix([-y / rho_squared, x / rho_squared, 0])
            + chart_spin / (radius**2 + spin**2) * grad_radius
        )
        inverse_basis = sp.Matrix.vstack(grad_radius.T, grad_theta.T, grad_azimuth.T)
        require_zero_matrix(
            inverse_basis * basis - sp.eye(3),
            f"oblate gradients, branch={branch}",
        )

        cartesian_tangent = basis * spheroidal_tangent
        require_zero(
            (cartesian_covector.T * cartesian_tangent)[0]
            - (spheroidal_covector.T * spheroidal_tangent)[0],
            f"Cartesian/oblate tangent-covector pairing, branch={branch}",
        )


def verify_bl_tangent_covector_duality(geometry: Geometry) -> None:
    kt, kr, ktheta, kphi = sp.symbols("k_t k_r k_theta k_phi", real=True)
    pt, pr, ptheta, pphi = sp.symbols("p_t p_r p_theta p_phi", real=True)
    tangent_bl = sp.Matrix([kt, kr, ktheta, kphi])
    covector_ks = sp.Matrix([pt, pr, ptheta, pphi])

    for branch in (1, -1):
        jacobian = geometry.jacobian[branch]
        tangent_ks = jacobian * tangent_bl
        covector_bl = jacobian.T * covector_ks
        require_zero(
            (covector_ks.T * tangent_ks)[0] - (covector_bl.T * tangent_bl)[0],
            f"BL/KS dual pairing, branch={branch}",
        )
        require_zero_matrix(
            tangent_ks.T * geometry.ks_metric[branch] * tangent_ks
            - tangent_bl.T * geometry.bl_metric[branch] * tangent_bl,
            f"BL/KS tangent norm, branch={branch}",
        )

        expected_covector_bl = sp.Matrix(
            [
                pt,
                pr
                + branch * 2 * geometry.mass * geometry.radius / geometry.delta * pt
                + branch * geometry.spin / geometry.delta * pphi,
                ptheta,
                pphi,
            ]
        )
        require_zero_matrix(
            covector_bl - expected_covector_bl,
            f"BL covector formula, branch={branch}",
        )
        require_zero_matrix(
            jacobian.inv() * tangent_ks - tangent_bl,
            f"BL tangent round trip, branch={branch}",
        )
        require_zero_matrix(
            jacobian.T.inv() * covector_bl - covector_ks,
            f"BL covector round trip, branch={branch}",
        )


@dataclass(frozen=True)
class MinoSystem:
    energy: sp.Symbol
    impact: sp.Symbol
    carter: sp.Symbol
    mu: sp.Symbol
    radial_velocity: sp.Symbol
    polar_velocity: sp.Symbol
    bl_inverse_metric: dict[int, sp.Matrix]
    covector: dict[int, sp.Matrix]
    radial_potential: dict[int, sp.Expr]
    polar_potential: dict[int, sp.Expr]
    hamiltonian_identity_left: dict[int, sp.Expr]
    hamiltonian_identity_right: dict[int, sp.Expr]


def build_and_verify_mino_system(geometry: Geometry) -> MinoSystem:
    mass, radius, spin = geometry.mass, geometry.radius, geometry.spin
    energy = sp.symbols("E", real=True, nonzero=True)
    impact, carter = sp.symbols("b eta", real=True)
    mu, radial_velocity, polar_velocity = sp.symbols("mu v_r v_mu", real=True)
    sin_squared = 1 - mu**2
    delta = geometry.delta

    inverse_metrics: dict[int, sp.Matrix] = {}
    covectors: dict[int, sp.Matrix] = {}
    radial_potentials: dict[int, sp.Expr] = {}
    polar_potentials: dict[int, sp.Expr] = {}
    identity_left: dict[int, sp.Expr] = {}
    identity_right: dict[int, sp.Expr] = {}

    for branch in (1, -1):
        bl_spin = spin
        sigma = radius**2 + bl_spin**2 * mu**2
        inverse = sp.zeros(4)
        inverse[0, 0] = -(
            (radius**2 + bl_spin**2) ** 2 - bl_spin**2 * delta * sin_squared
        ) / (sigma * delta)
        inverse[0, 3] = inverse[3, 0] = -2 * mass * bl_spin * radius / (sigma * delta)
        inverse[1, 1] = delta / sigma
        inverse[2, 2] = 1 / sigma
        inverse[3, 3] = (delta - bl_spin**2 * sin_squared) / (
            sigma * delta * sin_squared
        )
        inverse_metrics[branch] = inverse

        momentum = sp.Matrix(
            [
                -energy,
                energy * radial_velocity / delta,
                -energy * polar_velocity / sp.sqrt(sin_squared),
                energy * impact,
            ]
        )
        covectors[branch] = momentum

        p_function = radius**2 + bl_spin**2 - bl_spin * impact
        a_function = (impact - bl_spin) ** 2 + carter
        radial_potential = p_function**2 - delta * a_function
        polar_potential = (
            carter + (bl_spin**2 - carter - impact**2) * mu**2 - bl_spin**2 * mu**4
        )
        radial_potentials[branch] = radial_potential
        polar_potentials[branch] = polar_potential

        hamiltonian = (momentum.T * inverse * momentum)[0] / 2
        separated = (radial_velocity**2 - radial_potential) / delta + (
            polar_velocity**2 - polar_potential
        ) / sin_squared
        identity_left[branch] = 2 * sigma * hamiltonian / energy**2
        identity_right[branch] = separated
        require_zero(
            identity_left[branch] - identity_right[branch],
            f"separated null Hamiltonian, branch={branch}",
        )

        # d/dtau = (Sigma/E) d/dsigma.  The canonical BL equations give
        # dr/dsigma = Delta p_r/Sigma and dmu/dsigma =
        # -sin(theta) p_theta/Sigma.
        affine_r = inverse[1, 1] * momentum[1]
        affine_mu = -sp.sqrt(sin_squared) * inverse[2, 2] * momentum[2]
        require_zero(
            sigma / energy * affine_r - radial_velocity,
            f"affine/Mino radial scale, branch={branch}",
        )
        require_zero(
            sigma / energy * affine_mu - polar_velocity,
            f"affine/Mino polar scale, branch={branch}",
        )
        require_zero(
            sp.diff(radial_potential, radius) / 2
            - (2 * radius * p_function - (radius - mass) * a_function),
            f"Mino radial acceleration, branch={branch}",
        )
        require_zero(
            sp.diff(polar_potential, mu) / 2
            - ((bl_spin**2 - carter - impact**2) * mu - 2 * bl_spin**2 * mu**3),
            f"Mino polar acceleration, branch={branch}",
        )

    return MinoSystem(
        energy=energy,
        impact=impact,
        carter=carter,
        mu=mu,
        radial_velocity=radial_velocity,
        polar_velocity=polar_velocity,
        bl_inverse_metric=inverse_metrics,
        covector=covectors,
        radial_potential=radial_potentials,
        polar_potential=polar_potentials,
        hamiltonian_identity_left=identity_left,
        hamiltonian_identity_right=identity_right,
    )


def random_float(rng: random.Random, low: float, high: float) -> sp.Float:
    value = low + (high - low) * rng.random()
    return sp.Float(f"{value:.17g}", PRECISION_DIGITS)


def verify_boundary_substitutions(
    geometry: Geometry, mino: MinoSystem
) -> dict[str, tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr]]:
    """Stress outgoing expressions at defined points near three chart seams."""

    rng = random.Random(SEED)
    mass_value = sp.Float("1", PRECISION_DIGITS)
    ten = sp.Integer(10)

    axis_spin = random_float(rng, 0.65, 0.85)
    axis_radius = random_float(rng, 3.0, 9.0)
    axis_u = (1 + random_float(rng, 0.0, 1.0)) / ten**70

    horizon_spin = random_float(rng, 0.65, 0.85)
    horizon_plus = mass_value + sp.sqrt(mass_value**2 - horizon_spin**2)
    horizon_radius = horizon_plus + (1 + random_float(rng, 0.0, 1.0)) / ten**60
    horizon_u = random_float(rng, 0.2, 0.8)

    extremal_spin = mass_value - (1 + random_float(rng, 0.0, 1.0)) / ten**60
    extremal_plus = mass_value + sp.sqrt(mass_value**2 - extremal_spin**2)
    extremal_radius = extremal_plus + (1 + random_float(rng, 0.0, 1.0)) / ten**50
    extremal_u = random_float(rng, 0.2, 0.8)

    cases = {
        "near_axis": (axis_spin, axis_radius, axis_u),
        "near_horizon": (horizon_spin, horizon_radius, horizon_u),
        "near_extremality": (extremal_spin, extremal_radius, extremal_u),
    }

    kt, kr, ktheta, kphi = sp.symbols("k_t k_r k_theta k_phi", real=True)
    pt, pr, ptheta, pphi = sp.symbols("p_t p_r p_theta p_phi", real=True)
    tangent_bl = sp.Matrix([kt, kr, ktheta, kphi])
    covector_ks = sp.Matrix([pt, pr, ptheta, pphi])
    outgoing_jacobian = geometry.jacobian[-1]
    tangent_ks = outgoing_jacobian * tangent_bl
    covector_bl = outgoing_jacobian.T * covector_ks
    dual_left = [(covector_ks.T * tangent_ks)[0]]
    dual_right = [(covector_bl.T * tangent_bl)[0]]

    results: dict[str, tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr]] = {}
    for name, (spin_value, radius_value, u_value) in cases.items():
        geometry_substitutions = {
            geometry.mass: mass_value,
            geometry.spin: spin_value,
            geometry.radius: radius_value,
            geometry.sin_theta_squared: u_value,
        }
        sigma_value = sp.N(
            geometry.sigma.subs(geometry_substitutions), PRECISION_DIGITS
        )
        delta_value = sp.N(
            geometry.delta.subs(geometry_substitutions), PRECISION_DIGITS
        )
        denominators = (
            sigma_value,
            delta_value,
            u_value,
            radius_value,
            radius_value**2 + spin_value**2,
            (radius_value**2 + spin_value**2) * u_value,
        )
        if any(value == 0 for value in denominators):
            raise AssertionError(f"{name}: substitution hit a coordinate denominator")

        metric_residual = max_normalized_residual(
            geometry.pullback[-1],
            geometry.bl_metric[-1],
            geometry_substitutions,
        )

        state_substitutions = dict(geometry_substitutions)
        for symbol in (kt, kr, ktheta, kphi, pt, pr, ptheta, pphi):
            state_substitutions[symbol] = random_float(rng, -2.0, 2.0)
        dual_residual = max_normalized_residual(
            dual_left, dual_right, state_substitutions
        )

        mu_value = sp.sqrt(1 - u_value)
        energy_value = random_float(rng, 0.8, 1.8)
        impact_value = sp.Float("0", PRECISION_DIGITS)
        carter_value = random_float(rng, 0.5, 2.0)
        potential_substitutions = {
            **geometry_substitutions,
            mino.energy: energy_value,
            mino.impact: impact_value,
            mino.carter: carter_value,
            mino.mu: mu_value,
        }
        radial_potential_value = sp.N(
            mino.radial_potential[-1].subs(potential_substitutions),
            PRECISION_DIGITS,
        )
        polar_potential_value = sp.N(
            mino.polar_potential[-1].subs(potential_substitutions),
            PRECISION_DIGITS,
        )
        if radial_potential_value <= 0 or polar_potential_value <= 0:
            raise AssertionError(f"{name}: seeded Mino point is outside R,U >= 0")
        potential_substitutions[mino.radial_velocity] = sp.sqrt(radial_potential_value)
        potential_substitutions[mino.polar_velocity] = -sp.sqrt(polar_potential_value)
        hamiltonian_residual = max_normalized_residual(
            [mino.hamiltonian_identity_left[-1]],
            [mino.hamiltonian_identity_right[-1]],
            potential_substitutions,
        )
        sin_squared = 1 - mino.mu**2
        sigma = geometry.radius**2 + geometry.spin**2 * mino.mu**2
        momentum = mino.covector[-1]
        inverse = mino.bl_inverse_metric[-1]
        affine_mino_left = [
            sigma / mino.energy * inverse[1, 1] * momentum[1],
            sigma / mino.energy * (-sp.sqrt(sin_squared) * inverse[2, 2] * momentum[2]),
        ]
        affine_mino_right = [mino.radial_velocity, mino.polar_velocity]
        affine_residual = max_normalized_residual(
            affine_mino_left,
            affine_mino_right,
            potential_substitutions,
        )
        mino_residual = max(hamiltonian_residual, affine_residual)

        for label, residual in (
            ("metric", metric_residual),
            ("duality", dual_residual),
            ("Mino", mino_residual),
        ):
            if residual >= BOUNDARY_TOLERANCE:
                raise AssertionError(
                    f"{name}: {label} residual {residual} >= {BOUNDARY_TOLERANCE}"
                )
        extremality_gap = sp.N(mass_value**2 - spin_value**2, PRECISION_DIGITS)
        results[name] = (
            u_value,
            abs(delta_value),
            abs(extremality_gap),
            metric_residual,
            dual_residual,
            mino_residual,
        )

    return results


def short_scientific(value: sp.Expr) -> str:
    if value == 0:
        return "0"
    return str(sp.N(value, 8))


def main() -> None:
    geometry = build_geometry()
    verify_metric_pullback(geometry)
    mismatch = verify_legacy_same_spin_outgoing_mismatch(geometry)
    verify_cartesian_oblate_map(geometry)
    verify_bl_tangent_covector_duality(geometry)
    mino = build_and_verify_mino_system(geometry)
    boundary_results = verify_boundary_substitutions(geometry, mino)

    print(f"python={platform.python_version()}")
    print(f"sympy={sp.__version__}")
    print("symbolic.metric_pullback=PASS branches=ingoing,outgoing")
    print("symbolic.cartesian_oblate_map=PASS")
    print("symbolic.tangent_covector_duality=PASS")
    print("symbolic.affine_mino=PASS branches=ingoing,outgoing")
    print("symbolic.corrected_physical_spin=PASS branches=ingoing,outgoing")
    print(f"symbolic.legacy_outgoing=RED_AS_EXPECTED mismatch={mismatch}")
    print("symbolic.legacy_outgoing_sample=RED_AS_EXPECTED g_tphi=g_phit=360/1591")
    print(f"boundary.seed=0x{SEED:08X} precision_digits={PRECISION_DIGITS}")
    for name, (
        u_value,
        abs_delta,
        extremality_gap,
        metric,
        duality,
        mino_residual,
    ) in boundary_results.items():
        print(
            f"boundary.{name}=PASS "
            f"u={short_scientific(u_value)} "
            f"abs_delta={short_scientific(abs_delta)} "
            f"M2_minus_a2={short_scientific(extremality_gap)} "
            f"metric={short_scientific(metric)} "
            f"duality={short_scientific(duality)} "
            f"mino={short_scientific(mino_residual)}"
        )
    print("RESULT=PASS")


if __name__ == "__main__":
    main()

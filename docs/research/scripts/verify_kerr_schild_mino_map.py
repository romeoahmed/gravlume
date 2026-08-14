"""Reproduce the formal checks in docs/research/kerr-schild-mino-map.md.

This is deliberately a research tool rather than production code. It proves
both the legacy outgoing handedness defect and the corrected, physical-spin
zero-step seam. Every failed identity raises and makes the process exit nonzero.
"""

from __future__ import annotations

import platform
import random
from dataclasses import dataclass
from enum import IntEnum

import sympy as sp
from sympy_checks import (
    evaluate_real,
    maximum_relative_residual,
    rational_form,
    require_equal,
    require_matrix_equal,
    require_zero,
    trigonometric_rational_form,
)

SEED = 0x4B534D53  # ASCII-ish "KSMS"
PRECISION_DIGITS = 180
BOUNDARY_TOLERANCE = sp.Rational(1, 10**80)


class Chart(IntEnum):
    INGOING = 1
    OUTGOING = -1


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    spin: sp.Expr
    radius: sp.Expr
    sin_theta_squared: sp.Expr


@dataclass(frozen=True)
class BoundaryProbe:
    state_symbols: tuple[sp.Symbol, ...]
    duality_left: tuple[sp.Expr, ...]
    duality_right: tuple[sp.Expr, ...]
    affine_mino_left: tuple[sp.Expr, ...]
    affine_mino_right: tuple[sp.Expr, ...]


@dataclass(frozen=True)
class BoundaryResult:
    sin_theta_squared: sp.Expr
    absolute_delta: sp.Expr
    extremality_gap: sp.Expr
    metric_residual: sp.Expr
    duality_residual: sp.Expr
    mino_residual: sp.Expr


@dataclass(frozen=True)
class Geometry:
    mass: sp.Symbol
    radius: sp.Symbol
    spin: sp.Symbol
    sin_theta_squared: sp.Symbol
    sigma: sp.Expr
    delta: sp.Expr
    ks_metrics: dict[Chart, sp.MatrixBase]
    bl_metric: sp.MatrixBase
    jacobians: dict[Chart, sp.MatrixBase]
    pullbacks: dict[Chart, sp.MatrixBase]
    legacy_outgoing_pullback: sp.MatrixBase


def build_geometry() -> Geometry:
    mass, radius = sp.symbols("M r", positive=True)
    spin = sp.symbols("a", real=True)
    sin_theta_squared = sp.symbols("u", positive=True)
    sigma = radius**2 + spin**2 * (1 - sin_theta_squared)
    delta = radius**2 - 2 * mass * radius + spin**2

    bl_metric = sp.zeros(4)
    bl_metric[0, 0] = -1 + 2 * mass * radius / sigma
    bl_metric[0, 3] = bl_metric[3, 0] = (
        -2 * mass * radius * spin * sin_theta_squared / sigma
    )
    bl_metric[1, 1] = sigma / delta
    bl_metric[2, 2] = sigma
    bl_metric[3, 3] = (
        sin_theta_squared
        * ((radius**2 + spin**2) ** 2 - spin**2 * delta * sin_theta_squared)
        / sigma
    )

    ks_metrics: dict[Chart, sp.MatrixBase] = {}
    jacobians: dict[Chart, sp.MatrixBase] = {}
    pullbacks: dict[Chart, sp.MatrixBase] = {}
    for chart in Chart:
        branch = int(chart)
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
        ks_metric = (
            flat_metric
            + (2 * mass * radius / sigma) * principal_covector * principal_covector.T
        )

        # q_s = (t_s, r, theta, phi_s), q_B = (t_B, r, theta, phi_B).
        jacobian = sp.Matrix(
            [
                [1, branch * 2 * mass * radius / delta, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, branch * spin / delta, 0, 1],
            ]
        )
        ks_metrics[chart] = ks_metric
        jacobians[chart] = jacobian
        pullbacks[chart] = jacobian.T * ks_metric * jacobian

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
        ks_metrics=ks_metrics,
        bl_metric=bl_metric.as_immutable(),
        jacobians=jacobians,
        pullbacks=pullbacks,
        legacy_outgoing_pullback=(legacy_jacobian.T * legacy_metric * legacy_jacobian),
    )


def verify_metric_pullback(geometry: Geometry) -> None:
    for chart in Chart:
        require_matrix_equal(
            geometry.pullbacks[chart],
            geometry.bl_metric,
            f"metric pullback, chart={chart.name.lower()}",
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
    require_matrix_equal(
        geometry.legacy_outgoing_pullback - geometry.bl_metric,
        expected,
        "legacy same-spin outgoing mismatch support",
    )

    sample = rational_form(
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

    for chart in Chart:
        branch = int(chart)
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
        require_matrix_equal(
            basis.T * basis,
            expected_flat_spatial,
            f"Cartesian oblate flat metric, chart={chart.name.lower()}",
            reduce=trigonometric_rational_form,
        )

        principal_cartesian = sp.Matrix(
            [
                (branch * radius * x + spin * y) / (radius**2 + spin**2),
                (branch * radius * y - spin * x) / (radius**2 + spin**2),
                branch * z / radius,
            ]
        )
        require_matrix_equal(
            basis.T * principal_cartesian,
            sp.Matrix([branch, 0, -spin * sin_theta**2]),
            f"Cartesian principal covector, chart={chart.name.lower()}",
            reduce=trigonometric_rational_form,
        )

        spheroidal_covector = basis.T * cartesian_covector
        expected_covector = sp.Matrix(
            [
                sin_theta * (cos_phi * px + sin_phi * py) + cos_theta * pz,
                sp.cot(theta) * (x * px + y * py) - radius * sin_theta * pz,
                x * py - y * px,
            ]
        )
        require_matrix_equal(
            spheroidal_covector,
            expected_covector,
            f"Cartesian to oblate covector, chart={chart.name.lower()}",
            reduce=trigonometric_rational_form,
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
        require_matrix_equal(
            inverse_basis * basis,
            sp.eye(3),
            f"oblate gradients, chart={chart.name.lower()}",
            reduce=trigonometric_rational_form,
        )

        cartesian_tangent = basis * spheroidal_tangent
        require_zero(
            (cartesian_covector.T * cartesian_tangent)[0]
            - (spheroidal_covector.T * spheroidal_tangent)[0],
            f"Cartesian/oblate tangent-covector pairing, chart={chart.name.lower()}",
            reduce=trigonometric_rational_form,
        )


def verify_bl_tangent_covector_duality(geometry: Geometry) -> None:
    kt, kr, ktheta, kphi = sp.symbols("k_t k_r k_theta k_phi", real=True)
    pt, pr, ptheta, pphi = sp.symbols("p_t p_r p_theta p_phi", real=True)
    tangent_bl = sp.Matrix([kt, kr, ktheta, kphi])
    covector_ks = sp.Matrix([pt, pr, ptheta, pphi])

    for chart in Chart:
        branch = int(chart)
        jacobian = geometry.jacobians[chart]
        tangent_ks = jacobian * tangent_bl
        covector_bl = jacobian.T * covector_ks
        require_zero(
            (covector_ks.T * tangent_ks)[0] - (covector_bl.T * tangent_bl)[0],
            f"BL/KS dual pairing, chart={chart.name.lower()}",
        )
        require_matrix_equal(
            tangent_ks.T * geometry.ks_metrics[chart] * tangent_ks,
            tangent_bl.T * geometry.bl_metric * tangent_bl,
            f"BL/KS tangent norm, chart={chart.name.lower()}",
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
        require_matrix_equal(
            covector_bl,
            expected_covector_bl,
            f"BL covector formula, chart={chart.name.lower()}",
        )
        require_matrix_equal(
            jacobian.inv() * tangent_ks,
            tangent_bl,
            f"BL tangent round trip, chart={chart.name.lower()}",
        )
        require_matrix_equal(
            jacobian.T.inv() * covector_bl,
            covector_ks,
            f"BL covector round trip, chart={chart.name.lower()}",
        )


@dataclass(frozen=True)
class MinoSystem:
    energy: sp.Symbol
    impact: sp.Symbol
    carter: sp.Symbol
    mu: sp.Symbol
    radial_velocity: sp.Symbol
    polar_velocity: sp.Symbol
    bl_inverse_metric: sp.MatrixBase
    covector: sp.MatrixBase
    radial_potential: sp.Expr
    polar_potential: sp.Expr
    hamiltonian_identity_left: sp.Expr
    hamiltonian_identity_right: sp.Expr


def build_and_verify_mino_system(geometry: Geometry) -> MinoSystem:
    mass, radius, spin = geometry.mass, geometry.radius, geometry.spin
    energy = sp.symbols("E", real=True, nonzero=True)
    impact, carter = sp.symbols("b eta", real=True)
    mu, radial_velocity, polar_velocity = sp.symbols("mu v_r v_mu", real=True)
    sin_squared = 1 - mu**2
    delta = geometry.delta

    sigma = radius**2 + spin**2 * mu**2
    inverse = sp.zeros(4)
    inverse[0, 0] = -((radius**2 + spin**2) ** 2 - spin**2 * delta * sin_squared) / (
        sigma * delta
    )
    inverse[0, 3] = inverse[3, 0] = -2 * mass * spin * radius / (sigma * delta)
    inverse[1, 1] = delta / sigma
    inverse[2, 2] = 1 / sigma
    inverse[3, 3] = (delta - spin**2 * sin_squared) / (sigma * delta * sin_squared)

    momentum = sp.Matrix(
        [
            -energy,
            energy * radial_velocity / delta,
            -energy * polar_velocity / sp.sqrt(sin_squared),
            energy * impact,
        ]
    )
    radial_factor = radius**2 + spin**2 - spin * impact
    separation = (impact - spin) ** 2 + carter
    radial_potential = radial_factor**2 - delta * separation
    polar_potential = carter + (spin**2 - carter - impact**2) * mu**2 - spin**2 * mu**4

    hamiltonian = (momentum.T * inverse * momentum)[0] / 2
    separated_hamiltonian = (radial_velocity**2 - radial_potential) / delta + (
        polar_velocity**2 - polar_potential
    ) / sin_squared
    hamiltonian_identity = 2 * sigma * hamiltonian / energy**2
    require_equal(
        hamiltonian_identity,
        separated_hamiltonian,
        "separated null Hamiltonian",
    )

    # d/dtau = (Sigma/E) d/dsigma. The canonical BL equations give
    # dr/dsigma = Delta p_r/Sigma and dmu/dsigma = -sin(theta) p_theta/Sigma.
    affine_r = inverse[1, 1] * momentum[1]
    affine_mu = -sp.sqrt(sin_squared) * inverse[2, 2] * momentum[2]
    require_equal(
        sigma / energy * affine_r,
        radial_velocity,
        "affine/Mino radial scale",
    )
    require_equal(
        sigma / energy * affine_mu,
        polar_velocity,
        "affine/Mino polar scale",
    )
    require_equal(
        sp.diff(radial_potential, radius) / 2,
        2 * radius * radial_factor - (radius - mass) * separation,
        "Mino radial acceleration",
    )
    require_equal(
        sp.diff(polar_potential, mu) / 2,
        (spin**2 - carter - impact**2) * mu - 2 * spin**2 * mu**3,
        "Mino polar acceleration",
    )

    return MinoSystem(
        energy=energy,
        impact=impact,
        carter=carter,
        mu=mu,
        radial_velocity=radial_velocity,
        polar_velocity=polar_velocity,
        bl_inverse_metric=inverse.as_immutable(),
        covector=momentum.as_immutable(),
        radial_potential=radial_potential,
        polar_potential=polar_potential,
        hamiltonian_identity_left=hamiltonian_identity,
        hamiltonian_identity_right=separated_hamiltonian,
    )


def random_rational(rng: random.Random, low: float, high: float) -> sp.Rational:
    value = low + (high - low) * rng.random()
    return sp.Rational(f"{value:.17g}")


def build_boundary_cases(rng: random.Random) -> tuple[BoundaryCase, ...]:
    ten = sp.Integer(10)

    axis_spin = random_rational(rng, 0.65, 0.85)
    axis_radius = random_rational(rng, 3.0, 9.0)
    axis_u = (1 + random_rational(rng, 0.0, 1.0)) / ten**70

    horizon_spin = random_rational(rng, 0.65, 0.85)
    horizon_plus = 1 + sp.sqrt(1 - horizon_spin**2)
    horizon_radius = horizon_plus + (1 + random_rational(rng, 0.0, 1.0)) / ten**60
    horizon_u = random_rational(rng, 0.2, 0.8)

    extremal_spin = 1 - (1 + random_rational(rng, 0.0, 1.0)) / ten**60
    extremal_plus = 1 + sp.sqrt(1 - extremal_spin**2)
    extremal_radius = extremal_plus + (1 + random_rational(rng, 0.0, 1.0)) / ten**50
    extremal_u = random_rational(rng, 0.2, 0.8)

    return (
        BoundaryCase("near_axis", axis_spin, axis_radius, axis_u),
        BoundaryCase("near_horizon", horizon_spin, horizon_radius, horizon_u),
        BoundaryCase(
            "near_extremality",
            extremal_spin,
            extremal_radius,
            extremal_u,
        ),
    )


def build_boundary_probe(geometry: Geometry, mino: MinoSystem) -> BoundaryProbe:
    kt, kr, ktheta, kphi = sp.symbols("k_t k_r k_theta k_phi", real=True)
    pt, pr, ptheta, pphi = sp.symbols("p_t p_r p_theta p_phi", real=True)
    tangent_bl = sp.Matrix([kt, kr, ktheta, kphi])
    covector_ks = sp.Matrix([pt, pr, ptheta, pphi])
    outgoing_jacobian = geometry.jacobians[Chart.OUTGOING]
    tangent_ks = outgoing_jacobian * tangent_bl
    covector_bl = outgoing_jacobian.T * covector_ks
    sin_squared = 1 - mino.mu**2
    sigma = geometry.radius**2 + geometry.spin**2 * mino.mu**2
    momentum = mino.covector
    inverse = mino.bl_inverse_metric
    return BoundaryProbe(
        state_symbols=(kt, kr, ktheta, kphi, pt, pr, ptheta, pphi),
        duality_left=((covector_ks.T * tangent_ks)[0],),
        duality_right=((covector_bl.T * tangent_bl)[0],),
        affine_mino_left=(
            sigma / mino.energy * inverse[1, 1] * momentum[1],
            sigma / mino.energy * (-sp.sqrt(sin_squared) * inverse[2, 2] * momentum[2]),
        ),
        affine_mino_right=(mino.radial_velocity, mino.polar_velocity),
    )


def require_boundary_tolerance(name: str, result: BoundaryResult) -> None:
    for label, residual in (
        ("metric", result.metric_residual),
        ("duality", result.duality_residual),
        ("Mino", result.mino_residual),
    ):
        if residual >= BOUNDARY_TOLERANCE:
            raise AssertionError(
                f"{name}: {label} residual {residual} >= {BOUNDARY_TOLERANCE}"
            )


def evaluate_boundary_case(
    geometry: Geometry,
    mino: MinoSystem,
    probe: BoundaryProbe,
    rng: random.Random,
    case: BoundaryCase,
) -> BoundaryResult:
    name = case.name
    mass_value = sp.Integer(1)
    spin_value = case.spin
    radius_value = case.radius
    u_value = case.sin_theta_squared
    geometry_substitutions = {
        geometry.mass: mass_value,
        geometry.spin: spin_value,
        geometry.radius: radius_value,
        geometry.sin_theta_squared: u_value,
    }
    sigma_value = evaluate_real(
        geometry.sigma, geometry_substitutions, PRECISION_DIGITS
    )
    delta_value = evaluate_real(
        geometry.delta, geometry_substitutions, PRECISION_DIGITS
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

    metric_residual = maximum_relative_residual(
        geometry.pullbacks[Chart.OUTGOING],
        geometry.bl_metric,
        geometry_substitutions,
        PRECISION_DIGITS,
    )

    state_substitutions = dict(geometry_substitutions)
    for symbol in probe.state_symbols:
        state_substitutions[symbol] = random_rational(rng, -2.0, 2.0)
    duality_residual = maximum_relative_residual(
        probe.duality_left,
        probe.duality_right,
        state_substitutions,
        PRECISION_DIGITS,
    )

    potential_substitutions = {
        **geometry_substitutions,
        mino.energy: random_rational(rng, 0.8, 1.8),
        mino.impact: sp.Integer(0),
        mino.carter: random_rational(rng, 0.5, 2.0),
        mino.mu: sp.sqrt(1 - u_value),
    }
    radial_potential = mino.radial_potential.xreplace(potential_substitutions)
    polar_potential = mino.polar_potential.xreplace(potential_substitutions)
    radial_potential_value = evaluate_real(
        radial_potential,
        {},
        PRECISION_DIGITS,
    )
    polar_potential_value = evaluate_real(
        polar_potential,
        {},
        PRECISION_DIGITS,
    )
    if radial_potential_value <= 0 or polar_potential_value <= 0:
        raise AssertionError(f"{name}: seeded Mino point is outside R,U >= 0")
    potential_substitutions[mino.radial_velocity] = sp.sqrt(radial_potential)
    potential_substitutions[mino.polar_velocity] = -sp.sqrt(polar_potential)
    hamiltonian_residual = maximum_relative_residual(
        (mino.hamiltonian_identity_left,),
        (mino.hamiltonian_identity_right,),
        potential_substitutions,
        PRECISION_DIGITS,
    )
    affine_residual = maximum_relative_residual(
        probe.affine_mino_left,
        probe.affine_mino_right,
        potential_substitutions,
        PRECISION_DIGITS,
    )
    result = BoundaryResult(
        sin_theta_squared=u_value,
        absolute_delta=abs(delta_value),
        extremality_gap=abs((mass_value**2 - spin_value**2).evalf(PRECISION_DIGITS)),
        metric_residual=metric_residual,
        duality_residual=duality_residual,
        mino_residual=max(hamiltonian_residual, affine_residual),
    )
    require_boundary_tolerance(name, result)
    return result


def verify_boundary_substitutions(
    geometry: Geometry, mino: MinoSystem
) -> dict[str, BoundaryResult]:
    """Stress outgoing expressions at defined points near three chart seams."""

    rng = random.Random(SEED)
    probe = build_boundary_probe(geometry, mino)
    return {
        case.name: evaluate_boundary_case(geometry, mino, probe, rng, case)
        for case in build_boundary_cases(rng)
    }


def short_scientific(value: sp.Expr) -> str:
    if value == 0:
        return "0"
    return str(value.evalf(8))


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
    print("symbolic.affine_mino=PASS")
    print("symbolic.corrected_physical_spin=PASS branches=ingoing,outgoing")
    print(f"symbolic.legacy_outgoing=RED_AS_EXPECTED mismatch={mismatch}")
    print("symbolic.legacy_outgoing_sample=RED_AS_EXPECTED g_tphi=g_phit=360/1591")
    print(f"boundary.seed=0x{SEED:08X} precision_digits={PRECISION_DIGITS}")
    for name, result in boundary_results.items():
        print(
            f"boundary.{name}=PASS "
            f"u={short_scientific(result.sin_theta_squared)} "
            f"abs_delta={short_scientific(result.absolute_delta)} "
            f"M2_minus_a2={short_scientific(result.extremality_gap)} "
            f"metric={short_scientific(result.metric_residual)} "
            f"duality={short_scientific(result.duality_residual)} "
            f"mino={short_scientific(result.mino_residual)}"
        )
    print("RESULT=PASS")


if __name__ == "__main__":
    main()

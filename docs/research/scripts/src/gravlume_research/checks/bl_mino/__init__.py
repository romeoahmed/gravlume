"""Run independent high-precision BL/Mino witnesses for named Kerr rays.

The private proof package reconstructs observations from decimal inputs, maps
their photon covectors to Boyer--Lindquist constants, and solves certified
events with turning-segment Mino quadratures.  It neither imports Gravlume nor
acts as a general Kerr solver.

Primary mathematical sources:

* Carter separation: https://doi.org/10.1103/PhysRev.174.1559
* Real Kerr integrals and root classes: https://doi.org/10.1103/PhysRevD.101.044032
* Mino reparameterization: https://doi.org/10.1103/PhysRevD.67.084027
* Signed circular-orbit branches: https://doi.org/10.1086/151796
"""

import platform

import mpmath as mp

from ._certification import (
    _build_critical_curve_precision_certificate,
    _build_negative_spin_precision_certificate,
    _build_source_edge_corpus_precision_certificate,
)
from ._model import (
    _CriticalSurfaceWitness,
    _EscapeWitness,
    _HorizonWitness,
    _NegativeSpinSurfaceWitness,
    _SurfaceWitness,
)


def _scientific(value: mp.mpf, digits: int = 110) -> str:
    return mp.nstr(value, digits, strip_zeros=False)


def _print_path_identity(
    witness: _SurfaceWitness
    | _EscapeWitness
    | _CriticalSurfaceWitness
    | _HorizonWitness
    | _NegativeSpinSurfaceWitness,
) -> None:
    print(f"terminal={witness.terminal}")
    print(
        "branch="
        f"initial_polar_side:{witness.initial_polar_side},"
        f"radial_turnings:{witness.radial_turnings},"
        f"polar_turnings:{witness.polar_turnings},"
        "equatorial_crossings_before_terminal:"
        f"{witness.equatorial_crossings_before_terminal},"
        f"azimuth_winding:{witness.azimuth_winding}"
    )


def _print_turning_and_residuals(
    witness: _SurfaceWitness
    | _EscapeWitness
    | _CriticalSurfaceWitness
    | _NegativeSpinSurfaceWitness,
) -> None:
    print(f"energy={_scientific(witness.energy)}")
    print(f"impact_parameter={_scientific(witness.impact_parameter)}")
    print(f"carter_parameter={_scientific(witness.carter_parameter)}")
    print(
        "radial_turning_derivative="
        f"{_scientific(witness.radial_turning_derivative, 20)}"
    )
    print(
        f"polar_turning_derivative={_scientific(witness.polar_turning_derivative, 20)}"
    )
    print(f"initial_null_residual={_scientific(witness.initial_null_residual, 12)}")
    print(
        f"mino_constraint_residual={_scientific(witness.mino_constraint_residual, 12)}"
    )
    print(
        f"chart_primitive_residual={_scientific(witness.chart_primitive_residual, 12)}"
    )


def _print_surface_observables(
    witness: _SurfaceWitness | _CriticalSurfaceWitness | _NegativeSpinSurfaceWitness,
    *,
    outer_edge_signed_margin: mp.mpf | None = None,
) -> None:
    print(f"source_radius_m={_scientific(witness.source_radius_m)}")
    if outer_edge_signed_margin is not None:
        print(f"outer_edge_signed_margin_m={_scientific(outer_edge_signed_margin)}")
    print(f"source_azimuth_rad={_scientific(witness.source_azimuth_rad)}")
    print(f"frequency_ratio={_scientific(witness.frequency_ratio)}")
    print(f"travel_time_m={_scientific(witness.travel_time_m)}")
    print(
        "emitted_bolometric_intensity="
        f"{_scientific(witness.emitted_bolometric_intensity)}"
    )
    print(
        "observed_bolometric_intensity="
        f"{_scientific(witness.observed_bolometric_intensity)}"
    )


def _print_negative_spin_observables(
    witness: _NegativeSpinSurfaceWitness,
) -> None:
    print(f"physical_spin_m={_scientific(witness.physical_spin_m)}")
    print(f"chart_sign={witness.chart_sign}")
    print(f"emitter_branch_sign={witness.emitter_branch_sign}")
    print(f"radial_root_class={witness.radial_root_class}")
    print(f"exterior_radial_root_count={witness.exterior_radial_root_count}")
    print(
        f"radial_stationary_radius_m={_scientific(witness.radial_stationary_radius_m)}"
    )
    print(
        "radial_classification_margin="
        f"{_scientific(witness.radial_classification_margin)}"
    )
    print(f"radial_turning_radius_m={_scientific(witness.radial_turning_radius_m)}")
    print(
        "radial_turning_above_horizon_margin_m="
        f"{_scientific(witness.radial_turning_above_horizon_margin_m)}"
    )
    print(f"source_mino_duration={_scientific(witness.source_mino_duration)}")
    print(f"radial_turn_mino_duration={_scientific(witness.radial_turn_mino_duration)}")
    print(
        "source_after_radial_turn_mino_margin="
        f"{_scientific(witness.source_after_radial_turn_mino_margin)}"
    )
    print(
        "next_crossing_after_source_mino_margin="
        f"{_scientific(witness.next_crossing_after_source_mino_margin)}"
    )
    print(f"source_inner_margin_m={_scientific(witness.source_inner_margin_m)}")
    print(f"source_outer_margin_m={_scientific(witness.source_outer_margin_m)}")
    _print_surface_observables(witness)
    print(
        "source_azimuth_unwrapped_rad="
        f"{_scientific(witness.source_azimuth_unwrapped_rad)}"
    )
    print(
        "emitter_angular_velocity_per_m="
        f"{_scientific(witness.emitter_angular_velocity_per_m)}"
    )


def _print_escape_observables(
    witness: _EscapeWitness,
    *,
    outer_edge_signed_margin: mp.mpf,
) -> None:
    print(
        "first_equatorial_crossing_radius_m="
        f"{_scientific(witness.first_equatorial_crossing_radius_m)}"
    )
    print(f"outer_edge_signed_margin_m={_scientific(outer_edge_signed_margin)}")
    print(
        "escape_position_xyz_m="
        + ",".join(_scientific(value) for value in witness.escape_position_xyz_m)
    )
    print(
        "escape_direction_xyz="
        + ",".join(_scientific(value) for value in witness.escape_direction_xyz)
    )
    print(f"travel_time_m={_scientific(witness.travel_time_m)}")
    print(
        "escape_before_next_crossing_mino_margin="
        f"{_scientific(witness.escape_before_next_crossing_mino_margin)}"
    )


def _print_critical_surface_observables(witness: _CriticalSurfaceWitness) -> None:
    print(
        "first_equatorial_crossing_mino_duration="
        f"{_scientific(witness.first_equatorial_crossing_mino_duration)}"
    )
    print(
        "terminal_equatorial_crossing_mino_duration="
        f"{_scientific(witness.terminal_equatorial_crossing_mino_duration)}"
    )
    print(
        "terminal_after_first_crossing_mino_margin="
        f"{_scientific(witness.terminal_after_first_crossing_mino_margin)}"
    )
    print(
        "first_equatorial_crossing_radius_m="
        f"{_scientific(witness.first_equatorial_crossing_radius_m)}"
    )
    print(
        "first_crossing_below_surface_margin_m="
        f"{_scientific(witness.first_crossing_below_surface_margin_m)}"
    )
    print(
        "radial_turning_above_horizon_margin_m="
        f"{_scientific(witness.radial_turning_above_horizon_margin_m)}"
    )
    _print_surface_observables(witness)
    print(
        "source_azimuth_unwrapped_rad="
        f"{_scientific(witness.source_azimuth_unwrapped_rad)}"
    )


def _print_horizon_observables(witness: _HorizonWitness) -> None:
    print(
        "first_equatorial_crossing_mino_duration="
        f"{_scientific(witness.first_equatorial_crossing_mino_duration)}"
    )
    print(f"horizon_mino_duration={_scientific(witness.horizon_mino_duration)}")
    print(
        "first_equatorial_crossing_radius_m="
        f"{_scientific(witness.first_equatorial_crossing_radius_m)}"
    )
    print(
        "first_crossing_below_surface_margin_m="
        f"{_scientific(witness.first_crossing_below_surface_margin_m)}"
    )
    print(
        "horizon_after_first_crossing_mino_margin="
        f"{_scientific(witness.horizon_after_first_crossing_mino_margin)}"
    )
    print(f"horizon_radius_m={_scientific(witness.horizon_radius_m)}")
    print(f"horizon_mu={_scientific(witness.horizon_mu)}")
    print(
        "horizon_azimuth_unwrapped_rad="
        f"{_scientific(witness.horizon_azimuth_unwrapped_rad)}"
    )
    print(f"horizon_azimuth_rad={_scientific(witness.horizon_azimuth_rad)}")
    print(
        "horizon_position_xyz_m="
        + ",".join(_scientific(value) for value in witness.horizon_position_xyz_m)
    )
    print(f"travel_time_m={_scientific(witness.travel_time_m)}")
    print(f"energy={_scientific(witness.energy)}")
    print(f"impact_parameter={_scientific(witness.impact_parameter)}")
    print(f"carter_parameter={_scientific(witness.carter_parameter)}")
    print(
        f"polar_turning_derivative={_scientific(witness.polar_turning_derivative, 20)}"
    )
    print(f"initial_null_residual={_scientific(witness.initial_null_residual, 12)}")
    print(
        f"mino_constraint_residual={_scientific(witness.mino_constraint_residual, 12)}"
    )
    print(
        "horizon_cancellation_residual="
        f"{_scientific(witness.horizon_cancellation_residual, 12)}"
    )


def run() -> None:
    source_edge_certificate = _build_source_edge_corpus_precision_certificate()
    print(f"python={platform.python_version()}")
    print(f"mpmath={mp.__version__}")
    print(
        "source_edge_corpus_precision="
        f"{source_edge_certificate.precision.policy.low_digits},"
        f"{source_edge_certificate.precision.policy.high_digits} "
        "stable_digits>="
        f"{source_edge_certificate.precision.policy.required_digits} "
        "maximum_normalized_delta="
        f"{_scientific(source_edge_certificate.precision.maximum_normalized_delta, 12)}"
    )
    for case in source_edge_certificate.witness.cases:
        pixel_x, pixel_y = case.pixel
        print(f"case=kerr-exterior-observation-v1:{pixel_x}:{pixel_y}:0.5:0.5")
        _print_path_identity(case.witness)
        if isinstance(case.witness, _EscapeWitness):
            _print_escape_observables(
                case.witness,
                outer_edge_signed_margin=case.outer_edge_signed_margin_m,
            )
        else:
            _print_surface_observables(
                case.witness,
                outer_edge_signed_margin=case.outer_edge_signed_margin_m,
            )
        _print_turning_and_residuals(case.witness)

    critical_certificate = _build_critical_curve_precision_certificate()
    critical_corpus = critical_certificate.witness
    print(
        "critical_curve_corpus_precision="
        f"{critical_certificate.precision.policy.low_digits},"
        f"{critical_certificate.precision.policy.high_digits} "
        f"stable_digits>={critical_certificate.precision.policy.required_digits} "
        "maximum_normalized_delta="
        f"{_scientific(critical_certificate.precision.maximum_normalized_delta, 12)}"
    )
    print(f"critical_root_class={critical_corpus.critical_root_class}")
    print(f"critical_sample_y={_scientific(critical_corpus.critical_sample_y)}")
    print(f"critical_radius_m={_scientific(critical_corpus.critical_radius_m)}")
    print(
        "critical_potential_residual="
        f"{_scientific(critical_corpus.critical_potential_residual, 12)}"
    )
    print(
        "critical_derivative_residual="
        f"{_scientific(critical_corpus.critical_derivative_residual, 12)}"
    )
    print(
        "critical_second_derivative="
        f"{_scientific(critical_corpus.critical_second_derivative)}"
    )
    for case in critical_corpus.cases:
        pixel_x, pixel_y = case.pixel
        print(f"case=kerr-critical-outgoing-v1:{pixel_x}:{pixel_y}:0.5:0.5")
        print(
            "signed_critical_distance_pixels="
            f"{_scientific(case.signed_critical_distance_pixels)}"
        )
        print(f"exterior_radial_root_count={case.exterior_radial_root_count}")
        print(
            "radial_classification_margin="
            f"{_scientific(case.radial_classification_margin)}"
        )
        _print_path_identity(case.witness)
        if isinstance(case.witness, _CriticalSurfaceWitness):
            _print_critical_surface_observables(case.witness)
            _print_turning_and_residuals(case.witness)
        else:
            _print_horizon_observables(case.witness)

    negative_spin_certificate = _build_negative_spin_precision_certificate()
    negative_spin = negative_spin_certificate.witness
    print(
        "negative_spin_surface_precision="
        f"{negative_spin_certificate.precision.policy.low_digits},"
        f"{negative_spin_certificate.precision.policy.high_digits} "
        "stable_digits>="
        f"{negative_spin_certificate.precision.policy.required_digits} "
        "maximum_normalized_delta="
        f"{_scientific(negative_spin_certificate.precision.maximum_normalized_delta, 12)}"
    )
    pixel_x, pixel_y = negative_spin.pixel
    print(f"case=kerr-negative-spin-outgoing-v1:{pixel_x}:{pixel_y}:0.5:0.5")
    _print_path_identity(negative_spin)
    _print_negative_spin_observables(negative_spin)
    _print_turning_and_residuals(negative_spin)
    print("RESULT=PASS")

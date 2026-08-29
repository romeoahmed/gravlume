"""Run the private Kerr topology and Carlson proof modules together."""

import platform

import mpmath as mp

from ..._precision import _SCIENTIFIC_PRECISION
from ._carlson import _build_carlson_certificate
from ._topology import _build_topology_certificate


def _scientific(value: mp.mpf, digits: int = 12) -> str:
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits):
        return mp.nstr(value, digits, strip_zeros=False)


def run() -> None:
    topology = _build_topology_certificate()
    carlson = _build_carlson_certificate()
    print(f"python={platform.python_version()}")
    print(
        "precision="
        f"{_SCIENTIFIC_PRECISION.low_digits},"
        f"{_SCIENTIFIC_PRECISION.high_digits} "
        f"stable_digits>={_SCIENTIFIC_PRECISION.required_digits}"
    )
    print(
        "topology.maximum_normalized_delta="
        f"{_scientific(topology.precision.maximum_normalized_delta)}"
    )
    print(f"topology.false_acceptance_count={topology.false_acceptance_count}")
    for report in topology.reports:
        reasons = ",".join(report.fallback_reasons) or "none"
        print(
            f"topology.case={report.name} class={report.topology_id} "
            f"branch={report.motion_branch or 'none'} "
            f"decision={report.decision.value} reasons={reasons}"
        )
        vieta = (
            _scientific(report.vieta_residual)
            if report.vieta_residual is not None
            else "not-applicable"
        )
        sign_changes = sum(
            lower_sign * upper_sign < 0
            for lower_sign, upper_sign in report.real_root_bracket_signs
        )
        print(
            f"topology.metrics.{report.name}="
            f"real:{len(report.ordered_real_roots)} "
            f"complex:{report.complex_root_count} "
            f"sign_brackets:{sign_changes} "
            f"root_separation:{_scientific(report.minimum_root_separation)} "
            "horizon_separation:"
            f"{_scientific(report.minimum_horizon_root_separation)} "
            f"initial_separation:{_scientific(report.minimum_initial_root_separation)} "
            f"root_derivative:{_scientific(report.minimum_root_derivative)} "
            f"root_residual:{_scientific(report.maximum_root_residual)} "
            f"vieta:{vieta}"
        )
    print(
        "carlson.maximum_definition_residual="
        f"{_scientific(carlson.maximum_definition_residual)}"
    )
    print(
        "carlson.maximum_identity_residual="
        f"{_scientific(carlson.maximum_identity_residual)}"
    )
    print(
        "carlson.maximum_normalized_delta="
        f"{_scientific(carlson.precision.maximum_normalized_delta)}"
    )
    print(
        "carlson.rd_xy_symmetry_residual="
        f"{_scientific(carlson.rd_xy_symmetry_residual)}"
    )
    print(
        "carlson.rd_full_permutation_delta="
        f"{_scientific(carlson.rd_full_permutation_delta)}"
    )
    print(
        "carlson.kerr_radial_reduction_residual="
        f"{_scientific(carlson.kerr_radial_reduction_residual)}"
    )
    print("RESULT=PASS")

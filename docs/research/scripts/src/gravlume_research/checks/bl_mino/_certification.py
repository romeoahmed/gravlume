"""Certify precision doubling for the three independent BL/Mino proofs."""

import mpmath as mp

from ..._precision import _SCIENTIFIC_PRECISION
from ._critical_curve import _critical_curve_corpus_witness
from ._model import (
    _CriticalCurveCorpusWitness,
    _CriticalSurfaceWitness,
    _EscapeWitness,
    _HorizonWitness,
    _NegativeSpinSurfaceWitness,
    _PrecisionCertificate,
    _SourceEdgeCorpusWitness,
    _SurfaceWitness,
)
from ._negative_spin import _negative_spin_surface_witness
from ._source_edge import _source_edge_corpus_witness

_SURFACE_PRECISION_FIELDS = (
    "source_radius_m",
    "source_azimuth_rad",
    "frequency_ratio",
    "travel_time_m",
    "emitted_bolometric_intensity",
    "observed_bolometric_intensity",
    "energy",
    "impact_parameter",
    "carter_parameter",
    "radial_turning_derivative",
    "polar_turning_derivative",
)
_ESCAPE_PRECISION_FIELDS = (
    "first_equatorial_crossing_radius_m",
    "escape_radius_m",
    "travel_time_m",
    "escape_before_next_crossing_mino_margin",
    "energy",
    "impact_parameter",
    "carter_parameter",
    "radial_turning_derivative",
    "polar_turning_derivative",
)
_CRITICAL_SURFACE_PRECISION_FIELDS = (
    "first_equatorial_crossing_mino_duration",
    "terminal_equatorial_crossing_mino_duration",
    "terminal_after_first_crossing_mino_margin",
    "first_equatorial_crossing_radius_m",
    "first_crossing_below_surface_margin_m",
    "radial_turning_above_horizon_margin_m",
    "source_radius_m",
    "source_azimuth_unwrapped_rad",
    "source_azimuth_rad",
    "frequency_ratio",
    "travel_time_m",
    "emitted_bolometric_intensity",
    "observed_bolometric_intensity",
    "energy",
    "impact_parameter",
    "carter_parameter",
    "radial_turning_derivative",
    "polar_turning_derivative",
)
_HORIZON_PRECISION_FIELDS = (
    "first_equatorial_crossing_mino_duration",
    "horizon_mino_duration",
    "first_equatorial_crossing_radius_m",
    "first_crossing_below_surface_margin_m",
    "horizon_after_first_crossing_mino_margin",
    "horizon_radius_m",
    "horizon_mu",
    "horizon_azimuth_unwrapped_rad",
    "horizon_azimuth_rad",
    "travel_time_m",
    "energy",
    "impact_parameter",
    "carter_parameter",
    "polar_turning_derivative",
)
_NEGATIVE_SPIN_PRECISION_FIELDS = (
    "physical_spin_m",
    "source_mino_duration",
    "radial_turn_mino_duration",
    "source_after_radial_turn_mino_margin",
    "next_crossing_after_source_mino_margin",
    "radial_stationary_radius_m",
    "radial_classification_margin",
    "radial_turning_radius_m",
    "radial_turning_above_horizon_margin_m",
    "source_radius_m",
    "source_inner_margin_m",
    "source_outer_margin_m",
    "source_azimuth_unwrapped_rad",
    "source_azimuth_rad",
    "emitter_angular_velocity_per_m",
    "frequency_ratio",
    "travel_time_m",
    "emitted_bolometric_intensity",
    "observed_bolometric_intensity",
    "energy",
    "impact_parameter",
    "carter_parameter",
    "radial_turning_derivative",
    "polar_turning_derivative",
)


def _witness_precision_pairs(
    low: _SurfaceWitness | _EscapeWitness,
    high: _SurfaceWitness | _EscapeWitness,
) -> list[tuple[mp.mpf, mp.mpf]]:
    if isinstance(low, _SurfaceWitness) and isinstance(high, _SurfaceWitness):
        return [
            (getattr(low, field), getattr(high, field))
            for field in _SURFACE_PRECISION_FIELDS
        ]
    if isinstance(low, _EscapeWitness) and isinstance(high, _EscapeWitness):
        value_pairs = [
            (getattr(low, field), getattr(high, field))
            for field in _ESCAPE_PRECISION_FIELDS
        ]
        value_pairs.extend(
            zip(
                low.escape_position_xyz_m,
                high.escape_position_xyz_m,
                strict=True,
            )
        )
        value_pairs.extend(
            zip(
                low.escape_direction_xyz,
                high.escape_direction_xyz,
                strict=True,
            )
        )
        return value_pairs
    raise AssertionError("precision doubling changed a source-edge terminal stratum")


def _critical_witness_precision_pairs(
    low: _CriticalSurfaceWitness | _HorizonWitness,
    high: _CriticalSurfaceWitness | _HorizonWitness,
) -> list[tuple[mp.mpf, mp.mpf]]:
    if isinstance(low, _CriticalSurfaceWitness) and isinstance(
        high,
        _CriticalSurfaceWitness,
    ):
        return [
            (getattr(low, field), getattr(high, field))
            for field in _CRITICAL_SURFACE_PRECISION_FIELDS
        ]
    if isinstance(low, _HorizonWitness) and isinstance(high, _HorizonWitness):
        value_pairs = [
            (getattr(low, field), getattr(high, field))
            for field in _HORIZON_PRECISION_FIELDS
        ]
        value_pairs.extend(
            zip(
                low.horizon_position_xyz_m,
                high.horizon_position_xyz_m,
                strict=True,
            )
        )
        return value_pairs
    raise AssertionError("precision doubling changed a critical-curve terminal stratum")


def _build_source_edge_corpus_precision_certificate() -> _PrecisionCertificate[
    _SourceEdgeCorpusWitness
]:
    """Recompute all nine cases at 120/180 digits and certify every observable."""

    low = _source_edge_corpus_witness(precision_digits=_SCIENTIFIC_PRECISION.low_digits)
    high = _source_edge_corpus_witness(
        precision_digits=_SCIENTIFIC_PRECISION.high_digits
    )
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits):
        value_pairs = []
        for low_case, high_case in zip(low.cases, high.cases, strict=True):
            value_pairs.extend(
                _witness_precision_pairs(low_case.witness, high_case.witness)
            )
        precision = _SCIENTIFIC_PRECISION.certify_pairs(
            value_pairs,
            subject="source-edge corpus",
        )
    return _PrecisionCertificate(
        precision=precision,
        witness=high,
    )


def _build_critical_curve_precision_certificate() -> _PrecisionCertificate[
    _CriticalCurveCorpusWitness
]:
    """Rebuild both critical-curve cases at 120/180 digits."""

    low = _critical_curve_corpus_witness(
        precision_digits=_SCIENTIFIC_PRECISION.low_digits
    )
    high = _critical_curve_corpus_witness(
        precision_digits=_SCIENTIFIC_PRECISION.high_digits
    )
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits):
        value_pairs = [
            (low.critical_sample_y, high.critical_sample_y),
            (low.critical_radius_m, high.critical_radius_m),
            (low.critical_second_derivative, high.critical_second_derivative),
        ]
        for low_case, high_case in zip(low.cases, high.cases, strict=True):
            value_pairs.extend(
                (
                    (
                        low_case.signed_critical_distance_pixels,
                        high_case.signed_critical_distance_pixels,
                    ),
                    (
                        low_case.radial_classification_margin,
                        high_case.radial_classification_margin,
                    ),
                )
            )
            value_pairs.extend(
                _critical_witness_precision_pairs(
                    low_case.witness,
                    high_case.witness,
                )
            )
        precision = _SCIENTIFIC_PRECISION.certify_pairs(
            value_pairs,
            subject="critical-curve corpus",
        )
    return _PrecisionCertificate(
        precision=precision,
        witness=high,
    )


def _build_negative_spin_precision_certificate() -> _PrecisionCertificate[
    _NegativeSpinSurfaceWitness
]:
    """Rebuild the named negative-spin surface ray at 120/180 digits."""

    low = _negative_spin_surface_witness(
        precision_digits=_SCIENTIFIC_PRECISION.low_digits
    )
    high = _negative_spin_surface_witness(
        precision_digits=_SCIENTIFIC_PRECISION.high_digits
    )
    with mp.workdps(_SCIENTIFIC_PRECISION.high_digits):
        precision = _SCIENTIFIC_PRECISION.certify_pairs(
            (
                (getattr(low, field), getattr(high, field))
                for field in _NEGATIVE_SPIN_PRECISION_FIELDS
            ),
            subject="negative-spin surface",
        )
    return _PrecisionCertificate(
        precision=precision,
        witness=high,
    )

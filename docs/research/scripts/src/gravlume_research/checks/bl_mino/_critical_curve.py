"""Build and validate the adjacent critical-curve surface/capture proof."""

import mpmath as mp

from ._geometry import (
    _azimuth_winding,
    _canonical_initial_ray,
    _critical_curve_geometry,
    _oblate_position,
    _separated_initial_state,
    _wrap_angle,
)
from ._model import (
    _CRITICAL_CAPTURE_AZIMUTH_WINDING,
    _CRITICAL_CAPTURE_POLAR_TURNINGS,
    _CRITICAL_CURVE_PIXELS,
    _CRITICAL_EQUATORIAL_CROSSINGS,
    _CRITICAL_ROOT_CLASS,
    _CRITICAL_SURFACE_AZIMUTH_WINDING,
    _CRITICAL_SURFACE_PIXEL,
    _CRITICAL_SURFACE_POLAR_TURNINGS,
    _HORIZON_TERMINAL,
    _SOURCE_EDGE_INITIAL_POLAR_SIDE,
    _SOURCE_EDGE_RADIAL_TURNINGS,
    _SURFACE_TERMINAL,
    RESIDUAL_GUARD_DIGITS,
    SURFACE_INNER_RADIUS_M,
    SURFACE_OUTER_RADIUS_M,
    _CriticalCurveCaseWitness,
    _CriticalCurveCorpusWitness,
    _CriticalSurfaceWitness,
    _HorizonWitness,
    _InitialRay,
    _ObservationGeometry,
    _RadialClassification,
    _SeparatedState,
    _UnsupportedWitnessError,
    _validate_precision_digits,
)
from ._motion import (
    _build_capture_radial_motion,
    _build_polar_motion,
    _build_radial_motion,
    _classify_radial_barrier,
    _integrate_path_observables,
    _integrate_polar_observables,
    _solve_capture_polar_endpoint,
    _solve_capture_radius,
    _solve_critical_point,
    _solve_equatorial_crossing_radius,
    _solve_inbound_radius,
    _surface_transfer_observables,
    _unit_integrand,
)


def _compute_critical_surface_witness(
    geometry: _ObservationGeometry,
    initial_ray: _InitialRay,
    separated: _SeparatedState,
    classification: _RadialClassification,
    precision_digits: int,
) -> _CriticalSurfaceWitness:
    if separated.radial_velocity <= 0 or separated.polar_velocity >= 0:
        raise _UnsupportedWitnessError(
            "critical surface witness requires the named outgoing polar branch"
        )
    if classification.margin >= 0 or len(classification.exterior_roots) != 2:
        raise _UnsupportedWitnessError(
            "critical surface witness lacks the two-root scattering topology"
        )
    initial_mu = mp.cos(geometry.theta)
    polar = _build_polar_motion(
        geometry.spin,
        separated.impact,
        separated.carter,
        initial_mu,
    )
    initial_to_turn = polar.integrate_to_turn(_unit_integrand, initial_mu)
    equator_to_turn = polar.integrate_to_turn(_unit_integrand, mp.mpf(0))
    first_crossing_duration = initial_to_turn + equator_to_turn
    second_crossing_duration = initial_to_turn + 3 * equator_to_turn

    radial = _build_radial_motion(geometry, separated, precision_digits)
    first_crossing_radius = _solve_inbound_radius(
        radial,
        geometry.radius,
        first_crossing_duration,
        precision_digits,
    )
    source_radius = _solve_equatorial_crossing_radius(
        radial,
        geometry.radius,
        second_crossing_duration,
        precision_digits,
        mp.mpf(SURFACE_INNER_RADIUS_M),
        mp.mpf(SURFACE_OUTER_RADIUS_M),
    )
    path = _integrate_path_observables(
        geometry,
        polar,
        radial,
        initial_mu,
        source_radius,
        completed_polar_oscillations=1,
    )
    transfer = _surface_transfer_observables(
        geometry,
        initial_ray,
        separated,
        radial,
        source_radius,
        emitter_branch_sign=1,
    )
    horizon = geometry.mass + mp.sqrt(geometry.mass**2 - geometry.spin**2)
    return _CriticalSurfaceWitness(
        precision_digits=precision_digits,
        terminal=_SURFACE_TERMINAL,
        initial_polar_side=_SOURCE_EDGE_INITIAL_POLAR_SIDE,
        radial_turnings=_SOURCE_EDGE_RADIAL_TURNINGS,
        polar_turnings=_CRITICAL_SURFACE_POLAR_TURNINGS,
        equatorial_crossings_before_terminal=_CRITICAL_EQUATORIAL_CROSSINGS,
        azimuth_winding=path.azimuth_winding,
        first_equatorial_crossing_mino_duration=first_crossing_duration,
        terminal_equatorial_crossing_mino_duration=second_crossing_duration,
        terminal_after_first_crossing_mino_margin=(
            second_crossing_duration - first_crossing_duration
        ),
        first_equatorial_crossing_radius_m=first_crossing_radius,
        first_crossing_below_surface_margin_m=(
            mp.mpf(SURFACE_INNER_RADIUS_M) - first_crossing_radius
        ),
        radial_turning_above_horizon_margin_m=radial.turning - horizon,
        source_radius_m=source_radius,
        source_azimuth_unwrapped_rad=path.terminal_azimuth_unwrapped,
        source_azimuth_rad=path.terminal_azimuth,
        frequency_ratio=transfer.frequency_ratio,
        travel_time_m=path.travel_time,
        emitted_bolometric_intensity=transfer.emitted_intensity,
        observed_bolometric_intensity=transfer.observed_intensity,
        energy=separated.energy,
        impact_parameter=separated.impact,
        carter_parameter=separated.carter,
        radial_turning_derivative=radial.turning_derivative,
        polar_turning_derivative=polar.turning_derivative,
        initial_null_residual=initial_ray.initial_null_residual,
        mino_constraint_residual=separated.constraint_residual,
        chart_primitive_residual=path.chart_primitive_residual,
    )


def _compute_horizon_witness(
    geometry: _ObservationGeometry,
    initial_ray: _InitialRay,
    separated: _SeparatedState,
    classification: _RadialClassification,
    precision_digits: int,
) -> _HorizonWitness:
    if separated.radial_velocity <= 0 or separated.polar_velocity >= 0:
        raise _UnsupportedWitnessError(
            "horizon witness requires the named outgoing polar branch"
        )
    polar = _build_polar_motion(
        geometry.spin,
        separated.impact,
        separated.carter,
        mp.cos(geometry.theta),
    )
    radial = _build_capture_radial_motion(
        geometry,
        separated,
        classification,
    )
    capture_mino_duration = radial.mino_duration()
    initial_mu = mp.cos(geometry.theta)
    (
        terminal_mu_magnitude,
        first_crossing_duration,
        horizon_after_first_crossing,
    ) = _solve_capture_polar_endpoint(
        polar,
        capture_mino_duration,
        initial_mu,
        precision_digits,
    )
    first_crossing_radius = _solve_capture_radius(
        radial,
        first_crossing_duration,
        precision_digits,
    )
    polar_time, polar_azimuth = _integrate_polar_observables(
        geometry,
        polar,
        separated.impact,
        initial_mu,
        terminal_mu_magnitude,
        0,
    )
    horizon_azimuth_unwrapped = geometry.chart_azimuth - (
        radial.azimuth_integral() + polar_azimuth
    )
    horizon_mu = -terminal_mu_magnitude
    horizon_position = _oblate_position(
        geometry,
        radial.horizon,
        horizon_mu,
        horizon_azimuth_unwrapped,
    )
    return _HorizonWitness(
        precision_digits=precision_digits,
        terminal=_HORIZON_TERMINAL,
        initial_polar_side=_SOURCE_EDGE_INITIAL_POLAR_SIDE,
        radial_turnings=0,
        polar_turnings=_CRITICAL_CAPTURE_POLAR_TURNINGS,
        equatorial_crossings_before_terminal=_CRITICAL_EQUATORIAL_CROSSINGS,
        azimuth_winding=_azimuth_winding(
            geometry,
            radial.horizon,
            horizon_azimuth_unwrapped,
        ),
        first_equatorial_crossing_mino_duration=first_crossing_duration,
        horizon_mino_duration=capture_mino_duration,
        first_equatorial_crossing_radius_m=first_crossing_radius,
        first_crossing_below_surface_margin_m=(
            mp.mpf(SURFACE_INNER_RADIUS_M) - first_crossing_radius
        ),
        horizon_after_first_crossing_mino_margin=horizon_after_first_crossing,
        horizon_radius_m=radial.horizon,
        horizon_mu=horizon_mu,
        horizon_azimuth_unwrapped_rad=horizon_azimuth_unwrapped,
        horizon_azimuth_rad=_wrap_angle(horizon_azimuth_unwrapped),
        horizon_position_xyz_m=horizon_position,
        travel_time_m=radial.time_integral() + polar_time,
        energy=separated.energy,
        impact_parameter=separated.impact,
        carter_parameter=separated.carter,
        polar_turning_derivative=polar.turning_derivative,
        initial_null_residual=initial_ray.initial_null_residual,
        mino_constraint_residual=separated.constraint_residual,
        horizon_cancellation_residual=radial.cancellation_residual(),
    )


def _critical_curve_corpus_witness(
    *,
    precision_digits: int,
) -> _CriticalCurveCorpusWitness:
    """Recompute the adjacent scattering/capture pair from canonical inputs."""

    _validate_precision_digits(precision_digits)
    with mp.workdps(precision_digits):
        geometry = _critical_curve_geometry()
        critical = _solve_critical_point(geometry, precision_digits)
        cases = []
        for pixel in _CRITICAL_CURVE_PIXELS:
            initial_ray = _canonical_initial_ray(geometry, *pixel)
            separated = _separated_initial_state(geometry, initial_ray)
            classification = _classify_radial_barrier(geometry, separated)
            if pixel == _CRITICAL_SURFACE_PIXEL:
                witness = _compute_critical_surface_witness(
                    geometry,
                    initial_ray,
                    separated,
                    classification,
                    precision_digits,
                )
            else:
                witness = _compute_horizon_witness(
                    geometry,
                    initial_ray,
                    separated,
                    classification,
                    precision_digits,
                )
            sample_y = mp.mpf(pixel[1]) + mp.mpf(1) / 2
            cases.append(
                _CriticalCurveCaseWitness(
                    pixel=pixel,
                    witness=witness,
                    exterior_radial_root_count=len(classification.exterior_roots),
                    signed_critical_distance_pixels=sample_y - critical.sample_y,
                    radial_classification_margin=classification.margin,
                )
            )
        return _CriticalCurveCorpusWitness(
            critical_root_class=_CRITICAL_ROOT_CLASS,
            critical_sample_y=critical.sample_y,
            critical_radius_m=critical.radius,
            critical_potential_residual=critical.potential_residual,
            critical_derivative_residual=critical.derivative_residual,
            critical_second_derivative=critical.second_derivative,
            cases=tuple(cases),
        )


def _validate_critical_surface_witness(witness: _CriticalSurfaceWitness) -> None:
    _validate_precision_digits(witness.precision_digits)
    discrete_identity = (
        (witness.terminal, _SURFACE_TERMINAL),
        (witness.initial_polar_side, _SOURCE_EDGE_INITIAL_POLAR_SIDE),
        (witness.radial_turnings, _SOURCE_EDGE_RADIAL_TURNINGS),
        (witness.polar_turnings, _CRITICAL_SURFACE_POLAR_TURNINGS),
        (
            witness.equatorial_crossings_before_terminal,
            _CRITICAL_EQUATORIAL_CROSSINGS,
        ),
        (witness.azimuth_winding, _CRITICAL_SURFACE_AZIMUTH_WINDING),
    )
    if any(
        type(actual) is not type(expected) or actual != expected
        for actual, expected in discrete_identity
    ):
        raise _UnsupportedWitnessError(
            "critical surface witness does not match its discrete path identity"
        )
    continuous_fields = (
        witness.first_equatorial_crossing_mino_duration,
        witness.terminal_equatorial_crossing_mino_duration,
        witness.terminal_after_first_crossing_mino_margin,
        witness.first_equatorial_crossing_radius_m,
        witness.first_crossing_below_surface_margin_m,
        witness.radial_turning_above_horizon_margin_m,
        witness.source_radius_m,
        witness.source_azimuth_unwrapped_rad,
        witness.source_azimuth_rad,
        witness.frequency_ratio,
        witness.travel_time_m,
        witness.emitted_bolometric_intensity,
        witness.observed_bolometric_intensity,
        witness.energy,
        witness.impact_parameter,
        witness.carter_parameter,
        witness.radial_turning_derivative,
        witness.polar_turning_derivative,
        witness.initial_null_residual,
        witness.mino_constraint_residual,
        witness.chart_primitive_residual,
    )
    if not all(
        isinstance(value, mp.mpf) and mp.isfinite(value) for value in continuous_fields
    ):
        raise _UnsupportedWitnessError(
            "critical surface witness contains a non-real or non-finite value"
        )
    if not (
        witness.first_equatorial_crossing_radius_m < SURFACE_INNER_RADIUS_M
        and SURFACE_INNER_RADIUS_M <= witness.source_radius_m <= SURFACE_OUTER_RADIUS_M
    ):
        raise _UnsupportedWitnessError(
            "critical surface crossings do not certify the named event order"
        )
    positive_fields = (
        witness.first_equatorial_crossing_mino_duration,
        witness.terminal_equatorial_crossing_mino_duration,
        witness.terminal_after_first_crossing_mino_margin,
        witness.first_crossing_below_surface_margin_m,
        witness.radial_turning_above_horizon_margin_m,
        witness.frequency_ratio,
        witness.travel_time_m,
        witness.emitted_bolometric_intensity,
        witness.observed_bolometric_intensity,
        witness.energy,
        witness.radial_turning_derivative,
        witness.polar_turning_derivative,
    )
    if any(value <= 0 for value in positive_fields):
        raise _UnsupportedWitnessError(
            "critical surface witness lacks a positive physical or event margin"
        )

    with mp.workdps(witness.precision_digits):
        residual_limit = mp.power(
            10,
            RESIDUAL_GUARD_DIGITS - witness.precision_digits,
        )
        margin_residual = abs(
            witness.first_crossing_below_surface_margin_m
            - (
                mp.mpf(SURFACE_INNER_RADIUS_M)
                - witness.first_equatorial_crossing_radius_m
            )
        )
        event_residual = abs(
            witness.terminal_after_first_crossing_mino_margin
            - (
                witness.terminal_equatorial_crossing_mino_duration
                - witness.first_equatorial_crossing_mino_duration
            )
        )
        phase_residual = abs(
            _wrap_angle(witness.source_azimuth_unwrapped_rad)
            - witness.source_azimuth_rad
        )
        transfer_residual = abs(
            witness.observed_bolometric_intensity
            - witness.emitted_bolometric_intensity * witness.frequency_ratio**4
        ) / max(mp.mpf(1), abs(witness.observed_bolometric_intensity))
        residuals = (
            witness.initial_null_residual,
            witness.mino_constraint_residual,
            witness.chart_primitive_residual,
            margin_residual,
            event_residual,
            phase_residual,
            transfer_residual,
        )
        if any(residual < 0 or residual >= residual_limit for residual in residuals):
            raise _UnsupportedWitnessError(
                "critical surface equation or identity residual is too large"
            )
        if not -mp.pi <= witness.source_azimuth_rad < mp.pi:
            raise _UnsupportedWitnessError("critical surface phase is not canonical")
        geometry = _critical_curve_geometry()
        if (
            _azimuth_winding(
                geometry,
                witness.source_radius_m,
                witness.source_azimuth_unwrapped_rad,
            )
            != witness.azimuth_winding
        ):
            raise _UnsupportedWitnessError(
                "critical surface unwrapped phase disagrees with its winding"
            )


def _validate_horizon_witness(witness: _HorizonWitness) -> None:
    _validate_precision_digits(witness.precision_digits)
    discrete_identity = (
        (witness.terminal, _HORIZON_TERMINAL),
        (witness.initial_polar_side, _SOURCE_EDGE_INITIAL_POLAR_SIDE),
        (witness.radial_turnings, 0),
        (witness.polar_turnings, _CRITICAL_CAPTURE_POLAR_TURNINGS),
        (
            witness.equatorial_crossings_before_terminal,
            _CRITICAL_EQUATORIAL_CROSSINGS,
        ),
        (witness.azimuth_winding, _CRITICAL_CAPTURE_AZIMUTH_WINDING),
    )
    if any(
        type(actual) is not type(expected) or actual != expected
        for actual, expected in discrete_identity
    ):
        raise _UnsupportedWitnessError(
            "horizon witness does not match its discrete path identity"
        )
    if len(witness.horizon_position_xyz_m) != 3:
        raise _UnsupportedWitnessError(
            "horizon position must contain exactly three lanes"
        )
    continuous_fields = (
        witness.first_equatorial_crossing_mino_duration,
        witness.horizon_mino_duration,
        witness.first_equatorial_crossing_radius_m,
        witness.first_crossing_below_surface_margin_m,
        witness.horizon_after_first_crossing_mino_margin,
        witness.horizon_radius_m,
        witness.horizon_mu,
        witness.horizon_azimuth_unwrapped_rad,
        witness.horizon_azimuth_rad,
        *witness.horizon_position_xyz_m,
        witness.travel_time_m,
        witness.energy,
        witness.impact_parameter,
        witness.carter_parameter,
        witness.polar_turning_derivative,
        witness.initial_null_residual,
        witness.mino_constraint_residual,
        witness.horizon_cancellation_residual,
    )
    if not all(
        isinstance(value, mp.mpf) and mp.isfinite(value) for value in continuous_fields
    ):
        raise _UnsupportedWitnessError(
            "horizon witness contains a non-real or non-finite value"
        )
    if not (
        witness.horizon_radius_m
        < witness.first_equatorial_crossing_radius_m
        < SURFACE_INNER_RADIUS_M
    ):
        raise _UnsupportedWitnessError(
            "horizon witness does not certify the first non-surface crossing"
        )
    if not -1 < witness.horizon_mu < 0:
        raise _UnsupportedWitnessError(
            "horizon endpoint is not on the certified southern polar segment"
        )
    positive_fields = (
        witness.first_equatorial_crossing_mino_duration,
        witness.horizon_mino_duration,
        witness.first_crossing_below_surface_margin_m,
        witness.horizon_after_first_crossing_mino_margin,
        witness.travel_time_m,
        witness.energy,
        witness.polar_turning_derivative,
    )
    if any(value <= 0 for value in positive_fields):
        raise _UnsupportedWitnessError(
            "horizon witness lacks a positive physical or event margin"
        )

    with mp.workdps(witness.precision_digits):
        expected_horizon = mp.mpf(8) / 5
        residual_limit = mp.power(
            10,
            RESIDUAL_GUARD_DIGITS - witness.precision_digits,
        )
        horizon_residual = abs(witness.horizon_radius_m - expected_horizon)
        event_residual = abs(
            witness.horizon_after_first_crossing_mino_margin
            - (
                witness.horizon_mino_duration
                - witness.first_equatorial_crossing_mino_duration
            )
        )
        phase_residual = abs(
            _wrap_angle(witness.horizon_azimuth_unwrapped_rad)
            - witness.horizon_azimuth_rad
        )
        margin_residual = abs(
            witness.first_crossing_below_surface_margin_m
            - (
                mp.mpf(SURFACE_INNER_RADIUS_M)
                - witness.first_equatorial_crossing_radius_m
            )
        )
        sin_theta = mp.sqrt(1 - witness.horizon_mu**2)
        azimuth = witness.horizon_azimuth_rad
        spin = mp.mpf(4) / 5
        expected_position = (
            (witness.horizon_radius_m * mp.cos(azimuth) + spin * mp.sin(azimuth))
            * sin_theta,
            (witness.horizon_radius_m * mp.sin(azimuth) - spin * mp.cos(azimuth))
            * sin_theta,
            witness.horizon_radius_m * witness.horizon_mu,
        )
        position_residual = max(
            abs(actual - expected)
            for actual, expected in zip(
                witness.horizon_position_xyz_m,
                expected_position,
                strict=True,
            )
        ) / max(
            mp.mpf(1),
            *(abs(component) for component in expected_position),
        )
        residuals = (
            witness.initial_null_residual,
            witness.mino_constraint_residual,
            witness.horizon_cancellation_residual,
            horizon_residual,
            event_residual,
            phase_residual,
            margin_residual,
            position_residual,
        )
        if any(residual < 0 or residual >= residual_limit for residual in residuals):
            raise _UnsupportedWitnessError(
                "horizon equation or identity residual is too large"
            )
        if not -mp.pi <= witness.horizon_azimuth_rad < mp.pi:
            raise _UnsupportedWitnessError("horizon phase is not canonical")
        geometry = _critical_curve_geometry()
        if (
            _azimuth_winding(
                geometry,
                witness.horizon_radius_m,
                witness.horizon_azimuth_unwrapped_rad,
            )
            != witness.azimuth_winding
        ):
            raise _UnsupportedWitnessError(
                "horizon unwrapped phase disagrees with its winding"
            )

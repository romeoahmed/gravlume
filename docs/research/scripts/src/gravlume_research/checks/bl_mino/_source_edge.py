"""Build and validate the ordered nine-ray source-edge proof corpus."""

import mpmath as mp

from ._geometry import (
    _canonical_geometry,
    _canonical_initial_ray,
    _separated_initial_state,
)
from ._model import (
    _SOURCE_EDGE_AZIMUTH_WINDING,
    _SOURCE_EDGE_ESCAPE_EQUATORIAL_CROSSINGS,
    _SOURCE_EDGE_ESCAPE_PIXELS,
    _SOURCE_EDGE_ESCAPE_TERMINAL,
    _SOURCE_EDGE_INITIAL_POLAR_SIDE,
    _SOURCE_EDGE_PIXELS,
    _SOURCE_EDGE_POLAR_TURNINGS,
    _SOURCE_EDGE_RADIAL_TURNINGS,
    _SURFACE_EQUATORIAL_CROSSINGS_BEFORE_TERMINAL,
    _SURFACE_TERMINAL,
    ESCAPE_RADIUS_M,
    RESIDUAL_GUARD_DIGITS,
    SURFACE_INNER_RADIUS_M,
    SURFACE_OUTER_RADIUS_M,
    _EscapeWitness,
    _SourceEdgeCaseWitness,
    _SourceEdgeCorpusWitness,
    _SurfaceWitness,
    _UnsupportedWitnessError,
    _validate_precision_digits,
)
from ._motion import (
    _build_polar_motion,
    _build_radial_motion,
    _escape_position_and_direction,
    _integrate_path_observables,
    _solve_equatorial_crossing_radius,
    _solve_escape_polar_endpoint,
    _solve_source_radius,
    _surface_transfer_observables,
    _unit_integrand,
)


def _compute_source_edge_escape_witness(
    pixel_x: int,
    pixel_y: int,
    precision_digits: int,
) -> _EscapeWitness:
    geometry = _canonical_geometry()
    initial_ray = _canonical_initial_ray(geometry, pixel_x, pixel_y)
    separated = _separated_initial_state(geometry, initial_ray)
    if separated.radial_velocity <= 0 or separated.polar_velocity >= 0:
        raise _UnsupportedWitnessError(
            "source-edge escape requires a future-outgoing ray after one "
            "northern polar turning"
        )
    initial_mu = mp.cos(geometry.theta)
    polar = _build_polar_motion(
        geometry.spin,
        separated.impact,
        separated.carter,
        initial_mu,
    )
    radial = _build_radial_motion(geometry, separated, precision_digits)
    escape_radius = mp.mpf(ESCAPE_RADIUS_M)
    (
        terminal_mu_magnitude,
        first_crossing_duration,
        next_crossing_margin,
    ) = _solve_escape_polar_endpoint(
        polar,
        radial,
        geometry.radius,
        escape_radius,
        initial_mu,
        precision_digits,
    )
    first_crossing_radius = _solve_equatorial_crossing_radius(
        radial,
        geometry.radius,
        first_crossing_duration,
        precision_digits,
        mp.mpf(SURFACE_OUTER_RADIUS_M),
        geometry.radius,
    )
    path = _integrate_path_observables(
        geometry,
        polar,
        radial,
        initial_mu,
        escape_radius,
        terminal_mu_magnitude,
    )
    position, direction = _escape_position_and_direction(
        geometry,
        radial,
        polar,
        escape_radius,
        terminal_mu_magnitude,
        path.terminal_azimuth,
    )
    return _EscapeWitness(
        precision_digits=precision_digits,
        terminal=_SOURCE_EDGE_ESCAPE_TERMINAL,
        initial_polar_side=_SOURCE_EDGE_INITIAL_POLAR_SIDE,
        radial_turnings=_SOURCE_EDGE_RADIAL_TURNINGS,
        polar_turnings=_SOURCE_EDGE_POLAR_TURNINGS,
        equatorial_crossings_before_terminal=(_SOURCE_EDGE_ESCAPE_EQUATORIAL_CROSSINGS),
        azimuth_winding=path.azimuth_winding,
        first_equatorial_crossing_radius_m=first_crossing_radius,
        escape_radius_m=escape_radius,
        escape_position_xyz_m=position,
        escape_direction_xyz=direction,
        travel_time_m=path.travel_time,
        escape_before_next_crossing_mino_margin=next_crossing_margin,
        energy=separated.energy,
        impact_parameter=separated.impact,
        carter_parameter=separated.carter,
        radial_turning_derivative=radial.turning_derivative,
        polar_turning_derivative=polar.turning_derivative,
        initial_null_residual=initial_ray.initial_null_residual,
        mino_constraint_residual=separated.constraint_residual,
        chart_primitive_residual=path.chart_primitive_residual,
    )


def _compute_surface_witness(
    pixel_x: int, pixel_y: int, precision_digits: int
) -> _SurfaceWitness:
    geometry = _canonical_geometry()
    initial_ray = _canonical_initial_ray(geometry, pixel_x, pixel_y)
    separated = _separated_initial_state(geometry, initial_ray)
    if separated.radial_velocity <= 0 or separated.polar_velocity >= 0:
        raise _UnsupportedWitnessError(
            "named surface witness requires a future-outgoing ray after one northern "
            "polar turning"
        )
    initial_mu = mp.cos(geometry.theta)
    polar = _build_polar_motion(
        geometry.spin,
        separated.impact,
        separated.carter,
        initial_mu,
    )
    polar_mino_duration = polar.integrate_to_turn(_unit_integrand, mp.mpf(0))
    polar_mino_duration += polar.integrate_to_turn(_unit_integrand, initial_mu)

    radial = _build_radial_motion(geometry, separated, precision_digits)
    source_radius = _solve_source_radius(
        radial,
        geometry.radius,
        polar_mino_duration,
        precision_digits,
    )

    path = _integrate_path_observables(
        geometry,
        polar,
        radial,
        initial_mu,
        source_radius,
    )
    transfer = _surface_transfer_observables(
        geometry,
        initial_ray,
        separated,
        radial,
        source_radius,
        emitter_branch_sign=1,
    )
    return _SurfaceWitness(
        precision_digits=precision_digits,
        terminal=_SURFACE_TERMINAL,
        initial_polar_side=_SOURCE_EDGE_INITIAL_POLAR_SIDE,
        radial_turnings=_SOURCE_EDGE_RADIAL_TURNINGS,
        polar_turnings=_SOURCE_EDGE_POLAR_TURNINGS,
        equatorial_crossings_before_terminal=(
            _SURFACE_EQUATORIAL_CROSSINGS_BEFORE_TERMINAL
        ),
        azimuth_winding=path.azimuth_winding,
        source_radius_m=source_radius,
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


def _source_edge_corpus_witness(*, precision_digits: int) -> _SourceEdgeCorpusWitness:
    """Compute the fixed ordered source-edge corpus from canonical inputs."""

    _validate_precision_digits(precision_digits)
    with mp.workdps(precision_digits):
        return _SourceEdgeCorpusWitness(
            cases=tuple(
                _SourceEdgeCaseWitness(
                    pixel=pixel,
                    witness=(
                        _compute_source_edge_escape_witness(
                            *pixel,
                            precision_digits,
                        )
                        if pixel in _SOURCE_EDGE_ESCAPE_PIXELS
                        else _compute_surface_witness(
                            *pixel,
                            precision_digits,
                        )
                    ),
                )
                for pixel in _SOURCE_EDGE_PIXELS
            )
        )


def _validate_surface_witness(witness: _SurfaceWitness) -> None:
    _validate_precision_digits(witness.precision_digits)
    discrete_identity = (
        (witness.terminal, _SURFACE_TERMINAL),
        (witness.initial_polar_side, _SOURCE_EDGE_INITIAL_POLAR_SIDE),
        (witness.radial_turnings, _SOURCE_EDGE_RADIAL_TURNINGS),
        (witness.polar_turnings, _SOURCE_EDGE_POLAR_TURNINGS),
        (
            witness.equatorial_crossings_before_terminal,
            _SURFACE_EQUATORIAL_CROSSINGS_BEFORE_TERMINAL,
        ),
        (witness.azimuth_winding, _SOURCE_EDGE_AZIMUTH_WINDING),
    )
    if any(
        type(actual) is not type(expected) or actual != expected
        for actual, expected in discrete_identity
    ):
        raise _UnsupportedWitnessError(
            "surface witness does not match the named discrete path identity"
        )
    continuous_fields = (
        witness.source_radius_m,
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
            "witness contains a non-real or non-finite value"
        )
    if not (
        SURFACE_INNER_RADIUS_M <= witness.source_radius_m <= SURFACE_OUTER_RADIUS_M
    ):
        raise _UnsupportedWitnessError("crossing lies outside the canonical surface")
    positive_fields = (
        witness.frequency_ratio,
        witness.travel_time_m,
        witness.emitted_bolometric_intensity,
        witness.observed_bolometric_intensity,
        witness.energy,
    )
    if any(value <= 0 for value in positive_fields):
        raise _UnsupportedWitnessError("witness contains a non-positive physical value")
    if witness.radial_turning_derivative <= 0 or witness.polar_turning_derivative <= 0:
        raise _UnsupportedWitnessError("separated turning root is not simple")

    residuals = (
        witness.initial_null_residual,
        witness.mino_constraint_residual,
        witness.chart_primitive_residual,
    )
    with mp.workdps(witness.precision_digits):
        residual_limit = mp.power(
            10,
            RESIDUAL_GUARD_DIGITS - witness.precision_digits,
        )
        if any(residual < 0 or residual >= residual_limit for residual in residuals):
            certified_digits = witness.precision_digits - RESIDUAL_GUARD_DIGITS
            raise _UnsupportedWitnessError(
                "equation residual does not retain the required "
                f"{certified_digits} decimal digits"
            )


def _validate_escape_witness(witness: _EscapeWitness) -> None:
    _validate_precision_digits(witness.precision_digits)
    discrete_identity = (
        (witness.terminal, _SOURCE_EDGE_ESCAPE_TERMINAL),
        (witness.initial_polar_side, _SOURCE_EDGE_INITIAL_POLAR_SIDE),
        (witness.radial_turnings, _SOURCE_EDGE_RADIAL_TURNINGS),
        (witness.polar_turnings, _SOURCE_EDGE_POLAR_TURNINGS),
        (
            witness.equatorial_crossings_before_terminal,
            _SOURCE_EDGE_ESCAPE_EQUATORIAL_CROSSINGS,
        ),
        (witness.azimuth_winding, _SOURCE_EDGE_AZIMUTH_WINDING),
    )
    if any(
        type(actual) is not type(expected) or actual != expected
        for actual, expected in discrete_identity
    ):
        raise _UnsupportedWitnessError(
            "escape witness does not match the named discrete path identity"
        )
    if (
        len(witness.escape_position_xyz_m) != 3
        or len(witness.escape_direction_xyz) != 3
    ):
        raise _UnsupportedWitnessError(
            "escape vectors must contain exactly three lanes"
        )
    continuous_fields = (
        witness.first_equatorial_crossing_radius_m,
        witness.escape_radius_m,
        *witness.escape_position_xyz_m,
        *witness.escape_direction_xyz,
        witness.travel_time_m,
        witness.escape_before_next_crossing_mino_margin,
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
            "escape witness contains a non-real or non-finite value"
        )
    if witness.escape_radius_m != ESCAPE_RADIUS_M:
        raise _UnsupportedWitnessError("escape witness uses the wrong terminal radius")
    if not (
        SURFACE_OUTER_RADIUS_M
        < witness.first_equatorial_crossing_radius_m
        < witness.escape_radius_m
    ):
        raise _UnsupportedWitnessError(
            "escape witness first crossing is not ordered between the outer "
            "source edge and escape terminal"
        )
    positive_fields = (
        witness.travel_time_m,
        witness.escape_before_next_crossing_mino_margin,
        witness.energy,
        witness.radial_turning_derivative,
        witness.polar_turning_derivative,
    )
    if any(value <= 0 for value in positive_fields):
        raise _UnsupportedWitnessError(
            "escape witness contains a non-positive physical or event margin"
        )

    with mp.workdps(witness.precision_digits):
        residual_limit = mp.power(
            10,
            RESIDUAL_GUARD_DIGITS - witness.precision_digits,
        )
        direction_norm_squared = mp.fsum(
            component**2 for component in witness.escape_direction_xyz
        )
        if abs(direction_norm_squared - 1) >= residual_limit:
            raise _UnsupportedWitnessError(
                "escape traversal direction is not normalized"
            )
        if (
            mp.fsum(
                position * direction
                for position, direction in zip(
                    witness.escape_position_xyz_m,
                    witness.escape_direction_xyz,
                    strict=True,
                )
            )
            <= 0
        ):
            raise _UnsupportedWitnessError("escape traversal direction is not outward")

        x, y, z = witness.escape_position_xyz_m
        spin = mp.mpf(4) / 5
        cylindrical_squared = x**2 + y**2
        oblate_term = cylindrical_squared + z**2 - spin**2
        recovered_radius_squared = (
            oblate_term + mp.sqrt(oblate_term**2 + 4 * spin**2 * z**2)
        ) / 2
        radius_residual = (
            abs(mp.sqrt(recovered_radius_squared) - witness.escape_radius_m)
            / witness.escape_radius_m
        )
        if radius_residual >= residual_limit:
            raise _UnsupportedWitnessError(
                "escape position does not lie on the named oblate radius"
            )

        residuals = (
            witness.initial_null_residual,
            witness.mino_constraint_residual,
            witness.chart_primitive_residual,
        )
        if any(residual < 0 or residual >= residual_limit for residual in residuals):
            certified_digits = witness.precision_digits - RESIDUAL_GUARD_DIGITS
            raise _UnsupportedWitnessError(
                "escape equation residual does not retain the required "
                f"{certified_digits} decimal digits"
            )

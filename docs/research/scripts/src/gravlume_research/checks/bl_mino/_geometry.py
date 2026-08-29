"""Construct observations and map Kerr--Schild rays to separated BL state."""

from collections.abc import Iterable, Sequence

import mpmath as mp

from ._model import (
    _CRITICAL_VIEWPORT_HEIGHT,
    _CRITICAL_VIEWPORT_WIDTH,
    _INGOING_CHART_SIGN,
    _NEGATIVE_SPIN_VIEWPORT_HEIGHT,
    _NEGATIVE_SPIN_VIEWPORT_WIDTH,
    _OUTGOING_CHART_SIGN,
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
    _InitialRay,
    _ObservationGeometry,
    _SeparatedState,
    _UnsupportedWitnessError,
)


def _vector_add(left: Sequence[mp.mpf], right: Sequence[mp.mpf]) -> tuple[mp.mpf, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _vector_scale(vector: Sequence[mp.mpf], scalar: mp.mpf) -> tuple[mp.mpf, ...]:
    return tuple(scalar * component for component in vector)


def _metric_dot(
    metric: Sequence[Sequence[mp.mpf]],
    left: Sequence[mp.mpf],
    right: Sequence[mp.mpf],
) -> mp.mpf:
    return mp.fsum(
        metric[row][column] * left[row] * right[column]
        for row in range(4)
        for column in range(4)
    )


def _lower(
    metric: Sequence[Sequence[mp.mpf]], vector: Sequence[mp.mpf]
) -> tuple[mp.mpf, ...]:
    return tuple(
        mp.fsum(metric[row][column] * vector[column] for column in range(4))
        for row in range(4)
    )


def _project_and_normalize(
    metric: Sequence[Sequence[mp.mpf]],
    four_velocity: Sequence[mp.mpf],
    seed: Sequence[mp.mpf],
    orthogonal_to: Iterable[Sequence[mp.mpf]],
) -> tuple[mp.mpf, ...]:
    projected = _vector_add(
        seed,
        _vector_scale(four_velocity, _metric_dot(metric, four_velocity, seed)),
    )
    for basis in orthogonal_to:
        projected = _vector_add(
            projected,
            _vector_scale(basis, -_metric_dot(metric, projected, basis)),
        )
    norm_squared = _metric_dot(metric, projected, projected)
    if norm_squared <= 0:
        raise _UnsupportedWitnessError("canonical observer frame seed is degenerate")
    return _vector_scale(projected, 1 / mp.sqrt(norm_squared))


def _orientation_determinant(columns: Sequence[Sequence[mp.mpf]]) -> mp.mpf:
    matrix = mp.matrix(
        [[columns[column][row] for column in range(4)] for row in range(4)]
    )
    return mp.det(matrix)


def _build_observation_geometry(
    *,
    spin: mp.mpf,
    chart_sign: int,
    radius: mp.mpf,
    theta: mp.mpf,
    chart_azimuth: mp.mpf,
    viewport_width: int,
    viewport_height: int,
    vertical_fov: mp.mpf,
) -> _ObservationGeometry:
    """Reconstruct one pure-Kerr observation without importing Rust state."""

    if chart_sign not in (_INGOING_CHART_SIGN, _OUTGOING_CHART_SIGN):
        raise _UnsupportedWitnessError("Kerr--Schild chart sign must be +1 or -1")
    mass = mp.mpf(1)
    sin_theta = mp.sin(theta)
    cos_theta = mp.cos(theta)
    sin_azimuth = mp.sin(chart_azimuth)
    cos_azimuth = mp.cos(chart_azimuth)
    chart_spin = chart_sign * spin
    x = (radius * cos_azimuth - chart_spin * sin_azimuth) * sin_theta
    y = (radius * sin_azimuth + chart_spin * cos_azimuth) * sin_theta
    z = radius * cos_theta
    sigma = radius**2 + spin**2 * cos_theta**2
    scalar_f = 2 * mass * radius / sigma
    principal = (
        mp.mpf(1),
        (chart_sign * radius * x + spin * y) / (radius**2 + spin**2),
        (chart_sign * radius * y - spin * x) / (radius**2 + spin**2),
        chart_sign * z / radius,
    )
    minkowski = (-1, 1, 1, 1)
    metric = tuple(
        tuple(
            (mp.mpf(minkowski[row]) if row == column else mp.mpf(0))
            + scalar_f * principal[row] * principal[column]
            for column in range(4)
        )
        for row in range(4)
    )
    return _ObservationGeometry(
        mass=mass,
        spin=spin,
        chart_sign=chart_sign,
        radius=radius,
        theta=theta,
        chart_azimuth=chart_azimuth,
        position=(x, y, z),
        metric=metric,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        vertical_fov=vertical_fov,
    )


def _canonical_geometry() -> _ObservationGeometry:
    return _build_observation_geometry(
        spin=mp.mpf(4) / 5,
        chart_sign=_INGOING_CHART_SIGN,
        radius=mp.mpf(30),
        theta=mp.pi / 3,
        chart_azimuth=mp.mpf(0),
        viewport_width=VIEWPORT_WIDTH,
        viewport_height=VIEWPORT_HEIGHT,
        vertical_fov=mp.pi / 4,
    )


def _critical_curve_geometry() -> _ObservationGeometry:
    return _build_observation_geometry(
        spin=mp.mpf(4) / 5,
        chart_sign=_OUTGOING_CHART_SIGN,
        radius=mp.mpf(30),
        theta=mp.pi / 3,
        chart_azimuth=mp.mpf(0),
        viewport_width=_CRITICAL_VIEWPORT_WIDTH,
        viewport_height=_CRITICAL_VIEWPORT_HEIGHT,
        vertical_fov=mp.pi / 4,
    )


def _negative_spin_geometry() -> _ObservationGeometry:
    return _build_observation_geometry(
        spin=-mp.mpf(4) / 5,
        chart_sign=_OUTGOING_CHART_SIGN,
        radius=mp.mpf(12),
        theta=mp.pi / 3,
        chart_azimuth=mp.mpf(0),
        viewport_width=_NEGATIVE_SPIN_VIEWPORT_WIDTH,
        viewport_height=_NEGATIVE_SPIN_VIEWPORT_HEIGHT,
        vertical_fov=mp.pi / 4,
    )


def _try_image_right_axis(
    metric: Sequence[Sequence[mp.mpf]],
    four_velocity: Sequence[mp.mpf],
    sight: Sequence[mp.mpf],
    up: Sequence[mp.mpf],
    axis: Sequence[mp.mpf],
) -> tuple[mp.mpf, ...] | None:
    try:
        return _project_and_normalize(metric, four_velocity, axis, (sight, up))
    except _UnsupportedWitnessError:
        return None


def _canonical_initial_ray(
    geometry: _ObservationGeometry,
    pixel_x: int | mp.mpf,
    pixel_y: int | mp.mpf,
    *,
    coordinates_are_centers: bool = False,
) -> _InitialRay:
    x, y, z = geometry.position
    metric = geometry.metric
    g_tt = metric[0][0]
    four_velocity = (1 / mp.sqrt(-g_tt), mp.mpf(0), mp.mpf(0), mp.mpf(0))
    sight = _project_and_normalize(
        metric,
        four_velocity,
        (mp.mpf(0), -x, -y, -z),
        (),
    )
    arrival = _vector_scale(sight, -1)
    up = _project_and_normalize(
        metric,
        four_velocity,
        (mp.mpf(0), mp.mpf(0), mp.mpf(0), mp.mpf(1)),
        (sight,),
    )
    right_candidates = tuple(
        candidate
        for candidate in (
            _try_image_right_axis(
                metric,
                four_velocity,
                sight,
                up,
                axis,
            )
            for axis in (
                (mp.mpf(0), mp.mpf(1), mp.mpf(0), mp.mpf(0)),
                (mp.mpf(0), mp.mpf(0), mp.mpf(1), mp.mpf(0)),
                (mp.mpf(0), mp.mpf(0), mp.mpf(0), mp.mpf(1)),
            )
        )
        if candidate is not None
    )
    if not right_candidates:
        raise _UnsupportedWitnessError(
            "canonical observer frame has no image-right axis"
        )
    right = max(
        right_candidates,
        key=lambda candidate: max(abs(component) for component in candidate[1:]),
    )
    if _orientation_determinant((four_velocity, right, up, arrival)) < 0:
        right = _vector_scale(right, -1)

    width = mp.mpf(geometry.viewport_width)
    height = mp.mpf(geometry.viewport_height)
    half = mp.mpf(1) / 2
    sample_x = mp.mpf(pixel_x) if coordinates_are_centers else mp.mpf(pixel_x) + half
    sample_y = mp.mpf(pixel_y) if coordinates_are_centers else mp.mpf(pixel_y) + half
    normalized_x = 2 * sample_x / width - 1
    normalized_y = 1 - 2 * sample_y / height
    tangent_half_fov = mp.tan(geometry.vertical_fov / 2)
    sight_x = width / height * tangent_half_fov * normalized_x
    sight_y = tangent_half_fov * normalized_y
    normalization = 1 / mp.sqrt(1 + sight_x**2 + sight_y**2)
    sight_direction = _vector_scale(
        _vector_add(
            _vector_add(_vector_scale(right, sight_x), _vector_scale(up, sight_y)),
            _vector_scale(arrival, -1),
        ),
        normalization,
    )
    photon_arrival = _vector_scale(sight_direction, -1)
    momentum_contravariant = _vector_add(four_velocity, photon_arrival)
    momentum_covariant = _lower(metric, momentum_contravariant)
    observer_frequency = -mp.fsum(
        covector * vector
        for covector, vector in zip(momentum_covariant, four_velocity, strict=True)
    )
    null_value = _metric_dot(metric, momentum_contravariant, momentum_contravariant)
    null_term_norm = mp.fsum(
        abs(
            metric[row][column]
            * momentum_contravariant[row]
            * momentum_contravariant[column]
        )
        for row in range(4)
        for column in range(4)
    )
    initial_null_residual = abs(null_value) / max(mp.mpf(1), null_term_norm)
    return _InitialRay(
        momentum_covariant=momentum_covariant,
        observer_frequency=observer_frequency,
        initial_null_residual=initial_null_residual,
    )


def _separated_initial_state(
    geometry: _ObservationGeometry,
    initial_ray: _InitialRay,
) -> _SeparatedState:
    mass = geometry.mass
    spin = geometry.spin
    radius = geometry.radius
    theta = geometry.theta
    x, y, _ = geometry.position
    sin_theta = mp.sin(theta)
    cos_theta = mp.cos(theta)
    p_t, p_x, p_y, p_z = initial_ray.momentum_covariant
    p_r_ks = (
        sin_theta * mp.cos(geometry.chart_azimuth) * p_x
        + sin_theta * mp.sin(geometry.chart_azimuth) * p_y
        + cos_theta * p_z
    )
    p_theta = mp.cos(theta) / sin_theta * (x * p_x + y * p_y) - radius * sin_theta * p_z
    p_phi = x * p_y - y * p_x
    delta = radius**2 - 2 * mass * radius + spin**2
    p_r_bl = p_r_ks + geometry.chart_sign * (
        2 * mass * radius / delta * p_t + spin / delta * p_phi
    )
    energy = -p_t
    impact = p_phi / energy
    mu = cos_theta
    carter = (p_theta / energy) ** 2 + mu**2 * (impact**2 / (1 - mu**2) - spin**2)
    radial_velocity = delta * p_r_bl / energy
    polar_velocity = -sin_theta * p_theta / energy

    def radial_potential(value: mp.mpf) -> mp.mpf:
        radial_factor = value**2 + spin**2 - spin * impact
        separation = (impact - spin) ** 2 + carter
        return radial_factor**2 - (value**2 - 2 * mass * value + spin**2) * separation

    def polar_potential(value: mp.mpf) -> mp.mpf:
        return carter + (spin**2 - carter - impact**2) * value**2 - spin**2 * value**4

    radial_residual = abs(radial_velocity**2 - radial_potential(radius)) / max(
        mp.mpf(1),
        abs(radial_velocity**2),
        abs(radial_potential(radius)),
    )
    polar_residual = abs(polar_velocity**2 - polar_potential(mu)) / max(
        mp.mpf(1),
        abs(polar_velocity**2),
        abs(polar_potential(mu)),
    )
    return _SeparatedState(
        energy=energy,
        impact=impact,
        carter=carter,
        radial_velocity=radial_velocity,
        polar_velocity=polar_velocity,
        constraint_residual=max(radial_residual, polar_residual),
    )


def _real_polynomial_roots(coefficients: Sequence[mp.mpf]) -> tuple[mp.mpf, ...]:
    roots = mp.polyroots(
        coefficients,
        maxsteps=400,
        cleanup=False,
        extraprec=80,
    )
    imaginary_tolerance = mp.power(10, -(mp.mp.dps - 30))
    return tuple(
        mp.re(root) for root in roots if abs(mp.im(root)) <= imaginary_tolerance
    )


def _chart_primitives(
    lower: mp.mpf,
    upper: mp.mpf,
    mass: mp.mpf,
    spin: mp.mpf,
) -> tuple[mp.mpf, mp.mpf]:
    horizon_gap = mp.sqrt(mass**2 - spin**2)
    outer = mass + horizon_gap
    inner = mass - horizon_gap
    denominator = outer - inner

    def time_primitive(radius: mp.mpf) -> mp.mpf:
        return 2 * mass * outer / denominator * mp.log(
            radius - outer
        ) - 2 * mass * inner / denominator * mp.log(radius - inner)

    def azimuth_primitive(radius: mp.mpf) -> mp.mpf:
        return spin / denominator * (mp.log(radius - outer) - mp.log(radius - inner))

    return (
        time_primitive(upper) - time_primitive(lower),
        azimuth_primitive(upper) - azimuth_primitive(lower),
    )


def _wrap_angle(angle: mp.mpf) -> mp.mpf:
    return angle - 2 * mp.pi * mp.floor((angle + mp.pi) / (2 * mp.pi))


def _oblate_position(
    geometry: _ObservationGeometry,
    radius: mp.mpf,
    mu: mp.mpf,
    chart_azimuth: mp.mpf,
) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Map one signed BL polar endpoint into the selected KS chart."""

    sin_theta = mp.sqrt(1 - mu**2)
    sin_azimuth = mp.sin(chart_azimuth)
    cos_azimuth = mp.cos(chart_azimuth)
    chart_spin = geometry.chart_sign * geometry.spin
    return (
        (radius * cos_azimuth - chart_spin * sin_azimuth) * sin_theta,
        (radius * sin_azimuth + chart_spin * cos_azimuth) * sin_theta,
        radius * mu,
    )


def _azimuth_winding(
    geometry: _ObservationGeometry,
    terminal_radius: mp.mpf,
    terminal_chart_azimuth_unwrapped: mp.mpf,
) -> int:
    observer_cartesian_azimuth = geometry.chart_azimuth + mp.atan2(
        geometry.chart_sign * geometry.spin,
        geometry.radius,
    )
    terminal_cartesian_azimuth = terminal_chart_azimuth_unwrapped + mp.atan2(
        geometry.chart_sign * geometry.spin,
        terminal_radius,
    )
    observer_cycle = mp.floor((observer_cartesian_azimuth + mp.pi) / (2 * mp.pi))
    terminal_cycle = mp.floor((terminal_cartesian_azimuth + mp.pi) / (2 * mp.pi))
    return int(terminal_cycle - observer_cycle)

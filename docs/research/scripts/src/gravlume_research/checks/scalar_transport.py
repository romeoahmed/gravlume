"""Verify the invariant scalar-transfer and blackbody-band contracts.

SymPy checks identities without sharing the Rust implementation. mpmath then
provides 80-decimal numerical oracles for the versioned observer-frame boxcar
bands and the cancellation-sensitive homogeneous-slab limits. This is a
research/validation tool, not a runtime dependency.
"""

import math

import mpmath as mp
import sympy as sp

from .._binary32 import round_binary32
from .._sympy import require_equal

ORACLE_PRECISION_DIGITS = 80
SECOND_RADIATION_CONSTANT_M_K_DECIMAL = "0.014387768775039337"
BANDS_NM = (
    ("red", 600, 700),
    ("green", 500, 600),
    ("blue", 400, 500),
)
ORACLE_TEMPERATURES_K = (1000, 3000, 5778, 6000, 10_000)


def verify_blackbody_redshift_identity() -> None:
    frequency, temperature, ratio = sp.symbols("nu T g", finite=True, positive=True)

    def planck_shape(nu: sp.Expr, temp: sp.Expr) -> sp.Expr:
        return nu**3 / (sp.exp(nu / temp) - 1)

    observed = ratio**3 * planck_shape(frequency / ratio, temperature)
    shifted = planck_shape(frequency, ratio * temperature)
    require_equal(observed, shifted, "I_nu/nu^3 blackbody temperature shift")

    emitted_bolometric = sp.symbols("I_em", finite=True, nonnegative=True)
    # d(nu_obs) = g d(nu_em), while spectral intensity contributes g^3.
    require_equal(
        ratio**3 * emitted_bolometric * ratio,
        ratio**4 * emitted_bolometric,
        "bolometric g^4 transport",
    )


def verify_homogeneous_slab_identities() -> None:
    incoming, source, first_depth, second_depth = sp.symbols(
        "I S tau_1 tau_2", finite=True, nonnegative=True
    )

    def transport(intensity: sp.Expr, optical_depth: sp.Expr) -> sp.Expr:
        return intensity * sp.exp(-optical_depth) + source * (
            1 - sp.exp(-optical_depth)
        )

    require_equal(
        transport(transport(incoming, first_depth), second_depth),
        transport(incoming, first_depth + second_depth),
        "constant-source slab partition invariance",
    )
    require_equal(
        transport(incoming, sp.Integer(0)),
        incoming,
        "zero-depth vacuum limit",
    )

    absorption, emissivity, length = sp.symbols(
        "alpha j length", finite=True, positive=True
    )
    coefficient_form = incoming * sp.exp(-absorption * length) + (
        emissivity / absorption
    ) * (1 - sp.exp(-absorption * length))
    pure_emission_limit = sp.limit(coefficient_form, absorption, 0, dir="+")
    require_equal(
        pure_emission_limit,
        incoming + emissivity * length,
        "zero-absorption pure-emission limit",
    )


def _dimensionless_planck_bounds(
    temperature_kelvin: mp.mpf,
    lower_nm: int | mp.mpf,
    upper_nm: int | mp.mpf,
) -> tuple[mp.mpf, mp.mpf]:
    second_radiation_constant = mp.mpf(SECOND_RADIATION_CONSTANT_M_K_DECIMAL)
    nanometers_to_meters = mp.mpf("1e-9")
    return (
        second_radiation_constant
        / (mp.mpf(upper_nm) * nanometers_to_meters * temperature_kelvin),
        second_radiation_constant
        / (mp.mpf(lower_nm) * nanometers_to_meters * temperature_kelvin),
    )


def planck_band_fraction(
    temperature_kelvin: mp.mpf,
    lower_nm: int | mp.mpf,
    upper_nm: int | mp.mpf,
) -> mp.mpf:
    lower_x, upper_x = _dimensionless_planck_bounds(
        temperature_kelvin, lower_nm, upper_nm
    )
    integral = mp.quad(lambda x: x**3 / mp.expm1(x), [lower_x, upper_x])
    return integral / (mp.pi**4 / 15)


def planck_tail(dimensionless_frequency: mp.mpf) -> mp.mpf:
    x = dimensionless_frequency
    exponential = mp.exp(-x)
    return (
        -(x**3) * mp.log1p(-exponential)
        + 3 * x**2 * mp.polylog(2, exponential)
        + 6 * x * mp.polylog(3, exponential)
        + 6 * mp.polylog(4, exponential)
    )


def planck_band_fraction_fast(
    temperature_kelvin: mp.mpf,
    lower_nm: int | mp.mpf,
    upper_nm: int | mp.mpf,
) -> mp.mpf:
    lower_x, upper_x = _dimensionless_planck_bounds(
        temperature_kelvin, lower_nm, upper_nm
    )
    return (planck_tail(lower_x) - planck_tail(upper_x)) / (mp.pi**4 / 15)


def verify_log_temperature_lut() -> tuple[mp.mpf, mp.mpf]:
    minimum_log2 = mp.mpf(-8)
    intervals_per_octave = mp.mpf(128)
    grid: list[tuple[float, ...]] = []
    for index in range(4097):
        temperature = mp.power(2, minimum_log2 + index / intervals_per_octave)
        grid.append(
            tuple(
                round_binary32(
                    float(
                        mp.log(planck_band_fraction_fast(temperature, lower, upper), 2)
                    )
                )
                for _, lower, upper in BANDS_NM
            )
        )

    worst_absolute = mp.mpf(0)
    worst_visible_relative = mp.mpf(0)
    relative_floor = mp.mpf("1e-6")
    for index in range(4096):
        coordinate = mp.mpf(index) + mp.mpf("0.5")
        temperature = mp.power(2, minimum_log2 + coordinate / intervals_per_octave)
        for channel, (_, lower, upper) in enumerate(BANDS_NM):
            expected = planck_band_fraction_fast(temperature, lower, upper)
            interpolated_log2 = mp.mpf(grid[index][channel]) + mp.mpf("0.5") * (
                mp.mpf(grid[index + 1][channel]) - mp.mpf(grid[index][channel])
            )
            interpolated = mp.power(2, interpolated_log2)
            error = abs(interpolated - expected)
            worst_absolute = max(worst_absolute, error)
            if expected >= relative_floor:
                worst_visible_relative = max(worst_visible_relative, error / expected)

    direct = planck_band_fraction(mp.mpf(6000), mp.mpf(600), mp.mpf(700))
    fast = planck_band_fraction_fast(mp.mpf(6000), mp.mpf(600), mp.mpf(700))
    if not mp.almosteq(direct, fast, rel_eps=mp.mpf("1e-70")):
        raise AssertionError(f"Planck tail formula mismatch: {direct - fast}")
    if worst_absolute > mp.mpf("3e-6"):
        raise AssertionError(
            f"LUT absolute interpolation budget exceeded: {worst_absolute}"
        )
    if worst_visible_relative > mp.mpf("0.002"):
        raise AssertionError(
            "LUT visible-relative interpolation budget exceeded: "
            f"{worst_visible_relative}"
        )
    return worst_absolute, worst_visible_relative


def verify_scaled_low_temperature_spectrum() -> tuple[mp.mpf, ...]:
    temperature = mp.mpf(220)
    bolometric_intensity = mp.mpf("1e38")
    bands = tuple(
        bolometric_intensity * planck_band_fraction_fast(temperature, lower, upper)
        for _, lower, upper in BANDS_NM
    )
    if bands[0] <= 1 or bands[1] <= mp.mpf("1e-5"):
        raise AssertionError(f"low-temperature diluted spectrum lost scale: {bands}")
    return bands


def verify_high_precision_oracles() -> list[tuple[int, tuple[mp.mpf, ...]]]:
    total = mp.quad(lambda x: x**3 / mp.expm1(x), [0, 1, 4, 12, mp.inf])
    expected_total = mp.pi**4 / 15
    if not mp.almosteq(total, expected_total, rel_eps=mp.mpf("1e-75")):
        raise AssertionError(f"Planck normalization mismatch: {total - expected_total}")

    rows: list[tuple[int, tuple[mp.mpf, ...]]] = []
    for temperature in ORACLE_TEMPERATURES_K:
        fractions = tuple(
            planck_band_fraction(mp.mpf(temperature), lower, upper)
            for _, lower, upper in BANDS_NM
        )
        if not all(mp.isfinite(value) and value > 0 for value in fractions):
            raise AssertionError(
                f"invalid band fractions at {temperature} K: {fractions}"
            )
        if sum(fractions) >= 1:
            raise AssertionError(f"visible bands exceed bolometric power: {fractions}")
        rows.append((temperature, fractions))
    return rows


def verify_cancellation_sensitive_weight() -> None:
    optical_depth = 2.0**-55
    naive = 1.0 - math.exp(-optical_depth)
    stable = -math.expm1(-optical_depth)
    reference = -mp.expm1(-(mp.mpf(2) ** -55))
    if naive != 0.0:
        raise AssertionError(f"expected naive binary64 cancellation, got {naive}")
    relative_error = abs(mp.mpf(stable) - reference) / reference
    if stable == 0.0 or relative_error > mp.mpf(2) ** -52:
        raise AssertionError(
            f"stable emission weight lost the thin-slab limit: {stable}"
        )


def print_surface_transport_fixture_oracles() -> None:
    radius = mp.mpf("19.6506789846041094")
    ratio = mp.mpf("0.953264138194626409")
    vacuum_intensity = mp.mpf("0.0235057486961945464")
    emitted_temperature = mp.mpf(6000) / (radius / 6) ** mp.mpf("0.75")
    observed_temperature = ratio * emitted_temperature
    incoming_bands = tuple(
        vacuum_intensity * planck_band_fraction(observed_temperature, lower, upper)
        for _, lower, upper in BANDS_NM
    )
    cases = (
        ("vacuum", mp.mpf(0), mp.mpf(0), None),
        ("pure-absorption", mp.mpf("0.75"), mp.mpf(0), None),
        (
            "constant-blackbody",
            mp.mpf("0.35"),
            mp.mpf("0.05") * -mp.expm1(mp.mpf("-0.35")),
            mp.mpf(4500),
        ),
        ("pure-emission-blackbody", mp.mpf(0), mp.mpf("0.003"), mp.mpf(4500)),
    )
    print(f"fixture_emitted_temperature_K={mp.nstr(emitted_temperature, 30)}")
    print(f"fixture_observed_temperature_K={mp.nstr(observed_temperature, 30)}")
    for name, optical_depth, integrated_emission, emission_temperature in cases:
        transmittance = mp.exp(-optical_depth)
        observed_bolometric = vacuum_intensity * transmittance + integrated_emission
        emission_bands = (mp.mpf(0),) * 3
        if emission_temperature is not None:
            emission_bands = tuple(
                integrated_emission
                * planck_band_fraction(emission_temperature, lower, upper)
                for _, lower, upper in BANDS_NM
            )
        observed_bands = tuple(
            incoming * transmittance + emission
            for incoming, emission in zip(incoming_bands, emission_bands, strict=True)
        )
        print(f"fixture_{name}_bolometric={mp.nstr(observed_bolometric, 30)}")
        print(
            f"fixture_{name}_bands="
            + ",".join(mp.nstr(value, 30) for value in observed_bands)
        )


def run() -> None:
    with mp.workdps(ORACLE_PRECISION_DIGITS):
        verify_blackbody_redshift_identity()
        verify_homogeneous_slab_identities()
        rows = verify_high_precision_oracles()
        verify_cancellation_sensitive_weight()
        lut_absolute, lut_visible_relative = verify_log_temperature_lut()
        scaled_low_temperature = verify_scaled_low_temperature_spectrum()

        print("I_nu/nu^3 blackbody shift and bolometric g^4 identities: PASS")
        print("Homogeneous-slab limits and partition invariance: PASS")
        print("Planck normalization and binary64 cancellation oracle: PASS")
        print(
            "4097-entry log2-temperature/log2-fraction LUT midpoint scan: "
            f"max_abs={mp.nstr(lut_absolute, 12)}, "
            f"max_rel_for_fraction_ge_1e-6={mp.nstr(lut_visible_relative, 12)}"
        )
        print(
            "scaled_220K_I1e38_bands="
            + ",".join(mp.nstr(value, 24) for value in scaled_low_temperature)
        )
        print("temperature_K,red_600_700,green_500_600,blue_400_500")
        for temperature, fractions in rows:
            values = ",".join(mp.nstr(value, 24) for value in fractions)
            print(f"{temperature},{values}")
        print_surface_transport_fixture_oracles()
        print("RESULT=PASS")

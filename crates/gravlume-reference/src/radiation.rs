use std::f64::consts::PI;

use gravlume_domain::{
    EquatorialEmissionModel, EquatorialSurface, ScalarSlabEmissionModel, SpectralBand,
    SurfaceTransport, VISIBLE_BOXCAR_BANDS_V1,
};

const SECOND_RADIATION_CONSTANT_M_K: f64 = 0.014_387_768_775_039_337;
const PLANCK_INTEGRAL: f64 = PI * PI * PI * PI / 15.0;
const MAX_DIMENSIONLESS_FREQUENCY: f64 = 80.0;
const PLANCK_UNIT_INTERVAL_COUNT: u32 = 80;

pub fn blackbody_band_intensities(
    bolometric_intensity: f64,
    temperature_kelvin: f64,
) -> Option<[f64; 3]> {
    if !bolometric_intensity.is_finite()
        || bolometric_intensity < 0.0
        || !temperature_kelvin.is_finite()
        || temperature_kelvin <= 0.0
    {
        return None;
    }

    let mut intensities = [0.0; 3];
    for (intensity, band) in intensities.iter_mut().zip(VISIBLE_BOXCAR_BANDS_V1) {
        let fraction = blackbody_band_fraction(temperature_kelvin, band)?;
        *intensity = bolometric_intensity * fraction;
        if !intensity.is_finite() || *intensity < 0.0 {
            return None;
        }
    }
    Some(intensities)
}

pub fn transport_bolometric_intensity(
    incoming: f64,
    transport: SurfaceTransport,
) -> Option<(f64, f64, f64)> {
    let slab = match transport {
        SurfaceTransport::Vacuum => return Some((incoming, 0.0, 1.0)),
        SurfaceTransport::HomogeneousScalar(slab) => slab,
    };
    let optical_depth = slab.optical_depth();
    let transmittance = (-optical_depth).exp();
    let outgoing = incoming.mul_add(transmittance, slab.integrated_bolometric_emission());
    if outgoing.is_finite() && outgoing >= 0.0 {
        Some((outgoing, optical_depth, transmittance))
    } else {
        None
    }
}

pub fn transport_blackbody_bands(
    incoming: [f64; 3],
    surface: EquatorialSurface,
) -> Option<[f64; 3]> {
    debug_assert!(matches!(
        surface.emitter().emission_model(),
        EquatorialEmissionModel::InverseCubeBlackbodyV1 { .. }
    ));
    let slab = match surface.transport() {
        SurfaceTransport::Vacuum => return Some(incoming),
        SurfaceTransport::HomogeneousScalar(slab) => slab,
    };
    let transmittance = (-slab.optical_depth()).exp();
    let integrated_emission = slab.integrated_bolometric_emission();
    let source = if integrated_emission == 0.0 {
        [0.0; 3]
    } else {
        let ScalarSlabEmissionModel::BlackbodyV1 { temperature_kelvin } = slab.emission_model()
        else {
            return None;
        };
        blackbody_band_intensities(integrated_emission, temperature_kelvin)?
    };

    let mut outgoing = [0.0; 3];
    for ((outgoing, incoming), source) in outgoing.iter_mut().zip(incoming).zip(source) {
        *outgoing = incoming.mul_add(transmittance, source);
        if !outgoing.is_finite() || *outgoing < 0.0 {
            return None;
        }
    }
    Some(outgoing)
}

fn blackbody_band_fraction(temperature_kelvin: f64, band: SpectralBand) -> Option<f64> {
    let lower_wavelength_m = band.lower_wavelength_nm() * 1.0e-9;
    let upper_wavelength_m = band.upper_wavelength_nm() * 1.0e-9;
    let lower_x = SECOND_RADIATION_CONSTANT_M_K / (upper_wavelength_m * temperature_kelvin);
    let upper_x = SECOND_RADIATION_CONSTANT_M_K / (lower_wavelength_m * temperature_kelvin);
    if !lower_x.is_finite() || !upper_x.is_finite() || lower_x < 0.0 || upper_x <= lower_x {
        return None;
    }
    if lower_x >= MAX_DIMENSIONLESS_FREQUENCY {
        return Some(0.0);
    }
    let upper_x = upper_x.min(MAX_DIMENSIONLESS_FREQUENCY);
    let integral = integrate_planck_kernel(lower_x, upper_x);
    let fraction = integral / PLANCK_INTEGRAL;
    (fraction.is_finite() && (0.0..=1.0).contains(&fraction)).then_some(fraction)
}

fn integrate_planck_kernel(lower: f64, upper: f64) -> f64 {
    (0..PLANCK_UNIT_INTERVAL_COUNT)
        .filter_map(|index| {
            let left = lower.max(f64::from(index));
            let right = upper.min(f64::from(index + 1));
            (left < right).then_some((left, right))
        })
        .map(|(left, right)| {
            let midpoint = 0.5 * (left + right);
            let whole = simpson(
                left,
                right,
                planck_kernel(left),
                planck_kernel(midpoint),
                planck_kernel(right),
            );
            let tolerance = 8.0 * f64::EPSILON * whole.abs().max(f64::MIN_POSITIVE);
            adaptive_simpson(left, right, whole, tolerance, 20)
        })
        .sum()
}

fn adaptive_simpson(left: f64, right: f64, whole: f64, tolerance: f64, depth: u32) -> f64 {
    let midpoint = 0.5 * (left + right);
    let left_midpoint = 0.5 * (left + midpoint);
    let right_midpoint = 0.5 * (midpoint + right);
    let left_integral = simpson(
        left,
        midpoint,
        planck_kernel(left),
        planck_kernel(left_midpoint),
        planck_kernel(midpoint),
    );
    let right_integral = simpson(
        midpoint,
        right,
        planck_kernel(midpoint),
        planck_kernel(right_midpoint),
        planck_kernel(right),
    );
    let refined = left_integral + right_integral;
    let correction = refined - whole;
    if depth == 0 || correction.abs() <= 15.0 * tolerance {
        refined + correction / 15.0
    } else {
        adaptive_simpson(left, midpoint, left_integral, 0.5 * tolerance, depth - 1)
            + adaptive_simpson(midpoint, right, right_integral, 0.5 * tolerance, depth - 1)
    }
}

fn simpson(left: f64, right: f64, left_value: f64, midpoint_value: f64, right_value: f64) -> f64 {
    (right - left) * (4.0_f64.mul_add(midpoint_value, left_value) + right_value) / 6.0
}

fn planck_kernel(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if x > 50.0 {
        let exponential = (-x).exp();
        x * x * x * exponential / (1.0 - exponential)
    } else {
        x * x * x / x.exp_m1()
    }
}

#[cfg(test)]
mod tests {
    use gravlume_domain::{HomogeneousScalarSlab, SurfaceTransport};
    use proptest::prelude::*;

    use super::{
        VISIBLE_BOXCAR_BANDS_V1, blackbody_band_intensities, integrate_planck_kernel,
        transport_bolometric_intensity,
    };

    fn lut_log2_temperature() -> impl Strategy<Value = f64> {
        prop_oneof![Just(-8.0), Just(24.0), -8.0_f64..=24.0]
    }

    fn non_negative_intensity() -> impl Strategy<Value = f64> {
        prop_oneof![Just(0.0), 0.0_f64..=1.0e6]
    }

    fn optical_depth() -> impl Strategy<Value = f64> {
        prop_oneof![Just(0.0), Just(20.0), 0.0_f64..=20.0]
    }

    #[test]
    fn dimensionless_planck_integral_closes_stefan_boltzmann_normalization() {
        let expected = std::f64::consts::PI.powi(4) / 15.0;
        let actual = integrate_planck_kernel(0.0, 80.0);

        assert!((actual - expected).abs() <= 4.0e-14);
    }

    #[test]
    fn visible_boxcar_v1_preserves_its_versioned_channel_order() {
        assert_eq!(
            VISIBLE_BOXCAR_BANDS_V1.map(|band| (
                band.name(),
                band.lower_wavelength_nm(),
                band.upper_wavelength_nm(),
            )),
            [
                ("red", 600.0, 700.0),
                ("green", 500.0, 600.0),
                ("blue", 400.0, 500.0),
            ]
        );
    }

    #[test]
    fn six_thousand_kelvin_bands_match_the_independent_high_precision_oracle() {
        let actual = blackbody_band_intensities(1.0, 6_000.0)
            .expect("the fixture temperature is representable");
        let expected = [
            0.112_401_203_671_287_7,
            0.130_369_212_437_614_94,
            0.132_971_877_537_021_7,
        ];

        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() <= 3.0e-15);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn visible_boxcar_bands_remain_a_bounded_bolometric_fraction(
            log2_temperature in lut_log2_temperature(),
        ) {
            let bands = blackbody_band_intensities(1.0, log2_temperature.exp2())
                .expect("generated LUT-domain temperature is representable");

            prop_assert!(bands.into_iter().all(|value| (0.0..=1.0).contains(&value)));
            prop_assert!(
                bands.into_iter().sum::<f64>() <= 16.0_f64.mul_add(f64::EPSILON, 1.0)
            );
        }

        #[test]
        fn constant_source_transport_is_positive_and_partition_invariant(
            incoming in non_negative_intensity(),
            source in non_negative_intensity(),
            first_depth in optical_depth(),
            second_depth in optical_depth(),
        ) {
            let first = HomogeneousScalarSlab::constant_bolometric_v1(first_depth, source)
                .expect("generated first partition is valid");
            let second = HomogeneousScalarSlab::constant_bolometric_v1(second_depth, source)
                .expect("generated second partition is valid");
            let combined = HomogeneousScalarSlab::constant_bolometric_v1(
                first_depth + second_depth,
                source,
            )
            .expect("generated combined slab is valid");
            let (after_first, _, first_transmittance) =
                transport_bolometric_intensity(incoming, SurfaceTransport::HomogeneousScalar(first))
                    .expect("first transport is representable");
            let (partitioned, _, second_transmittance) =
                transport_bolometric_intensity(after_first, SurfaceTransport::HomogeneousScalar(second))
                    .expect("second transport is representable");
            let (atomic, _, combined_transmittance) =
                transport_bolometric_intensity(incoming, SurfaceTransport::HomogeneousScalar(combined))
                    .expect("combined transport is representable");
            let scale = partitioned.abs().max(atomic.abs()).max(1.0);

            prop_assert!(partitioned >= 0.0 && atomic >= 0.0);
            prop_assert!(combined_transmittance <= first_transmittance);
            prop_assert!(combined_transmittance <= second_transmittance);
            prop_assert!(
                first_transmittance
                    .mul_add(-second_transmittance, combined_transmittance)
                    .abs()
                    <= 16.0 * f64::EPSILON,
            );
            prop_assert!((partitioned - atomic).abs() <= 32.0 * f64::EPSILON * scale);
        }
    }
}

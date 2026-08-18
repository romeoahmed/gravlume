use std::f64::consts::{LN_2, PI};

use gravlume_domain::VISIBLE_BOXCAR_BANDS_V1;

const BLACKBODY_LUT_ENTRY_COUNT: usize = 4_097;
pub const BLACKBODY_LUT_BYTE_SIZE: u64 = 65_552;
const BLACKBODY_LUT_MIN_LOG2_TEMPERATURE: f64 = -8.0;
const BLACKBODY_LUT_MAX_LOG2_TEMPERATURE: f64 = 24.0;
const BLACKBODY_LUT_INTERVALS_PER_OCTAVE: f64 = 128.0;

const SECOND_RADIATION_CONSTANT_M_K: f64 = 0.014_387_768_775_039_337;
const PLANCK_INTEGRAL: f64 = PI * PI * PI * PI / 15.0;
const HIGH_FREQUENCY_LOG_THRESHOLD: f64 = 50.0;
const BLACKBODY_LUT_LAST_INDEX: u32 = 4_096;
const INTEGRATION_BREAKS: [f64; 7] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 50.0];
const GAUSS_NODES: [f64; 8] = [
    0.095_012_509_837_637_44,
    0.281_603_550_779_258_9,
    0.458_016_777_657_227_4,
    0.617_876_244_402_643_8,
    0.755_404_408_355_003,
    0.865_631_202_387_831_8,
    0.944_575_023_073_232_6,
    0.989_400_934_991_649_9,
];
const GAUSS_WEIGHTS: [f64; 8] = [
    0.189_450_610_455_068_5,
    0.182_603_415_044_923_6,
    0.169_156_519_395_002_54,
    0.149_595_988_816_576_73,
    0.124_628_971_255_533_87,
    0.095_158_511_682_492_78,
    0.062_253_523_938_647_89,
    0.027_152_459_411_754_096,
];

const _: () = assert!(BLACKBODY_LUT_ENTRY_COUNT * size_of::<[f32; 4]>() == 65_552);

pub fn blackbody_log2_fraction_lut() -> Vec<[f32; 4]> {
    (0..=BLACKBODY_LUT_LAST_INDEX)
        .map(|index| {
            let log2_temperature = BLACKBODY_LUT_MIN_LOG2_TEMPERATURE
                + f64::from(index) / BLACKBODY_LUT_INTERVALS_PER_OCTAVE;
            let log2_fractions = blackbody_band_log2_fractions(log2_temperature.exp2());
            [
                binary32_log2_fraction(log2_fractions[0]),
                binary32_log2_fraction(log2_fractions[1]),
                binary32_log2_fraction(log2_fractions[2]),
                0.0,
            ]
        })
        .collect()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the versioned LUT intentionally rounds finite non-positive logarithms to binary32 and is checked against an 80-digit oracle"
)]
fn binary32_log2_fraction(value: f64) -> f32 {
    debug_assert!(value.is_finite() && value <= 0.0);
    value as f32
}

pub fn minimum_temperature_kelvin() -> f64 {
    BLACKBODY_LUT_MIN_LOG2_TEMPERATURE.exp2()
}

pub fn maximum_temperature_kelvin() -> f64 {
    BLACKBODY_LUT_MAX_LOG2_TEMPERATURE.exp2()
}

fn blackbody_band_log2_fractions(temperature_kelvin: f64) -> [f64; 3] {
    VISIBLE_BOXCAR_BANDS_V1.map(|band| {
        let lower_nm = band.lower_wavelength_nm();
        let upper_nm = band.upper_wavelength_nm();
        let lower_x = SECOND_RADIATION_CONSTANT_M_K / (upper_nm * 1.0e-9 * temperature_kelvin);
        let upper_x = SECOND_RADIATION_CONSTANT_M_K / (lower_nm * 1.0e-9 * temperature_kelvin);
        let log_integral = if lower_x >= HIGH_FREQUENCY_LOG_THRESHOLD {
            log_high_frequency_planck_band_integral(lower_x, upper_x)
        } else {
            integrate_planck_kernel(lower_x, upper_x).ln()
        };
        (log_integral - PLANCK_INTEGRAL.ln()) / LN_2
    })
}

fn integrate_planck_kernel(lower: f64, upper: f64) -> f64 {
    let mut integral = 0.0;
    let mut segment_start = lower;
    for segment_end in INTEGRATION_BREAKS
        .into_iter()
        .filter(|end| *end > lower && *end < upper)
        .chain([upper])
    {
        integral += integrate_gauss_legendre_segment(segment_start, segment_end);
        segment_start = segment_end;
    }
    integral
}

fn log_high_frequency_planck_band_integral(lower: f64, upper: f64) -> f64 {
    let lower_tail = log_high_frequency_planck_tail(lower);
    let upper_tail = log_high_frequency_planck_tail(upper);
    lower_tail + (-(upper_tail - lower_tail).exp()).ln_1p()
}

fn log_high_frequency_planck_tail(x: f64) -> f64 {
    // For x >= 50, replacing 1 / (exp(x) - 1) with exp(-x) has relative error below
    // exp(-50). The resulting x^3 exp(-x) tail has this closed logarithmic form.
    let inverse = x.recip();
    let polynomial_correction = inverse * inverse.mul_add(6.0_f64.mul_add(inverse, 6.0), 3.0);
    3.0_f64.mul_add(x.ln(), -x) + polynomial_correction.ln_1p()
}

fn integrate_gauss_legendre_segment(lower: f64, upper: f64) -> f64 {
    let midpoint = 0.5 * (lower + upper);
    let half_width = 0.5 * (upper - lower);
    let weighted_sum = GAUSS_NODES
        .into_iter()
        .zip(GAUSS_WEIGHTS)
        .map(|(node, weight)| {
            let offset = half_width * node;
            weight * (planck_kernel(midpoint - offset) + planck_kernel(midpoint + offset))
        })
        .sum::<f64>();
    half_width * weighted_sum
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
    use approx::assert_abs_diff_eq;

    use super::{blackbody_band_log2_fractions, blackbody_log2_fraction_lut};

    #[test]
    fn every_lut_entry_is_a_finite_non_positive_logarithm_with_zero_padding() {
        let lut = blackbody_log2_fraction_lut();

        assert!(lut.iter().all(|[red, green, blue, padding]| {
            [*red, *green, *blue]
                .into_iter()
                .all(|log2_fraction| log2_fraction.is_finite() && log2_fraction <= 0.0)
                && *padding == 0.0
        }));
    }

    #[test]
    fn gauss_legendre_generator_matches_the_independent_six_thousand_kelvin_oracle() {
        let actual = blackbody_band_log2_fractions(6_000.0).map(f64::exp2);
        let expected = [
            0.112_401_203_671_287_7,
            0.130_369_212_437_614_94,
            0.132_971_877_537_021_7,
        ];

        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_abs_diff_eq!(actual, expected, epsilon = 2.0e-14);
        }
    }
}

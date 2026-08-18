use std::f64::consts::PI;

use gravlume_domain::VISIBLE_BOXCAR_BANDS_V1;

const BLACKBODY_LUT_ENTRY_COUNT: usize = 4_097;
pub const BLACKBODY_LUT_BYTE_SIZE: u64 = 65_552;
const BLACKBODY_LUT_MIN_LOG2_TEMPERATURE: f64 = -8.0;
const BLACKBODY_LUT_MAX_LOG2_TEMPERATURE: f64 = 24.0;
const BLACKBODY_LUT_INTERVALS_PER_OCTAVE: f64 = 128.0;

const SECOND_RADIATION_CONSTANT_M_K: f64 = 0.014_387_768_775_039_337;
const PLANCK_INTEGRAL: f64 = PI * PI * PI * PI / 15.0;
const MAX_DIMENSIONLESS_FREQUENCY: f64 = 80.0;
const BLACKBODY_LUT_LAST_INDEX: u32 = 4_096;
const INTEGRATION_BREAKS: [f64; 7] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 80.0];
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

pub fn blackbody_lut() -> Vec<[f32; 4]> {
    (0..=BLACKBODY_LUT_LAST_INDEX)
        .map(|index| {
            let log2_temperature = BLACKBODY_LUT_MIN_LOG2_TEMPERATURE
                + f64::from(index) / BLACKBODY_LUT_INTERVALS_PER_OCTAVE;
            let fractions = blackbody_band_fractions(log2_temperature.exp2());
            [
                binary32_fraction(fractions[0]),
                binary32_fraction(fractions[1]),
                binary32_fraction(fractions[2]),
                0.0,
            ]
        })
        .collect()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the versioned LUT intentionally rounds finite [0, 1] fractions to binary32 and is checked against an 80-digit oracle"
)]
const fn binary32_fraction(value: f64) -> f32 {
    let fraction = value as f32;
    // WGSL permits subnormal inputs and outputs of numeric built-ins to be flushed to zero.
    // Canonical zero is therefore the only backend-independent LUT representation here, and its
    // error is many orders of magnitude below the declared 3e-6 absolute fraction budget.
    if fraction.is_subnormal() {
        0.0
    } else {
        fraction
    }
}

pub fn minimum_temperature_kelvin() -> f64 {
    BLACKBODY_LUT_MIN_LOG2_TEMPERATURE.exp2()
}

pub fn maximum_temperature_kelvin() -> f64 {
    BLACKBODY_LUT_MAX_LOG2_TEMPERATURE.exp2()
}

fn blackbody_band_fractions(temperature_kelvin: f64) -> [f64; 3] {
    VISIBLE_BOXCAR_BANDS_V1.map(|band| {
        let lower_nm = band.lower_wavelength_nm();
        let upper_nm = band.upper_wavelength_nm();
        let lower_x = SECOND_RADIATION_CONSTANT_M_K / (upper_nm * 1.0e-9 * temperature_kelvin);
        let upper_x = SECOND_RADIATION_CONSTANT_M_K / (lower_nm * 1.0e-9 * temperature_kelvin);
        integrate_planck_kernel(lower_x, upper_x) / PLANCK_INTEGRAL
    })
}

fn integrate_planck_kernel(lower: f64, upper: f64) -> f64 {
    if lower >= MAX_DIMENSIONLESS_FREQUENCY {
        return 0.0;
    }
    let upper = upper.min(MAX_DIMENSIONLESS_FREQUENCY);
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
    use super::{blackbody_band_fractions, blackbody_lut};

    #[test]
    fn every_lut_entry_is_a_bounded_fraction_with_zero_padding() {
        let lut = blackbody_lut();

        assert!(lut.iter().all(|[red, green, blue, padding]| {
            [*red, *green, *blue]
                .into_iter()
                .all(|fraction| (0.0..=1.0).contains(&fraction) && !fraction.is_subnormal())
                && red + green + blue <= 4.0_f32.mul_add(f32::EPSILON, 1.0)
                && *padding == 0.0
        }));
    }

    #[test]
    fn gauss_legendre_generator_matches_the_independent_six_thousand_kelvin_oracle() {
        let actual = blackbody_band_fractions(6_000.0);
        let expected = [
            0.112_401_203_671_287_7,
            0.130_369_212_437_614_94,
            0.132_971_877_537_021_7,
        ];

        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() <= 2.0e-14);
        }
    }
}

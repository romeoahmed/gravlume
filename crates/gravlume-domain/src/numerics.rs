use crate::state::FourVector;

pub fn normalized_quadratic_form_residual(matrix: [[f64; 4]; 4], vector: FourVector) -> f64 {
    let vector = vector.to_array();
    if !matrix.into_iter().flatten().all(f64::is_finite) || !vector.into_iter().all(f64::is_finite)
    {
        return f64::NAN;
    }
    let matrix_scale = matrix
        .into_iter()
        .flatten()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let vector_scale = vector.into_iter().map(f64::abs).fold(0.0_f64, f64::max);
    if matrix_scale == 0.0 || vector_scale == 0.0 {
        return 0.0;
    }
    let scaled_vector = vector.map(|component| component / vector_scale);
    let mut contraction = 0.0;
    let mut term_norm = 0.0;
    for row in 0..4 {
        for column in 0..4 {
            let term =
                (matrix[row][column] / matrix_scale) * scaled_vector[row] * scaled_vector[column];
            contraction += term;
            term_norm += term.abs();
        }
    }
    if term_norm == 0.0 {
        return 0.0;
    }
    let relative_residual = contraction.abs() / term_norm;
    let denominator_scale =
        positive_product_capped_at_one([term_norm, matrix_scale, vector_scale, vector_scale]);
    relative_residual * denominator_scale
}

pub fn normalized_inner_product_residual<const N: usize>(
    left: [f64; N],
    right: [f64; N],
    expected: f64,
) -> Option<f64> {
    if !expected.is_finite()
        || !left.into_iter().all(f64::is_finite)
        || !right.into_iter().all(f64::is_finite)
    {
        return None;
    }

    let mut terms = [(0.0, 0_i32); N];
    let mut maximum_exponent = None;
    for (index, (left, right)) in left.into_iter().zip(right).enumerate() {
        if left == 0.0 || right == 0.0 {
            continue;
        }
        let (left_significand, left_exponent) = positive_binary_decomposition(left.abs());
        let (right_significand, right_exponent) = positive_binary_decomposition(right.abs());
        let mut significand = left_significand * right_significand;
        let mut exponent = left_exponent + right_exponent;
        if significand >= 2.0 {
            significand *= 0.5;
            exponent += 1;
        }
        if left.is_sign_negative() != right.is_sign_negative() {
            significand = -significand;
        }
        terms[index] = (significand, exponent);
        maximum_exponent =
            Some(maximum_exponent.map_or(exponent, |current: i32| current.max(exponent)));
    }

    let Some(maximum_exponent) = maximum_exponent else {
        return Some(expected.abs());
    };
    let (contraction, term_norm) = terms.into_iter().fold(
        (0.0, 0.0),
        |(contraction, term_norm), (significand, exponent)| {
            if significand == 0.0 {
                return (contraction, term_norm);
            }
            let scaled = significand * binary_power(exponent - maximum_exponent);
            (contraction + scaled, term_norm + scaled.abs())
        },
    );
    if !contraction.is_finite() || !term_norm.is_finite() || term_norm == 0.0 {
        return None;
    }

    let (_, term_norm_exponent) = positive_binary_decomposition(term_norm);
    let residual = if maximum_exponent + term_norm_exponent >= 0 {
        let expected_scaled = expected * binary_power(-maximum_exponent);
        (contraction - expected_scaled).abs() / term_norm
    } else {
        let product = contraction * binary_power(maximum_exponent);
        (product - expected).abs()
    };
    residual.is_finite().then_some(residual)
}

fn positive_product_capped_at_one(factors: [f64; 4]) -> f64 {
    positive_product(factors).min(1.0)
}

pub fn positive_product<const N: usize>(factors: [f64; N]) -> f64 {
    if factors.contains(&0.0) {
        return 0.0;
    }
    let mut significand = 1.0;
    let mut exponent = 0_i32;
    for factor in factors {
        let (factor_significand, factor_exponent) = positive_binary_decomposition(factor);
        significand *= factor_significand;
        exponent += factor_exponent;
        if significand >= 2.0 {
            significand *= 0.5;
            exponent += 1;
        }
    }
    if exponent < -1075 {
        return 0.0;
    }
    if exponent == -1075 {
        return (significand * 0.5) * f64::from_bits(1);
    }
    significand * binary_power(exponent)
}

fn positive_binary_decomposition(value: f64) -> (f64, i32) {
    let (significand, exponent) = binary64_magnitude(value);
    let Ok(highest_bit) = i32::try_from(significand.ilog2()) else {
        return (f64::NAN, 0);
    };
    let normalized_exponent = exponent + highest_bit;
    (
        value / binary_power(normalized_exponent),
        normalized_exponent,
    )
}

pub fn binary64_magnitude(value: f64) -> (u64, i32) {
    let bits = value.to_bits();
    let stored_exponent = (bits >> 52) & 0x7ff;
    let fraction = bits & ((1_u64 << 52) - 1);
    if stored_exponent == 0 {
        (fraction, -1074)
    } else {
        (
            (1_u64 << 52) | fraction,
            i32::try_from(stored_exponent).unwrap_or_default() - 1075,
        )
    }
}

pub fn binary_power(exponent: i32) -> f64 {
    if exponent < -1074 {
        return 0.0;
    }
    if exponent < -1022 {
        let Ok(fraction_bit) = u32::try_from(exponent + 1074) else {
            return 0.0;
        };
        return f64::from_bits(1_u64 << fraction_bit);
    }
    let Ok(stored_exponent) = u64::try_from(exponent + 1023) else {
        return f64::INFINITY;
    };
    if stored_exponent >= 0x7ff {
        f64::INFINITY
    } else {
        f64::from_bits(stored_exponent << 52)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{binary_power, binary64_magnitude};

    fn positive_finite_binary64() -> impl Strategy<Value = f64> {
        (1_u64..=0x7fef_ffff_ffff_ffff).prop_map(f64::from_bits)
    }

    fn binary64_significand_as_f64(significand: u64) -> f64 {
        let upper = u32::try_from(significand >> 32)
            .expect("a binary64 significand contains at most 53 bits");
        let lower = u32::try_from(significand & u64::from(u32::MAX))
            .expect("the lower half contains exactly 32 bits");
        f64::from(upper).mul_add(4_294_967_296.0, f64::from(lower))
    }

    proptest! {
        #[test]
        fn binary_magnitude_round_trips_every_positive_finite_value(
            value in positive_finite_binary64(),
        ) {
            let (significand, exponent) = binary64_magnitude(value);
            let reconstructed = binary64_significand_as_f64(significand) * binary_power(exponent);

            prop_assert_eq!(reconstructed.to_bits(), value.to_bits());
        }
    }
}

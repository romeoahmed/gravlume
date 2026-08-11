use crate::{
    ValidationIssue, ValidationIssueCode, ValidationReport, validation::validate_finite_array,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpacetimeEvent {
    txyz: [f64; 4],
}

impl SpacetimeEvent {
    /// Creates an event in canonical `(t, x, y, z)` component order.
    ///
    /// # Errors
    ///
    /// Returns a validation report when any component is non-finite.
    pub fn from_txyz(txyz: [f64; 4]) -> Result<Self, ValidationReport> {
        let mut report = ValidationReport::default();
        validate_finite_array(&mut report, txyz, "event.txyz");
        report.into_result(Self { txyz })
    }

    pub(super) const fn from_validated(txyz: [f64; 4]) -> Self {
        Self { txyz }
    }

    #[must_use]
    pub const fn to_txyz(self) -> [f64; 4] {
        self.txyz
    }

    pub(super) const fn x(self) -> f64 {
        self.txyz[1]
    }

    pub(super) const fn y(self) -> f64 {
        self.txyz[2]
    }

    pub(super) const fn z(self) -> f64 {
        self.txyz[3]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FourVector([f64; 4]);

impl FourVector {
    pub(super) const fn new(txyz: [f64; 4]) -> Self {
        Self(txyz)
    }

    pub(super) const fn to_array(self) -> [f64; 4] {
        self.0
    }

    pub(super) fn add(self, other: Self) -> Self {
        Self(std::array::from_fn(|index| self.0[index] + other.0[index]))
    }

    pub(super) fn subtract(self, other: Self) -> Self {
        Self(std::array::from_fn(|index| self.0[index] - other.0[index]))
    }

    pub(super) fn scaled(self, scalar: f64) -> Self {
        Self(self.0.map(|component| component * scalar))
    }

    pub(super) fn is_finite(self) -> bool {
        self.0.into_iter().all(f64::is_finite)
    }
}

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
    if exponent >= 0 {
        return 1.0;
    }
    if exponent < -1075 {
        return 0.0;
    }
    if exponent == -1075 {
        return (significand * 0.5) * f64::from_bits(1);
    }
    let power_of_two = if exponent >= -1022 {
        let Ok(stored_exponent) = u64::try_from(exponent + 1023) else {
            return 0.0;
        };
        f64::from_bits(stored_exponent << 52)
    } else {
        let Ok(fraction_bit) = u32::try_from(exponent + 1074) else {
            return 0.0;
        };
        f64::from_bits(1_u64 << fraction_bit)
    };
    significand * power_of_two
}

fn positive_binary_decomposition(value: f64) -> (f64, i32) {
    let bits = value.to_bits();
    let stored_exponent = (bits >> 52) & 0x7ff;
    let fraction = bits & ((1_u64 << 52) - 1);
    if stored_exponent == 0 {
        let highest_fraction_bit = fraction.ilog2();
        let power_of_two = f64::from_bits(1_u64 << highest_fraction_bit);
        let factor_exponent = match i32::try_from(highest_fraction_bit) {
            Ok(bit) => bit - 1074,
            Err(_) => return (f64::NAN, 0),
        };
        (value / power_of_two, factor_exponent)
    } else {
        let factor_exponent = match i32::try_from(stored_exponent) {
            Ok(stored) => stored - 1023,
            Err(_) => return (f64::NAN, 0),
        };
        (f64::from_bits((1023_u64 << 52) | fraction), factor_exponent)
    }
}

fn binary_power(exponent: i32) -> f64 {
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodesicState {
    components: [f64; 8],
}

impl GeodesicState {
    /// Creates a canonical Hamilton state `(t, x, y, z, p_t, p_x, p_y, p_z)`.
    ///
    /// # Errors
    ///
    /// Returns a validation report when any component is non-finite.
    pub fn new(
        position_txyz: [f64; 4],
        momentum_covariant_txyz: [f64; 4],
    ) -> Result<Self, ValidationReport> {
        let mut report = ValidationReport::default();
        validate_finite_array(&mut report, position_txyz, "geodesic_state.position_txyz");
        validate_finite_array(
            &mut report,
            momentum_covariant_txyz,
            "geodesic_state.momentum_covariant_txyz",
        );
        let components = std::array::from_fn(|index| {
            if index < 4 {
                position_txyz[index]
            } else {
                momentum_covariant_txyz[index - 4]
            }
        });
        report.into_result(Self { components })
    }

    /// Reconstructs a state produced by a numerical integrator.
    ///
    /// # Errors
    ///
    /// Returns a validation report when the numerical state is non-finite.
    pub fn from_components(components: [f64; 8]) -> Result<Self, ValidationReport> {
        if components.into_iter().all(f64::is_finite) {
            Ok(Self { components })
        } else {
            Err(ValidationReport::from_issue(ValidationIssue::error(
                ValidationIssueCode::NonFinite,
                "geodesic_state",
                "every numerical state component must be finite",
            )))
        }
    }

    pub(super) const fn from_validated(components: [f64; 8]) -> Self {
        Self { components }
    }

    #[must_use]
    pub const fn components(self) -> [f64; 8] {
        self.components
    }

    #[must_use]
    pub const fn event(self) -> SpacetimeEvent {
        SpacetimeEvent::from_validated([
            self.components[0],
            self.components[1],
            self.components[2],
            self.components[3],
        ])
    }

    #[must_use]
    pub const fn momentum_covariant_txyz(self) -> [f64; 4] {
        [
            self.components[4],
            self.components[5],
            self.components[6],
            self.components[7],
        ]
    }
}

impl ValidationReport {
    fn from_issue(issue: ValidationIssue) -> Self {
        let mut report = Self::default();
        report.push(issue);
        report
    }
}

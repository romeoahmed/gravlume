use std::cmp::Ordering;

use glam::DVec3;

use crate::{
    GeodesicState, SpacetimeEvent, ValidationIssue, ValidationIssueCode, ValidationReport,
    math::{
        FourVector, binary_power, binary64_magnitude, normalized_inner_product_residual,
        normalized_quadratic_form_residual, positive_product,
    },
    validation::validate_finite,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterState {
    Subextremal,
    Extremal,
    Superextremal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KerrSchildCoordinates {
    Ingoing,
    Outgoing,
}

impl KerrSchildCoordinates {
    const fn principal_direction(self) -> f64 {
        match self {
            Self::Ingoing => 1.0,
            Self::Outgoing => -1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodesicInvariants {
    energy: f64,
    angular_momentum_z: f64,
    carter_constant: f64,
    normalized_null_residual: f64,
}

impl GeodesicInvariants {
    #[must_use]
    pub const fn energy(self) -> f64 {
        self.energy
    }

    #[must_use]
    pub const fn angular_momentum_z(self) -> f64 {
        self.angular_momentum_z
    }

    #[must_use]
    pub const fn carter_constant(self) -> f64 {
        self.carter_constant
    }

    #[must_use]
    pub const fn normalized_null_residual(self) -> f64 {
        self.normalized_null_residual
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GeometryError {
    #[error("geometry input contains a non-finite component")]
    NonFinite,
    #[error("the Kerr-Schild radius is undefined at the ring singularity")]
    RingSingularity,
    #[error("the non-negative Kerr-Schild chart is undefined on its r = 0 branch disk")]
    ChartBoundary,
    #[error("a Kerr-Schild denominator is non-positive or non-finite")]
    InvalidDenominator,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KerrNewmanSpacetime {
    mass_m: f64,
    spin_m: f64,
    charge_m: f64,
    coordinates: KerrSchildCoordinates,
    parameter_state: ParameterState,
}

#[derive(Clone, Copy)]
struct Geometry {
    radius: f64,
    radius_gradient: DVec3,
    scalar_f: f64,
    scalar_f_gradient: DVec3,
    null_covector: [f64; 4],
    null_vector: [f64; 4],
    null_vector_gradient: [[f64; 4]; 3],
}

#[derive(Clone, Copy)]
struct ScaledCoordinates {
    scale: f64,
    x: f64,
    y: f64,
    z: f64,
    spin: f64,
    radius_squared: f64,
    radius: f64,
    sigma: f64,
}

impl ScaledCoordinates {
    fn singularity_measure(self) -> f64 {
        let spin_squared = self.spin * self.spin;
        positive_product([
            self.scale,
            self.scale,
            self.scale,
            self.scale,
            self.radius_squared
                .mul_add(self.radius_squared, spin_squared * self.z * self.z),
        ])
    }
}

impl KerrNewmanSpacetime {
    /// Creates validated Kerr-Newman parameters in geometric units.
    ///
    /// # Errors
    ///
    /// Returns every finite/positive boundary violation as a structured validation report.
    pub fn new(
        mass_m: f64,
        spin_m: f64,
        charge_m: f64,
        coordinates: KerrSchildCoordinates,
    ) -> Result<Self, ValidationReport> {
        Self::validated_with_prefix(mass_m, spin_m, charge_m, coordinates, "spacetime")
    }

    pub(super) fn validated_with_prefix(
        mass_m: f64,
        spin_m: f64,
        charge_m: f64,
        coordinates: KerrSchildCoordinates,
        prefix: &str,
    ) -> Result<Self, ValidationReport> {
        let mut report = ValidationReport::default();
        let mass_path = format!("{prefix}.mass_m");
        let spin_path = format!("{prefix}.spin_m");
        let charge_path = format!("{prefix}.charge_m");
        if !mass_m.is_finite() {
            report.push(ValidationIssue::error(
                ValidationIssueCode::NonFinite,
                mass_path,
                "mass scale must be finite",
            ));
        } else if mass_m <= 0.0 {
            report.push(ValidationIssue::error(
                ValidationIssueCode::NonPositive,
                mass_path,
                "mass scale must be positive",
            ));
        }
        validate_finite(&mut report, spin_m, spin_path);
        validate_finite(&mut report, charge_m, charge_path);

        if !report.is_empty() {
            return Err(report);
        }

        let mass_squared = ExactBinary::square(mass_m);
        let angular_squared = ExactBinary::sum_of_squares(spin_m, charge_m);
        let parameter_state = match mass_squared.cmp(&angular_squared) {
            Ordering::Greater => ParameterState::Subextremal,
            Ordering::Equal => ParameterState::Extremal,
            Ordering::Less => ParameterState::Superextremal,
        };
        let spacetime = Self {
            mass_m,
            spin_m,
            charge_m,
            coordinates,
            parameter_state,
        };
        if spacetime
            .outer_horizon_radius()
            .is_some_and(|radius| !radius.is_finite())
        {
            report.push(ValidationIssue::error(
                ValidationIssueCode::NonFinite,
                format!("{prefix}.outer_horizon_radius_m"),
                "outer horizon radius must be representable",
            ));
        }
        report.into_result(spacetime)
    }

    #[must_use]
    pub const fn mass_m(self) -> f64 {
        self.mass_m
    }

    #[must_use]
    pub const fn spin_m(self) -> f64 {
        self.spin_m
    }

    #[must_use]
    pub const fn charge_m(self) -> f64 {
        self.charge_m
    }

    #[must_use]
    pub const fn coordinates(self) -> KerrSchildCoordinates {
        self.coordinates
    }

    #[must_use]
    pub const fn parameter_state(self) -> ParameterState {
        self.parameter_state
    }

    #[must_use]
    pub fn outer_horizon_radius(self) -> Option<f64> {
        match self.parameter_state {
            ParameterState::Superextremal => None,
            ParameterState::Extremal => Some(self.mass_m),
            ParameterState::Subextremal => {
                let mass_squared = ExactBinary::square(self.mass_m);
                let angular_squared = ExactBinary::sum_of_squares(self.spin_m, self.charge_m);
                let discriminant_root = mass_squared.subtract(&angular_squared).square_root();
                Some(self.mass_m + discriminant_root)
            }
        }
    }

    /// Converts oblate coordinates to this spacetime's Cartesian Kerr-Schild coordinates.
    #[must_use]
    pub fn oblate_to_cartesian(self, radius: f64, polar: f64, azimuth: f64) -> [f64; 3] {
        let (sin_theta, cos_theta) = polar.sin_cos();
        let (sin_phi, cos_phi) = azimuth.sin_cos();
        [
            self.spin_m.mul_add(-sin_phi, radius * cos_phi) * sin_theta,
            self.spin_m.mul_add(cos_phi, radius * sin_phi) * sin_theta,
            radius * cos_theta,
        ]
    }

    /// Evaluates the non-negative oblate Kerr-Schild radius.
    ///
    /// # Errors
    ///
    /// Returns a typed numerical error at non-finite input or the ring singularity.
    pub fn radius(self, event: SpacetimeEvent) -> Result<f64, GeometryError> {
        self.geometry(event).map(|geometry| geometry.radius)
    }

    /// Returns `g_tt` in the canonical mostly-plus convention.
    ///
    /// # Errors
    ///
    /// Returns a typed numerical error when the metric is undefined.
    pub fn metric_component_tt(self, event: SpacetimeEvent) -> Result<f64, GeometryError> {
        self.geometry(event)
            .map(|geometry| -1.0 + geometry.scalar_f)
    }

    /// Measures the term-normalized maximum residual of `g * g^-1 = I`.
    ///
    /// # Errors
    ///
    /// Returns a typed numerical error when the metric is undefined.
    pub fn metric_inverse_residual(self, event: SpacetimeEvent) -> Result<f64, GeometryError> {
        let geometry = self.geometry(event)?;
        let covariant = metric_covariant(geometry);
        let inverse = metric_inverse(geometry);
        let mut residual = 0.0_f64;
        for (row, covariant_row) in covariant.iter().enumerate() {
            for (column, _) in inverse[0].iter().enumerate() {
                let expected = if row == column { 1.0 } else { 0.0 };
                let inverse_column = std::array::from_fn(|index| inverse[index][column]);
                let normalized =
                    normalized_inner_product_residual(*covariant_row, inverse_column, expected)
                        .ok_or(GeometryError::NonFinite)?;
                if normalized > residual {
                    residual = normalized;
                }
            }
        }
        Ok(residual)
    }

    /// Evaluates the canonical Hamilton right-hand side.
    ///
    /// # Errors
    ///
    /// Returns a typed numerical error before evaluating an invalid denominator.
    pub fn hamiltonian_rhs(self, state: GeodesicState) -> Result<[f64; 8], GeometryError> {
        let event = state.event();
        let geometry = self.geometry(event)?;
        let momentum = state.momentum_covariant_txyz();
        let contraction = dot_components(geometry.null_vector, momentum);
        let minkowski_raised = [-momentum[0], momentum[1], momentum[2], momentum[3]];
        let velocity: [f64; 4] = std::array::from_fn(|index| {
            (geometry.scalar_f * contraction)
                .mul_add(-geometry.null_vector[index], minkowski_raised[index])
        });
        let momentum_derivative: [f64; 3] = std::array::from_fn(|spatial_index| {
            let null_derivative_contraction =
                dot_components(momentum, geometry.null_vector_gradient[spatial_index]);
            (geometry.scalar_f * contraction).mul_add(
                null_derivative_contraction,
                0.5 * contraction * contraction * geometry.scalar_f_gradient[spatial_index],
            )
        });
        let derivative = [
            velocity[0],
            velocity[1],
            velocity[2],
            velocity[3],
            0.0,
            momentum_derivative[0],
            momentum_derivative[1],
            momentum_derivative[2],
        ];
        if derivative.into_iter().all(f64::is_finite) {
            Ok(derivative)
        } else {
            Err(GeometryError::NonFinite)
        }
    }

    /// Computes conserved quantities and the normalized null residual.
    ///
    /// # Errors
    ///
    /// Returns a typed numerical error when the metric is undefined.
    pub fn invariants(self, state: GeodesicState) -> Result<GeodesicInvariants, GeometryError> {
        let event = state.event();
        let momentum = state.momentum_covariant_txyz();
        let radius = self.radius(event)?;
        let energy = -momentum[0];
        let angular_momentum_z = event.y().mul_add(-momentum[1], event.x() * momentum[2]);
        let rho = event.x().hypot(event.y());
        let spin_squared = self.spin_m * self.spin_m;
        let carter_constant = if rho == 0.0 {
            let transverse_momentum_squared =
                momentum[1].mul_add(momentum[1], momentum[2] * momentum[2]);
            let leading = (radius.mul_add(radius, spin_squared)) * transverse_momentum_squared;
            (-spin_squared * energy).mul_add(energy, leading)
        } else {
            let cos_theta = event.z() / radius;
            let sin_theta = rho / radius.mul_add(radius, spin_squared).sqrt();
            let projected_momentum = event.x().mul_add(momentum[1], event.y() * momentum[2]);
            let p_theta = (-radius * sin_theta)
                .mul_add(momentum[3], (cos_theta / sin_theta) * projected_momentum);
            let lz_over_sin = angular_momentum_z / sin_theta;
            let angular_term = (-spin_squared * energy).mul_add(energy, lz_over_sin * lz_over_sin);
            (cos_theta * cos_theta).mul_add(angular_term, p_theta * p_theta)
        };
        let normalized_null_residual = self.normalized_null_residual(state)?;
        if ![
            energy,
            angular_momentum_z,
            carter_constant,
            normalized_null_residual,
        ]
        .into_iter()
        .all(f64::is_finite)
        {
            return Err(GeometryError::NonFinite);
        }
        Ok(GeodesicInvariants {
            energy,
            angular_momentum_z,
            carter_constant,
            normalized_null_residual,
        })
    }

    /// Returns `|g^mn p_m p_n|` normalized by the term one-norm.
    ///
    /// # Errors
    ///
    /// Returns a typed numerical error when the metric is undefined.
    pub fn normalized_null_residual(self, state: GeodesicState) -> Result<f64, GeometryError> {
        let inverse = metric_inverse(self.geometry(state.event())?);
        let momentum = FourVector::new(state.momentum_covariant_txyz());
        let residual = normalized_quadratic_form_residual(inverse, momentum);
        residual
            .is_finite()
            .then_some(residual)
            .ok_or(GeometryError::NonFinite)
    }

    /// Returns `dr/dlambda` for the canonical Hamilton traversal direction.
    ///
    /// # Errors
    ///
    /// Returns a typed numerical error when the metric is undefined.
    pub fn radial_velocity(self, state: GeodesicState) -> Result<f64, GeometryError> {
        let geometry = self.geometry(state.event())?;
        let derivative = self.hamiltonian_rhs(state)?;
        Ok(geometry
            .radius_gradient
            .dot(DVec3::new(derivative[1], derivative[2], derivative[3])))
    }

    /// Returns `r^4 + a^2 z^2`, the canonical singularity guard observable.
    ///
    /// # Errors
    ///
    /// Returns a typed numerical error when the metric is undefined.
    pub fn singularity_measure(self, event: SpacetimeEvent) -> Result<f64, GeometryError> {
        let measure = self.scaled_coordinates(event)?.singularity_measure();
        measure
            .is_finite()
            .then_some(measure)
            .ok_or(GeometryError::NonFinite)
    }

    /// Returns the finite signed residual of the canonical singularity guard observable.
    ///
    /// Values above the representable `f64` range saturate positively. This preserves the event
    /// side without weakening exact evaluation near the finite guard surface.
    ///
    /// # Errors
    ///
    /// Returns a typed numerical error for an invalid guard or undefined geometry.
    pub fn singularity_guard_residual(
        self,
        event: SpacetimeEvent,
        guard: f64,
    ) -> Result<f64, GeometryError> {
        if !guard.is_finite() || guard <= 0.0 {
            return Err(GeometryError::InvalidDenominator);
        }
        let measure = self.scaled_coordinates(event)?.singularity_measure();
        if measure.is_infinite() {
            Ok(f64::MAX)
        } else {
            Ok(measure - guard)
        }
    }

    pub(super) fn metric_dot(
        self,
        event: SpacetimeEvent,
        left: FourVector,
        right: FourVector,
    ) -> Result<f64, GeometryError> {
        self.metric_dot_and_term_norm(event, left, right)
            .map(|(contraction, _)| contraction)
    }

    pub(super) fn metric_dot_and_term_norm(
        self,
        event: SpacetimeEvent,
        left: FourVector,
        right: FourVector,
    ) -> Result<(f64, f64), GeometryError> {
        let metric = metric_covariant(self.geometry(event)?);
        let left = left.to_array();
        let right = right.to_array();
        let result = (0..4)
            .flat_map(|row| (0..4).map(move |column| (row, column)))
            .map(|(row, column)| metric[row][column] * left[row] * right[column])
            .fold((0.0, 0.0_f64), |(sum, norm), term| {
                (sum + term, norm + term.abs())
            });
        if result.0.is_finite() && result.1.is_finite() {
            Ok(result)
        } else {
            Err(GeometryError::NonFinite)
        }
    }

    pub(super) fn metric_covariant_at(
        self,
        event: SpacetimeEvent,
    ) -> Result<[[f64; 4]; 4], GeometryError> {
        self.geometry(event).map(metric_covariant)
    }

    fn geometry(self, event: SpacetimeEvent) -> Result<Geometry, GeometryError> {
        let scaled = self.scaled_coordinates(event)?;
        if scaled.x == 0.0 && scaled.y == 0.0 {
            return self.axis_geometry(scaled);
        }
        let ScaledCoordinates {
            scale,
            x,
            y,
            z,
            spin,
            radius_squared,
            radius: scaled_radius,
            sigma,
        } = scaled;
        let radius = scale * scaled_radius;
        if !radius.is_finite() {
            return Err(GeometryError::InvalidDenominator);
        }
        let spin_squared = spin * spin;
        let radius_gradient = DVec3::new(
            x * scaled_radius / sigma,
            y * scaled_radius / sigma,
            z * (radius_squared + spin_squared) / (scaled_radius * sigma),
        );
        let mass = self.mass_m / scale;
        let charge = self.charge_m / scale;
        let numerator = (-charge).mul_add(charge, 2.0 * mass * scaled_radius);
        let scalar_f = numerator / sigma;
        let vertical_sigma = spin_squared * z * z / radius_squared;
        let sigma_radius_factor = 2.0 * (scaled_radius - vertical_sigma / scaled_radius);
        let sigma_gradient = DVec3::new(
            sigma_radius_factor * radius_gradient.x,
            sigma_radius_factor * radius_gradient.y,
            sigma_radius_factor.mul_add(radius_gradient.z, 2.0 * spin_squared * z / radius_squared),
        );
        let numerator_gradient = radius_gradient * (2.0 * mass);
        let scalar_f_gradient =
            (numerator_gradient * sigma - sigma_gradient * numerator) / (sigma * sigma * scale);
        let radial_denominator = radius_squared + spin_squared;
        if radial_denominator <= 0.0 || !radial_denominator.is_finite() {
            return Err(GeometryError::InvalidDenominator);
        }
        let null_spatial = self.coordinates.principal_direction()
            * DVec3::new(
                spin.mul_add(y, scaled_radius * x) / radial_denominator,
                spin.mul_add(-x, scaled_radius * y) / radial_denominator,
                z / scaled_radius,
            );
        let null_covector = [1.0, null_spatial.x, null_spatial.y, null_spatial.z];
        let null_vector = [-1.0, null_spatial.x, null_spatial.y, null_spatial.z];
        let coordinates = [x, y, z];
        let radius_gradients = radius_gradient.to_array();
        let direction = self.coordinates.principal_direction();
        let null_vector_gradient = std::array::from_fn(|index| {
            let radius_i = radius_gradients[index];
            let delta_x = f64::from(index == 0);
            let delta_y = f64::from(index == 1);
            let delta_z = f64::from(index == 2);
            let numerator_x = spin.mul_add(delta_y, radius_i.mul_add(x, scaled_radius * delta_x));
            let numerator_y = spin.mul_add(-delta_x, radius_i.mul_add(y, scaled_radius * delta_y));
            let radial_derivative = 2.0 * scaled_radius * radius_i;
            let derivative_x = ((-direction * null_spatial.x * radial_derivative)
                .mul_add(radial_denominator.recip(), numerator_x / radial_denominator))
                / scale;
            let derivative_y = ((-direction * null_spatial.y * radial_derivative)
                .mul_add(radial_denominator.recip(), numerator_y / radial_denominator))
                / scale;
            let derivative_z =
                (delta_z / scaled_radius - coordinates[2] * radius_i / radius_squared) / scale;
            [
                0.0,
                direction * derivative_x,
                direction * derivative_y,
                direction * derivative_z,
            ]
        });
        let geometry = Geometry {
            radius,
            radius_gradient,
            scalar_f,
            scalar_f_gradient,
            null_covector,
            null_vector,
            null_vector_gradient,
        };
        validate_geometry(geometry)
    }

    fn axis_geometry(self, scaled: ScaledCoordinates) -> Result<Geometry, GeometryError> {
        let ScaledCoordinates {
            scale,
            z,
            spin,
            radius: scaled_radius,
            sigma,
            ..
        } = scaled;
        let radius = scale * scaled_radius;
        if !radius.is_finite() || scaled_radius <= 0.0 || sigma <= 0.0 {
            return Err(GeometryError::InvalidDenominator);
        }
        let axis_sign = z.signum();
        let radius_gradient = DVec3::new(0.0, 0.0, axis_sign);
        let mass = self.mass_m / scale;
        let charge = self.charge_m / scale;
        let numerator = (-charge).mul_add(charge, 2.0 * mass * scaled_radius);
        let scalar_f = numerator / sigma;
        let sigma_gradient = DVec3::new(0.0, 0.0, 2.0 * scaled_radius * axis_sign);
        let numerator_gradient = radius_gradient * (2.0 * mass);
        let scalar_f_gradient =
            (numerator_gradient * sigma - sigma_gradient * numerator) / (sigma * sigma * scale);
        let direction = self.coordinates.principal_direction();
        let null_covector = [1.0, 0.0, 0.0, direction * axis_sign];
        let null_vector = [-1.0, 0.0, 0.0, direction * axis_sign];
        let inverse_denominator_scale = sigma.recip() / scale;
        let null_vector_gradient = [
            [
                0.0,
                direction * scaled_radius * inverse_denominator_scale,
                -direction * spin * inverse_denominator_scale,
                0.0,
            ],
            [
                0.0,
                direction * spin * inverse_denominator_scale,
                direction * scaled_radius * inverse_denominator_scale,
                0.0,
            ],
            [0.0; 4],
        ];
        validate_geometry(Geometry {
            radius,
            radius_gradient,
            scalar_f,
            scalar_f_gradient,
            null_covector,
            null_vector,
            null_vector_gradient,
        })
    }

    fn scaled_coordinates(self, event: SpacetimeEvent) -> Result<ScaledCoordinates, GeometryError> {
        let [_, physical_x, physical_y, physical_z] = event.to_txyz();
        if ![physical_x, physical_y, physical_z]
            .into_iter()
            .all(f64::is_finite)
        {
            return Err(GeometryError::NonFinite);
        }
        let scale = physical_x
            .abs()
            .max(physical_y.abs())
            .max(physical_z.abs())
            .max(self.spin_m.abs());
        if scale == 0.0 {
            return Err(GeometryError::RingSingularity);
        }
        let x = physical_x / scale;
        let y = physical_y / scale;
        let z = physical_z / scale;
        let spin = self.spin_m / scale;
        let spin_squared = spin * spin;
        if physical_x == 0.0 && physical_y == 0.0 && physical_z != 0.0 {
            let radius = z.abs();
            let radius_squared = radius * radius;
            let sigma = radius_squared + spin_squared;
            if radius > 0.0 && sigma.is_finite() && sigma > 0.0 {
                return Ok(ScaledCoordinates {
                    scale,
                    x,
                    y,
                    z,
                    spin,
                    radius_squared,
                    radius,
                    sigma,
                });
            }
        }
        let radius_squared_3d = x.mul_add(x, y.mul_add(y, z * z));
        let b = radius_squared_3d - spin_squared;
        let discriminant = b.mul_add(b, 4.0 * spin_squared * z * z);
        if !discriminant.is_finite() || discriminant < 0.0 {
            return Err(GeometryError::InvalidDenominator);
        }
        let root = discriminant.sqrt();
        let radius_squared = if b >= 0.0 {
            0.5 * (b + root)
        } else {
            let denominator = root - b;
            if denominator <= 0.0 || !denominator.is_finite() {
                return Err(GeometryError::InvalidDenominator);
            }
            2.0 * spin_squared * z * z / denominator
        };
        if !radius_squared.is_finite() {
            return Err(GeometryError::InvalidDenominator);
        }
        if radius_squared <= 0.0 {
            return Err(classify_zero_radius(b));
        }
        let radius = radius_squared.sqrt();
        let sigma = radius_squared + spin_squared * z * z / radius_squared;
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(GeometryError::InvalidDenominator);
        }
        Ok(ScaledCoordinates {
            scale,
            x,
            y,
            z,
            spin,
            radius_squared,
            radius,
            sigma,
        })
    }
}

// A finite binary64 magnitude is `significand * 2^exponent` with a 53-bit
// significand and exponent in `[-1074, 971]`. Sixty-six limbs therefore hold
// the exact sum of any two squared magnitudes, including its carry bit.
const EXACT_BINARY_LIMBS: usize = 66;
const MINIMUM_SQUARED_EXPONENT: i32 = -2148;
const BINARY_WORD_BITS: usize = 64;

#[derive(Clone, Eq, PartialEq)]
struct ExactBinary {
    limbs: [u64; EXACT_BINARY_LIMBS],
}

impl ExactBinary {
    fn square(value: f64) -> Self {
        let mut result = Self::default();
        result.add_square(value);
        result
    }

    fn sum_of_squares(first: f64, second: f64) -> Self {
        let mut result = Self::default();
        result.add_square(first);
        result.add_square(second);
        result
    }

    fn add_square(&mut self, value: f64) {
        let (significand, exponent) = binary64_magnitude(value.abs());
        if significand == 0 {
            return;
        }
        let coefficient = u128::from(significand) * u128::from(significand);
        let shift = usize::try_from(2 * exponent - MINIMUM_SQUARED_EXPONENT).unwrap_or_default();
        self.add_shifted(coefficient, shift);
    }

    fn add_shifted(&mut self, coefficient: u128, shift: usize) {
        let word = shift / BINARY_WORD_BITS;
        let bit = shift % BINARY_WORD_BITS;
        let low = u64::try_from(coefficient & u128::from(u64::MAX)).unwrap_or_default();
        let high = u64::try_from(coefficient >> BINARY_WORD_BITS).unwrap_or_default();
        if bit == 0 {
            self.add_word(word, low);
            self.add_word(word + 1, high);
        } else {
            self.add_word(word, low << bit);
            self.add_word(word + 1, low >> (BINARY_WORD_BITS - bit));
            self.add_word(word + 1, high << bit);
            self.add_word(word + 2, high >> (BINARY_WORD_BITS - bit));
        }
    }

    fn add_word(&mut self, mut index: usize, mut value: u64) {
        while value != 0 {
            let (sum, carry) = self.limbs[index].overflowing_add(value);
            self.limbs[index] = sum;
            value = u64::from(carry);
            index += 1;
        }
    }

    fn subtract(&self, smaller: &Self) -> Self {
        let mut result = Self::default();
        let mut borrow = false;
        for index in 0..EXACT_BINARY_LIMBS {
            let (difference, first_borrow) =
                self.limbs[index].overflowing_sub(smaller.limbs[index]);
            let (difference, second_borrow) = difference.overflowing_sub(u64::from(borrow));
            result.limbs[index] = difference;
            borrow = first_borrow || second_borrow;
        }
        debug_assert!(!borrow);
        result
    }

    fn square_root(&self) -> f64 {
        let highest_bit = self.highest_bit().unwrap_or_default();
        let mut significand = 0_u64;
        for offset in 0..f64::MANTISSA_DIGITS as usize {
            significand <<= 1;
            if highest_bit >= offset && self.bit(highest_bit - offset) {
                significand |= 1;
            }
        }
        let discarded_bits = highest_bit.saturating_sub(f64::MANTISSA_DIGITS as usize - 1);
        if discarded_bits > 0 {
            let guard = self.bit(discarded_bits - 1);
            let sticky = (0..discarded_bits - 1).any(|bit| self.bit(bit));
            if guard && (sticky || significand & 1 == 1) {
                significand += 1;
            }
        }
        let high = u32::try_from(significand >> 32).unwrap_or_default();
        let low = u32::try_from(significand & u64::from(u32::MAX)).unwrap_or_default();
        let normalized =
            f64::from(high).mul_add(2.0_f64.powi(32), f64::from(low)) / 2.0_f64.powi(52);
        let exponent = MINIMUM_SQUARED_EXPONENT + i32::try_from(highest_bit).unwrap_or_default();
        if exponent & 1 == 0 {
            normalized.sqrt() * binary_power(exponent / 2)
        } else {
            (2.0 * normalized).sqrt() * binary_power((exponent - 1) / 2)
        }
    }

    fn highest_bit(&self) -> Option<usize> {
        self.limbs.iter().rposition(|limb| *limb != 0).map(|word| {
            word * BINARY_WORD_BITS
                + usize::try_from(u64::BITS - 1 - self.limbs[word].leading_zeros())
                    .unwrap_or_default()
        })
    }

    const fn bit(&self, index: usize) -> bool {
        let word = index / BINARY_WORD_BITS;
        let bit = index % BINARY_WORD_BITS;
        self.limbs[word] & (1_u64 << bit) != 0
    }
}

impl Default for ExactBinary {
    fn default() -> Self {
        Self {
            limbs: [0; EXACT_BINARY_LIMBS],
        }
    }
}

impl PartialOrd for ExactBinary {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ExactBinary {
    fn cmp(&self, other: &Self) -> Ordering {
        self.limbs.iter().rev().cmp(other.limbs.iter().rev())
    }
}

const fn classify_zero_radius(b: f64) -> GeometryError {
    if b == 0.0 {
        GeometryError::RingSingularity
    } else {
        GeometryError::ChartBoundary
    }
}

fn validate_geometry(geometry: Geometry) -> Result<Geometry, GeometryError> {
    let all_finite = geometry.radius_gradient.is_finite()
        && geometry.scalar_f.is_finite()
        && geometry.scalar_f_gradient.is_finite()
        && geometry.null_covector.into_iter().all(f64::is_finite)
        && geometry
            .null_vector_gradient
            .into_iter()
            .flatten()
            .all(f64::is_finite);
    all_finite
        .then_some(geometry)
        .ok_or(GeometryError::NonFinite)
}

fn metric_covariant(geometry: Geometry) -> [[f64; 4]; 4] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            let minkowski = if row == column {
                if row == 0 { -1.0 } else { 1.0 }
            } else {
                0.0
            };
            (geometry.scalar_f * geometry.null_covector[row])
                .mul_add(geometry.null_covector[column], minkowski)
        })
    })
}

fn metric_inverse(geometry: Geometry) -> [[f64; 4]; 4] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            let minkowski = if row == column {
                if row == 0 { -1.0 } else { 1.0 }
            } else {
                0.0
            };
            (-geometry.scalar_f * geometry.null_vector[row])
                .mul_add(geometry.null_vector[column], minkowski)
        })
    })
}

fn dot_components(left: [f64; 4], right: [f64; 4]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

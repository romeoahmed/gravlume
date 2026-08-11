use glam::DVec3;

use crate::{
    GeodesicState, SpacetimeEvent, ValidationIssue, ValidationIssueCode, ValidationReport,
    math::{FourVector, normalized_quadratic_form_residual},
    validation::validate_finite,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterState {
    Subextremal,
    Extremal,
    Superextremal,
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
    singularity_measure: f64,
}

impl KerrNewmanSpacetime {
    /// Creates validated Kerr-Newman parameters in geometric units.
    ///
    /// # Errors
    ///
    /// Returns every finite/positive boundary violation as a structured validation report.
    pub fn new(mass_m: f64, spin_m: f64, charge_m: f64) -> Result<Self, ValidationReport> {
        Self::validated_with_prefix(mass_m, spin_m, charge_m, "spacetime")
    }

    pub(super) fn validated_with_prefix(
        mass_m: f64,
        spin_m: f64,
        charge_m: f64,
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

        let extremality = mass_m.mul_add(mass_m, -spin_m.mul_add(spin_m, charge_m * charge_m));
        let parameter_state = if extremality > 0.0 {
            ParameterState::Subextremal
        } else if extremality == 0.0 {
            ParameterState::Extremal
        } else {
            ParameterState::Superextremal
        };
        Ok(Self {
            mass_m,
            spin_m,
            charge_m,
            parameter_state,
        })
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
    pub const fn parameter_state(self) -> ParameterState {
        self.parameter_state
    }

    #[must_use]
    pub fn outer_horizon_radius(self) -> Option<f64> {
        let discriminant = self.mass_m.mul_add(
            self.mass_m,
            -self
                .spin_m
                .mul_add(self.spin_m, self.charge_m * self.charge_m),
        );
        (discriminant >= 0.0).then(|| self.mass_m + discriminant.sqrt())
    }

    /// Converts ingoing oblate coordinates to canonical Cartesian spatial coordinates.
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
                let (product, term_norm) = covariant_row
                    .iter()
                    .zip(inverse.iter())
                    .map(|(left, inverse_row)| left * inverse_row[column])
                    .fold((0.0, 0.0_f64), |(sum, norm), term| {
                        (sum + term, norm + term.abs())
                    });
                let expected = if row == column { 1.0 } else { 0.0 };
                let normalized = (product - expected).abs() / term_norm.max(1.0);
                residual = residual.max(normalized);
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
        self.geometry(event)
            .map(|geometry| geometry.singularity_measure)
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
        let [_, x, y, z] = event.to_txyz();
        if ![x, y, z].into_iter().all(f64::is_finite) {
            return Err(GeometryError::NonFinite);
        }
        let radius_squared_3d = x.mul_add(x, y.mul_add(y, z * z));
        let spin_squared = self.spin_m * self.spin_m;
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
        let radius_fourth = radius_squared * radius_squared;
        let singularity_measure = (spin_squared * z).mul_add(z, radius_fourth);
        if singularity_measure <= 0.0 || !singularity_measure.is_finite() {
            return Err(GeometryError::InvalidDenominator);
        }
        let radius_gradient = DVec3::new(
            x * radius.powi(3) / singularity_measure,
            y * radius.powi(3) / singularity_measure,
            z * radius * (radius_squared + spin_squared) / singularity_measure,
        );
        let radius_cubed = radius.powi(3);
        let charge_squared = self.charge_m * self.charge_m;
        let numerator = (-charge_squared).mul_add(radius_squared, 2.0 * self.mass_m * radius_cubed);
        let scalar_f = numerator / singularity_measure;
        let numerator_factor =
            (-2.0 * charge_squared).mul_add(radius, 6.0 * self.mass_m * radius_squared);
        let numerator_gradient = radius_gradient * numerator_factor;
        let denominator_gradient = DVec3::new(
            4.0 * radius_cubed * radius_gradient.x,
            4.0 * radius_cubed * radius_gradient.y,
            (4.0 * radius_cubed).mul_add(radius_gradient.z, 2.0 * spin_squared * z),
        );
        let denominator_squared = singularity_measure * singularity_measure;
        let scalar_f_gradient = (numerator_gradient * singularity_measure
            - denominator_gradient * numerator)
            / denominator_squared;
        let radial_denominator = radius_squared + spin_squared;
        if radial_denominator <= 0.0 || !radial_denominator.is_finite() {
            return Err(GeometryError::InvalidDenominator);
        }
        let null_spatial = DVec3::new(
            self.spin_m.mul_add(y, radius * x) / radial_denominator,
            self.spin_m.mul_add(-x, radius * y) / radial_denominator,
            z / radius,
        );
        let null_covector = [1.0, null_spatial.x, null_spatial.y, null_spatial.z];
        let null_vector = [-1.0, null_spatial.x, null_spatial.y, null_spatial.z];
        let coordinates = [x, y, z];
        let radius_gradients = radius_gradient.to_array();
        let null_vector_gradient = std::array::from_fn(|index| {
            let radius_i = radius_gradients[index];
            let delta_x = f64::from(index == 0);
            let delta_y = f64::from(index == 1);
            let delta_z = f64::from(index == 2);
            let numerator_x = self
                .spin_m
                .mul_add(delta_y, radius_i.mul_add(x, radius * delta_x));
            let numerator_y = self
                .spin_m
                .mul_add(-delta_x, radius_i.mul_add(y, radius * delta_y));
            let radial_derivative = 2.0 * radius * radius_i;
            let derivative_x = (-null_spatial.x * radial_derivative)
                .mul_add(radial_denominator.recip(), numerator_x / radial_denominator);
            let derivative_y = (-null_spatial.y * radial_derivative)
                .mul_add(radial_denominator.recip(), numerator_y / radial_denominator);
            let derivative_z = delta_z / radius - coordinates[2] * radius_i / radius_squared;
            [0.0, derivative_x, derivative_y, derivative_z]
        });
        let geometry = Geometry {
            radius,
            radius_gradient,
            scalar_f,
            scalar_f_gradient,
            null_covector,
            null_vector,
            null_vector_gradient,
            singularity_measure,
        };
        validate_geometry(geometry)
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

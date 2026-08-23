use std::cmp::Ordering;

use glam::DVec3;

mod exact_binary;

use exact_binary::ExactBinary;

use crate::{
    GeodesicState, SpacetimeEvent, ValidationIssue, ValidationIssueCode, ValidationReport,
    numerics::{
        normalized_inner_product_residual, normalized_quadratic_form_residual, positive_product,
    },
    state::FourVector,
    validation::validate_finite,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extremality {
    Subextremal,
    Extremal,
    Superextremal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KerrSchildChart {
    Ingoing,
    Outgoing,
}

impl KerrSchildChart {
    const fn branch_sign(self) -> f64 {
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
    chart: KerrSchildChart,
    extremality: Extremality,
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
        chart: KerrSchildChart,
    ) -> Result<Self, ValidationReport> {
        Self::validated_with_prefix(mass_m, spin_m, charge_m, chart, "spacetime")
    }

    pub(super) fn validated_with_prefix(
        mass_m: f64,
        spin_m: f64,
        charge_m: f64,
        chart: KerrSchildChart,
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
        let extremality = match mass_squared.cmp(&angular_squared) {
            Ordering::Greater => Extremality::Subextremal,
            Ordering::Equal => Extremality::Extremal,
            Ordering::Less => Extremality::Superextremal,
        };
        let spacetime = Self {
            mass_m,
            spin_m,
            charge_m,
            chart,
            extremality,
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
    pub const fn chart(self) -> KerrSchildChart {
        self.chart
    }

    #[must_use]
    pub const fn extremality(self) -> Extremality {
        self.extremality
    }

    #[must_use]
    pub fn outer_horizon_radius(self) -> Option<f64> {
        match self.extremality {
            Extremality::Superextremal => None,
            Extremality::Extremal => Some(self.mass_m),
            Extremality::Subextremal => {
                let mass_squared = ExactBinary::square(self.mass_m);
                let angular_squared = ExactBinary::sum_of_squares(self.spin_m, self.charge_m);
                let discriminant_root = mass_squared.subtract(&angular_squared).square_root();
                Some(self.mass_m + discriminant_root)
            }
        }
    }

    /// Converts this chart's oblate coordinates to Cartesian Kerr-Schild coordinates.
    ///
    /// `azimuth` is the selected chart's azimuth. `spin_m` remains the physical
    /// `J / M`; the outgoing chart applies the opposite oblate spatial twist.
    #[must_use]
    pub fn oblate_to_cartesian(self, radius: f64, polar: f64, azimuth: f64) -> [f64; 3] {
        let (sin_theta, cos_theta) = polar.sin_cos();
        let (sin_phi, cos_phi) = azimuth.sin_cos();
        let chart_spin = self.chart.branch_sign() * self.spin_m;
        [
            chart_spin.mul_add(-sin_phi, radius * cos_phi) * sin_theta,
            chart_spin.mul_add(cos_phi, radius * sin_phi) * sin_theta,
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
        let branch_sign = self.chart.branch_sign();
        let chart_spin = branch_sign * spin;
        let null_spatial = branch_sign
            * DVec3::new(
                chart_spin.mul_add(y, scaled_radius * x) / radial_denominator,
                chart_spin.mul_add(-x, scaled_radius * y) / radial_denominator,
                z / scaled_radius,
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
            let numerator_x =
                chart_spin.mul_add(delta_y, radius_i.mul_add(x, scaled_radius * delta_x));
            let numerator_y =
                chart_spin.mul_add(-delta_x, radius_i.mul_add(y, scaled_radius * delta_y));
            let radial_derivative = 2.0 * scaled_radius * radius_i;
            let derivative_x = ((-branch_sign * null_spatial.x * radial_derivative)
                .mul_add(radial_denominator.recip(), numerator_x / radial_denominator))
                / scale;
            let derivative_y = ((-branch_sign * null_spatial.y * radial_derivative)
                .mul_add(radial_denominator.recip(), numerator_y / radial_denominator))
                / scale;
            let derivative_z =
                (delta_z / scaled_radius - coordinates[2] * radius_i / radius_squared) / scale;
            [
                0.0,
                branch_sign * derivative_x,
                branch_sign * derivative_y,
                branch_sign * derivative_z,
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
        let branch_sign = self.chart.branch_sign();
        let null_covector = [1.0, 0.0, 0.0, branch_sign * axis_sign];
        let null_vector = [-1.0, 0.0, 0.0, branch_sign * axis_sign];
        let inverse_denominator_scale = sigma.recip() / scale;
        let null_vector_gradient = [
            [
                0.0,
                branch_sign * scaled_radius * inverse_denominator_scale,
                -spin * inverse_denominator_scale,
                0.0,
            ],
            [
                0.0,
                spin * inverse_denominator_scale,
                branch_sign * scaled_radius * inverse_denominator_scale,
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
            b.midpoint(root)
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

#[cfg(test)]
mod tests {
    use approx::{abs_diff_eq, assert_abs_diff_eq};
    use proptest::prelude::*;

    use super::{GeodesicState, KerrNewmanSpacetime, KerrSchildChart, SpacetimeEvent};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn both_charts_pull_back_to_the_same_physical_kerr_newman_metric(
            mass in 0.5_f64..4.0,
            spin_fraction in -0.9_f64..0.9,
            charge_fraction in -0.2_f64..0.2,
            radius_fraction in 3.0_f64..20.0,
            polar in 0.1_f64..(std::f64::consts::PI - 0.1),
            azimuth in -std::f64::consts::PI..std::f64::consts::PI,
        ) {
        let spin = mass * spin_fraction;
        let charge = mass * charge_fraction;
        let radius = mass * radius_fraction;
        let (sin_theta, cos_theta) = polar.sin_cos();
        let sin_theta_squared = sin_theta * sin_theta;
            let sigma = radius.mul_add(radius, spin * spin * cos_theta * cos_theta);
            let spin_charge_squared = charge.mul_add(charge, spin * spin);
            let delta = radius.mul_add(radius, (2.0 * mass).mul_add(-radius, spin_charge_squared));
            let radial_numerator = charge.mul_add(-charge, 2.0 * mass * radius);
            let radial_factor = spin.mul_add(spin, radius * radius);
            let expected = [
                [
                    -1.0 + radial_numerator / sigma,
                    0.0,
                    0.0,
                    -radial_numerator * spin * sin_theta_squared / sigma,
                ],
                [0.0, sigma / delta, 0.0, 0.0],
                [0.0, 0.0, sigma, 0.0],
                [
                    -radial_numerator * spin * sin_theta_squared / sigma,
                    0.0,
                    0.0,
                    sin_theta_squared
                        * (spin * spin * delta)
                            .mul_add(-sin_theta_squared, radial_factor * radial_factor)
                        / sigma,
                ],
            ];

            for (chart, branch_sign) in [
                (KerrSchildChart::Ingoing, 1.0),
                (KerrSchildChart::Outgoing, -1.0),
            ] {
                let spacetime = KerrNewmanSpacetime::new(mass, spin, charge, chart)
                    .expect("fixture parameters are valid");
                let [x, y, z] = spacetime.oblate_to_cartesian(radius, polar, azimuth);
                let event =
                    SpacetimeEvent::from_txyz([0.0, x, y, z]).expect("fixture event is finite");
                let metric = spacetime
                    .metric_covariant_at(event)
                    .expect("fixture metric is regular");
                let (sin_phi, cos_phi) = azimuth.sin_cos();
                let chart_spin = branch_sign * spin;
                let chart_basis = [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, sin_theta * cos_phi, sin_theta * sin_phi, cos_theta],
                    [
                        0.0,
                        cos_theta * chart_spin.mul_add(-sin_phi, radius * cos_phi),
                        cos_theta * chart_spin.mul_add(cos_phi, radius * sin_phi),
                        -radius * sin_theta,
                    ],
                    [0.0, -y, x, 0.0],
                ];
                let mut boyer_lindquist_basis = chart_basis;
                for component in 0..4 {
                    let radial_shift = (spin / delta).mul_add(
                        chart_basis[3][component],
                        radial_numerator / delta * chart_basis[0][component],
                    );
                    boyer_lindquist_basis[1][component] =
                        branch_sign.mul_add(radial_shift, boyer_lindquist_basis[1][component]);
                }

                for row in 0..4 {
                    for column in 0..4 {
                        let (actual, term_norm) = metric_pairing(
                            metric,
                            boyer_lindquist_basis[row],
                            boyer_lindquist_basis[column],
                        );
                        let scale = term_norm.max(expected[row][column].abs()).max(1.0);
                        prop_assert!(
                            abs_diff_eq!(
                                actual,
                                expected[row][column],
                                epsilon = 128.0 * f64::EPSILON * scale
                            ),
                            "{chart:?}, a={spin}: g[{row},{column}] was {actual:e}, expected {:e}",
                            expected[row][column]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn axis_hamiltonian_rhs_preserves_physical_spin_handedness() {
        let spin = 0.8_f64;
        let radius = 4.0_f64;
        let radial_denominator = radius.mul_add(radius, spin * spin);
        let scalar_f = 2.0 * radius / radial_denominator;
        let expected_spin_force = scalar_f * spin / radial_denominator;
        let event =
            SpacetimeEvent::from_txyz([0.0, 0.0, 0.0, radius]).expect("fixture event is finite");
        let x_force_state = GeodesicState::new(event.to_txyz(), [-1.0, 0.0, 1.0, 0.0])
            .expect("fixture state is finite");
        let y_force_state = GeodesicState::new(event.to_txyz(), [-1.0, 1.0, 0.0, 0.0])
            .expect("fixture state is finite");

        for chart in [KerrSchildChart::Ingoing, KerrSchildChart::Outgoing] {
            let spacetime = KerrNewmanSpacetime::new(1.0, spin, 0.0, chart)
                .expect("fixture parameters are valid");
            let x_force = spacetime
                .hamiltonian_rhs(x_force_state)
                .expect("axis geometry is regular");
            let y_force = spacetime
                .hamiltonian_rhs(y_force_state)
                .expect("axis geometry is regular");

            assert_abs_diff_eq!(
                x_force[5],
                -expected_spin_force,
                epsilon = 8.0 * f64::EPSILON
            );
            assert_abs_diff_eq!(
                y_force[6],
                expected_spin_force,
                epsilon = 8.0 * f64::EPSILON
            );
        }
    }

    fn metric_pairing(metric: [[f64; 4]; 4], left: [f64; 4], right: [f64; 4]) -> (f64, f64) {
        let terms = metric.into_iter().zip(left).flat_map(|(row, left)| {
            row.into_iter()
                .zip(right)
                .map(move |(component, right)| left * component * right)
        });
        terms.fold((0.0, 0.0), |(sum, norm), term| {
            (sum + term, norm + term.abs())
        })
    }
}

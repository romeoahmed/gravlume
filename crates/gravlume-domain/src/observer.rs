use crate::{
    KerrNewmanSpacetime, SpacetimeEvent, ValidationIssue, ValidationIssueCode, ValidationReport,
    numerics::normalized_quadratic_form_residual, state::FourVector,
    validation::validate_finite_array,
};

const BASIS_TOLERANCE: f64 = 1.0e-12;

#[derive(Clone, Debug)]
pub struct StationaryObserverInput {
    event_txyz_m: [f64; 4],
    target_txyz_m: [f64; 4],
    up_hint_xyz: [f64; 3],
    measured_frequency: f64,
}

impl StationaryObserverInput {
    #[must_use]
    pub const fn new(
        event_txyz_m: [f64; 4],
        target_txyz_m: [f64; 4],
        up_hint_xyz: [f64; 3],
        measured_frequency: f64,
    ) -> Self {
        Self {
            event_txyz_m,
            target_txyz_m,
            up_hint_xyz,
            measured_frequency,
        }
    }

    pub(super) fn validate(&self, prefix: &str) -> ValidationReport {
        let mut report = ValidationReport::default();
        validate_finite_array(
            &mut report,
            self.event_txyz_m,
            format!("{prefix}.event_txyz_m"),
        );
        validate_finite_array(
            &mut report,
            self.target_txyz_m,
            format!("{prefix}.target_txyz_m"),
        );
        validate_finite_array(
            &mut report,
            self.up_hint_xyz,
            format!("{prefix}.up_hint_xyz"),
        );
        let frequency_path = format!("{prefix}.measured_frequency");
        if !self.measured_frequency.is_finite() {
            report.push(ValidationIssue::error(
                ValidationIssueCode::NonFinite,
                frequency_path,
                "measured frequency must be finite",
            ));
        } else if self.measured_frequency <= 0.0 {
            report.push(ValidationIssue::error(
                ValidationIssueCode::NonPositive,
                frequency_path,
                "measured frequency must be positive",
            ));
        }
        report
    }

    pub(super) fn build(
        self,
        spacetime: KerrNewmanSpacetime,
        prefix: &str,
    ) -> Result<StationaryObserver, ValidationReport> {
        let event = SpacetimeEvent::from_validated(self.event_txyz_m);
        let metric_g_tt = spacetime.metric_component_tt(event).map_err(|error| {
            ValidationReport::from_error(
                ValidationIssueCode::OutOfRange,
                format!("{prefix}.event_txyz_m"),
                error.to_string(),
            )
        })?;
        if metric_g_tt >= 0.0 {
            return Err(ValidationReport::from_error(
                ValidationIssueCode::NonStationaryObserver,
                format!("{prefix}.event_txyz_m"),
                "a stationary observer requires g_tt < 0",
            ));
        }
        let four_velocity = FourVector::new([(-metric_g_tt).sqrt().recip(), 0.0, 0.0, 0.0]);
        let frame = construct_frame(spacetime, event, four_velocity, &self, prefix)?;
        let metric_covariant = spacetime.metric_covariant_at(event).map_err(|error| {
            ValidationReport::from_error(
                ValidationIssueCode::InternalInvariant,
                prefix,
                error.to_string(),
            )
        })?;
        Ok(StationaryObserver {
            event,
            frame,
            measured_frequency: self.measured_frequency,
            metric_g_tt,
            metric_covariant,
        })
    }
}

fn construct_frame(
    spacetime: KerrNewmanSpacetime,
    event: SpacetimeEvent,
    four_velocity: FourVector,
    input: &StationaryObserverInput,
    prefix: &str,
) -> Result<ObserverFrame, ValidationReport> {
    let target_seed = FourVector::new(std::array::from_fn(|index| {
        input.target_txyz_m[index] - input.event_txyz_m[index]
    }));
    let Ok(sight) = project_and_normalize(spacetime, event, four_velocity, target_seed, &[]) else {
        return Err(ValidationReport::from_error(
            ValidationIssueCode::DegenerateDirection,
            format!("{prefix}.target_txyz_m"),
            "target does not define a spatial direction in the observer rest frame",
        ));
    };
    let arrival = sight.scaled(-1.0);
    let requested_up = FourVector::new([
        0.0,
        input.up_hint_xyz[0],
        input.up_hint_xyz[1],
        input.up_hint_xyz[2],
    ]);
    let requested_up =
        project_and_normalize(spacetime, event, four_velocity, requested_up, &[sight]);
    let (up, used_up_fallback) = if let Ok(up) = requested_up {
        (up, false)
    } else {
        let Some(up) = best_coordinate_axis(spacetime, event, four_velocity, &[sight]) else {
            return Err(ValidationReport::from_error(
                ValidationIssueCode::DegenerateDirection,
                format!("{prefix}.up_hint_xyz"),
                "no stable image-up axis could be constructed",
            ));
        };
        (up, true)
    };
    let Some(mut right) = best_coordinate_axis(spacetime, event, four_velocity, &[sight, up])
    else {
        return Err(ValidationReport::from_error(
            ValidationIssueCode::InternalInvariant,
            prefix,
            "observer tetrad could not be completed",
        ));
    };
    let mut orientation = determinant4([
        four_velocity.to_array(),
        right.to_array(),
        up.to_array(),
        arrival.to_array(),
    ]);
    if orientation < 0.0 {
        right = right.scaled(-1.0);
        orientation = -orientation;
    }
    let frame = ObserverFrame {
        four_velocity,
        image_right: right,
        image_up: up,
        arrival,
        gram_residual: gram_residual(spacetime, event, [four_velocity, right, up, arrival]),
        orientation_determinant: orientation,
        used_up_fallback,
    };
    if !frame.gram_residual.is_finite()
        || frame.gram_residual > BASIS_TOLERANCE
        || !frame.orientation_determinant.is_finite()
        || frame.orientation_determinant <= 0.0
    {
        return Err(ValidationReport::from_error(
            ValidationIssueCode::InternalInvariant,
            prefix,
            "constructed observer frame failed its Gram/orientation contract",
        ));
    }
    Ok(frame)
}

#[derive(Clone, Copy, Debug)]
pub struct ObserverFrame {
    four_velocity: FourVector,
    image_right: FourVector,
    image_up: FourVector,
    arrival: FourVector,
    gram_residual: f64,
    orientation_determinant: f64,
    used_up_fallback: bool,
}

impl ObserverFrame {
    #[must_use]
    pub const fn four_velocity_txyz(&self) -> [f64; 4] {
        self.four_velocity.to_array()
    }

    #[must_use]
    pub const fn image_right_txyz(&self) -> [f64; 4] {
        self.image_right.to_array()
    }

    #[must_use]
    pub const fn image_up_txyz(&self) -> [f64; 4] {
        self.image_up.to_array()
    }

    #[must_use]
    pub const fn arrival_direction_txyz(&self) -> [f64; 4] {
        self.arrival.to_array()
    }

    #[must_use]
    pub const fn gram_residual(&self) -> f64 {
        self.gram_residual
    }

    #[must_use]
    pub const fn orientation_determinant(&self) -> f64 {
        self.orientation_determinant
    }

    #[must_use]
    pub const fn used_up_fallback(&self) -> bool {
        self.used_up_fallback
    }

    pub(super) const fn four_velocity(&self) -> FourVector {
        self.four_velocity
    }

    pub(super) const fn image_right(&self) -> FourVector {
        self.image_right
    }

    pub(super) const fn image_up(&self) -> FourVector {
        self.image_up
    }

    pub(super) const fn arrival(&self) -> FourVector {
        self.arrival
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StationaryObserver {
    event: SpacetimeEvent,
    frame: ObserverFrame,
    measured_frequency: f64,
    metric_g_tt: f64,
    metric_covariant: [[f64; 4]; 4],
}

impl StationaryObserver {
    pub(super) const fn event(&self) -> SpacetimeEvent {
        self.event
    }

    pub(super) const fn frame(&self) -> &ObserverFrame {
        &self.frame
    }

    pub(super) const fn measured_frequency(&self) -> f64 {
        self.measured_frequency
    }

    pub(super) const fn metric_g_tt(&self) -> f64 {
        self.metric_g_tt
    }

    pub(super) fn lower(&self, vector: FourVector) -> [f64; 4] {
        let vector = vector.to_array();
        std::array::from_fn(|row| {
            (0..4)
                .map(|column| self.metric_covariant[row][column] * vector[column])
                .sum()
        })
    }

    pub(super) fn normalized_null_residual(&self, momentum: FourVector) -> f64 {
        normalized_quadratic_form_residual(self.metric_covariant, momentum)
    }
}

fn project_and_normalize(
    spacetime: KerrNewmanSpacetime,
    event: SpacetimeEvent,
    four_velocity: FourVector,
    seed: FourVector,
    orthogonal_to: &[FourVector],
) -> Result<FourVector, ()> {
    if !seed.is_finite() {
        return Err(());
    }
    let u_dot_seed = spacetime
        .metric_dot(event, four_velocity, seed)
        .map_err(|_| ())?;
    let mut projected = seed.add(four_velocity.scaled(u_dot_seed));
    for basis in orthogonal_to {
        let component = spacetime
            .metric_dot(event, projected, *basis)
            .map_err(|_| ())?;
        projected = projected.subtract(basis.scaled(component));
    }
    let norm_squared = spacetime
        .metric_dot(event, projected, projected)
        .map_err(|_| ())?;
    if !norm_squared.is_finite() || norm_squared <= BASIS_TOLERANCE * BASIS_TOLERANCE {
        return Err(());
    }
    let normalized = projected.scaled(norm_squared.sqrt().recip());
    normalized.is_finite().then_some(normalized).ok_or(())
}

fn best_coordinate_axis(
    spacetime: KerrNewmanSpacetime,
    event: SpacetimeEvent,
    four_velocity: FourVector,
    orthogonal_to: &[FourVector],
) -> Option<FourVector> {
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    .into_iter()
    .filter_map(|axis| {
        project_and_normalize(
            spacetime,
            event,
            four_velocity,
            FourVector::new(axis),
            orthogonal_to,
        )
        .ok()
    })
    .max_by(|left, right| {
        let left_score = left.to_array()[1..]
            .iter()
            .map(|component| component.abs())
            .fold(0.0_f64, f64::max);
        let right_score = right.to_array()[1..]
            .iter()
            .map(|component| component.abs())
            .fold(0.0_f64, f64::max);
        left_score.total_cmp(&right_score)
    })
}

fn gram_residual(
    spacetime: KerrNewmanSpacetime,
    event: SpacetimeEvent,
    basis: [FourVector; 4],
) -> f64 {
    let mut residual = 0.0_f64;
    for row in 0..4 {
        for column in 0..4 {
            let Ok((actual, term_norm)) =
                spacetime.metric_dot_and_term_norm(event, basis[row], basis[column])
            else {
                return f64::INFINITY;
            };
            let expected = if row == column {
                if row == 0 { -1.0 } else { 1.0 }
            } else {
                0.0
            };
            residual = residual.max((actual - expected).abs() / term_norm.max(1.0));
        }
    }
    residual
}

fn determinant4(columns: [[f64; 4]; 4]) -> f64 {
    let mut matrix: [[f64; 4]; 4] =
        std::array::from_fn(|row| std::array::from_fn(|column| columns[column][row]));
    let mut determinant = 1.0;
    for pivot_column in 0..4 {
        let Some(pivot_row) = (pivot_column..4).max_by(|left, right| {
            matrix[*left][pivot_column]
                .abs()
                .total_cmp(&matrix[*right][pivot_column].abs())
        }) else {
            return 0.0;
        };
        if matrix[pivot_row][pivot_column] == 0.0 {
            return 0.0;
        }
        if pivot_row != pivot_column {
            matrix.swap(pivot_row, pivot_column);
            determinant = -determinant;
        }
        let pivot = matrix[pivot_column][pivot_column];
        determinant *= pivot;
        let pivot_values = matrix[pivot_column];
        for row_values in matrix.iter_mut().skip(pivot_column + 1) {
            let factor = row_values[pivot_column] / pivot;
            for (column, value) in row_values.iter_mut().enumerate().skip(pivot_column + 1) {
                *value = factor.mul_add(-pivot_values[column], *value);
            }
        }
    }
    determinant
}

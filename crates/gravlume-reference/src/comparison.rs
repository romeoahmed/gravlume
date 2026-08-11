use crate::{ReferenceOutcome, Termination};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonIssue {
    TerminationMismatch,
    EventPositionBudgetExceeded,
    EscapeDirectionBudgetExceeded,
    TravelTimeBudgetExceeded,
    NullDriftBudgetExceeded,
    EnergyDriftBudgetExceeded,
    AngularMomentumZDriftBudgetExceeded,
    CarterDriftBudgetExceeded,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceComparison {
    baseline_policy: &'static str,
    candidate_policy: &'static str,
    event_position_distance_m: Option<f64>,
    escape_direction_angle_rad: Option<f64>,
    travel_time_difference_m: f64,
    issues: Vec<ComparisonIssue>,
}

impl ReferenceComparison {
    #[must_use]
    pub fn baseline_v1(baseline: &ReferenceOutcome, candidate: &ReferenceOutcome) -> Self {
        let mut issues = Vec::new();
        if baseline.termination() != candidate.termination() {
            issues.push(ComparisonIssue::TerminationMismatch);
        }
        let event_position_distance_m = event_position_distance(baseline, candidate);
        if event_position_distance_m.is_some_and(|distance| distance > 2.0e-9) {
            issues.push(ComparisonIssue::EventPositionBudgetExceeded);
        }
        let escape_direction_angle_rad = if baseline.termination() == Termination::Escape
            && candidate.termination() == Termination::Escape
        {
            escape_direction_angle(baseline, candidate)
        } else {
            None
        };
        if escape_direction_angle_rad.is_some_and(|angle| angle > 2.0e-9) {
            issues.push(ComparisonIssue::EscapeDirectionBudgetExceeded);
        }
        let travel_time_difference_m = (baseline.travel_time_m() - candidate.travel_time_m()).abs();
        if travel_time_difference_m > 2.0e-8 {
            issues.push(ComparisonIssue::TravelTimeBudgetExceeded);
        }
        let baseline_diagnostics = baseline.diagnostics();
        let candidate_diagnostics = candidate.diagnostics();
        if baseline_diagnostics
            .maximum_null_residual()
            .max(candidate_diagnostics.maximum_null_residual())
            > 5.0e-9
        {
            issues.push(ComparisonIssue::NullDriftBudgetExceeded);
        }
        if baseline_diagnostics
            .maximum_energy_drift()
            .max(candidate_diagnostics.maximum_energy_drift())
            > 5.0e-9
        {
            issues.push(ComparisonIssue::EnergyDriftBudgetExceeded);
        }
        if baseline_diagnostics
            .maximum_angular_momentum_z_drift()
            .max(candidate_diagnostics.maximum_angular_momentum_z_drift())
            > 5.0e-9
        {
            issues.push(ComparisonIssue::AngularMomentumZDriftBudgetExceeded);
        }
        if baseline_diagnostics
            .maximum_carter_drift()
            .max(candidate_diagnostics.maximum_carter_drift())
            > 5.0e-9
        {
            issues.push(ComparisonIssue::CarterDriftBudgetExceeded);
        }
        Self {
            baseline_policy: baseline.policy_id(),
            candidate_policy: candidate.policy_id(),
            event_position_distance_m,
            escape_direction_angle_rad,
            travel_time_difference_m,
            issues,
        }
    }

    #[must_use]
    pub const fn baseline_policy(&self) -> &'static str {
        self.baseline_policy
    }

    #[must_use]
    pub const fn candidate_policy(&self) -> &'static str {
        self.candidate_policy
    }

    #[must_use]
    pub const fn event_position_distance_m(&self) -> Option<f64> {
        self.event_position_distance_m
    }

    #[must_use]
    pub const fn escape_direction_angle_rad(&self) -> Option<f64> {
        self.escape_direction_angle_rad
    }

    #[must_use]
    pub const fn travel_time_difference_m(&self) -> f64 {
        self.travel_time_difference_m
    }

    #[must_use]
    pub fn issues(&self) -> &[ComparisonIssue] {
        &self.issues
    }

    #[must_use]
    pub const fn is_accepted(&self) -> bool {
        self.issues.is_empty()
    }
}

fn event_position_distance(left: &ReferenceOutcome, right: &ReferenceOutcome) -> Option<f64> {
    left.event().zip(right.event()).map(|_| {
        let left = left.state().components();
        let right = right.state().components();
        let dx = left[1] - right[1];
        let dy = left[2] - right[2];
        let dz = left[3] - right[3];
        dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt()
    })
}

fn escape_direction_angle(left: &ReferenceOutcome, right: &ReferenceOutcome) -> Option<f64> {
    let left = unit_spatial_position(left)?;
    let right = unit_spatial_position(right)?;
    let dot = left[0]
        .mul_add(right[0], left[1].mul_add(right[1], left[2] * right[2]))
        .clamp(-1.0, 1.0);
    let cross = [
        left[1].mul_add(right[2], -left[2] * right[1]),
        left[2].mul_add(right[0], -left[0] * right[2]),
        left[0].mul_add(right[1], -left[1] * right[0]),
    ];
    let cross_norm = cross[0]
        .mul_add(cross[0], cross[1].mul_add(cross[1], cross[2] * cross[2]))
        .sqrt();
    Some(cross_norm.atan2(dot))
}

fn unit_spatial_position(outcome: &ReferenceOutcome) -> Option<[f64; 3]> {
    let components = outcome.state().components();
    let norm = components[1]
        .mul_add(
            components[1],
            components[2].mul_add(components[2], components[3] * components[3]),
        )
        .sqrt();
    (norm > 0.0 && norm.is_finite()).then(|| {
        [
            components[1] / norm,
            components[2] / norm,
            components[3] / norm,
        ]
    })
}

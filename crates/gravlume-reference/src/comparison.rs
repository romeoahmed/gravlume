use crate::{
    ReferenceOutcome, Termination, TraceInputId,
    policy::{REGULAR_V1_ID, STRICT_V1_ID},
};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ComparisonError {
    #[error("baseline policy must be {expected}, got {actual}")]
    UnexpectedBaselinePolicy {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("candidate policy must be {expected}, got {actual}")]
    UnexpectedCandidatePolicy {
        expected: &'static str,
        actual: &'static str,
    },
    #[error(
        "comparison inputs differ: baseline {baseline_input_id}, candidate {candidate_input_id}"
    )]
    InputMismatch {
        baseline_input_id: TraceInputId,
        candidate_input_id: TraceInputId,
    },
    #[error("comparison input label {input_id} resolves to different canonical inputs")]
    InputIdentityCollision { input_id: TraceInputId },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonIssue {
    TerminationMismatch,
    UnsuccessfulTermination,
    EventPositionBudgetExceeded,
    EscapeDirectionUnavailable,
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
    /// Compares regular and strict outcomes for the same input under the v1 budget.
    ///
    /// # Errors
    ///
    /// Rejects outcomes with the wrong policy roles or different input identities.
    pub fn baseline_v1(
        baseline: &ReferenceOutcome,
        candidate: &ReferenceOutcome,
    ) -> Result<Self, ComparisonError> {
        if baseline.policy_id() != REGULAR_V1_ID {
            return Err(ComparisonError::UnexpectedBaselinePolicy {
                expected: REGULAR_V1_ID,
                actual: baseline.policy_id(),
            });
        }
        if candidate.policy_id() != STRICT_V1_ID {
            return Err(ComparisonError::UnexpectedCandidatePolicy {
                expected: STRICT_V1_ID,
                actual: candidate.policy_id(),
            });
        }
        if baseline.input_id().as_str() != candidate.input_id().as_str() {
            return Err(ComparisonError::InputMismatch {
                baseline_input_id: baseline.input_id().logical(),
                candidate_input_id: candidate.input_id().logical(),
            });
        }
        if baseline.input_id() != candidate.input_id() {
            return Err(ComparisonError::InputIdentityCollision {
                input_id: baseline.input_id().logical(),
            });
        }
        let mut issues = Vec::new();
        if baseline.termination() != candidate.termination() {
            issues.push(ComparisonIssue::TerminationMismatch);
        }
        if is_unsuccessful(baseline.termination()) || is_unsuccessful(candidate.termination()) {
            issues.push(ComparisonIssue::UnsuccessfulTermination);
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
        if baseline.termination() == Termination::Escape
            && candidate.termination() == Termination::Escape
        {
            match escape_direction_angle_rad {
                Some(angle) if angle > 2.0e-9 => {
                    issues.push(ComparisonIssue::EscapeDirectionBudgetExceeded);
                }
                None => issues.push(ComparisonIssue::EscapeDirectionUnavailable),
                Some(_) => {}
            }
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
        Ok(Self {
            baseline_policy: baseline.policy_id(),
            candidate_policy: candidate.policy_id(),
            event_position_distance_m,
            escape_direction_angle_rad,
            travel_time_difference_m,
            issues,
        })
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

const fn is_unsuccessful(termination: Termination) -> bool {
    matches!(
        termination,
        Termination::StepExhaustion
            | Termination::RejectExhaustion
            | Termination::NumericalFailure(_)
    )
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
    let left = left.escape_direction_xyz()?;
    let right = right.escape_direction_xyz()?;
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

#[cfg(test)]
mod tests {
    use gravlume_domain::GeodesicState;

    use super::{ComparisonIssue, ReferenceComparison};
    use crate::{NumericalFailure, ReferenceOutcome, Termination, TraceDiagnostics, TraceInputId};

    #[test]
    fn matching_unsuccessful_terminations_are_not_convergence_evidence() {
        for termination in [
            Termination::StepExhaustion,
            Termination::RejectExhaustion,
            Termination::NumericalFailure(NumericalFailure::NonFinite),
        ] {
            let mut baseline = escape_outcome("reference-regular-v1", [1.0, 0.0, 0.0]);
            baseline.termination = termination;
            baseline.escape_direction_xyz = None;
            let mut candidate = escape_outcome("reference-strict-v1", [1.0, 0.0, 0.0]);
            candidate.termination = termination;
            candidate.escape_direction_xyz = None;

            let comparison = ReferenceComparison::baseline_v1(&baseline, &candidate)
                .expect("policy roles and input identity match");

            assert!(!comparison.is_accepted(), "accepted {termination:?}");
        }
    }

    #[test]
    fn escape_direction_gate_uses_terminal_momentum_not_event_position() {
        let baseline = escape_outcome("reference-regular-v1", [1.0, 0.0, 0.0]);
        let candidate = escape_outcome("reference-strict-v1", [0.0, 1.0, 0.0]);

        let comparison = ReferenceComparison::baseline_v1(&baseline, &candidate)
            .expect("policy roles and input identity match");

        assert!(
            comparison
                .issues()
                .contains(&ComparisonIssue::EscapeDirectionBudgetExceeded)
        );
    }

    #[test]
    fn escape_direction_gate_rejects_a_missing_terminal_direction() {
        let baseline = escape_outcome("reference-regular-v1", [1.0, 0.0, 0.0]);
        let mut candidate = escape_outcome("reference-strict-v1", [1.0, 0.0, 0.0]);
        candidate.escape_direction_xyz = None;

        let comparison = ReferenceComparison::baseline_v1(&baseline, &candidate)
            .expect("policy roles and input identity match");

        assert!(
            comparison
                .issues()
                .contains(&ComparisonIssue::EscapeDirectionUnavailable)
        );
    }

    fn escape_outcome(policy_id: &'static str, spatial_momentum: [f64; 3]) -> ReferenceOutcome {
        ReferenceOutcome {
            input_id: TraceInputId::new("same-input"),
            policy_id,
            termination: Termination::Escape,
            state: GeodesicState::new(
                [0.0, 200.0, 0.0, 0.0],
                [
                    -1.0,
                    spatial_momentum[0],
                    spatial_momentum[1],
                    spatial_momentum[2],
                ],
            )
            .expect("state is finite"),
            affine_parameter_m: 1.0,
            event: None,
            escape_direction_xyz: Some(spatial_momentum),
            turning_radius_m: None,
            azimuth_advance_rad: 0.0,
            travel_time_m: 1.0,
            diagnostics: TraceDiagnostics {
                accepted_steps: 1,
                rejected_steps: 0,
                rhs_evaluations: 7,
                minimum_step_m: Some(1.0),
                maximum_step_m: Some(1.0),
                maximum_null_residual: 0.0,
                maximum_energy_drift: 0.0,
                maximum_angular_momentum_z_drift: 0.0,
                maximum_carter_drift: 0.0,
            },
        }
    }
}

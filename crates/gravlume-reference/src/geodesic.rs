use std::f64::consts::PI;

use gravlume_domain::{GeodesicInvariants, GeodesicState, GeometryError, KerrNewmanSpacetime};

use crate::{
    events::{EventConfiguration, EventKind, escape_event_is_armed},
    fixture::GeodesicFixture,
    integrator::{DenseOutput, attempt_step, derivative},
    outcome::{
        GeodesicTrace, LocalizedEvent, NumericalFailure, ReferenceOutcome, Termination,
        TraceDiagnostics,
    },
    policy::ReferencePolicy,
};
#[derive(Clone, Debug)]
pub struct GeodesicTracer {
    spacetime: KerrNewmanSpacetime,
    policy: ReferencePolicy,
    events: EventConfiguration,
}

impl GeodesicTracer {
    /// Creates a tracer for the `M = 1` normalization required by the v1 policies.
    ///
    /// # Errors
    ///
    /// Rejects a spacetime that has not been normalized to unit mass.
    pub const fn new(
        spacetime: KerrNewmanSpacetime,
        policy: ReferencePolicy,
        events: EventConfiguration,
    ) -> Result<Self, GeodesicConfigurationError> {
        if spacetime.mass_m().to_bits() != 1.0_f64.to_bits() {
            return Err(GeodesicConfigurationError::NonNormalizedMass);
        }
        Ok(Self {
            spacetime,
            policy,
            events,
        })
    }

    /// Creates a tracer from a fixture whose normalization and events were validated while parsing.
    #[must_use]
    pub const fn from_fixture(fixture: &GeodesicFixture, policy: ReferencePolicy) -> Self {
        Self {
            spacetime: fixture.spacetime(),
            policy,
            events: fixture.event_configuration(),
        }
    }

    #[must_use]
    pub fn trace(&self, mut request: GeodesicTrace) -> ReferenceOutcome {
        request.input_id = request.input_id.bind(
            self.spacetime,
            request.initial_state,
            request.affine_direction,
            self.events,
        );
        TraceExecution::new(self, request).run()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GeodesicConfigurationError {
    #[error("reference-v1 inputs must be normalized to M = 1")]
    NonNormalizedMass,
}

struct TraceExecution<'tracer> {
    tracer: &'tracer GeodesicTracer,
    request: GeodesicTrace,
    state: GeodesicState,
    affine_parameter_m: f64,
    step_m: f64,
    accepted_steps: u64,
    rejected_steps: u64,
    rhs_evaluations: u64,
    consecutive_rejects: u32,
    minimum_step_m: Option<f64>,
    maximum_step_m: Option<f64>,
    initial_invariants: Option<GeodesicInvariants>,
    drifts: InvariantDrifts,
    event_arming: EventArming,
    turning_radius_m: Option<f64>,
    azimuth_advance_rad: f64,
    previous_azimuth_rad: f64,
    coordinate_time_delta_m: CompensatedSum,
}

impl<'tracer> TraceExecution<'tracer> {
    fn new(tracer: &'tracer GeodesicTracer, request: GeodesicTrace) -> Self {
        let state = request.initial_state;
        let traversal_sign = request.affine_direction.sign();
        let components = state.components();
        let initial_invariants = tracer.spacetime.invariants(state).ok();
        let event_arming = EventArming::new(tracer, state);
        Self {
            tracer,
            request,
            state,
            affine_parameter_m: 0.0,
            step_m: traversal_sign * tracer.policy.initial_step_m(),
            accepted_steps: 0,
            rejected_steps: 0,
            rhs_evaluations: 0,
            consecutive_rejects: 0,
            minimum_step_m: None,
            maximum_step_m: None,
            initial_invariants,
            drifts: InvariantDrifts::default(),
            event_arming,
            turning_radius_m: None,
            azimuth_advance_rad: 0.0,
            previous_azimuth_rad: components[2].atan2(components[1]),
            coordinate_time_delta_m: CompensatedSum::default(),
        }
    }

    fn run(mut self) -> ReferenceOutcome {
        let initial_components = self.state.components();
        let mut start_derivative =
            match self.evaluate_rhs(|spacetime| derivative(spacetime, initial_components)) {
                Ok(derivative) => derivative,
                Err(error) => return self.finish_failure(error.into()),
            };
        self.update_invariants(self.state);
        loop {
            if self.accepted_steps >= self.tracer.policy.maximum_accepted_steps() {
                let state = self.state;
                return self.finish(Termination::StepExhaustion, state, None);
            }
            self.record_step(self.step_m.abs());
            let attempt = match attempt_step(
                self.tracer.spacetime,
                self.state.components(),
                self.step_m,
                start_derivative,
            ) {
                Ok(attempt) => {
                    self.rhs_evaluations += 6;
                    attempt
                }
                Err(failure) => {
                    self.rhs_evaluations += u64::from(failure.evaluations());
                    return self.finish_failure(failure.error().into());
                }
            };
            let mut error_start = self.state.components();
            let mut error_end = attempt.end;
            error_start[0] = self.coordinate_time_delta_m.total();
            error_end[0] = error_start[0] + attempt.dense.time_increment(1.0);
            let error_norm = self
                .tracer
                .policy
                .error_norm(error_start, error_end, attempt.error);
            if !error_norm.is_finite() {
                return self.finish_failure(NumericalFailure::NonFinite);
            }
            if error_norm > 1.0 {
                self.rejected_steps += 1;
                self.consecutive_rejects += 1;
                if self.consecutive_rejects >= self.tracer.policy.maximum_consecutive_rejects() {
                    let state = self.state;
                    return self.finish(Termination::RejectExhaustion, state, None);
                }
                if self.step_m.abs() <= self.tracer.policy.minimum_step_m() {
                    return self.finish_failure(NumericalFailure::MinimumStep);
                }
                self.resize_step(error_norm, false);
                continue;
            }

            self.accepted_steps += 1;
            self.consecutive_rejects = 0;
            let Ok(end_state) = GeodesicState::from_components(attempt.end) else {
                return self.finish_failure(NumericalFailure::NonFinite);
            };
            let event = match self.find_event(&attempt.dense, end_state) {
                Ok(event) => event,
                Err(error) => return self.finish_failure(error.into()),
            };
            let committed_theta = event.as_ref().map_or(1.0, |event| event.theta);
            let committed_state = event.as_ref().map_or(end_state, |event| event.state);
            self.record_turning_point(&attempt.dense, end_state, committed_theta);
            self.coordinate_time_delta_m
                .add(attempt.dense.time_increment(committed_theta));
            self.commit_observables(committed_state);
            if let Some(event) = event {
                self.affine_parameter_m = self.step_m.mul_add(event.theta, self.affine_parameter_m);
                let localized = self.event_diagnostic(&event);
                return self.finish(
                    termination_for_event(event.kind),
                    committed_state,
                    Some(localized),
                );
            }

            self.affine_parameter_m += self.step_m;
            self.state = end_state;
            start_derivative = attempt.end_derivative;
            self.event_arming.update(self.tracer, end_state);
            self.resize_step(error_norm, true);
        }
    }

    fn resize_step(&mut self, error_norm: f64, accepted: bool) {
        let magnitude =
            self.tracer
                .policy
                .next_step_magnitude(self.step_m.abs(), error_norm, accepted);
        self.step_m = self.request.affine_direction.sign() * magnitude;
    }

    fn record_step(&mut self, magnitude: f64) {
        self.minimum_step_m = Some(
            self.minimum_step_m
                .map_or(magnitude, |value| value.min(magnitude)),
        );
        self.maximum_step_m = Some(
            self.maximum_step_m
                .map_or(magnitude, |value| value.max(magnitude)),
        );
    }

    fn find_event(
        &self,
        dense: &DenseOutput,
        end_state: GeodesicState,
    ) -> Result<Option<LocalizedRoot>, GeometryError> {
        let mut roots: [Option<LocalizedRoot>; 4] = std::array::from_fn(|_| None);
        let mut root_count = 0;
        for kind in EventKind::ordered() {
            if !self.event_arming.is_armed(kind) || !self.event_is_installed(kind) {
                continue;
            }
            let start_value = self.event_value(kind, self.state)?;
            let end_value = self.event_value(kind, end_state)?;
            if crosses(kind, start_value, end_value) {
                let root = self.localize_event(dense, kind, start_value, end_value)?;
                if kind != EventKind::EquatorialSurface
                    || self
                        .tracer
                        .events
                        .equatorial_surface()
                        .is_some_and(|surface| surface.contains(self.tracer.spacetime, root.state))
                {
                    roots[root_count] = Some(root);
                    root_count += 1;
                }
            }
        }
        Ok(select_earliest_event(
            &roots,
            self.step_m,
            self.tracer.policy.event_tie_tolerance_m(),
        ))
    }

    fn localize_event(
        &self,
        dense: &DenseOutput,
        kind: EventKind,
        start_value: f64,
        end_value: f64,
    ) -> Result<LocalizedRoot, GeometryError> {
        let root = localize_dense_root(
            dense,
            self.step_m,
            self.tracer.policy.event_affine_tolerance_m(),
            start_value,
            end_value,
            |state| self.event_value(kind, state),
        )?;
        let residual = self.event_value(kind, root.state)?.abs() / self.event_scale(kind);
        Ok(LocalizedRoot {
            kind,
            candidates: vec![kind],
            theta: root.theta,
            state: root.state,
            bracket_width_m: root.bracket_width_m,
            normalized_residual: residual,
        })
    }

    fn event_diagnostic(&self, root: &LocalizedRoot) -> LocalizedEvent {
        LocalizedEvent {
            kind: root.kind,
            candidates: root.candidates.clone(),
            affine_parameter_m: self.step_m.mul_add(root.theta, self.affine_parameter_m),
            bracket_width_m: root.bracket_width_m,
            normalized_residual: root.normalized_residual,
        }
    }

    fn event_is_installed(&self, kind: EventKind) -> bool {
        match kind {
            EventKind::SingularityGuard => true,
            EventKind::Horizon => self.tracer.spacetime.outer_horizon_radius().is_some(),
            EventKind::EquatorialSurface => self.tracer.events.equatorial_surface().is_some(),
            EventKind::Escape => self.tracer.events.escape_radius_m().is_some(),
        }
    }

    fn event_value(&self, kind: EventKind, state: GeodesicState) -> Result<f64, GeometryError> {
        event_value_for(self.tracer, kind, state)
    }

    fn event_scale(&self, kind: EventKind) -> f64 {
        match kind {
            EventKind::SingularityGuard => self.tracer.spacetime.mass_m().powi(4).max(1.0),
            EventKind::Horizon => self
                .tracer
                .spacetime
                .outer_horizon_radius()
                .unwrap_or(1.0)
                .max(1.0),
            EventKind::EquatorialSurface => self.tracer.spacetime.mass_m().max(1.0),
            EventKind::Escape => self.tracer.events.escape_radius_m().unwrap_or(1.0).max(1.0),
        }
    }

    fn record_turning_point(
        &mut self,
        dense: &DenseOutput,
        end_state: GeodesicState,
        maximum_theta: f64,
    ) {
        let traversal_sign = self.request.affine_direction.sign();
        let start_state = self.state;
        let start_velocity = self
            .evaluate_rhs(|spacetime| spacetime.radial_velocity(start_state))
            .map(|velocity| traversal_sign * velocity);
        let end_velocity = self
            .evaluate_rhs(|spacetime| spacetime.radial_velocity(end_state))
            .map(|velocity| traversal_sign * velocity);
        if let (Ok(start), Ok(end)) = (start_velocity, end_velocity)
            && start < 0.0
            && end >= 0.0
            && let Ok((theta, state)) = self.localize_turning_point(dense, start, end)
            && theta <= maximum_theta
            && let Ok(radius) = self.tracer.spacetime.radius(state.event())
        {
            self.turning_radius_m = Some(
                self.turning_radius_m
                    .map_or(radius, |minimum| minimum.min(radius)),
            );
        }
    }

    fn localize_turning_point(
        &mut self,
        dense: &DenseOutput,
        start_value: f64,
        end_value: f64,
    ) -> Result<(f64, GeodesicState), GeometryError> {
        let traversal_sign = self.request.affine_direction.sign();
        let root = localize_dense_root(
            dense,
            self.step_m,
            self.tracer.policy.event_affine_tolerance_m(),
            start_value,
            end_value,
            |state| {
                self.evaluate_rhs(|spacetime| spacetime.radial_velocity(state))
                    .map(|velocity| traversal_sign * velocity)
            },
        )?;
        Ok((root.theta, root.state))
    }

    fn commit_observables(&mut self, state: GeodesicState) {
        let components = state.components();
        let azimuth = components[2].atan2(components[1]);
        self.azimuth_advance_rad += wrapped_angle_difference(azimuth, self.previous_azimuth_rad);
        self.previous_azimuth_rad = azimuth;
        self.update_invariants(state);
    }

    fn update_invariants(&mut self, state: GeodesicState) {
        if let (Some(initial), Ok(current)) = (
            self.initial_invariants,
            self.tracer.spacetime.invariants(state),
        ) {
            self.drifts.update(initial, current);
        }
    }

    fn finish_failure(self, failure: NumericalFailure) -> ReferenceOutcome {
        let state = self.state;
        self.finish(Termination::NumericalFailure(failure), state, None)
    }

    fn finish(
        mut self,
        termination: Termination,
        state: GeodesicState,
        event: Option<LocalizedEvent>,
    ) -> ReferenceOutcome {
        let escape_direction_xyz = self.escape_direction_xyz(termination, state);
        ReferenceOutcome {
            input_id: self.request.input_id,
            policy_id: self.tracer.policy.id(),
            termination,
            state,
            affine_parameter_m: self.affine_parameter_m,
            event,
            escape_direction_xyz,
            turning_radius_m: self.turning_radius_m,
            azimuth_advance_rad: self.azimuth_advance_rad,
            travel_time_m: self.request.affine_direction.sign()
                * self.coordinate_time_delta_m.total(),
            surface_observable: None,
            diagnostics: TraceDiagnostics {
                accepted_steps: self.accepted_steps,
                rejected_steps: self.rejected_steps,
                rhs_evaluations: self.rhs_evaluations,
                minimum_step_m: self.minimum_step_m,
                maximum_step_m: self.maximum_step_m,
                maximum_null_residual: self.drifts.null,
                maximum_energy_drift: self.drifts.energy,
                maximum_angular_momentum_z_drift: self.drifts.angular_momentum_z,
                maximum_carter_drift: self.drifts.carter,
            },
        }
    }

    fn escape_direction_xyz(
        &mut self,
        termination: Termination,
        state: GeodesicState,
    ) -> Option<[f64; 3]> {
        if termination != Termination::Escape {
            return None;
        }
        let derivative = self
            .evaluate_rhs(|spacetime| spacetime.hamiltonian_rhs(state))
            .ok()?;
        let traversal_sign = self.request.affine_direction.sign();
        let direction = [
            traversal_sign * derivative[1],
            traversal_sign * derivative[2],
            traversal_sign * derivative[3],
        ];
        let norm = direction[0]
            .mul_add(
                direction[0],
                direction[1].mul_add(direction[1], direction[2] * direction[2]),
            )
            .sqrt();
        (norm > 0.0 && norm.is_finite()).then(|| direction.map(|component| component / norm))
    }

    fn evaluate_rhs<T>(
        &mut self,
        evaluate: impl FnOnce(KerrNewmanSpacetime) -> Result<T, GeometryError>,
    ) -> Result<T, GeometryError> {
        self.rhs_evaluations += 1;
        evaluate(self.tracer.spacetime)
    }
}

#[derive(Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    fn total(&self) -> f64 {
        self.sum + self.correction
    }
}

#[derive(Clone, Copy, Debug)]
struct EventArming {
    armed: [bool; 4],
}

impl EventArming {
    fn new(tracer: &GeodesicTracer, state: GeodesicState) -> Self {
        let band = tracer.policy.event_arming_band_m();
        Self {
            armed: EventKind::ordered().map(|kind| {
                event_value_for(tracer, kind, state)
                    .is_ok_and(|value| clears_arming_band(kind, value, band))
            }),
        }
    }

    const fn is_armed(self, kind: EventKind) -> bool {
        self.armed[kind.index()]
    }

    fn update(&mut self, tracer: &GeodesicTracer, state: GeodesicState) {
        let band = tracer.policy.event_arming_band_m();
        for kind in EventKind::ordered() {
            if !self.is_armed(kind)
                && event_value_for(tracer, kind, state)
                    .is_ok_and(|value| clears_arming_band(kind, value, band))
            {
                self.arm(kind);
            }
        }
    }

    const fn arm(&mut self, kind: EventKind) {
        self.armed[kind.index()] = true;
    }
}

const fn clears_arming_band(kind: EventKind, value: f64, band: f64) -> bool {
    match kind {
        EventKind::SingularityGuard | EventKind::Horizon => value > band,
        EventKind::EquatorialSurface => value.abs() > band,
        EventKind::Escape => escape_event_is_armed(value, band),
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct InvariantDrifts {
    null: f64,
    energy: f64,
    angular_momentum_z: f64,
    carter: f64,
}

impl InvariantDrifts {
    fn update(&mut self, initial: GeodesicInvariants, current: GeodesicInvariants) {
        self.null = self.null.max(current.normalized_null_residual());
        self.energy = self
            .energy
            .max(relative_drift(initial.energy(), current.energy()));
        self.angular_momentum_z = self.angular_momentum_z.max(relative_drift(
            initial.angular_momentum_z(),
            current.angular_momentum_z(),
        ));
        self.carter = self.carter.max(relative_drift(
            initial.carter_constant(),
            current.carter_constant(),
        ));
    }
}

#[derive(Clone, Debug)]
struct LocalizedRoot {
    kind: EventKind,
    candidates: Vec<EventKind>,
    theta: f64,
    state: GeodesicState,
    bracket_width_m: f64,
    normalized_residual: f64,
}

struct DenseRoot {
    theta: f64,
    state: GeodesicState,
    bracket_width_m: f64,
}

impl EventKind {
    const fn ordered() -> [Self; 4] {
        [
            Self::SingularityGuard,
            Self::Horizon,
            Self::EquatorialSurface,
            Self::Escape,
        ]
    }

    const fn index(self) -> usize {
        match self {
            Self::SingularityGuard => 0,
            Self::Horizon => 1,
            Self::EquatorialSurface => 2,
            Self::Escape => 3,
        }
    }
}

fn state_from_dense(dense: &DenseOutput, theta: f64) -> Result<GeodesicState, GeometryError> {
    GeodesicState::from_components(dense.evaluate(theta)).map_err(|_| GeometryError::NonFinite)
}

fn localize_dense_root(
    dense: &DenseOutput,
    step_m: f64,
    affine_tolerance_m: f64,
    start_value: f64,
    end_value: f64,
    mut value_at: impl FnMut(GeodesicState) -> Result<f64, GeometryError>,
) -> Result<DenseRoot, GeometryError> {
    let mut lower = 0.0;
    let mut upper = 1.0;
    let mut lower_value = start_value;
    let mut upper_value = end_value;
    for _ in 0..64 {
        if (upper - lower) * step_m.abs() <= affine_tolerance_m {
            break;
        }
        let middle = 0.5 * (lower + upper);
        let value = value_at(state_from_dense(dense, middle)?)?;
        if same_sign(lower_value, value) {
            lower = middle;
            lower_value = value;
        } else {
            upper = middle;
            upper_value = value;
        }
    }
    let theta = if lower_value.abs() <= upper_value.abs() {
        lower
    } else {
        upper
    };
    Ok(DenseRoot {
        theta,
        state: state_from_dense(dense, theta)?,
        bracket_width_m: (upper - lower) * step_m.abs(),
    })
}

fn event_value_for(
    tracer: &GeodesicTracer,
    kind: EventKind,
    state: GeodesicState,
) -> Result<f64, GeometryError> {
    let spacetime = tracer.spacetime;
    match kind {
        EventKind::SingularityGuard => {
            let mass_squared = spacetime.mass_m() * spacetime.mass_m();
            let guard = tracer.policy.singularity_guard_d_over_m4() * mass_squared * mass_squared;
            spacetime.singularity_guard_residual(state.event(), guard)
        }
        EventKind::Horizon => {
            Ok(spacetime.radius(state.event())?
                - spacetime.outer_horizon_radius().unwrap_or_default())
        }
        EventKind::EquatorialSurface => Ok(state.components()[3]),
        EventKind::Escape => {
            Ok(spacetime.radius(state.event())?
                - tracer.events.escape_radius_m().unwrap_or_default())
        }
    }
}

const fn termination_for_event(kind: EventKind) -> Termination {
    match kind {
        EventKind::SingularityGuard => Termination::SingularityGuard,
        EventKind::Horizon => Termination::HorizonCrossing,
        EventKind::EquatorialSurface => Termination::EquatorialSurface,
        EventKind::Escape => Termination::Escape,
    }
}

const fn crosses(kind: EventKind, start: f64, end: f64) -> bool {
    match kind {
        EventKind::SingularityGuard | EventKind::Horizon => start > 0.0 && end <= 0.0,
        EventKind::EquatorialSurface => (start < 0.0 && end >= 0.0) || (start > 0.0 && end <= 0.0),
        EventKind::Escape => start < 0.0 && end >= 0.0,
    }
}

const fn same_sign(left: f64, right: f64) -> bool {
    left.is_sign_negative() == right.is_sign_negative()
}

fn wrapped_angle_difference(current: f64, previous: f64) -> f64 {
    (current - previous + PI).rem_euclid(2.0 * PI) - PI
}

fn relative_drift(initial: f64, current: f64) -> f64 {
    (current - initial).abs() / initial.abs().max(1.0)
}

fn select_earliest_event(
    roots: &[Option<LocalizedRoot>; 4],
    step_m: f64,
    tie_tolerance_m: f64,
) -> Option<LocalizedRoot> {
    let mut first = roots
        .iter()
        .flatten()
        .min_by(|left, right| left.theta.total_cmp(&right.theta))?
        .clone();
    first.candidates = roots
        .iter()
        .flatten()
        .filter(|root| (root.theta - first.theta).abs() * step_m.abs() <= tie_tolerance_m)
        .map(|root| root.kind)
        .collect();
    first.candidates.sort_unstable();
    Some(first)
}

#[cfg(test)]
mod tests {
    use gravlume_domain::{GeodesicState, KerrNewmanSpacetime, KerrSchildChart};
    use proptest::prelude::*;

    use super::{
        EventArming, EventConfiguration, EventKind, GeodesicTrace, GeodesicTracer, ReferencePolicy,
        Termination, event_value_for, select_earliest_event,
    };
    use crate::{AffineDirection, TraceInputId};

    #[test]
    fn accepted_step_limit_is_a_typed_terminal_condition() {
        let spacetime = KerrNewmanSpacetime::new(1.0, 0.0, 0.0, KerrSchildChart::Ingoing)
            .expect("spacetime is valid");
        let state = GeodesicState::new([0.0, 50.0, 0.0, 0.0], [-1.0, -0.99, 0.1, 0.0])
            .expect("state is finite");
        let policy = ReferencePolicy::regular_v1().limited_to_one_step_for_test();
        let tracer = GeodesicTracer::new(spacetime, policy, EventConfiguration::horizon_only())
            .expect("mass is normalized");
        let outcome = tracer.trace(GeodesicTrace::new(
            TraceInputId::new("accepted-step-limit"),
            state,
            AffineDirection::Positive,
        ));

        assert_eq!(outcome.termination(), Termination::StepExhaustion);
        assert_eq!(outcome.diagnostics().accepted_steps(), 1);
    }

    #[test]
    fn singularity_guard_uses_the_unclamped_d_observable() {
        let spacetime = KerrNewmanSpacetime::new(1.0, 2.0, 0.0, KerrSchildChart::Ingoing)
            .expect("spacetime is valid");
        let policy = ReferencePolicy::regular_v1();
        let tracer = GeodesicTracer::new(spacetime, policy, EventConfiguration::horizon_only())
            .expect("mass is normalized");
        let guard_radius = policy.singularity_guard_d_over_m4().sqrt().sqrt();
        let event_value = |factor: f64| {
            let radius = factor * guard_radius;
            let x = radius.mul_add(radius, 4.0).sqrt();
            let state = GeodesicState::new([0.0, x, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0])
                .expect("state is finite");
            event_value_for(&tracer, EventKind::SingularityGuard, state)
                .expect("point is outside the ring")
        };

        assert!(event_value(1.1) > 0.0);
        assert!(event_value(0.9) < 0.0);
    }

    #[test]
    fn singularity_guard_arms_only_after_leaving_the_full_band() {
        let spacetime = KerrNewmanSpacetime::new(1.0, 2.0, 0.0, KerrSchildChart::Ingoing)
            .expect("spacetime is valid");
        let policy = ReferencePolicy::regular_v1();
        let tracer = GeodesicTracer::new(spacetime, policy, EventConfiguration::horizon_only())
            .expect("mass is normalized");
        let state_at_measure = |measure: f64| {
            let radius = measure.sqrt().sqrt();
            let x = radius.mul_add(radius, 4.0).sqrt();
            GeodesicState::new([0.0, x, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]).expect("state is finite")
        };
        let guard = policy.singularity_guard_d_over_m4();
        let band = policy.event_arming_band_m();
        let surface = state_at_measure(guard);
        let inside_band = state_at_measure(0.5_f64.mul_add(band, guard));
        let outside_band = state_at_measure(2.0_f64.mul_add(band, guard));
        let mut arming = EventArming::new(&tracer, surface);

        arming.update(&tracer, inside_band);
        assert!(!arming.is_armed(EventKind::SingularityGuard));
        arming.update(&tracer, outside_band);
        assert!(arming.is_armed(EventKind::SingularityGuard));
    }

    #[test]
    fn superextremal_ring_approach_stops_at_the_singularity_guard() {
        let spacetime = KerrNewmanSpacetime::new(1.0, 2.0, 0.0, KerrSchildChart::Ingoing)
            .expect("spacetime is valid");
        let state = GeodesicState::new([0.0, 2.1, 0.0, 0.0], [-1.0, -1.0, 0.0, 0.0])
            .expect("state is finite");
        let tracer = GeodesicTracer::new(
            spacetime,
            ReferencePolicy::regular_v1(),
            EventConfiguration::horizon_only(),
        )
        .expect("mass is normalized");
        let outcome = tracer.trace(GeodesicTrace::new(
            TraceInputId::new("superextremal-ring-approach"),
            state,
            AffineDirection::Positive,
        ));

        assert_eq!(outcome.termination(), Termination::SingularityGuard);
    }

    proptest! {
        #[test]
        fn same_step_event_selection_is_relative_to_the_global_affine_minimum(
            optional_events in any::<[bool; 3]>(),
            offsets in prop::array::uniform4(prop_oneof![Just(0.0), 0.0_f64..=2.0]),
            step_magnitude in 0.005_f64..=8.0,
            is_backward in any::<bool>(),
            tie_tolerance_m in 1.0e-12_f64..=1.0e-6,
        ) {
            let state = GeodesicState::new([0.0; 4], [-1.0, 0.0, 0.0, 0.0])
                .expect("state is finite");
            let kinds = EventKind::ordered();
            let step_m = if is_backward {
                -step_magnitude
            } else {
                step_magnitude
            };
            let roots = std::array::from_fn(|index| {
                let is_present = index == 0 || optional_events[index - 1];
                is_present.then(|| super::LocalizedRoot {
                    kind: kinds[index],
                    candidates: vec![kinds[index]],
                    theta: 0.25 + offsets[index] * tie_tolerance_m / step_magnitude,
                    state,
                    bracket_width_m: 1.0e-12,
                    normalized_residual: 0.0,
                })
            });
            let earliest_theta = roots
                .iter()
                .flatten()
                .map(|root| root.theta)
                .fold(f64::INFINITY, f64::min);
            let expected_kind = roots
                .iter()
                .flatten()
                .find(|root| root.theta.to_bits() == earliest_theta.to_bits())
                .expect("the singularity guard is always present")
                .kind;
            let expected_candidates = roots
                .iter()
                .flatten()
                .filter(|root| {
                    (root.theta - earliest_theta).abs() * step_magnitude <= tie_tolerance_m
                })
                .map(|root| root.kind)
                .collect::<Vec<_>>();

            let selected = select_earliest_event(&roots, step_m, tie_tolerance_m)
                .expect("the singularity guard is always present");

            prop_assert_eq!(selected.theta.to_bits(), earliest_theta.to_bits());
            prop_assert_eq!(selected.kind, expected_kind);
            prop_assert_eq!(selected.candidates, expected_candidates);
        }
    }
}

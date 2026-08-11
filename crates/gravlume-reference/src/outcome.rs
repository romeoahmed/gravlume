use gravlume_domain::{GeodesicState, GeometryError};

use crate::EventKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffineDirection {
    Positive,
    Negative,
}

impl AffineDirection {
    pub(super) const fn sign(self) -> f64 {
        match self {
            Self::Positive => 1.0,
            Self::Negative => -1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceRequest {
    pub(super) input_id: u64,
    pub(super) initial_state: GeodesicState,
    pub(super) affine_direction: AffineDirection,
}

impl TraceRequest {
    #[must_use]
    pub const fn new(
        input_id: u64,
        initial_state: GeodesicState,
        affine_direction: AffineDirection,
    ) -> Self {
        Self {
            input_id,
            initial_state,
            affine_direction,
        }
    }

    #[must_use]
    pub const fn input_id(self) -> u64 {
        self.input_id
    }

    #[must_use]
    pub const fn initial_state(self) -> GeodesicState {
        self.initial_state
    }

    #[must_use]
    pub const fn affine_direction(self) -> AffineDirection {
        self.affine_direction
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericalFailure {
    NonFinite,
    RingSingularity,
    ChartBoundary,
    InvalidDenominator,
    MinimumStep,
}

impl From<GeometryError> for NumericalFailure {
    fn from(error: GeometryError) -> Self {
        match error {
            GeometryError::NonFinite => Self::NonFinite,
            GeometryError::RingSingularity => Self::RingSingularity,
            GeometryError::ChartBoundary => Self::ChartBoundary,
            GeometryError::InvalidDenominator => Self::InvalidDenominator,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    HorizonCrossing,
    Escape,
    EquatorialSurface,
    SingularityGuard,
    StepExhaustion,
    RejectExhaustion,
    NumericalFailure(NumericalFailure),
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalizedEvent {
    pub(super) kind: EventKind,
    pub(super) candidates: Vec<EventKind>,
    pub(super) affine_parameter_m: f64,
    pub(super) bracket_width_m: f64,
    pub(super) normalized_residual: f64,
}

impl LocalizedEvent {
    #[must_use]
    pub const fn kind(&self) -> EventKind {
        self.kind
    }

    #[must_use]
    pub fn candidates(&self) -> &[EventKind] {
        &self.candidates
    }

    #[must_use]
    pub const fn affine_parameter_m(&self) -> f64 {
        self.affine_parameter_m
    }

    #[must_use]
    pub const fn bracket_width_m(&self) -> f64 {
        self.bracket_width_m
    }

    #[must_use]
    pub const fn normalized_residual(&self) -> f64 {
        self.normalized_residual
    }

    #[must_use]
    pub const fn is_ambiguous(&self) -> bool {
        self.candidates.len() > 1
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceDiagnostics {
    pub(super) accepted_steps: u64,
    pub(super) rejected_steps: u64,
    pub(super) rhs_evaluations: u64,
    pub(super) minimum_step_m: Option<f64>,
    pub(super) maximum_step_m: Option<f64>,
    pub(super) maximum_null_residual: f64,
    pub(super) maximum_energy_drift: f64,
    pub(super) maximum_angular_momentum_z_drift: f64,
    pub(super) maximum_carter_drift: f64,
}

impl TraceDiagnostics {
    #[must_use]
    pub const fn accepted_steps(self) -> u64 {
        self.accepted_steps
    }

    #[must_use]
    pub const fn rejected_steps(self) -> u64 {
        self.rejected_steps
    }

    #[must_use]
    pub const fn rhs_evaluations(self) -> u64 {
        self.rhs_evaluations
    }

    #[must_use]
    pub const fn minimum_step_m(self) -> Option<f64> {
        self.minimum_step_m
    }

    #[must_use]
    pub const fn maximum_step_m(self) -> Option<f64> {
        self.maximum_step_m
    }

    #[must_use]
    pub const fn maximum_null_residual(self) -> f64 {
        self.maximum_null_residual
    }

    #[must_use]
    pub const fn maximum_energy_drift(self) -> f64 {
        self.maximum_energy_drift
    }

    #[must_use]
    pub const fn maximum_angular_momentum_z_drift(self) -> f64 {
        self.maximum_angular_momentum_z_drift
    }

    #[must_use]
    pub const fn maximum_carter_drift(self) -> f64 {
        self.maximum_carter_drift
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceOutcome {
    pub(super) input_id: u64,
    pub(super) policy_id: &'static str,
    pub(super) termination: Termination,
    pub(super) state: GeodesicState,
    pub(super) affine_parameter_m: f64,
    pub(super) event: Option<LocalizedEvent>,
    pub(super) turning_radius_m: Option<f64>,
    pub(super) azimuth_advance_rad: f64,
    pub(super) travel_time_m: f64,
    pub(super) diagnostics: TraceDiagnostics,
}

impl ReferenceOutcome {
    #[must_use]
    pub const fn input_id(&self) -> u64 {
        self.input_id
    }

    #[must_use]
    pub const fn policy_id(&self) -> &'static str {
        self.policy_id
    }

    #[must_use]
    pub const fn termination(&self) -> Termination {
        self.termination
    }

    #[must_use]
    pub const fn state(&self) -> GeodesicState {
        self.state
    }

    #[must_use]
    pub const fn affine_parameter_m(&self) -> f64 {
        self.affine_parameter_m
    }

    #[must_use]
    pub const fn event(&self) -> Option<&LocalizedEvent> {
        self.event.as_ref()
    }

    #[must_use]
    pub const fn turning_radius_m(&self) -> Option<f64> {
        self.turning_radius_m
    }

    #[must_use]
    pub const fn azimuth_advance_rad(&self) -> f64 {
        self.azimuth_advance_rad
    }

    #[must_use]
    pub const fn travel_time_m(&self) -> f64 {
        self.travel_time_m
    }

    #[must_use]
    pub const fn diagnostics(&self) -> TraceDiagnostics {
        self.diagnostics
    }
}

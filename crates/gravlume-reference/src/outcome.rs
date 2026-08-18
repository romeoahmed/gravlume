use std::{fmt, sync::Arc};

use gravlume_domain::{GeodesicState, GeometryError, KerrNewmanSpacetime};

use crate::{EventKind, SurfaceObservable, events::EventConfiguration};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CanonicalTraceInput {
    spacetime: [u64; 3],
    state: [u64; 8],
    affine_direction: AffineDirection,
    events: [u64; 5],
}

/// Stable logical identity shared by policy variants of exactly the same effective trace input.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraceInputId {
    logical: Arc<str>,
    canonical: Option<Arc<CanonicalTraceInput>>,
}

impl TraceInputId {
    /// Creates a caller-defined logical label.
    ///
    /// The reference tracer binds this label to the canonical spacetime, state, affine direction,
    /// and event configuration before producing an outcome.
    #[must_use]
    pub fn new(value: impl Into<Arc<str>>) -> Self {
        Self {
            logical: value.into(),
            canonical: None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.logical
    }

    pub(super) fn bind(
        &self,
        spacetime: KerrNewmanSpacetime,
        state: GeodesicState,
        affine_direction: AffineDirection,
        events: EventConfiguration,
    ) -> Self {
        let canonical = CanonicalTraceInput {
            spacetime: [
                spacetime.mass_m().to_bits(),
                spacetime.spin_m().to_bits(),
                spacetime.charge_m().to_bits(),
            ],
            state: state.components().map(f64::to_bits),
            affine_direction,
            events: events.canonical_bits(),
        };
        if self.canonical.as_deref() == Some(&canonical) {
            return self.clone();
        }
        Self {
            logical: Arc::clone(&self.logical),
            canonical: Some(Arc::new(canonical)),
        }
    }

    pub(super) fn logical(&self) -> Self {
        Self::new(Arc::clone(&self.logical))
    }
}

impl fmt::Display for TraceInputId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.logical)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeodesicTrace {
    pub(super) input_id: TraceInputId,
    pub(super) initial_state: GeodesicState,
    pub(super) affine_direction: AffineDirection,
}

impl GeodesicTrace {
    #[must_use]
    pub const fn new(
        input_id: TraceInputId,
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
    pub const fn input_id(&self) -> &TraceInputId {
        &self.input_id
    }

    #[must_use]
    pub const fn initial_state(&self) -> GeodesicState {
        self.initial_state
    }

    #[must_use]
    pub const fn affine_direction(&self) -> AffineDirection {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PolarSide {
    Negative,
    Equatorial,
    Positive,
}

impl PolarSide {
    pub(super) const fn from_height(height_m: f64) -> Self {
        if height_m > 0.0 {
            Self::Positive
        } else if height_m < 0.0 {
            Self::Negative
        } else {
            Self::Equatorial
        }
    }
}

/// Discrete path semantics that must agree before source-space differencing or reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TraceBranchKey {
    initial_polar_side: PolarSide,
    radial_turnings: u32,
    equatorial_crossings: u32,
    azimuth_winding: i32,
}

impl TraceBranchKey {
    pub(super) const fn new(
        initial_polar_side: PolarSide,
        radial_turnings: u32,
        equatorial_crossings: u32,
        azimuth_winding: i32,
    ) -> Self {
        Self {
            initial_polar_side,
            radial_turnings,
            equatorial_crossings,
            azimuth_winding,
        }
    }

    #[must_use]
    pub const fn initial_polar_side(self) -> PolarSide {
        self.initial_polar_side
    }

    #[must_use]
    pub const fn radial_turnings(self) -> u32 {
        self.radial_turnings
    }

    #[must_use]
    pub const fn equatorial_crossings(self) -> u32 {
        self.equatorial_crossings
    }

    #[must_use]
    pub const fn azimuth_winding(self) -> i32 {
        self.azimuth_winding
    }
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

/// Availability of the normalized coordinate traversal direction at an escape event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EscapeDirection {
    Resolved([f64; 3]),
    Unavailable,
}

impl EscapeDirection {
    #[must_use]
    pub const fn xyz(self) -> Option<[f64; 3]> {
        match self {
            Self::Resolved(direction) => Some(direction),
            Self::Unavailable => None,
        }
    }
}

/// Terminal classification together with exactly the evidence available for that branch.
#[derive(Clone, Debug, PartialEq)]
pub enum ReferenceTerminal {
    HorizonCrossing {
        event: LocalizedEvent,
    },
    Escape {
        event: LocalizedEvent,
        direction: EscapeDirection,
    },
    EquatorialSurface {
        event: LocalizedEvent,
    },
    ObservedSurface {
        event: LocalizedEvent,
        observable: SurfaceObservable,
    },
    SingularityGuard {
        event: LocalizedEvent,
    },
    StepExhaustion,
    RejectExhaustion,
    NumericalFailure(NumericalFailure),
}

impl ReferenceTerminal {
    #[must_use]
    pub const fn termination(&self) -> Termination {
        match self {
            Self::HorizonCrossing { .. } => Termination::HorizonCrossing,
            Self::Escape { .. } => Termination::Escape,
            Self::EquatorialSurface { .. } | Self::ObservedSurface { .. } => {
                Termination::EquatorialSurface
            }
            Self::SingularityGuard { .. } => Termination::SingularityGuard,
            Self::StepExhaustion => Termination::StepExhaustion,
            Self::RejectExhaustion => Termination::RejectExhaustion,
            Self::NumericalFailure(failure) => Termination::NumericalFailure(*failure),
        }
    }

    #[must_use]
    pub const fn event(&self) -> Option<&LocalizedEvent> {
        match self {
            Self::HorizonCrossing { event }
            | Self::Escape { event, .. }
            | Self::EquatorialSurface { event }
            | Self::ObservedSurface { event, .. }
            | Self::SingularityGuard { event } => Some(event),
            Self::StepExhaustion | Self::RejectExhaustion | Self::NumericalFailure(_) => None,
        }
    }

    #[must_use]
    pub const fn escape_direction(&self) -> Option<EscapeDirection> {
        match self {
            Self::Escape { direction, .. } => Some(*direction),
            Self::HorizonCrossing { .. }
            | Self::EquatorialSurface { .. }
            | Self::ObservedSurface { .. }
            | Self::SingularityGuard { .. }
            | Self::StepExhaustion
            | Self::RejectExhaustion
            | Self::NumericalFailure(_) => None,
        }
    }

    #[must_use]
    pub const fn surface_observable(&self) -> Option<SurfaceObservable> {
        match self {
            Self::ObservedSurface { observable, .. } => Some(*observable),
            Self::HorizonCrossing { .. }
            | Self::Escape { .. }
            | Self::EquatorialSurface { .. }
            | Self::SingularityGuard { .. }
            | Self::StepExhaustion
            | Self::RejectExhaustion
            | Self::NumericalFailure(_) => None,
        }
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
    pub(super) input_id: TraceInputId,
    pub(super) policy_id: &'static str,
    pub(super) terminal: ReferenceTerminal,
    pub(super) state: GeodesicState,
    pub(super) affine_parameter_m: f64,
    pub(super) turning_radius_m: Option<f64>,
    pub(super) azimuth_advance_rad: f64,
    pub(super) travel_time_m: f64,
    pub(super) branch_key: TraceBranchKey,
    pub(super) diagnostics: TraceDiagnostics,
}

impl ReferenceOutcome {
    #[must_use]
    pub const fn input_id(&self) -> &TraceInputId {
        &self.input_id
    }

    #[must_use]
    pub const fn policy_id(&self) -> &'static str {
        self.policy_id
    }

    #[must_use]
    pub const fn termination(&self) -> Termination {
        self.terminal.termination()
    }

    #[must_use]
    pub const fn terminal(&self) -> &ReferenceTerminal {
        &self.terminal
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
    pub const fn turning_radius_m(&self) -> Option<f64> {
        self.turning_radius_m
    }

    #[must_use]
    pub const fn azimuth_advance_rad(&self) -> f64 {
        self.azimuth_advance_rad
    }

    /// Returns the coordinate-time duration along the actual affine traversal.
    #[must_use]
    pub const fn travel_time_m(&self) -> f64 {
        self.travel_time_m
    }

    #[must_use]
    pub const fn branch_key(&self) -> TraceBranchKey {
        self.branch_key
    }

    #[must_use]
    pub const fn diagnostics(&self) -> TraceDiagnostics {
        self.diagnostics
    }
}

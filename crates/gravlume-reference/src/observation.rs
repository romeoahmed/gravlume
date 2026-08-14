use gravlume_domain::{
    ImageSample, InitialViewRay, KerrNewmanSpacetime, Observation, ValidationReport,
};

use crate::{
    AffineDirection, EquatorialCircularSurface, EventConfiguration, EventConfigurationError,
    GeodesicTrace, GeodesicTracer, ReferenceOutcome, ReferencePolicy, SurfaceObservableError,
    Termination, TraceInputId,
};

#[derive(Clone, Debug)]
pub struct ObservationTrace {
    input_id: TraceInputId,
    spacetime: KerrNewmanSpacetime,
    initial_ray: InitialViewRay,
    policy: ReferencePolicy,
    equatorial_circular_surface: Option<EquatorialCircularSurface>,
}

impl ObservationTrace {
    /// Binds a stable logical identity and sample to the observation, then resolves its initial ray.
    ///
    /// # Errors
    ///
    /// Rejects a sample that is invalid for the request observation's view.
    pub fn new(
        input_id: TraceInputId,
        observation: &Observation,
        sample: ImageSample,
        policy: ReferencePolicy,
    ) -> Result<Self, ValidationReport> {
        let initial_ray = observation.initial_ray(sample)?;
        Ok(Self::from_initial_ray(
            input_id,
            *observation.scene().spacetime(),
            initial_ray,
            policy,
        ))
    }

    pub(crate) const fn from_initial_ray(
        input_id: TraceInputId,
        spacetime: KerrNewmanSpacetime,
        initial_ray: InitialViewRay,
        policy: ReferencePolicy,
    ) -> Self {
        Self {
            input_id,
            spacetime,
            initial_ray,
            policy,
            equatorial_circular_surface: None,
        }
    }

    /// Requests the physical observables of the first hit on a prograde circular surface.
    #[must_use]
    pub const fn with_equatorial_circular_surface(
        mut self,
        surface: EquatorialCircularSurface,
    ) -> Self {
        self.equatorial_circular_surface = Some(surface);
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ObservationTracer {
    _private: (),
}

impl ObservationTracer {
    #[must_use]
    pub const fn baseline_v1() -> Self {
        Self { _private: () }
    }

    /// Traces an image sample backward while preserving future-directed photon momentum.
    ///
    /// # Errors
    ///
    /// Returns an error if the observation is not normalized for reference-v1. Numerical
    /// integration failures are successful, typed outcomes.
    pub fn trace(
        &self,
        request: ObservationTrace,
    ) -> Result<ReferenceOutcome, ObservationTraceError> {
        let ObservationTrace {
            input_id,
            spacetime,
            initial_ray,
            policy,
            equatorial_circular_surface,
        } = request;
        if (initial_ray.observer_frequency() - 1.0).abs() > 32.0 * f64::EPSILON {
            return Err(ObservationTraceError::NonNormalizedReferenceInput);
        }
        let mut events = EventConfiguration::observation_baseline_v1();
        if let Some(surface) = equatorial_circular_surface {
            events = events
                .with_equatorial_surface(surface.inner_radius_m(), surface.outer_radius_m())?;
        }
        let tracer = GeodesicTracer::new(spacetime, policy, events)
            .map_err(|_| ObservationTraceError::NonNormalizedReferenceInput)?;
        let observer_frequency = initial_ray.observer_frequency();
        let mut outcome = tracer.trace(GeodesicTrace::new(
            input_id,
            initial_ray.state(),
            AffineDirection::Negative,
        ));
        if let Some(surface) = equatorial_circular_surface
            && outcome.termination() == Termination::EquatorialSurface
        {
            outcome.surface_observable =
                Some(surface.observable_at(spacetime, outcome.state(), observer_frequency)?);
        }
        Ok(outcome)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ObservationTraceError {
    #[error("reference-v1 observation inputs must be normalized to M = omega = 1")]
    NonNormalizedReferenceInput,
    #[error(transparent)]
    EventConfiguration(#[from] EventConfigurationError),
    #[error(transparent)]
    SurfaceObservable(#[from] SurfaceObservableError),
}

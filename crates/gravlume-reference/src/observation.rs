use gravlume_domain::{
    ImageSample, InitialViewRay, KerrNewmanSpacetime, Observation, SceneRadiance, ValidationReport,
};

use crate::{
    AffineDirection, EventConfiguration, EventConfigurationError, GeodesicTrace, GeodesicTracer,
    ReferenceOutcome, ReferencePolicy, ReferenceTerminal, SurfaceObservableError, TraceInputId,
    surface::observable_at,
};

#[derive(Clone, Debug)]
pub struct ObservationTrace {
    input_id: TraceInputId,
    spacetime: KerrNewmanSpacetime,
    initial_ray: InitialViewRay,
    policy: ReferencePolicy,
    radiance: SceneRadiance,
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
        Ok(Self {
            input_id,
            spacetime: *observation.scene().spacetime(),
            initial_ray,
            policy,
            radiance: observation.scene().radiance(),
        })
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
            radiance,
        } = request;
        if (initial_ray.observer_frequency() - 1.0).abs() > 32.0 * f64::EPSILON {
            return Err(ObservationTraceError::NonNormalizedReferenceInput);
        }
        let mut events = EventConfiguration::observation_baseline_v1();
        let surface = match radiance {
            SceneRadiance::AnalyticSky => None,
            SceneRadiance::EquatorialSurface(surface) => {
                let emitter = surface.emitter();
                events = events
                    .with_equatorial_surface(emitter.inner_radius_m(), emitter.outer_radius_m())?;
                Some(surface)
            }
        };
        let tracer = GeodesicTracer::new(spacetime, policy, events)
            .map_err(|_| ObservationTraceError::NonNormalizedReferenceInput)?;
        let observer_frequency = initial_ray.observer_frequency();
        let mut outcome = tracer.trace(GeodesicTrace::new(
            input_id,
            initial_ray.state(),
            AffineDirection::Negative,
        ));
        let Some(surface) = surface else {
            return Ok(outcome);
        };
        let ReferenceTerminal::EquatorialSurface { event } = outcome.terminal() else {
            return Ok(outcome);
        };
        let event = event.clone();
        let observable = observable_at(surface, spacetime, outcome.state(), observer_frequency)?;
        outcome.terminal = ReferenceTerminal::ObservedSurface { event, observable };
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

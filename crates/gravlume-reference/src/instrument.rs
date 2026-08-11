use std::sync::Arc;

use gravlume_domain::{Observation, ViewportSample};

use crate::{
    AffineDirection, EventConfiguration, ReferenceOutcome, ReferencePolicy, ReferenceTracer,
    TraceRequest,
};

#[derive(Clone, Debug)]
pub struct ReferenceRequest {
    observation: Arc<Observation>,
    sample: ViewportSample,
    policy: ReferencePolicy,
}

impl ReferenceRequest {
    #[must_use]
    pub const fn new(
        observation: Arc<Observation>,
        sample: ViewportSample,
        policy: ReferencePolicy,
    ) -> Self {
        Self {
            observation,
            sample,
            policy,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReferenceInstrument {
    explicit_events: Option<EventConfiguration>,
}

impl ReferenceInstrument {
    #[must_use]
    pub const fn baseline_v1() -> Self {
        Self {
            explicit_events: None,
        }
    }

    #[must_use]
    pub const fn with_events(events: EventConfiguration) -> Self {
        Self {
            explicit_events: Some(events),
        }
    }

    /// Traces a viewport sample backward while preserving future-directed photon momentum.
    ///
    /// # Errors
    ///
    /// Returns an error only if an impossible internal initial-ray or baseline event invariant is
    /// observed. Numerical integration failures are successful, typed outcomes.
    pub fn trace(
        &self,
        request: ReferenceRequest,
    ) -> Result<ReferenceOutcome, ReferenceRuntimeError> {
        let ReferenceRequest {
            observation,
            sample,
            policy,
        } = request;
        let initial_ray = observation.initial_ray(sample);
        if !initial_ray.is_future_directed()
            || !initial_ray.normalized_null_residual().is_finite()
            || initial_ray.normalized_null_residual() > 2.0e-12
        {
            return Err(ReferenceRuntimeError::InvalidInitialRay);
        }
        let spacetime = *observation.scene().spacetime();
        if (spacetime.mass_m() - 1.0).abs() > 32.0 * f64::EPSILON
            || (initial_ray.observer_frequency() - 1.0).abs() > 32.0 * f64::EPSILON
        {
            return Err(ReferenceRuntimeError::NonNormalizedReferenceInput);
        }
        let events = match self.explicit_events {
            Some(events) => events,
            None => EventConfiguration::with_escape_radius(200.0)
                .map_err(|_| ReferenceRuntimeError::InvalidBaselineEvents)?,
        };
        let tracer = ReferenceTracer::new(spacetime, policy, events)
            .map_err(|_| ReferenceRuntimeError::NonNormalizedReferenceInput)?;
        Ok(tracer.trace(TraceRequest::new(
            0,
            initial_ray.state(),
            AffineDirection::Negative,
        )))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReferenceRuntimeError {
    #[error("validated observation produced an invalid initial ray")]
    InvalidInitialRay,
    #[error("baseline event configuration could not be constructed")]
    InvalidBaselineEvents,
    #[error("reference-v1 observation inputs must be normalized to M = omega = 1")]
    NonNormalizedReferenceInput,
}

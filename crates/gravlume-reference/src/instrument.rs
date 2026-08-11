use std::sync::Arc;

use gravlume_domain::{InitialViewRay, Observation, ValidationReport, ViewportSample};

use crate::{
    AffineDirection, EventConfiguration, ReferenceOutcome, ReferencePolicy, ReferenceTracer,
    TraceInputId, TraceRequest,
};

#[derive(Clone, Debug)]
pub struct ReferenceRequest {
    input_id: TraceInputId,
    observation: Arc<Observation>,
    initial_ray: InitialViewRay,
    policy: ReferencePolicy,
}

impl ReferenceRequest {
    /// Binds a stable logical identity and sample to the observation, then resolves its initial ray.
    ///
    /// # Errors
    ///
    /// Rejects a sample that is invalid for the request observation's projection.
    pub fn new(
        input_id: TraceInputId,
        observation: Arc<Observation>,
        sample: ViewportSample,
        policy: ReferencePolicy,
    ) -> Result<Self, ValidationReport> {
        let initial_ray = observation.initial_ray(sample)?;
        Ok(Self {
            input_id,
            observation,
            initial_ray,
            policy,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReferenceInstrument {
    events: EventConfiguration,
}

impl Default for ReferenceInstrument {
    fn default() -> Self {
        Self::baseline_v1()
    }
}

impl ReferenceInstrument {
    #[must_use]
    pub const fn baseline_v1() -> Self {
        Self {
            events: EventConfiguration::observation_baseline_v1(),
        }
    }

    #[must_use]
    pub const fn with_events(events: EventConfiguration) -> Self {
        Self { events }
    }

    /// Traces a viewport sample backward while preserving future-directed photon momentum.
    ///
    /// # Errors
    ///
    /// Returns an error if an impossible internal initial-ray invariant is observed or the
    /// observation is not normalized for reference-v1. Numerical integration failures are
    /// successful, typed outcomes.
    pub fn trace(
        &self,
        request: ReferenceRequest,
    ) -> Result<ReferenceOutcome, ReferenceRuntimeError> {
        let ReferenceRequest {
            input_id,
            observation,
            initial_ray,
            policy,
        } = request;
        let spacetime = *observation.scene().spacetime();
        if spacetime.mass_m().to_bits() != 1.0_f64.to_bits()
            || (initial_ray.observer_frequency() - 1.0).abs() > 32.0 * f64::EPSILON
        {
            return Err(ReferenceRuntimeError::NonNormalizedReferenceInput);
        }
        let tracer = ReferenceTracer::new(spacetime, policy, self.events)
            .map_err(|_| ReferenceRuntimeError::NonNormalizedReferenceInput)?;
        Ok(tracer.trace(TraceRequest::new(
            input_id,
            initial_ray.state(),
            AffineDirection::Negative,
        )))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReferenceRuntimeError {
    #[error("reference-v1 observation inputs must be normalized to M = omega = 1")]
    NonNormalizedReferenceInput,
}

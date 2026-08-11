use gravlume_domain::{
    InitialViewRay, KerrNewmanSpacetime, Observation, ValidationReport, ViewportSample,
};

use crate::{
    AffineDirection, EventConfiguration, ReferenceOutcome, ReferencePolicy, ReferenceTracer,
    TraceInputId, TraceRequest,
};

#[derive(Clone, Debug)]
pub struct ReferenceRequest {
    input_id: TraceInputId,
    spacetime: KerrNewmanSpacetime,
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
        observation: &Observation,
        sample: ViewportSample,
        policy: ReferencePolicy,
    ) -> Result<Self, ValidationReport> {
        let initial_ray = observation.initial_ray(sample)?;
        Ok(Self {
            input_id,
            spacetime: *observation.scene().spacetime(),
            initial_ray,
            policy,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReferenceInstrument {
    _private: (),
}

impl ReferenceInstrument {
    #[must_use]
    pub const fn baseline_v1() -> Self {
        Self { _private: () }
    }

    /// Traces a viewport sample backward while preserving future-directed photon momentum.
    ///
    /// # Errors
    ///
    /// Returns an error if the observation is not normalized for reference-v1. Numerical
    /// integration failures are successful, typed outcomes.
    pub fn trace(
        &self,
        request: ReferenceRequest,
    ) -> Result<ReferenceOutcome, ReferenceRuntimeError> {
        let ReferenceRequest {
            input_id,
            spacetime,
            initial_ray,
            policy,
        } = request;
        if (initial_ray.observer_frequency() - 1.0).abs() > 32.0 * f64::EPSILON {
            return Err(ReferenceRuntimeError::NonNormalizedReferenceInput);
        }
        let tracer = ReferenceTracer::new(
            spacetime,
            policy,
            EventConfiguration::observation_baseline_v1(),
        )
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

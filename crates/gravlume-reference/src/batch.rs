use std::num::NonZeroUsize;

use rayon::prelude::*;

use crate::{GeodesicTrace, GeodesicTracer, ReferenceOutcome};

const MAX_REFERENCE_THREADS: usize = 256;

pub struct GeodesicBatch {
    pool: rayon::ThreadPool,
}

impl GeodesicBatch {
    /// Builds a dedicated reference-computation pool.
    ///
    /// # Errors
    ///
    /// Returns an error if Rayon cannot create the requested worker threads.
    pub fn new(thread_count: NonZeroUsize) -> Result<Self, GeodesicBatchError> {
        if thread_count.get() > MAX_REFERENCE_THREADS {
            return Err(GeodesicBatchError::TooManyThreads);
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count.get())
            .build()
            .map_err(|_| GeodesicBatchError::BuildFailed)?;
        Ok(Self { pool })
    }

    #[must_use]
    pub fn trace_ordered(
        &self,
        tracer: &GeodesicTracer,
        requests: &[GeodesicTrace],
    ) -> Vec<ReferenceOutcome> {
        self.pool.install(|| {
            requests
                .par_iter()
                .cloned()
                .map(|request| tracer.trace(request))
                .collect()
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GeodesicBatchError {
    #[error("the requested reference pool exceeds the 256-thread safety limit")]
    TooManyThreads,
    #[error("the dedicated Rayon reference pool could not be built")]
    BuildFailed,
}

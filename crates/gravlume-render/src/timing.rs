use std::sync::mpsc::{self, TryRecvError};

use num_traits::ToPrimitive as _;

const QUERY_COUNT: u32 = 2;

fn query_bytes() -> u64 {
    u64::from(QUERY_COUNT) * u64::from(wgpu::QUERY_SIZE)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QueryTicks {
    beginning: u64,
    end: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimingSample {
    compute_ms: f64,
}

impl TimingSample {
    fn from_ticks(ticks: QueryTicks, timestamp_period_ns: f32) -> Self {
        let elapsed_ticks = ticks.end.saturating_sub(ticks.beginning);
        let milliseconds_per_tick = f64::from(timestamp_period_ns) / 1_000_000.0;
        Self {
            compute_ms: elapsed_ticks.to_f64().unwrap_or(f64::INFINITY) * milliseconds_per_tick,
        }
    }

    pub(crate) const fn compute_ms(self) -> f64 {
        self.compute_ms
    }
}

const fn decode_query_ticks(bytes: &[u8]) -> Option<QueryTicks> {
    let (encoded_ticks, []) = bytes.as_chunks::<{ size_of::<u64>() }>() else {
        return None;
    };
    let [beginning, end] = encoded_ticks else {
        return None;
    };
    Some(QueryTicks {
        beginning: u64::from_le_bytes(*beginning),
        end: u64::from_le_bytes(*end),
    })
}

#[derive(Debug, thiserror::Error)]
pub enum TimingError {
    #[error("a GPU timestamp readback is already pending")]
    ReadbackAlreadyPending,
    #[error("non-blocking GPU poll failed: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("GPU timestamp readback mapping failed: {0}")]
    Map(#[from] wgpu::BufferAsyncError),
    #[error("GPU timestamp mapped range was unavailable: {0}")]
    MappedRange(#[from] wgpu::MapRangeError),
    #[error("GPU timestamp callback channel disconnected")]
    CallbackDisconnected,
    #[error("GPU timestamp readback had an invalid byte count")]
    InvalidReadback,
}

struct PendingReadback<C> {
    receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    context: C,
}

pub struct GpuTimings<C> {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    pending: Option<PendingReadback<C>>,
}

impl<C> GpuTimings<C> {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame trace timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_COUNT,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame timestamp resolve buffer"),
            size: query_bytes(),
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame timestamp readback buffer"),
            size: query_bytes(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            pending: None,
        }
    }

    pub(crate) const fn capture_available(&self) -> bool {
        self.pending.is_none()
    }

    pub(crate) const fn trace_writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    pub(crate) fn encode_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        context: C,
    ) -> Result<(), TimingError> {
        if self.pending.is_some() {
            return Err(TimingError::ReadbackAlreadyPending);
        }
        encoder.resolve_query_set(&self.query_set, 0..QUERY_COUNT, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            query_bytes(),
        );

        let (sender, receiver) = mpsc::channel();
        // Scheduling map on the producing encoder makes copy completion and mapping one ordered
        // submission lifecycle. The callback is driven by `Device::poll` after queue submission.
        // Source: https://docs.rs/wgpu/30.0.1/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit
        encoder.map_buffer_on_submit(
            &self.readback_buffer,
            wgpu::MapMode::Read,
            ..,
            move |result| {
                if sender.send(result).is_err() {
                    tracing::debug!("timestamp callback receiver dropped");
                }
            },
        );
        self.pending = Some(PendingReadback { receiver, context });
        Ok(())
    }

    pub(crate) fn poll(
        &mut self,
        device: &wgpu::Device,
        timestamp_period_ns: f32,
    ) -> Result<Option<(C, TimingSample)>, TimingError> {
        if self.pending.is_none() {
            return Ok(None);
        }

        device.poll(wgpu::PollType::Poll)?;
        let Some(pending) = self.pending.take() else {
            return Ok(None);
        };
        // Only a successful callback grants a mapped range that must later be unmapped.
        // Source: https://docs.rs/wgpu/30.0.1/wgpu/struct.Buffer.html#method.map_async
        match pending.receiver.try_recv() {
            Ok(Ok(())) => {
                let ticks = self.read_ticks();
                self.readback_buffer.unmap();
                let sample = TimingSample::from_ticks(ticks?, timestamp_period_ns);
                Ok(Some((pending.context, sample)))
            }
            Ok(Err(error)) => Err(error.into()),
            Err(TryRecvError::Empty) => {
                self.pending = Some(pending);
                Ok(None)
            }
            Err(TryRecvError::Disconnected) => Err(TimingError::CallbackDisconnected),
        }
    }

    fn read_ticks(&self) -> Result<QueryTicks, TimingError> {
        let mapped = self.readback_buffer.get_mapped_range(..)?;
        let ticks = decode_query_ticks(&mapped).ok_or(TimingError::InvalidReadback);
        drop(mapped);
        ticks
    }

    pub(crate) const fn has_pending_readback(&self) -> bool {
        self.pending.is_some()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use approx::relative_eq;
    use num_traits::ToPrimitive as _;
    use proptest::prelude::*;

    use super::{GpuTimings, PendingReadback, QueryTicks, TimingError, TimingSample, query_bytes};
    use crate::error::GpuErrorScopes;

    proptest! {
        #[test]
        fn trace_timestamp_pair_roundtrips(ticks in any::<[u64; 2]>()) {
            let bytes: Vec<u8> = ticks.into_iter().flat_map(u64::to_le_bytes).collect();
            prop_assert_eq!(
                super::decode_query_ticks(&bytes),
                Some(QueryTicks { beginning: ticks[0], end: ticks[1] })
            );
        }

        #[test]
        fn timestamp_decoder_rejects_incorrect_lengths_around_the_pair_boundary(
            length in (0_usize..=64).prop_filter(
                "the valid timestamp-pair byte count is tested separately",
                |length| *length != usize::try_from(query_bytes()).expect("query bytes fit usize"),
            ),
        ) {
            prop_assert_eq!(super::decode_query_ticks(&vec![0; length]), None);
        }

        #[test]
        fn trace_duration_is_saturating_and_nonnegative(
            beginning: u64,
            end: u64,
            timestamp_period_ns in f32::MIN_POSITIVE..=1_000_000.0_f32,
        ) {
            let actual = TimingSample::from_ticks(
                QueryTicks { beginning, end },
                timestamp_period_ns,
            ).compute_ms();
            let expected = end
                .saturating_sub(beginning)
                .to_f64()
                .expect("u64 converts to finite f64")
                * f64::from(timestamp_period_ns)
                / 1_000_000.0;

            prop_assert!(actual.is_finite());
            prop_assert!(actual >= 0.0);
            prop_assert!(relative_eq!(
                actual,
                expected,
                epsilon = f64::EPSILON,
                max_relative = 4.0 * f64::EPSILON,
            ));
        }
    }

    #[test]
    fn map_failure_does_not_emit_a_secondary_unmap_validation_error() {
        let gpu = crate::test_device::native_gpu();
        let mut timings = GpuTimings::new(&gpu.device);
        let (sender, receiver) = mpsc::channel();
        sender
            .send(Err(wgpu::BufferAsyncError))
            .expect("synthetic map completion is delivered");
        timings.pending = Some(PendingReadback {
            receiver,
            context: 17_u64,
        });
        let scopes = GpuErrorScopes::push(&gpu.device);

        assert!(matches!(
            timings.poll(&gpu.device, gpu.queue.get_timestamp_period()),
            Err(TimingError::Map(_))
        ));
        let secondary_error = pollster::block_on(scopes.finish());
        assert!(
            secondary_error.is_ok(),
            "the typed map failure must be the only reported error: {secondary_error:?}"
        );
    }

    #[test]
    fn one_shot_readback_completes_after_submission_goes_idle() {
        let gpu = crate::test_device::native_gpu();
        let mut timings = GpuTimings::new(&gpu.device);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("one-shot timestamp encoder"),
            });
        {
            let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("one-shot timestamp trace pass"),
                timestamp_writes: Some(timings.trace_writes()),
            });
        }
        timings
            .encode_readback(&mut encoder, 17_u64)
            .expect("timestamp readback is available");
        let submission = gpu.queue.submit([encoder.finish()]);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("timestamp submission completes");

        let (context, sample) = timings
            .poll(&gpu.device, gpu.queue.get_timestamp_period())
            .expect("timestamp readback succeeds")
            .expect("timestamp readback completed after its submission");

        assert_eq!(context, 17);
        assert!(sample.compute_ms().is_finite());
        assert!(!timings.has_pending_readback());
    }
}

use std::sync::mpsc::{self, TryRecvError};

use num_traits::ToPrimitive as _;

const QUERY_COUNT: u32 = 4;
const QUERY_BYTES: u64 = QUERY_COUNT as u64 * wgpu::QUERY_SIZE as u64;
const QUERY_BYTE_COUNT: usize = QUERY_COUNT as usize * size_of::<u64>();

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimingSample {
    compute_ms: f64,
}

impl TimingSample {
    pub(crate) fn from_ticks(ticks: [u64; 4], timestamp_period_ns: f32) -> Self {
        let milliseconds_per_tick = f64::from(timestamp_period_ns) / 1_000_000.0;
        let escape_map_ticks = ticks[1].saturating_sub(ticks[0]);
        let trace_ticks = ticks[3].saturating_sub(ticks[2]);
        Self {
            compute_ms: escape_map_ticks
                .saturating_add(trace_ticks)
                .to_f64()
                .unwrap_or(f64::INFINITY)
                * milliseconds_per_tick,
        }
    }

    pub(crate) const fn compute_ms(self) -> f64 {
        self.compute_ms
    }
}

fn decode_query_ticks(bytes: &[u8]) -> Option<[u64; QUERY_COUNT as usize]> {
    if bytes.len() != QUERY_BYTE_COUNT {
        return None;
    }

    let mut ticks = [0; QUERY_COUNT as usize];
    for (tick, encoded) in ticks.iter_mut().zip(bytes.chunks_exact(8)) {
        *tick = u64::from_le_bytes(encoded.try_into().ok()?);
    }
    Some(ticks)
}

#[derive(Debug, thiserror::Error)]
pub enum TimingError {
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

pub struct GpuTimings {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    callback_receiver: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl GpuTimings {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame pass timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_COUNT,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame timestamp resolve buffer"),
            size: QUERY_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame timestamp readback buffer"),
            size: QUERY_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            callback_receiver: None,
        }
    }

    pub(crate) const fn capture_available(&self) -> bool {
        self.callback_receiver.is_none()
    }

    pub(crate) const fn escape_map_writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    pub(crate) const fn trace_writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(2),
            end_of_pass_write_index: Some(3),
        }
    }

    pub(crate) fn encode_resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..QUERY_COUNT, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            QUERY_BYTES,
        );
    }

    pub(crate) fn begin_readback(&mut self) {
        if self.callback_receiver.is_some() {
            return;
        }

        let (sender, receiver) = mpsc::channel();
        self.readback_buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                if sender.send(result).is_err() {
                    tracing::debug!("timestamp callback receiver dropped");
                }
            });
        self.callback_receiver = Some(receiver);
    }

    pub(crate) fn poll(
        &mut self,
        device: &wgpu::Device,
        timestamp_period_ns: f32,
    ) -> Result<Option<TimingSample>, TimingError> {
        let Some(receiver) = self.callback_receiver.as_ref() else {
            return Ok(None);
        };

        device.poll(wgpu::PollType::Poll)?;
        match receiver.try_recv() {
            Ok(Ok(())) => {
                let ticks = self.read_ticks();
                self.finish_readback();
                let ticks = ticks?;
                let sample = TimingSample::from_ticks(ticks, timestamp_period_ns);
                Ok(Some(sample))
            }
            Ok(Err(error)) => {
                self.finish_readback();
                Err(error.into())
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                self.finish_readback();
                Err(TimingError::CallbackDisconnected)
            }
        }
    }

    fn read_ticks(&self) -> Result<[u64; QUERY_COUNT as usize], TimingError> {
        let mapped = self.readback_buffer.get_mapped_range(..)?;
        let ticks = decode_query_ticks(&mapped).ok_or(TimingError::InvalidReadback);
        drop(mapped);
        ticks
    }

    fn finish_readback(&mut self) {
        self.readback_buffer.unmap();
        self.callback_receiver = None;
    }

    pub(crate) const fn has_pending_readback(&self) -> bool {
        self.callback_receiver.is_some()
    }
}

#[cfg(test)]
mod tests {
    use num_traits::ToPrimitive as _;
    use proptest::prelude::*;

    use super::{GpuTimings, TimingSample};

    proptest! {
        #[test]
        fn timestamp_encoding_roundtrips(ticks in any::<[u64; 4]>()) {
            let bytes: Vec<u8> = ticks.into_iter().flat_map(u64::to_le_bytes).collect();
            prop_assert_eq!(super::decode_query_ticks(&bytes), Some(ticks));
        }

        #[test]
        fn timestamp_decoder_rejects_every_wrong_byte_count(length in 0_usize..64) {
            prop_assume!(length != 32);
            prop_assert_eq!(super::decode_query_ticks(&vec![0; length]), None);
        }
    }

    #[test]
    fn timestamp_duration_combines_passes_and_saturates() {
        let ordinary = TimingSample::from_ticks([10, 15, 20, 27], 1_000.0);
        assert!((ordinary.compute_ms() - 0.012).abs() <= f64::EPSILON);

        let saturated = TimingSample::from_ticks([0, u64::MAX, 0, u64::MAX], 1.0);
        let expected = u64::MAX.to_f64().expect("u64 converts to finite f64") / 1_000_000.0;
        assert!((saturated.compute_ms() - expected).abs() <= expected * f64::EPSILON);
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
                label: Some("one-shot timestamp compute pass"),
                timestamp_writes: Some(timings.escape_map_writes()),
            });
        }
        {
            let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("one-shot timestamp trace pass"),
                timestamp_writes: Some(timings.trace_writes()),
            });
        }
        timings.encode_resolve(&mut encoder);
        let submission = gpu.queue.submit([encoder.finish()]);
        timings.begin_readback();
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("timestamp submission completes");

        let sample = timings
            .poll(&gpu.device, gpu.queue.get_timestamp_period())
            .expect("timestamp readback succeeds")
            .expect("timestamp readback completed after its submission");

        assert!(sample.compute_ms().is_finite());
        assert!(!timings.has_pending_readback());
    }
}

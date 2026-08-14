use std::sync::mpsc::{self, TryRecvError};

use num_traits::ToPrimitive as _;

const MAXIMUM_QUERY_COUNT: usize = 4;

#[derive(Clone, Copy)]
enum TimingLayout {
    TraceOnly,
    EscapeMapAndTrace,
}

impl TimingLayout {
    const fn for_escape_map(has_escape_map: bool) -> Self {
        if has_escape_map {
            Self::EscapeMapAndTrace
        } else {
            Self::TraceOnly
        }
    }

    const fn query_count(self) -> u32 {
        match self {
            Self::TraceOnly => 2,
            Self::EscapeMapAndTrace => 4,
        }
    }

    fn query_bytes(self) -> u64 {
        u64::from(self.query_count()) * u64::from(wgpu::QUERY_SIZE)
    }

    const fn trace_indices(self) -> [u32; 2] {
        match self {
            Self::TraceOnly => [0, 1],
            Self::EscapeMapAndTrace => [2, 3],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QueryTicks {
    values: [u64; MAXIMUM_QUERY_COUNT],
    count: usize,
}

impl QueryTicks {
    fn as_slice(&self) -> &[u64] {
        &self.values[..self.count]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimingSample {
    compute_ms: f64,
}

impl TimingSample {
    fn from_ticks(ticks: QueryTicks, timestamp_period_ns: f32) -> Self {
        let milliseconds_per_tick = f64::from(timestamp_period_ns) / 1_000_000.0;
        let elapsed_ticks = ticks
            .as_slice()
            .chunks_exact(2)
            .map(|pair| pair[1].saturating_sub(pair[0]))
            .fold(0, u64::saturating_add);
        Self {
            compute_ms: elapsed_ticks.to_f64().unwrap_or(f64::INFINITY) * milliseconds_per_tick,
        }
    }

    pub(crate) const fn compute_ms(self) -> f64 {
        self.compute_ms
    }
}

fn decode_query_ticks(bytes: &[u8]) -> Option<QueryTicks> {
    let count = bytes.len() / size_of::<u64>();
    if !matches!(count, 2 | 4) || bytes.len() != count * size_of::<u64>() {
        return None;
    }

    let mut values = [0; MAXIMUM_QUERY_COUNT];
    for (tick, encoded) in values.iter_mut().zip(bytes.chunks_exact(8)) {
        *tick = u64::from_le_bytes(encoded.try_into().ok()?);
    }
    Some(QueryTicks { values, count })
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
    layout: TimingLayout,
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    callback_receiver: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl GpuTimings {
    pub(crate) fn new(device: &wgpu::Device, has_escape_map: bool) -> Self {
        let layout = TimingLayout::for_escape_map(has_escape_map);
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame pass timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: layout.query_count(),
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame timestamp resolve buffer"),
            size: layout.query_bytes(),
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame timestamp readback buffer"),
            size: layout.query_bytes(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            layout,
            query_set,
            resolve_buffer,
            readback_buffer,
            callback_receiver: None,
        }
    }

    pub(crate) const fn capture_available(&self) -> bool {
        self.callback_receiver.is_none()
    }

    pub(crate) const fn escape_map_writes(&self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        match self.layout {
            TimingLayout::TraceOnly => None,
            TimingLayout::EscapeMapAndTrace => Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        }
    }

    pub(crate) const fn trace_writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        let [beginning, end] = self.layout.trace_indices();
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(beginning),
            end_of_pass_write_index: Some(end),
        }
    }

    pub(crate) fn encode_resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.query_set,
            0..self.layout.query_count(),
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            self.layout.query_bytes(),
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

    fn read_ticks(&self) -> Result<QueryTicks, TimingError> {
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

    use super::{GpuTimings, QueryTicks, TimingSample};

    proptest! {
        #[test]
        fn timestamp_encoding_roundtrips(
            ticks in any::<[u64; 4]>(),
            count in prop_oneof![Just(2_usize), Just(4_usize)],
        ) {
            let bytes: Vec<u8> = ticks[..count]
                .iter()
                .copied()
                .flat_map(u64::to_le_bytes)
                .collect();
            let mut expected = [0; 4];
            expected[..count].copy_from_slice(&ticks[..count]);
            prop_assert_eq!(
                super::decode_query_ticks(&bytes),
                Some(QueryTicks { values: expected, count })
            );
        }

        #[test]
        fn timestamp_decoder_rejects_every_wrong_byte_count(length in 0_usize..64) {
            prop_assume!(!matches!(length, 16 | 32));
            prop_assert_eq!(super::decode_query_ticks(&vec![0; length]), None);
        }
    }

    #[test]
    fn timestamp_duration_combines_passes_and_saturates() {
        let ordinary = TimingSample::from_ticks(
            QueryTicks {
                values: [10, 15, 20, 27],
                count: 4,
            },
            1_000.0,
        );
        assert!((ordinary.compute_ms() - 0.012).abs() <= f64::EPSILON);

        let saturated = TimingSample::from_ticks(
            QueryTicks {
                values: [0, u64::MAX, 0, u64::MAX],
                count: 4,
            },
            1.0,
        );
        let expected = u64::MAX.to_f64().expect("u64 converts to finite f64") / 1_000_000.0;
        assert!((saturated.compute_ms() - expected).abs() <= expected * f64::EPSILON);
    }

    #[test]
    fn one_shot_readback_completes_after_submission_goes_idle() {
        let gpu = crate::test_device::native_gpu();
        for has_escape_map in [false, true] {
            let mut timings = GpuTimings::new(&gpu.device, has_escape_map);
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("one-shot timestamp encoder"),
                });
            if let Some(timestamp_writes) = timings.escape_map_writes() {
                let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("one-shot timestamp escape-map pass"),
                    timestamp_writes: Some(timestamp_writes),
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
}

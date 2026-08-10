use std::sync::mpsc::{self, TryRecvError};

const QUERY_COUNT: u32 = 4;
const QUERY_BYTES: u64 = QUERY_COUNT as u64 * wgpu::QUERY_SIZE as u64;
const QUERY_BYTE_COUNT: usize = QUERY_COUNT as usize * size_of::<u64>();

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimingSample {
    compute_ms: f64,
    display_ms: f64,
}

impl TimingSample {
    #[expect(
        clippy::cast_precision_loss,
        reason = "single-pass GPU tick deltas stay far below f64's exact integer range"
    )]
    pub(crate) fn from_ticks(ticks: [u64; 4], timestamp_period_ns: f32) -> Self {
        let milliseconds_per_tick = f64::from(timestamp_period_ns) / 1_000_000.0;
        Self {
            compute_ms: ticks[1].saturating_sub(ticks[0]) as f64 * milliseconds_per_tick,
            display_ms: ticks[3].saturating_sub(ticks[2]) as f64 * milliseconds_per_tick,
        }
    }

    pub(crate) const fn compute_ms(self) -> f64 {
        self.compute_ms
    }

    pub(crate) const fn display_ms(self) -> f64 {
        self.display_ms
    }
}

#[derive(Debug, Default)]
pub struct TimingState {
    map_pending: bool,
}

impl TimingState {
    pub(crate) const fn begin_map(&mut self) -> bool {
        if self.map_pending {
            false
        } else {
            self.map_pending = true;
            true
        }
    }

    pub(crate) const fn has_pending_map(&self) -> bool {
        self.map_pending
    }

    pub(crate) const fn finish_map(&mut self) {
        self.map_pending = false;
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
    callback_sender: mpsc::SyncSender<Result<(), wgpu::BufferAsyncError>>,
    callback_receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    state: TimingState,
    latest: Option<TimingSample>,
}

impl GpuTimings {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Phase 0 pass timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_COUNT,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Phase 0 timestamp resolve buffer"),
            size: QUERY_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Phase 0 timestamp readback buffer"),
            size: QUERY_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let (callback_sender, callback_receiver) = mpsc::sync_channel(1);

        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            callback_sender,
            callback_receiver,
            state: TimingState::default(),
            latest: None,
        }
    }

    pub(crate) const fn capture_available(&self) -> bool {
        !self.state.has_pending_map()
    }

    pub(crate) const fn compute_writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    pub(crate) const fn display_writes(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
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
        if !self.state.begin_map() {
            return;
        }

        let sender = self.callback_sender.clone();
        self.readback_buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                if let Err(error) = sender.try_send(result) {
                    tracing::debug!(?error, "timestamp callback could not publish its result");
                }
            });
    }

    pub(crate) fn poll(
        &mut self,
        device: &wgpu::Device,
        timestamp_period_ns: f32,
    ) -> Result<Option<TimingSample>, TimingError> {
        if !self.state.has_pending_map() {
            return Ok(None);
        }

        device.poll(wgpu::PollType::Poll)?;
        match self.callback_receiver.try_recv() {
            Ok(Ok(())) => {
                let ticks = match self.readback_buffer.get_mapped_range(..) {
                    Ok(mapped) => {
                        let ticks = decode_query_ticks(&mapped);
                        drop(mapped);
                        ticks
                    }
                    Err(error) => {
                        self.readback_buffer.unmap();
                        self.state.finish_map();
                        return Err(error.into());
                    }
                };
                self.readback_buffer.unmap();
                self.state.finish_map();
                let ticks = ticks.ok_or(TimingError::InvalidReadback)?;
                let sample = TimingSample::from_ticks(ticks, timestamp_period_ns);
                self.latest = Some(sample);
                Ok(Some(sample))
            }
            Ok(Err(error)) => {
                self.readback_buffer.unmap();
                self.state.finish_map();
                Err(error.into())
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                self.state.finish_map();
                Err(TimingError::CallbackDisconnected)
            }
        }
    }

    pub(crate) const fn has_pending_readback(&self) -> bool {
        self.state.has_pending_map()
    }

    pub(crate) const fn latest(&self) -> Option<TimingSample> {
        self.latest
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{GpuTimings, TimingSample, TimingState};

    #[test]
    fn timestamp_ticks_are_converted_with_queue_period() {
        let sample = TimingSample::from_ticks([100, 250, 300, 550], 2.0);

        assert!((sample.compute_ms() - 0.000_3).abs() < f64::EPSILON);
        assert!((sample.display_ms() - 0.000_5).abs() < f64::EPSILON);
    }

    #[test]
    fn timing_state_allows_only_one_pending_map() {
        let mut state = TimingState::default();

        assert!(state.begin_map());
        assert!(!state.begin_map());
        assert!(state.has_pending_map());

        state.finish_map();

        assert!(!state.has_pending_map());
        assert!(state.begin_map());
    }

    #[test]
    fn mapped_query_bytes_are_decoded_without_alignment_assumptions() {
        let bytes: Vec<u8> = [10_u64, 30, 40, 90]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();

        assert_eq!(super::decode_query_ticks(&bytes), Some([10, 30, 40, 90]));
        assert_eq!(super::decode_query_ticks(&bytes[..31]), None);
    }

    #[test]
    fn one_shot_readback_completes_after_submission_goes_idle() {
        pollster::block_on(async {
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
            descriptor.backends = crate::native_backends();
            let instance = wgpu::Instance::new(descriptor);
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                })
                .await
                .expect("native adapter is available");
            let adapter_limits = adapter.limits();
            let required_limits = wgpu::Limits::default()
                .using_resolution(adapter_limits.clone())
                .using_alignment(adapter_limits);
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("one-shot timestamp contract device"),
                    required_features: crate::capabilities::BASELINE_FEATURES,
                    required_limits,
                    ..Default::default()
                })
                .await
                .expect("Phase 0 device request succeeds");
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("one-shot timestamp render target"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let mut timings = GpuTimings::new(&device);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("one-shot timestamp encoder"),
            });
            {
                let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("one-shot timestamp compute pass"),
                    timestamp_writes: Some(timings.compute_writes()),
                });
            }
            {
                let attachment = Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                });
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("one-shot timestamp display pass"),
                    color_attachments: &[attachment],
                    depth_stencil_attachment: None,
                    timestamp_writes: Some(timings.display_writes()),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            timings.encode_resolve(&mut encoder);
            queue.submit([encoder.finish()]);
            timings.begin_readback();

            let deadline = Instant::now() + Duration::from_secs(5);
            while timings.latest().is_none() && Instant::now() < deadline {
                let _completed = timings
                    .poll(&device, queue.get_timestamp_period())
                    .expect("non-blocking timestamp poll succeeds");
                std::thread::yield_now();
            }

            assert!(timings.latest().is_some());
            assert!(!timings.has_pending_readback());
        });
    }
}

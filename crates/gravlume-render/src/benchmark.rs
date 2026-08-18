//! Minimal Criterion seam for the current production trace pipeline.
//!
//! The benchmark deliberately reuses the renderer's trace and timestamp implementations. Historical
//! A/B variants and their validation artifacts live in `docs/research`; they are not permanent
//! runtime APIs.

use std::{num::NonZeroU32, time::Duration};

use criterion::{Criterion, SamplingMode, Throughput};
use gravlume_domain::{
    Angle, EquatorialCircularEmitter, EquatorialSurface, KerrNewmanSpacetime, KerrSchildChart,
    Observation, PerspectiveView, PhysicalScene, PhysicalSceneInput, StationaryObserverInput,
    SurfaceTransport, ValidationReport,
};

use crate::{
    CapabilityError, GpuTraceInputError, TimingError,
    capabilities::{BASELINE_FEATURES, check_baseline_adapter, required_device_limits},
    extent::RenderExtent,
    ray_tracer::{RayTracer, TileRegion, TraceBatchOptions, TraceImage},
    timing::GpuTimings,
};

const BENCHMARK_WIDTH: u32 = 1_280;
const BENCHMARK_HEIGHT: u32 = 720;

#[derive(Debug, thiserror::Error)]
enum TraceBenchmarkError {
    #[error("failed to construct the benchmark observation: {0}")]
    Observation(#[from] ValidationReport),
    #[error(transparent)]
    Capability(#[from] CapabilityError),
    #[error("failed to request a native benchmark adapter: {0}")]
    Adapter(#[from] wgpu::RequestAdapterError),
    #[error("failed to request the native benchmark device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error(transparent)]
    TraceInput(#[from] GpuTraceInputError),
    #[error(transparent)]
    Timing(#[from] TimingError),
    #[error("waiting for a benchmark submission failed: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("the native adapter cannot dispatch the fixed 1280x720 benchmark")]
    UnsupportedExtent,
    #[error("the summed GPU duration overflowed the platform Duration representation")]
    DurationOverflow,
    #[error("the timestamp readback did not complete after its submission")]
    MissingTimingSample,
}

struct TraceGpuBenchmark {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute: RayTracer,
    target: TraceImage,
    tiles: TileRegion,
    timings: GpuTimings<()>,
    adapter_info: wgpu::AdapterInfo,
}

impl TraceGpuBenchmark {
    const NAME: &str = "production_1280x720";
    const PIXELS: u64 = BENCHMARK_WIDTH as u64 * BENCHMARK_HEIGHT as u64;

    /// Creates the fixed production benchmark workload and reusable GPU resources.
    ///
    /// # Errors
    ///
    /// Returns an error when the scene is invalid or the native adapter cannot satisfy the
    /// renderer's baseline trace contract.
    fn new() -> Result<Self, TraceBenchmarkError> {
        let (device, queue, adapter_info) = pollster::block_on(request_benchmark_device())?;
        let extent = benchmark_extent(&device.limits())?;
        let observation = benchmark_observation()?;
        let compute = RayTracer::new(&device, &observation)?;
        let target = compute.create_target(&device, extent);
        let tiles = TileRegion::all(extent);
        let timings = GpuTimings::new(&device, compute.has_escape_map());

        Ok(Self {
            device,
            queue,
            compute,
            target,
            tiles,
            timings,
            adapter_info,
        })
    }

    const fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Executes `iterations` traces and returns the sum of their GPU timestamp durations.
    ///
    /// Command encoding, submission, waiting, and readback are intentionally excluded from the
    /// returned duration. Criterion's `iter_custom` uses this value as its measurement.
    ///
    /// # Errors
    ///
    /// Returns an error when submission or timestamp readback fails.
    fn measure_gpu(&mut self, iterations: u64) -> Result<Duration, TraceBenchmarkError> {
        (0..iterations).try_fold(Duration::ZERO, |total, _| {
            total
                .checked_add(self.measure_trace()?)
                .ok_or(TraceBenchmarkError::DurationOverflow)
        })
    }

    fn measure_trace(&mut self) -> Result<Duration, TraceBenchmarkError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("production trace benchmark encoder"),
            });
        self.compute.encode_batch(
            &self.queue,
            &mut encoder,
            &self.target,
            self.tiles,
            TraceBatchOptions::new(
                self.timings.escape_map_writes(),
                Some(self.timings.trace_writes()),
                true,
            ),
        );
        self.timings.encode_readback(&mut encoder, ())?;
        let submission = self.queue.submit([encoder.finish()]);
        self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        let (_, sample) = self
            .timings
            .poll(&self.device, self.queue.get_timestamp_period())?
            .ok_or(TraceBenchmarkError::MissingTimingSample)?;
        Duration::try_from_secs_f64(sample.compute_ms() / 1_000.0)
            .map_err(|_| TraceBenchmarkError::DurationOverflow)
    }
}

/// Registers the fixed production trace benchmark with Criterion.
///
/// This is the feature-gated public seam used by Cargo's separate benchmark crate.
///
/// # Panics
///
/// Panics when the native adapter cannot create or execute the fixed benchmark workload. Cargo
/// benchmarks have no caller to which a typed setup or measurement error can be propagated.
pub fn register(criterion: &mut Criterion) {
    let mut trace = TraceGpuBenchmark::new()
        .unwrap_or_else(|error| panic!("failed to create trace GPU benchmark: {error}"));
    let adapter = trace.adapter_info();
    eprintln!(
        "trace GPU benchmark adapter: {} ({:?}, {:?})",
        adapter.name, adapter.backend, adapter.device_type
    );

    let mut group = criterion.benchmark_group("trace_gpu");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(15))
        .sampling_mode(SamplingMode::Flat)
        .noise_threshold(0.05)
        .throughput(Throughput::Elements(TraceGpuBenchmark::PIXELS));
    group.bench_function(TraceGpuBenchmark::NAME, |bencher| {
        bencher.iter_custom(|iterations| {
            trace
                .measure_gpu(iterations)
                .unwrap_or_else(|error| panic!("trace GPU benchmark failed: {error}"))
        });
    });
    group.finish();
}

fn benchmark_extent(limits: &wgpu::Limits) -> Result<RenderExtent, TraceBenchmarkError> {
    let extent = RenderExtent::new(BENCHMARK_WIDTH, BENCHMARK_HEIGHT)
        .ok_or(TraceBenchmarkError::UnsupportedExtent)?;
    let [workgroups_x, workgroups_y] = crate::ray_tracer::tile_grid(extent);
    if BENCHMARK_WIDTH > limits.max_texture_dimension_2d
        || BENCHMARK_HEIGHT > limits.max_texture_dimension_2d
        || workgroups_x > limits.max_compute_workgroups_per_dimension
        || workgroups_y > limits.max_compute_workgroups_per_dimension
    {
        return Err(TraceBenchmarkError::UnsupportedExtent);
    }
    Ok(extent)
}

async fn request_benchmark_device()
-> Result<(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo), TraceBenchmarkError> {
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
        .await?;
    let adapter_info = adapter.get_info();
    let hdr_usages = adapter
        .get_texture_format_features(wgpu::TextureFormat::Rgba16Float)
        .allowed_usages;
    check_baseline_adapter(
        adapter_info.device_type,
        adapter.get_downlevel_capabilities().is_webgpu_compliant(),
        adapter.features(),
        hdr_usages,
    )?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("native trace benchmark device"),
            required_features: BASELINE_FEATURES,
            required_limits: required_device_limits(adapter.limits()),
            ..Default::default()
        })
        .await?;
    Ok((device, queue, adapter_info))
}

fn benchmark_observation() -> Result<Observation, TraceBenchmarkError> {
    let spacetime = KerrNewmanSpacetime::new(1.0, 0.8, 0.0, KerrSchildChart::Outgoing)?;
    let observer_xyz = spacetime.oblate_to_cartesian(30.0, std::f64::consts::FRAC_PI_3, 0.0);
    let observer = StationaryObserverInput::new(
        [0.0, observer_xyz[0], observer_xyz[1], observer_xyz[2]],
        [0.0; 4],
        [0.0, 0.0, 1.0],
        1.0,
    );
    let emitter = EquatorialCircularEmitter::inverse_cube_bolometric_v1(6.0, 20.0, 1.0)?;
    let surface = EquatorialSurface::new(emitter, SurfaceTransport::Vacuum)?;
    let scene = PhysicalScene::new(PhysicalSceneInput::new(
        1.0,
        0.8,
        0.0,
        KerrSchildChart::Outgoing,
        observer,
    ))?
    .with_equatorial_surface(surface);
    let view = PerspectiveView::new(
        NonZeroU32::new(BENCHMARK_WIDTH).ok_or(TraceBenchmarkError::UnsupportedExtent)?,
        NonZeroU32::new(BENCHMARK_HEIGHT).ok_or(TraceBenchmarkError::UnsupportedExtent)?,
        Angle::from_radians(std::f64::consts::FRAC_PI_4)?,
    )?;
    Ok(Observation::new(scene, view))
}

use std::sync::{Arc, mpsc};

use gravlume_domain::Observation;
use gravlume_native_display::DynamicRange;
use num_traits::ToPrimitive as _;

use crate::{
    capabilities::{
        BASELINE_FEATURES, OutputMode, SurfaceSelection, check_baseline_adapter,
        required_device_limits, select_surface,
    },
    display::{DisplayPipeline, DisplayTarget, PublishedScene, ScenePresentation, UI_FORMAT},
    error::{
        DeviceEvent, GpuErrorScopes, RendererError, RendererInitError, ResizeError,
        install_device_callbacks, scoped_gpu_operation,
    },
    extent::{ExtentChange, ExtentTracker, RenderExtent},
    ray_tracer::{
        RayTracer, TileRegion, TraceImage, direction_reconstruction_scratch_bytes,
        shadow_coverage_scratch_bytes, tile_grid,
    },
    timing::GpuTimings,
};

const MAXIMUM_NATIVE_TRACE_PIXELS: u64 = 3_840 * 2_160;
const HDR_BYTES_PER_PIXEL: u64 = 8;
const UI_BYTES_PER_PIXEL: u64 = 4;
const MAXIMUM_CORE_RESOURCE_BYTES: u64 = 256 * 1024 * 1024;
const INITIAL_TRACE_TILES_PER_BATCH: u32 = 512;
const TARGET_TRACE_BATCH_MS: f64 = 32.0;
const MAXIMUM_TRACE_BATCH_MS: f64 = 50.0;
const MAXIMUM_BATCH_SCALE: f64 = 1.5;
const MINIMUM_BATCH_SCALE: f64 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentSkip {
    ZeroExtent,
    Suspended,
    Timeout,
    Occluded,
    Outdated,
    Lost,
    Validation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentResult {
    Presented,
    Skipped(PresentSkip),
}

#[derive(Debug, Default)]
pub struct RendererUpdate {
    published_generation: Option<u64>,
    completed_present_generation: Option<u64>,
    events: Vec<DeviceEvent>,
}

impl RendererUpdate {
    #[must_use]
    pub const fn published_generation(&self) -> Option<u64> {
        self.published_generation
    }

    #[must_use]
    pub const fn completed_present_generation(&self) -> Option<u64> {
        self.completed_present_generation
    }

    pub fn take_events(&mut self) -> Vec<DeviceEvent> {
        std::mem::take(&mut self.events)
    }
}

pub struct RendererDiagnostics<'a> {
    adapter_name: &'a str,
    backend: &'a str,
    driver: &'a str,
    surface_format: &'a str,
    color_space: &'a str,
    display_transfer: &'static str,
    trace_progress: Option<TraceProgressDiagnostics>,
}

#[derive(Clone, Copy, Debug)]
struct TraceProgressDiagnostics {
    completed_tiles: u32,
    total_tiles: u32,
    completed_batches: u32,
    total_compute_ms: f64,
    maximum_batch_ms: f64,
}

impl RendererDiagnostics<'_> {
    #[must_use]
    pub const fn adapter_name(&self) -> &str {
        self.adapter_name
    }

    #[must_use]
    pub const fn backend(&self) -> &str {
        self.backend
    }

    #[must_use]
    pub const fn driver(&self) -> &str {
        self.driver
    }

    #[must_use]
    pub const fn surface_format(&self) -> &str {
        self.surface_format
    }

    #[must_use]
    pub const fn color_space(&self) -> &str {
        self.color_space
    }

    #[must_use]
    pub const fn display_transfer(&self) -> &'static str {
        self.display_transfer
    }

    #[must_use]
    pub fn trace_completion(&self) -> Option<f64> {
        self.trace_progress
            .map(|progress| f64::from(progress.completed_tiles) / f64::from(progress.total_tiles))
    }

    #[must_use]
    pub fn completed_trace_batches(&self) -> Option<u32> {
        self.trace_progress
            .map(|progress| progress.completed_batches)
    }

    #[must_use]
    pub fn total_trace_compute_ms(&self) -> Option<f64> {
        self.trace_progress
            .map(|progress| progress.total_compute_ms)
    }

    #[must_use]
    pub fn maximum_trace_batch_ms(&self) -> Option<f64> {
        self.trace_progress
            .map(|progress| progress.maximum_batch_ms)
    }
}

struct FrameResources {
    candidate: Option<TraceCandidate>,
    display: DisplayTarget,
    presentation: ScenePresentation,
    presentation_extent: RenderExtent,
    completed_batches: u32,
    total_compute_ms: f64,
    maximum_batch_ms: f64,
}

struct TraceCandidate {
    trace: TraceImage,
    completed_presentation: ScenePresentation,
    progress: TraceProgress,
}

struct CompletedCandidate {
    view: wgpu::TextureView,
    presentation: ScenePresentation,
}

#[derive(Debug)]
struct TraceProgress {
    grid: [u32; 2],
    total_tiles: u32,
    next_tile: u32,
    tiles_per_batch: u32,
    maximum_batch_tiles: u32,
    maximum_dispatch_dimension: u32,
    in_flight: Option<TileRegion>,
    completed_batches: u32,
    total_compute_ms: f64,
    maximum_batch_ms: f64,
}

enum PreparedSurfaceFrame {
    Render {
        texture: wgpu::SurfaceTexture,
        reconfigure_after_present: bool,
    },
    Skip(PresentSkip),
}

struct SurfaceUpdate {
    selection: SurfaceSelection,
    presentation_pipeline: Option<wgpu::RenderPipeline>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TraceSubmission {
    extent_generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CoreResourcePlan {
    published: u64,
    installed: FrameResourceFootprint,
    replacement: FrameResourceFootprint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrameResourceFootprint {
    ui: u64,
    candidate: u64,
    trace_scratch: u64,
}

impl FrameResourceFootprint {
    const EMPTY: Self = Self {
        ui: 0,
        candidate: 0,
        trace_scratch: 0,
    };

    const fn display_only(extent: RenderExtent) -> Self {
        Self {
            ui: extent_pixels(extent),
            candidate: 0,
            trace_scratch: 0,
        }
    }

    fn tracing(extent: RenderExtent) -> Self {
        Self {
            ui: extent_pixels(extent),
            candidate: extent_pixels(extent),
            trace_scratch: shadow_coverage_scratch_bytes(extent)
                .saturating_add(direction_reconstruction_scratch_bytes(extent)),
        }
    }

    const fn required_bytes(self) -> u64 {
        self.ui
            .saturating_mul(UI_BYTES_PER_PIXEL)
            .saturating_add(self.candidate.saturating_mul(HDR_BYTES_PER_PIXEL))
            .saturating_add(self.trace_scratch)
    }
}

impl CoreResourcePlan {
    fn without_installed_frame(published: RenderExtent, replacement: RenderExtent) -> Self {
        Self {
            published: extent_pixels(published),
            installed: FrameResourceFootprint::EMPTY,
            replacement: FrameResourceFootprint::tracing(replacement),
        }
    }

    fn rebuild(
        published: RenderExtent,
        installed: FrameResourceFootprint,
        replacement: RenderExtent,
    ) -> Self {
        Self {
            published: extent_pixels(published),
            installed,
            replacement: FrameResourceFootprint::tracing(replacement),
        }
    }

    const fn required_bytes(self) -> u64 {
        self.published
            .saturating_mul(HDR_BYTES_PER_PIXEL)
            .saturating_add(self.installed.required_bytes())
            .saturating_add(self.replacement.required_bytes())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TraceCompletion {
    Stale,
    Pending,
    Ready,
}

impl FrameResources {
    fn new(
        device: &wgpu::Device,
        trace: &RayTracer,
        display: &DisplayPipeline,
        published: &PublishedScene,
        extent: RenderExtent,
    ) -> Self {
        let display_target = DisplayPipeline::create_target(device, extent);
        let presentation = display.bind_scene(device, published.view(), &display_target);
        let candidate = Self::create_candidate(device, trace, display, &display_target, extent);
        Self {
            candidate: Some(candidate),
            display: display_target,
            presentation,
            presentation_extent: extent,
            completed_batches: 0,
            total_compute_ms: 0.0,
            maximum_batch_ms: 0.0,
        }
    }

    fn create_candidate(
        device: &wgpu::Device,
        trace: &RayTracer,
        display: &DisplayPipeline,
        display_target: &DisplayTarget,
        extent: RenderExtent,
    ) -> TraceCandidate {
        let trace = trace.create_target(device, extent);
        let completed_presentation = display.bind_scene(device, trace.view(), display_target);
        TraceCandidate {
            trace,
            completed_presentation,
            progress: TraceProgress::new(
                extent,
                device.limits().max_compute_workgroups_per_dimension,
            ),
        }
    }

    fn next_batch(&self) -> Option<TileRegion> {
        self.candidate.as_ref()?.progress.next_batch()
    }

    fn submitted(&mut self, batch: TileRegion) {
        if let Some(candidate) = self.candidate.as_mut() {
            candidate.progress.submitted(batch);
        }
    }

    fn complete_submission(
        &mut self,
        submission: TraceSubmission,
        current_generation: u64,
        compute_ms: f64,
    ) -> Option<CompletedCandidate> {
        let completion = self
            .candidate
            .as_mut()
            .map_or(TraceCompletion::Stale, |candidate| {
                candidate
                    .progress
                    .complete_submission(submission, current_generation, compute_ms)
            });
        if completion == TraceCompletion::Stale {
            return None;
        }
        self.completed_batches += 1;
        if compute_ms.is_finite() {
            self.total_compute_ms += compute_ms;
            self.maximum_batch_ms = self.maximum_batch_ms.max(compute_ms);
        }
        if completion != TraceCompletion::Ready {
            return None;
        }
        let candidate = self.candidate.take()?;
        Some(CompletedCandidate {
            view: candidate.trace.view().clone(),
            presentation: candidate.completed_presentation,
        })
    }

    fn diagnostics(&self) -> TraceProgressDiagnostics {
        let (completed_tiles, total_tiles) = self.candidate.as_ref().map_or_else(
            || {
                let [columns, rows] = tile_grid(self.presentation_extent);
                let tiles = columns * rows;
                (tiles, tiles)
            },
            |candidate| {
                let diagnostics = candidate.progress.diagnostics();
                (diagnostics.completed_tiles, diagnostics.total_tiles)
            },
        );
        TraceProgressDiagnostics {
            completed_tiles,
            total_tiles,
            completed_batches: self.completed_batches,
            total_compute_ms: self.total_compute_ms,
            maximum_batch_ms: self.maximum_batch_ms,
        }
    }

    fn resource_footprint(&self) -> FrameResourceFootprint {
        if self.candidate.is_some() {
            FrameResourceFootprint::tracing(self.presentation_extent)
        } else {
            FrameResourceFootprint::display_only(self.presentation_extent)
        }
    }
}

impl TraceProgress {
    const fn new(extent: RenderExtent, maximum_dispatch_dimension: u32) -> Self {
        debug_assert!(maximum_dispatch_dimension > 0);
        let grid = tile_grid(extent);
        let total_tiles = grid[0] * grid[1];
        let maximum_batch_tiles = if grid[0] > maximum_dispatch_dimension {
            maximum_dispatch_dimension
        } else {
            grid[0].saturating_mul(if grid[1] < maximum_dispatch_dimension {
                grid[1]
            } else {
                maximum_dispatch_dimension
            })
        };
        Self {
            grid,
            total_tiles,
            next_tile: 0,
            tiles_per_batch: if INITIAL_TRACE_TILES_PER_BATCH < maximum_batch_tiles {
                INITIAL_TRACE_TILES_PER_BATCH
            } else {
                maximum_batch_tiles
            },
            maximum_batch_tiles,
            maximum_dispatch_dimension,
            in_flight: None,
            completed_batches: 0,
            total_compute_ms: 0.0,
            maximum_batch_ms: 0.0,
        }
    }

    fn next_batch(&self) -> Option<TileRegion> {
        if self.in_flight.is_some() || self.next_tile == self.total_tiles {
            return None;
        }
        let tile_x = self.next_tile % self.grid[0];
        let tile_y = self.next_tile / self.grid[0];
        let remaining_tiles = self.total_tiles - self.next_tile;
        let budget = self.tiles_per_batch.min(remaining_tiles);
        let remaining_columns = self.grid[0] - tile_x;
        let workgroups_x = budget
            .min(remaining_columns)
            .min(self.maximum_dispatch_dimension);
        let workgroups_y = if tile_x == 0 && workgroups_x == self.grid[0] && budget >= self.grid[0]
        {
            (budget / self.grid[0])
                .min(self.grid[1] - tile_y)
                .min(self.maximum_dispatch_dimension)
        } else {
            1
        };
        Some(TileRegion::new(
            [tile_x, tile_y],
            [workgroups_x, workgroups_y],
        ))
    }

    fn submitted(&mut self, batch: TileRegion) {
        let origin = batch.origin();
        debug_assert_eq!(origin[1] * self.grid[0] + origin[0], self.next_tile);
        debug_assert!(self.in_flight.is_none());
        self.next_tile += batch.len();
        self.in_flight = Some(batch);
    }

    fn completed(&mut self, compute_ms: f64) {
        let Some(batch) = self.in_flight.take() else {
            return;
        };
        self.completed_batches += 1;
        if compute_ms.is_finite() {
            self.total_compute_ms += compute_ms;
            self.maximum_batch_ms = self.maximum_batch_ms.max(compute_ms);
        }
        if self.next_tile == self.total_tiles || !compute_ms.is_finite() || compute_ms <= 0.0 {
            return;
        }

        let scale = if compute_ms > MAXIMUM_TRACE_BATCH_MS {
            MINIMUM_BATCH_SCALE
        } else {
            (TARGET_TRACE_BATCH_MS / compute_ms).clamp(MINIMUM_BATCH_SCALE, MAXIMUM_BATCH_SCALE)
        };
        let scaled = (f64::from(batch.len()) * scale).round().clamp(
            1.0,
            f64::from(self.total_tiles.min(self.maximum_batch_tiles)),
        );
        self.tiles_per_batch = scaled.to_u32().unwrap_or(self.total_tiles);
    }

    fn complete_submission(
        &mut self,
        submission: TraceSubmission,
        current_generation: u64,
        compute_ms: f64,
    ) -> TraceCompletion {
        if submission.extent_generation != current_generation {
            return TraceCompletion::Stale;
        }
        self.completed(compute_ms);
        if self.is_complete() {
            TraceCompletion::Ready
        } else {
            TraceCompletion::Pending
        }
    }

    const fn is_complete(&self) -> bool {
        self.next_tile == self.total_tiles && self.in_flight.is_none()
    }

    const fn diagnostics(&self) -> TraceProgressDiagnostics {
        let completed_tiles = match self.in_flight {
            Some(batch) => self.next_tile - batch.len(),
            None => self.next_tile,
        };
        TraceProgressDiagnostics {
            completed_tiles,
            total_tiles: self.total_tiles,
            completed_batches: self.completed_batches,
            total_compute_ms: self.total_compute_ms,
            maximum_batch_ms: self.maximum_batch_ms,
        }
    }
}

struct DiagnosticLabels {
    adapter_name: String,
    backend: String,
    driver: String,
    surface_format: String,
    color_space: String,
    display_transfer: &'static str,
}

impl DiagnosticLabels {
    fn new(adapter: &wgpu::AdapterInfo, selection: SurfaceSelection) -> Self {
        Self {
            adapter_name: adapter.name.clone(),
            backend: format!("{:?}", adapter.backend),
            driver: format!("{} {}", adapter.driver, adapter.driver_info)
                .trim()
                .to_owned(),
            surface_format: format!("{:?}", selection.format()),
            color_space: format!("{:?}", selection.color_space()),
            display_transfer: display_transfer_label(selection),
        }
    }

    fn update_surface(&mut self, selection: SurfaceSelection) {
        self.surface_format = format!("{:?}", selection.format());
        self.color_space = format!("{:?}", selection.color_space());
        self.display_transfer = display_transfer_label(selection);
    }
}

const fn display_transfer_label(selection: SurfaceSelection) -> &'static str {
    match selection.output_mode() {
        OutputMode::Hdr => "HDR",
        OutputMode::Sdr(_) => "SDR",
    }
}

pub struct Renderer {
    surface: Option<wgpu::Surface<'static>>,
    surface_suspended: bool,
    instance: wgpu::Instance,
    window: Arc<winit::window::Window>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    selection: SurfaceSelection,
    display_state: DynamicRange,
    extent: ExtentTracker,
    frame_resources: Option<FrameResources>,
    published_scene: PublishedScene,
    trace: RayTracer,
    display: DisplayPipeline,
    egui: egui_wgpu::Renderer,
    timings: GpuTimings,
    trace_submission: Option<TraceSubmission>,
    pending_present_generation: Option<u64>,
    diagnostic_labels: DiagnosticLabels,
    device_events: mpsc::Receiver<DeviceEvent>,
}

impl Renderer {
    /// Creates the GPU device, ray tracer, and presentation resources.
    ///
    /// # Errors
    ///
    /// Returns an error when the native surface, adapter, required capabilities, or device cannot
    /// be initialized.
    pub async fn new(
        window: Arc<winit::window::Window>,
        observation: &Observation,
        display_state: DynamicRange,
    ) -> Result<Self, RendererInitError> {
        let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_descriptor.backends = crate::native_backends();
        let instance = wgpu::Instance::new(instance_descriptor);
        let surface = instance.create_surface(Arc::clone(&window))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await?;
        let adapter_info = adapter.get_info();
        let hdr_features = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba16Float);
        check_baseline_adapter(
            adapter_info.device_type,
            adapter.get_downlevel_capabilities().is_webgpu_compliant(),
            adapter.features(),
            hdr_features.allowed_usages,
        )
        .map_err(|reason| RendererInitError::UnsupportedAdapter {
            adapter: adapter_info.name.clone(),
            reason: reason.to_string(),
        })?;
        let selection = select_surface(&surface.get_capabilities(&adapter), display_state)?;
        let diagnostic_labels = DiagnosticLabels::new(&adapter_info, selection);
        let required_limits = required_device_limits(adapter.limits());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Gravlume renderer device"),
                required_features: BASELINE_FEATURES,
                required_limits,
                ..Default::default()
            })
            .await?;

        let (_device_event_sender, device_events) = install_device_callbacks(&device);
        let resource_scopes = GpuErrorScopes::push(&device);
        let trace = RayTracer::new(&device, observation)?;
        let display = DisplayPipeline::new(&device, selection);
        let published_scene = DisplayPipeline::create_initial_scene(&device, &queue);
        let egui =
            egui_wgpu::Renderer::new(&device, UI_FORMAT, egui_wgpu::RendererOptions::default());
        let timings = GpuTimings::new(&device);
        let initial_size = window.inner_size();
        let mut renderer = Self {
            surface: Some(surface),
            surface_suspended: false,
            instance,
            window,
            adapter,
            device,
            queue,
            selection,
            display_state,
            extent: ExtentTracker::default(),
            frame_resources: None,
            published_scene,
            trace,
            display,
            egui,
            timings,
            trace_submission: None,
            pending_present_generation: None,
            diagnostic_labels,
            device_events,
        };
        let resize_result = renderer.resize(initial_size.width, initial_size.height);
        resource_scopes
            .finish()
            .await
            .map_err(|source| RendererInitError::GpuResource {
                stage: "ray-tracing resources",
                source,
            })?;
        resize_result.map_err(RendererInitError::InitialResize)?;
        Ok(renderer)
    }

    /// Rebuilds every size-dependent resource as one transaction.
    ///
    /// # Errors
    ///
    /// Returns a typed error without changing the installed extent generation or frame-resource
    /// bundle when the requested extent, surface configuration, or GPU allocation is invalid.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), ResizeError> {
        // Native resize events can continue while the application has no surface. The desktop
        // reapplies the latest physical window size after resume; allocating here would let
        // successive replacements overlap resources retained by an in-flight submission.
        if self.surface_suspended {
            return Ok(());
        }
        let (candidate_extent, change) = self.extent.updated(width, height);
        match change {
            ExtentChange::Unchanged => return Ok(()),
            ExtentChange::Paused => {
                self.extent = candidate_extent;
            }
            ExtentChange::Rebuild { extent, .. } => {
                let resource_plan = self.resource_plan_for_rebuild(extent);
                validate_render_extent(extent, &self.device.limits(), resource_plan)?;
                let selection = if let Some(surface) = &self.surface {
                    let capabilities = surface.get_capabilities(&self.adapter);
                    select_surface(&capabilities, self.display_state)?
                } else {
                    self.selection
                };

                let pipeline_changed = selection.format() != self.selection.format()
                    || selection.fragment_entry() != self.selection.fragment_entry();
                let (replacement, presentation_pipeline) =
                    scoped_gpu_operation(&self.device, || {
                        let replacement = FrameResources::new(
                            &self.device,
                            &self.trace,
                            &self.display,
                            &self.published_scene,
                            extent,
                        );
                        let presentation_pipeline = pipeline_changed.then(|| {
                            self.display
                                .create_presentation_pipeline(&self.device, selection)
                        });
                        (replacement, presentation_pipeline)
                    })
                    .map_err(|source| ResizeError::GpuResource {
                        stage: "create frame and presentation resources",
                        source,
                    })?;
                if let Some(surface) = &self.surface {
                    configure_surface_scoped(surface, &self.device, selection, extent).map_err(
                        |source| ResizeError::GpuResource {
                            stage: "configure the presentation surface",
                            source,
                        },
                    )?;
                }

                self.install_surface_selection(selection, presentation_pipeline);
                self.extent = candidate_extent;
                self.frame_resources = Some(replacement);
                self.pending_present_generation = None;
            }
        }
        Ok(())
    }

    /// Encodes and presents the latest complete scene with the current UI.
    ///
    /// # Errors
    ///
    /// Returns an error when surface recovery or frame-resource validation fails.
    pub fn present(
        &mut self,
        paint_jobs: &[egui::ClippedPrimitive],
        textures_delta: &egui::TexturesDelta,
        pixels_per_point: f32,
    ) -> Result<PresentResult, RendererError> {
        let Some(extent) = self.extent.extent() else {
            return Ok(PresentResult::Skipped(PresentSkip::ZeroExtent));
        };
        let (surface_texture, reconfigure_after_present) =
            match self.prepare_surface_frame(extent)? {
                PreparedSurfaceFrame::Render {
                    texture,
                    reconfigure_after_present,
                } => (texture, reconfigure_after_present),
                PreparedSurfaceFrame::Skip(reason) => return Ok(PresentResult::Skipped(reason)),
            };

        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [extent.width(), extent.height()],
            pixels_per_point,
        };
        for (texture_id, image_deltas) in &textures_delta.set {
            for image_delta in image_deltas {
                self.egui
                    .update_texture(&self.device, &self.queue, *texture_id, image_delta);
            }
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gravlume frame encoder"),
            });
        let callback_buffers =
            self.egui
                .update_buffers(&self.device, &self.queue, &mut encoder, paint_jobs, &screen);
        let frame = self
            .frame_resources
            .as_mut()
            .ok_or(RendererError::MissingFrameResources)?;
        encode_egui(
            &self.egui,
            &mut encoder,
            frame.display.ui_view(),
            paint_jobs,
            &screen,
        );
        self.display
            .encode_presentation(&mut encoder, &surface_view, &frame.presentation);
        let main_buffer = encoder.finish();
        self.queue
            .submit(callback_buffers.into_iter().chain([main_buffer]));
        free_egui_textures_after_submit(&mut self.egui, textures_delta);
        self.window.pre_present_notify();
        self.queue.present(surface_texture);
        self.pending_present_generation = self.published_scene.generation();

        if reconfigure_after_present {
            self.reconfigure_surface(extent, true)?;
        }
        Ok(PresentResult::Presented)
    }

    /// Submits one hidden full-resolution trace batch without acquiring or presenting a surface.
    ///
    /// # Errors
    ///
    /// Returns an error when size-dependent resources are unavailable.
    pub fn advance_trace(&mut self) -> Result<(), RendererError> {
        if self.surface_suspended
            || self.extent.extent().is_none()
            || !self.timings.capture_available()
        {
            return Ok(());
        }
        let Some(frame) = self.frame_resources.as_mut() else {
            return Ok(());
        };
        let Some(batch) = frame.next_batch() else {
            return Ok(());
        };
        let candidate = frame
            .candidate
            .as_ref()
            .ok_or(RendererError::MissingFrameResources)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gravlume trace encoder"),
            });
        self.trace.encode_node_pass(
            &self.queue,
            &mut encoder,
            &candidate.trace,
            batch,
            Some(self.timings.node_writes()),
        );
        self.trace.encode_resolve_pass(
            &mut encoder,
            &candidate.trace,
            batch,
            Some(self.timings.resolve_writes()),
            true,
        );
        self.timings.encode_resolve(&mut encoder);
        self.queue.submit([encoder.finish()]);
        frame.submitted(batch);
        self.timings.begin_readback();
        self.trace_submission = Some(TraceSubmission {
            extent_generation: self.extent.generation(),
        });
        Ok(())
    }

    /// Advances pending GPU work without blocking the event loop.
    ///
    /// # Errors
    ///
    /// Returns an error when device polling or timestamp readback fails.
    pub fn poll(&mut self) -> Result<RendererUpdate, RendererError> {
        let mut poll_status = None;
        let timing = if self.timings.has_pending_readback() {
            self.timings
                .poll(&self.device, self.queue.get_timestamp_period())?
        } else {
            poll_status = Some(self.device.poll(wgpu::PollType::Poll)?);
            None
        };
        let mut published_generation = None;
        if let Some(timing) = timing {
            let submission = self.trace_submission.take();
            let generation = self.extent.generation();
            if let Some(submission) = submission
                && let Some(frame) = self.frame_resources.as_mut()
                && let Some(completed) =
                    frame.complete_submission(submission, generation, timing.compute_ms())
            {
                let extent = frame.presentation_extent;
                self.published_scene =
                    PublishedScene::from_candidate(completed.view, extent, generation);
                frame.presentation = completed.presentation;
                published_generation = Some(generation);
            }
        }
        let completed_present_generation = poll_status
            .is_some_and(|status| status.is_queue_empty())
            .then(|| self.pending_present_generation.take())
            .flatten();
        let events = self.device_events.try_iter().collect();
        Ok(RendererUpdate {
            published_generation,
            completed_present_generation,
            events,
        })
    }

    pub const fn has_pending_work(&self) -> bool {
        self.timings.has_pending_readback() || self.pending_present_generation.is_some()
    }

    pub const fn generation(&self) -> u64 {
        self.extent.generation()
    }

    pub fn suspend(&mut self) {
        self.surface = None;
        self.surface_suspended = true;
    }

    /// Restores the presentation surface after a desktop resume event.
    ///
    /// # Errors
    ///
    /// Returns an error when the native surface cannot be recreated or its capabilities changed.
    pub fn resume_surface(&mut self) -> Result<(), RendererError> {
        if self.surface.is_none() {
            self.recreate_surface()?;
        }
        self.surface_suspended = false;
        Ok(())
    }

    pub fn diagnostics(&self) -> RendererDiagnostics<'_> {
        RendererDiagnostics {
            adapter_name: &self.diagnostic_labels.adapter_name,
            backend: &self.diagnostic_labels.backend,
            driver: &self.diagnostic_labels.driver,
            surface_format: &self.diagnostic_labels.surface_format,
            color_space: &self.diagnostic_labels.color_space,
            display_transfer: self.diagnostic_labels.display_transfer,
            trace_progress: self
                .frame_resources
                .as_ref()
                .map(FrameResources::diagnostics),
        }
    }

    /// Re-resolves the output transport without invalidating the published trace scene.
    ///
    /// # Errors
    ///
    /// Returns an error when the live surface no longer has a presentable SDR fallback.
    pub fn update_output(&mut self, display_state: DynamicRange) -> Result<(), RendererError> {
        self.display_state = display_state;
        if self.surface_suspended {
            return Ok(());
        }
        let Some(extent) = self.extent.extent() else {
            return Ok(());
        };
        self.reconfigure_surface(extent, false)?;
        Ok(())
    }

    fn prepare_surface_frame(
        &mut self,
        extent: RenderExtent,
    ) -> Result<PreparedSurfaceFrame, RendererError> {
        let acquisition = match self.surface.as_ref() {
            Some(surface) => surface.get_current_texture(),
            None if self.surface_suspended => {
                return Ok(PreparedSurfaceFrame::Skip(PresentSkip::Suspended));
            }
            None => {
                self.recreate_surface()?;
                return Ok(PreparedSurfaceFrame::Skip(PresentSkip::Lost));
            }
        };

        let frame = match acquisition {
            wgpu::CurrentSurfaceTexture::Success(texture) => PreparedSurfaceFrame::Render {
                texture,
                reconfigure_after_present: false,
            },
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => PreparedSurfaceFrame::Render {
                texture,
                reconfigure_after_present: true,
            },
            wgpu::CurrentSurfaceTexture::Timeout => {
                PreparedSurfaceFrame::Skip(PresentSkip::Timeout)
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                PreparedSurfaceFrame::Skip(PresentSkip::Occluded)
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.reconfigure_surface(extent, true)?;
                PreparedSurfaceFrame::Skip(PresentSkip::Outdated)
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.recreate_surface()?;
                PreparedSurfaceFrame::Skip(PresentSkip::Lost)
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                PreparedSurfaceFrame::Skip(PresentSkip::Validation)
            }
        };
        Ok(frame)
    }

    fn reconfigure_surface(
        &mut self,
        extent: RenderExtent,
        force: bool,
    ) -> Result<(), RendererError> {
        let surface = self
            .surface
            .as_ref()
            .ok_or(RendererError::MissingPresentationSurface)?;
        let capabilities = surface.get_capabilities(&self.adapter);
        let update = self.prepare_runtime_surface_update(&capabilities)?;
        if (force || surface_configuration_changed(self.selection, update.selection))
            && let Err(source) =
                configure_surface_scoped(surface, &self.device, update.selection, extent)
        {
            return Err(RendererError::GpuResource {
                stage: "configure the presentation surface",
                source,
            });
        }
        self.install_surface_selection(update.selection, update.presentation_pipeline);
        Ok(())
    }

    fn resource_plan_for_rebuild(&self, replacement: RenderExtent) -> CoreResourcePlan {
        let published = self.published_scene.extent();
        self.frame_resources.as_ref().map_or_else(
            || CoreResourcePlan::without_installed_frame(published, replacement),
            |frame| CoreResourcePlan::rebuild(published, frame.resource_footprint(), replacement),
        )
    }

    fn recreate_surface(&mut self) -> Result<(), RendererError> {
        let replacement = self.instance.create_surface(Arc::clone(&self.window))?;
        let capabilities = replacement.get_capabilities(&self.adapter);
        let update = self.prepare_runtime_surface_update(&capabilities)?;
        if let Some(extent) = self.extent.extent()
            && let Err(source) =
                configure_surface_scoped(&replacement, &self.device, update.selection, extent)
        {
            return Err(RendererError::GpuResource {
                stage: "configure the recreated presentation surface",
                source,
            });
        }
        self.install_surface_selection(update.selection, update.presentation_pipeline);
        self.surface = Some(replacement);
        Ok(())
    }

    fn prepare_runtime_surface_update(
        &self,
        capabilities: &wgpu::SurfaceCapabilities,
    ) -> Result<SurfaceUpdate, RendererError> {
        let selection = select_surface(capabilities, self.display_state)?;
        let presentation_pipeline = self
            .create_presentation_pipeline_if_needed(selection)
            .map_err(|source| RendererError::GpuResource {
                stage: "rebuild the surface presentation pipeline",
                source,
            })?;
        Ok(SurfaceUpdate {
            selection,
            presentation_pipeline,
        })
    }

    fn create_presentation_pipeline_if_needed(
        &self,
        selection: SurfaceSelection,
    ) -> Result<Option<wgpu::RenderPipeline>, wgpu::Error> {
        if selection.format() == self.selection.format()
            && selection.fragment_entry() == self.selection.fragment_entry()
        {
            return Ok(None);
        }
        scoped_gpu_operation(&self.device, || {
            self.display
                .create_presentation_pipeline(&self.device, selection)
        })
        .map(Some)
    }

    fn install_surface_selection(
        &mut self,
        selection: SurfaceSelection,
        presentation_pipeline: Option<wgpu::RenderPipeline>,
    ) {
        self.display
            .install_output(&self.queue, selection, presentation_pipeline);
        self.selection = selection;
        self.diagnostic_labels.update_surface(selection);
    }
}

const fn validate_render_extent(
    extent: RenderExtent,
    limits: &wgpu::Limits,
    resource_plan: CoreResourcePlan,
) -> Result<(), ResizeError> {
    if extent.width() > limits.max_texture_dimension_2d
        || extent.height() > limits.max_texture_dimension_2d
    {
        return Err(ResizeError::ExtentLimit {
            width: extent.width(),
            height: extent.height(),
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
        });
    }
    if extent_pixels(extent) > MAXIMUM_NATIVE_TRACE_PIXELS {
        return Err(ResizeError::NativePixelBudget {
            width: extent.width(),
            height: extent.height(),
            maximum_pixels: MAXIMUM_NATIVE_TRACE_PIXELS,
        });
    }
    let required_bytes = resource_plan.required_bytes();
    if required_bytes > MAXIMUM_CORE_RESOURCE_BYTES {
        return Err(ResizeError::FrameResourceBudget {
            width: extent.width(),
            height: extent.height(),
            required_bytes,
            maximum_bytes: MAXIMUM_CORE_RESOURCE_BYTES,
        });
    }
    Ok(())
}

const fn extent_pixels(extent: RenderExtent) -> u64 {
    extent.width() as u64 * extent.height() as u64
}

fn surface_configuration_changed(current: SurfaceSelection, next: SurfaceSelection) -> bool {
    current.format() != next.format()
        || current.color_space() != next.color_space()
        || current.present_mode() != next.present_mode()
        || current.alpha_mode() != next.alpha_mode()
}

fn configure_surface_scoped(
    surface: &wgpu::Surface<'_>,
    device: &wgpu::Device,
    selection: SurfaceSelection,
    extent: RenderExtent,
) -> Result<(), wgpu::Error> {
    scoped_gpu_operation(device, || {
        surface.configure(
            device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: selection.format(),
                color_space: selection.color_space(),
                width: extent.width(),
                height: extent.height(),
                present_mode: selection.present_mode(),
                desired_maximum_frame_latency: 2,
                alpha_mode: selection.alpha_mode(),
                view_formats: vec![],
            },
        );
    })
}

fn encode_egui(
    renderer: &egui_wgpu::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    surface_view: &wgpu::TextureView,
    paint_jobs: &[egui::ClippedPrimitive],
    screen: &egui_wgpu::ScreenDescriptor,
) {
    let color_attachment = Some(wgpu::RenderPassColorAttachment {
        view: surface_view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        },
    });
    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("egui overlay pass"),
        color_attachments: &[color_attachment],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let mut pass = pass.forget_lifetime();
    renderer.render(&mut pass, paint_jobs, screen);
}

// egui-wgpu defers destruction until after submit so current command buffers stay valid.
// Source: https://docs.rs/egui-wgpu/0.36.1/src/egui_wgpu/winit.rs.html#740-748
fn free_egui_textures_after_submit(
    renderer: &mut egui_wgpu::Renderer,
    textures_delta: &egui::TexturesDelta,
) {
    for texture_id in &textures_delta.free {
        renderer.free_texture(texture_id);
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{
        CoreResourcePlan, FrameResourceFootprint, MAXIMUM_CORE_RESOURCE_BYTES,
        MAXIMUM_NATIVE_TRACE_PIXELS, ResizeError, TraceCompletion, TraceProgress, TraceSubmission,
        validate_render_extent,
    };
    use crate::{extent::RenderExtent, ray_tracer::tile_grid};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn trace_progress_covers_each_tile_once_with_bounded_batches(
            width in 1_u32..=257,
            height in 1_u32..=257,
            maximum_dispatch_dimension in 1_u32..=16,
        ) {
        let extent = RenderExtent::new(width, height).expect("generated extent is nonzero");
        let mut progress = TraceProgress::new(extent, maximum_dispatch_dimension);
        let [tile_columns, tile_rows] = tile_grid(extent);
        let mut covered =
            vec![false; usize::try_from(tile_columns * tile_rows).expect("small grid")];

        while let Some(batch) = progress.next_batch() {
            let [origin_x, origin_y] = batch.origin();
            let [workgroups_x, workgroups_y] = batch.workgroups();
            prop_assert!(workgroups_x <= maximum_dispatch_dimension);
            prop_assert!(workgroups_y <= maximum_dispatch_dimension);
            for tile_y in origin_y..origin_y + workgroups_y {
                for tile_x in origin_x..origin_x + workgroups_x {
                    prop_assert!(tile_x < tile_columns && tile_y < tile_rows);
                    let index =
                        usize::try_from(tile_y * tile_columns + tile_x).expect("small grid index");
                    prop_assert!(!covered[index], "tile ({tile_x}, {tile_y}) was repeated");
                    covered[index] = true;
                }
            }
            progress.submitted(batch);
            prop_assert!(progress.next_batch().is_none(), "one batch stays in flight");
            progress.completed(super::TARGET_TRACE_BATCH_MS);
        }

        prop_assert!(covered.into_iter().all(|tile| tile));
        prop_assert_eq!(progress.next_tile, progress.total_tiles);
        prop_assert!(progress.in_flight.is_none());
        prop_assert!(
            progress.next_batch().is_none(),
            "complete traces are reused"
        );
        }
    }

    #[test]
    fn publication_gate_requires_the_complete_current_generation() {
        let extent = RenderExtent::new(4_097, 9).expect("extent is nonzero");
        let mut progress = TraceProgress::new(
            extent,
            wgpu::Limits::default().max_compute_workgroups_per_dimension,
        );
        let submission = TraceSubmission {
            extent_generation: 7,
        };
        let mut covered_tiles = 0;

        while let Some(batch) = progress.next_batch() {
            progress.submitted(batch);
            covered_tiles += batch.len();
            let expected = if covered_tiles == progress.total_tiles {
                TraceCompletion::Ready
            } else {
                TraceCompletion::Pending
            };
            assert_eq!(batch.finishes(extent), expected == TraceCompletion::Ready);
            assert_eq!(
                progress.complete_submission(submission, 7, super::TARGET_TRACE_BATCH_MS),
                expected
            );
        }

        let stale_extent = RenderExtent::new(1, 1).expect("extent is nonzero");
        let mut stale = TraceProgress::new(
            stale_extent,
            wgpu::Limits::default().max_compute_workgroups_per_dimension,
        );
        let batch = stale
            .next_batch()
            .expect("one tile requires one submission");
        stale.submitted(batch);
        assert_eq!(
            stale.complete_submission(submission, 8, super::TARGET_TRACE_BATCH_MS),
            TraceCompletion::Stale
        );
    }

    #[test]
    fn core_resource_budget_accounts_for_transactional_4k_rebuild() {
        let limits = wgpu::Limits::default();
        let extent = RenderExtent::new(3_840, 2_160).expect("extent is nonzero");
        let initial = CoreResourcePlan::without_installed_frame(RenderExtent::ONE, extent);
        let active_rebuild =
            CoreResourcePlan::rebuild(extent, FrameResourceFootprint::tracing(extent), extent);
        let completed_rebuild =
            CoreResourcePlan::rebuild(extent, FrameResourceFootprint::display_only(extent), extent);
        let cold_rebuild = CoreResourcePlan::rebuild(
            RenderExtent::ONE,
            FrameResourceFootprint::tracing(extent),
            extent,
        );
        assert_eq!(super::extent_pixels(extent), MAXIMUM_NATIVE_TRACE_PIXELS);
        assert!(initial.required_bytes() <= MAXIMUM_CORE_RESOURCE_BYTES);
        assert!(validate_render_extent(extent, &limits, initial).is_ok());
        assert!(validate_render_extent(extent, &limits, cold_rebuild).is_ok());
        assert!(active_rebuild.required_bytes() > MAXIMUM_CORE_RESOURCE_BYTES);
        assert!(matches!(
            validate_render_extent(extent, &limits, active_rebuild),
            Err(ResizeError::FrameResourceBudget { .. })
        ));
        assert!(validate_render_extent(extent, &limits, completed_rebuild).is_ok());
    }

    #[test]
    fn resize_rejects_pixels_beyond_the_native_4k_policy() {
        let limits = wgpu::Limits::default();
        let extent = RenderExtent::new(3_840, 2_161).expect("extent is nonzero");

        assert!(matches!(
            validate_render_extent(
                extent,
                &limits,
                CoreResourcePlan::without_installed_frame(RenderExtent::ONE, extent),
            ),
            Err(ResizeError::NativePixelBudget {
                width: 3_840,
                height: 2_161,
                maximum_pixels: MAXIMUM_NATIVE_TRACE_PIXELS,
            })
        ));
    }

    #[test]
    fn trace_batches_respect_the_device_dispatch_dimension_at_4k() {
        let extent = RenderExtent::new(3_840, 2_160).expect("extent is nonzero");
        let maximum_dispatch_dimension = 512;
        let mut progress = TraceProgress::new(extent, maximum_dispatch_dimension);
        let [tile_columns, tile_rows] = tile_grid(extent);
        let mut covered_tiles = 0;

        while let Some(batch) = progress.next_batch() {
            let [workgroups_x, workgroups_y] = batch.workgroups();
            assert!(workgroups_x <= maximum_dispatch_dimension);
            assert!(workgroups_y <= maximum_dispatch_dimension);
            let [origin_x, origin_y] = batch.origin();
            assert_eq!(origin_y * tile_columns + origin_x, covered_tiles);
            progress.submitted(batch);
            covered_tiles += batch.len();
            progress.completed(f64::MIN_POSITIVE);
        }

        assert_eq!(covered_tiles, tile_columns * tile_rows);
    }

    #[test]
    fn resize_rejects_each_excess_texture_dimension() {
        let limits = wgpu::Limits::default();
        let maximum = limits.max_texture_dimension_2d;
        let too_wide = RenderExtent::new(maximum + 1, 1).expect("extent is nonzero");
        let too_tall = RenderExtent::new(1, maximum + 1).expect("extent is nonzero");

        assert!(matches!(
            validate_render_extent(
                too_wide,
                &limits,
                CoreResourcePlan::without_installed_frame(RenderExtent::ONE, too_wide),
            ),
            Err(ResizeError::ExtentLimit {
                width,
                height: 1,
                max_texture_dimension_2d,
            }) if width == maximum + 1 && max_texture_dimension_2d == maximum
        ));
        assert!(matches!(
            validate_render_extent(
                too_tall,
                &limits,
                CoreResourcePlan::without_installed_frame(RenderExtent::ONE, too_tall),
            ),
            Err(ResizeError::ExtentLimit {
                width: 1,
                height,
                max_texture_dimension_2d,
            }) if height == maximum + 1 && max_texture_dimension_2d == maximum
        ));
    }
}

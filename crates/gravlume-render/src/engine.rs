use std::sync::{Arc, mpsc};

use gravlume_domain::Observation;
use num_traits::ToPrimitive as _;

use crate::{
    capabilities::{
        BASELINE_FEATURES, DisplayState, OutputMode, SurfaceSelection, check_baseline_adapter,
        required_device_limits, select_surface,
    },
    display::{CandidatePublication, DisplayPipeline, DisplayTarget, PublishedScene, UI_FORMAT},
    extent::{ExtentChange, ExtentTracker, RenderExtent},
    gpu_error::{
        DeviceEvent, GpuErrorScopes, RenderInitError, RenderRuntimeError, ResizeError,
        install_device_callbacks, scoped_gpu_operation,
    },
    timing::GpuTimings,
    trace::{
        TRACE_WORKGROUP_HEIGHT, TRACE_WORKGROUP_WIDTH, TraceCompute, TracePixels, TraceTarget,
        trace_record_plane_size,
    },
};

const MAXIMUM_NATIVE_TRACE_PIXELS: u64 = 2_560 * 1_440;
const FRAME_RESOURCE_BYTES_PER_PIXEL: u64 = 3 * 16 + 8 + 8 + 4;
const MAXIMUM_FRAME_RESOURCE_BYTES: u64 =
    MAXIMUM_NATIVE_TRACE_PIXELS * FRAME_RESOURCE_BYTES_PER_PIXEL;
const INITIAL_TRACE_PIXELS_PER_BATCH: u32 = 32_768;
const TARGET_TRACE_BATCH_MS: f64 = 32.0;
const MAXIMUM_TRACE_BATCH_MS: f64 = 50.0;
const MAXIMUM_BATCH_SCALE: f64 = 1.5;
const MINIMUM_BATCH_SCALE: f64 = 0.5;
const TRACE_WORKGROUP_PIXELS: u32 = TRACE_WORKGROUP_WIDTH * TRACE_WORKGROUP_HEIGHT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameSkip {
    ZeroExtent,
    Suspended,
    Timeout,
    Occluded,
    Outdated,
    Lost,
    Validation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameStatus {
    Presented,
    Skipped(FrameSkip),
}

#[derive(Debug, Default)]
pub struct PollOutcome {
    completed_readback: bool,
    events: Vec<DeviceEvent>,
}

impl PollOutcome {
    #[must_use]
    pub fn into_parts(self) -> (bool, Vec<DeviceEvent>) {
        (self.completed_readback, self.events)
    }
}

pub struct RenderDiagnostics<'a> {
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
    completed_pixels: u32,
    total_pixels: u32,
    completed_batches: u32,
    total_compute_ms: f64,
    maximum_batch_ms: f64,
}

impl RenderDiagnostics<'_> {
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
            .map(|progress| f64::from(progress.completed_pixels) / f64::from(progress.total_pixels))
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
    published: PublishedScene,
    display: DisplayTarget,
    presentation_extent: RenderExtent,
    completed_batches: u32,
    total_compute_ms: f64,
    maximum_batch_ms: f64,
}

struct TraceCandidate {
    trace: TraceTarget,
    publication: CandidatePublication,
    progress: TraceProgress,
}

#[derive(Debug)]
struct TraceProgress {
    total_pixels: u32,
    next_pixel: u32,
    pixels_per_batch: u32,
    in_flight: Option<TracePixels>,
    completed_batches: u32,
    total_compute_ms: f64,
    maximum_batch_ms: f64,
}

enum PreparedSurfaceFrame {
    Render {
        texture: wgpu::SurfaceTexture,
        reconfigure_after_present: bool,
    },
    Skip(FrameSkip),
}

struct SurfaceUpdate {
    selection: SurfaceSelection,
    presentation_pipeline: Option<wgpu::RenderPipeline>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TraceSubmission {
    extent_generation: u64,
}

impl FrameResources {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        trace: &TraceCompute,
        display: &DisplayPipeline,
        extent: RenderExtent,
    ) -> Self {
        let display_target = DisplayPipeline::create_target(device, extent);
        let published = display.create_published_scene(device, queue, &display_target, extent);
        let candidate = Self::create_candidate(device, trace, display, extent);
        Self {
            candidate: Some(candidate),
            published,
            display: display_target,
            presentation_extent: extent,
            completed_batches: 0,
            total_compute_ms: 0.0,
            maximum_batch_ms: 0.0,
        }
    }

    fn create_candidate(
        device: &wgpu::Device,
        trace: &TraceCompute,
        display: &DisplayPipeline,
        extent: RenderExtent,
    ) -> TraceCandidate {
        let trace = trace.create_target(device, extent);
        let publication = display.bind_candidate(device, trace.view());
        TraceCandidate {
            trace,
            publication,
            progress: TraceProgress::new(extent),
        }
    }

    fn next_batch(&self) -> Option<TracePixels> {
        self.candidate.as_ref()?.progress.next_batch()
    }

    fn submitted(&mut self, batch: TracePixels) {
        if let Some(candidate) = self.candidate.as_mut() {
            candidate.progress.submitted(batch);
        }
    }

    fn publishes(&self, batch: TracePixels) -> bool {
        self.candidate
            .as_ref()
            .is_some_and(|candidate| batch.start() + batch.len() == candidate.progress.total_pixels)
    }

    fn completed(&mut self, compute_ms: f64) -> bool {
        let candidate_complete = self.candidate.as_mut().is_some_and(|candidate| {
            candidate.progress.completed(compute_ms);
            candidate.progress.is_complete()
        });
        self.completed_batches += 1;
        if compute_ms.is_finite() {
            self.total_compute_ms += compute_ms;
            self.maximum_batch_ms = self.maximum_batch_ms.max(compute_ms);
        }
        candidate_complete
    }

    fn release_completed_candidate(&mut self) {
        // Timestamp readback proves the publication submission is complete and the native-sized
        // candidate may be released. Incomplete candidates are never displayable.
        self.candidate = None;
    }

    fn diagnostics(&self) -> TraceProgressDiagnostics {
        let (completed_pixels, total_pixels) = self.candidate.as_ref().map_or_else(
            || {
                let pixels = self.presentation_extent.width() * self.presentation_extent.height();
                (pixels, pixels)
            },
            |candidate| {
                let diagnostics = candidate.progress.diagnostics();
                (diagnostics.completed_pixels, diagnostics.total_pixels)
            },
        );
        TraceProgressDiagnostics {
            completed_pixels,
            total_pixels,
            completed_batches: self.completed_batches,
            total_compute_ms: self.total_compute_ms,
            maximum_batch_ms: self.maximum_batch_ms,
        }
    }
}

impl TraceProgress {
    const fn new(extent: RenderExtent) -> Self {
        Self {
            total_pixels: extent.width() * extent.height(),
            next_pixel: 0,
            pixels_per_batch: INITIAL_TRACE_PIXELS_PER_BATCH,
            in_flight: None,
            completed_batches: 0,
            total_compute_ms: 0.0,
            maximum_batch_ms: 0.0,
        }
    }

    fn next_batch(&self) -> Option<TracePixels> {
        if self.in_flight.is_some() || self.next_pixel == self.total_pixels {
            return None;
        }
        Some(TracePixels::new(
            self.next_pixel,
            self.next_pixel
                .saturating_add(self.pixels_per_batch)
                .min(self.total_pixels),
        ))
    }

    fn submitted(&mut self, batch: TracePixels) {
        debug_assert_eq!(batch.start(), self.next_pixel);
        debug_assert!(self.in_flight.is_none());
        self.next_pixel += batch.len();
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
        if self.next_pixel == self.total_pixels || !compute_ms.is_finite() || compute_ms <= 0.0 {
            return;
        }

        let scale = if compute_ms > MAXIMUM_TRACE_BATCH_MS {
            MINIMUM_BATCH_SCALE
        } else {
            (TARGET_TRACE_BATCH_MS / compute_ms).clamp(MINIMUM_BATCH_SCALE, MAXIMUM_BATCH_SCALE)
        };
        let scaled = (f64::from(batch.len()) * scale).round().clamp(
            f64::from(TRACE_WORKGROUP_PIXELS),
            f64::from(self.total_pixels),
        );
        let scaled = scaled.to_u32().unwrap_or(self.total_pixels);
        self.pixels_per_batch = scaled.div_ceil(TRACE_WORKGROUP_PIXELS) * TRACE_WORKGROUP_PIXELS;
    }

    const fn is_complete(&self) -> bool {
        self.next_pixel == self.total_pixels && self.in_flight.is_none()
    }

    const fn diagnostics(&self) -> TraceProgressDiagnostics {
        let completed_pixels = match self.in_flight {
            Some(batch) => batch.start(),
            None => self.next_pixel,
        };
        TraceProgressDiagnostics {
            completed_pixels,
            total_pixels: self.total_pixels,
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

pub struct GpuEngine {
    surface: Option<wgpu::Surface<'static>>,
    surface_suspended: bool,
    instance: wgpu::Instance,
    window: Arc<winit::window::Window>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    selection: SurfaceSelection,
    display_state: DisplayState,
    extent: ExtentTracker,
    frame_resources: Option<FrameResources>,
    trace: TraceCompute,
    display: DisplayPipeline,
    egui_renderer: egui_wgpu::Renderer,
    timings: GpuTimings,
    trace_submission: Option<TraceSubmission>,
    diagnostic_labels: DiagnosticLabels,
    device_event_sender: mpsc::Sender<DeviceEvent>,
    device_events: mpsc::Receiver<DeviceEvent>,
}

impl GpuEngine {
    /// Creates the GPU device, interactive trace, and presentation resources.
    ///
    /// # Errors
    ///
    /// Returns an error when the native surface, adapter, required capabilities, or device cannot
    /// be initialized.
    pub async fn new(
        window: Arc<winit::window::Window>,
        observation: &Observation,
        display_state: DisplayState,
    ) -> Result<Self, RenderInitError> {
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
        .map_err(|reason| RenderInitError::UnsupportedAdapter {
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

        let (device_event_sender, device_events) = install_device_callbacks(&device);
        let resource_scopes = GpuErrorScopes::push(&device);
        let trace = TraceCompute::new(&device, observation)?;
        let display = DisplayPipeline::new(&device, selection);
        let egui_renderer =
            egui_wgpu::Renderer::new(&device, UI_FORMAT, egui_wgpu::RendererOptions::default());
        let timings = GpuTimings::new(&device);
        let initial_size = window.inner_size();
        let mut engine = Self {
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
            trace,
            display,
            egui_renderer,
            timings,
            trace_submission: None,
            diagnostic_labels,
            device_event_sender,
            device_events,
        };
        let resize_result = engine.resize(initial_size.width, initial_size.height);
        resource_scopes
            .finish()
            .await
            .map_err(|source| RenderInitError::GpuResource {
                stage: "interactive trace GPU resources",
                source,
            })?;
        resize_result.map_err(RenderInitError::InitialResize)?;
        Ok(engine)
    }

    /// Rebuilds every size-dependent resource as one transaction.
    ///
    /// # Errors
    ///
    /// Returns a typed error without changing the installed extent generation or frame-resource
    /// bundle when the requested extent, surface configuration, or GPU allocation is invalid.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), ResizeError> {
        let (candidate_extent, change) = self.extent.updated(width, height);
        match change {
            ExtentChange::Unchanged => return Ok(()),
            ExtentChange::Paused => {
                self.extent = candidate_extent;
                self.frame_resources = None;
            }
            ExtentChange::Rebuild { extent, .. } => {
                validate_render_extent(extent, &self.device.limits())?;
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
                            &self.queue,
                            &self.trace,
                            &self.display,
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
            }
        }
        Ok(())
    }

    /// Encodes, submits, and presents one complete frame transaction.
    ///
    /// # Errors
    ///
    /// Returns an error when surface recovery or frame-resource validation fails.
    pub fn render(
        &mut self,
        paint_jobs: &[egui::ClippedPrimitive],
        textures_delta: &egui::TexturesDelta,
        pixels_per_point: f32,
    ) -> Result<FrameStatus, RenderRuntimeError> {
        let Some(extent) = self.extent.extent() else {
            return Ok(FrameStatus::Skipped(FrameSkip::ZeroExtent));
        };
        let (surface_texture, reconfigure_after_present) =
            match self.prepare_surface_frame(extent)? {
                PreparedSurfaceFrame::Render {
                    texture,
                    reconfigure_after_present,
                } => (texture, reconfigure_after_present),
                PreparedSurfaceFrame::Skip(reason) => return Ok(FrameStatus::Skipped(reason)),
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
                self.egui_renderer.update_texture(
                    &self.device,
                    &self.queue,
                    *texture_id,
                    image_delta,
                );
            }
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gravlume frame encoder"),
            });
        let callback_buffers = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            paint_jobs,
            &screen,
        );
        let frame = self
            .frame_resources
            .as_mut()
            .ok_or(RenderRuntimeError::MissingFrameResources)?;
        let trace_batch = self
            .timings
            .capture_available()
            .then(|| frame.next_batch())
            .flatten();
        if let Some(batch) = trace_batch
            && let Some(candidate) = frame.candidate.as_ref()
        {
            self.trace.encode(
                &self.queue,
                &mut encoder,
                &candidate.trace,
                batch,
                Some(self.timings.compute_writes()),
            );
        }
        let publishes_candidate = trace_batch.is_some_and(|batch| frame.publishes(batch));
        if publishes_candidate && let Some(candidate) = frame.candidate.as_ref() {
            self.display
                .encode_publication(&mut encoder, &frame.published, &candidate.publication);
        }
        encode_egui(
            &self.egui_renderer,
            &mut encoder,
            frame.display.ui_view(),
            paint_jobs,
            &screen,
        );
        self.display
            .encode_presentation(&mut encoder, &surface_view, &frame.published);
        if trace_batch.is_some() {
            self.timings.encode_resolve(&mut encoder);
        }
        let main_buffer = encoder.finish();
        self.queue
            .submit(callback_buffers.into_iter().chain([main_buffer]));
        free_egui_textures_after_submit(&mut self.egui_renderer, textures_delta);
        if let Some(batch) = trace_batch {
            frame.submitted(batch);
            self.timings.begin_readback();
            self.trace_submission = Some(TraceSubmission {
                extent_generation: self.extent.generation(),
            });
        }
        self.window.pre_present_notify();
        self.queue.present(surface_texture);

        if reconfigure_after_present {
            self.reconfigure_surface(extent)?;
        }
        Ok(FrameStatus::Presented)
    }

    /// Advances pending GPU work without blocking the event loop.
    ///
    /// # Errors
    ///
    /// Returns an error when device polling or timestamp readback fails.
    pub fn poll(&mut self) -> Result<PollOutcome, RenderRuntimeError> {
        let timing = if self.timings.has_pending_readback() {
            self.timings
                .poll(&self.device, self.queue.get_timestamp_period())?
        } else {
            self.device.poll(wgpu::PollType::Poll)?;
            None
        };
        if let Some(timing) = timing {
            let submission = self.trace_submission.take();
            if submission
                .is_some_and(|submission| submission.extent_generation == self.extent.generation())
                && let Some(frame) = self.frame_resources.as_mut()
                && frame.completed(timing.compute_ms())
            {
                frame.release_completed_candidate();
            }
        }
        let events = self.device_events.try_iter().collect();
        Ok(PollOutcome {
            completed_readback: timing.is_some(),
            events,
        })
    }

    pub const fn has_pending_gpu_work(&self) -> bool {
        self.timings.has_pending_readback()
    }

    pub fn trace_is_complete(&self) -> bool {
        self.frame_resources
            .as_ref()
            .is_none_or(|frame| frame.candidate.is_none())
    }

    pub fn trace_needs_redraw(&self) -> bool {
        !self.surface_suspended
            && self.timings.capture_available()
            && self
                .frame_resources
                .as_ref()
                .is_some_and(|frame| frame.next_batch().is_some())
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
    pub fn resume_surface(&mut self) -> Result<(), RenderRuntimeError> {
        self.surface_suspended = false;
        if self.surface.is_none() {
            self.recreate_surface()?;
        }
        Ok(())
    }

    pub fn diagnostics(&self) -> RenderDiagnostics<'_> {
        RenderDiagnostics {
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
    pub fn refresh_output(
        &mut self,
        display_state: DisplayState,
    ) -> Result<(), RenderRuntimeError> {
        self.display_state = display_state;
        let Some(extent) = self.extent.extent() else {
            return Ok(());
        };
        let _ = self.reconfigure_surface(extent)?;
        Ok(())
    }

    fn prepare_surface_frame(
        &mut self,
        extent: RenderExtent,
    ) -> Result<PreparedSurfaceFrame, RenderRuntimeError> {
        let acquisition = match self.surface.as_ref() {
            Some(surface) => surface.get_current_texture(),
            None if self.surface_suspended => {
                return Ok(PreparedSurfaceFrame::Skip(FrameSkip::Suspended));
            }
            None => {
                return Ok(PreparedSurfaceFrame::Skip(if self.recreate_surface()? {
                    FrameSkip::Lost
                } else {
                    FrameSkip::Validation
                }));
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
            wgpu::CurrentSurfaceTexture::Timeout => PreparedSurfaceFrame::Skip(FrameSkip::Timeout),
            wgpu::CurrentSurfaceTexture::Occluded => {
                PreparedSurfaceFrame::Skip(FrameSkip::Occluded)
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                PreparedSurfaceFrame::Skip(if self.reconfigure_surface(extent)? {
                    FrameSkip::Outdated
                } else {
                    FrameSkip::Validation
                })
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                PreparedSurfaceFrame::Skip(if self.recreate_surface()? {
                    FrameSkip::Lost
                } else {
                    FrameSkip::Validation
                })
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                PreparedSurfaceFrame::Skip(FrameSkip::Validation)
            }
        };
        Ok(frame)
    }

    fn reconfigure_surface(&mut self, extent: RenderExtent) -> Result<bool, RenderRuntimeError> {
        let Some(surface) = self.surface.as_ref() else {
            return Ok(false);
        };
        let capabilities = surface.get_capabilities(&self.adapter);
        let Some(update) = self.prepare_runtime_surface_update(&capabilities)? else {
            return Ok(false);
        };
        if surface_configuration_changed(self.selection, update.selection)
            && let Err(error) =
                configure_surface_scoped(surface, &self.device, update.selection, extent)
        {
            self.enqueue_device_event(DeviceEvent::from_wgpu(
                "failed to configure the presentation surface",
                error,
            ));
            return Ok(false);
        }
        self.install_surface_selection(update.selection, update.presentation_pipeline);
        Ok(true)
    }

    fn recreate_surface(&mut self) -> Result<bool, RenderRuntimeError> {
        let replacement = self.instance.create_surface(Arc::clone(&self.window))?;
        let capabilities = replacement.get_capabilities(&self.adapter);
        let Some(update) = self.prepare_runtime_surface_update(&capabilities)? else {
            return Ok(false);
        };
        if let Some(extent) = self.extent.extent()
            && let Err(error) =
                configure_surface_scoped(&replacement, &self.device, update.selection, extent)
        {
            let event =
                DeviceEvent::from_wgpu("failed to configure the presentation surface", error);
            self.enqueue_device_event(event);
            return Ok(false);
        }
        self.install_surface_selection(update.selection, update.presentation_pipeline);
        self.surface = Some(replacement);
        Ok(true)
    }

    fn prepare_runtime_surface_update(
        &self,
        capabilities: &wgpu::SurfaceCapabilities,
    ) -> Result<Option<SurfaceUpdate>, RenderRuntimeError> {
        let selection = select_surface(capabilities, self.display_state)?;
        match self.create_presentation_pipeline_if_needed(selection) {
            Ok(presentation_pipeline) => Ok(Some(SurfaceUpdate {
                selection,
                presentation_pipeline,
            })),
            Err(error) => {
                self.enqueue_device_event(DeviceEvent::from_wgpu(
                    "failed to rebuild surface presentation pipeline",
                    error,
                ));
                Ok(None)
            }
        }
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

    fn enqueue_device_event(&self, event: DeviceEvent) {
        if self.device_event_sender.send(event).is_err() {
            tracing::debug!("device event receiver dropped");
        }
    }
}

fn validate_render_extent(extent: RenderExtent, limits: &wgpu::Limits) -> Result<(), ResizeError> {
    if extent.width() > limits.max_texture_dimension_2d
        || extent.height() > limits.max_texture_dimension_2d
    {
        return Err(ResizeError::ExtentLimit {
            width: extent.width(),
            height: extent.height(),
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
        });
    }
    let pixels = u64::from(extent.width()) * u64::from(extent.height());
    let required_bytes = pixels.saturating_mul(FRAME_RESOURCE_BYTES_PER_PIXEL);
    if required_bytes > MAXIMUM_FRAME_RESOURCE_BYTES {
        return Err(ResizeError::FrameResourceBudget {
            width: extent.width(),
            height: extent.height(),
            required_bytes,
            maximum_bytes: MAXIMUM_FRAME_RESOURCE_BYTES,
        });
    }
    let required_bytes = trace_record_plane_size(extent);
    if required_bytes > limits.max_storage_buffer_binding_size
        || required_bytes > limits.max_buffer_size
    {
        return Err(ResizeError::TraceRecordLimit {
            width: extent.width(),
            height: extent.height(),
            required_bytes,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
        });
    }
    Ok(())
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
    use super::{
        INITIAL_TRACE_PIXELS_PER_BATCH, MAXIMUM_FRAME_RESOURCE_BYTES, ResizeError, TraceProgress,
        validate_render_extent,
    };
    use crate::extent::RenderExtent;

    #[test]
    fn trace_progress_covers_a_generation_once_with_bounded_batches() {
        let extent =
            RenderExtent::new(INITIAL_TRACE_PIXELS_PER_BATCH + 13, 2).expect("extent is nonzero");
        let mut progress = TraceProgress::new(extent);
        let mut covered = 0;

        while let Some(batch) = progress.next_batch() {
            assert_eq!(batch.start(), covered);
            assert!(batch.len() <= progress.pixels_per_batch);
            progress.submitted(batch);
            assert!(progress.next_batch().is_none(), "one batch stays in flight");
            covered += batch.len();
            progress.completed(super::TARGET_TRACE_BATCH_MS);
        }

        assert_eq!(covered, extent.width() * extent.height());
        assert_eq!(progress.next_pixel, progress.total_pixels);
        assert!(progress.in_flight.is_none());
        assert!(
            progress.next_batch().is_none(),
            "complete traces are reused"
        );
    }

    #[test]
    fn resize_accepts_the_native_trace_budget_boundary() {
        let limits = wgpu::Limits::default();
        let extent = RenderExtent::new(2_560, 1_440).expect("extent is nonzero");

        assert!(validate_render_extent(extent, &limits).is_ok());
    }

    #[test]
    fn resize_rejects_4k_native_trace_before_allocation() {
        let limits = wgpu::Limits::default();
        let extent = RenderExtent::new(3_840, 2_160).expect("extent is nonzero");

        assert!(matches!(
            validate_render_extent(extent, &limits),
            Err(ResizeError::FrameResourceBudget {
                width: 3_840,
                height: 2_160,
                maximum_bytes: MAXIMUM_FRAME_RESOURCE_BYTES,
                ..
            })
        ));
    }

    #[test]
    fn resize_rejects_each_excess_texture_dimension() {
        let limits = wgpu::Limits::default();
        let maximum = limits.max_texture_dimension_2d;
        let too_wide = RenderExtent::new(maximum + 1, 1).expect("extent is nonzero");
        let too_tall = RenderExtent::new(1, maximum + 1).expect("extent is nonzero");

        assert!(matches!(
            validate_render_extent(too_wide, &limits),
            Err(ResizeError::ExtentLimit {
                width,
                height: 1,
                max_texture_dimension_2d,
            }) if width == maximum + 1 && max_texture_dimension_2d == maximum
        ));
        assert!(matches!(
            validate_render_extent(too_tall, &limits),
            Err(ResizeError::ExtentLimit {
                width: 1,
                height,
                max_texture_dimension_2d,
            }) if height == maximum + 1 && max_texture_dimension_2d == maximum
        ));
    }
}

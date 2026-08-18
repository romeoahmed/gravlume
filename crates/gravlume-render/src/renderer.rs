use std::sync::{Arc, mpsc};

use gravlume_domain::Observation;
use gravlume_native_display::DynamicRange;

mod frame;

use frame::{CoreResourcePlan, FrameResources, TraceProgressDiagnostics, validate_extent};

use crate::{
    capabilities::{
        BASELINE_FEATURES, OutputMode, SurfaceSelection, check_baseline_adapter,
        required_device_limits, select_surface,
    },
    display::{DisplayPipeline, PublishedScene, UI_FORMAT},
    error::{
        DeviceEvent, GpuErrorScopes, RendererError, RendererInitError, ResizeError,
        install_device_callbacks, scoped_gpu_operation,
    },
    extent::{ExtentChange, ExtentTracker, RenderExtent},
    ray_tracer::{RayTracer, TraceBatchOptions},
    scientific_capture::{ScientificCapture, ScientificCaptureError, capture_texture},
    timing::GpuTimings,
};

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
            .map(TraceProgressDiagnostics::completion)
    }

    #[must_use]
    pub fn completed_trace_batches(&self) -> Option<u32> {
        self.trace_progress
            .map(TraceProgressDiagnostics::completed_batches)
    }

    #[must_use]
    pub fn total_trace_compute_ms(&self) -> Option<f64> {
        self.trace_progress
            .map(TraceProgressDiagnostics::total_compute_ms)
    }

    #[must_use]
    pub fn maximum_trace_batch_ms(&self) -> Option<f64> {
        self.trace_progress
            .map(TraceProgressDiagnostics::maximum_batch_ms)
    }
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
    timings: GpuTimings<u64>,
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
        let timings = GpuTimings::new(&device, trace.has_escape_map());
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
                validate_extent(extent, &self.device.limits(), resource_plan)?;
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
            frame.ui_view(),
            paint_jobs,
            &screen,
        );
        self.display
            .encode_presentation(&mut encoder, &surface_view, frame.presentation());
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

    /// Reads the latest complete physical surface image before display mapping or UI composition.
    ///
    /// This explicit export operation waits for its copy submission. The returned binary16 words
    /// retain the published `Rgba16Float` texels and include per-pixel kind plus source, transport,
    /// channel, and numerical metadata. Consumers must accept only `SurfaceRadiance` texels;
    /// analytic escape RGB remains an orientation preview, not an unspecified spectrum.
    ///
    /// # Errors
    ///
    /// Returns an error before the first complete generation, for an analytic-sky-only scene, or
    /// when GPU copy, polling, or buffer mapping fails.
    pub fn capture_scene_linear(&self) -> Result<ScientificCapture, ScientificCaptureError> {
        let generation = self
            .published_scene
            .generation()
            .ok_or(ScientificCaptureError::NoPublishedScene)?;
        let metadata = self
            .trace
            .scientific_capture_metadata()
            .cloned()
            .ok_or(ScientificCaptureError::NoPhysicalSurfaceSource)?;
        capture_texture(
            &self.device,
            &self.queue,
            self.published_scene.view().texture(),
            self.published_scene.extent(),
            generation,
            metadata,
        )
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
            .candidate_trace()
            .ok_or(RendererError::MissingFrameResources)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gravlume trace encoder"),
            });
        self.trace.encode_batch(
            &self.queue,
            &mut encoder,
            candidate,
            batch,
            TraceBatchOptions::new(
                self.timings.escape_map_writes(),
                Some(self.timings.trace_writes()),
                true,
            ),
        );
        let generation = self.extent.generation();
        self.timings.encode_readback(&mut encoder, generation)?;
        self.queue.submit([encoder.finish()]);
        frame.submitted(batch);
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
        if let Some((submission_generation, timing)) = timing {
            let generation = self.extent.generation();
            if let Some(frame) = self.frame_resources.as_mut()
                && let Some(completed) = frame.complete_submission(
                    submission_generation,
                    generation,
                    timing.compute_ms(),
                )
            {
                let extent = frame.extent();
                let (view, presentation) = completed.into_parts();
                self.published_scene = PublishedScene::from_candidate(view, extent, generation);
                frame.install_presentation(presentation);
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
        let replacement_trace_scratch = self.trace.scratch_bytes(replacement);
        self.frame_resources.as_ref().map_or_else(
            || {
                CoreResourcePlan::without_installed_frame(
                    published,
                    replacement,
                    replacement_trace_scratch,
                )
            },
            |frame| frame.rebuild_plan(published, replacement, replacement_trace_scratch),
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

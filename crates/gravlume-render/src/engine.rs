use std::sync::{Arc, mpsc};

use gravlume_domain::Observation;

use crate::{
    capabilities::{
        BASELINE_FEATURES, SurfaceSelection, check_baseline_adapter, required_device_limits,
        select_surface,
    },
    display::{COMPOSITE_FORMAT, DisplayPipeline, DisplayTarget},
    extent::{ExtentChange, ExtentTracker, RenderExtent},
    gpu_error::{
        DeviceEvent, GpuErrorScopes, RenderInitError, RenderRuntimeError, ResizeError,
        install_device_callbacks, scoped_gpu_operation,
    },
    timing::{GpuTimings, TimingSample},
    trace::{TraceCompute, TraceTarget, trace_record_plane_size},
};

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
    extent_generation: u64,
    timing: Option<TimingSample>,
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
    pub const fn extent_generation(&self) -> u64 {
        self.extent_generation
    }

    pub fn compute_ms(&self) -> Option<f64> {
        self.timing.map(TimingSample::compute_ms)
    }

    pub fn display_ms(&self) -> Option<f64> {
        self.timing.map(TimingSample::display_ms)
    }
}

struct FrameResources {
    trace: TraceTarget,
    display: DisplayTarget,
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

impl FrameResources {
    fn new(
        device: &wgpu::Device,
        trace: &TraceCompute,
        display: &DisplayPipeline,
        extent: RenderExtent,
    ) -> Self {
        let trace = trace.create_target(device, extent);
        let display = display.create_target(device, trace.view(), extent);
        Self { trace, display }
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

fn display_transfer_label(selection: SurfaceSelection) -> &'static str {
    if selection.requires_manual_srgb_encoding() {
        "gamma composite passthrough"
    } else {
        "surface sRGB re-encoding"
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
    extent: ExtentTracker,
    frame_resources: Option<FrameResources>,
    trace: TraceCompute,
    display: DisplayPipeline,
    egui_renderer: egui_wgpu::Renderer,
    timings: GpuTimings,
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
        let selection = select_surface(&surface.get_capabilities(&adapter))?;
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
        let display = DisplayPipeline::new(&device, selection.format());
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            COMPOSITE_FORMAT,
            egui_wgpu::RendererOptions::default(),
        );
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
            extent: ExtentTracker::default(),
            frame_resources: None,
            trace,
            display,
            egui_renderer,
            timings,
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
                    select_surface(&capabilities)?
                } else {
                    self.selection
                };

                let format_changed = selection.format() != self.selection.format();
                let (replacement, presentation_pipeline) =
                    scoped_gpu_operation(&self.device, || {
                        let replacement =
                            FrameResources::new(&self.device, &self.trace, &self.display, extent);
                        let presentation_pipeline = format_changed.then(|| {
                            self.display
                                .create_presentation_pipeline(&self.device, selection.format())
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
        let capture_timing = self.timings.capture_available();
        let frame = self
            .frame_resources
            .as_mut()
            .ok_or(RenderRuntimeError::MissingFrameResources)?;
        debug_assert_eq!(frame.trace.extent(), extent);
        let compute_writes = capture_timing.then(|| self.timings.compute_writes());
        self.trace
            .encode(&mut encoder, &frame.trace, compute_writes);
        let display_begin_writes = capture_timing.then(|| self.timings.display_begin_writes());
        self.display
            .encode_display(&mut encoder, &frame.display, display_begin_writes);
        encode_egui(
            &self.egui_renderer,
            &mut encoder,
            frame.display.view(),
            paint_jobs,
            &screen,
        );
        let display_end_writes = capture_timing.then(|| self.timings.display_end_writes());
        self.display.encode_presentation(
            &mut encoder,
            &surface_view,
            &frame.display,
            display_end_writes,
        );
        if capture_timing {
            self.timings.encode_resolve(&mut encoder);
        }
        let main_buffer = encoder.finish();
        self.queue
            .submit(callback_buffers.into_iter().chain([main_buffer]));
        free_egui_textures_after_submit(&mut self.egui_renderer, textures_delta);
        if capture_timing {
            self.timings.begin_readback();
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
        let completed_readback = if self.timings.has_pending_readback() {
            self.timings
                .poll(&self.device, self.queue.get_timestamp_period())?
                .is_some()
        } else {
            self.device.poll(wgpu::PollType::Poll)?;
            false
        };
        let events = self.device_events.try_iter().collect();
        Ok(PollOutcome {
            completed_readback,
            events,
        })
    }

    pub const fn has_pending_gpu_work(&self) -> bool {
        self.timings.has_pending_readback()
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
            extent_generation: self.extent.generation(),
            timing: self.timings.latest(),
        }
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
        if let Err(error) =
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
        let selection = select_surface(capabilities)?;
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
        if selection.format() == self.selection.format() {
            return Ok(None);
        }
        scoped_gpu_operation(&self.device, || {
            self.display
                .create_presentation_pipeline(&self.device, selection.format())
        })
        .map(Some)
    }

    fn install_surface_selection(
        &mut self,
        selection: SurfaceSelection,
        presentation_pipeline: Option<wgpu::RenderPipeline>,
    ) {
        if let Some(pipeline) = presentation_pipeline {
            self.display.install_presentation_pipeline(pipeline);
        }
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
            load: wgpu::LoadOp::Load,
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
    use super::{ResizeError, validate_render_extent};
    use crate::extent::RenderExtent;

    #[test]
    fn resize_accepts_4k_with_webgpu_default_limits() {
        let limits = wgpu::Limits::default();
        let extent = RenderExtent::new(3_840, 2_160).expect("extent is nonzero");

        assert!(validate_render_extent(extent, &limits).is_ok());
    }

    #[test]
    fn resize_rejects_each_excess_texture_dimension() {
        let limits = wgpu::Limits::default();
        let maximum = limits.max_texture_dimension_2d;
        let boundary = RenderExtent::new(maximum, maximum).expect("extent is nonzero");
        let too_wide = RenderExtent::new(maximum + 1, 1).expect("extent is nonzero");
        let too_tall = RenderExtent::new(1, maximum + 1).expect("extent is nonzero");

        assert!(matches!(
            validate_render_extent(boundary, &limits),
            Err(ResizeError::TraceRecordLimit { .. })
        ));
        assert!(matches!(
            validate_render_extent(too_wide, &limits),
            Err(ResizeError::ExtentLimit {
                width: 8_193,
                height: 1,
                max_texture_dimension_2d: 8_192,
            })
        ));
        assert!(matches!(
            validate_render_extent(too_tall, &limits),
            Err(ResizeError::ExtentLimit {
                width: 1,
                height: 8_193,
                max_texture_dimension_2d: 8_192,
            })
        ));
    }
}

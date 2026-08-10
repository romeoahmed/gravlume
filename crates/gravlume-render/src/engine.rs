use std::sync::{Arc, mpsc};

use crate::{
    capabilities::{BASELINE_FEATURES, SurfaceSelection, check_baseline_adapter, select_surface},
    display::DisplayPipeline,
    extent::{ExtentChange, ExtentTracker, RenderExtent},
    scene::{SceneCompute, SceneTarget},
    surface::{AcquireOutcome, FrameProtocol, FrameProtocolError, SurfaceDirective, directive_for},
    timing::{GpuTimings, TimingSample},
};

#[derive(Debug, thiserror::Error)]
pub enum RenderInitError {
    #[error("failed to create the native presentation surface: {0}")]
    CreateSurface(#[from] wgpu::CreateSurfaceError),
    #[error("no adapter was available for the native surface: {0}")]
    RequestAdapter(#[from] wgpu::RequestAdapterError),
    #[error("adapter {adapter:?} does not satisfy Phase 0: {reason}")]
    UnsupportedAdapter { adapter: String, reason: String },
    #[error("surface does not satisfy the SDR presentation contract: {0}")]
    SurfaceCapabilities(String),
    #[error("failed to create the Phase 0 device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
}

#[derive(Debug, thiserror::Error)]
pub enum RenderRuntimeError {
    #[error("GPU timing/readback failed: {0}")]
    Timing(String),
    #[error("non-blocking GPU poll failed: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("frame protocol violation: {0}")]
    FrameProtocol(String),
    #[error("surface acquisition reported a validation error")]
    SurfaceValidation,
    #[error("failed to recreate a lost surface: {0}")]
    RecreateSurface(#[from] wgpu::CreateSurfaceError),
    #[error("surface capabilities changed incompatibly while recovering")]
    SurfaceCapabilitiesChanged,
    #[error("nonzero extent has no matching frame-resource bundle")]
    MissingFrameResources,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameSkip {
    ZeroExtent,
    Suspended,
    Timeout,
    Occluded,
    Outdated,
    Lost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameStatus {
    Presented,
    Skipped(FrameSkip),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceEventKind {
    Validation,
    Internal,
    OutOfMemory,
    Lost,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceEvent {
    kind: DeviceEventKind,
    message: String,
}

impl DeviceEvent {
    pub const fn kind(&self) -> DeviceEventKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub const fn is_fatal(&self) -> bool {
        matches!(
            self.kind,
            DeviceEventKind::Internal | DeviceEventKind::OutOfMemory | DeviceEventKind::Lost
        )
    }
}

#[derive(Debug, Default)]
pub struct PollOutcome {
    completed_readback: bool,
    events: Vec<DeviceEvent>,
}

impl PollOutcome {
    pub const fn completed_readback(&self) -> bool {
        self.completed_readback
    }

    pub fn events(&self) -> &[DeviceEvent] {
        &self.events
    }
}

#[derive(Clone, Debug)]
pub struct RenderDiagnostics {
    adapter_name: String,
    backend: String,
    driver: String,
    surface_format: String,
    color_space: String,
    display_transfer: &'static str,
    extent_generation: u64,
    timing: Option<TimingSample>,
}

impl RenderDiagnostics {
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn driver(&self) -> &str {
        &self.driver
    }

    pub fn surface_format(&self) -> &str {
        &self.surface_format
    }

    pub fn color_space(&self) -> &str {
        &self.color_space
    }

    pub const fn display_transfer(&self) -> &'static str {
        self.display_transfer
    }

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
    scene: SceneTarget,
    display_bind_group: wgpu::BindGroup,
    needs_clear: bool,
}

impl FrameResources {
    fn new(
        device: &wgpu::Device,
        scene_compute: &SceneCompute,
        display: &DisplayPipeline,
        extent: RenderExtent,
    ) -> Self {
        let scene = scene_compute.create_target(device, extent);
        let display_bind_group = display.bind_scene(device, scene.view());
        Self {
            scene,
            display_bind_group,
            needs_clear: true,
        }
    }
}

pub struct GpuEngine {
    surface: Option<wgpu::Surface<'static>>,
    instance: wgpu::Instance,
    window: Arc<winit::window::Window>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    selection: SurfaceSelection,
    extent: ExtentTracker,
    frame_resources: Option<FrameResources>,
    scene_compute: SceneCompute,
    display: DisplayPipeline,
    egui_renderer: egui_wgpu::Renderer,
    timings: GpuTimings,
    adapter_info: wgpu::AdapterInfo,
    device_events: mpsc::Receiver<DeviceEvent>,
}

impl GpuEngine {
    pub async fn new(window: Arc<winit::window::Window>) -> Result<Self, RenderInitError> {
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
        let selection = select_surface(&surface.get_capabilities(&adapter))
            .map_err(|error| RenderInitError::SurfaceCapabilities(error.to_string()))?;
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits::default()
            .using_resolution(adapter_limits.clone())
            .using_alignment(adapter_limits);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Gravlume Phase 0 device"),
                required_features: BASELINE_FEATURES,
                required_limits,
                ..Default::default()
            })
            .await?;

        let device_events = install_device_callbacks(&device);
        let scene_compute = SceneCompute::new(&device);
        let display = DisplayPipeline::new(&device, selection.format());
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            selection.format(),
            egui_wgpu::RendererOptions::default(),
        );
        let timings = GpuTimings::new(&device);
        let initial_size = window.inner_size();
        let mut engine = Self {
            surface: Some(surface),
            instance,
            window,
            adapter,
            device,
            queue,
            selection,
            extent: ExtentTracker::default(),
            frame_resources: None,
            scene_compute,
            display,
            egui_renderer,
            timings,
            adapter_info,
            device_events,
        };
        engine.resize(initial_size.width, initial_size.height);
        Ok(engine)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        match self.extent.update(width, height) {
            ExtentChange::Unchanged => {}
            ExtentChange::Paused => self.frame_resources = None,
            ExtentChange::Rebuild { extent, .. } => {
                let replacement =
                    FrameResources::new(&self.device, &self.scene_compute, &self.display, extent);
                if let Some(surface) = &self.surface {
                    configure_surface(surface, &self.device, self.selection, extent);
                }
                self.frame_resources = Some(replacement);
            }
        }
    }

    pub fn render(
        &mut self,
        paint_jobs: &[egui::ClippedPrimitive],
        textures_delta: &egui::TexturesDelta,
        pixels_per_point: f32,
    ) -> Result<FrameStatus, RenderRuntimeError> {
        let Some(extent) = self.extent.extent() else {
            return Ok(FrameStatus::Skipped(FrameSkip::ZeroExtent));
        };
        let Some(surface) = self.surface.as_ref() else {
            return Ok(FrameStatus::Skipped(FrameSkip::Suspended));
        };

        let (surface_texture, reconfigure_after_present) = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => {
                debug_assert_eq!(
                    directive_for(AcquireOutcome::Success),
                    SurfaceDirective::Render {
                        reconfigure_after_present: false
                    }
                );
                (texture, false)
            }
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                debug_assert_eq!(
                    directive_for(AcquireOutcome::Suboptimal),
                    SurfaceDirective::Render {
                        reconfigure_after_present: true
                    }
                );
                (texture, true)
            }
            wgpu::CurrentSurfaceTexture::Timeout => {
                debug_assert_eq!(
                    directive_for(AcquireOutcome::Timeout),
                    SurfaceDirective::Skip
                );
                return Ok(FrameStatus::Skipped(FrameSkip::Timeout));
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                debug_assert_eq!(
                    directive_for(AcquireOutcome::Occluded),
                    SurfaceDirective::Skip
                );
                return Ok(FrameStatus::Skipped(FrameSkip::Occluded));
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                debug_assert_eq!(
                    directive_for(AcquireOutcome::Outdated),
                    SurfaceDirective::Reconfigure
                );
                configure_surface(surface, &self.device, self.selection, extent);
                return Ok(FrameStatus::Skipped(FrameSkip::Outdated));
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                debug_assert_eq!(
                    directive_for(AcquireOutcome::Lost),
                    SurfaceDirective::Recreate
                );
                self.recreate_surface()?;
                return Ok(FrameStatus::Skipped(FrameSkip::Lost));
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                debug_assert_eq!(
                    directive_for(AcquireOutcome::Validation),
                    SurfaceDirective::ReportValidation
                );
                return Err(RenderRuntimeError::SurfaceValidation);
            }
        };

        let mut protocol = FrameProtocol::default();
        protocol.acquired().map_err(frame_protocol_error)?;
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
                label: Some("Gravlume Phase 0 frame encoder"),
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
        debug_assert_eq!(frame.scene.extent(), extent);
        if frame.needs_clear {
            encoder.clear_texture(
                frame.scene.texture(),
                &wgpu::ImageSubresourceRange::default(),
            );
            frame.needs_clear = false;
        }
        let compute_writes = capture_timing.then(|| self.timings.compute_writes());
        self.scene_compute
            .encode(&mut encoder, &frame.scene, compute_writes);
        let display_writes = capture_timing.then(|| self.timings.display_writes());
        self.display.encode(
            &mut encoder,
            &surface_view,
            &frame.display_bind_group,
            display_writes,
        );
        encode_egui(
            &self.egui_renderer,
            &mut encoder,
            &surface_view,
            paint_jobs,
            &screen,
        );
        if capture_timing {
            self.timings.encode_resolve(&mut encoder);
        }
        for texture_id in &textures_delta.free {
            self.egui_renderer.free_texture(texture_id);
        }

        let main_buffer = encoder.finish();
        self.queue
            .submit(callback_buffers.into_iter().chain([main_buffer]));
        protocol.submitted().map_err(frame_protocol_error)?;
        if capture_timing {
            self.timings.begin_readback();
        }
        self.window.pre_present_notify();
        self.queue.present(surface_texture);
        protocol.presented().map_err(frame_protocol_error)?;
        debug_assert!(protocol.is_complete());

        if reconfigure_after_present {
            configure_surface(surface, &self.device, self.selection, extent);
        }
        Ok(FrameStatus::Presented)
    }

    pub fn poll(&mut self) -> Result<PollOutcome, RenderRuntimeError> {
        let completed_readback = if self.timings.has_pending_readback() {
            self.timings
                .poll(&self.device, self.queue.get_timestamp_period())
                .map_err(|error| RenderRuntimeError::Timing(error.to_string()))?
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
    }

    pub fn resume_surface(&mut self) -> Result<(), RenderRuntimeError> {
        if self.surface.is_none() {
            self.recreate_surface()?;
        }
        Ok(())
    }

    pub fn diagnostics(&self) -> RenderDiagnostics {
        RenderDiagnostics {
            adapter_name: self.adapter_info.name.clone(),
            backend: format!("{:?}", self.adapter_info.backend),
            driver: format!(
                "{} {}",
                self.adapter_info.driver, self.adapter_info.driver_info
            )
            .trim()
            .to_owned(),
            surface_format: format!("{:?}", self.selection.format()),
            color_space: format!("{:?}", self.selection.color_space()),
            display_transfer: if self.selection.requires_manual_srgb_encoding() {
                "shader sRGB encoding"
            } else {
                "surface sRGB encoding"
            },
            extent_generation: self.extent.generation(),
            timing: self.timings.latest(),
        }
    }

    fn recreate_surface(&mut self) -> Result<(), RenderRuntimeError> {
        let replacement = self.instance.create_surface(Arc::clone(&self.window))?;
        let selection = select_surface(&replacement.get_capabilities(&self.adapter))
            .map_err(|_| RenderRuntimeError::SurfaceCapabilitiesChanged)?;
        if selection != self.selection {
            return Err(RenderRuntimeError::SurfaceCapabilitiesChanged);
        }
        if let Some(extent) = self.extent.extent() {
            configure_surface(&replacement, &self.device, selection, extent);
        }
        self.surface = Some(replacement);
        Ok(())
    }
}

fn frame_protocol_error(error: FrameProtocolError) -> RenderRuntimeError {
    RenderRuntimeError::FrameProtocol(error.to_string())
}

fn configure_surface(
    surface: &wgpu::Surface<'_>,
    device: &wgpu::Device,
    selection: SurfaceSelection,
    extent: RenderExtent,
) {
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

fn install_device_callbacks(device: &wgpu::Device) -> mpsc::Receiver<DeviceEvent> {
    let (sender, receiver) = mpsc::channel();
    let uncaptured_sender = sender.clone();
    device.on_uncaptured_error(Arc::new(move |error| {
        let (kind, message) = match error {
            wgpu::Error::Validation { description, .. } => {
                (DeviceEventKind::Validation, description)
            }
            wgpu::Error::Internal { description, .. } => (DeviceEventKind::Internal, description),
            wgpu::Error::OutOfMemory { .. } => {
                (DeviceEventKind::OutOfMemory, "GPU out of memory".to_owned())
            }
        };
        if uncaptured_sender
            .send(DeviceEvent { kind, message })
            .is_err()
        {
            tracing::debug!("device event receiver dropped");
        }
    }));
    device.set_device_lost_callback(move |reason, message| {
        let message = format!("{reason:?}: {message}");
        if sender
            .send(DeviceEvent {
                kind: DeviceEventKind::Lost,
                message,
            })
            .is_err()
        {
            tracing::debug!("device event receiver dropped");
        }
    });
    receiver
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_backend_is_narrowed_to_the_release_contract() {
        #[cfg(target_os = "macos")]
        assert_eq!(crate::native_backends(), wgpu::Backends::METAL);

        #[cfg(any(target_os = "windows", target_os = "linux"))]
        assert_eq!(crate::native_backends(), wgpu::Backends::VULKAN);
    }
}

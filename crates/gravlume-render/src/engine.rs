use std::sync::{Arc, mpsc};

use crate::{
    capabilities::{BASELINE_FEATURES, SurfaceSelection, check_baseline_adapter, select_surface},
    display::DisplayPipeline,
    extent::{ExtentChange, ExtentTracker, RenderExtent},
    gpu_error::{
        DeviceEvent, GpuErrorScopes, RenderInitError, RenderRuntimeError, ResizeError,
        install_device_callbacks, scoped_gpu_operation,
    },
    scene::{SceneCompute, SceneTarget},
    surface::{
        AcquireOutcome, FrameProtocol, FrameProtocolError, FrameSkip, SurfaceDirective,
        directive_for,
    },
    timing::{GpuTimings, TimingSample},
};

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
    pub const fn completed_readback(&self) -> bool {
        self.completed_readback
    }

    #[must_use]
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
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    #[must_use]
    pub fn backend(&self) -> &str {
        &self.backend
    }

    #[must_use]
    pub fn driver(&self) -> &str {
        &self.driver
    }

    #[must_use]
    pub fn surface_format(&self) -> &str {
        &self.surface_format
    }

    #[must_use]
    pub fn color_space(&self) -> &str {
        &self.color_space
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
    scene: SceneTarget,
    display_bind_group: wgpu::BindGroup,
    needs_clear: bool,
}

enum SurfaceAcquisition {
    Render {
        texture: wgpu::SurfaceTexture,
        reconfigure_after_present: bool,
    },
    Skip(FrameSkip),
    Reconfigure,
    Recreate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissingSurfaceDirective {
    SkipSuspended,
    Recreate,
}

const fn missing_surface_directive(surface_suspended: bool) -> MissingSurfaceDirective {
    if surface_suspended {
        MissingSurfaceDirective::SkipSuspended
    } else {
        MissingSurfaceDirective::Recreate
    }
}

enum PreparedSurfaceFrame {
    Render {
        texture: wgpu::SurfaceTexture,
        reconfigure_after_present: bool,
    },
    Skip(FrameSkip),
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
    surface_suspended: bool,
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
    device_event_sender: mpsc::Sender<DeviceEvent>,
    device_events: mpsc::Receiver<DeviceEvent>,
}

impl GpuEngine {
    /// Creates the Phase 0 GPU device and presentation resources for `window`.
    ///
    /// # Errors
    ///
    /// Returns an error when the native surface, adapter, required capabilities, or device cannot
    /// be initialized.
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

        let (device_event_sender, device_events) = install_device_callbacks(&device);
        let resource_scopes = GpuErrorScopes::push(&device);
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
            surface_suspended: false,
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
            device_event_sender,
            device_events,
        };
        let resize_result = engine.resize(initial_size.width, initial_size.height);
        resource_scopes
            .finish()
            .await
            .map_err(|source| RenderInitError::GpuResource {
                stage: "Phase 0 GPU resources",
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
        let (candidate_extent, change) = self.extent.preview_update(width, height);
        match change {
            ExtentChange::Unchanged => return Ok(()),
            ExtentChange::Paused => {
                self.extent = candidate_extent;
                self.frame_resources = None;
            }
            ExtentChange::Rebuild { extent, .. } => {
                validate_extent_limit(extent, self.device.limits().max_texture_dimension_2d)?;
                if let Some(surface) = &self.surface {
                    let capabilities = surface.get_capabilities(&self.adapter);
                    if !self.selection.is_supported_by(&capabilities) {
                        return Err(ResizeError::SurfaceCapabilitiesChanged);
                    }
                }

                let replacement = scoped_gpu_operation(&self.device, || {
                    FrameResources::new(&self.device, &self.scene_compute, &self.display, extent)
                })
                .map_err(|source| ResizeError::GpuResource {
                    stage: "create frame resources",
                    source,
                })?;
                if let Some(surface) = &self.surface {
                    configure_surface_scoped(surface, &self.device, self.selection, extent)
                        .map_err(|source| ResizeError::GpuResource {
                            stage: "configure the presentation surface",
                            source,
                        })?;
                }

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
    /// Returns an error when surface recovery, frame-resource validation, or the internal frame
    /// protocol fails.
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
        let main_buffer = encoder.finish();
        self.queue
            .submit(callback_buffers.into_iter().chain([main_buffer]));
        protocol.submitted().map_err(frame_protocol_error)?;
        free_egui_textures_after_submit(&mut self.egui_renderer, textures_delta);
        protocol.textures_released().map_err(frame_protocol_error)?;
        if capture_timing {
            self.timings.begin_readback();
        }
        self.window.pre_present_notify();
        self.queue.present(surface_texture);
        protocol.presented().map_err(frame_protocol_error)?;
        debug_assert!(protocol.is_complete());

        if reconfigure_after_present {
            self.reconfigure_surface(extent);
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

    fn prepare_surface_frame(
        &mut self,
        extent: RenderExtent,
    ) -> Result<PreparedSurfaceFrame, RenderRuntimeError> {
        let acquisition = {
            match self.surface.as_ref() {
                Some(surface) => acquire_surface_frame(surface),
                None => match missing_surface_directive(self.surface_suspended) {
                    MissingSurfaceDirective::SkipSuspended => {
                        return Ok(PreparedSurfaceFrame::Skip(FrameSkip::Suspended));
                    }
                    MissingSurfaceDirective::Recreate => SurfaceAcquisition::Recreate,
                },
            }
        };
        let frame = match acquisition {
            SurfaceAcquisition::Render {
                texture,
                reconfigure_after_present,
            } => PreparedSurfaceFrame::Render {
                texture,
                reconfigure_after_present,
            },
            SurfaceAcquisition::Skip(reason) => PreparedSurfaceFrame::Skip(reason),
            SurfaceAcquisition::Reconfigure => {
                PreparedSurfaceFrame::Skip(if self.reconfigure_surface(extent) {
                    FrameSkip::Outdated
                } else {
                    FrameSkip::Validation
                })
            }
            SurfaceAcquisition::Recreate => {
                PreparedSurfaceFrame::Skip(if self.recreate_surface()? {
                    FrameSkip::Lost
                } else {
                    FrameSkip::Validation
                })
            }
        };
        Ok(frame)
    }

    fn reconfigure_surface(&self, extent: RenderExtent) -> bool {
        let result = {
            let Some(surface) = self.surface.as_ref() else {
                return false;
            };
            configure_surface_checked(surface, &self.adapter, &self.device, self.selection, extent)
        };
        match result {
            Ok(()) => true,
            Err(event) => {
                self.enqueue_device_event(event);
                false
            }
        }
    }

    fn recreate_surface(&mut self) -> Result<bool, RenderRuntimeError> {
        let replacement = self.instance.create_surface(Arc::clone(&self.window))?;
        let selection = select_surface(&replacement.get_capabilities(&self.adapter))
            .map_err(|_| RenderRuntimeError::SurfaceCapabilitiesChanged)?;
        if selection != self.selection {
            return Err(RenderRuntimeError::SurfaceCapabilitiesChanged);
        }
        if let Some(extent) = self.extent.extent()
            && let Err(event) = configure_surface_checked(
                &replacement,
                &self.adapter,
                &self.device,
                selection,
                extent,
            )
        {
            self.enqueue_device_event(event);
            return Ok(false);
        }
        self.surface = Some(replacement);
        Ok(true)
    }

    fn enqueue_device_event(&self, event: DeviceEvent) {
        if self.device_event_sender.send(event).is_err() {
            tracing::debug!("device event receiver dropped");
        }
    }
}

fn frame_protocol_error(error: FrameProtocolError) -> RenderRuntimeError {
    RenderRuntimeError::FrameProtocol(error.to_string())
}

const fn validate_extent_limit(
    extent: RenderExtent,
    max_texture_dimension_2d: u32,
) -> Result<(), ResizeError> {
    if extent.width() > max_texture_dimension_2d || extent.height() > max_texture_dimension_2d {
        return Err(ResizeError::ExtentLimit {
            width: extent.width(),
            height: extent.height(),
            max_texture_dimension_2d,
        });
    }
    Ok(())
}

fn acquire_surface_frame(surface: &wgpu::Surface<'_>) -> SurfaceAcquisition {
    match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture) => {
            debug_assert_eq!(
                directive_for(AcquireOutcome::Success),
                SurfaceDirective::Render {
                    reconfigure_after_present: false
                }
            );
            SurfaceAcquisition::Render {
                texture,
                reconfigure_after_present: false,
            }
        }
        wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
            debug_assert_eq!(
                directive_for(AcquireOutcome::Suboptimal),
                SurfaceDirective::Render {
                    reconfigure_after_present: true
                }
            );
            SurfaceAcquisition::Render {
                texture,
                reconfigure_after_present: true,
            }
        }
        wgpu::CurrentSurfaceTexture::Timeout => {
            debug_assert_eq!(
                directive_for(AcquireOutcome::Timeout),
                SurfaceDirective::Skip(FrameSkip::Timeout)
            );
            SurfaceAcquisition::Skip(FrameSkip::Timeout)
        }
        wgpu::CurrentSurfaceTexture::Occluded => {
            debug_assert_eq!(
                directive_for(AcquireOutcome::Occluded),
                SurfaceDirective::Skip(FrameSkip::Occluded)
            );
            SurfaceAcquisition::Skip(FrameSkip::Occluded)
        }
        wgpu::CurrentSurfaceTexture::Outdated => {
            debug_assert_eq!(
                directive_for(AcquireOutcome::Outdated),
                SurfaceDirective::Reconfigure
            );
            SurfaceAcquisition::Reconfigure
        }
        wgpu::CurrentSurfaceTexture::Lost => {
            debug_assert_eq!(
                directive_for(AcquireOutcome::Lost),
                SurfaceDirective::Recreate
            );
            SurfaceAcquisition::Recreate
        }
        wgpu::CurrentSurfaceTexture::Validation => {
            debug_assert_eq!(
                directive_for(AcquireOutcome::Validation),
                SurfaceDirective::Skip(FrameSkip::Validation)
            );
            SurfaceAcquisition::Skip(FrameSkip::Validation)
        }
    }
}

fn configure_surface_checked(
    surface: &wgpu::Surface<'_>,
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    selection: SurfaceSelection,
    extent: RenderExtent,
) -> Result<(), DeviceEvent> {
    if !selection.is_supported_by(&surface.get_capabilities(adapter)) {
        return Err(DeviceEvent::validation(
            "surface no longer supports the active presentation configuration",
        ));
    }
    configure_surface_scoped(surface, device, selection, extent).map_err(|error| {
        DeviceEvent::from_wgpu("failed to configure the presentation surface", error)
    })
}

fn configure_surface_scoped(
    surface: &wgpu::Surface<'_>,
    device: &wgpu::Device,
    selection: SurfaceSelection,
    extent: RenderExtent,
) -> Result<(), wgpu::Error> {
    scoped_gpu_operation(device, || {
        configure_surface(surface, device, selection, extent);
    })
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
        MissingSurfaceDirective, ResizeError, missing_surface_directive, validate_extent_limit,
    };
    use crate::extent::RenderExtent;

    #[test]
    fn native_backend_is_narrowed_to_the_release_contract() {
        #[cfg(target_os = "macos")]
        assert_eq!(crate::native_backends(), wgpu::Backends::METAL);

        #[cfg(any(target_os = "windows", target_os = "linux"))]
        assert_eq!(crate::native_backends(), wgpu::Backends::VULKAN);
    }

    #[test]
    fn resize_rejects_each_dimension_above_the_device_limit_before_allocation() {
        let maximum = 8_192;
        let too_wide = RenderExtent::new(maximum + 1, 1).expect("extent is nonzero");
        let too_tall = RenderExtent::new(1, maximum + 1).expect("extent is nonzero");

        assert!(matches!(
            validate_extent_limit(too_wide, maximum),
            Err(ResizeError::ExtentLimit {
                width: 8_193,
                height: 1,
                max_texture_dimension_2d: 8_192,
            })
        ));
        assert!(matches!(
            validate_extent_limit(too_tall, maximum),
            Err(ResizeError::ExtentLimit {
                width: 1,
                height: 8_193,
                max_texture_dimension_2d: 8_192,
            })
        ));
    }

    #[test]
    fn missing_surface_retries_only_after_the_application_resumes() {
        assert_eq!(
            missing_surface_directive(true),
            MissingSurfaceDirective::SkipSuspended
        );
        assert_eq!(
            missing_surface_directive(false),
            MissingSurfaceDirective::Recreate
        );
    }
}

use std::{
    ffi::OsStr,
    num::NonZeroU32,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use gravlume_domain::{
    Angle, KerrNewmanSpacetime, KerrSchildCoordinates, Observation, PhysicalScene,
    PhysicalSceneDraft, StationaryObserverDraft, ValidationReport, ViewportProjection,
};
use gravlume_native_display::DisplayMonitor;
use gravlume_render::{
    DeviceEvent, DisplayState, FrameSkip, FrameStatus, GpuEngine, HdrParameters, RenderDiagnostics,
    RenderInitError, RenderRuntimeError, ResizeError, UnknownDisplayState,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    error::{EventLoopError, OsError},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    window::{Window, WindowId},
};

use crate::{Launch, lifecycle::Lifecycle};

const PENDING_GPU_POLL_INTERVAL: Duration = Duration::from_millis(2);
const RETRY_FRAME_INTERVAL: Duration = Duration::from_millis(16);
const SMOKE_ONCE_ENV: &str = "GRAVLUME_SMOKE_ONCE";
const UI_FONT_NAME: &str = "Noto Sans SC";
const UI_FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/NotoSansSC-Regular.otf");

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("failed to create the desktop event loop: {0}")]
    EventLoop(#[from] EventLoopError),
    #[error("failed to create the native window: {0}")]
    Window(#[from] OsError),
    #[error("failed to monitor the native display: {0}")]
    NativeDisplay(#[from] gravlume_native_display::MonitorError),
    #[error("failed to initialize the GPU renderer: {0}")]
    RenderInit(#[from] RenderInitError),
    #[error("failed to construct the validated default observation: {0}")]
    DefaultObservation(#[from] ValidationReport),
    #[error("rendering failed: {0}")]
    RenderRuntime(#[from] RenderRuntimeError),
    #[error("fatal resize failure: {0}")]
    Resize(#[from] ResizeError),
    #[error("fatal GPU event: {0}")]
    Device(#[from] DeviceEvent),
}

/// Runs the native Gravlume application until the event loop exits.
///
/// # Errors
///
/// Returns an error when event-loop, window, renderer, or device initialization/runtime fails.
pub fn run(launch: Launch) -> Result<(), RunError> {
    let mut builder = EventLoop::<AppEvent>::with_user_event();
    let event_loop = builder.build()?;
    let mut app = DesktopApp::new(launch, event_loop.create_proxy());
    event_loop.run_app(&mut app)?;
    app.fatal_error.take().map_or(Ok(()), Err)
}

struct WindowState {
    // Native observers must be removed before their NSWindow/HWND is destroyed.
    display_monitor: DisplayMonitor,
    output_event_pending: Arc<AtomicBool>,
    egui: egui_winit::State,
    window: Arc<Window>,
}

#[derive(Clone, Copy, Debug)]
enum AppEvent {
    RepaintAt(Instant),
    OutputStateDirty,
}

struct DesktopApp {
    launch: Launch,
    lifecycle: Lifecycle,
    window: Option<WindowState>,
    renderer: Option<GpuEngine>,
    egui_context: egui::Context,
    event_proxy: EventLoopProxy<AppEvent>,
    schedule: EventLoopSchedule,
    pending_textures: egui::TexturesDelta,
    last_device_event: Option<DeviceEvent>,
    fatal_error: Option<RunError>,
    presented_frames: u64,
    smoke_once: bool,
    exit_requested: bool,
}

impl DesktopApp {
    fn new(launch: Launch, event_proxy: EventLoopProxy<AppEvent>) -> Self {
        let egui_context = egui::Context::default();
        install_ui_font(&egui_context);
        Self {
            launch,
            lifecycle: Lifecycle::default(),
            window: None,
            renderer: None,
            egui_context,
            event_proxy,
            schedule: EventLoopSchedule::default(),
            pending_textures: egui::TexturesDelta::default(),
            last_device_event: None,
            fatal_error: None,
            presented_frames: 0,
            smoke_once: std::env::var_os(SMOKE_ONCE_ENV)
                .is_some_and(|value| value == OsStr::new("1")),
            exit_requested: false,
        }
    }

    fn initialize(&mut self, event_loop: &ActiveEventLoop) -> Result<(), RunError> {
        let window = if let Some(window_state) = &self.window {
            Arc::clone(&window_state.window)
        } else {
            let window_preferences = self.launch.window();
            let attributes = Window::default_attributes()
                .with_title(window_preferences.title())
                .with_inner_size(PhysicalSize::new(
                    window_preferences.width(),
                    window_preferences.height(),
                ));
            let window = Arc::new(event_loop.create_window(attributes)?);
            let egui = egui_winit::State::new(
                self.egui_context.clone(),
                egui::ViewportId::ROOT,
                window.as_ref(),
                None,
                window.theme(),
                None,
            );
            let repaint_proxy = self.event_proxy.clone();
            self.egui_context
                .set_request_repaint_callback(move |request| {
                    if request.delay == Duration::MAX {
                        return;
                    }
                    let now = Instant::now();
                    let deadline = now.checked_add(request.delay).unwrap_or(now);
                    if repaint_proxy
                        .send_event(AppEvent::RepaintAt(deadline))
                        .is_err()
                    {
                        tracing::debug!("event loop closed before egui repaint request");
                    }
                });
            let output_proxy = self.event_proxy.clone();
            let output_event_pending = Arc::new(AtomicBool::new(false));
            let callback_pending = Arc::clone(&output_event_pending);
            let display_monitor = DisplayMonitor::new(window.as_ref(), move || {
                if callback_pending.swap(true, Ordering::AcqRel) {
                    return;
                }
                if output_proxy.send_event(AppEvent::OutputStateDirty).is_err() {
                    callback_pending.store(false, Ordering::Release);
                    tracing::debug!("event loop closed before display-state notification");
                }
            })?;
            self.window = Some(WindowState {
                display_monitor,
                output_event_pending,
                egui,
                window: Arc::clone(&window),
            });
            window
        };

        let display_state = self.current_display_state();
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.resume_surface()?;
            renderer.refresh_output(display_state)?;
            window.request_redraw();
            return Ok(());
        }
        let size = window.inner_size();
        let observation = default_observation(size.width, size.height)?;
        let mut renderer =
            pollster::block_on(GpuEngine::new(window.clone(), &observation, display_state))?;
        // Configuring extended-linear output can arm macOS EDR. Re-read the live snapshot after
        // the surface exists so current headroom is not left at its pre-configuration value.
        renderer.refresh_output(self.current_display_state())?;
        let diagnostics = renderer.diagnostics();
        tracing::info!(
            adapter = diagnostics.adapter_name(),
            backend = diagnostics.backend(),
            driver = diagnostics.driver(),
            format = diagnostics.surface_format(),
            color_space = diagnostics.color_space(),
            transfer = diagnostics.display_transfer(),
            "initialized interactive desktop renderer"
        );
        self.renderer = Some(renderer);
        window.request_redraw();
        Ok(())
    }

    fn draw_frame(&mut self, event_loop: &ActiveEventLoop) {
        let render_result = {
            let (Some(window_state), Some(renderer)) =
                (self.window.as_mut(), self.renderer.as_mut())
            else {
                return;
            };
            let window = &window_state.window;

            let diagnostics = renderer.diagnostics();
            let device_event = self.last_device_event.as_ref();
            let raw_input = window_state.egui.take_egui_input(window);
            let output = self.egui_context.run_ui(raw_input, |root_ui| {
                show_overlay(root_ui.ctx(), &diagnostics, device_event);
            });
            let egui::FullOutput {
                platform_output,
                textures_delta,
                shapes,
                pixels_per_point,
                ..
            } = output;
            window_state.egui.handle_platform_output_with_event_loop(
                window,
                event_loop,
                platform_output,
            );
            let paint_jobs = self.egui_context.tessellate(shapes, pixels_per_point);
            self.pending_textures.append(textures_delta);
            renderer.render(&paint_jobs, &self.pending_textures, pixels_per_point)
        };

        match render_result {
            Ok(FrameStatus::Presented) => {
                self.pending_textures.clear();
                self.presented_frames += 1;
            }
            Ok(FrameStatus::Skipped(FrameSkip::Timeout)) => {
                self.schedule
                    .request_repaint(Instant::now(), RETRY_FRAME_INTERVAL);
            }
            Ok(FrameStatus::Skipped(FrameSkip::Outdated | FrameSkip::Lost)) => {
                self.schedule
                    .request_repaint(Instant::now(), Duration::ZERO);
            }
            Ok(FrameStatus::Skipped(
                FrameSkip::Validation
                | FrameSkip::Occluded
                | FrameSkip::ZeroExtent
                | FrameSkip::Suspended,
            )) => {}
            Err(error) => self.fail(event_loop, error.into()),
        }
    }

    fn process_device_events(&mut self, event_loop: &ActiveEventLoop, events: Vec<DeviceEvent>) {
        let mut report_changed = false;
        for event in events {
            tracing::error!(kind = ?event.kind(), message = event.message(), "GPU device event");
            if event.is_fatal() {
                self.fail(event_loop, event.into());
                return;
            }
            if self.last_device_event.as_ref() != Some(&event) {
                self.last_device_event = Some(event);
                report_changed = true;
            }
        }
        if report_changed {
            self.request_redraw();
        }
    }

    fn resize_renderer(&mut self, event_loop: &ActiveEventLoop, width: u32, height: u32) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        match renderer.resize(width, height) {
            Ok(()) => {
                if width != 0 && height != 0 {
                    self.request_redraw();
                }
            }
            Err(error) => {
                let is_fatal = error.is_fatal();
                let event = DeviceEvent::from(&error);
                tracing::error!(kind = ?event.kind(), message = event.message(), width, height, "GPU resize rejected");
                let report_changed = self.last_device_event.as_ref() != Some(&event);
                if report_changed {
                    self.last_device_event = Some(event);
                }
                if is_fatal {
                    self.fail(event_loop, error.into());
                } else if report_changed {
                    self.request_redraw();
                }
            }
        }
    }

    fn request_redraw(&self) {
        if let Some(window) = &self.window {
            window.window.request_redraw();
        }
    }

    fn current_display_state(&self) -> DisplayState {
        self.window.as_ref().map_or(
            DisplayState::Unknown(UnknownDisplayState::PlatformIntegrationUnavailable),
            |state| map_dynamic_range(state.display_monitor.dynamic_range()),
        )
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: RunError) {
        if self.fatal_error.is_none() {
            tracing::error!(error = %error, "desktop runtime is stopping");
            self.lifecycle.fail();
            self.pending_textures.clear();
            self.fatal_error = Some(error);
        }
        self.request_exit(event_loop);
    }

    fn request_exit(&mut self, event_loop: &ActiveEventLoop) {
        if !self.exit_requested {
            self.exit_requested = true;
            self.pending_textures.clear();
            if let Some(window) = self.window.as_mut() {
                window.display_monitor.shutdown();
            }
        }
        self.finish_exit(event_loop);
    }

    fn finish_exit(&self, event_loop: &ActiveEventLoop) {
        let shutdown_complete = self
            .window
            .as_ref()
            .is_none_or(|window| window.display_monitor.shutdown_complete());
        if shutdown_complete {
            event_loop.exit();
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

fn install_ui_font(context: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        UI_FONT_NAME.to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(UI_FONT_BYTES)),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push(UI_FONT_NAME.to_owned());
    }
    context.set_fonts(fonts);
}

impl ApplicationHandler<AppEvent> for DesktopApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.lifecycle.resume()
            && let Err(error) = self.initialize(event_loop)
        {
            self.fail(event_loop, error);
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::RepaintAt(deadline) if !self.exit_requested => {
                self.schedule.request_repaint_at(deadline);
            }
            AppEvent::RepaintAt(_) => {}
            AppEvent::OutputStateDirty => {
                if let Some(window) = self.window.as_ref() {
                    window.output_event_pending.store(false, Ordering::Release);
                }
                if self.exit_requested {
                    self.finish_exit(event_loop);
                    return;
                }
                let display_state = self.current_display_state();
                if let Some(renderer) = self.renderer.as_mut()
                    && let Err(error) = renderer.refresh_output(display_state)
                {
                    self.fail(event_loop, error.into());
                    return;
                }
                self.request_redraw();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self.exit_requested {
            return;
        }
        let Some(window_state) = self.window.as_mut() else {
            return;
        };
        let window = Arc::clone(&window_state.window);
        if window.id() != window_id {
            return;
        }

        let response = window_state.egui.on_window_event(&window, &event);
        // This event is the requested repaint; echoing it would queue another frame.
        if response.repaint && !matches!(&event, WindowEvent::RedrawRequested) {
            window.request_redraw();
        }

        match event {
            WindowEvent::CloseRequested => self.request_exit(event_loop),
            WindowEvent::Resized(size) => {
                self.resize_renderer(event_loop, size.width, size.height);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = window.inner_size();
                self.resize_renderer(event_loop, size.width, size.height);
            }
            WindowEvent::RedrawRequested => self.draw_frame(event_loop),
            _ if response.consumed => {}
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.exit_requested {
            self.finish_exit(event_loop);
            return;
        }
        let poll_result = self.renderer.as_mut().map(GpuEngine::poll);
        match poll_result {
            Some(Ok(outcome)) => {
                let (completed_readback, events) = outcome.into_parts();
                self.process_device_events(event_loop, events);
                let trace_complete = self
                    .renderer
                    .as_ref()
                    .is_some_and(GpuEngine::trace_is_complete);
                if self.smoke_once
                    && self.presented_frames > 0
                    && completed_readback
                    && trace_complete
                {
                    if let Some(diagnostics) = self.renderer.as_ref().map(GpuEngine::diagnostics) {
                        tracing::info!(
                            trace_batches = diagnostics.completed_trace_batches(),
                            total_trace_compute_ms = diagnostics.total_trace_compute_ms(),
                            maximum_trace_batch_ms = diagnostics.maximum_trace_batch_ms(),
                            "interactive one-frame smoke completed"
                        );
                    }
                    self.request_exit(event_loop);
                    return;
                }
            }
            Some(Err(error)) => {
                self.fail(event_loop, error.into());
                return;
            }
            None => {}
        }
        if self.fatal_error.is_some() {
            return;
        }

        if let Some(window) = self.window.as_mut() {
            window.display_monitor.refresh();
        }

        let now = Instant::now();
        let has_pending_gpu_work = self
            .renderer
            .as_ref()
            .is_some_and(GpuEngine::has_pending_gpu_work);
        self.schedule.after_gpu_poll(now, has_pending_gpu_work);
        if self
            .renderer
            .as_ref()
            .is_some_and(GpuEngine::trace_needs_redraw)
        {
            self.schedule.request_repaint(now, Duration::ZERO);
        }
        if self.schedule.take_due_repaint(now) {
            self.request_redraw();
        }
        let native_dispatch_deadline = self
            .window
            .as_ref()
            .and_then(|window| window.display_monitor.next_dispatch_deadline());
        match earliest_deadline(self.schedule.next_wake(), native_dispatch_deadline) {
            Some(deadline) => event_loop.set_control_flow(ControlFlow::WaitUntil(deadline)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if self.lifecycle.suspend()
            && let Some(renderer) = self.renderer.as_mut()
        {
            renderer.suspend();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_mut() {
            window.display_monitor.shutdown();
        }
    }
}

fn map_dynamic_range(dynamic_range: gravlume_native_display::DynamicRange) -> DisplayState {
    match dynamic_range {
        gravlume_native_display::DynamicRange::Hdr {
            tone_map_headroom,
            reference_white_scale,
        } => HdrParameters::new(tone_map_headroom, reference_white_scale).map_or(
            DisplayState::Unknown(UnknownDisplayState::StateQueryFailed),
            DisplayState::Hdr,
        ),
        gravlume_native_display::DynamicRange::Sdr => DisplayState::Sdr,
        gravlume_native_display::DynamicRange::Suppressed => DisplayState::Suppressed,
        gravlume_native_display::DynamicRange::Unknown(reason) => {
            DisplayState::Unknown(match reason {
                gravlume_native_display::UnknownDisplayState::PlatformIntegrationUnavailable => {
                    UnknownDisplayState::PlatformIntegrationUnavailable
                }
                gravlume_native_display::UnknownDisplayState::UnsupportedOsVersion => {
                    UnknownDisplayState::UnsupportedOsVersion
                }
                gravlume_native_display::UnknownDisplayState::StateQueryFailed => {
                    UnknownDisplayState::StateQueryFailed
                }
                gravlume_native_display::UnknownDisplayState::WaylandColorManagementUnavailable => {
                    UnknownDisplayState::WaylandColorManagementUnavailable
                }
                gravlume_native_display::UnknownDisplayState::WaylandProtocolTooOld => {
                    UnknownDisplayState::WaylandProtocolTooOld
                }
                gravlume_native_display::UnknownDisplayState::WaylandEncodingUnavailable => {
                    UnknownDisplayState::WaylandEncodingUnavailable
                }
            })
        }
    }
}

fn earliest_deadline(first: Option<Instant>, second: Option<Instant>) -> Option<Instant> {
    match (first, second) {
        (Some(first), Some(second)) => Some(first.min(second)),
        (Some(deadline), None) | (None, Some(deadline)) => Some(deadline),
        (None, None) => None,
    }
}

fn show_overlay(
    context: &egui::Context,
    diagnostics: &RenderDiagnostics<'_>,
    device_event: Option<&DeviceEvent>,
) {
    egui::Window::new("Gravlume")
        .default_pos([16.0, 16.0])
        .resizable(false)
        .collapsible(false)
        .show(context, |ui| {
            ui.strong("Interactive Kerr black-hole lensing");
            match diagnostics.trace_completion() {
                Some(completion) if completion < 1.0 => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(format!(
                            "Tracing the native-resolution image: {:.0}%",
                            completion * 100.0
                        ));
                    });
                }
                _ => {
                    ui.label("Ready — the complete native-resolution image is published.");
                }
            }
            ui.label("Kerr spin 0.8 · observer radius 30 M · vertical field of view 45°.");
            ui.label(
                "Black is the event-horizon shadow; the colored grid is a lensed sky test pattern.",
            );
            ui.label("This preview validates geometry; it is not an accretion-disk simulation.");
            ui.label(format!(
                "Display output: {}",
                diagnostics.display_transfer()
            ));
            if let Some(event) = device_event {
                ui.separator();
                ui.colored_label(
                    egui::Color32::from_rgb(255, 170, 80),
                    format!("GPU {:?}: {}", event.kind(), event.message()),
                );
            }
        });
}

#[cfg(test)]
mod font_tests {
    use super::install_ui_font;

    #[test]
    fn bundled_ui_font_covers_required_unicode_fallbacks() {
        let context = egui::Context::default();
        install_ui_font(&context);
        let mut output = context.run_ui(egui::RawInput::default(), |_| {});

        context.fonts_mut(|fonts| {
            assert!(fonts.has_glyphs(
                &egui::FontId::proportional(14.0),
                "\u{4e2d}\u{6587} \u{2192} Kerr\u{2013}Schild \u{00b7} HDR/SDR"
            ));
        });
        output.textures_delta.clear();
    }
}

fn default_observation(width: u32, height: u32) -> Result<Observation, ValidationReport> {
    let spacetime = KerrNewmanSpacetime::new(1.0, 0.8, 0.0, KerrSchildCoordinates::Outgoing)?;
    let observer_xyz = spacetime.oblate_to_cartesian(30.0, std::f64::consts::FRAC_PI_3, 0.0);
    let observer = StationaryObserverDraft::new(
        [0.0, observer_xyz[0], observer_xyz[1], observer_xyz[2]],
        [0.0; 4],
        [0.0, 0.0, 1.0],
        1.0,
    );
    let scene = PhysicalScene::commit(PhysicalSceneDraft::new(
        1.0,
        0.8,
        0.0,
        KerrSchildCoordinates::Outgoing,
        observer,
    ))?;
    let projection = ViewportProjection::perspective(
        NonZeroU32::new(width).unwrap_or(NonZeroU32::MIN),
        NonZeroU32::new(height).unwrap_or(NonZeroU32::MIN),
        Angle::from_radians(std::f64::consts::FRAC_PI_4)?,
    )?;
    Ok(Observation::new(scene, projection))
}

/// Keeps timer wakeups for GPU progress separate from repaint requests.
///
/// Source: <https://docs.rs/winit/0.30.13/winit/application/trait.ApplicationHandler.html#method.about_to_wait>
#[derive(Debug, Default)]
struct EventLoopSchedule {
    repaint_deadline: Option<Instant>,
    gpu_poll_deadline: Option<Instant>,
}

impl EventLoopSchedule {
    fn request_repaint(&mut self, now: Instant, delay: Duration) {
        if delay != Duration::MAX {
            self.request_repaint_at(now.checked_add(delay).unwrap_or(now));
        }
    }

    fn request_repaint_at(&mut self, deadline: Instant) {
        self.repaint_deadline = Some(
            self.repaint_deadline
                .map_or(deadline, |current| current.min(deadline)),
        );
    }

    fn after_gpu_poll(&mut self, now: Instant, has_pending_work: bool) {
        if !has_pending_work {
            self.gpu_poll_deadline = None;
            return;
        }
        if self
            .gpu_poll_deadline
            .is_none_or(|deadline| deadline <= now)
        {
            self.gpu_poll_deadline =
                Some(now.checked_add(PENDING_GPU_POLL_INTERVAL).unwrap_or(now));
        }
    }

    fn take_due_repaint(&mut self, now: Instant) -> bool {
        if self
            .repaint_deadline
            .is_some_and(|deadline| deadline <= now)
        {
            self.repaint_deadline = None;
            true
        } else {
            false
        }
    }

    fn next_wake(&self) -> Option<Instant> {
        match (self.repaint_deadline, self.gpu_poll_deadline) {
            (Some(repaint), Some(gpu_poll)) => Some(repaint.min(gpu_poll)),
            (Some(deadline), None) | (None, Some(deadline)) => Some(deadline),
            (None, None) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::EventLoopSchedule;

    #[test]
    fn gpu_progress_never_consumes_or_requests_a_repaint() {
        let now = std::time::Instant::now();
        let mut schedule = EventLoopSchedule::default();

        schedule.request_repaint(now, Duration::from_millis(20));
        schedule.request_repaint(now, Duration::from_millis(10));
        schedule.request_repaint(now, Duration::from_millis(15));
        schedule.after_gpu_poll(now, true);

        let first_poll = now + super::PENDING_GPU_POLL_INTERVAL;
        assert_eq!(schedule.next_wake(), Some(first_poll));
        assert!(!schedule.take_due_repaint(first_poll));

        schedule.after_gpu_poll(first_poll, true);
        let second_poll = first_poll + super::PENDING_GPU_POLL_INTERVAL;
        assert_eq!(schedule.next_wake(), Some(second_poll));
        assert!(!schedule.take_due_repaint(second_poll));

        schedule.after_gpu_poll(second_poll, false);
        let repaint = now + Duration::from_millis(10);
        assert_eq!(schedule.next_wake(), Some(repaint));
        assert!(!schedule.take_due_repaint(now + Duration::from_millis(9)));
        assert!(schedule.take_due_repaint(repaint));
        assert_eq!(schedule.next_wake(), None);
    }
}

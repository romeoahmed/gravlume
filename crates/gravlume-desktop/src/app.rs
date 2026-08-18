use std::{
    ffi::OsStr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use gravlume_domain::ValidationReport;
use gravlume_native_display::{DisplayMonitor, DynamicRange, UnknownDisplayState};
use gravlume_render::{
    DeviceEvent, PresentResult, PresentSkip, Renderer, RendererError, RendererInitError,
    ResizeError,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    error::{EventLoopError, OsError},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    window::{Window, WindowId},
};

use crate::{
    lifecycle::Lifecycle,
    preview::DEFAULT_PREVIEW,
    schedule::{DesktopSchedule, ResizeAction},
    ui::{install_fonts, show_overlay},
};

const WINDOW_TITLE: &str = "Gravlume";
const INITIAL_RENDER_EXTENT: PhysicalSize<u32> = PhysicalSize::new(1280, 720);
const RETRY_FRAME_INTERVAL: Duration = Duration::from_millis(16);
const SMOKE_ONCE_ENV: &str = "GRAVLUME_SMOKE_ONCE";

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("failed to create the desktop event loop: {0}")]
    EventLoop(#[from] EventLoopError),
    #[error("failed to create the native window: {0}")]
    Window(#[from] OsError),
    #[error("failed to monitor the native display: {0}")]
    NativeDisplay(#[from] gravlume_native_display::MonitorError),
    #[error("failed to initialize the GPU renderer: {0}")]
    RenderInit(#[from] RendererInitError),
    #[error("failed to construct the validated preview scene: {0}")]
    Preview(#[from] ValidationReport),
    #[error("rendering failed: {0}")]
    RenderRuntime(#[from] RendererError),
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
pub fn run() -> Result<(), RunError> {
    let mut builder = EventLoop::<AppEvent>::with_user_event();
    let event_loop = builder.build()?;
    let mut app = DesktopApp::new(event_loop.create_proxy());
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
    lifecycle: Lifecycle,
    window: Option<WindowState>,
    renderer: Option<Renderer>,
    egui_context: egui::Context,
    event_proxy: EventLoopProxy<AppEvent>,
    schedule: DesktopSchedule,
    pending_textures: egui::TexturesDelta,
    last_device_event: Option<DeviceEvent>,
    last_resize_event: Option<DeviceEvent>,
    fatal_error: Option<RunError>,
    completed_present_generation: Option<u64>,
    smoke_once: bool,
    exit_requested: bool,
}

impl DesktopApp {
    fn new(event_proxy: EventLoopProxy<AppEvent>) -> Self {
        let egui_context = egui::Context::default();
        install_fonts(&egui_context);
        Self {
            lifecycle: Lifecycle::default(),
            window: None,
            renderer: None,
            egui_context,
            event_proxy,
            schedule: DesktopSchedule::default(),
            pending_textures: egui::TexturesDelta::default(),
            last_device_event: None,
            last_resize_event: None,
            fatal_error: None,
            completed_present_generation: None,
            smoke_once: std::env::var_os(SMOKE_ONCE_ENV)
                .is_some_and(|value| value == OsStr::new("1")),
            exit_requested: false,
        }
    }

    fn initialize(&mut self, event_loop: &ActiveEventLoop) -> Result<(), RunError> {
        let window = if let Some(window_state) = &self.window {
            Arc::clone(&window_state.window)
        } else {
            let attributes = Window::default_attributes()
                .with_title(WINDOW_TITLE)
                .with_inner_size(INITIAL_RENDER_EXTENT);
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
            renderer.update_output(display_state)?;
            let size = window.inner_size();
            self.resize_renderer(event_loop, size.width, size.height);
            return Ok(());
        }
        let size = window.inner_size();
        let observation = DEFAULT_PREVIEW.observation(size.width, size.height)?;
        let mut renderer =
            pollster::block_on(Renderer::new(window.clone(), &observation, display_state))?;
        // Configuring extended-linear output can arm macOS EDR. Re-read the live snapshot after
        // the surface exists so current headroom is not left at its pre-configuration value.
        renderer.update_output(self.current_display_state())?;
        let diagnostics = renderer.diagnostics();
        tracing::info!(
            adapter = diagnostics.adapter_name(),
            backend = diagnostics.backend(),
            driver = diagnostics.driver(),
            format = diagnostics.surface_format(),
            color_space = diagnostics.color_space(),
            transfer = diagnostics.display_transfer(),
            "initialized desktop renderer"
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
            let resize_event = self.last_resize_event.as_ref();
            let raw_input = window_state.egui.take_egui_input(window);
            let output = self.egui_context.run_ui(raw_input, |root_ui| {
                show_overlay(
                    root_ui.ctx(),
                    &diagnostics,
                    DEFAULT_PREVIEW,
                    device_event,
                    resize_event,
                );
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
            renderer.present(&paint_jobs, &self.pending_textures, pixels_per_point)
        };

        match render_result {
            Ok(PresentResult::Presented) => {
                self.pending_textures.clear();
            }
            Ok(PresentResult::Skipped(PresentSkip::Timeout)) => {
                self.schedule
                    .request_repaint(Instant::now(), RETRY_FRAME_INTERVAL);
            }
            Ok(PresentResult::Skipped(PresentSkip::Outdated | PresentSkip::Lost)) => {
                self.schedule
                    .request_repaint(Instant::now(), Duration::ZERO);
            }
            Ok(PresentResult::Skipped(
                PresentSkip::Validation
                | PresentSkip::Occluded
                | PresentSkip::ZeroExtent
                | PresentSkip::Suspended,
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

    fn request_resize(&mut self, event_loop: &ActiveEventLoop, size: PhysicalSize<u32>) {
        if let ResizeAction::ApplyNow(size) = self.schedule.request_resize(Instant::now(), size) {
            self.resize_renderer(event_loop, size.width, size.height);
        }
    }

    fn apply_pending_resize(&mut self, event_loop: &ActiveEventLoop, gpu_idle: bool) {
        if let Some(size) = self.schedule.take_ready_resize(Instant::now(), gpu_idle) {
            self.resize_renderer(event_loop, size.width, size.height);
        }
    }

    fn resize_renderer(&mut self, event_loop: &ActiveEventLoop, width: u32, height: u32) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        match renderer.resize(width, height) {
            Ok(()) => {
                if width != 0 && height != 0 {
                    self.last_resize_event = None;
                    self.request_redraw();
                }
            }
            Err(error) => {
                let is_fatal = error.is_fatal();
                let event = DeviceEvent::from(&error);
                tracing::error!(kind = ?event.kind(), message = event.message(), width, height, "GPU resize rejected");
                let report_changed = self.last_resize_event.as_ref() != Some(&event);
                if report_changed {
                    self.last_resize_event = Some(event);
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

    fn current_display_state(&self) -> DynamicRange {
        self.window.as_ref().map_or(
            DynamicRange::Unknown(UnknownDisplayState::PlatformIntegrationUnavailable),
            |state| state.display_monitor.dynamic_range(),
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
                    && let Err(error) = renderer.update_output(display_state)
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
        if response.repaint
            && !matches!(
                &event,
                WindowEvent::RedrawRequested
                    | WindowEvent::Resized(_)
                    | WindowEvent::ScaleFactorChanged { .. }
            )
        {
            window.request_redraw();
        }

        match event {
            WindowEvent::CloseRequested => self.request_exit(event_loop),
            WindowEvent::Resized(size) => {
                self.request_resize(event_loop, size);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = window.inner_size();
                self.request_resize(event_loop, size);
            }
            WindowEvent::Occluded(false) => window.request_redraw(),
            WindowEvent::RedrawRequested if self.schedule.redraw_allowed() => {
                self.draw_frame(event_loop);
            }
            WindowEvent::Occluded(true) | WindowEvent::RedrawRequested => {}
            _ if response.consumed => {}
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.exit_requested {
            self.finish_exit(event_loop);
            return;
        }
        let poll_result = self.renderer.as_mut().map(Renderer::poll);
        match poll_result {
            Some(Ok(mut update)) => {
                let published_generation = update.published_generation();
                let completed_present_generation = update.completed_present_generation();
                self.process_device_events(event_loop, update.take_events());
                if published_generation.is_some() {
                    self.request_redraw();
                }
                if let Some(generation) = completed_present_generation {
                    self.completed_present_generation = Some(generation);
                }
                let current_generation = self.renderer.as_ref().map(Renderer::generation);
                if self.smoke_once
                    && current_generation.is_some()
                    && self.completed_present_generation == current_generation
                {
                    if let Some(diagnostics) = self.renderer.as_ref().map(Renderer::diagnostics) {
                        tracing::info!(
                            trace_batches = diagnostics.completed_trace_batches(),
                            total_trace_compute_ms = diagnostics.total_trace_compute_ms(),
                            maximum_trace_batch_ms = diagnostics.maximum_trace_batch_ms(),
                            "one-frame renderer smoke completed"
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
        let gpu_idle = self
            .renderer
            .as_ref()
            .is_none_or(|renderer| !renderer.has_pending_work());
        self.apply_pending_resize(event_loop, gpu_idle);
        if !self.schedule.resize_pending()
            && let Some(Err(error)) = self.renderer.as_mut().map(Renderer::advance_trace)
        {
            self.fail(event_loop, error.into());
            return;
        }
        let has_pending_work = self
            .renderer
            .as_ref()
            .is_some_and(Renderer::has_pending_work);
        self.schedule.after_gpu_poll(now, has_pending_work);
        if self.schedule.take_due_repaint(now) {
            self.request_redraw();
        }
        let native_dispatch_deadline = self
            .window
            .as_ref()
            .and_then(|window| window.display_monitor.next_dispatch_deadline());
        match self.schedule.next_wake(native_dispatch_deadline, gpu_idle) {
            Some(deadline) => event_loop.set_control_flow(ControlFlow::WaitUntil(deadline)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.schedule.clear_resize();
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

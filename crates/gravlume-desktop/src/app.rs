use std::{
    ffi::OsStr,
    sync::Arc,
    time::{Duration, Instant},
};

use gravlume_render::{
    DeviceEvent, DeviceEventKind, FrameSkip, FrameStatus, GpuEngine, RenderDiagnostics,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    window::{Window, WindowId},
};

use crate::{
    Launch,
    lifecycle::{Lifecycle, LifecycleAction},
};

const PENDING_GPU_POLL_INTERVAL: Duration = Duration::from_millis(2);
const RETRY_FRAME_INTERVAL: Duration = Duration::from_millis(16);
const SMOKE_ONCE_ENV: &str = "GRAVLUME_SMOKE_ONCE";

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("failed to create the desktop event loop: {0}")]
    EventLoop(String),
    #[error("failed to create the native window: {0}")]
    Window(String),
    #[error("failed to initialize the GPU renderer: {0}")]
    RenderInit(String),
    #[error("rendering failed: {0}")]
    RenderRuntime(String),
    #[error("fatal GPU event: {0}")]
    Device(String),
}

/// Runs the native Gravlume application until the event loop exits.
///
/// # Errors
///
/// Returns an error when event-loop, window, renderer, or device initialization/runtime fails.
pub fn run(launch: Launch) -> Result<(), RunError> {
    let mut builder = EventLoop::<UserEvent>::with_user_event();
    let event_loop = builder
        .build()
        .map_err(|error| RunError::EventLoop(error.to_string()))?;
    let mut app = DesktopApp::new(launch, event_loop.create_proxy());
    event_loop
        .run_app(&mut app)
        .map_err(|error| RunError::EventLoop(error.to_string()))?;
    app.fatal_error.take().map_or(Ok(()), Err)
}

#[derive(Clone, Copy, Debug)]
enum UserEvent {
    RequestRepaint(Instant),
}

struct DesktopApp {
    launch: Launch,
    lifecycle: Lifecycle,
    window: Option<Arc<Window>>,
    renderer: Option<GpuEngine>,
    egui_context: egui::Context,
    egui_state: Option<egui_winit::State>,
    event_proxy: EventLoopProxy<UserEvent>,
    schedule: EventLoopSchedule,
    pending_textures: egui::TexturesDelta,
    last_device_event: Option<(DeviceEventKind, String)>,
    fatal_error: Option<RunError>,
    presented_frames: u64,
    smoke_once: bool,
}

impl DesktopApp {
    fn new(launch: Launch, event_proxy: EventLoopProxy<UserEvent>) -> Self {
        Self {
            launch,
            lifecycle: Lifecycle::default(),
            window: None,
            renderer: None,
            egui_context: egui::Context::default(),
            egui_state: None,
            event_proxy,
            schedule: EventLoopSchedule::default(),
            pending_textures: egui::TexturesDelta::default(),
            last_device_event: None,
            fatal_error: None,
            presented_frames: 0,
            smoke_once: smoke_once_value(std::env::var_os(SMOKE_ONCE_ENV).as_deref()),
        }
    }

    fn initialize(&mut self, event_loop: &ActiveEventLoop) -> Result<(), RunError> {
        if self.window.is_none() {
            let window_preferences = self.launch.window();
            let attributes = Window::default_attributes()
                .with_title(window_preferences.title())
                .with_inner_size(PhysicalSize::new(
                    window_preferences.width(),
                    window_preferences.height(),
                ));
            let window = Arc::new(
                event_loop
                    .create_window(attributes)
                    .map_err(|error| RunError::Window(error.to_string()))?,
            );
            let state = egui_winit::State::new(
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
                        .send_event(UserEvent::RequestRepaint(deadline))
                        .is_err()
                    {
                        tracing::debug!("event loop closed before egui repaint request");
                    }
                });
            self.egui_state = Some(state);
            self.window = Some(window);
        }

        let Some(window) = self.window.as_ref().map(Arc::clone) else {
            return Err(RunError::Window(
                "window lifecycle did not retain the created window".to_owned(),
            ));
        };
        if let Some(renderer) = self.renderer.as_mut() {
            renderer
                .resume_surface()
                .map_err(|error| RunError::RenderRuntime(error.to_string()))?;
            window.request_redraw();
            return Ok(());
        }
        let renderer = pollster::block_on(GpuEngine::new(window.clone()))
            .map_err(|error| RunError::RenderInit(error.to_string()))?;
        let diagnostics = renderer.diagnostics();
        tracing::info!(
            adapter = diagnostics.adapter_name(),
            backend = diagnostics.backend(),
            driver = diagnostics.driver(),
            format = diagnostics.surface_format(),
            color_space = diagnostics.color_space(),
            transfer = diagnostics.display_transfer(),
            "initialized Phase 0 desktop renderer"
        );
        self.renderer = Some(renderer);
        window.request_redraw();
        Ok(())
    }

    fn draw_frame(&mut self, event_loop: &ActiveEventLoop) {
        let render_result = {
            let (Some(window), Some(state), Some(renderer)) = (
                self.window.as_ref(),
                self.egui_state.as_mut(),
                self.renderer.as_mut(),
            ) else {
                return;
            };

            let diagnostics = renderer.diagnostics();
            let device_event = self.last_device_event.as_ref();
            let raw_input = state.take_egui_input(window);
            let output = self.egui_context.run_ui(raw_input, |root_ui| {
                show_overlay(root_ui.ctx(), &diagnostics, device_event);
            });
            let egui::FullOutput {
                platform_output,
                textures_delta,
                shapes,
                pixels_per_point,
                viewport_output,
            } = output;
            state.handle_platform_output_with_event_loop(window, event_loop, platform_output);
            if let Some(viewport) = viewport_output.get(&egui::ViewportId::ROOT) {
                self.schedule
                    .request_repaint(Instant::now(), viewport.repaint_delay);
            }
            let paint_jobs = self.egui_context.tessellate(shapes, pixels_per_point);
            self.pending_textures.append(textures_delta);
            renderer.render(&paint_jobs, &self.pending_textures, pixels_per_point)
        };

        match render_result {
            Ok(FrameStatus::Presented) => {
                self.pending_textures.clear();
                self.presented_frames = self.presented_frames.saturating_add(1);
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
            Err(error) => self.fail(event_loop, RunError::RenderRuntime(error.to_string())),
        }
    }

    fn process_device_events(&mut self, event_loop: &ActiveEventLoop, events: &[DeviceEvent]) {
        let mut report_changed = false;
        for event in events {
            tracing::error!(kind = ?event.kind(), message = event.message(), "GPU device event");
            report_changed |= record_device_event(&mut self.last_device_event, event);
            if event.is_fatal() {
                self.fail(event_loop, RunError::Device(event.message().to_owned()));
                return;
            }
        }
        if report_changed {
            self.request_redraw();
        }
    }

    fn resize_renderer(&mut self, event_loop: &ActiveEventLoop, width: u32, height: u32) -> bool {
        let Some(result) = self
            .renderer
            .as_mut()
            .map(|renderer| renderer.resize(width, height))
        else {
            return false;
        };
        match result {
            Ok(()) => true,
            Err(error) => {
                let kind = error.kind();
                let message = error.to_string();
                tracing::error!(?kind, %message, width, height, "GPU resize rejected");
                self.last_device_event = Some((kind, message.clone()));
                if error.is_fatal() {
                    self.fail(event_loop, RunError::Device(message));
                }
                false
            }
        }
    }

    fn request_redraw(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: RunError) {
        if self.fatal_error.is_none() {
            tracing::error!(error = %error, "desktop runtime is stopping");
            self.lifecycle.fail();
            self.pending_textures.clear();
            self.fatal_error = Some(error);
        }
        event_loop.exit();
    }
}

impl Drop for DesktopApp {
    fn drop(&mut self) {
        self.pending_textures.clear();
    }
}

impl ApplicationHandler<UserEvent> for DesktopApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.lifecycle.resume() == LifecycleAction::Initialize
            && let Err(error) = self.initialize(event_loop)
        {
            self.fail(event_loop, error);
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::RequestRepaint(deadline) => {
                self.schedule.request_repaint_at(deadline);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref().map(Arc::clone) else {
            return;
        };
        if window.id() != window_id {
            return;
        }

        let response = self
            .egui_state
            .as_mut()
            .map(|state| state.on_window_event(&window, &event))
            .unwrap_or_default();
        if response.repaint {
            window.request_redraw();
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if self.resize_renderer(event_loop, size.width, size.height)
                    && size.width != 0
                    && size.height != 0
                {
                    window.request_redraw();
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = window.inner_size();
                if self.resize_renderer(event_loop, size.width, size.height)
                    && size.width != 0
                    && size.height != 0
                {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.draw_frame(event_loop),
            _ if response.consumed => {}
            _ => {
                // Observer controls arrive here once Phase 1 introduces an Observer Frame.
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let poll_result = self.renderer.as_mut().map(GpuEngine::poll);
        match poll_result {
            Some(Ok(outcome)) => {
                self.process_device_events(event_loop, outcome.events());
                if self.smoke_once && self.presented_frames > 0 && outcome.completed_readback() {
                    tracing::info!("Phase 0 one-frame smoke completed");
                    event_loop.exit();
                    return;
                }
            }
            Some(Err(error)) => {
                self.fail(event_loop, RunError::RenderRuntime(error.to_string()));
                return;
            }
            None => {}
        }
        if self.fatal_error.is_some() {
            return;
        }

        let now = Instant::now();
        let has_pending_gpu_work = self
            .renderer
            .as_ref()
            .is_some_and(GpuEngine::has_pending_gpu_work);
        self.schedule.after_gpu_poll(now, has_pending_gpu_work);
        if self.schedule.take_due_repaint(now) {
            self.request_redraw();
        }
        match self.schedule.next_wake() {
            Some(deadline) => event_loop.set_control_flow(ControlFlow::WaitUntil(deadline)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if self.lifecycle.suspend() == LifecycleAction::ReleaseSurface
            && let Some(renderer) = self.renderer.as_mut()
        {
            renderer.suspend();
        }
    }
}

fn show_overlay(
    context: &egui::Context,
    diagnostics: &RenderDiagnostics,
    device_event: Option<&(DeviceEventKind, String)>,
) {
    egui::Window::new("Phase 0 · Desktop stack")
        .default_pos([16.0, 16.0])
        .resizable(false)
        .collapsible(false)
        .show(context, |ui| {
            ui.heading("Gravlume");
            ui.label("scene-linear HDR compute → neutral display → egui");
            ui.separator();
            ui.monospace(format!(
                "{} · {}",
                diagnostics.adapter_name(),
                diagnostics.backend()
            ));
            if !diagnostics.driver().is_empty() {
                ui.monospace(diagnostics.driver());
            }
            ui.monospace(format!(
                "{} / {} / {}",
                diagnostics.surface_format(),
                diagnostics.color_space(),
                diagnostics.display_transfer()
            ));
            ui.monospace(format!(
                "extent generation {}",
                diagnostics.extent_generation()
            ));
            if let (Some(compute_ms), Some(display_ms)) =
                (diagnostics.compute_ms(), diagnostics.display_ms())
            {
                ui.monospace(format!(
                    "GPU compute {compute_ms:.3} ms · display {display_ms:.3} ms"
                ));
            } else {
                ui.monospace("GPU timing pending");
            }
            if let Some((kind, message)) = device_event {
                ui.separator();
                ui.colored_label(
                    egui::Color32::from_rgb(255, 170, 80),
                    format!("GPU {kind:?}: {message}"),
                );
            }
        });
}

/// Keeps timer wakeups for GPU progress separate from repaint requests.
///
/// Source: <https://docs.rs/winit/0.30.13/winit/application/trait.ApplicationHandler.html#method.about_to_wait>
#[derive(Debug, Default)]
struct EventLoopSchedule {
    repaint: RepaintSchedule,
    gpu_poll_deadline: Option<Instant>,
}

impl EventLoopSchedule {
    fn request_repaint(&mut self, now: Instant, delay: Duration) {
        self.repaint.request(now, delay);
    }

    fn request_repaint_at(&mut self, deadline: Instant) {
        self.repaint.request_at(deadline);
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
        self.repaint.take_due(now)
    }

    fn next_wake(&self) -> Option<Instant> {
        match (self.repaint.deadline(), self.gpu_poll_deadline) {
            (Some(repaint), Some(gpu_poll)) => Some(repaint.min(gpu_poll)),
            (Some(deadline), None) | (None, Some(deadline)) => Some(deadline),
            (None, None) => None,
        }
    }
}

#[derive(Debug, Default)]
struct RepaintSchedule {
    deadline: Option<Instant>,
}

impl RepaintSchedule {
    fn request(&mut self, now: Instant, delay: Duration) {
        if delay == Duration::MAX {
            return;
        }
        self.request_at(now.checked_add(delay).unwrap_or(now));
    }

    fn request_at(&mut self, deadline: Instant) {
        self.deadline = Some(
            self.deadline
                .map_or(deadline, |current| current.min(deadline)),
        );
    }

    fn take_due(&mut self, now: Instant) -> bool {
        if self.deadline.is_some_and(|deadline| deadline <= now) {
            self.deadline = None;
            true
        } else {
            false
        }
    }

    const fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
}

fn smoke_once_value(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| value == OsStr::new("1"))
}

fn record_device_event(
    last_event: &mut Option<(DeviceEventKind, String)>,
    event: &DeviceEvent,
) -> bool {
    if last_event
        .as_ref()
        .is_some_and(|(kind, message)| *kind == event.kind() && message == event.message())
    {
        return false;
    }
    *last_event = Some((event.kind(), event.message().to_owned()));
    true
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsStr, time::Duration};

    use gravlume_render::DeviceEvent;

    use super::{EventLoopSchedule, RepaintSchedule, record_device_event, smoke_once_value};

    #[test]
    fn repaint_schedule_keeps_the_earliest_deadline() {
        let now = std::time::Instant::now();
        let mut schedule = RepaintSchedule::default();

        schedule.request(now, Duration::from_millis(20));
        schedule.request(now, Duration::from_millis(5));
        schedule.request(now, Duration::from_millis(10));

        assert!(!schedule.take_due(now + Duration::from_millis(4)));
        assert!(schedule.take_due(now + Duration::from_millis(5)));
        assert_eq!(schedule.deadline(), None);
    }

    #[test]
    fn smoke_hook_accepts_only_the_exact_opt_in_value() {
        assert!(!smoke_once_value(None));
        assert!(smoke_once_value(Some(OsStr::new("1"))));
        assert!(!smoke_once_value(Some(OsStr::new("true"))));
    }

    #[test]
    fn gpu_poll_deadline_wakes_without_requesting_a_redraw() {
        let now = std::time::Instant::now();
        let mut schedule = EventLoopSchedule::default();

        schedule.after_gpu_poll(now, true);
        let poll_deadline = now + super::PENDING_GPU_POLL_INTERVAL;
        assert_eq!(schedule.next_wake(), Some(poll_deadline));
        assert!(!schedule.take_due_repaint(poll_deadline));

        schedule.after_gpu_poll(poll_deadline, true);
        let next_poll_deadline = poll_deadline + super::PENDING_GPU_POLL_INTERVAL;
        assert_eq!(schedule.next_wake(), Some(next_poll_deadline));
        assert!(!schedule.take_due_repaint(next_poll_deadline));

        schedule.after_gpu_poll(next_poll_deadline, false);
        assert_eq!(schedule.next_wake(), None);
    }

    #[test]
    fn repaint_and_gpu_poll_keep_independent_deadlines() {
        let now = std::time::Instant::now();
        let mut schedule = EventLoopSchedule::default();

        schedule.request_repaint(now, Duration::from_millis(10));
        schedule.after_gpu_poll(now, true);

        let poll_deadline = now + super::PENDING_GPU_POLL_INTERVAL;
        assert_eq!(schedule.next_wake(), Some(poll_deadline));
        assert!(!schedule.take_due_repaint(poll_deadline));
        assert!(schedule.take_due_repaint(now + Duration::from_millis(10)));
    }

    #[test]
    fn duplicate_device_event_does_not_request_another_report_redraw() {
        let event = DeviceEvent::validation("surface configuration was rejected");
        let mut last_event = None;

        assert!(record_device_event(&mut last_event, &event));
        assert!(!record_device_event(&mut last_event, &event));
    }
}

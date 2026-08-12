#![deny(unsafe_code)]

//! Narrow, safe ownership boundary for native display-state notifications.

use std::time::Instant;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownDisplayState {
    PlatformIntegrationUnavailable,
    UnsupportedOsVersion,
    StateQueryFailed,
    WaylandColorManagementUnavailable,
    WaylandProtocolTooOld,
    WaylandEncodingUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DynamicRange {
    Hdr {
        tone_map_headroom: f32,
        reference_white_scale: f32,
    },
    Sdr,
    Suppressed,
    Unknown(UnknownDisplayState),
}

#[derive(Debug, thiserror::Error)]
pub enum MonitorError {
    #[error("native window handle is unavailable: {0}")]
    WindowHandle(#[from] raw_window_handle::HandleError),
    #[error("native display monitoring must be created on the platform UI thread")]
    WrongThread,
    #[error("the native window handle does not match this platform")]
    WrongWindowHandle,
    #[error("the native view is not attached to a window")]
    MissingWindow,
}

pub struct DisplayMonitor {
    platform: platform::Monitor,
}

trait PlatformMonitor {
    fn refresh(&mut self);
    fn dynamic_range(&self) -> DynamicRange;
    fn next_dispatch_deadline(&self) -> Option<Instant>;
    fn shutdown(&mut self);
    fn shutdown_complete(&self) -> bool;
}

impl DisplayMonitor {
    /// Creates a live monitor for the native display backing `window`.
    ///
    /// # Errors
    ///
    /// Returns an error when native handles are unavailable or do not match the current platform,
    /// or when a UI-thread-only platform API is called from another thread.
    pub fn new(
        window: &(impl HasDisplayHandle + HasWindowHandle),
        notify: impl Fn() + Send + Sync + 'static,
    ) -> Result<Self, MonitorError> {
        platform::Monitor::new(window, notify).map(|platform| Self { platform })
    }

    /// Dispatches native events that share the window system's connection.
    ///
    /// This is a no-op on platforms whose native callback mechanism is driven directly by the
    /// event loop.
    pub fn refresh(&mut self) {
        self.platform.refresh();
    }

    #[must_use]
    pub fn dynamic_range(&self) -> DynamicRange {
        self.platform.dynamic_range()
    }

    /// Returns a distant wake guard needed to make externally queued display events observable.
    ///
    /// The deadline is not a polling interval. On Wayland it prevents winit from discarding a
    /// readable display fd as a spurious wake when the event belongs only to this monitor's guest
    /// queue.
    #[must_use]
    pub fn next_dispatch_deadline(&self) -> Option<Instant> {
        self.platform.next_dispatch_deadline()
    }

    /// Removes native observers and begins platform queue shutdown while the event loop is alive.
    pub fn shutdown(&mut self) {
        self.platform.shutdown();
    }

    /// Reports whether platform shutdown has completed and the event loop may stop pumping.
    #[must_use]
    pub fn shutdown_complete(&self) -> bool {
        self.platform.shutdown_complete()
    }
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
#[path = "platform/macos.rs"]
mod platform;

#[cfg(target_os = "windows")]
#[allow(unsafe_code)]
#[path = "platform/windows.rs"]
mod platform;

#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
#[path = "platform/wayland.rs"]
mod platform;

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
mod platform {
    use std::time::Instant;

    use raw_window_handle::HasWindowHandle;

    use crate::{DynamicRange, MonitorError, UnknownDisplayState};

    pub struct Monitor;

    impl Monitor {
        pub(super) fn new(
            window: &impl HasWindowHandle,
            _notify: impl Fn() + Send + Sync + 'static,
        ) -> Result<Self, MonitorError> {
            let _ = window.window_handle()?;
            Ok(Self)
        }
    }

    impl crate::PlatformMonitor for Monitor {
        fn dynamic_range(&self) -> DynamicRange {
            DynamicRange::Unknown(UnknownDisplayState::PlatformIntegrationUnavailable)
        }

        fn refresh(&mut self) {}

        fn next_dispatch_deadline(&self) -> Option<Instant> {
            None
        }

        fn shutdown(&mut self) {}

        fn shutdown_complete(&self) -> bool {
            true
        }
    }
}

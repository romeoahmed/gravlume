use std::{sync::Arc, time::Instant};

use block2::RcBlock;
use num_traits::ToPrimitive as _;
use objc2::{MainThreadMarker, rc::Retained, runtime::ProtocolObject};
use objc2_app_kit::{
    NSApplication, NSApplicationDidChangeScreenParametersNotification,
    NSApplicationShouldBeginSuppressingHighDynamicRangeContentNotification,
    NSApplicationShouldEndSuppressingHighDynamicRangeContentNotification, NSScreen, NSView,
    NSWindow, NSWindowDidChangeScreenNotification,
};
use objc2_foundation::{NSNotification, NSNotificationCenter, NSObjectProtocol};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

use crate::{DynamicRange, MonitorError, PlatformMonitor, UnknownDisplayState};

pub struct Monitor {
    application: Retained<NSApplication>,
    window: Retained<NSWindow>,
    center: Retained<NSNotificationCenter>,
    observers: Vec<Retained<ProtocolObject<dyn NSObjectProtocol>>>,
}

impl Monitor {
    pub(super) fn new(
        window: &impl HasWindowHandle,
        notify: impl Fn() + Send + Sync + 'static,
    ) -> Result<Self, MonitorError> {
        let main_thread = MainThreadMarker::new().ok_or(MonitorError::WrongThread)?;
        let RawWindowHandle::AppKit(handle) = window.window_handle()?.as_raw() else {
            return Err(MonitorError::WrongWindowHandle);
        };
        // SAFETY: raw-window-handle guarantees that `ns_view` is the live NSView owned by
        // `window`. The monitor is stored beside that window and is dropped first.
        let view = unsafe { handle.ns_view.cast::<NSView>().as_ref() };
        let native_window = view.window().ok_or(MonitorError::MissingWindow)?;
        let application = NSApplication::sharedApplication(main_thread);
        let center = NSNotificationCenter::defaultCenter();
        let notify: Arc<dyn Fn() + Send + Sync> = Arc::new(notify);
        let mut observers = Vec::with_capacity(4);

        // SAFETY: every block captures only a Send + Sync callback, the notification center
        // copies escaping blocks, and every observer token remains retained until Drop.
        unsafe {
            observers.push(observe(
                &center,
                NSWindowDidChangeScreenNotification,
                Some(native_window.as_ref()),
                Arc::clone(&notify),
            ));
            for name in [
                NSApplicationDidChangeScreenParametersNotification,
                NSApplicationShouldBeginSuppressingHighDynamicRangeContentNotification,
                NSApplicationShouldEndSuppressingHighDynamicRangeContentNotification,
            ] {
                observers.push(observe(&center, name, None, Arc::clone(&notify)));
            }
        }

        Ok(Self {
            application,
            window: native_window,
            center,
            observers,
        })
    }
}

impl PlatformMonitor for Monitor {
    fn dynamic_range(&self) -> DynamicRange {
        if self
            .application
            .applicationShouldSuppressHighDynamicRangeContent()
        {
            return DynamicRange::Suppressed;
        }
        self.window.screen().map_or(
            DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed),
            |screen| dynamic_range_for_screen(&screen),
        )
    }

    fn refresh(&mut self) {}

    fn next_dispatch_deadline(&self) -> Option<Instant> {
        None
    }

    fn shutdown(&mut self) {
        for observer in self.observers.drain(..) {
            // SAFETY: each token came from this center and remains a valid Objective-C object.
            unsafe { self.center.removeObserver(observer.as_ref()) };
        }
    }

    fn shutdown_complete(&self) -> bool {
        self.observers.is_empty()
    }
}

fn dynamic_range_for_screen(screen: &NSScreen) -> DynamicRange {
    let Some(current) = screen
        .maximumExtendedDynamicRangeColorComponentValue()
        .to_f32()
    else {
        return DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed);
    };
    let Some(potential) = screen
        .maximumPotentialExtendedDynamicRangeColorComponentValue()
        .to_f32()
    else {
        return DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed);
    };
    if !current.is_finite() || !potential.is_finite() || current <= 0.0 || potential <= 0.0 {
        return DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed);
    }
    if potential > 1.0 || current > 1.0 {
        DynamicRange::Hdr {
            tone_map_headroom: current.max(1.0),
            reference_white_scale: 1.0,
        }
    } else {
        DynamicRange::Sdr
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

unsafe fn observe(
    center: &NSNotificationCenter,
    name: &objc2_foundation::NSNotificationName,
    object: Option<&objc2::runtime::AnyObject>,
    notify: Arc<dyn Fn() + Send + Sync>,
) -> Retained<ProtocolObject<dyn NSObjectProtocol>> {
    let block = RcBlock::new(move |_notification: std::ptr::NonNull<NSNotification>| notify());
    // SAFETY: the caller establishes block sendability and the object type; `queue = None`
    // delivers on the posting thread and the callback only sends a winit user event.
    unsafe { center.addObserverForName_object_queue_usingBlock(Some(name), object, None, &block) }
}

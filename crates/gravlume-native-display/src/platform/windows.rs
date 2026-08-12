use std::{
    ffi::c_void,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use windows::{
    Foundation::TypedEventHandler,
    Graphics::Display::{AdvancedColorKind, DisplayInformation},
    System::{DispatcherQueue, DispatcherQueueController},
    Win32::{
        Foundation::{E_INVALIDARG, E_NOINTERFACE, HWND, REGDB_E_CLASSNOTREG},
        Graphics::Gdi::HMONITOR,
        System::WinRT::{
            CreateDispatcherQueueController, DQTAT_COM_NONE, DQTYPE_THREAD_CURRENT,
            DispatcherQueueOptions,
        },
    },
    core::Interface as _,
};

use crate::{DynamicRange, MonitorError, PlatformMonitor, UnknownDisplayState};
use windows_core::Type as _;
use windows_future::IAsyncAction;

const NOMINAL_SCRGB_WHITE_NITS: f32 = 80.0;

pub struct Monitor {
    display: Option<DisplayInformation>,
    event_token: Option<i64>,
    owned_dispatcher: Option<DispatcherQueueController>,
    shutdown_action: Option<IAsyncAction>,
    shutdown_complete: Arc<AtomicBool>,
    notify: Arc<dyn Fn() + Send + Sync>,
    unavailable_reason: UnknownDisplayState,
}

impl Monitor {
    pub(super) fn new(
        window: &impl HasWindowHandle,
        notify: impl Fn() + Send + Sync + 'static,
    ) -> Result<Self, MonitorError> {
        let RawWindowHandle::Win32(handle) = window.window_handle()?.as_raw() else {
            return Err(MonitorError::WrongWindowHandle);
        };
        let hwnd = HWND(handle.hwnd.get() as *mut c_void);
        let notify: Arc<dyn Fn() + Send + Sync> = Arc::new(notify);
        let shutdown_complete = Arc::new(AtomicBool::new(false));
        let owned_dispatcher = match ensure_dispatcher_queue() {
            Ok(controller) => controller,
            Err(error) => {
                return Ok(Self::unavailable(&error, None, notify, shutdown_complete));
            }
        };
        let display = match display_information_for_window(hwnd) {
            Ok(display) => display,
            Err(error) => {
                return Ok(Self::unavailable(
                    &error,
                    owned_dispatcher,
                    notify,
                    shutdown_complete,
                ));
            }
        };
        let event_notify = Arc::clone(&notify);
        let event_token =
            match display.AdvancedColorInfoChanged(&TypedEventHandler::new(move |_, _| {
                event_notify();
                Ok(())
            })) {
                Ok(token) => token,
                Err(error) => {
                    return Ok(Self::unavailable(
                        &error,
                        owned_dispatcher,
                        notify,
                        shutdown_complete,
                    ));
                }
            };
        Ok(Self {
            display: Some(display),
            event_token: Some(event_token),
            owned_dispatcher,
            shutdown_action: None,
            shutdown_complete,
            notify,
            unavailable_reason: UnknownDisplayState::StateQueryFailed,
        })
    }

    fn unavailable(
        error: &windows::core::Error,
        owned_dispatcher: Option<DispatcherQueueController>,
        notify: Arc<dyn Fn() + Send + Sync>,
        shutdown_complete: Arc<AtomicBool>,
    ) -> Self {
        Self {
            display: None,
            event_token: None,
            owned_dispatcher,
            shutdown_action: None,
            shutdown_complete,
            notify,
            unavailable_reason: unavailable_reason(error),
        }
    }
}

impl PlatformMonitor for Monitor {
    fn dynamic_range(&self) -> DynamicRange {
        self.display
            .as_ref()
            .map_or(DynamicRange::Unknown(self.unavailable_reason), |display| {
                display
                    .GetAdvancedColorInfo()
                    .ok()
                    .and_then(|info| {
                        let kind = info.CurrentAdvancedColorKind().ok()?;
                        if kind != AdvancedColorKind::HighDynamicRange {
                            return Some(DynamicRange::Sdr);
                        }
                        let peak_nits = finite_positive(info.MaxLuminanceInNits().ok()?);
                        let sdr_white_nits = finite_positive(info.SdrWhiteLevelInNits().ok()?);
                        match (peak_nits, sdr_white_nits) {
                            (Some(peak), Some(white)) => Some(DynamicRange::Hdr {
                                tone_map_headroom: (peak / white).max(1.0),
                                reference_white_scale: white / NOMINAL_SCRGB_WHITE_NITS,
                            }),
                            _ => Some(DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed)),
                        }
                    })
                    .unwrap_or(DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed))
            })
    }

    fn refresh(&mut self) {}

    fn next_dispatch_deadline(&self) -> Option<Instant> {
        None
    }

    fn shutdown(&mut self) {
        if let (Some(display), Some(token)) = (&self.display, self.event_token.take()) {
            let _ = display.RemoveAdvancedColorInfoChanged(token);
        }
        self.display = None;
        if self.shutdown_action.is_some() || self.shutdown_complete() {
            return;
        }
        let Some(controller) = self.owned_dispatcher.as_ref() else {
            self.shutdown_complete.store(true, Ordering::Release);
            return;
        };
        let shutdown = match controller.ShutdownQueueAsync() {
            Ok(shutdown) => shutdown,
            Err(error) => {
                tracing::debug!(%error, "failed to begin Windows display dispatcher shutdown");
                self.shutdown_complete.store(true, Ordering::Release);
                return;
            }
        };
        let completed = Arc::clone(&self.shutdown_complete);
        let notify = Arc::clone(&self.notify);
        match shutdown.when(move |result| {
            if let Err(error) = result {
                tracing::debug!(%error, "Windows display dispatcher shutdown failed");
            }
            completed.store(true, Ordering::Release);
            notify();
        }) {
            Ok(()) => self.shutdown_action = Some(shutdown),
            Err(error) => {
                tracing::debug!(%error, "failed to observe Windows display dispatcher shutdown");
                self.shutdown_complete.store(true, Ordering::Release);
            }
        }
    }

    fn shutdown_complete(&self) -> bool {
        self.shutdown_complete.load(Ordering::Acquire)
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn finite_positive(value: f32) -> Option<f32> {
    (value.is_finite() && value > 0.0).then_some(value)
}

const fn unavailable_reason(error: &windows::core::Error) -> UnknownDisplayState {
    if matches!(error.code(), E_NOINTERFACE | REGDB_E_CLASSNOTREG) {
        UnknownDisplayState::UnsupportedOsVersion
    } else {
        UnknownDisplayState::StateQueryFailed
    }
}

fn ensure_dispatcher_queue() -> windows::core::Result<Option<DispatcherQueueController>> {
    if DispatcherQueue::GetForCurrentThread().is_ok() {
        return Ok(None);
    }
    let dw_size = u32::try_from(size_of::<DispatcherQueueOptions>())
        .map_err(|_| windows::core::Error::from(E_INVALIDARG))?;
    let options = DispatcherQueueOptions {
        dwSize: dw_size,
        threadType: DQTYPE_THREAD_CURRENT,
        apartmentType: DQTAT_COM_NONE,
    };
    // SAFETY: winit owns and pumps this top-level window thread. The options request an inbox
    // Windows.System queue on the current thread without changing its existing COM apartment.
    unsafe { CreateDispatcherQueueController(options).map(Some) }
}

fn display_information_for_window(hwnd: HWND) -> windows::core::Result<DisplayInformation> {
    let factory = activation_factory::<IDisplayInformationStaticsInterop>()?;
    let mut display = std::ptr::null_mut();
    // SAFETY: `hwnd` is the live, current-thread top-level handle supplied by raw-window-handle;
    // `display` is an initialized out pointer, and DisplayInformation::IID matches the requested
    // WinRT class interface. Successful ownership transfers into `from_abi` exactly once.
    unsafe {
        (IDisplayInformationStaticsInterop::vtable(&factory).get_for_window)(
            IDisplayInformationStaticsInterop::as_raw(&factory),
            hwnd,
            &DisplayInformation::IID,
            &raw mut display,
        )
        .and_then(|| DisplayInformation::from_abi(display))
    }
}

fn activation_factory<I: windows::core::Interface>() -> windows::core::Result<I> {
    struct DisplayInformationFactory;

    impl windows::core::RuntimeName for DisplayInformationFactory {
        const NAME: &'static str = "Windows.Graphics.Display.DisplayInformation";
    }

    windows::core::factory::<DisplayInformationFactory, I>()
}

// Private projection of the official header-only interface in
// windows.graphics.display.interop.h. The current windows crate projects the WinRT class but not
// this desktop activation-factory interop interface.
windows_core::imp::define_interface!(
    IDisplayInformationStaticsInterop,
    IDisplayInformationStaticsInteropVtable,
    0x7449121c_382b_4705_8da7_a795ba482013
);
#[repr(C)]
pub struct IDisplayInformationStaticsInteropVtable {
    base: windows::core::IInspectable_Vtbl,
    get_for_window: unsafe extern "system" fn(
        *mut c_void,
        HWND,
        *const windows::core::GUID,
        *mut *mut c_void,
    ) -> windows::core::HRESULT,
    get_for_monitor: unsafe extern "system" fn(
        *mut c_void,
        HMONITOR,
        *const windows::core::GUID,
        *mut *mut c_void,
    ) -> windows::core::HRESULT,
}

impl windows::core::RuntimeName for IDisplayInformationStaticsInterop {}

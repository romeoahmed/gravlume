use std::{sync::Arc, time::Instant};

use num_traits::ToPrimitive as _;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use wayland_client::{
    Connection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum,
    backend::{Backend, ObjectId},
    globals::{GlobalListContents, registry_queue_init},
    protocol::{wl_registry, wl_surface},
};
use wayland_protocols::wp::color_management::v1::client::{
    wp_color_management_surface_feedback_v1::{self, WpColorManagementSurfaceFeedbackV1},
    wp_color_manager_v1::{self, Feature, WpColorManagerV1},
    wp_image_description_info_v1::{self, WpImageDescriptionInfoV1},
    wp_image_description_v1::{self, WpImageDescriptionV1},
};

use crate::{DynamicRange, MonitorError, PlatformMonitor, UnknownDisplayState};

const MINIMUM_COLOR_MANAGEMENT_VERSION: u32 = 2;
const MAXIMUM_COLOR_MANAGEMENT_VERSION: u32 = 3;
const SCRGB_UNIT_NITS: f64 = 80.0;

/// Long enough to be irrelevant as a timer, but finite so winit does not discard a Wayland fd
/// wake that contains events for this guest queue only.
const EVENT_LOOP_WAKE_GUARD: std::time::Duration = std::time::Duration::from_hours(24);

pub struct Monitor {
    live: Option<LiveMonitor>,
    snapshot: DynamicRange,
}

struct LiveMonitor {
    _connection: Connection,
    event_queue: EventQueue<State>,
    state: State,
    feedback: WpColorManagementSurfaceFeedbackV1,
    manager: WpColorManagerV1,
}

struct State {
    notify: Arc<dyn Fn() + Send + Sync>,
    manager_ready: bool,
    parametric_descriptions_supported: bool,
    preferred_identity: Option<u64>,
    pending: Option<PendingDescription>,
    description: Option<Description>,
    query_generation: u64,
    dispatch_failed: bool,
}

struct PendingDescription {
    generation: u64,
    preferred_identity: Option<u64>,
    image_identity: Option<u64>,
    image: WpImageDescriptionV1,
    info: Information,
}

#[derive(Clone, Copy, Default)]
struct Information {
    reference_white_nits: Option<f64>,
    maximum_luminance_nits: Option<f64>,
}

#[derive(Clone, Copy)]
struct Description {
    preferred_identity: u64,
    information: Information,
}

#[derive(Clone, Copy)]
struct DescriptionQuery(u64);

impl Monitor {
    pub(super) fn new(
        window: &(impl HasDisplayHandle + HasWindowHandle),
        notify: impl Fn() + Send + Sync + 'static,
    ) -> Result<Self, MonitorError> {
        let RawDisplayHandle::Wayland(display) = window.display_handle()?.as_raw() else {
            return Err(MonitorError::WrongWindowHandle);
        };
        let RawWindowHandle::Wayland(window) = window.window_handle()?.as_raw() else {
            return Err(MonitorError::WrongWindowHandle);
        };

        // SAFETY: raw-window-handle guarantees that this is the live `wl_display` backing the
        // window. `LiveMonitor` is stored beside that window and is dropped first. The guest
        // backend neither owns nor closes the foreign connection.
        let backend = unsafe { Backend::from_foreign_display(display.display.as_ptr().cast()) };
        let connection = Connection::from_backend(backend);
        let mut state = State::new(Arc::new(notify));
        let (globals, mut event_queue) = match registry_queue_init::<State>(&connection) {
            Ok(initialized) => initialized,
            Err(error) => {
                tracing::debug!(%error, "Wayland color-management registry initialization failed");
                return Ok(Self::unavailable(UnknownDisplayState::StateQueryFailed));
            }
        };
        let qh = event_queue.handle();
        let manager: WpColorManagerV1 = match globals.bind(
            &qh,
            MINIMUM_COLOR_MANAGEMENT_VERSION..=MAXIMUM_COLOR_MANAGEMENT_VERSION,
            (),
        ) {
            Ok(manager) => manager,
            Err(wayland_client::globals::BindError::NotPresent) => {
                return Ok(Self::unavailable(
                    UnknownDisplayState::WaylandColorManagementUnavailable,
                ));
            }
            Err(wayland_client::globals::BindError::UnsupportedVersion) => {
                return Ok(Self::unavailable(
                    UnknownDisplayState::WaylandProtocolTooOld,
                ));
            }
        };

        // SAFETY: raw-window-handle guarantees the pointer is a live `wl_surface` on the same
        // `wl_display`; the protocol interface is checked by `ObjectId::from_ptr`. The proxy is
        // borrowed only as a request argument and is never adopted or destroyed by this module.
        let surface_id = unsafe {
            ObjectId::from_ptr(
                wl_surface::WlSurface::interface(),
                window.surface.as_ptr().cast(),
            )
        }
        .map_err(|_| MonitorError::WrongWindowHandle)?;
        let surface = wl_surface::WlSurface::from_id(&connection, surface_id)
            .map_err(|_| MonitorError::WrongWindowHandle)?;
        let feedback = manager.get_surface_feedback(&surface, &qh, ());

        if let Err(error) = event_queue.roundtrip(&mut state) {
            tracing::debug!(%error, "Wayland color-management initial roundtrip failed");
            feedback.destroy();
            manager.destroy();
            return Ok(Self::unavailable(UnknownDisplayState::StateQueryFailed));
        }
        if state.parametric_descriptions_supported && state.pending.is_none() {
            state.request_preferred(&feedback, &qh);
        }
        for _ in 0..2 {
            if state.pending.is_none() || state.description.is_some() {
                break;
            }
            if let Err(error) = event_queue.roundtrip(&mut state) {
                tracing::debug!(%error, "Wayland preferred color-description roundtrip failed");
                feedback.destroy();
                manager.destroy();
                return Ok(Self::unavailable(UnknownDisplayState::StateQueryFailed));
            }
        }

        let snapshot = state.dynamic_range();
        Ok(Self {
            live: Some(LiveMonitor {
                _connection: connection,
                event_queue,
                state,
                feedback,
                manager,
            }),
            snapshot,
        })
    }

    const fn unavailable(reason: UnknownDisplayState) -> Self {
        Self {
            live: None,
            snapshot: DynamicRange::Unknown(reason),
        }
    }
}

impl PlatformMonitor for Monitor {
    fn refresh(&mut self) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        if let Err(error) = live.event_queue.dispatch_pending(&mut live.state) {
            tracing::debug!(%error, "Wayland color-management event dispatch failed");
            live.state.dispatch_failed = true;
        }
        if let Err(error) = live.event_queue.flush() {
            tracing::debug!(%error, "Wayland color-management request flush failed");
            live.state.dispatch_failed = true;
        }
        let next = live.state.dynamic_range();
        if next != self.snapshot {
            self.snapshot = next;
            (live.state.notify)();
        }
    }

    fn dynamic_range(&self) -> DynamicRange {
        self.snapshot
    }

    fn next_dispatch_deadline(&self) -> Option<Instant> {
        self.live
            .as_ref()
            .map(|_| Instant::now() + EVENT_LOOP_WAKE_GUARD)
    }

    fn shutdown(&mut self) {
        let Some(mut live) = self.live.take() else {
            return;
        };
        if let Some(pending) = live.state.pending.take() {
            pending.image.destroy();
        }
        live.feedback.destroy();
        live.manager.destroy();
        if let Err(error) = live.event_queue.flush() {
            tracing::debug!(%error, "Wayland color-management shutdown flush failed");
        }
    }

    fn shutdown_complete(&self) -> bool {
        self.live.is_none()
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl State {
    fn new(notify: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self {
            notify,
            manager_ready: false,
            parametric_descriptions_supported: false,
            preferred_identity: None,
            pending: None,
            description: None,
            query_generation: 0,
            dispatch_failed: false,
        }
    }

    fn request_preferred(
        &mut self,
        feedback: &WpColorManagementSurfaceFeedbackV1,
        qh: &QueueHandle<Self>,
    ) {
        if let Some(pending) = self.pending.take() {
            pending.image.destroy();
        }
        self.query_generation = self.query_generation.wrapping_add(1);
        let generation = self.query_generation;
        let image = feedback.get_preferred_parametric(qh, DescriptionQuery(generation));
        self.pending = Some(PendingDescription {
            generation,
            preferred_identity: self.preferred_identity,
            image_identity: None,
            image,
            info: Information::default(),
        });
    }

    fn dynamic_range(&self) -> DynamicRange {
        if self.dispatch_failed {
            return DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed);
        }
        if !self.manager_ready {
            return DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed);
        }
        if !self.parametric_descriptions_supported {
            return DynamicRange::Unknown(UnknownDisplayState::WaylandEncodingUnavailable);
        }
        let Some(info) = self.description.map(|description| description.information) else {
            return DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed);
        };
        let (Some(maximum), Some(reference)) =
            (info.maximum_luminance_nits, info.reference_white_nits)
        else {
            return DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed);
        };
        if !maximum.is_finite() || !reference.is_finite() || maximum <= 0.0 || reference <= 0.0 {
            return DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed);
        }
        if maximum <= reference {
            return DynamicRange::Sdr;
        }
        let Some(tone_map_headroom) = (maximum / reference).to_f32() else {
            return DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed);
        };
        let Some(reference_white_scale) = (reference / SCRGB_UNIT_NITS).to_f32() else {
            return DynamicRange::Unknown(UnknownDisplayState::StateQueryFailed);
        };
        DynamicRange::Hdr {
            tone_map_headroom,
            reference_white_scale,
        }
    }
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wl_registry::WlRegistry,
        _event: wl_registry::Event,
        _data: &GlobalListContents,
        _connection: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WpColorManagerV1, ()> for State {
    fn event(
        state: &mut Self,
        _proxy: &WpColorManagerV1,
        event: wp_color_manager_v1::Event,
        _data: &(),
        _connection: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            wp_color_manager_v1::Event::SupportedFeature {
                feature: WEnum::Value(Feature::Parametric),
            } => state.parametric_descriptions_supported = true,
            wp_color_manager_v1::Event::Done => state.manager_ready = true,
            _ => {}
        }
    }
}

impl Dispatch<WpColorManagementSurfaceFeedbackV1, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &WpColorManagementSurfaceFeedbackV1,
        event: wp_color_management_surface_feedback_v1::Event,
        _data: &(),
        _connection: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wp_color_management_surface_feedback_v1::Event::PreferredChanged2 {
            identity_hi,
            identity_lo,
        } = event
        {
            let identity = (u64::from(identity_hi) << 32) | u64::from(identity_lo);
            if state.preferred_identity != Some(identity) {
                state.preferred_identity = Some(identity);
                if state.parametric_descriptions_supported
                    && state
                        .description
                        .is_none_or(|description| description.preferred_identity != identity)
                {
                    state.request_preferred(proxy, qh);
                }
            }
        }
    }
}

impl Dispatch<WpImageDescriptionV1, DescriptionQuery> for State {
    fn event(
        state: &mut Self,
        proxy: &WpImageDescriptionV1,
        event: wp_image_description_v1::Event,
        query: &DescriptionQuery,
        _connection: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if state
            .pending
            .as_ref()
            .is_none_or(|pending| pending.generation != query.0)
        {
            return;
        }
        match event {
            wp_image_description_v1::Event::Ready2 {
                identity_hi,
                identity_lo,
            } => {
                let identity = (u64::from(identity_hi) << 32) | u64::from(identity_lo);
                if identity == 0 {
                    state.dispatch_failed = true;
                    return;
                }
                if let Some(pending) = state.pending.as_mut() {
                    pending.image_identity = Some(identity);
                }
                let _information = proxy.get_information(qh, DescriptionQuery(query.0));
            }
            wp_image_description_v1::Event::Failed { cause, msg } => {
                tracing::debug!(?cause, %msg, "Wayland preferred color description failed");
                if let Some(pending) = state.pending.take() {
                    pending.image.destroy();
                }
                state.description = None;
            }
            _ => {}
        }
    }
}

impl Dispatch<WpImageDescriptionInfoV1, DescriptionQuery> for State {
    fn event(
        state: &mut Self,
        _proxy: &WpImageDescriptionInfoV1,
        event: wp_image_description_info_v1::Event,
        query: &DescriptionQuery,
        _connection: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        let Some(pending) = state
            .pending
            .as_mut()
            .filter(|pending| pending.generation == query.0)
        else {
            return;
        };
        match event {
            wp_image_description_info_v1::Event::Luminances {
                max_lum,
                reference_lum,
                ..
            } => {
                pending.info.maximum_luminance_nits = Some(f64::from(max_lum));
                pending.info.reference_white_nits = Some(f64::from(reference_lum));
            }
            wp_image_description_info_v1::Event::Done => {
                let Some(completed) = state.pending.take() else {
                    return;
                };
                if completed.image_identity.is_none() {
                    state.dispatch_failed = true;
                    completed.image.destroy();
                    return;
                }
                let Some(preferred_identity) = completed
                    .preferred_identity
                    .or(state.preferred_identity)
                    .or(completed.image_identity)
                else {
                    state.dispatch_failed = true;
                    completed.image.destroy();
                    return;
                };
                state.description = Some(Description {
                    preferred_identity,
                    information: completed.info,
                });
                completed.image.destroy();
            }
            _ => {}
        }
    }
}

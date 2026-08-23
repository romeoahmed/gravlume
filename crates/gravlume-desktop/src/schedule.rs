use std::time::{Duration, Instant};

use winit::dpi::PhysicalSize;

const GPU_POLL_INTERVAL: Duration = Duration::from_millis(2);
const RESIZE_SETTLE_INTERVAL: Duration = Duration::from_millis(40);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeAction {
    ApplyNow(PhysicalSize<u32>),
    Deferred,
}

/// Owns all application deadlines and the live-resize publication gate.
///
/// GPU polling remains independent from repaint requests. `about_to_wait` supplies only the native
/// monitor deadline and whether resize owners have drained, then installs the single returned wake
/// deadline.
/// Source: <https://docs.rs/winit/0.30.13/winit/application/trait.ApplicationHandler.html#method.about_to_wait>
#[derive(Debug, Default)]
pub struct DesktopSchedule {
    resize_request: Option<(PhysicalSize<u32>, Instant)>,
    repaint_deadline: Option<Instant>,
    gpu_poll_deadline: Option<Instant>,
}

impl DesktopSchedule {
    pub fn request_resize(&mut self, now: Instant, size: PhysicalSize<u32>) -> ResizeAction {
        if size.width == 0 || size.height == 0 {
            self.resize_request = None;
            return ResizeAction::ApplyNow(size);
        }
        let deadline = now.checked_add(RESIZE_SETTLE_INTERVAL).unwrap_or(now);
        self.resize_request = Some((size, deadline));
        ResizeAction::Deferred
    }

    pub fn take_ready_resize(
        &mut self,
        now: Instant,
        resize_ready: bool,
    ) -> Option<PhysicalSize<u32>> {
        let (size, deadline) = self.resize_request?;
        if !resize_ready || deadline > now {
            return None;
        }
        self.resize_request = None;
        Some(size)
    }

    pub const fn resize_pending(&self) -> bool {
        self.resize_request.is_some()
    }

    pub const fn redraw_allowed(&self) -> bool {
        !self.resize_pending()
    }

    pub const fn clear_resize(&mut self) {
        self.resize_request = None;
    }

    pub fn request_repaint(&mut self, now: Instant, delay: Duration) {
        if delay != Duration::MAX {
            self.request_repaint_at(now.checked_add(delay).unwrap_or(now));
        }
    }

    pub fn request_repaint_at(&mut self, deadline: Instant) {
        self.repaint_deadline = Some(
            self.repaint_deadline
                .map_or(deadline, |current| current.min(deadline)),
        );
    }

    pub fn after_gpu_poll(&mut self, now: Instant, has_pending_work: bool) {
        if !has_pending_work {
            self.gpu_poll_deadline = None;
            return;
        }
        if self
            .gpu_poll_deadline
            .is_none_or(|deadline| deadline <= now)
        {
            self.gpu_poll_deadline = Some(now.checked_add(GPU_POLL_INTERVAL).unwrap_or(now));
        }
    }

    pub fn take_due_repaint(&mut self, now: Instant) -> bool {
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

    pub fn next_wake(
        &self,
        native_dispatch_deadline: Option<Instant>,
        resize_ready: bool,
    ) -> Option<Instant> {
        let resize_deadline = if resize_ready {
            self.resize_request.map(|(_, deadline)| deadline)
        } else {
            None
        };
        [
            self.repaint_deadline,
            self.gpu_poll_deadline,
            resize_deadline,
            native_dispatch_deadline,
        ]
        .into_iter()
        .flatten()
        .min()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn resize_coalescing_matches_the_latest_nonzero_request(
            requests in prop::collection::vec((0_u32..=4_096, 0_u32..=2_160), 0..64),
        ) {
            let start = Instant::now();
            let mut schedule = DesktopSchedule::default();
            let mut expected = None;

            for (index, (width, height)) in requests.into_iter().enumerate() {
                let elapsed_ms = u64::try_from(index)
                    .expect("generated resize sequence length fits in u64");
                let now = start + Duration::from_millis(elapsed_ms);
                let size = PhysicalSize::new(width, height);
                let action = schedule.request_resize(now, size);

                if width == 0 || height == 0 {
                    prop_assert_eq!(action, ResizeAction::ApplyNow(size));
                    expected = None;
                } else {
                    prop_assert_eq!(action, ResizeAction::Deferred);
                    let deadline = now + RESIZE_SETTLE_INTERVAL;
                    expected = Some((size, deadline));
                }
                prop_assert_eq!(schedule.resize_pending(), expected.is_some());
                prop_assert_eq!(
                    schedule.next_wake(None, true),
                    expected.map(|(_, deadline)| deadline),
                );
            }

            if let Some((size, deadline)) = expected {
                prop_assert_eq!(schedule.take_ready_resize(deadline, false), None);
                prop_assert_eq!(schedule.take_ready_resize(deadline, true), Some(size));
                prop_assert!(!schedule.resize_pending());
            } else {
                prop_assert_eq!(
                    schedule.take_ready_resize(start + Duration::from_secs(1), true),
                    None,
                );
            }
        }

        #[test]
        fn next_wake_is_the_minimum_eligible_application_or_native_deadline(
            repaint_ms in prop::option::of(0_u64..=100),
            native_ms in prop::option::of(0_u64..=100),
            resize_pending in any::<bool>(),
            has_pending_work in any::<bool>(),
            resize_ready in any::<bool>(),
        ) {
            let now = Instant::now();
            let mut schedule = DesktopSchedule::default();
            if let Some(delay_ms) = repaint_ms {
                schedule.request_repaint(now, Duration::from_millis(delay_ms));
            }
            if resize_pending {
                let action = schedule.request_resize(now, PhysicalSize::new(1_280, 720));
                prop_assert_eq!(action, ResizeAction::Deferred);
            }
            schedule.after_gpu_poll(now, has_pending_work);
            let native = native_ms.map(|delay_ms| now + Duration::from_millis(delay_ms));
            let expected = [
                repaint_ms.map(|delay_ms| now + Duration::from_millis(delay_ms)),
                has_pending_work.then_some(now + GPU_POLL_INTERVAL),
                (resize_pending && resize_ready).then_some(now + RESIZE_SETTLE_INTERVAL),
                native,
            ]
            .into_iter()
            .flatten()
            .min();

            prop_assert_eq!(schedule.next_wake(native, resize_ready), expected);
        }
    }

    #[test]
    fn gpu_polling_does_not_consume_or_request_a_repaint() {
        let now = Instant::now();
        let mut schedule = DesktopSchedule::default();

        schedule.request_repaint(now, Duration::from_millis(10));
        schedule.after_gpu_poll(now, true);

        let poll = now + GPU_POLL_INTERVAL;
        assert_eq!(schedule.next_wake(None, false), Some(poll));
        assert!(!schedule.take_due_repaint(poll));

        schedule.after_gpu_poll(poll, false);
        let repaint = now + Duration::from_millis(10);
        assert_eq!(schedule.next_wake(None, true), Some(repaint));
        assert!(schedule.take_due_repaint(repaint));
        assert_eq!(schedule.next_wake(None, true), None);
    }
}

use std::time::{Duration, Instant};

use winit::dpi::PhysicalSize;

const GPU_POLL_INTERVAL: Duration = Duration::from_millis(2);
const RESIZE_SETTLE_INTERVAL: Duration = Duration::from_millis(40);

pub fn earliest(first: Option<Instant>, second: Option<Instant>) -> Option<Instant> {
    match (first, second) {
        (Some(first), Some(second)) => Some(first.min(second)),
        (Some(deadline), None) | (None, Some(deadline)) => Some(deadline),
        (None, None) => None,
    }
}

/// Coalesces live-resize events until the GPU can replace resources safely.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PendingResize {
    request: Option<(PhysicalSize<u32>, Instant)>,
}

impl PendingResize {
    pub fn request(&mut self, now: Instant, size: PhysicalSize<u32>) -> bool {
        if size.width == 0 || size.height == 0 {
            self.clear();
            return true;
        }
        let deadline = now.checked_add(RESIZE_SETTLE_INTERVAL).unwrap_or(now);
        self.request = Some((size, deadline));
        false
    }

    pub fn take_due(&mut self, now: Instant, gpu_idle: bool) -> Option<PhysicalSize<u32>> {
        let (size, deadline) = self.request?;
        if !gpu_idle || deadline > now {
            return None;
        }
        self.request = None;
        Some(size)
    }

    pub const fn is_pending(self) -> bool {
        self.request.is_some()
    }

    pub const fn deadline(self) -> Option<Instant> {
        match self.request {
            Some((_, deadline)) => Some(deadline),
            None => None,
        }
    }

    pub const fn clear(&mut self) {
        self.request = None;
    }
}

/// Keeps GPU polling independent from requests to repaint the surface.
///
/// Source: <https://docs.rs/winit/0.30.13/winit/application/trait.ApplicationHandler.html#method.about_to_wait>
#[derive(Debug, Default)]
pub struct EventLoopSchedule {
    repaint_deadline: Option<Instant>,
    gpu_poll_deadline: Option<Instant>,
}

impl EventLoopSchedule {
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

    pub fn next_wake(&self) -> Option<Instant> {
        earliest(self.repaint_deadline, self.gpu_poll_deadline)
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
            let mut pending = PendingResize::default();
            let mut expected = None;

            for (index, (width, height)) in requests.into_iter().enumerate() {
                let elapsed_ms = u64::try_from(index)
                    .expect("generated resize sequence length fits in u64");
                let now = start + Duration::from_millis(elapsed_ms);
                let size = PhysicalSize::new(width, height);
                let pauses = pending.request(now, size);

                if width == 0 || height == 0 {
                    prop_assert!(pauses);
                    expected = None;
                } else {
                    prop_assert!(!pauses);
                    let deadline = now + RESIZE_SETTLE_INTERVAL;
                    expected = Some((size, deadline));
                }
                prop_assert_eq!(pending.is_pending(), expected.is_some());
                prop_assert_eq!(pending.deadline(), expected.map(|(_, deadline)| deadline));
            }

            if let Some((size, deadline)) = expected {
                prop_assert_eq!(pending.take_due(deadline, false), None);
                prop_assert_eq!(pending.take_due(deadline, true), Some(size));
                prop_assert!(!pending.is_pending());
            } else {
                prop_assert_eq!(pending.take_due(start + Duration::from_secs(1), true), None);
            }
        }
    }

    #[test]
    fn gpu_polling_does_not_consume_or_request_a_repaint() {
        let now = Instant::now();
        let mut schedule = EventLoopSchedule::default();

        schedule.request_repaint(now, Duration::from_millis(10));
        schedule.after_gpu_poll(now, true);

        let poll = now + GPU_POLL_INTERVAL;
        assert_eq!(schedule.next_wake(), Some(poll));
        assert!(!schedule.take_due_repaint(poll));

        schedule.after_gpu_poll(poll, false);
        let repaint = now + Duration::from_millis(10);
        assert_eq!(schedule.next_wake(), Some(repaint));
        assert!(schedule.take_due_repaint(repaint));
        assert_eq!(schedule.next_wake(), None);
    }
}

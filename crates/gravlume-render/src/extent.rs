use std::num::NonZeroU32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RenderExtent {
    width: NonZeroU32,
    height: NonZeroU32,
}

impl RenderExtent {
    pub(crate) fn new(width: u32, height: u32) -> Option<Self> {
        Some(Self {
            width: NonZeroU32::new(width)?,
            height: NonZeroU32::new(height)?,
        })
    }

    pub(crate) const fn width(self) -> u32 {
        self.width.get()
    }

    pub(crate) const fn height(self) -> u32 {
        self.height.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExtentChange {
    Unchanged,
    Paused,
    Rebuild {
        extent: RenderExtent,
        generation: u64,
    },
}

#[derive(Debug, Default)]
pub(crate) struct ExtentTracker {
    extent: Option<RenderExtent>,
    generation: u64,
    is_paused: bool,
}

impl ExtentTracker {
    pub(crate) fn update(&mut self, width: u32, height: u32) -> ExtentChange {
        let Some(next) = RenderExtent::new(width, height) else {
            if self.is_paused {
                return ExtentChange::Unchanged;
            }
            self.extent = None;
            self.is_paused = true;
            return ExtentChange::Paused;
        };

        if !self.is_paused && self.extent == Some(next) {
            return ExtentChange::Unchanged;
        }

        self.is_paused = false;
        self.extent = Some(next);
        self.generation += 1;
        ExtentChange::Rebuild {
            extent: next,
            generation: self.generation,
        }
    }

    pub(crate) const fn extent(&self) -> Option<RenderExtent> {
        self.extent
    }

    pub(crate) const fn generation(&self) -> u64 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::{ExtentChange, ExtentTracker, RenderExtent};

    #[test]
    fn zero_extent_pauses_without_advancing_generation() {
        let mut tracker = ExtentTracker::default();

        assert_eq!(tracker.update(0, 720), ExtentChange::Paused);
        assert_eq!(tracker.extent(), None);
        assert_eq!(tracker.generation(), 0);
    }

    #[test]
    fn nonzero_extent_advances_generation_only_when_dimensions_change() {
        let mut tracker = ExtentTracker::default();
        let odd = RenderExtent::new(1279, 719).expect("both dimensions are nonzero");

        assert_eq!(
            tracker.update(1279, 719),
            ExtentChange::Rebuild {
                extent: odd,
                generation: 1,
            }
        );
        assert_eq!(tracker.update(1279, 719), ExtentChange::Unchanged);
        assert_eq!(tracker.generation(), 1);

        assert_eq!(
            tracker.update(1280, 719),
            ExtentChange::Rebuild {
                extent: RenderExtent::new(1280, 719).expect("both dimensions are nonzero"),
                generation: 2,
            }
        );
    }

    #[test]
    fn returning_from_zero_rebuilds_the_extent_generation() {
        let mut tracker = ExtentTracker::default();
        assert!(matches!(
            tracker.update(800, 600),
            ExtentChange::Rebuild { .. }
        ));

        assert_eq!(tracker.update(0, 0), ExtentChange::Paused);
        assert_eq!(tracker.update(0, 0), ExtentChange::Unchanged);
        assert_eq!(
            tracker.update(800, 600),
            ExtentChange::Rebuild {
                extent: RenderExtent::new(800, 600).expect("both dimensions are nonzero"),
                generation: 2,
            }
        );
    }
}

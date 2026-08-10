use std::num::NonZeroU32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderExtent {
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
pub enum ExtentChange {
    Unchanged,
    Paused,
    Rebuild {
        extent: RenderExtent,
        generation: u64,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ExtentTracker {
    extent: Option<RenderExtent>,
    generation: u64,
    is_paused: bool,
}

impl ExtentTracker {
    pub(crate) fn updated(mut self, width: u32, height: u32) -> (Self, ExtentChange) {
        let Some(next) = RenderExtent::new(width, height) else {
            if self.is_paused {
                return (self, ExtentChange::Unchanged);
            }
            self.extent = None;
            self.is_paused = true;
            return (self, ExtentChange::Paused);
        };

        if !self.is_paused && self.extent == Some(next) {
            return (self, ExtentChange::Unchanged);
        }

        self.is_paused = false;
        self.extent = Some(next);
        self.generation += 1;
        let generation = self.generation;
        (
            self,
            ExtentChange::Rebuild {
                extent: next,
                generation,
            },
        )
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

    fn extent(width: u32, height: u32) -> RenderExtent {
        RenderExtent::new(width, height).expect("test extent is nonzero")
    }

    #[test]
    fn updates_are_transactional_and_generation_based() {
        let original = ExtentTracker::default();

        let (active, change) = original.updated(1279, 719);
        assert_eq!(
            change,
            ExtentChange::Rebuild {
                extent: extent(1279, 719),
                generation: 1,
            }
        );
        assert_eq!(original.extent(), None);
        assert_eq!(original.generation(), 0);

        let (active, change) = active.updated(1279, 719);
        assert_eq!(change, ExtentChange::Unchanged);
        assert_eq!(active.generation(), 1);

        let (paused, change) = active.updated(0, 719);
        assert_eq!(change, ExtentChange::Paused);
        assert_eq!(paused.extent(), None);
        assert_eq!(paused.generation(), 1);

        let (paused, change) = paused.updated(0, 0);
        assert_eq!(change, ExtentChange::Unchanged);

        let (resumed, change) = paused.updated(1279, 719);
        assert_eq!(
            change,
            ExtentChange::Rebuild {
                extent: extent(1279, 719),
                generation: 2,
            }
        );
        assert_eq!(resumed.extent(), Some(extent(1279, 719)));
    }
}

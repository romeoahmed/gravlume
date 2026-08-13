use std::num::NonZeroU32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderExtent {
    width: NonZeroU32,
    height: NonZeroU32,
}

impl RenderExtent {
    pub(crate) const ONE: Self = Self {
        width: NonZeroU32::MIN,
        height: NonZeroU32::MIN,
    };

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
    use proptest::prelude::*;

    use super::{ExtentChange, ExtentTracker, RenderExtent};

    proptest! {
        #[test]
        fn updates_follow_extent_and_generation_contract(updates in prop::collection::vec((0_u32..=64, 0_u32..=64), 0..128)) {
            let mut tracker = ExtentTracker::default();
            let mut model_extent = None;
            let mut model_generation = 0;
            let mut model_paused = false;

            for (width, height) in updates {
                let previous = tracker;
                let (next, change) = tracker.updated(width, height);
                let requested = RenderExtent::new(width, height);
                let expected = match requested {
                    None if model_paused => ExtentChange::Unchanged,
                    None => {
                        model_extent = None;
                        model_paused = true;
                        ExtentChange::Paused
                    }
                    Some(extent) if !model_paused && model_extent == Some(extent) => ExtentChange::Unchanged,
                    Some(extent) => {
                        model_extent = Some(extent);
                        model_paused = false;
                        model_generation += 1;
                        ExtentChange::Rebuild { extent, generation: model_generation }
                    }
                };

                prop_assert_eq!(change, expected);
                prop_assert_eq!(next.extent(), model_extent);
                prop_assert_eq!(next.generation(), model_generation);
                if change == ExtentChange::Unchanged {
                    prop_assert_eq!(next.extent(), previous.extent());
                    prop_assert_eq!(next.generation(), previous.generation());
                }
                tracker = next;
            }
        }
    }
}

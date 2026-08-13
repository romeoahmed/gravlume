#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum LifecycleState {
    #[default]
    AwaitingResume,
    Active,
    Suspended,
    Fatal,
}

#[derive(Debug, Default)]
pub struct Lifecycle {
    state: LifecycleState,
}

impl Lifecycle {
    pub(crate) const fn resume(&mut self) -> bool {
        match self.state {
            LifecycleState::AwaitingResume | LifecycleState::Suspended => {
                self.state = LifecycleState::Active;
                true
            }
            LifecycleState::Active | LifecycleState::Fatal => false,
        }
    }

    pub(crate) const fn suspend(&mut self) -> bool {
        match self.state {
            LifecycleState::Active => {
                self.state = LifecycleState::Suspended;
                true
            }
            LifecycleState::AwaitingResume | LifecycleState::Suspended | LifecycleState::Fatal => {
                false
            }
        }
    }

    pub(crate) const fn fail(&mut self) {
        self.state = LifecycleState::Fatal;
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::Lifecycle;

    #[derive(Clone, Copy, Debug)]
    enum Event {
        Resume,
        Suspend,
        Fail,
    }

    fn event() -> impl Strategy<Value = Event> {
        prop_oneof![Just(Event::Resume), Just(Event::Suspend), Just(Event::Fail),]
    }

    proptest! {
        #[test]
        fn lifecycle_matches_its_small_transition_model(events in prop::collection::vec(event(), 0..64)) {
            let mut lifecycle = Lifecycle::default();
            let mut active = false;
            let mut fatal = false;

            for event in events {
                match event {
                    Event::Resume => {
                        let expected = !fatal && !active;
                        prop_assert_eq!(lifecycle.resume(), expected);
                        active |= expected;
                    }
                    Event::Suspend => {
                        let expected = !fatal && active;
                        prop_assert_eq!(lifecycle.suspend(), expected);
                        active &= !expected;
                    }
                    Event::Fail => {
                        lifecycle.fail();
                        active = false;
                        fatal = true;
                    }
                }
            }
        }
    }
}

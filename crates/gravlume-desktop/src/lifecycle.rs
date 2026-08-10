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
    use super::Lifecycle;

    #[test]
    fn redundant_resume_and_suspend_events_are_idempotent() {
        let mut lifecycle = Lifecycle::default();

        assert!(lifecycle.resume());
        assert!(!lifecycle.resume());
        assert!(lifecycle.suspend());
        assert!(!lifecycle.suspend());
        assert!(lifecycle.resume());
    }

    #[test]
    fn fatal_state_never_attempts_to_initialize_again() {
        let mut lifecycle = Lifecycle::default();
        assert!(lifecycle.resume());

        lifecycle.fail();

        assert!(!lifecycle.suspend());
        assert!(!lifecycle.resume());
    }
}

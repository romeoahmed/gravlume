#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum LifecycleState {
    #[default]
    AwaitingResume,
    Active,
    Suspended,
    Fatal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LifecycleAction {
    None,
    Initialize,
    ReleaseSurface,
}

#[derive(Debug, Default)]
pub(crate) struct Lifecycle {
    state: LifecycleState,
}

impl Lifecycle {
    pub(crate) fn resume(&mut self) -> LifecycleAction {
        match self.state {
            LifecycleState::AwaitingResume | LifecycleState::Suspended => {
                self.state = LifecycleState::Active;
                LifecycleAction::Initialize
            }
            LifecycleState::Active | LifecycleState::Fatal => LifecycleAction::None,
        }
    }

    pub(crate) fn suspend(&mut self) -> LifecycleAction {
        match self.state {
            LifecycleState::Active => {
                self.state = LifecycleState::Suspended;
                LifecycleAction::ReleaseSurface
            }
            LifecycleState::AwaitingResume | LifecycleState::Suspended | LifecycleState::Fatal => {
                LifecycleAction::None
            }
        }
    }

    pub(crate) fn fail(&mut self) {
        self.state = LifecycleState::Fatal;
    }
}

#[cfg(test)]
mod tests {
    use super::{Lifecycle, LifecycleAction};

    #[test]
    fn redundant_resume_and_suspend_events_are_idempotent() {
        let mut lifecycle = Lifecycle::default();

        assert_eq!(lifecycle.resume(), LifecycleAction::Initialize);
        assert_eq!(lifecycle.resume(), LifecycleAction::None);
        assert_eq!(lifecycle.suspend(), LifecycleAction::ReleaseSurface);
        assert_eq!(lifecycle.suspend(), LifecycleAction::None);
        assert_eq!(lifecycle.resume(), LifecycleAction::Initialize);
    }

    #[test]
    fn fatal_state_never_attempts_to_initialize_again() {
        let mut lifecycle = Lifecycle::default();
        assert_eq!(lifecycle.resume(), LifecycleAction::Initialize);

        lifecycle.fail();

        assert_eq!(lifecycle.suspend(), LifecycleAction::None);
        assert_eq!(lifecycle.resume(), LifecycleAction::None);
    }
}

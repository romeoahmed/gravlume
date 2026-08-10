#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AcquireOutcome {
    Success,
    Suboptimal,
    Timeout,
    Occluded,
    Outdated,
    Lost,
    Validation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SurfaceDirective {
    Render { reconfigure_after_present: bool },
    Skip,
    Reconfigure,
    Recreate,
    ReportValidation,
}

pub(crate) const fn directive_for(outcome: AcquireOutcome) -> SurfaceDirective {
    match outcome {
        AcquireOutcome::Success => SurfaceDirective::Render {
            reconfigure_after_present: false,
        },
        AcquireOutcome::Suboptimal => SurfaceDirective::Render {
            reconfigure_after_present: true,
        },
        AcquireOutcome::Timeout | AcquireOutcome::Occluded => SurfaceDirective::Skip,
        AcquireOutcome::Outdated => SurfaceDirective::Reconfigure,
        AcquireOutcome::Lost => SurfaceDirective::Recreate,
        AcquireOutcome::Validation => SurfaceDirective::ReportValidation,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum FrameStage {
    #[default]
    Ready,
    Acquired,
    Submitted,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum FrameProtocolError {
    #[error("frame submitted before acquiring a surface texture")]
    SubmitBeforeAcquire,
    #[error("frame presented before its commands were submitted")]
    PresentBeforeSubmit,
    #[error("surface texture acquired more than once in one frame")]
    DuplicateAcquire,
    #[error("frame protocol is already complete")]
    AlreadyComplete,
}

#[derive(Debug, Default)]
pub(crate) struct FrameProtocol {
    stage: FrameStage,
}

impl FrameProtocol {
    pub(crate) fn acquired(&mut self) -> Result<(), FrameProtocolError> {
        match self.stage {
            FrameStage::Ready => {
                self.stage = FrameStage::Acquired;
                Ok(())
            }
            FrameStage::Acquired | FrameStage::Submitted => {
                Err(FrameProtocolError::DuplicateAcquire)
            }
            FrameStage::Complete => Err(FrameProtocolError::AlreadyComplete),
        }
    }

    pub(crate) fn submitted(&mut self) -> Result<(), FrameProtocolError> {
        match self.stage {
            FrameStage::Ready => Err(FrameProtocolError::SubmitBeforeAcquire),
            FrameStage::Acquired => {
                self.stage = FrameStage::Submitted;
                Ok(())
            }
            FrameStage::Submitted => Err(FrameProtocolError::AlreadyComplete),
            FrameStage::Complete => Err(FrameProtocolError::AlreadyComplete),
        }
    }

    pub(crate) fn presented(&mut self) -> Result<(), FrameProtocolError> {
        match self.stage {
            FrameStage::Ready | FrameStage::Acquired => {
                Err(FrameProtocolError::PresentBeforeSubmit)
            }
            FrameStage::Submitted => {
                self.stage = FrameStage::Complete;
                Ok(())
            }
            FrameStage::Complete => Err(FrameProtocolError::AlreadyComplete),
        }
    }

    pub(crate) const fn is_complete(&self) -> bool {
        matches!(self.stage, FrameStage::Complete)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AcquireOutcome, FrameProtocol, FrameProtocolError, SurfaceDirective, directive_for,
    };

    #[test]
    fn every_wgpu_30_acquire_outcome_has_an_explicit_directive() {
        let cases = [
            (
                AcquireOutcome::Success,
                SurfaceDirective::Render {
                    reconfigure_after_present: false,
                },
            ),
            (
                AcquireOutcome::Suboptimal,
                SurfaceDirective::Render {
                    reconfigure_after_present: true,
                },
            ),
            (AcquireOutcome::Timeout, SurfaceDirective::Skip),
            (AcquireOutcome::Occluded, SurfaceDirective::Skip),
            (AcquireOutcome::Outdated, SurfaceDirective::Reconfigure),
            (AcquireOutcome::Lost, SurfaceDirective::Recreate),
            (
                AcquireOutcome::Validation,
                SurfaceDirective::ReportValidation,
            ),
        ];

        for (outcome, expected) in cases {
            assert_eq!(directive_for(outcome), expected, "outcome: {outcome:?}");
        }
    }

    #[test]
    fn successful_frame_allows_exactly_one_acquire_submit_and_present() {
        let mut protocol = FrameProtocol::default();

        protocol.acquired().expect("first acquire is valid");
        protocol.submitted().expect("submit follows acquire");
        protocol.presented().expect("present follows submit");

        assert!(protocol.is_complete());
        assert_eq!(
            protocol.acquired(),
            Err(FrameProtocolError::AlreadyComplete)
        );
    }

    #[test]
    fn frame_protocol_rejects_out_of_order_transitions() {
        let mut protocol = FrameProtocol::default();
        assert_eq!(
            protocol.submitted(),
            Err(FrameProtocolError::SubmitBeforeAcquire)
        );

        protocol.acquired().expect("first acquire is valid");
        assert_eq!(
            protocol.presented(),
            Err(FrameProtocolError::PresentBeforeSubmit)
        );
        assert_eq!(
            protocol.acquired(),
            Err(FrameProtocolError::DuplicateAcquire)
        );
    }
}

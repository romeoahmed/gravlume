use gravlume_render::{
    SampleInspectionCompletion, SampleInspectionRequestError, SampleInspectionTicket,
};
use winit::dpi::{PhysicalPosition, PhysicalSize};

#[derive(Debug, Default)]
pub enum InspectionStatus {
    #[default]
    Idle,
    ViewportChanging,
    Pending(SampleInspectionTicket),
    Rejected(SampleInspectionRequestError),
    Finished(SampleInspectionCompletion),
}

impl InspectionStatus {
    pub(crate) fn on_publication(&mut self, generation: u64) {
        let invalidated = match self {
            Self::Idle => false,
            Self::ViewportChanging | Self::Rejected(_) => true,
            Self::Pending(ticket) => ticket.generation() != generation,
            Self::Finished(completion) => completion.ticket().generation() != generation,
        };
        if invalidated {
            *self = Self::Idle;
        }
    }

    /// Installs a terminal completion only while it still belongs to the active viewport.
    pub(crate) fn on_completion(
        &mut self,
        completion: SampleInspectionCompletion,
        current_generation: u64,
    ) -> bool {
        if !completion_is_current(self, completion.ticket().generation(), current_generation) {
            return false;
        }
        *self = Self::Finished(completion);
        true
    }

    /// Ends a viewport wait only when the renderer still targets the current complete publication.
    /// Returns whether visible state changed so the caller can schedule a redraw.
    pub(crate) fn on_viewport_settled(&mut self, has_current_publication: bool) -> bool {
        let changed = matches!(self, Self::ViewportChanging) && has_current_publication;
        if changed {
            *self = Self::Idle;
        }
        changed
    }
}

const fn completion_is_current(
    status: &InspectionStatus,
    ticket_generation: u64,
    current_generation: u64,
) -> bool {
    matches!(
        status,
        InspectionStatus::Pending(ticket)
            if ticket.generation() == ticket_generation
                && ticket_generation == current_generation
    )
}

pub fn cursor_pixel(
    position: PhysicalPosition<f64>,
    extent: PhysicalSize<u32>,
) -> Option<[u32; 2]> {
    let within_extent = position.x.is_finite()
        && position.y.is_finite()
        && position.x >= 0.0
        && position.y >= 0.0
        && position.x < f64::from(extent.width)
        && position.y < f64::from(extent.height);
    if !within_extent {
        return None;
    }
    Some([
        floor_valid_coordinate(position.x),
        floor_valid_coordinate(position.y),
    ])
}

const fn floor_valid_coordinate(coordinate: f64) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the caller proves the coordinate finite, nonnegative, and below a u32 extent"
    )]
    {
        coordinate.floor() as u32
    }
}

#[cfg(test)]
mod tests {
    use winit::dpi::{PhysicalPosition, PhysicalSize};

    use super::{InspectionStatus, completion_is_current, cursor_pixel};

    #[test]
    fn publication_reconciliation_expires_waits_and_old_generations() {
        let mut publication_wait = InspectionStatus::ViewportChanging;
        publication_wait.on_publication(9);
        assert!(matches!(publication_wait, InspectionStatus::Idle));

        let mut retained_publication_wait = InspectionStatus::ViewportChanging;
        assert!(retained_publication_wait.on_viewport_settled(true));
        assert!(matches!(retained_publication_wait, InspectionStatus::Idle));

        let mut unpublished_wait = InspectionStatus::ViewportChanging;
        assert!(!unpublished_wait.on_viewport_settled(false));
        assert!(matches!(
            unpublished_wait,
            InspectionStatus::ViewportChanging
        ));
    }

    #[test]
    fn non_pending_states_reject_same_or_old_generation_completions() {
        assert!(!completion_is_current(
            &InspectionStatus::ViewportChanging,
            9,
            9
        ));
        assert!(!completion_is_current(&InspectionStatus::Idle, 8, 9));
        assert!(!completion_is_current(&InspectionStatus::Idle, 9, 9));
    }

    #[test]
    fn finite_window_positions_map_to_the_containing_physical_pixel() {
        let extent = PhysicalSize::new(1280, 720);

        assert_eq!(
            cursor_pixel(PhysicalPosition::new(0.0, 0.0), extent),
            Some([0, 0])
        );
        assert_eq!(
            cursor_pixel(PhysicalPosition::new(41.875, 12.25), extent),
            Some([41, 12])
        );
        assert_eq!(
            cursor_pixel(PhysicalPosition::new(1279.999, 719.999), extent),
            Some([1279, 719])
        );
    }

    #[test]
    fn invalid_or_outside_window_positions_do_not_name_a_pixel() {
        let extent = PhysicalSize::new(1280, 720);
        let invalid = [
            PhysicalPosition::new(-0.001, 0.0),
            PhysicalPosition::new(0.0, -0.001),
            PhysicalPosition::new(f64::NAN, 0.0),
            PhysicalPosition::new(0.0, f64::INFINITY),
            PhysicalPosition::new(1280.0, 0.0),
            PhysicalPosition::new(0.0, 720.0),
        ];

        for position in invalid {
            assert_eq!(cursor_pixel(position, extent), None);
        }
        assert_eq!(
            cursor_pixel(PhysicalPosition::new(0.0, 0.0), PhysicalSize::new(0, 720)),
            None
        );
    }
}

use gravlume_render::{
    SampleInspectionEvent, SampleInspectionRequestError, SampleInspectionRequestId,
};
use winit::dpi::{PhysicalPosition, PhysicalSize};

#[derive(Debug, Default)]
pub enum InspectionStatus {
    #[default]
    Idle,
    ViewportChanging,
    Pending {
        request_id: SampleInspectionRequestId,
        generation: u64,
    },
    Rejected(SampleInspectionRequestError),
    Finished(SampleInspectionEvent),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PublicationAffinity {
    None,
    NextPublication,
    Generation(u64),
}

impl PublicationAffinity {
    const fn invalidated_by(self, generation: u64) -> bool {
        match self {
            Self::None => false,
            Self::NextPublication => true,
            Self::Generation(bound) => bound != generation,
        }
    }
}

impl InspectionStatus {
    pub fn on_publication(&mut self, generation: u64) {
        let affinity = match self {
            Self::Idle => PublicationAffinity::None,
            Self::ViewportChanging | Self::Rejected(_) => PublicationAffinity::NextPublication,
            Self::Pending { generation, .. } => PublicationAffinity::Generation(*generation),
            Self::Finished(event) => inspection_generation(event).map_or(
                PublicationAffinity::NextPublication,
                PublicationAffinity::Generation,
            ),
        };
        if affinity.invalidated_by(generation) {
            *self = Self::Idle;
        }
    }

    pub fn on_unchanged_viewport(&mut self) {
        if matches!(self, Self::ViewportChanging) {
            *self = Self::Idle;
        }
    }
}

const fn inspection_generation(event: &SampleInspectionEvent) -> Option<u64> {
    match event {
        SampleInspectionEvent::Completed(inspection) => Some(inspection.identity().generation()),
        SampleInspectionEvent::Cancelled(identity)
        | SampleInspectionEvent::Superseded(identity)
        | SampleInspectionEvent::Failed { identity, .. } => Some(identity.generation()),
        _ => None,
    }
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

    use super::{InspectionStatus, PublicationAffinity, cursor_pixel};

    #[test]
    fn publication_reconciliation_expires_waits_and_old_generations() {
        let mut publication_wait = InspectionStatus::ViewportChanging;
        publication_wait.on_publication(9);
        assert!(matches!(publication_wait, InspectionStatus::Idle));

        let mut unchanged_wait = InspectionStatus::ViewportChanging;
        unchanged_wait.on_unchanged_viewport();
        assert!(matches!(unchanged_wait, InspectionStatus::Idle));

        assert!(PublicationAffinity::NextPublication.invalidated_by(9));
        assert!(!PublicationAffinity::Generation(9).invalidated_by(9));
        assert!(PublicationAffinity::Generation(8).invalidated_by(9));
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

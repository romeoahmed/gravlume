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
    pub(crate) fn on_publication(
        &mut self,
        generation: u64,
        viewport_has_current_publication: bool,
    ) {
        if !viewport_has_current_publication {
            *self = Self::ViewportChanging;
            return;
        }
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
        let ticket_generation = completion.ticket().generation();
        let is_current = matches!(
            self,
            Self::Pending(ticket)
                if ticket.generation() == ticket_generation
                    && ticket_generation == current_generation
        );
        if !is_current {
            return false;
        }
        *self = Self::Finished(completion);
        true
    }

    /// Ends a viewport wait only when the renderer still targets the current complete publication.
    /// Returns whether visible state changed so the caller can schedule a redraw.
    pub(crate) fn on_viewport_settled(&mut self, viewport_has_current_publication: bool) -> bool {
        let changed = matches!(self, Self::ViewportChanging) && viewport_has_current_publication;
        if changed {
            *self = Self::Idle;
        }
        changed
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
    use proptest::prelude::*;
    use winit::dpi::{PhysicalPosition, PhysicalSize};

    use super::{InspectionStatus, cursor_pixel};

    #[test]
    fn current_publication_settles_a_viewport_wait() {
        let mut publication_wait = InspectionStatus::ViewportChanging;
        publication_wait.on_publication(9, true);
        assert!(matches!(publication_wait, InspectionStatus::Idle));

        let mut current_viewport_wait = InspectionStatus::ViewportChanging;
        assert!(current_viewport_wait.on_viewport_settled(true));
        assert!(matches!(current_viewport_wait, InspectionStatus::Idle));

        let mut unpublished_wait = InspectionStatus::ViewportChanging;
        assert!(!unpublished_wait.on_viewport_settled(false));
        assert!(matches!(
            unpublished_wait,
            InspectionStatus::ViewportChanging
        ));

        let mut mismatched_publication = InspectionStatus::Idle;
        mismatched_publication.on_publication(9, false);
        assert!(matches!(
            mismatched_publication,
            InspectionStatus::ViewportChanging
        ));
    }

    fn position_inside_physical_pixel()
    -> impl Strategy<Value = (PhysicalPosition<f64>, PhysicalSize<u32>, [u32; 2])> {
        (1_u32..=u32::MAX, 1_u32..=u32::MAX)
            .prop_flat_map(|(width, height)| {
                (
                    Just(width),
                    Just(height),
                    0..width,
                    0..height,
                    0_u16..=1023,
                    0_u16..=1023,
                )
            })
            .prop_map(
                |(width, height, pixel_x, pixel_y, subpixel_x, subpixel_y)| {
                    let position = PhysicalPosition::new(
                        f64::from(pixel_x) + f64::from(subpixel_x) / 1024.0,
                        f64::from(pixel_y) + f64::from(subpixel_y) / 1024.0,
                    );
                    (
                        position,
                        PhysicalSize::new(width, height),
                        [pixel_x, pixel_y],
                    )
                },
            )
    }

    fn position_outside_physical_extent()
    -> impl Strategy<Value = (PhysicalPosition<f64>, PhysicalSize<u32>)> {
        (1_u32..=u32::MAX, 1_u32..=u32::MAX, 0_u8..4, 0_u16..=1023).prop_map(
            |(width, height, edge, offset)| {
                let inside = 0.5;
                let negative_offset = (f64::from(offset) + 1.0) / 1024.0;
                let positive_offset = f64::from(offset) / 1024.0;
                let position = match edge {
                    0 => PhysicalPosition::new(-negative_offset, inside),
                    1 => PhysicalPosition::new(f64::from(width) + positive_offset, inside),
                    2 => PhysicalPosition::new(inside, -negative_offset),
                    _ => PhysicalPosition::new(inside, f64::from(height) + positive_offset),
                };
                (position, PhysicalSize::new(width, height))
            },
        )
    }

    proptest! {
        #[test]
        fn finite_positions_map_to_their_containing_physical_pixel(
            (position, extent, expected) in position_inside_physical_pixel(),
        ) {
            prop_assert_eq!(cursor_pixel(position, extent), Some(expected));
        }

        #[test]
        fn finite_positions_outside_each_physical_edge_are_rejected(
            (position, extent) in position_outside_physical_extent(),
        ) {
            prop_assert_eq!(cursor_pixel(position, extent), None);
        }
    }

    #[test]
    fn non_finite_positions_and_empty_extents_are_rejected() {
        let extent = PhysicalSize::new(1280, 720);
        for position in [
            PhysicalPosition::new(f64::NAN, 0.0),
            PhysicalPosition::new(f64::NEG_INFINITY, 0.0),
            PhysicalPosition::new(0.0, f64::INFINITY),
        ] {
            assert_eq!(cursor_pixel(position, extent), None);
        }
        for empty_extent in [PhysicalSize::new(0, 720), PhysicalSize::new(1280, 0)] {
            assert_eq!(
                cursor_pixel(PhysicalPosition::new(0.0, 0.0), empty_extent),
                None
            );
        }
    }
}

use std::{f64::consts::PI, num::NonZeroU32};

use crate::{ValidationIssue, ValidationIssueCode, ValidationReport, validation::validate_finite};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Angle {
    radians: f64,
}

impl Angle {
    /// Creates a finite angle in radians.
    ///
    /// # Errors
    ///
    /// Returns a validation report for a non-finite value.
    pub fn from_radians(radians: f64) -> Result<Self, ValidationReport> {
        let mut report = ValidationReport::default();
        validate_finite(&mut report, radians, "angle.radians");
        report.into_result(Self { radians })
    }

    #[must_use]
    pub const fn radians(self) -> f64 {
        self.radians
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerspectiveView {
    width: NonZeroU32,
    height: NonZeroU32,
    vertical_fov: Angle,
    tangent_half_fov: f64,
}

impl PerspectiveView {
    /// Creates a top-left-origin perspective view.
    ///
    /// # Errors
    ///
    /// Returns a validation report unless the vertical field of view lies in `(0, pi)`.
    pub fn new(
        width: NonZeroU32,
        height: NonZeroU32,
        vertical_fov: Angle,
    ) -> Result<Self, ValidationReport> {
        let mut report = ValidationReport::default();
        if vertical_fov.radians <= 0.0 || vertical_fov.radians >= PI {
            report.push(ValidationIssue::error(
                ValidationIssueCode::OutOfRange,
                "view.vertical_fov",
                "vertical field of view must lie in (0, pi)",
            ));
        }
        let tangent_half_fov = (vertical_fov.radians * 0.5).tan();
        if !tangent_half_fov.is_finite() {
            report.push(ValidationIssue::error(
                ValidationIssueCode::NonFinite,
                "view.vertical_fov",
                "vertical field of view produced a non-finite view scale",
            ));
        }
        report.into_result(Self {
            width,
            height,
            vertical_fov,
            tangent_half_fov,
        })
    }

    #[must_use]
    pub const fn width(self) -> NonZeroU32 {
        self.width
    }

    #[must_use]
    pub const fn height(self) -> NonZeroU32 {
        self.height
    }

    #[must_use]
    pub const fn vertical_fov(self) -> Angle {
        self.vertical_fov
    }

    /// Validates and stores view-independent pixel/subpixel coordinates.
    ///
    /// # Errors
    ///
    /// Returns every out-of-range or non-finite seam issue as a validation report.
    pub fn sample(
        self,
        pixel_x: u32,
        pixel_y: u32,
        subpixel_x: f64,
        subpixel_y: f64,
    ) -> Result<ImageSample, ValidationReport> {
        self.validate_sample(pixel_x, pixel_y, subpixel_x, subpixel_y)?;
        Ok(ImageSample {
            pixel_x,
            pixel_y,
            subpixel_x,
            subpixel_y,
        })
    }

    pub(super) fn sight_plane(self, sample: ImageSample) -> Result<[f64; 2], ValidationReport> {
        self.validate_sample(
            sample.pixel_x,
            sample.pixel_y,
            sample.subpixel_x,
            sample.subpixel_y,
        )?;
        let width = f64::from(self.width.get());
        let height = f64::from(self.height.get());
        let normalized_x = 2.0 * (f64::from(sample.pixel_x) + sample.subpixel_x) / width - 1.0;
        let normalized_y = 1.0 - 2.0 * (f64::from(sample.pixel_y) + sample.subpixel_y) / height;
        let aspect = width / height;
        Ok([
            aspect * self.tangent_half_fov * normalized_x,
            self.tangent_half_fov * normalized_y,
        ])
    }

    fn validate_sample(
        self,
        pixel_x: u32,
        pixel_y: u32,
        subpixel_x: f64,
        subpixel_y: f64,
    ) -> Result<(), ValidationReport> {
        let mut report = ValidationReport::default();
        if pixel_x >= self.width.get() {
            report.push(ValidationIssue::error(
                ValidationIssueCode::OutOfRange,
                "image_sample.pixel_x",
                "pixel x must be smaller than the physical width",
            ));
        }
        if pixel_y >= self.height.get() {
            report.push(ValidationIssue::error(
                ValidationIssueCode::OutOfRange,
                "image_sample.pixel_y",
                "pixel y must be smaller than the physical height",
            ));
        }
        validate_subpixel(&mut report, subpixel_x, "image_sample.subpixel_x");
        validate_subpixel(&mut report, subpixel_y, "image_sample.subpixel_y");
        report.into_result(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageSample {
    pixel_x: u32,
    pixel_y: u32,
    subpixel_x: f64,
    subpixel_y: f64,
}

impl ImageSample {
    #[must_use]
    pub const fn pixel(self) -> [u32; 2] {
        [self.pixel_x, self.pixel_y]
    }

    #[must_use]
    pub const fn subpixel(self) -> [f64; 2] {
        [self.subpixel_x, self.subpixel_y]
    }
}

fn validate_subpixel(report: &mut ValidationReport, value: f64, field_path: &'static str) {
    if !value.is_finite() {
        report.push(ValidationIssue::error(
            ValidationIssueCode::NonFinite,
            field_path,
            "subpixel offset must be finite",
        ));
    } else if !(0.0..=1.0).contains(&value) {
        report.push(ValidationIssue::error(
            ValidationIssueCode::OutOfRange,
            field_path,
            "subpixel offset must lie in [0, 1]",
        ));
    }
}

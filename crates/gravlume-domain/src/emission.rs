use crate::{ValidationIssue, ValidationIssueCode, ValidationReport, validation::validate_finite};

/// A prograde circular equatorial emitter with the baseline inverse-cube bolometric profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EquatorialCircularEmitter {
    inner_radius: f64,
    outer_radius: f64,
    intensity_at_six: f64,
}

impl EquatorialCircularEmitter {
    /// Creates the baseline equatorial source over an inclusive radial interval.
    ///
    /// The emitted intensity is `I_em(r) = I_6 / (r / 6 M)^3`. Circular-orbit existence and
    /// timelikeness remain hit-local properties because they depend on the selected spacetime.
    ///
    /// # Errors
    ///
    /// Rejects non-finite values, non-positive radii, reversed bounds, and negative intensity.
    pub fn inverse_cube_bolometric_v1(
        inner_radius_m: f64,
        outer_radius_m: f64,
        intensity_at_six_m: f64,
    ) -> Result<Self, ValidationReport> {
        let mut report = ValidationReport::default();
        validate_finite(
            &mut report,
            inner_radius_m,
            "equatorial_circular_emitter.inner_radius_m",
        );
        validate_finite(
            &mut report,
            outer_radius_m,
            "equatorial_circular_emitter.outer_radius_m",
        );
        validate_finite(
            &mut report,
            intensity_at_six_m,
            "equatorial_circular_emitter.intensity_at_six_m",
        );
        if inner_radius_m.is_finite() && inner_radius_m <= 0.0 {
            report.push(ValidationIssue::error(
                ValidationIssueCode::NonPositive,
                "equatorial_circular_emitter.inner_radius_m",
                "inner radius must be positive",
            ));
        }
        if outer_radius_m.is_finite() && outer_radius_m <= 0.0 {
            report.push(ValidationIssue::error(
                ValidationIssueCode::NonPositive,
                "equatorial_circular_emitter.outer_radius_m",
                "outer radius must be positive",
            ));
        }
        if inner_radius_m.is_finite()
            && outer_radius_m.is_finite()
            && outer_radius_m < inner_radius_m
        {
            report.push(ValidationIssue::error(
                ValidationIssueCode::OutOfRange,
                "equatorial_circular_emitter.outer_radius_m",
                "outer radius must not be smaller than inner radius",
            ));
        }
        if intensity_at_six_m.is_finite() && intensity_at_six_m < 0.0 {
            report.push(ValidationIssue::error(
                ValidationIssueCode::OutOfRange,
                "equatorial_circular_emitter.intensity_at_six_m",
                "bolometric intensity must be non-negative",
            ));
        }
        report.into_result(Self {
            inner_radius: inner_radius_m,
            outer_radius: outer_radius_m,
            intensity_at_six: intensity_at_six_m,
        })
    }

    #[must_use]
    pub const fn inner_radius_m(self) -> f64 {
        self.inner_radius
    }

    #[must_use]
    pub const fn outer_radius_m(self) -> f64 {
        self.outer_radius
    }

    #[must_use]
    pub const fn intensity_at_six_m(self) -> f64 {
        self.intensity_at_six
    }
}

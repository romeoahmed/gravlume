use gravlume_domain::{GeodesicState, KerrNewmanSpacetime};

pub const OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_M: f64 = 200.0;
pub const OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_DECIMAL: &str = "200";

pub const fn escape_event_is_armed(event_value: f64, arming_band_m: f64) -> bool {
    event_value < -arming_band_m
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventKind {
    SingularityGuard,
    Horizon,
    EquatorialSurface,
    Escape,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EventConfiguration {
    escape_radius_m: Option<f64>,
    equatorial_surface: Option<EquatorialSurface>,
}

impl EventConfiguration {
    #[must_use]
    pub const fn observation_baseline_v1() -> Self {
        Self {
            escape_radius_m: Some(OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_M),
            equatorial_surface: None,
        }
    }

    #[must_use]
    pub const fn horizon_only() -> Self {
        Self {
            escape_radius_m: None,
            equatorial_surface: None,
        }
    }

    /// Installs a finite outer escape surface.
    ///
    /// # Errors
    ///
    /// Rejects a non-finite or non-positive radius.
    pub fn with_escape_radius(escape_radius_m: f64) -> Result<Self, EventConfigurationError> {
        if !escape_radius_m.is_finite() || escape_radius_m <= 0.0 {
            return Err(EventConfigurationError::InvalidEscapeRadius);
        }
        Ok(Self {
            escape_radius_m: Some(escape_radius_m),
            equatorial_surface: None,
        })
    }

    /// Adds an equatorial terminal surface with an inclusive radial interval.
    ///
    /// # Errors
    ///
    /// Rejects non-finite, non-positive, or reversed radii.
    pub fn with_equatorial_surface(
        mut self,
        inner_radius_m: f64,
        outer_radius_m: f64,
    ) -> Result<Self, EventConfigurationError> {
        if !inner_radius_m.is_finite()
            || !outer_radius_m.is_finite()
            || inner_radius_m <= 0.0
            || outer_radius_m < inner_radius_m
        {
            return Err(EventConfigurationError::InvalidEquatorialSurface);
        }
        self.equatorial_surface = Some(EquatorialSurface {
            inner_radius_m,
            outer_radius_m,
        });
        Ok(self)
    }

    pub(super) const fn escape_radius_m(self) -> Option<f64> {
        self.escape_radius_m
    }

    pub(super) const fn equatorial_surface(self) -> Option<EquatorialSurface> {
        self.equatorial_surface
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EventConfigurationError {
    #[error("escape radius must be finite and positive")]
    InvalidEscapeRadius,
    #[error("equatorial surface radii must be finite, positive, and ordered")]
    InvalidEquatorialSurface,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EquatorialSurface {
    inner_radius_m: f64,
    outer_radius_m: f64,
}

impl EquatorialSurface {
    pub(super) fn contains(self, spacetime: KerrNewmanSpacetime, state: GeodesicState) -> bool {
        spacetime
            .radius(state.event())
            .is_ok_and(|radius| (self.inner_radius_m..=self.outer_radius_m).contains(&radius))
    }
}

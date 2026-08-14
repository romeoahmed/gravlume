use std::f64::consts::PI;

use gravlume_domain::{GeodesicState, GeometryError, KerrNewmanSpacetime, KerrSchildChart};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EquatorialCircularSurface {
    inner_radius_m: f64,
    outer_radius_m: f64,
}

impl EquatorialCircularSurface {
    /// Creates a prograde circular emitter over an inclusive radial interval.
    ///
    /// Orbit existence and timelikeness are evaluated at the localized source anchor because they
    /// depend on the trace spacetime.
    ///
    /// # Errors
    ///
    /// Rejects non-finite, non-positive, or reversed radii.
    pub fn new(
        inner_radius_m: f64,
        outer_radius_m: f64,
    ) -> Result<Self, SurfaceConfigurationError> {
        if !inner_radius_m.is_finite()
            || !outer_radius_m.is_finite()
            || inner_radius_m <= 0.0
            || outer_radius_m < inner_radius_m
        {
            return Err(SurfaceConfigurationError::InvalidRadialInterval);
        }
        Ok(Self {
            inner_radius_m,
            outer_radius_m,
        })
    }

    #[must_use]
    pub const fn inner_radius_m(self) -> f64 {
        self.inner_radius_m
    }

    #[must_use]
    pub const fn outer_radius_m(self) -> f64 {
        self.outer_radius_m
    }

    pub(crate) fn observable_at(
        self,
        spacetime: KerrNewmanSpacetime,
        state: GeodesicState,
        observer_frequency: f64,
    ) -> Result<SurfaceObservable, SurfaceObservableError> {
        let radius_m = spacetime.radius(state.event())?;
        if !(self.inner_radius_m..=self.outer_radius_m).contains(&radius_m) {
            return Err(SurfaceObservableError::HitOutsideConfiguredSurface);
        }

        let charge_squared = spacetime.charge_m() * spacetime.charge_m();
        let circular_root_squared = spacetime.mass_m().mul_add(radius_m, -charge_squared);
        if !circular_root_squared.is_finite() || circular_root_squared < 0.0 {
            return Err(SurfaceObservableError::CircularOrbitUnavailable);
        }
        let circular_root = circular_root_squared.sqrt();
        let branch_sign = if spacetime.spin_m() < 0.0 { -1.0 } else { 1.0 };
        let radius_squared = radius_m * radius_m;
        let angular_velocity_denominator =
            (branch_sign * spacetime.spin_m()).mul_add(circular_root, radius_squared);
        if !angular_velocity_denominator.is_finite() || angular_velocity_denominator <= 0.0 {
            return Err(SurfaceObservableError::CircularOrbitUnavailable);
        }
        let angular_velocity = branch_sign * circular_root / angular_velocity_denominator;

        let radial_numerator = (2.0 * spacetime.mass_m()).mul_add(radius_m, -charge_squared);
        let spin_squared = spacetime.spin_m() * spacetime.spin_m();
        let timelike_discriminant = (-3.0 * spacetime.mass_m()).mul_add(
            radius_m,
            (2.0 * branch_sign * spacetime.spin_m()).mul_add(
                circular_root,
                2.0_f64.mul_add(charge_squared, radius_squared),
            ),
        );
        if !timelike_discriminant.is_finite() || timelike_discriminant <= 0.0 {
            return Err(SurfaceObservableError::CircularOrbitIsNotTimelike);
        }
        let delta = radius_squared.mul_add(
            1.0,
            (2.0 * spacetime.mass_m()).mul_add(-radius_m, spin_squared + charge_squared),
        );
        let g_tt = -1.0 + radial_numerator / radius_squared;
        let g_t_phi = -radial_numerator * spacetime.spin_m() / radius_squared;
        let radial_factor = radius_squared + spin_squared;
        let g_phi_phi =
            radial_factor.mul_add(radial_factor, -spin_squared * delta) / radius_squared;
        let circular_norm_squared = angular_velocity.mul_add(
            angular_velocity * g_phi_phi,
            (2.0 * g_t_phi).mul_add(angular_velocity, g_tt),
        );
        if !circular_norm_squared.is_finite() || circular_norm_squared >= 0.0 {
            return Err(SurfaceObservableError::CircularOrbitIsNotTimelike);
        }
        let time_component = (-circular_norm_squared).sqrt().recip();

        let invariants = spacetime.invariants(state)?;
        let emitter_frequency = time_component
            * angular_velocity.mul_add(-invariants.angular_momentum_z(), invariants.energy());
        if !emitter_frequency.is_finite() || emitter_frequency <= 0.0 {
            return Err(SurfaceObservableError::NonPositiveEmitterFrequency);
        }
        if !observer_frequency.is_finite() || observer_frequency <= 0.0 {
            return Err(SurfaceObservableError::InvalidObserverFrequency);
        }
        let frequency_ratio = observer_frequency / emitter_frequency;
        if !frequency_ratio.is_finite() || frequency_ratio <= 0.0 {
            return Err(SurfaceObservableError::NonRepresentableFrequencyRatio);
        }

        let event = state.event().to_txyz();
        let chart_spin = match spacetime.chart() {
            KerrSchildChart::Ingoing => spacetime.spin_m(),
            KerrSchildChart::Outgoing => -spacetime.spin_m(),
        };
        let cartesian_azimuth = event[2].atan2(event[1]);
        let azimuth_rad = wrap_angle(cartesian_azimuth - chart_spin.atan2(radius_m));
        if !azimuth_rad.is_finite() {
            return Err(SurfaceObservableError::NonRepresentableSourceAnchor);
        }

        Ok(SurfaceObservable {
            source_anchor: SourceAnchor::EquatorialSurface(EquatorialSurfaceAnchor {
                radius_m,
                azimuth_rad,
            }),
            frequency_ratio: FrequencyRatio(frequency_ratio),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SurfaceConfigurationError {
    #[error("equatorial surface radii must be finite, positive, and ordered")]
    InvalidRadialInterval,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SourceAnchor {
    EquatorialSurface(EquatorialSurfaceAnchor),
}

impl SourceAnchor {
    #[must_use]
    pub const fn as_equatorial_surface(self) -> Option<EquatorialSurfaceAnchor> {
        match self {
            Self::EquatorialSurface(anchor) => Some(anchor),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EquatorialSurfaceAnchor {
    radius_m: f64,
    azimuth_rad: f64,
}

impl EquatorialSurfaceAnchor {
    #[must_use]
    pub const fn radius_m(self) -> f64 {
        self.radius_m
    }

    /// Returns the selected Kerr-Schild chart's oblate azimuth in `[-pi, pi)`.
    #[must_use]
    pub const fn azimuth_rad(self) -> f64 {
        self.azimuth_rad
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrequencyRatio(f64);

impl FrequencyRatio {
    #[must_use]
    pub const fn value(self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceObservable {
    source_anchor: SourceAnchor,
    frequency_ratio: FrequencyRatio,
}

impl SurfaceObservable {
    #[must_use]
    pub const fn source_anchor(self) -> SourceAnchor {
        self.source_anchor
    }

    #[must_use]
    pub const fn frequency_ratio(self) -> FrequencyRatio {
        self.frequency_ratio
    }

    /// Applies vacuum bolometric transport, `I_obs = g^4 I_em`.
    ///
    /// # Errors
    ///
    /// Rejects negative/non-finite emitted intensity or an unrepresentable result.
    pub fn vacuum_observed_bolometric_intensity(
        self,
        emitted_bolometric_intensity: f64,
    ) -> Result<f64, SurfaceObservableError> {
        if !emitted_bolometric_intensity.is_finite() || emitted_bolometric_intensity < 0.0 {
            return Err(SurfaceObservableError::InvalidEmittedBolometricIntensity);
        }
        let observed = self.frequency_ratio.0.powi(4) * emitted_bolometric_intensity;
        if observed.is_finite() && observed >= 0.0 {
            Ok(observed)
        } else {
            Err(SurfaceObservableError::NonRepresentableObservedIntensity)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SurfaceObservableError {
    #[error("source geometry is undefined: {0}")]
    Geometry(#[from] GeometryError),
    #[error("localized source anchor lies outside the configured surface")]
    HitOutsideConfiguredSurface,
    #[error("the selected prograde circular orbit does not exist at the source anchor")]
    CircularOrbitUnavailable,
    #[error("the selected prograde circular orbit is not timelike at the source anchor")]
    CircularOrbitIsNotTimelike,
    #[error("the emitter-frame photon frequency is not finite and positive")]
    NonPositiveEmitterFrequency,
    #[error("the observer-frame photon frequency is not finite and positive")]
    InvalidObserverFrequency,
    #[error("the observer-to-emitter frequency ratio is not representable")]
    NonRepresentableFrequencyRatio,
    #[error("the source anchor is not representable")]
    NonRepresentableSourceAnchor,
    #[error("emitted bolometric intensity must be finite and non-negative")]
    InvalidEmittedBolometricIntensity,
    #[error("observed bolometric intensity is not representable")]
    NonRepresentableObservedIntensity,
}

fn wrap_angle(angle: f64) -> f64 {
    (angle + PI).rem_euclid(2.0 * PI) - PI
}

#[cfg(test)]
mod tests {
    use gravlume_domain::{GeodesicState, KerrNewmanSpacetime, KerrSchildChart};

    use super::{EquatorialCircularSurface, SurfaceObservableError, wrap_angle};

    #[test]
    fn schwarzschild_frequency_ratio_matches_the_circular_orbit_closed_form() {
        let spacetime = KerrNewmanSpacetime::new(1.0, 0.0, 0.0, KerrSchildChart::Ingoing)
            .expect("Schwarzschild spacetime is valid");
        let state = GeodesicState::new([0.0, 6.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0])
            .expect("state is finite");
        let observable = EquatorialCircularSurface::new(6.0, 20.0)
            .expect("surface is valid")
            .observable_at(spacetime, state, 1.0)
            .expect("the r = 6 M circular orbit is timelike");

        let expected = 0.5_f64.sqrt();
        assert!((observable.frequency_ratio().value() - expected).abs() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn source_anchor_inverts_each_chart_handed_oblate_map() {
        let radius_m = 10.0;
        let expected_azimuth = 1.2;
        for chart in [KerrSchildChart::Ingoing, KerrSchildChart::Outgoing] {
            let spacetime =
                KerrNewmanSpacetime::new(1.0, 0.8, 0.0, chart).expect("Kerr spacetime is valid");
            let [x, y, z] = spacetime.oblate_to_cartesian(
                radius_m,
                std::f64::consts::FRAC_PI_2,
                expected_azimuth,
            );
            let state =
                GeodesicState::new([0.0, x, y, z], [-1.0, 0.0, 0.0, 0.0]).expect("state is finite");
            let observable = EquatorialCircularSurface::new(6.0, 20.0)
                .expect("surface is valid")
                .observable_at(spacetime, state, 1.0)
                .expect("the source orbit is timelike");
            let anchor = observable
                .source_anchor()
                .as_equatorial_surface()
                .expect("source is equatorial");

            assert!((anchor.radius_m() - radius_m).abs() <= 8.0 * f64::EPSILON);
            assert!(
                wrap_angle(anchor.azimuth_rad() - expected_azimuth).abs() <= 8.0 * f64::EPSILON
            );
        }
    }

    #[test]
    fn null_circular_limit_is_not_clamped_into_a_timelike_emitter() {
        let spacetime = KerrNewmanSpacetime::new(1.0, 0.0, 0.0, KerrSchildChart::Ingoing)
            .expect("Schwarzschild spacetime is valid");
        let state = GeodesicState::new([0.0, 3.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0])
            .expect("state is finite");
        let error = EquatorialCircularSurface::new(3.0, 3.0)
            .expect("surface interval is valid")
            .observable_at(spacetime, state, 1.0)
            .expect_err("the photon orbit is not a timelike emitter orbit");

        assert_eq!(error, SurfaceObservableError::CircularOrbitIsNotTimelike);
    }
}

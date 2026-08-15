use std::f64::consts::{PI, TAU};

use gravlume_domain::{
    EquatorialCircularEmitter, GeodesicState, GeometryError, HomogeneousScalarSlab,
    KerrNewmanSpacetime, KerrSchildChart,
};

use crate::radiation::{
    SpectralTransportError, blackbody_band_intensities, transport_blackbody_bands,
    transport_bolometric_intensity,
};

pub fn observable_at(
    emitter: EquatorialCircularEmitter,
    spacetime: KerrNewmanSpacetime,
    state: GeodesicState,
    observer_frequency: f64,
    homogeneous_scalar_slab: Option<HomogeneousScalarSlab>,
) -> Result<SurfaceObservable, SurfaceObservableError> {
    let (radius_m, frequency_ratio) =
        surface_radius_and_frequency_ratio(emitter, spacetime, state, observer_frequency)?;
    let mass_m = spacetime.mass_m();
    let emitted_bolometric_intensity = emitted_bolometric_intensity(emitter, mass_m, radius_m)?;
    let vacuum_observed_bolometric_intensity =
        vacuum_observed_bolometric_intensity(emitted_bolometric_intensity, frequency_ratio)?;
    let emitted_temperature_kelvin = emitted_blackbody_temperature(emitter, radius_m / mass_m)?;
    let vacuum_observed_temperature_kelvin = emitted_temperature_kelvin
        .map(|temperature| temperature * frequency_ratio)
        .map(|temperature| {
            if temperature.is_finite() && temperature > 0.0 {
                Ok(temperature)
            } else {
                Err(SurfaceObservableError::NonRepresentableTemperature)
            }
        })
        .transpose()?;
    let vacuum_spectral_band_intensities = vacuum_observed_temperature_kelvin
        .map(|temperature| {
            blackbody_band_intensities(vacuum_observed_bolometric_intensity, temperature)
                .ok_or(SurfaceObservableError::NonRepresentableSpectrum)
        })
        .transpose()?;
    let (observed_bolometric_intensity, optical_depth, _) = transport_bolometric_intensity(
        vacuum_observed_bolometric_intensity,
        homogeneous_scalar_slab,
    )
    .ok_or(SurfaceObservableError::NonRepresentableObservedIntensity)?;
    let observed_spectral_band_intensities = vacuum_spectral_band_intensities
        .map(|bands| {
            transport_blackbody_bands(bands, homogeneous_scalar_slab).map_err(|error| match error {
                SpectralTransportError::UnresolvedSourceSpectrum => {
                    SurfaceObservableError::UnresolvedSlabSourceSpectrum
                }
                SpectralTransportError::NonRepresentable => {
                    SurfaceObservableError::NonRepresentableSpectrum
                }
            })
        })
        .transpose()?;

    let [_, x, y, _] = state.event().to_txyz();
    let chart_spin = match spacetime.chart() {
        KerrSchildChart::Ingoing => spacetime.spin_m(),
        KerrSchildChart::Outgoing => -spacetime.spin_m(),
    };
    let azimuth_rad = wrapped_angle_difference(y.atan2(x), chart_spin.atan2(radius_m));

    Ok(SurfaceObservable {
        source_anchor: SourceAnchor {
            radius_m,
            azimuth_rad,
        },
        frequency_ratio: FrequencyRatio(frequency_ratio),
        emitted_bolometric_intensity,
        vacuum_observed_bolometric_intensity,
        observed_bolometric_intensity,
        optical_depth,
        emitted_temperature_kelvin,
        vacuum_observed_temperature_kelvin,
        observed_spectral_band_intensities,
    })
}

fn surface_radius_and_frequency_ratio(
    emitter: EquatorialCircularEmitter,
    spacetime: KerrNewmanSpacetime,
    state: GeodesicState,
    observer_frequency: f64,
) -> Result<(f64, f64), SurfaceObservableError> {
    let radius_m = spacetime.radius(state.event())?;
    if !(emitter.inner_radius_m()..=emitter.outer_radius_m()).contains(&radius_m) {
        return Err(SurfaceObservableError::HitOutsideConfiguredSurface);
    }

    let mass_m = spacetime.mass_m();
    let spin_m = spacetime.spin_m();
    let charge_squared = spacetime.charge_m() * spacetime.charge_m();
    let circular_root_squared = mass_m.mul_add(radius_m, -charge_squared);
    if !circular_root_squared.is_finite() || circular_root_squared < 0.0 {
        return Err(SurfaceObservableError::CircularOrbitUnavailable);
    }
    let circular_root = circular_root_squared.sqrt();
    let branch_sign = if spin_m < 0.0 { -1.0 } else { 1.0 };
    let radius_squared = radius_m * radius_m;
    let angular_velocity_denominator =
        (branch_sign * spin_m).mul_add(circular_root, radius_squared);
    if !angular_velocity_denominator.is_finite() || angular_velocity_denominator <= 0.0 {
        return Err(SurfaceObservableError::CircularOrbitUnavailable);
    }
    let angular_velocity = branch_sign * circular_root / angular_velocity_denominator;

    let radial_numerator = (2.0 * mass_m).mul_add(radius_m, -charge_squared);
    let spin_squared = spin_m * spin_m;
    let timelike_discriminant = (-3.0 * mass_m).mul_add(
        radius_m,
        (2.0 * branch_sign * spin_m).mul_add(
            circular_root,
            2.0_f64.mul_add(charge_squared, radius_squared),
        ),
    );
    if !timelike_discriminant.is_finite() || timelike_discriminant <= 0.0 {
        return Err(SurfaceObservableError::CircularOrbitIsNotTimelike);
    }
    let delta = radius_squared.mul_add(
        1.0,
        (2.0 * mass_m).mul_add(-radius_m, spin_squared + charge_squared),
    );
    let g_tt = -1.0 + radial_numerator / radius_squared;
    let g_t_phi = -radial_numerator * spin_m / radius_squared;
    let radial_factor = radius_squared + spin_squared;
    let g_phi_phi = radial_factor.mul_add(radial_factor, -spin_squared * delta) / radius_squared;
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
    Ok((radius_m, frequency_ratio))
}

pub fn emitted_blackbody_temperature(
    emitter: EquatorialCircularEmitter,
    radius_over_mass: f64,
) -> Result<Option<f64>, SurfaceObservableError> {
    let Some(temperature_at_six_kelvin) = emitter.blackbody_temperature_at_six_kelvin() else {
        return Ok(None);
    };
    let radius_ratio = radius_over_mass / 6.0;
    let temperature = temperature_at_six_kelvin / (radius_ratio * radius_ratio.sqrt()).sqrt();
    if temperature.is_finite() && temperature > 0.0 {
        Ok(Some(temperature))
    } else {
        Err(SurfaceObservableError::NonRepresentableTemperature)
    }
}

pub fn emitted_bolometric_intensity(
    emitter: EquatorialCircularEmitter,
    mass_m: f64,
    radius_m: f64,
) -> Result<f64, SurfaceObservableError> {
    let radius_ratio = radius_m / mass_m / 6.0;
    if !radius_ratio.is_finite() || radius_ratio <= 0.0 {
        return Err(SurfaceObservableError::NonRepresentableEmittedIntensity);
    }
    let mut intensity = emitter.intensity_at_six_m();
    for _ in 0..3 {
        intensity /= radius_ratio;
    }
    if intensity.is_finite() && intensity >= 0.0 {
        Ok(intensity)
    } else {
        Err(SurfaceObservableError::NonRepresentableEmittedIntensity)
    }
}

pub fn vacuum_observed_bolometric_intensity(
    emitted_bolometric_intensity: f64,
    frequency_ratio: f64,
) -> Result<f64, SurfaceObservableError> {
    // Starting from intensity avoids an out-of-range g^4 when the final product is finite.
    let mut observed = emitted_bolometric_intensity;
    for _ in 0..4 {
        observed *= frequency_ratio;
    }
    if observed.is_finite() && observed >= 0.0 {
        Ok(observed)
    } else {
        Err(SurfaceObservableError::NonRepresentableObservedIntensity)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceAnchor {
    radius_m: f64,
    azimuth_rad: f64,
}

impl SourceAnchor {
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
    emitted_bolometric_intensity: f64,
    vacuum_observed_bolometric_intensity: f64,
    observed_bolometric_intensity: f64,
    optical_depth: f64,
    emitted_temperature_kelvin: Option<f64>,
    vacuum_observed_temperature_kelvin: Option<f64>,
    observed_spectral_band_intensities: Option<[f64; 3]>,
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

    #[must_use]
    pub const fn emitted_bolometric_intensity(self) -> f64 {
        self.emitted_bolometric_intensity
    }

    #[must_use]
    pub const fn vacuum_observed_bolometric_intensity(self) -> f64 {
        self.vacuum_observed_bolometric_intensity
    }

    #[must_use]
    pub const fn observed_bolometric_intensity(self) -> f64 {
        self.observed_bolometric_intensity
    }

    #[must_use]
    pub const fn optical_depth(self) -> f64 {
        self.optical_depth
    }

    #[must_use]
    pub const fn emitted_temperature_kelvin(self) -> Option<f64> {
        self.emitted_temperature_kelvin
    }

    #[must_use]
    pub const fn vacuum_observed_temperature_kelvin(self) -> Option<f64> {
        self.vacuum_observed_temperature_kelvin
    }

    #[must_use]
    pub const fn observed_spectral_band_intensities(self) -> Option<[f64; 3]> {
        self.observed_spectral_band_intensities
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
    #[error("the emitted bolometric intensity is not representable")]
    NonRepresentableEmittedIntensity,
    #[error("the observed bolometric intensity is not representable")]
    NonRepresentableObservedIntensity,
    #[error("the emitted or observed blackbody temperature is not representable")]
    NonRepresentableTemperature,
    #[error("the versioned boxcar spectrum is not representable")]
    NonRepresentableSpectrum,
    #[error("a non-zero neutral slab source cannot be combined with a resolved blackbody spectrum")]
    UnresolvedSlabSourceSpectrum,
}

pub fn wrapped_angle_difference(left: f64, right: f64) -> f64 {
    (left - right + PI).rem_euclid(TAU) - PI
}

#[cfg(test)]
mod tests {
    use gravlume_domain::{
        EquatorialCircularEmitter, GeodesicState, KerrNewmanSpacetime, KerrSchildChart,
    };
    use proptest::prelude::*;

    use super::{SurfaceObservableError, observable_at, vacuum_observed_bolometric_intensity};

    #[test]
    fn schwarzschild_frequency_ratio_matches_the_circular_orbit_closed_form() {
        let spacetime = KerrNewmanSpacetime::new(1.0, 0.0, 0.0, KerrSchildChart::Ingoing)
            .expect("Schwarzschild spacetime is valid");
        let state = GeodesicState::new([0.0, 6.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0])
            .expect("state is finite");
        let observable = observable_at(surface(6.0, 20.0), spacetime, state, 1.0, None)
            .expect("the r = 6 M circular orbit is timelike");

        let expected = 0.5_f64.sqrt();
        assert!((observable.frequency_ratio().value() - expected).abs() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn null_circular_limit_is_not_clamped_into_a_timelike_emitter() {
        let spacetime = KerrNewmanSpacetime::new(1.0, 0.0, 0.0, KerrSchildChart::Ingoing)
            .expect("Schwarzschild spacetime is valid");
        let state = GeodesicState::new([0.0, 3.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0])
            .expect("state is finite");
        let error = observable_at(surface(3.0, 3.0), spacetime, state, 1.0, None)
            .expect_err("the photon orbit is not a timelike emitter orbit");

        assert_eq!(error, SurfaceObservableError::CircularOrbitIsNotTimelike);
    }

    proptest! {
        #[test]
        fn source_anchor_round_trips_chart_oblate_coordinates(
            chart in prop_oneof![
                Just(KerrSchildChart::Ingoing),
                Just(KerrSchildChart::Outgoing),
            ],
            spin_m in -0.99_f64..=0.99,
            radius_m in 6.0_f64..=20.0,
            azimuth_rad in -std::f64::consts::PI..std::f64::consts::PI,
        ) {
            let spacetime = KerrNewmanSpacetime::new(1.0, spin_m, 0.0, chart)
                .expect("generated Kerr spacetime is subextremal");
            let [x, y, z] = spacetime.oblate_to_cartesian(
                radius_m,
                std::f64::consts::FRAC_PI_2,
                azimuth_rad,
            );
            let state = GeodesicState::new([0.0, x, y, z], [-1.0, 0.0, 0.0, 0.0])
                .expect("generated state is finite");
            let observable = observable_at(surface(6.0, 20.0), spacetime, state, 1.0, None)
                .expect("generated circular source orbit is timelike");
            let anchor = observable.source_anchor();
            let azimuth_difference = anchor.azimuth_rad() - azimuth_rad;
            let azimuth_error = azimuth_difference.sin().atan2(azimuth_difference.cos()).abs();

            prop_assert!(
                (anchor.radius_m() - radius_m).abs() <= 64.0 * f64::EPSILON * radius_m
            );
            prop_assert!(azimuth_error <= 64.0 * f64::EPSILON);
        }

        #[test]
        fn bolometric_transport_matches_representable_binary_scaling(
            frequency_exponent in -341_i32..=340,
        ) {
            let frequency_ratio = 2.0_f64.powi(frequency_exponent);
            let emitted = 2.0_f64.powi(-3 * frequency_exponent);
            let observed = vacuum_observed_bolometric_intensity(emitted, frequency_ratio)
                .expect("generated final intensity is representable");

            prop_assert_eq!(observed.to_bits(), frequency_ratio.to_bits());
        }

        #[test]
        fn zero_bolometric_intensity_is_fixed_for_every_finite_binary_shift(
            frequency_exponent in -1022_i32..=1023,
        ) {
            let observed = vacuum_observed_bolometric_intensity(
                0.0,
                2.0_f64.powi(frequency_exponent),
            )
                .expect("zero intensity remains representable");

            prop_assert_eq!(observed.to_bits(), 0.0_f64.to_bits());
        }
    }

    fn surface(inner_radius_m: f64, outer_radius_m: f64) -> EquatorialCircularEmitter {
        EquatorialCircularEmitter::inverse_cube_bolometric_v1(inner_radius_m, outer_radius_m, 1.0)
            .expect("surface is valid")
    }
}

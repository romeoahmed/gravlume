use crate::{ValidationIssue, ValidationIssueCode, ValidationReport, validation::validate_finite};

/// Versioned observer-frame boxcar bands, ordered as red, green, and blue output channels.
pub const VISIBLE_BOXCAR_BANDS_V1: [SpectralBand; 3] = [
    SpectralBand::new("red", 600.0, 700.0),
    SpectralBand::new("green", 500.0, 600.0),
    SpectralBand::new("blue", 400.0, 500.0),
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralBand {
    name: &'static str,
    lower_wavelength_nm: f64,
    upper_wavelength_nm: f64,
}

impl SpectralBand {
    const fn new(name: &'static str, lower_wavelength_nm: f64, upper_wavelength_nm: f64) -> Self {
        Self {
            name,
            lower_wavelength_nm,
            upper_wavelength_nm,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    #[must_use]
    pub const fn lower_wavelength_nm(self) -> f64 {
        self.lower_wavelength_nm
    }

    #[must_use]
    pub const fn upper_wavelength_nm(self) -> f64 {
        self.upper_wavelength_nm
    }
}

/// The spectral interpretation of a homogeneous scalar slab's emission term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarSlabEmissionModel {
    /// A neutral bolometric term with no resolved spectrum.
    NeutralBolometric,
    /// A diluted-blackbody term at one observer-frame temperature.
    BlackbodyV1 { temperature_kelvin: f64 },
}

/// A path-integrated homogeneous grey slab between a resolved source and the observer.
///
/// The slab stores total optical depth rather than a coordinate thickness. It compiles either a
/// constant source function or a pure-emission integral into the non-negative emission term of
/// `I_out = I_in exp(-tau) + E`. Spatially varying media require ordered path samples and are
/// outside this type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HomogeneousScalarSlab {
    optical_depth: f64,
    integrated_bolometric_emission: f64,
    emission_model: ScalarSlabEmissionModel,
}

impl HomogeneousScalarSlab {
    /// Creates a pure-absorption slab with zero source function.
    ///
    /// # Errors
    ///
    /// Rejects a non-finite or negative optical depth.
    pub fn pure_absorption_v1(optical_depth: f64) -> Result<Self, ValidationReport> {
        Self::constant_bolometric_v1(optical_depth, 0.0)
    }

    /// Creates a grey slab with a constant neutral bolometric source function.
    ///
    /// # Errors
    ///
    /// Rejects non-finite values, negative optical depth, and negative source intensity.
    pub fn constant_bolometric_v1(
        optical_depth: f64,
        source_bolometric_intensity: f64,
    ) -> Result<Self, ValidationReport> {
        Self::validated(
            optical_depth,
            EmissionInput::ConstantSource(source_bolometric_intensity),
            ScalarSlabEmissionModel::NeutralBolometric,
        )
    }

    /// Creates a grey slab with a constant diluted-blackbody source function.
    ///
    /// # Errors
    ///
    /// Rejects the bolometric slab's invalid inputs and a non-finite or non-positive temperature.
    pub fn constant_blackbody_v1(
        optical_depth: f64,
        source_bolometric_intensity: f64,
        source_temperature_kelvin: f64,
    ) -> Result<Self, ValidationReport> {
        Self::validated(
            optical_depth,
            EmissionInput::ConstantSource(source_bolometric_intensity),
            ScalarSlabEmissionModel::BlackbodyV1 {
                temperature_kelvin: source_temperature_kelvin,
            },
        )
    }

    /// Creates the zero-absorption limit with a path-integrated neutral emission term.
    ///
    /// # Errors
    ///
    /// Rejects a non-finite or negative integrated emission intensity.
    pub fn pure_emission_bolometric_v1(
        integrated_bolometric_emission: f64,
    ) -> Result<Self, ValidationReport> {
        Self::validated(
            0.0,
            EmissionInput::Integrated(integrated_bolometric_emission),
            ScalarSlabEmissionModel::NeutralBolometric,
        )
    }

    /// Creates the zero-absorption limit with a path-integrated blackbody emission term.
    ///
    /// # Errors
    ///
    /// Rejects invalid integrated emission and a non-finite or non-positive temperature.
    pub fn pure_emission_blackbody_v1(
        integrated_bolometric_emission: f64,
        emission_temperature_kelvin: f64,
    ) -> Result<Self, ValidationReport> {
        Self::validated(
            0.0,
            EmissionInput::Integrated(integrated_bolometric_emission),
            ScalarSlabEmissionModel::BlackbodyV1 {
                temperature_kelvin: emission_temperature_kelvin,
            },
        )
    }

    fn validated(
        optical_depth: f64,
        emission: EmissionInput,
        emission_model: ScalarSlabEmissionModel,
    ) -> Result<Self, ValidationReport> {
        let mut report = ValidationReport::default();
        validate_finite(
            &mut report,
            optical_depth,
            "homogeneous_scalar_slab.optical_depth",
        );
        let (emission_input, emission_field, emission_description) = match emission {
            EmissionInput::ConstantSource(value) => (
                value,
                "homogeneous_scalar_slab.source_bolometric_intensity",
                "source-function intensity must be non-negative",
            ),
            EmissionInput::Integrated(value) => (
                value,
                "homogeneous_scalar_slab.integrated_bolometric_emission",
                "integrated emission intensity must be non-negative",
            ),
        };
        validate_finite(&mut report, emission_input, emission_field);
        if optical_depth.is_finite() && optical_depth < 0.0 {
            report.push(ValidationIssue::error(
                ValidationIssueCode::OutOfRange,
                "homogeneous_scalar_slab.optical_depth",
                "optical depth must be non-negative",
            ));
        }
        if emission_input.is_finite() && emission_input < 0.0 {
            report.push(ValidationIssue::error(
                ValidationIssueCode::OutOfRange,
                emission_field,
                emission_description,
            ));
        }
        if let ScalarSlabEmissionModel::BlackbodyV1 { temperature_kelvin } = emission_model {
            validate_finite(
                &mut report,
                temperature_kelvin,
                "homogeneous_scalar_slab.emission_temperature_kelvin",
            );
            if temperature_kelvin.is_finite() && temperature_kelvin <= 0.0 {
                report.push(ValidationIssue::error(
                    ValidationIssueCode::NonPositive,
                    "homogeneous_scalar_slab.emission_temperature_kelvin",
                    "blackbody emission temperature must be positive",
                ));
            }
        }
        let integrated_bolometric_emission = match emission {
            EmissionInput::ConstantSource(source)
                if source.is_finite() && optical_depth.is_finite() =>
            {
                source * -(-optical_depth).exp_m1()
            }
            EmissionInput::Integrated(integrated) if integrated.is_finite() => integrated,
            // The original field already carries the validation error. This placeholder is never
            // observable because `ValidationReport::into_result` rejects the constructed value.
            EmissionInput::ConstantSource(_) | EmissionInput::Integrated(_) => 0.0,
        };
        if !integrated_bolometric_emission.is_finite() {
            report.push(ValidationIssue::error(
                ValidationIssueCode::OutOfRange,
                "homogeneous_scalar_slab.integrated_bolometric_emission",
                "derived integrated emission must be finite",
            ));
        }
        report.into_result(Self {
            optical_depth,
            integrated_bolometric_emission,
            emission_model,
        })
    }

    #[must_use]
    pub const fn optical_depth(self) -> f64 {
        self.optical_depth
    }

    #[must_use]
    pub const fn integrated_bolometric_emission(self) -> f64 {
        self.integrated_bolometric_emission
    }

    /// Returns the emission term's explicit spectral interpretation.
    #[must_use]
    pub const fn emission_model(self) -> ScalarSlabEmissionModel {
        self.emission_model
    }
}

#[derive(Clone, Copy)]
enum EmissionInput {
    ConstantSource(f64),
    Integrated(f64),
}

#[cfg(test)]
mod tests {
    use super::VISIBLE_BOXCAR_BANDS_V1;

    #[test]
    fn visible_boxcar_v1_preserves_its_versioned_channel_order() {
        assert_eq!(
            VISIBLE_BOXCAR_BANDS_V1.map(|band| (
                band.name(),
                band.lower_wavelength_nm(),
                band.upper_wavelength_nm(),
            )),
            [
                ("red", 600.0, 700.0),
                ("green", 500.0, 600.0),
                ("blue", 400.0, 500.0),
            ]
        );
    }
}

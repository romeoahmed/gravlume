//! Validated host-to-GPU trace ABI packing.

use gravlume_domain::{
    EquatorialEmissionModel, EquatorialSurface, Extremality, KerrNewmanSpacetime, KerrSchildChart,
    Observation, ScalarSlabEmissionModel, SceneRadiance, SurfaceTransport, ValidationReport,
};
use num_traits::ToPrimitive as _;

use crate::{
    scientific_capture::INVARIANT_RELATIVE_DRIFT_LIMIT,
    spectral_lut::{maximum_temperature_kelvin, minimum_temperature_kelvin},
};

const NORMALIZED_FREQUENCY_TOLERANCE: f64 = 32.0 * f64::EPSILON;
const OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_OVER_M: f32 = 200.0;
const GPU_EVENT_TIE_TOLERANCE_OVER_M: f32 = f32::from_bits(0x3700_0000);
const GPU_EVENT_ARMING_BAND_OVER_M: f32 = f32::from_bits(0x3980_0000);

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct TraceUniforms {
    spacetime: [f32; 4],
    observer: [f32; 4],
    observer_velocity: [f32; 4],
    image_right: [f32; 4],
    image_up: [f32; 4],
    arrival: [f32; 4],
    camera: [f32; 4],
    event_surfaces: [f32; 4],
    surface_emitter: [f32; 4],
    surface_transport: [f32; 4],
    step_policy: [f32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<TraceUniforms>() == 176);
    assert!(std::mem::align_of::<TraceUniforms>() == 16);
};

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub(super) struct TraceDispatch {
    pub(super) tile_origin: [u32; 2],
    pub(super) workgroup_count: [u32; 2],
}

const _: () = {
    assert!(std::mem::size_of::<TraceDispatch>() == 16);
    assert!(std::mem::align_of::<TraceDispatch>() == 16);
    assert!(std::mem::offset_of!(TraceDispatch, tile_origin) == 0);
    assert!(std::mem::offset_of!(TraceDispatch, workgroup_count) == 8);
};

impl TraceUniforms {
    pub fn from_observation(observation: &Observation) -> Result<Self, GpuTraceInputError> {
        let scene = observation.scene();
        let surface = match scene.radiance() {
            SceneRadiance::AnalyticSky => None,
            SceneRadiance::EquatorialSurface(surface) => Some(surface),
        };
        let physical_spacetime = *scene.spacetime();
        let mass = physical_spacetime.mass_m();
        let view = *observation.view();
        let frame = scene.observer_frame();
        let sample = view
            .sample(0, 0, 0.5, 0.5)
            .map_err(GpuTraceInputError::DomainInvariant)?;
        let observer_frequency = observation
            .initial_ray(sample)
            .map_err(GpuTraceInputError::DomainInvariant)?
            .observer_frequency();
        if (observer_frequency - 1.0).abs() > NORMALIZED_FREQUENCY_TOLERANCE {
            return Err(GpuTraceInputError::NonNormalizedObserverFrequency { observer_frequency });
        }
        let spacetime_uniform = pack4(
            [
                1.0,
                physical_spacetime.spin_m() / mass,
                physical_spacetime.charge_m() / mass,
                match physical_spacetime.chart() {
                    KerrSchildChart::Ingoing => 1.0,
                    KerrSchildChart::Outgoing => -1.0,
                },
            ],
            "spacetime",
        )?;
        let gpu_spacetime = KerrNewmanSpacetime::new(
            f64::from(spacetime_uniform[0]),
            f64::from(spacetime_uniform[1]),
            f64::from(spacetime_uniform[2]),
            physical_spacetime.chart(),
        )
        .map_err(GpuTraceInputError::DomainInvariant)?;
        let canonical_state = physical_spacetime.extremality();
        let gpu_extremality = gpu_spacetime.extremality();
        if gpu_extremality != canonical_state {
            return Err(GpuTraceInputError::ExtremalityChangedByPacking {
                canonical_state,
                gpu_extremality,
            });
        }
        let horizon = gpu_spacetime.outer_horizon_radius().unwrap_or(-1.0);
        let [_, observer_x, observer_y, observer_z] = scene.observer_event().to_txyz();
        let surface_emitter = pack_surface_emitter(surface, mass)?;
        let surface_transport = pack_surface_transport(surface, mass)?;

        Ok(Self {
            spacetime: spacetime_uniform,
            // A nonzero binary64 height can round to zero in binary32. Preserve the discrete side
            // before narrowing; the shader never consumes coordinate time from this lane.
            // Source: https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast
            observer: pack4(
                [
                    f64::from(encode_initial_polar_side(observer_z)),
                    observer_x / mass,
                    observer_y / mass,
                    observer_z / mass,
                ],
                "observer",
            )?,
            observer_velocity: pack4(frame.four_velocity_txyz(), "observer_velocity")?,
            image_right: pack4(frame.image_right_txyz(), "image_right")?,
            image_up: pack4(frame.image_up_txyz(), "image_up")?,
            arrival: pack4(frame.arrival_direction_txyz(), "arrival")?,
            camera: pack4(
                [(view.vertical_fov().radians() * 0.5).tan(), 1.0, 0.5, 0.5],
                "camera",
            )?,
            event_surfaces: pack4(
                [
                    f64::from(OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_OVER_M),
                    f64::from(f32::from_bits(0x2b80_0000)),
                    horizon,
                    f64::from(GPU_EVENT_TIE_TOLERANCE_OVER_M),
                ],
                "event_surfaces",
            )?,
            surface_emitter,
            surface_transport,
            step_policy: [0.1, 0.005, 8.0, INVARIANT_RELATIVE_DRIFT_LIMIT],
        })
    }

    #[cfg(test)]
    pub(super) fn set_capture_subpixel(
        &mut self,
        subpixel: [f64; 2],
        field: &'static str,
    ) -> Result<(), GpuTraceInputError> {
        let packed = pack4([subpixel[0], subpixel[1], 0.0, 0.0], field)?;
        self.camera[2..].copy_from_slice(&packed[..2]);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn set_packed_subpixel(&mut self, subpixel: [f32; 2]) {
        self.camera[2..].copy_from_slice(&subpixel);
    }

    #[cfg(test)]
    pub(super) fn use_footprint_step_policy(&mut self) {
        // Source-chart differencing subtracts neighboring terminal anchors. Keep this diagnostic
        // path below the presentation step ceiling so RK4 phase error does not dominate J.
        self.step_policy[..3].copy_from_slice(&[0.0025, 0.000_125, 0.25]);
    }
}

fn pack_surface_transport(
    surface: Option<EquatorialSurface>,
    mass_m: f64,
) -> Result<[f32; 4], GpuTraceInputError> {
    let Some(surface) = surface else {
        return Ok([0.0, 0.0, 1.0, 0.0]);
    };
    let emitter = surface.emitter();
    let emitter_temperature = match emitter.emission_model() {
        EquatorialEmissionModel::InverseCubeBolometricV1 => None,
        EquatorialEmissionModel::InverseCubeBlackbodyV1 {
            temperature_at_six_kelvin,
        } => Some(temperature_at_six_kelvin),
    };
    if let Some(temperature_at_six_kelvin) = emitter_temperature {
        for (radius_m, field) in [
            (
                emitter.inner_radius_m(),
                "equatorial_circular_emitter.inner_temperature_kelvin",
            ),
            (
                emitter.outer_radius_m(),
                "equatorial_circular_emitter.outer_temperature_kelvin",
            ),
        ] {
            let radius_ratio = radius_m / mass_m / 6.0;
            let temperature =
                temperature_at_six_kelvin / (radius_ratio * radius_ratio.sqrt()).sqrt();
            validate_lut_temperature(temperature, field)?;
        }
    }

    let (slab, transmittance, weighted_source_intensity, source_emission) =
        match surface.transport() {
            SurfaceTransport::Vacuum => (None, 1.0, 0.0, None),
            SurfaceTransport::HomogeneousScalar(slab) => {
                let transmittance = (-slab.optical_depth()).exp();
                (
                    Some(slab),
                    transmittance,
                    slab.integrated_bolometric_emission(),
                    Some(slab.emission_model()),
                )
            }
        };
    let source_temperature = if emitter_temperature.is_some() && weighted_source_intensity > 0.0 {
        let Some(ScalarSlabEmissionModel::BlackbodyV1 { temperature_kelvin }) = source_emission
        else {
            // `EquatorialSurface` construction rejects this combination. Keeping this assertion
            // local documents the trusted domain invariant without restoring a duplicate error.
            debug_assert!(
                false,
                "validated blackbody transport has a resolved source spectrum"
            );
            return Err(GpuTraceInputError::NotRepresentable {
                field: "surface_transport",
            });
        };
        validate_lut_temperature(
            temperature_kelvin,
            "homogeneous_scalar_slab.emission_temperature_kelvin",
        )?;
        temperature_kelvin
    } else {
        0.0
    };
    let packed = pack4(
        [
            emitter_temperature.unwrap_or(0.0),
            source_temperature,
            transmittance,
            weighted_source_intensity,
        ],
        "surface_transport",
    )?;
    let transmittance_underflowed = slab.is_some_and(|slab| {
        slab.optical_depth() > 0.0 && (transmittance == 0.0 || packed[2] == 0.0)
    });
    let source_underflowed = weighted_source_intensity > 0.0 && packed[3] == 0.0;
    if transmittance_underflowed
        || source_underflowed
        || packed.iter().any(|value| value.is_subnormal())
    {
        return Err(GpuTraceInputError::NotRepresentable {
            field: "surface_transport",
        });
    }
    Ok(packed)
}

fn validate_lut_temperature(
    temperature_kelvin: f64,
    field: &'static str,
) -> Result<(), GpuTraceInputError> {
    let minimum_kelvin = minimum_temperature_kelvin();
    let maximum_kelvin = maximum_temperature_kelvin();
    if (minimum_kelvin..=maximum_kelvin).contains(&temperature_kelvin) {
        Ok(())
    } else {
        Err(GpuTraceInputError::TemperatureOutsideSpectralLut {
            field,
            temperature_kelvin,
            minimum_kelvin,
            maximum_kelvin,
        })
    }
}

fn pack_surface_emitter(
    surface: Option<EquatorialSurface>,
    mass_m: f64,
) -> Result<[f32; 4], GpuTraceInputError> {
    let Some(surface) = surface else {
        return Ok([0.0; 4]);
    };
    let emitter = surface.emitter();
    let outer_radius_over_m = emitter.outer_radius_m() / mass_m;
    if outer_radius_over_m >= f64::from(OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_OVER_M) {
        return Err(GpuTraceInputError::SurfaceOutsideEscapeBoundary {
            outer_radius_over_m,
            escape_radius_over_m: f64::from(OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_OVER_M),
        });
    }
    let packed = pack4(
        [
            emitter.inner_radius_m() / mass_m,
            outer_radius_over_m,
            emitter.intensity_at_six_m(),
            f64::from(GPU_EVENT_ARMING_BAND_OVER_M),
        ],
        "surface_emitter",
    )?;
    // A binary64 value inside the boundary can round onto it when packed for the shader.
    // Source: https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast
    if packed[1] >= OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_OVER_M {
        return Err(GpuTraceInputError::SurfaceOutsideEscapeBoundary {
            outer_radius_over_m,
            escape_radius_over_m: f64::from(OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_OVER_M),
        });
    }
    let interval_collapsed = emitter.outer_radius_m() > emitter.inner_radius_m()
        && packed[1].to_bits() == packed[0].to_bits();
    let intensity_underflowed = emitter.intensity_at_six_m() > 0.0 && packed[2] == 0.0;
    if packed[0] <= 0.0
        || packed[1] < packed[0]
        || packed[2] < 0.0
        || packed[..3].iter().any(|value| value.is_subnormal())
        || interval_collapsed
        || intensity_underflowed
    {
        return Err(GpuTraceInputError::NotRepresentable {
            field: "surface_emitter",
        });
    }
    Ok(packed)
}

fn pack4(values: [f64; 4], field: &'static str) -> Result<[f32; 4], GpuTraceInputError> {
    let [Some(a), Some(b), Some(c), Some(d)] =
        values.map(|value| value.to_f32().filter(|packed| packed.is_finite()))
    else {
        return Err(GpuTraceInputError::NotRepresentable { field });
    };
    Ok([a, b, c, d])
}

const fn encode_initial_polar_side(height_m: f64) -> u32 {
    if height_m < 0.0 {
        0
    } else if height_m > 0.0 {
        2
    } else {
        1
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuTraceInputError {
    #[error("validated observation failed to resolve its initial ray: {0}")]
    DomainInvariant(#[source] ValidationReport),
    #[error(
        "GPU trace inputs must be normalized to observer frequency 1, got {observer_frequency}"
    )]
    NonNormalizedObserverFrequency { observer_frequency: f64 },
    #[error(
        "spacetime extremality changes from {canonical_state:?} to {gpu_extremality:?} under the GPU f32 contract"
    )]
    ExtremalityChangedByPacking {
        canonical_state: Extremality,
        gpu_extremality: Extremality,
    },
    #[error("observation field {field} is not representable by the GPU f32 contract")]
    NotRepresentable { field: &'static str },
    #[error(
        "surface outer radius {outer_radius_over_m} M must be strictly inside the GPU escape boundary {escape_radius_over_m} M"
    )]
    SurfaceOutsideEscapeBoundary {
        outer_radius_over_m: f64,
        escape_radius_over_m: f64,
    },
    #[error(
        "{field} temperature {temperature_kelvin} K lies outside the GPU spectral LUT [{minimum_kelvin}, {maximum_kelvin}] K"
    )]
    TemperatureOutsideSpectralLut {
        field: &'static str,
        temperature_kelvin: f64,
        minimum_kelvin: f64,
        maximum_kelvin: f64,
    },
    #[cfg(test)]
    #[error("surface footprint capture requires an equatorial surface source")]
    SurfaceFootprintRequiresSurface,
    #[cfg(test)]
    #[error("surface transport capture requires an equatorial surface source")]
    SurfaceTransportRequiresSurface,
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use gravlume_domain::{
        Angle, KerrSchildChart, Observation, PerspectiveView, PhysicalScene, PhysicalSceneInput,
        StationaryObserverInput,
    };

    use super::TraceUniforms;

    #[test]
    fn trace_uniforms_preserve_polar_side_before_binary32_packing() {
        for (observer_z, expected_side) in [(-1.0e-50, 0.0_f32), (0.0, 1.0_f32), (1.0e-50, 2.0_f32)]
        {
            let observer = StationaryObserverInput::new(
                [0.0, 30.0, 0.0, observer_z],
                [0.0; 4],
                [0.0, 0.0, 1.0],
                1.0,
            );
            let scene = PhysicalScene::new(PhysicalSceneInput::new(
                1.0,
                0.0,
                0.0,
                KerrSchildChart::Outgoing,
                observer,
            ))
            .expect("the generated observer is stationary");
            let view = PerspectiveView::new(
                NonZeroU32::MIN,
                NonZeroU32::MIN,
                Angle::from_radians(std::f64::consts::FRAC_PI_4)
                    .expect("the test field of view is finite"),
            )
            .expect("the test view is valid");
            let uniforms = TraceUniforms::from_observation(&Observation::new(scene, view))
                .expect("the observation is representable by the GPU contract");

            assert_eq!(uniforms.observer[0].to_bits(), expected_side.to_bits());
        }
    }
}

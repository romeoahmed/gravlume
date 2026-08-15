use std::sync::mpsc;

use gravlume_domain::{Observation, SpectralBand, VISIBLE_BOXCAR_BANDS_V1};

use crate::{error::GpuErrorScopes, extent::RenderExtent, ray_tracer::INVARIANT_DRIFT_LIMIT};

const RGBA16_FLOAT_BYTES_PER_PIXEL: u32 = 8;
const SPECTRAL_LUT_ABSOLUTE_FRACTION_ERROR_BUDGET: f64 = 3.0e-6;
const SPECTRAL_LUT_VISIBLE_RELATIVE_ERROR_BUDGET: f64 = 2.0e-3;
const SPECTRAL_LUT_RELATIVE_ERROR_FRACTION_FLOOR: f64 = 1.0e-6;
const BOLOMETRIC_SURFACE_RELATIVE_ERROR_BUDGET: f64 = 2.0e-3;
const SPECTRAL_SURFACE_RELATIVE_ERROR_BUDGET: f64 = 4.0e-3;
const HALF_POSITIVE_ZERO_BITS: u16 = 0x0000;
const HALF_NEGATIVE_ZERO_BITS: u16 = 0x8000;
const HALF_ANALYTIC_ESCAPE_TAG_BITS: u16 = 0x3c00;
const HALF_SURFACE_RADIANCE_TAG_BITS: u16 = 0x4000;

/// A direct, tone-mapping-free readback of one atomically published scene.
pub struct ScientificCapture {
    extent: [u32; 2],
    generation: u64,
    rgba16_float_bits: Vec<[u16; 4]>,
    metadata: ScientificCaptureMetadata,
}

impl ScientificCapture {
    #[must_use]
    pub const fn extent(&self) -> [u32; 2] {
        self.extent
    }

    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns IEEE-754 binary16 bit patterns in `R`, `G`, `B`, `A` memory order.
    ///
    /// WebGPU texel copies preserve the numeric value of finite, normal channels but may
    /// canonicalize zero and other exceptional representations. Use [`Self::pixel_kind`] before
    /// interpreting RGB: only [`ScientificPixelKind::SurfaceRadiance`] is physical source output.
    #[must_use]
    pub fn rgba16_float_bits(&self) -> &[[u16; 4]] {
        &self.rgba16_float_bits
    }

    /// Classifies one row-major texel by the renderer's non-display alpha tag.
    #[must_use]
    pub fn pixel_kind(&self, index: usize) -> Option<ScientificPixelKind> {
        self.rgba16_float_bits
            .get(index)
            .map(|texel| classify_alpha_bits(texel[3]))
    }

    #[must_use]
    pub const fn metadata(&self) -> &ScientificCaptureMetadata {
        &self.metadata
    }
}

/// Meaning of one scientific-capture texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScientificPixelKind {
    /// RGB contains the channel model declared by [`ScientificCaptureMetadata::channels`].
    SurfaceRadiance,
    /// RGB is the orientation-only analytic sky preview and is not spectral radiance.
    AnalyticEscapePreview,
    /// The ray crossed the horizon; RGB is zero.
    Horizon,
    /// The trace did not produce a physical terminal observable.
    TraceFailure { termination_code: u32 },
    /// The alpha word is outside the renderer's published tagging contract.
    Unclassified { alpha_bits: u16 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScientificCaptureMetadata {
    mass_m: f64,
    source_inner_radius_m: f64,
    source_outer_radius_m: f64,
    intensity_at_six_m: f64,
    emission_model: ScientificEmissionModel,
    transport: ScientificTransportMetadata,
    channels: ScientificChannelModel,
    numerical: ScientificNumericalMetadata,
}

impl ScientificCaptureMetadata {
    #[must_use]
    pub const fn mass_m(&self) -> f64 {
        self.mass_m
    }

    #[must_use]
    pub const fn source_radial_interval_m(&self) -> [f64; 2] {
        [self.source_inner_radius_m, self.source_outer_radius_m]
    }

    #[must_use]
    pub const fn intensity_at_six_m(&self) -> f64 {
        self.intensity_at_six_m
    }

    #[must_use]
    pub const fn emission_model(&self) -> ScientificEmissionModel {
        self.emission_model
    }

    #[must_use]
    pub const fn transport(&self) -> ScientificTransportMetadata {
        self.transport
    }

    #[must_use]
    pub const fn channels(&self) -> ScientificChannelModel {
        self.channels
    }

    #[must_use]
    pub const fn numerical(&self) -> ScientificNumericalMetadata {
        self.numerical
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScientificEmissionModel {
    InverseCubeBolometricV1,
    InverseCubeBlackbodyV1 { temperature_at_six_kelvin: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScientificChannelModel {
    /// The same model-normalized bolometric intensity is stored in R, G, and B.
    BolometricRepeated,
    /// R, G, and B are band-integrated radiances for the listed observer-frame boxcars.
    VisibleBoxcarV1 { bands: [SpectralBand; 3] },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScientificTransportMetadata {
    optical_depth: f64,
    integrated_bolometric_emission: f64,
    emission_temperature_kelvin: Option<f64>,
}

impl ScientificTransportMetadata {
    #[must_use]
    pub const fn optical_depth(self) -> f64 {
        self.optical_depth
    }

    #[must_use]
    pub const fn integrated_bolometric_emission(self) -> f64 {
        self.integrated_bolometric_emission
    }

    #[must_use]
    pub const fn emission_temperature_kelvin(self) -> Option<f64> {
        self.emission_temperature_kelvin
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScientificNumericalMetadata {
    invariant_relative_drift_limit: f32,
    bolometric_surface_relative_error_budget: f64,
    spectral_surface_relative_error_budget: Option<f64>,
    spectral_lut_absolute_fraction_error_budget: Option<f64>,
    spectral_lut_visible_relative_error_budget: Option<f64>,
    spectral_lut_relative_error_fraction_floor: Option<f64>,
}

impl ScientificNumericalMetadata {
    #[must_use]
    pub const fn invariant_relative_drift_limit(self) -> f32 {
        self.invariant_relative_drift_limit
    }

    #[must_use]
    pub const fn bolometric_surface_relative_error_budget(self) -> f64 {
        self.bolometric_surface_relative_error_budget
    }

    #[must_use]
    pub const fn spectral_surface_relative_error_budget(self) -> Option<f64> {
        self.spectral_surface_relative_error_budget
    }

    #[must_use]
    pub const fn spectral_lut_absolute_fraction_error_budget(self) -> Option<f64> {
        self.spectral_lut_absolute_fraction_error_budget
    }

    #[must_use]
    pub const fn spectral_lut_visible_relative_error_budget(self) -> Option<f64> {
        self.spectral_lut_visible_relative_error_budget
    }

    #[must_use]
    pub const fn spectral_lut_relative_error_fraction_floor(self) -> Option<f64> {
        self.spectral_lut_relative_error_fraction_floor
    }
}

pub fn metadata_for_observation(observation: &Observation) -> Option<ScientificCaptureMetadata> {
    let scene = observation.scene();
    let emitter = scene.equatorial_circular_emitter()?;
    let (emission_model, channels, spectral_budget) =
        emitter.blackbody_temperature_at_six_kelvin().map_or(
            (
                ScientificEmissionModel::InverseCubeBolometricV1,
                ScientificChannelModel::BolometricRepeated,
                false,
            ),
            |temperature_at_six_kelvin| {
                (
                    ScientificEmissionModel::InverseCubeBlackbodyV1 {
                        temperature_at_six_kelvin,
                    },
                    ScientificChannelModel::VisibleBoxcarV1 {
                        bands: VISIBLE_BOXCAR_BANDS_V1,
                    },
                    true,
                )
            },
        );
    let transport = scene.homogeneous_scalar_slab().map_or(
        ScientificTransportMetadata {
            optical_depth: 0.0,
            integrated_bolometric_emission: 0.0,
            emission_temperature_kelvin: None,
        },
        |slab| ScientificTransportMetadata {
            optical_depth: slab.optical_depth(),
            integrated_bolometric_emission: slab.integrated_bolometric_emission(),
            emission_temperature_kelvin: slab.emission_temperature_kelvin(),
        },
    );
    let optional_budget = |value| spectral_budget.then_some(value);
    Some(ScientificCaptureMetadata {
        mass_m: scene.spacetime().mass_m(),
        source_inner_radius_m: emitter.inner_radius_m(),
        source_outer_radius_m: emitter.outer_radius_m(),
        intensity_at_six_m: emitter.intensity_at_six_m(),
        emission_model,
        transport,
        channels,
        numerical: ScientificNumericalMetadata {
            invariant_relative_drift_limit: INVARIANT_DRIFT_LIMIT,
            bolometric_surface_relative_error_budget: BOLOMETRIC_SURFACE_RELATIVE_ERROR_BUDGET,
            spectral_surface_relative_error_budget: optional_budget(
                SPECTRAL_SURFACE_RELATIVE_ERROR_BUDGET,
            ),
            spectral_lut_absolute_fraction_error_budget: optional_budget(
                SPECTRAL_LUT_ABSOLUTE_FRACTION_ERROR_BUDGET,
            ),
            spectral_lut_visible_relative_error_budget: optional_budget(
                SPECTRAL_LUT_VISIBLE_RELATIVE_ERROR_BUDGET,
            ),
            spectral_lut_relative_error_fraction_floor: optional_budget(
                SPECTRAL_LUT_RELATIVE_ERROR_FRACTION_FLOOR,
            ),
        },
    })
}

pub fn capture_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    extent: RenderExtent,
    generation: u64,
    metadata: ScientificCaptureMetadata,
) -> Result<ScientificCapture, ScientificCaptureError> {
    let unpadded_bytes_per_row = extent
        .width()
        .checked_mul(RGBA16_FLOAT_BYTES_PER_PIXEL)
        .ok_or(ScientificCaptureError::LayoutOverflow)?;
    let padded_bytes_per_row = unpadded_bytes_per_row
        .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        .checked_mul(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        .ok_or(ScientificCaptureError::LayoutOverflow)?;
    let buffer_size = u64::from(padded_bytes_per_row)
        .checked_mul(u64::from(extent.height()))
        .ok_or(ScientificCaptureError::LayoutOverflow)?;
    let scopes = GpuErrorScopes::push(device);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scientific scene-linear capture readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("scientific scene-linear capture encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(extent.height()),
            },
        },
        wgpu::Extent3d {
            width: extent.width(),
            height: extent.height(),
            depth_or_array_layers: 1,
        },
    );
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        let _send_result = sender.send(result);
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;
    let map_result = receiver
        .recv()
        .map_err(|_| ScientificCaptureError::MapCallbackDropped)?;
    pollster::block_on(scopes.finish()).map_err(ScientificCaptureError::GpuResource)?;
    map_result?;

    let mapped = readback.get_mapped_range(..)?;
    let unpadded = usize::try_from(unpadded_bytes_per_row)
        .map_err(|_| ScientificCaptureError::LayoutOverflow)?;
    let padded = usize::try_from(padded_bytes_per_row)
        .map_err(|_| ScientificCaptureError::LayoutOverflow)?;
    let height =
        usize::try_from(extent.height()).map_err(|_| ScientificCaptureError::LayoutOverflow)?;
    let texel_capacity = usize::try_from(extent.width())
        .ok()
        .and_then(|width| width.checked_mul(height))
        .ok_or(ScientificCaptureError::LayoutOverflow)?;
    let mut rgba16_float_bits = Vec::with_capacity(texel_capacity);
    for row in mapped.chunks_exact(padded) {
        for texel in row[..unpadded].chunks_exact(RGBA16_FLOAT_BYTES_PER_PIXEL as usize) {
            rgba16_float_bits.push(std::array::from_fn(|channel| {
                let offset = channel * size_of::<u16>();
                u16::from_le_bytes([texel[offset], texel[offset + 1]])
            }));
        }
    }
    drop(mapped);
    readback.unmap();
    Ok(ScientificCapture {
        extent: [extent.width(), extent.height()],
        generation,
        rgba16_float_bits,
        metadata,
    })
}

const fn classify_alpha_bits(alpha_bits: u16) -> ScientificPixelKind {
    match alpha_bits {
        HALF_SURFACE_RADIANCE_TAG_BITS => ScientificPixelKind::SurfaceRadiance,
        HALF_ANALYTIC_ESCAPE_TAG_BITS => ScientificPixelKind::AnalyticEscapePreview,
        HALF_POSITIVE_ZERO_BITS | HALF_NEGATIVE_ZERO_BITS => ScientificPixelKind::Horizon,
        0xc200 => ScientificPixelKind::TraceFailure {
            termination_code: 3,
        },
        0xc400 => ScientificPixelKind::TraceFailure {
            termination_code: 4,
        },
        0xc500 => ScientificPixelKind::TraceFailure {
            termination_code: 5,
        },
        0xc600 => ScientificPixelKind::TraceFailure {
            termination_code: 6,
        },
        alpha_bits => ScientificPixelKind::Unclassified { alpha_bits },
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ScientificCaptureError {
    #[error("no complete scene generation has been published")]
    NoPublishedScene,
    #[error("the active observation has no physical surface radiance model to capture")]
    NoPhysicalSurfaceSource,
    #[error("scientific capture row or buffer layout overflowed")]
    LayoutOverflow,
    #[error("failed to allocate or copy the scientific capture: {0}")]
    GpuResource(#[source] wgpu::Error),
    #[error("scientific capture GPU polling failed: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("scientific capture buffer mapping failed: {0}")]
    Map(#[from] wgpu::BufferAsyncError),
    #[error("scientific capture mapped-range access failed: {0}")]
    BufferAccess(#[from] wgpu::MapRangeError),
    #[error("scientific capture map callback was dropped")]
    MapCallbackDropped,
}

#[cfg(test)]
mod tests {
    use gravlume_domain::{EquatorialCircularEmitter, HomogeneousScalarSlab, Observation};

    use super::{
        ScientificChannelModel, ScientificEmissionModel, ScientificPixelKind, capture_texture,
        classify_alpha_bits, metadata_for_observation,
    };
    use crate::{extent::RenderExtent, gpu_trace_tests::default_observation};

    #[test]
    fn analytic_sky_is_not_labeled_as_scientific_radiance() {
        assert!(metadata_for_observation(&default_observation(1, 1)).is_none());
    }

    #[test]
    fn blackbody_capture_metadata_closes_source_transport_channels_and_budgets() {
        let base = default_observation(1, 1);
        let emitter = EquatorialCircularEmitter::inverse_cube_blackbody_v1(6.0, 20.0, 1.0, 6_000.0)
            .expect("test blackbody source is valid");
        let slab = HomogeneousScalarSlab::constant_blackbody_v1(0.35, 0.05, 4_500.0)
            .expect("test slab is valid");
        let observation = Observation::new(
            base.scene()
                .clone()
                .with_equatorial_circular_emitter(emitter)
                .with_homogeneous_scalar_slab(slab),
            *base.view(),
        );

        let metadata = metadata_for_observation(&observation)
            .expect("physical surface source has capture metadata");

        assert_eq!(
            metadata.source_radial_interval_m().map(f64::to_bits),
            [6.0_f64.to_bits(), 20.0_f64.to_bits()]
        );
        assert_eq!(
            metadata.emission_model(),
            ScientificEmissionModel::InverseCubeBlackbodyV1 {
                temperature_at_six_kelvin: 6_000.0
            }
        );
        assert!(matches!(
            metadata.channels(),
            ScientificChannelModel::VisibleBoxcarV1 { .. }
        ));
        assert_eq!(
            metadata.transport().optical_depth().to_bits(),
            0.35_f64.to_bits()
        );
        assert_eq!(
            metadata
                .transport()
                .integrated_bolometric_emission()
                .to_bits(),
            slab.integrated_bolometric_emission().to_bits()
        );
        assert_eq!(
            metadata.transport().emission_temperature_kelvin(),
            Some(4_500.0)
        );
        let numerical = metadata.numerical();
        assert!(numerical.invariant_relative_drift_limit() > 0.0);
        assert!(numerical.bolometric_surface_relative_error_budget() > 0.0);
        assert!(
            numerical
                .spectral_surface_relative_error_budget()
                .is_some_and(|budget| budget > 0.0)
        );
        assert!(
            numerical
                .spectral_lut_absolute_fraction_error_budget()
                .is_some_and(|budget| budget > 0.0)
        );
        assert!(
            numerical
                .spectral_lut_visible_relative_error_budget()
                .is_some_and(|budget| budget > 0.0)
        );
        assert!(
            numerical
                .spectral_lut_relative_error_fraction_floor()
                .is_some_and(|floor| floor > 0.0)
        );
    }

    #[test]
    fn texture_readback_preserves_raw_scene_linear_half_bits() {
        let gpu = crate::test_device::native_gpu();
        let extent = RenderExtent::new(2, 1).expect("test extent is nonzero");
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scientific capture raw-bit fixture"),
            size: wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let bytes = [
            0x00, 0x3c, 0x00, 0x38, 0x00, 0x00, 0x00, 0x40, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44,
            0x00, 0xc5,
        ];
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let mut observation = default_observation(2, 1);
        let emitter =
            gravlume_domain::EquatorialCircularEmitter::inverse_cube_bolometric_v1(6.0, 20.0, 1.0)
                .expect("test surface is valid");
        observation = gravlume_domain::Observation::new(
            observation
                .scene()
                .clone()
                .with_equatorial_circular_emitter(emitter),
            *observation.view(),
        );
        let metadata = metadata_for_observation(&observation)
            .expect("surface observation has scientific metadata");

        let capture = capture_texture(&gpu.device, &gpu.queue, &texture, extent, 17, metadata)
            .expect("raw texture capture succeeds");

        assert_eq!(capture.extent(), [2, 1]);
        assert_eq!(capture.generation(), 17);
        assert_eq!(
            capture.rgba16_float_bits(),
            [
                [0x3c00, 0x3800, 0x0000, 0x4000],
                [0x4000, 0x4200, 0x4400, 0xc500]
            ]
        );
        assert_eq!(
            capture.pixel_kind(0),
            Some(ScientificPixelKind::SurfaceRadiance)
        );
        assert_eq!(
            capture.pixel_kind(1),
            Some(ScientificPixelKind::TraceFailure {
                termination_code: 5
            })
        );
        assert_eq!(capture.pixel_kind(2), None);
    }

    #[test]
    fn alpha_tags_keep_preview_pixels_out_of_the_surface_radiance_domain() {
        assert_eq!(
            classify_alpha_bits(0x3c00),
            ScientificPixelKind::AnalyticEscapePreview
        );
        assert_eq!(classify_alpha_bits(0), ScientificPixelKind::Horizon);
        assert_eq!(classify_alpha_bits(0x8000), ScientificPixelKind::Horizon);
        assert_eq!(
            classify_alpha_bits(0xc200),
            ScientificPixelKind::TraceFailure {
                termination_code: 3
            }
        );
        assert_eq!(
            classify_alpha_bits(0x3555),
            ScientificPixelKind::Unclassified { alpha_bits: 0x3555 }
        );
    }
}

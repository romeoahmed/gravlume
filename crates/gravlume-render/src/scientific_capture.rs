use std::sync::mpsc;

use gravlume_domain::EquatorialSurface;

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
    texels: Vec<ScientificTexel>,
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

    /// Returns row-major texels with their raw representation and semantic kind kept together.
    ///
    /// Only [`ScientificPixelKind::SurfaceRadiance`] carries physical source output.
    #[must_use]
    pub fn texels(&self) -> &[ScientificTexel] {
        &self.texels
    }

    #[must_use]
    pub const fn metadata(&self) -> &ScientificCaptureMetadata {
        &self.metadata
    }
}

/// One scientific-capture texel and the renderer protocol needed to interpret it safely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScientificTexel {
    rgba16_float_bits: [u16; 4],
}

impl ScientificTexel {
    const fn from_rgba16_float_bits(rgba16_float_bits: [u16; 4]) -> Self {
        Self { rgba16_float_bits }
    }

    /// Returns IEEE-754 binary16 bit patterns in `R`, `G`, `B`, `A` memory order.
    ///
    /// WebGPU texel copies preserve the numeric value of finite, normal channels but may
    /// canonicalize zero and other exceptional representations.
    #[must_use]
    pub const fn rgba16_float_bits(self) -> [u16; 4] {
        self.rgba16_float_bits
    }

    #[must_use]
    pub const fn kind(self) -> ScientificPixelKind {
        classify_alpha_bits(self.rgba16_float_bits[3])
    }

    /// Returns physical scene-linear RGB words only when the alpha protocol identifies radiance.
    #[must_use]
    pub const fn surface_radiance_rgb16_float_bits(self) -> Option<[u16; 3]> {
        match self.kind() {
            ScientificPixelKind::SurfaceRadiance => Some([
                self.rgba16_float_bits[0],
                self.rgba16_float_bits[1],
                self.rgba16_float_bits[2],
            ]),
            ScientificPixelKind::AnalyticEscapePreview
            | ScientificPixelKind::Horizon
            | ScientificPixelKind::TraceFailure { .. }
            | ScientificPixelKind::Unclassified { .. } => None,
        }
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
    surface: EquatorialSurface,
    channels: ScientificChannelModel,
    numerical: ScientificNumericalMetadata,
}

impl ScientificCaptureMetadata {
    pub(crate) fn for_surface(
        mass_m: f64,
        surface: EquatorialSurface,
        channels: ScientificChannelModel,
    ) -> Self {
        let spectral_budget = channels == ScientificChannelModel::VisibleBoxcarV1;
        let optional_budget = |value| spectral_budget.then_some(value);
        Self {
            mass_m,
            surface,
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
        }
    }

    #[must_use]
    pub const fn mass_m(&self) -> f64 {
        self.mass_m
    }

    #[must_use]
    pub const fn surface(&self) -> EquatorialSurface {
        self.surface
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScientificChannelModel {
    /// The same model-normalized bolometric intensity is stored in R, G, and B.
    BolometricRepeated,
    /// R, G, and B are band-integrated radiances for `VISIBLE_BOXCAR_BANDS_V1`.
    VisibleBoxcarV1,
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
    let (sender, receiver) = mpsc::sync_channel(1);
    // Bind the mapping request to the producing encoder so wgpu orders it after the copy.
    // Source: https://docs.rs/wgpu/30.0.0/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit
    encoder.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |result| {
        let _send_result = sender.send(result);
    });
    let submission = queue.submit([encoder.finish()]);
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
    let mut texels = Vec::with_capacity(texel_capacity);
    for row in mapped.chunks_exact(padded) {
        for texel in row[..unpadded].chunks_exact(RGBA16_FLOAT_BYTES_PER_PIXEL as usize) {
            let rgba16_float_bits = std::array::from_fn(|channel| {
                let offset = channel * size_of::<u16>();
                u16::from_le_bytes([texel[offset], texel[offset + 1]])
            });
            texels.push(ScientificTexel::from_rgba16_float_bits(rgba16_float_bits));
        }
    }
    drop(mapped);
    readback.unmap();
    Ok(ScientificCapture {
        extent: [extent.width(), extent.height()],
        generation,
        texels,
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
    use gravlume_domain::{
        EquatorialCircularEmitter, EquatorialSurface, HomogeneousScalarSlab, SurfaceTransport,
    };

    use super::{
        BOLOMETRIC_SURFACE_RELATIVE_ERROR_BUDGET, INVARIANT_DRIFT_LIMIT,
        SPECTRAL_LUT_ABSOLUTE_FRACTION_ERROR_BUDGET, SPECTRAL_LUT_RELATIVE_ERROR_FRACTION_FLOOR,
        SPECTRAL_LUT_VISIBLE_RELATIVE_ERROR_BUDGET, SPECTRAL_SURFACE_RELATIVE_ERROR_BUDGET,
        ScientificCaptureMetadata, ScientificChannelModel, ScientificNumericalMetadata,
        ScientificPixelKind, capture_texture, classify_alpha_bits,
    };
    use crate::extent::RenderExtent;

    #[test]
    fn capture_metadata_keeps_validated_surface_semantics_and_exact_budgets() {
        let bolometric = EquatorialCircularEmitter::inverse_cube_bolometric_v1(6.0, 20.0, 1.0)
            .expect("test bolometric source is valid");
        let bolometric_surface = validated_surface(bolometric, SurfaceTransport::Vacuum);
        let metadata = ScientificCaptureMetadata::for_surface(
            1.0,
            bolometric_surface,
            ScientificChannelModel::BolometricRepeated,
        );

        assert_f64_contract(metadata.mass_m(), 1.0);
        assert_eq!(metadata.surface(), bolometric_surface);
        assert_eq!(
            metadata.channels(),
            ScientificChannelModel::BolometricRepeated
        );
        assert_numerical_contract(metadata.numerical(), false);

        let emitter = EquatorialCircularEmitter::inverse_cube_blackbody_v1(6.0, 20.0, 1.0, 6_000.0)
            .expect("test blackbody source is valid");
        let slab = HomogeneousScalarSlab::constant_blackbody_v1(0.35, 0.05, 4_500.0)
            .expect("test slab is valid");
        let surface = validated_surface(emitter, SurfaceTransport::HomogeneousScalar(slab));
        let metadata = ScientificCaptureMetadata::for_surface(
            1.0,
            surface,
            ScientificChannelModel::VisibleBoxcarV1,
        );

        assert_eq!(metadata.surface(), surface);
        assert_eq!(metadata.channels(), ScientificChannelModel::VisibleBoxcarV1);
        assert_numerical_contract(metadata.numerical(), true);
    }

    #[test]
    fn texture_readback_unpads_rows_and_preserves_normal_scene_linear_half_words() {
        let gpu = crate::test_device::native_gpu();
        let extent = RenderExtent::new(2, 2).expect("test extent is nonzero");
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scientific capture raw-bit fixture"),
            size: wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let first_row = [
            0x00, 0x3c, 0x00, 0x38, 0x00, 0x00, 0x00, 0x40, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44,
            0x00, 0xc5,
        ];
        let second_row = [
            0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        let mut bytes = vec![0_u8; 2 * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize];
        bytes[..first_row.len()].copy_from_slice(&first_row);
        let second_row_start = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        bytes[second_row_start..second_row_start + second_row.len()].copy_from_slice(&second_row);
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
                bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                rows_per_image: Some(2),
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
        let emitter =
            gravlume_domain::EquatorialCircularEmitter::inverse_cube_bolometric_v1(6.0, 20.0, 1.0)
                .expect("test surface is valid");
        let metadata = ScientificCaptureMetadata::for_surface(
            1.0,
            validated_surface(emitter, SurfaceTransport::Vacuum),
            ScientificChannelModel::BolometricRepeated,
        );

        let capture = capture_texture(&gpu.device, &gpu.queue, &texture, extent, 17, metadata)
            .expect("raw texture capture succeeds");

        assert_eq!(capture.extent(), [2, 2]);
        assert_eq!(capture.generation(), 17);
        let words = capture
            .texels()
            .iter()
            .copied()
            .map(super::ScientificTexel::rgba16_float_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            words,
            [
                [0x3c00, 0x3800, 0x0000, 0x4000],
                [0x4000, 0x4200, 0x4400, 0xc500],
                [0x3c00, 0x0000, 0x0000, 0x3c00],
                [0x0000, 0x0000, 0x0000, 0x0000],
            ]
        );
        assert_eq!(
            capture.texels()[0].kind(),
            ScientificPixelKind::SurfaceRadiance
        );
        assert_eq!(
            capture.texels()[0].surface_radiance_rgb16_float_bits(),
            Some([0x3c00, 0x3800, 0x0000])
        );
        assert_eq!(
            capture.texels()[1].kind(),
            ScientificPixelKind::TraceFailure {
                termination_code: 5
            }
        );
        assert_eq!(
            capture.texels()[2].kind(),
            ScientificPixelKind::AnalyticEscapePreview
        );
        assert_eq!(capture.texels()[3].kind(), ScientificPixelKind::Horizon);
        assert_eq!(
            capture.texels()[2].surface_radiance_rgb16_float_bits(),
            None
        );
    }

    #[test]
    fn alpha_tag_protocol_classifies_every_published_texel_kind() {
        let cases = [
            (0x4000, ScientificPixelKind::SurfaceRadiance),
            (0x3c00, ScientificPixelKind::AnalyticEscapePreview),
            (0x0000, ScientificPixelKind::Horizon),
            (0x8000, ScientificPixelKind::Horizon),
            (
                0xc200,
                ScientificPixelKind::TraceFailure {
                    termination_code: 3,
                },
            ),
            (
                0xc400,
                ScientificPixelKind::TraceFailure {
                    termination_code: 4,
                },
            ),
            (
                0xc500,
                ScientificPixelKind::TraceFailure {
                    termination_code: 5,
                },
            ),
            (
                0xc600,
                ScientificPixelKind::TraceFailure {
                    termination_code: 6,
                },
            ),
            (
                0x3555,
                ScientificPixelKind::Unclassified { alpha_bits: 0x3555 },
            ),
        ];

        for (alpha_bits, expected) in cases {
            assert_eq!(classify_alpha_bits(alpha_bits), expected);
        }
    }

    fn assert_numerical_contract(metadata: ScientificNumericalMetadata, has_spectrum: bool) {
        let spectral = |value| has_spectrum.then_some(value);

        assert_f64_contract(
            f64::from(metadata.invariant_relative_drift_limit()),
            f64::from(INVARIANT_DRIFT_LIMIT),
        );
        assert_f64_contract(
            metadata.bolometric_surface_relative_error_budget(),
            BOLOMETRIC_SURFACE_RELATIVE_ERROR_BUDGET,
        );
        assert_eq!(
            metadata.spectral_surface_relative_error_budget(),
            spectral(SPECTRAL_SURFACE_RELATIVE_ERROR_BUDGET)
        );
        assert_eq!(
            metadata.spectral_lut_absolute_fraction_error_budget(),
            spectral(SPECTRAL_LUT_ABSOLUTE_FRACTION_ERROR_BUDGET)
        );
        assert_eq!(
            metadata.spectral_lut_visible_relative_error_budget(),
            spectral(SPECTRAL_LUT_VISIBLE_RELATIVE_ERROR_BUDGET)
        );
        assert_eq!(
            metadata.spectral_lut_relative_error_fraction_floor(),
            spectral(SPECTRAL_LUT_RELATIVE_ERROR_FRACTION_FLOOR)
        );
    }

    fn assert_f64_contract(actual: f64, expected: f64) {
        let tolerance = f64::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual:e} differs from {expected:e} by more than {tolerance:e}"
        );
    }

    fn validated_surface(
        emitter: EquatorialCircularEmitter,
        transport: SurfaceTransport,
    ) -> EquatorialSurface {
        EquatorialSurface::new(emitter, transport)
            .expect("test surface and transport are compatible")
    }
}

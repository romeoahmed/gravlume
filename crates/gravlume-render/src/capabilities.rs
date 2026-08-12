pub const BASELINE_FEATURES: wgpu::Features = wgpu::Features::TIMESTAMP_QUERY;

/// Resolves the exact limits consumed by the native renderer.
///
/// Buffer limits remain at the WebGPU baseline because every trace plane fits within that project
/// contract. Requesting adapter maxima would turn hardware capability into an allocation policy.
/// Source: <https://docs.rs/wgpu/30.0.0/wgpu/struct.Limits.html#method.using_resolution>
pub fn required_device_limits(adapter: wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits::default()
        .using_resolution(adapter.clone())
        .using_alignment(adapter)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CapabilityError {
    #[error("software adapters are outside the native desktop baseline")]
    SoftwareAdapter,
    #[error("adapter is not WebGPU compliant")]
    DownlevelAdapter,
    #[error("adapter is missing required renderer features: {0:?}")]
    MissingFeatures(wgpu::Features),
    #[error("rgba16float is missing required usages: {0:?}")]
    MissingHdrTextureUsages(wgpu::TextureUsages),
    #[error("surface has no SDR sRGB format/color-space pair")]
    NoSdrSurfacePair,
    #[error("surface exposes no presentation mode")]
    NoPresentMode,
    #[error("surface exposes no composite alpha mode")]
    NoAlphaMode,
    #[error("surface cannot be used as a render attachment")]
    MissingRenderAttachment,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSelection {
    format: wgpu::TextureFormat,
    color_space: wgpu::SurfaceColorSpace,
    present_mode: wgpu::PresentMode,
    alpha_mode: wgpu::CompositeAlphaMode,
    output: OutputContract,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputContract {
    mode: OutputMode,
    tone_map_headroom: f32,
    reference_white_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    Hdr,
    Sdr(SdrReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdrReason {
    SystemSuppressed,
    HdrSurfacePairMissing,
    DisplayReportedSdr,
    PlatformIntegrationUnavailable,
    UnsupportedOsVersion,
    DisplayStateQueryFailed,
    WaylandColorManagementUnavailable,
    WaylandProtocolTooOld,
    WaylandEncodingUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownDisplayState {
    PlatformIntegrationUnavailable,
    UnsupportedOsVersion,
    StateQueryFailed,
    WaylandColorManagementUnavailable,
    WaylandProtocolTooOld,
    WaylandEncodingUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrParameters {
    tone_map_headroom: f32,
    reference_white_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DisplayState {
    Hdr(HdrParameters),
    Sdr,
    Suppressed,
    Unknown(UnknownDisplayState),
}

impl HdrParameters {
    #[must_use]
    pub fn new(tone_map_headroom: f32, reference_white_scale: f32) -> Option<Self> {
        (tone_map_headroom.is_finite()
            && tone_map_headroom >= 1.0
            && reference_white_scale.is_finite()
            && reference_white_scale > 0.0)
            .then_some(Self {
                tone_map_headroom,
                reference_white_scale,
            })
    }
}

impl SurfaceSelection {
    pub(crate) const fn format(self) -> wgpu::TextureFormat {
        self.format
    }

    pub(crate) const fn color_space(self) -> wgpu::SurfaceColorSpace {
        self.color_space
    }

    pub(crate) const fn present_mode(self) -> wgpu::PresentMode {
        self.present_mode
    }

    pub(crate) const fn alpha_mode(self) -> wgpu::CompositeAlphaMode {
        self.alpha_mode
    }

    pub(crate) fn requires_manual_srgb_encoding(self) -> bool {
        !self.format.is_srgb()
    }

    pub(crate) const fn output_mode(self) -> OutputMode {
        self.output.mode
    }

    pub(crate) const fn tone_map_headroom(self) -> f32 {
        self.output.tone_map_headroom
    }

    pub(crate) const fn reference_white_scale(self) -> f32 {
        self.output.reference_white_scale
    }

    pub(crate) fn fragment_entry(self) -> &'static str {
        match self.output.mode {
            OutputMode::Hdr => "present_hdr_extended_linear",
            OutputMode::Sdr(_) if self.format.is_srgb() => "present_sdr_to_linear_target",
            OutputMode::Sdr(_) => "present_sdr_to_gamma_target",
        }
    }
}

pub fn check_baseline_adapter(
    device_type: wgpu::DeviceType,
    is_webgpu_compliant: bool,
    available_features: wgpu::Features,
    hdr_allowed_usages: wgpu::TextureUsages,
) -> Result<(), CapabilityError> {
    if device_type == wgpu::DeviceType::Cpu {
        return Err(CapabilityError::SoftwareAdapter);
    }
    if !is_webgpu_compliant {
        return Err(CapabilityError::DownlevelAdapter);
    }
    let missing_features = BASELINE_FEATURES.difference(available_features);
    if !missing_features.is_empty() {
        return Err(CapabilityError::MissingFeatures(missing_features));
    }

    let required_hdr_usages = wgpu::TextureUsages::STORAGE_BINDING
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC
        | wgpu::TextureUsages::RENDER_ATTACHMENT;
    let missing_usages = required_hdr_usages.difference(hdr_allowed_usages);
    if !missing_usages.is_empty() {
        return Err(CapabilityError::MissingHdrTextureUsages(missing_usages));
    }
    Ok(())
}

pub fn select_surface(
    capabilities: &wgpu::SurfaceCapabilities,
    display: DisplayState,
) -> Result<SurfaceSelection, CapabilityError> {
    if !capabilities
        .usages
        .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
    {
        return Err(CapabilityError::MissingRenderAttachment);
    }

    let present_mode = capabilities
        .present_modes
        .iter()
        .copied()
        .find(|mode| *mode == wgpu::PresentMode::Fifo)
        .or_else(|| capabilities.present_modes.first().copied())
        .ok_or(CapabilityError::NoPresentMode)?;
    let alpha_mode = capabilities
        .alpha_modes
        .iter()
        .copied()
        .find(|mode| *mode == wgpu::CompositeAlphaMode::Opaque)
        .or_else(|| capabilities.alpha_modes.first().copied())
        .ok_or(CapabilityError::NoAlphaMode)?;

    let hdr_pair = capabilities.format_capabilities.iter().find(|candidate| {
        candidate.format == wgpu::TextureFormat::Rgba16Float
            && candidate
                .color_spaces
                .contains(wgpu::SurfaceColorSpaces::EXTENDED_SRGB_LINEAR)
    });
    if let (DisplayState::Hdr(parameters), Some(pair)) = (display, hdr_pair) {
        return Ok(SurfaceSelection {
            format: pair.format,
            color_space: wgpu::SurfaceColorSpace::ExtendedSrgbLinear,
            present_mode,
            alpha_mode,
            output: OutputContract {
                mode: OutputMode::Hdr,
                tone_map_headroom: parameters.tone_map_headroom,
                reference_white_scale: parameters.reference_white_scale,
            },
        });
    }
    let sdr_reason = match display {
        DisplayState::Suppressed => SdrReason::SystemSuppressed,
        DisplayState::Sdr => SdrReason::DisplayReportedSdr,
        DisplayState::Hdr(_) => SdrReason::HdrSurfacePairMissing,
        DisplayState::Unknown(reason) => match reason {
            UnknownDisplayState::PlatformIntegrationUnavailable => {
                SdrReason::PlatformIntegrationUnavailable
            }
            UnknownDisplayState::UnsupportedOsVersion => SdrReason::UnsupportedOsVersion,
            UnknownDisplayState::StateQueryFailed => SdrReason::DisplayStateQueryFailed,
            UnknownDisplayState::WaylandColorManagementUnavailable => {
                SdrReason::WaylandColorManagementUnavailable
            }
            UnknownDisplayState::WaylandProtocolTooOld => SdrReason::WaylandProtocolTooOld,
            UnknownDisplayState::WaylandEncodingUnavailable => {
                SdrReason::WaylandEncodingUnavailable
            }
        },
    };

    let supports_srgb = |candidate: &&wgpu::SurfaceFormatCapabilities| {
        candidate
            .color_spaces
            .contains(wgpu::SurfaceColorSpaces::SRGB)
    };
    let format = capabilities
        .format_capabilities
        .iter()
        .filter(supports_srgb)
        .find(|candidate| candidate.format.is_srgb())
        .or_else(|| {
            capabilities
                .format_capabilities
                .iter()
                .filter(supports_srgb)
                .find(|candidate| {
                    matches!(
                        candidate.format,
                        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                    )
                })
        })
        .ok_or(CapabilityError::NoSdrSurfacePair)?
        .format;
    Ok(SurfaceSelection {
        format,
        color_space: wgpu::SurfaceColorSpace::Srgb,
        present_mode,
        alpha_mode,
        output: OutputContract {
            mode: OutputMode::Sdr(sdr_reason),
            tone_map_headroom: 1.0,
            reference_white_scale: 1.0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::{
        BASELINE_FEATURES, CapabilityError, DisplayState, HdrParameters, OutputMode, SdrReason,
        UnknownDisplayState, check_baseline_adapter, required_device_limits, select_surface,
    };

    #[test]
    fn device_limits_do_not_copy_adapter_buffer_capacity() {
        let adapter = wgpu::Limits {
            max_texture_dimension_2d: 16_384,
            min_storage_buffer_offset_alignment: 64,
            max_storage_buffer_binding_size: 1 << 30,
            max_buffer_size: 2 << 30,
            ..wgpu::Limits::default()
        };

        let required = required_device_limits(adapter);

        assert_eq!(required.max_texture_dimension_2d, 16_384);
        assert_eq!(required.min_storage_buffer_offset_alignment, 64);
        assert_eq!(
            required.max_storage_buffer_binding_size,
            wgpu::Limits::default().max_storage_buffer_binding_size
        );
        assert_eq!(
            required.max_buffer_size,
            wgpu::Limits::default().max_buffer_size
        );
    }

    fn capabilities(
        formats: &[(wgpu::TextureFormat, wgpu::SurfaceColorSpaces)],
    ) -> wgpu::SurfaceCapabilities {
        wgpu::SurfaceCapabilities {
            formats: formats.iter().map(|(format, _)| *format).collect(),
            format_capabilities: formats
                .iter()
                .map(|(format, color_spaces)| wgpu::SurfaceFormatCapabilities {
                    format: *format,
                    color_spaces: *color_spaces,
                })
                .collect(),
            present_modes: vec![wgpu::PresentMode::Fifo, wgpu::PresentMode::Immediate],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::PreMultiplied,
                wgpu::CompositeAlphaMode::Opaque,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    #[test]
    fn output_resolver_selects_hdr_or_preserves_the_sdr_fallback_reason() {
        let caps = capabilities(&[
            (
                wgpu::TextureFormat::Rgba16Float,
                wgpu::SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
            ),
            (
                wgpu::TextureFormat::Bgra8UnormSrgb,
                wgpu::SurfaceColorSpaces::SRGB,
            ),
        ]);

        let selected = select_surface(
            &caps,
            DisplayState::Hdr(HdrParameters::new(4.0, 2.5).expect("valid HDR parameters")),
        )
        .expect("an HDR pair is available");

        assert_eq!(selected.format(), wgpu::TextureFormat::Rgba16Float);
        assert_eq!(
            selected.color_space(),
            wgpu::SurfaceColorSpace::ExtendedSrgbLinear
        );
        assert_eq!(selected.output_mode(), OutputMode::Hdr);
        assert_eq!(selected.present_mode(), wgpu::PresentMode::Fifo);
        assert_eq!(selected.alpha_mode(), wgpu::CompositeAlphaMode::Opaque);
        assert!((selected.tone_map_headroom() - 4.0).abs() <= f32::EPSILON);
        assert!((selected.reference_white_scale() - 2.5).abs() <= f32::EPSILON);
        assert!(HdrParameters::new(f32::NAN, 1.0).is_none());
        assert!(HdrParameters::new(2.0, 0.0).is_none());
        let fallback =
            select_surface(&caps, DisplayState::Suppressed).expect("an SDR fallback is available");
        assert_eq!(fallback.format(), wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(
            fallback.output_mode(),
            OutputMode::Sdr(SdrReason::SystemSuppressed)
        );
        let unknown = select_surface(
            &caps,
            DisplayState::Unknown(UnknownDisplayState::StateQueryFailed),
        )
        .expect("unknown display state has a color-correct SDR fallback");
        assert_eq!(
            unknown.output_mode(),
            OutputMode::Sdr(SdrReason::DisplayStateQueryFailed)
        );

        let manual_caps = capabilities(&[(
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::SurfaceColorSpaces::SRGB,
        )]);
        let manual = select_surface(&manual_caps, DisplayState::Sdr)
            .expect("a plain eight-bit SDR pair can use shader encoding");
        assert!(manual.requires_manual_srgb_encoding());
    }

    #[test]
    fn surface_selection_rejects_a_surface_without_srgb_pair() {
        let caps = capabilities(&[(
            wgpu::TextureFormat::Rgba16Float,
            wgpu::SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
        )]);

        assert_eq!(
            select_surface(
                &caps,
                DisplayState::Hdr(HdrParameters::new(1.0, 1.0).expect("valid HDR parameters")),
            ),
            Err(CapabilityError::NoSdrSurfacePair)
        );
    }

    #[test]
    fn adapter_gate_enforces_the_native_release_contract() {
        let required_hdr_usages = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT;
        let cases = [
            (
                "software adapter",
                wgpu::DeviceType::Cpu,
                true,
                BASELINE_FEATURES,
                required_hdr_usages,
                Err(CapabilityError::SoftwareAdapter),
            ),
            (
                "downlevel adapter",
                wgpu::DeviceType::IntegratedGpu,
                false,
                BASELINE_FEATURES,
                required_hdr_usages,
                Err(CapabilityError::DownlevelAdapter),
            ),
            (
                "missing timestamp queries",
                wgpu::DeviceType::IntegratedGpu,
                true,
                wgpu::Features::empty(),
                required_hdr_usages,
                Err(CapabilityError::MissingFeatures(
                    wgpu::Features::TIMESTAMP_QUERY,
                )),
            ),
            (
                "incomplete HDR usages",
                wgpu::DeviceType::DiscreteGpu,
                true,
                BASELINE_FEATURES,
                wgpu::TextureUsages::TEXTURE_BINDING,
                Err(CapabilityError::MissingHdrTextureUsages(
                    wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                )),
            ),
            (
                "supported adapter",
                wgpu::DeviceType::IntegratedGpu,
                true,
                BASELINE_FEATURES,
                required_hdr_usages,
                Ok(()),
            ),
        ];

        for (case, device_type, compliant, features, usages, expected) in cases {
            assert_eq!(
                check_baseline_adapter(device_type, compliant, features, usages),
                expected,
                "{case}"
            );
        }
    }
}

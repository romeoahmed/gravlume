pub(crate) const BASELINE_FEATURES: wgpu::Features =
    wgpu::Features::TIMESTAMP_QUERY.union(wgpu::Features::CLEAR_TEXTURE);

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum CapabilityError {
    #[error("software adapters are outside the native desktop baseline")]
    SoftwareAdapter,
    #[error("adapter is not WebGPU compliant")]
    DownlevelAdapter,
    #[error("adapter is missing Phase 0 features: {0:?}")]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SurfaceSelection {
    format: wgpu::TextureFormat,
    color_space: wgpu::SurfaceColorSpace,
    present_mode: wgpu::PresentMode,
    alpha_mode: wgpu::CompositeAlphaMode,
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
}

pub(crate) const fn missing_baseline_features(available: wgpu::Features) -> wgpu::Features {
    BASELINE_FEATURES.difference(available)
}

pub(crate) fn check_baseline_adapter(
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
    let missing_features = missing_baseline_features(available_features);
    if !missing_features.is_empty() {
        return Err(CapabilityError::MissingFeatures(missing_features));
    }

    let required_hdr_usages = wgpu::TextureUsages::STORAGE_BINDING
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC;
    let missing_usages = required_hdr_usages.difference(hdr_allowed_usages);
    if !missing_usages.is_empty() {
        return Err(CapabilityError::MissingHdrTextureUsages(missing_usages));
    }
    Ok(())
}

pub(crate) fn select_surface(
    capabilities: &wgpu::SurfaceCapabilities,
) -> Result<SurfaceSelection, CapabilityError> {
    if !capabilities
        .usages
        .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
    {
        return Err(CapabilityError::MissingRenderAttachment);
    }

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

    Ok(SurfaceSelection {
        format,
        color_space: wgpu::SurfaceColorSpace::Srgb,
        present_mode,
        alpha_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        BASELINE_FEATURES, CapabilityError, check_baseline_adapter, missing_baseline_features,
        select_surface,
    };

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
    fn surface_selection_prefers_srgb_format_and_opaque_alpha() {
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

        let selected = select_surface(&caps).expect("an SDR pair is available");

        assert_eq!(selected.format(), wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(selected.color_space(), wgpu::SurfaceColorSpace::Srgb);
        assert_eq!(selected.present_mode(), wgpu::PresentMode::Fifo);
        assert_eq!(selected.alpha_mode(), wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn surface_selection_accepts_gamma_space_sdr_when_srgb_format_is_absent() {
        let caps = capabilities(&[(
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::SurfaceColorSpaces::SRGB,
        )]);

        let selected = select_surface(&caps).expect("an explicit SDR pair is available");

        assert_eq!(selected.format(), wgpu::TextureFormat::Bgra8Unorm);
        assert!(selected.requires_manual_srgb_encoding());
    }

    #[test]
    fn surface_selection_rejects_a_surface_without_srgb_pair() {
        let caps = capabilities(&[(
            wgpu::TextureFormat::Rgba16Float,
            wgpu::SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
        )]);

        assert_eq!(
            select_surface(&caps),
            Err(CapabilityError::NoSdrSurfacePair)
        );
    }

    #[test]
    fn baseline_features_are_exact_and_missing_set_is_structured() {
        assert_eq!(
            BASELINE_FEATURES,
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::CLEAR_TEXTURE
        );
        assert_eq!(
            missing_baseline_features(wgpu::Features::TIMESTAMP_QUERY),
            wgpu::Features::CLEAR_TEXTURE
        );
        assert!(missing_baseline_features(BASELINE_FEATURES).is_empty());
    }

    #[test]
    fn adapter_gate_rejects_software_downlevel_and_incomplete_hdr_usage() {
        let hdr_usages = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;

        assert_eq!(
            check_baseline_adapter(wgpu::DeviceType::Cpu, true, BASELINE_FEATURES, hdr_usages,),
            Err(CapabilityError::SoftwareAdapter)
        );
        assert_eq!(
            check_baseline_adapter(
                wgpu::DeviceType::IntegratedGpu,
                false,
                BASELINE_FEATURES,
                hdr_usages,
            ),
            Err(CapabilityError::DownlevelAdapter)
        );
        assert_eq!(
            check_baseline_adapter(
                wgpu::DeviceType::DiscreteGpu,
                true,
                BASELINE_FEATURES,
                wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            Err(CapabilityError::MissingHdrTextureUsages(
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC
            ))
        );
    }

    #[test]
    fn adapter_gate_accepts_exact_phase_zero_capabilities() {
        assert_eq!(
            check_baseline_adapter(
                wgpu::DeviceType::IntegratedGpu,
                true,
                BASELINE_FEATURES,
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            ),
            Ok(())
        );
    }
}

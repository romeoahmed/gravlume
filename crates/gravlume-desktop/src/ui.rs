use gravlume_render::{DeviceEvent, RendererDiagnostics};

use crate::preview::Preview;

const FONT_NAME: &str = "Noto Sans SC";
const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/NotoSansSC-Regular.otf");

pub fn install_fonts(context: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        FONT_NAME.to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(FONT_BYTES)),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push(FONT_NAME.to_owned());
    }
    context.set_fonts(fonts);
}

pub fn show_overlay(
    context: &egui::Context,
    diagnostics: &RendererDiagnostics<'_>,
    preview: Preview,
    device_event: Option<&DeviceEvent>,
    resize_event: Option<&DeviceEvent>,
) {
    let style = context.style_of(context.theme());
    let panel_frame =
        egui::Frame::window(&style).fill(egui::Color32::from_rgba_unmultiplied(12, 14, 18, 228));
    let title_frame =
        egui::Frame::window(&style).fill(egui::Color32::from_rgba_unmultiplied(20, 22, 26, 236));
    egui::Window::new("Gravlume")
        .default_pos([16.0, 16.0])
        .default_width(380.0)
        .resizable(false)
        .collapsible(false)
        .frame(panel_frame)
        .title_frame(title_frame)
        .show(context, |ui| {
            ui.spacing_mut().item_spacing.y = 3.0;
            ui.strong("Kerr black-hole lensing");
            match diagnostics.trace_completion() {
                Some(completion) if completion < 1.0 => {
                    ui.label("Tracing a complete full-resolution view.");
                    ui.weak("Previous complete frame remains visible.");
                }
                _ => {
                    ui.label("Full-resolution trace complete.");
                }
            }
            ui.label(format!(
                "a/M {} | r_obs/M {} | vertical FOV {} deg",
                preview.spin_ratio(),
                preview.observer_radius_ratio(),
                preview.vertical_fov_degrees()
            ));
            ui.label("Black: horizon shadow | Color: lensed sky");
            ui.label(format!(
                "GPU Kerr geodesics | Output: {}",
                diagnostics.display_transfer()
            ));
            ui.weak("No accretion disk or radiative transfer.");
            if device_event.is_some() || resize_event.is_some() {
                ui.separator();
            }
            if let Some(event) = device_event {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 170, 80),
                    format!("GPU {:?}: {}", event.kind(), event.message()),
                );
            }
            if let Some(event) = resize_event {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 170, 80),
                    format!("Resize {:?}: {}", event.kind(), event.message()),
                );
            }
        });
}

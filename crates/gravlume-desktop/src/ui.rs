use std::num::NonZeroU32;

use gravlume_domain::{
    Angle, KerrNewmanSpacetime, KerrSchildChart, Observation, PerspectiveView, PhysicalScene,
    PhysicalSceneInput, StationaryObserverInput, ValidationReport,
};
use gravlume_render::{DeviceEvent, RendererDiagnostics};

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
            ui.label("a/M 0.8 | r_obs/M 30 | vertical FOV 45 deg");
            ui.label("Black: horizon shadow | Color: lensed sky");
            ui.label(format!(
                "GPU RK4 geometry preview | Output: {}",
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

pub fn default_observation(width: u32, height: u32) -> Result<Observation, ValidationReport> {
    let spacetime = KerrNewmanSpacetime::new(1.0, 0.8, 0.0, KerrSchildChart::Outgoing)?;
    let observer_xyz = spacetime.oblate_to_cartesian(30.0, std::f64::consts::FRAC_PI_3, 0.0);
    let observer = StationaryObserverInput::new(
        [0.0, observer_xyz[0], observer_xyz[1], observer_xyz[2]],
        [0.0; 4],
        [0.0, 0.0, 1.0],
        1.0,
    );
    let scene = PhysicalScene::new(PhysicalSceneInput::new(
        1.0,
        0.8,
        0.0,
        KerrSchildChart::Outgoing,
        observer,
    ))?;
    let view = PerspectiveView::new(
        NonZeroU32::new(width).unwrap_or(NonZeroU32::MIN),
        NonZeroU32::new(height).unwrap_or(NonZeroU32::MIN),
        Angle::from_radians(std::f64::consts::FRAC_PI_4)?,
    )?;
    Ok(Observation::new(scene, view))
}

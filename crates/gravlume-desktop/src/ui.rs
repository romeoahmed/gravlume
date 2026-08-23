use gravlume_render::{
    DeviceEvent, RendererDiagnostics, SampleInspection, SampleInspectionEvent,
    SampleInspectionSource, SampleSceneValue,
};

use crate::{inspection::InspectionStatus, preview::Preview};

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
    inspection: &InspectionStatus,
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
            ui.label("Black: horizon | Color: lensed sky / equatorial surface");
            ui.label(format!(
                "GPU Kerr geodesics | Output: {}",
                diagnostics.display_transfer()
            ));
            ui.weak("Thin equatorial source with vacuum g⁴ transport; not a stable disk model.");
            ui.separator();
            show_sample_inspection(ui, inspection);
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

fn show_sample_inspection(ui: &mut egui::Ui, status: &InspectionStatus) {
    ui.strong("Sample inspection");
    match status {
        InspectionStatus::Idle => {
            ui.weak("Click the image outside this panel to inspect one published pixel.");
        }
        InspectionStatus::ViewportChanging => {
            ui.weak("Inspection waits until the resized viewport has a current complete frame.");
        }
        InspectionStatus::Pending(request_id) => {
            ui.label(format!(
                "Request {} is tracing and reading back.",
                request_id.get()
            ));
        }
        InspectionStatus::Rejected(error) => {
            ui.colored_label(
                egui::Color32::from_rgb(255, 170, 80),
                format!("Inspection not started: {error}"),
            );
        }
        InspectionStatus::Finished(event) => show_inspection_event(ui, event),
    }
}

fn show_inspection_event(ui: &mut egui::Ui, event: &SampleInspectionEvent) {
    match event {
        SampleInspectionEvent::Completed(inspection) => show_completed_inspection(ui, inspection),
        SampleInspectionEvent::Cancelled(identity) => {
            ui.weak(format!(
                "Request {} was cancelled after GPU drain.",
                identity.request_id().get()
            ));
        }
        SampleInspectionEvent::Superseded(identity) => {
            ui.weak(format!(
                "Request {} was superseded after generation {} stopped being published.",
                identity.request_id().get(),
                identity.generation()
            ));
        }
        SampleInspectionEvent::Failed { identity, error } => {
            ui.colored_label(
                egui::Color32::from_rgb(255, 170, 80),
                format!("Request {} failed: {error}", identity.request_id().get()),
            );
        }
        _ => {
            ui.weak("The renderer returned a newer inspection event kind.");
        }
    }
}

fn show_completed_inspection(ui: &mut egui::Ui, inspection: &SampleInspection) {
    let identity = inspection.identity();
    let [pixel_x, pixel_y] = identity.sample().pixel();
    let [subpixel_x, subpixel_y] = identity.sample().subpixel();
    let [width, height] = identity.extent();
    ui.label(format!(
        "request {} | observation {} | generation {}",
        identity.request_id().get(),
        identity.observation_id().get(),
        identity.generation()
    ));
    ui.label(format!(
        "pixel ({pixel_x}, {pixel_y}) + ({subpixel_x:.3}, {subpixel_y:.3}) | {width}×{height}"
    ));
    ui.weak(format!(
        "{:?} | {:?} | {:?}",
        identity.profile(),
        identity.producer(),
        identity.arithmetic_domain()
    ));

    let texel = inspection.published_texel();
    let [red, green, blue, alpha] = texel.rgba16_float_bits();
    ui.monospace(format!(
        "published {:?}: {red:04x} {green:04x} {blue:04x} {alpha:04x}",
        texel.kind()
    ));
    ui.label(format!(
        "fresh {:?}: {}",
        inspection.termination(),
        scene_value_label(inspection.evaluated_scene_value())
    ));
    ui.label(source_label(inspection.source()));
    if let Some(branch) = inspection.branch_key() {
        ui.label(format!(
            "branch {:?} | radial {} | equatorial {} | winding {}",
            branch.initial_polar_side(),
            branch.radial_turnings(),
            branch.equatorial_crossings(),
            branch.azimuth_winding()
        ));
    } else {
        ui.weak("branch unavailable for this terminal result");
    }
    ui.label(format!(
        "Δt/M {:.6} | steps {} | candidates 0x{:x}",
        inspection.travel_time_over_m(),
        inspection.steps(),
        inspection.event_candidate_bits()
    ));
    ui.label(format!(
        "event residual {:.3e} | flags 0x{:x} | max drift {:?}",
        inspection.event_residual(),
        inspection.numerical_flag_bits(),
        inspection.maximum_invariant_drift()
    ));
    if let Some(channels) = inspection.channel_model() {
        ui.weak(format!("surface channels: {channels:?}"));
    }
}

fn source_label(source: SampleInspectionSource) -> String {
    match source {
        SampleInspectionSource::None => "source: none".to_owned(),
        SampleInspectionSource::AnalyticEscape { unit_direction } => {
            format!("source: analytic escape direction {unit_direction:?}")
        }
        SampleInspectionSource::EquatorialSurface {
            radius_over_m,
            azimuth_radians,
            frequency_ratio,
        } => format!(
            "source: r/M {radius_over_m:.6} | azimuth {azimuth_radians:.6} | g {frequency_ratio:.6}"
        ),
        _ => "source: newer renderer source kind".to_owned(),
    }
}

fn scene_value_label(value: SampleSceneValue) -> String {
    match value {
        SampleSceneValue::Horizon => "horizon black".to_owned(),
        SampleSceneValue::AnalyticEscapePreview(rgb) => {
            format!("analytic preview RGB {rgb:?}")
        }
        SampleSceneValue::SurfaceRadiance(rgb) => format!("surface radiance RGB {rgb:?}"),
        SampleSceneValue::TraceFailure {
            termination,
            visible_rgb,
        } => format!("visible failure {termination:?}, RGB {visible_rgb:?}"),
        _ => "newer renderer scene-value kind".to_owned(),
    }
}

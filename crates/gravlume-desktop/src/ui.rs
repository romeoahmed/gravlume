use gravlume_render::{
    DeviceEvent, RendererDiagnostics, SampleBranchKey, SampleInspection,
    SampleInspectionCompletion, SampleInspectionDisposition, SampleInspectionTicket, SampleRetrace,
    SampleSurfaceEvaluation, SampleTraceOutcome,
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
        InspectionStatus::Pending(ticket) => {
            ui.label(format!(
                "Generation {} sample is tracing and reading back.",
                ticket.generation()
            ));
        }
        InspectionStatus::Rejected(error) => {
            ui.colored_label(
                egui::Color32::from_rgb(255, 170, 80),
                format!("Inspection not started: {error}"),
            );
        }
        InspectionStatus::Finished(completion) => show_inspection_completion(ui, completion),
    }
}

fn show_inspection_completion(ui: &mut egui::Ui, completion: &SampleInspectionCompletion) {
    let ticket = completion.ticket();
    match completion.disposition() {
        SampleInspectionDisposition::Completed(inspection) => {
            show_completed_inspection(ui, ticket, inspection);
        }
        SampleInspectionDisposition::Cancelled => {
            ui.weak("Inspection was cancelled after GPU drain.");
        }
        SampleInspectionDisposition::Failed(error) => {
            ui.colored_label(
                egui::Color32::from_rgb(255, 170, 80),
                format!("Inspection failed: {error}"),
            );
        }
    }
}

fn show_completed_inspection(
    ui: &mut egui::Ui,
    ticket: SampleInspectionTicket,
    inspection: &SampleInspection,
) {
    let [pixel_x, pixel_y] = ticket.sample().pixel();
    let [subpixel_x, subpixel_y] = ticket.sample().subpixel();
    let [width, height] = ticket.extent();
    ui.label(format!("generation {}", ticket.generation()));
    ui.label(format!(
        "pixel ({pixel_x}, {pixel_y}) + ({subpixel_x:.3}, {subpixel_y:.3}) | {width}×{height}"
    ));
    ui.weak(SampleRetrace::METHOD_ID);

    let texel = inspection.published_texel();
    let [red, green, blue, alpha] = texel.rgba16_float_bits();
    ui.monospace(format!(
        "published {:?}: {red:04x} {green:04x} {blue:04x} {alpha:04x}",
        texel.kind()
    ));
    let retrace = inspection.fresh_retrace();
    let [effective_x, effective_y] = retrace.effective_subpixel();
    ui.weak(format!(
        "effective binary32 subpixel ({effective_x:.7}, {effective_y:.7})"
    ));
    show_trace_outcome(ui, retrace.outcome());
    let diagnostics = retrace.diagnostics();
    ui.label(format!(
        "Δt/M {:.6} | steps {} | candidates 0x{:x}",
        diagnostics.coordinate_time_delta_over_m(),
        diagnostics.steps(),
        diagnostics.event_candidate_bits()
    ));
    ui.label(format!(
        "event residual {:.3e} | flags 0x{:x} | max drift {:?}",
        diagnostics.event_residual(),
        diagnostics.numerical_flag_bits(),
        diagnostics.maximum_invariant_drift()
    ));
}

fn show_trace_outcome(ui: &mut egui::Ui, outcome: SampleTraceOutcome) {
    match outcome {
        SampleTraceOutcome::Horizon { branch } => {
            ui.label("fresh Horizon: black");
            show_branch(ui, "branch", branch);
        }
        SampleTraceOutcome::Escape {
            branch,
            unit_direction,
            preview_rgb,
        } => {
            ui.label(format!(
                "fresh Escape: analytic preview RGB {preview_rgb:?}"
            ));
            ui.label(format!(
                "source: analytic escape direction {unit_direction:?}"
            ));
            show_branch(ui, "branch", branch);
        }
        SampleTraceOutcome::EquatorialSurface {
            branch,
            radius_over_m,
            azimuth_radians,
            frequency_ratio,
            channels,
            evaluation,
        } => {
            match evaluation {
                SampleSurfaceEvaluation::Radiance(rgb) => {
                    ui.label(format!("fresh EquatorialSurface: radiance RGB {rgb:?}"));
                }
                SampleSurfaceEvaluation::NumericalFailure { visible_rgb } => {
                    ui.label(format!(
                        "fresh EquatorialSurface: numerical evaluation failure RGB {visible_rgb:?}"
                    ));
                }
            }
            ui.label(format!(
                "source: r/M {radius_over_m:.6} | azimuth {azimuth_radians:.6} | g {frequency_ratio:.6}"
            ));
            ui.weak(format!("surface channels: {channels:?}"));
            show_branch(ui, "branch", branch);
        }
        SampleTraceOutcome::SingularityGuard {
            branch,
            visible_rgb,
        } => {
            ui.label(format!(
                "fresh SingularityGuard: visible RGB {visible_rgb:?}"
            ));
            show_branch(ui, "branch", branch);
        }
        SampleTraceOutcome::StepExhausted {
            branch_prefix,
            visible_rgb,
        } => {
            ui.label(format!("fresh StepExhausted: visible RGB {visible_rgb:?}"));
            show_branch(ui, "branch prefix", branch_prefix);
        }
        SampleTraceOutcome::NumericalFailure { visible_rgb } => {
            ui.label(format!(
                "fresh NumericalFailure: visible RGB {visible_rgb:?}"
            ));
            ui.weak("branch unavailable for this terminal result");
        }
        SampleTraceOutcome::Uncertain { visible_rgb } => {
            ui.label(format!("fresh Uncertain: visible RGB {visible_rgb:?}"));
            ui.weak("branch unavailable for this terminal result");
        }
    }
}

fn show_branch(ui: &mut egui::Ui, label: &str, branch: SampleBranchKey) {
    ui.label(format!(
        "{label} {:?} | radial {} | equatorial {} | winding {}",
        branch.initial_polar_side(),
        branch.radial_turnings(),
        branch.equatorial_crossings(),
        branch.azimuth_winding()
    ));
}

use egui::epaint::text::{FontInsert, FontPriority, InsertFontFamily};
use gravlume_render::{
    DeviceEvent, RendererDiagnostics, SampleBranchKey, SampleInspection,
    SampleInspectionCompletion, SampleInspectionTicket, SampleRetrace, SampleSurfaceEvaluation,
    SampleTraceOutcome,
};

use crate::{inspection::InspectionStatus, preview::Preview};

const CJK_FALLBACK_NAME: &str = "Noto Sans SC";
const CJK_FALLBACK_BYTES: &[u8] = include_bytes!("../assets/fonts/NotoSansSC-Regular.otf");

pub fn install_cjk_fallback_font(context: &egui::Context) {
    context.add_font(FontInsert::new(
        CJK_FALLBACK_NAME,
        egui::FontData::from_static(CJK_FALLBACK_BYTES),
        vec![
            InsertFontFamily {
                family: egui::FontFamily::Proportional,
                priority: FontPriority::Lowest,
            },
            InsertFontFamily {
                family: egui::FontFamily::Monospace,
                priority: FontPriority::Lowest,
            },
        ],
    ));
}

pub fn show_overlay(
    context: &egui::Context,
    diagnostics: &RendererDiagnostics<'_>,
    preview: Preview,
    inspection: &InspectionStatus,
    device_event: Option<&DeviceEvent>,
    resize_event: Option<&DeviceEvent>,
) {
    let content_rect = context.content_rect();
    let maximum_height = (content_rect.height() - 32.0).max(240.0);
    egui::Window::new("Gravlume")
        .default_pos([16.0, 16.0])
        .default_size([400.0, maximum_height.min(560.0)])
        .min_width(320.0)
        .max_width(560.0)
        .max_height(maximum_height)
        .resizable(true)
        .collapsible(true)
        .vscroll(true)
        .constrain_to(content_rect)
        .show(context, |ui| {
            ui.spacing_mut().item_spacing.y = 6.0;
            ui.heading("Kerr black-hole lensing");
            show_trace_progress(ui, diagnostics.trace_completion());
            show_scene_summary(ui, diagnostics, preview);
            ui.separator();
            egui::CollapsingHeader::new("Sample inspection")
                .default_open(true)
                .show(ui, |ui| show_sample_inspection(ui, inspection));
            if device_event.is_some() || resize_event.is_some() {
                ui.separator();
                ui.strong("Runtime notices");
            }
            if let Some(event) = device_event {
                show_warning(ui, format!("GPU {:?}: {}", event.kind(), event.message()));
            }
            if let Some(event) = resize_event {
                show_warning(
                    ui,
                    format!("Resize {:?}: {}", event.kind(), event.message()),
                );
            }
        });
}

fn show_trace_progress(ui: &mut egui::Ui, completion: Option<f64>) {
    match completion {
        Some(completion) if completion < 1.0 => {
            ui.add(
                egui::ProgressBar::new(progress_fraction(completion))
                    .show_percentage()
                    .animate(true),
            );
            ui.weak(
                "Tracing the full-resolution view; the previous complete frame remains visible.",
            );
        }
        Some(_) => {
            ui.label("Full-resolution trace complete.");
        }
        None => {
            ui.weak("Waiting for a non-zero viewport before tracing.");
        }
    }
}

const fn progress_fraction(completion: f64) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value is clamped to [0, 1] and only drives a pixel-scale progress widget"
    )]
    {
        completion.clamp(0.0, 1.0) as f32
    }
}

fn show_scene_summary(ui: &mut egui::Ui, diagnostics: &RendererDiagnostics<'_>, preview: Preview) {
    egui::Grid::new("scene summary")
        .num_columns(2)
        .spacing([12.0, 4.0])
        .show(ui, |ui| {
            ui.weak("Spin");
            ui.label(format!("a/M = {}", preview.spin_ratio()));
            ui.end_row();

            ui.weak("Observer");
            ui.label(format!("r/M = {}", preview.observer_radius_ratio()));
            ui.end_row();

            ui.weak("Vertical FOV");
            ui.label(format!("{}°", preview.vertical_fov_degrees()));
            ui.end_row();

            ui.weak("Output");
            ui.label(diagnostics.display_transfer());
            ui.end_row();
        });
    ui.weak("Black: horizon · Color: lensed sky or equatorial surface");
    ui.weak("Thin equatorial source with vacuum g⁴ transport; not a stable disk model.");

    egui::CollapsingHeader::new("Renderer details")
        .default_open(false)
        .show(ui, |ui| {
            egui::Grid::new("renderer details")
                .num_columns(2)
                .spacing([12.0, 4.0])
                .show(ui, |ui| {
                    detail_row(ui, "Adapter", diagnostics.adapter_name());
                    detail_row(ui, "Backend", diagnostics.backend());
                    detail_row(ui, "Driver", diagnostics.driver());
                    detail_row(ui, "Surface", diagnostics.surface_format());
                    detail_row(ui, "Color space", diagnostics.color_space());
                });
            if let (Some(batches), Some(total_ms), Some(maximum_batch_ms)) = (
                diagnostics.completed_trace_batches(),
                diagnostics.total_trace_compute_ms(),
                diagnostics.maximum_trace_batch_ms(),
            ) {
                ui.weak(format!(
                    "{batches} batches · {total_ms:.3} ms GPU total · {maximum_batch_ms:.3} ms maximum batch"
                ));
            }
        });
}

fn detail_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.weak(label);
    ui.label(value);
    ui.end_row();
}

fn show_sample_inspection(ui: &mut egui::Ui, status: &InspectionStatus) {
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
            show_warning(ui, format!("Inspection not started: {error}"));
        }
        InspectionStatus::Finished(completion) => show_inspection_completion(ui, completion),
    }
}

fn show_inspection_completion(ui: &mut egui::Ui, completion: &SampleInspectionCompletion) {
    match completion {
        SampleInspectionCompletion::Completed { ticket, inspection } => {
            show_completed_inspection(ui, *ticket, inspection);
        }
        SampleInspectionCompletion::Cancelled { .. } => {
            ui.weak("Inspection was cancelled after GPU drain.");
        }
        SampleInspectionCompletion::Failed { error, .. } => {
            show_warning(ui, format!("Inspection failed: {error}"));
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
    ui.label(format!(
        "Generation {} · pixel ({pixel_x}, {pixel_y}) + ({subpixel_x:.3}, {subpixel_y:.3}) · {width}×{height}",
        ticket.generation()
    ));
    let retrace = inspection.fresh_retrace();
    show_trace_outcome(ui, retrace.outcome());
    egui::CollapsingHeader::new("Numerical evidence")
        .default_open(false)
        .show(ui, |ui| {
            ui.weak(SampleRetrace::METHOD_ID);
            let texel = inspection.published_texel();
            let [red, green, blue, alpha] = texel.rgba16_float_bits();
            ui.monospace(format!(
                "published {:?}: {red:04x} {green:04x} {blue:04x} {alpha:04x}",
                texel.kind()
            ));
            let [effective_x, effective_y] = retrace.effective_subpixel();
            ui.weak(format!(
                "effective binary32 subpixel ({effective_x:.7}, {effective_y:.7})"
            ));
            let diagnostics = retrace.diagnostics();
            ui.label(format!(
                "Δt/M {:.6} · steps {} · candidates 0x{:x}",
                diagnostics.coordinate_time_delta_over_m(),
                diagnostics.steps(),
                diagnostics.event_candidate_bits()
            ));
            ui.label(format!(
                "event residual {:.3e} · flags 0x{:x} · max drift {:?}",
                diagnostics.event_residual(),
                diagnostics.numerical_flag_bits(),
                diagnostics.maximum_invariant_drift()
            ));
        });
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
        SampleTraceOutcome::StepExhaustion {
            branch_prefix,
            visible_rgb,
        } => {
            ui.label(format!("fresh StepExhaustion: visible RGB {visible_rgb:?}"));
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

fn show_warning(ui: &mut egui::Ui, message: impl Into<egui::RichText>) {
    let color = ui.visuals().warn_fg_color;
    ui.colored_label(color, message);
}

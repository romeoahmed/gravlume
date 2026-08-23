//! Ordered WGSL composition for trace pipelines.

const TRACE_PROTOCOL: &str = include_str!("../shaders/trace_protocol.wgsl");
const KERR_SCHILD_DYNAMICS: &str = include_str!("../shaders/kerr_schild_dynamics.wgsl");
const GEODESIC_EVENTS: &str = include_str!("../shaders/geodesic_events.wgsl");
const GEODESIC_INTEGRATION: &str = include_str!("../shaders/geodesic_integration.wgsl");
const LENSING_PREVIEW: &str = include_str!("../shaders/lensing_preview.wgsl");
const GEODESIC_ACCELERATION: &str = include_str!("../shaders/geodesic_acceleration.wgsl");
const SHADOW_COVERAGE: &str = include_str!("../shaders/shadow_coverage.wgsl");
const SURFACE_TRANSPORT: &str = include_str!("../shaders/surface_transport.wgsl");
const BOLOMETRIC_SURFACE_PREVIEW: &str = include_str!("../shaders/bolometric_surface_preview.wgsl");
const BLACKBODY_SURFACE_PREVIEW: &str = include_str!("../shaders/blackbody_surface_preview.wgsl");
const SAMPLE_INSPECTION: &str = include_str!("../shaders/sample_inspection.wgsl");
const ANALYTIC_SAMPLE_INSPECTION: &str = include_str!("../shaders/analytic_sample_inspection.wgsl");
const SURFACE_SAMPLE_INSPECTION: &str = include_str!("../shaders/surface_sample_inspection.wgsl");

#[cfg(test)]
const TRACE_CAPTURE: &str = include_str!("../shaders/trace_capture.wgsl");
#[cfg(test)]
const SURFACE_TRACE_CAPTURE: &str = include_str!("../shaders/surface_trace_capture.wgsl");
#[cfg(test)]
const SURFACE_FOOTPRINT_CAPTURE: &str = include_str!("../shaders/surface_footprint_capture.wgsl");
#[cfg(test)]
const ACCELERATED_TRACE_CAPTURE: &str = include_str!("../shaders/accelerated_trace_capture.wgsl");
#[cfg(test)]
const INITIAL_RAY_CAPTURE: &str = include_str!("../shaders/initial_ray_capture.wgsl");
#[cfg(test)]
const INVARIANT_GATE_CAPTURE: &str = include_str!("../shaders/invariant_gate_capture.wgsl");
#[cfg(test)]
const EVENT_POLICY_CAPTURE: &str = include_str!("../shaders/event_policy_capture.wgsl");

const TRACE_CORE: [&str; 4] = [
    TRACE_PROTOCOL,
    KERR_SCHILD_DYNAMICS,
    GEODESIC_EVENTS,
    GEODESIC_INTEGRATION,
];

pub(super) fn accelerated_scene() -> String {
    assemble(&[LENSING_PREVIEW, GEODESIC_ACCELERATION])
}

pub(super) fn bolometric_surface_scene() -> String {
    assemble(&[
        LENSING_PREVIEW,
        SURFACE_TRANSPORT,
        BOLOMETRIC_SURFACE_PREVIEW,
    ])
}

pub(super) fn blackbody_surface_scene() -> String {
    assemble(&[
        LENSING_PREVIEW,
        SURFACE_TRANSPORT,
        BLACKBODY_SURFACE_PREVIEW,
    ])
}

pub(super) fn analytic_sample_inspection() -> String {
    assemble(&[
        LENSING_PREVIEW,
        SAMPLE_INSPECTION,
        ANALYTIC_SAMPLE_INSPECTION,
    ])
}

pub(super) fn bolometric_sample_inspection() -> String {
    assemble(&[
        LENSING_PREVIEW,
        SURFACE_TRANSPORT,
        BOLOMETRIC_SURFACE_PREVIEW,
        SAMPLE_INSPECTION,
        SURFACE_SAMPLE_INSPECTION,
    ])
}

pub(super) fn blackbody_sample_inspection() -> String {
    assemble(&[
        LENSING_PREVIEW,
        SURFACE_TRANSPORT,
        BLACKBODY_SURFACE_PREVIEW,
        SAMPLE_INSPECTION,
        SURFACE_SAMPLE_INSPECTION,
    ])
}

pub(super) fn shadow_coverage() -> String {
    assemble(&[LENSING_PREVIEW, GEODESIC_ACCELERATION, SHADOW_COVERAGE])
}

#[cfg(test)]
pub(super) fn trace_capture() -> String {
    assemble(&[LENSING_PREVIEW, TRACE_CAPTURE])
}

#[cfg(test)]
pub(super) fn bolometric_surface_capture() -> String {
    assemble(&[
        LENSING_PREVIEW,
        TRACE_CAPTURE,
        SURFACE_TRANSPORT,
        BOLOMETRIC_SURFACE_PREVIEW,
        SURFACE_TRACE_CAPTURE,
    ])
}

#[cfg(test)]
pub(super) fn blackbody_surface_capture() -> String {
    assemble(&[
        LENSING_PREVIEW,
        TRACE_CAPTURE,
        SURFACE_TRANSPORT,
        BLACKBODY_SURFACE_PREVIEW,
        SURFACE_TRACE_CAPTURE,
    ])
}

#[cfg(test)]
pub(super) fn bolometric_surface_footprint_capture() -> String {
    assemble(&[
        LENSING_PREVIEW,
        TRACE_CAPTURE,
        SURFACE_TRANSPORT,
        BOLOMETRIC_SURFACE_PREVIEW,
        SURFACE_FOOTPRINT_CAPTURE,
    ])
}

#[cfg(test)]
pub(super) fn blackbody_surface_footprint_capture() -> String {
    assemble(&[
        LENSING_PREVIEW,
        TRACE_CAPTURE,
        SURFACE_TRANSPORT,
        BLACKBODY_SURFACE_PREVIEW,
        SURFACE_FOOTPRINT_CAPTURE,
    ])
}

#[cfg(test)]
pub(super) fn accelerated_capture() -> String {
    assemble(&[
        LENSING_PREVIEW,
        TRACE_CAPTURE,
        GEODESIC_ACCELERATION,
        ACCELERATED_TRACE_CAPTURE,
    ])
}

#[cfg(test)]
pub(super) fn initial_ray_capture() -> String {
    assemble(&[LENSING_PREVIEW, TRACE_CAPTURE, INITIAL_RAY_CAPTURE])
}

#[cfg(test)]
pub(super) fn invariant_gate_capture() -> String {
    assemble(&[LENSING_PREVIEW, TRACE_CAPTURE, INVARIANT_GATE_CAPTURE])
}

#[cfg(test)]
pub(super) fn event_policy_capture() -> String {
    assemble(&[LENSING_PREVIEW, TRACE_CAPTURE, EVENT_POLICY_CAPTURE])
}

fn assemble(suffix: &[&str]) -> String {
    let fragments = TRACE_CORE.iter().copied().chain(suffix.iter().copied());
    let capacity = fragments.clone().map(|fragment| fragment.len() + 1).sum();
    let mut source = String::with_capacity(capacity);
    for fragment in fragments {
        if !source.is_empty() {
            source.push('\n');
        }
        source.push_str(fragment);
    }
    source
}

// Test-only single-sample protocol candidate. Every host-shared lane is one aligned vec4; no
// vec3, bool, implicit padding, trajectory, or extent-scaled record survives the invocation.

struct SampleInspectionRequest {
    pixel_extent: vec4<u32>,
    subpixel: vec4<f32>,
    identity: vec4<u32>,
}

struct SampleInspectionRecord {
    identity: vec4<u32>,
    // (ABI version, producer, arithmetic domain, output kind)
    protocol: vec4<u32>,
    // (termination, failure flags, steps, event candidates)
    metadata: vec4<u32>,
    // Exact only for a determinate, non-numerical-failure termination. All zeroes mean
    // unavailable for a numerical failure; Uncertain may carry provisional counters that must
    // not be decoded as an exact branch.
    branch_key: vec4<u32>,
    source_time: vec4<f32>,
    // (scene-linear RGB, event residual)
    scene_event: vec4<f32>,
    maximum_invariant_drift: vec4<f32>,
}

const SAMPLE_INSPECTION_ABI_VERSION: u32 = 1u;
const SAMPLE_INSPECTION_PRODUCER_FULL_KS_RETRACE: u32 = 1u;
const SAMPLE_INSPECTION_DOMAIN_WGSL_F32: u32 = 1u;
const SAMPLE_OUTPUT_HORIZON: u32 = 1u;
const SAMPLE_OUTPUT_ANALYTIC_ESCAPE_PREVIEW: u32 = 2u;
const SAMPLE_OUTPUT_SURFACE_RADIANCE: u32 = 3u;
const SAMPLE_OUTPUT_TRACE_FAILURE: u32 = 4u;

@group(0) @binding(3)
var<storage, read_write> inspected_sample: SampleInspectionRecord;

@group(0) @binding(9)
var<uniform> inspection_request: SampleInspectionRequest;

fn inspection_output_kind(value: vec4<f32>) -> u32 {
    if value.w == 0.0 {
        return SAMPLE_OUTPUT_HORIZON;
    }
    if value.w == 1.0 {
        return SAMPLE_OUTPUT_ANALYTIC_ESCAPE_PREVIEW;
    }
    if value.w == 2.0 {
        return SAMPLE_OUTPUT_SURFACE_RADIANCE;
    }
    return SAMPLE_OUTPUT_TRACE_FAILURE;
}

fn store_inspected_geometry(sample: GeometricSample) {
    inspected_sample.identity = inspection_request.identity;
    inspected_sample.protocol = vec4<u32>(
        SAMPLE_INSPECTION_ABI_VERSION,
        SAMPLE_INSPECTION_PRODUCER_FULL_KS_RETRACE,
        SAMPLE_INSPECTION_DOMAIN_WGSL_F32,
        0u,
    );
    inspected_sample.metadata = vec4<u32>(
        sample.termination,
        sample.flags,
        sample.steps,
        sample.event_candidates,
    );
    inspected_sample.branch_key = sample.branch_key;
    inspected_sample.source_time = vec4<f32>(
        sample.source_coordinates,
        sample.travel_time,
    );
    inspected_sample.scene_event = vec4<f32>(0.0, 0.0, 0.0, sample.event_residual);
    inspected_sample.maximum_invariant_drift = sample.maximum_drift;
}

fn store_inspected_scene(value: vec4<f32>) {
    inspected_sample.protocol.w = inspection_output_kind(value);
    inspected_sample.scene_event = vec4<f32>(value.xyz, inspected_sample.scene_event.w);
}

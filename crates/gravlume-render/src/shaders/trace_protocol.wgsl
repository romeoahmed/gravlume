// Cartesian Kerr-Schild null-geodesic solver. Appearance and optional shadow refinement are
// composed as separate WGSL fragments so scientific fields stay independent of display work.

struct TraceUniforms {
    // (mass, spin, charge, Kerr-Schild branch sign)
    spacetime: vec4<f32>,
    // (initial polar side, normalized observer x, y, z)
    observer: vec4<f32>,
    observer_velocity: vec4<f32>,
    image_right: vec4<f32>,
    image_up: vec4<f32>,
    arrival: vec4<f32>,
    // (tan(vertical FOV / 2), observer frequency, subpixel x, subpixel y)
    camera: vec4<f32>,
    // (escape radius, singularity guard, horizon radius, event tie tolerance in M)
    event_surfaces: vec4<f32>,
    // (surface inner radius, outer radius, intensity at 6 M, arming band in M)
    surface_emitter: vec4<f32>,
    // (emitter T at 6 M, slab source T, transmittance, weighted source intensity)
    surface_transport: vec4<f32>,
    // (radial step scale, minimum step, maximum step, invariant drift limit)
    step_policy: vec4<f32>,
}

struct TraceDispatch {
    tile_origin: vec2<u32>,
    workgroup_count: vec2<u32>,
}

struct Geometry {
    radius: f32,
    ks_profile: f32,
    ks_profile_gradient: vec3<f32>,
    principal_spatial: vec3<f32>,
    radius_gradient: vec3<f32>,
    inverse_scale: f32,
    inverse_radius: f32,
    inverse_oblate_factor: f32,
    singularity_measure: f32,
    flags: u32,
}

struct TraceState {
    position: vec3<f32>,
    momentum: vec3<f32>,
}

struct RhsResult {
    spacetime: vec4<f32>,
    momentum: vec3<f32>,
    flags: u32,
}

struct StepResult {
    state: TraceState,
    coordinate_time_increment: f32,
    flags: u32,
}

struct InitialState {
    state: TraceState,
    energy: f32,
    geometry: Geometry,
    rhs: RhsResult,
}

struct Invariants {
    values: vec4<f32>,
    flags: u32,
}

struct GeometricSample {
    termination: u32,
    flags: u32,
    event_candidates: u32,
    steps: u32,
    event_residual: f32,
    source_coordinates: vec3<f32>,
    travel_time: f32,
    maximum_drift: vec4<f32>,
    // (radial turnings, equatorial crossings, bitcast azimuth winding, initial polar side)
    branch_key: vec4<u32>,
}

const TERMINATION_HORIZON: u32 = 1u;
const TERMINATION_ESCAPE: u32 = 2u;
const TERMINATION_SINGULARITY: u32 = 3u;
const TERMINATION_STEP_EXHAUSTION: u32 = 4u;
const TERMINATION_NUMERICAL_FAILURE: u32 = 5u;
const TERMINATION_UNCERTAIN: u32 = 6u;
const TERMINATION_EQUATORIAL_SURFACE: u32 = 7u;
const FLAG_NON_FINITE: u32 = 1u;
const FLAG_INVALID_RADICAND: u32 = 2u;
const FLAG_INVALID_DENOMINATOR: u32 = 4u;
const EVENT_CANDIDATE_SINGULARITY: u32 = 1u;
const EVENT_CANDIDATE_HORIZON: u32 = 2u;
const EVENT_CANDIDATE_SURFACE: u32 = 4u;
const EVENT_CANDIDATE_ESCAPE: u32 = 8u;
const EVENT_INDEX_SINGULARITY: u32 = 0u;
const EVENT_INDEX_HORIZON: u32 = 1u;
const EVENT_INDEX_SURFACE: u32 = 2u;
const EVENT_INDEX_ESCAPE: u32 = 3u;
const TRACE_WORKGROUP_AXIS: u32 = 8u;
const MAXIMUM_STEPS: u32 = 2048u;
const MAXIMUM_FINITE_F32: f32 = 0x1.fffffep+127f;
const EVENT_REFINEMENT_ITERATIONS: u32 = 6u;
const TURNING_REFINEMENT_ITERATIONS: u32 = 20u;
const EVENT_DERIVATIVE_RELATIVE_FLOOR: f32 = 0x1p-11f;
override SURFACE_EVENTS_ENABLED: u32 = 0u;

alias EventCandidates = vec4<f32>;

struct EventSelection {
    termination: u32,
    fraction: f32,
    candidates: u32,
    ambiguous: u32,
}

struct SurfaceSource {
    coordinates: vec3<f32>,
    flags: u32,
}

struct RadialVelocitySample {
    value: f32,
    flags: u32,
}

struct RadialTurningBracket {
    fractions: vec2<f32>,
    valid: u32,
}

@group(0) @binding(0)
var<uniform> trace_uniforms: TraceUniforms;

@group(0) @binding(2)
var<uniform> trace_dispatch: TraceDispatch;

fn finite_scalar(value: f32) -> bool {
    return value == value && abs(value) <= MAXIMUM_FINITE_F32;
}

fn finite_vec4(value: vec4<f32>) -> bool {
    return all(value == value) && all(abs(value) <= vec4<f32>(MAXIMUM_FINITE_F32));
}

fn finite_vec3(value: vec3<f32>) -> bool {
    return all(value == value) && all(abs(value) <= vec3<f32>(MAXIMUM_FINITE_F32));
}

// WGSL runtime overflow may produce an indeterminate value under the finite-math assumption, so
// guard the multiplication rather than trying to clamp its result.
// Source: https://www.w3.org/TR/WGSL/#floating-point-evaluation
fn saturating_positive_product(left: f32, right: f32) -> f32 {
    if left == 0.0 || right == 0.0 {
        return 0.0;
    }
    if left > 1.0 && right > MAXIMUM_FINITE_F32 / left {
        return MAXIMUM_FINITE_F32;
    }
    return left * right;
}

fn saturating_positive_sum(left: f32, right: f32) -> f32 {
    if left > MAXIMUM_FINITE_F32 - right {
        return MAXIMUM_FINITE_F32;
    }
    return left + right;
}

fn positive_square(value: f32) -> f32 {
    return saturating_positive_product(abs(value), abs(value));
}

fn singularity_measure(radius: f32, spin: f32, z: f32) -> f32 {
    let radius_squared = positive_square(radius);
    let radius_fourth = positive_square(radius_squared);
    let spin_z = saturating_positive_product(abs(spin), abs(z));
    return saturating_positive_sum(radius_fourth, positive_square(spin_z));
}

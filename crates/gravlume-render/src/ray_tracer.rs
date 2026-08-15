use std::borrow::Cow;

use gravlume_domain::{
    EquatorialCircularEmitter, Extremality, HomogeneousScalarSlab, KerrNewmanSpacetime,
    KerrSchildChart, Observation, ValidationReport,
};
use num_traits::ToPrimitive as _;
use wgpu::util::DeviceExt as _;

use crate::{
    extent::RenderExtent,
    shadow_coverage::{ShadowCoverage, ShadowTarget},
    spectral_lut::{
        BLACKBODY_LUT_BYTE_SIZE, blackbody_lut, maximum_temperature_kelvin,
        minimum_temperature_kelvin,
    },
};

pub const KERR_SCHILD_TRACE_SHADER: &str = include_str!("shaders/kerr_schild_trace.wgsl");
pub const LENSING_PREVIEW_SHADER: &str = include_str!("shaders/lensing_preview.wgsl");
pub const GEODESIC_ACCELERATION_SHADER: &str = include_str!("shaders/geodesic_acceleration.wgsl");
pub const SHADOW_COVERAGE_SHADER: &str = include_str!("shaders/shadow_coverage.wgsl");
pub const SURFACE_PREVIEW_SHADER: &str = include_str!("shaders/surface_preview.wgsl");
pub const SURFACE_TRANSPORT_SHADER: &str = include_str!("shaders/surface_transport.wgsl");
pub const SPECTRAL_SURFACE_PREVIEW_SHADER: &str =
    include_str!("shaders/spectral_surface_preview.wgsl");
#[cfg(test)]
const TRACE_CAPTURE_SHADER: &str = include_str!("shaders/trace_capture.wgsl");
#[cfg(test)]
const SURFACE_TRACE_CAPTURE_SHADER: &str = include_str!("shaders/surface_trace_capture.wgsl");
#[cfg(test)]
const SURFACE_FOOTPRINT_CAPTURE_SHADER: &str =
    include_str!("shaders/surface_footprint_capture.wgsl");
#[cfg(test)]
const ACCELERATED_TRACE_CAPTURE_SHADER: &str =
    include_str!("shaders/accelerated_trace_capture.wgsl");
#[cfg(test)]
const INITIAL_RAY_CAPTURE_SHADER: &str = include_str!("shaders/initial_ray_capture.wgsl");
#[cfg(test)]
const INVARIANT_GATE_CAPTURE_SHADER: &str = include_str!("shaders/invariant_gate_capture.wgsl");
#[cfg(test)]
const EVENT_POLICY_CAPTURE_SHADER: &str = include_str!("shaders/event_policy_capture.wgsl");
pub const INVARIANT_DRIFT_LIMIT: f32 = 0.05;
const NORMALIZED_FREQUENCY_TOLERANCE: f64 = 32.0 * f64::EPSILON;
const OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_OVER_M: f32 = 200.0;
const GPU_EVENT_TIE_TOLERANCE_OVER_M: f32 = f32::from_bits(0x3700_0000);
const GPU_EVENT_ARMING_BAND_OVER_M: f32 = f32::from_bits(0x3980_0000);
const TRACE_RECORD_PLANE_ELEMENT_SIZE: u64 = 16;
pub const TRACE_WORKGROUP_WIDTH: u32 = 8;
pub const TRACE_WORKGROUP_HEIGHT: u32 = 8;

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct TraceUniforms {
    spacetime: [f32; 4],
    observer_event: [f32; 4],
    observer_velocity: [f32; 4],
    image_right: [f32; 4],
    image_up: [f32; 4],
    arrival: [f32; 4],
    camera: [f32; 4],
    event_surfaces: [f32; 4],
    surface_emitter: [f32; 4],
    surface_transport: [f32; 4],
    step_policy: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<TraceUniforms>() == 176);

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct TraceDispatch {
    tile_origin: [u32; 2],
    tile_count: [u32; 2],
}

impl TraceUniforms {
    pub fn from_observation(observation: &Observation) -> Result<Self, GpuTraceInputError> {
        let scene = observation.scene();
        let physical_spacetime = *scene.spacetime();
        let mass = physical_spacetime.mass_m();
        let view = *observation.view();
        let frame = scene.observer_frame();
        let sample = view
            .sample(0, 0, 0.5, 0.5)
            .map_err(GpuTraceInputError::DomainInvariant)?;
        let observer_frequency = observation
            .initial_ray(sample)
            .map_err(GpuTraceInputError::DomainInvariant)?
            .observer_frequency();
        if (observer_frequency - 1.0).abs() > NORMALIZED_FREQUENCY_TOLERANCE {
            return Err(GpuTraceInputError::NonNormalizedObserverFrequency { observer_frequency });
        }
        let spacetime_uniform = pack4(
            [
                1.0,
                physical_spacetime.spin_m() / mass,
                physical_spacetime.charge_m() / mass,
                match physical_spacetime.chart() {
                    KerrSchildChart::Ingoing => 1.0,
                    KerrSchildChart::Outgoing => -1.0,
                },
            ],
            "spacetime",
        )?;
        let gpu_spacetime = KerrNewmanSpacetime::new(
            f64::from(spacetime_uniform[0]),
            f64::from(spacetime_uniform[1]),
            f64::from(spacetime_uniform[2]),
            physical_spacetime.chart(),
        )
        .map_err(GpuTraceInputError::DomainInvariant)?;
        let canonical_state = physical_spacetime.extremality();
        let gpu_extremality = gpu_spacetime.extremality();
        if gpu_extremality != canonical_state {
            return Err(GpuTraceInputError::ExtremalityChangedByPacking {
                canonical_state,
                gpu_extremality,
            });
        }
        let horizon = gpu_spacetime.outer_horizon_radius().unwrap_or(-1.0);
        let [_, observer_x, observer_y, observer_z] = scene.observer_event().to_txyz();
        let emitter = scene.equatorial_circular_emitter();
        let slab = scene.homogeneous_scalar_slab();
        let surface_emitter = pack_surface_emitter(emitter, mass)?;
        let surface_transport = pack_surface_transport(emitter, slab, mass)?;

        Ok(Self {
            spacetime: spacetime_uniform,
            observer_event: pack4(
                [0.0, observer_x / mass, observer_y / mass, observer_z / mass],
                "observer_event",
            )?,
            observer_velocity: pack4(frame.four_velocity_txyz(), "observer_velocity")?,
            image_right: pack4(frame.image_right_txyz(), "image_right")?,
            image_up: pack4(frame.image_up_txyz(), "image_up")?,
            arrival: pack4(frame.arrival_direction_txyz(), "arrival")?,
            camera: pack4(
                [(view.vertical_fov().radians() * 0.5).tan(), 1.0, 0.5, 0.5],
                "camera",
            )?,
            event_surfaces: pack4(
                [
                    f64::from(OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_OVER_M),
                    f64::from(f32::from_bits(0x2b80_0000)),
                    horizon,
                    f64::from(GPU_EVENT_TIE_TOLERANCE_OVER_M),
                ],
                "event_surfaces",
            )?,
            surface_emitter,
            surface_transport,
            step_policy: [0.1, 0.005, 8.0, INVARIANT_DRIFT_LIMIT],
        })
    }
}

fn pack_surface_transport(
    emitter: Option<EquatorialCircularEmitter>,
    slab: Option<HomogeneousScalarSlab>,
    mass_m: f64,
) -> Result<[f32; 4], GpuTraceInputError> {
    let Some(emitter) = emitter else {
        if slab.is_some() {
            return Err(GpuTraceInputError::ScalarSlabRequiresSurface);
        }
        return Ok([0.0, 0.0, 1.0, 0.0]);
    };
    let emitter_temperature = emitter.blackbody_temperature_at_six_kelvin();
    if let Some(temperature_at_six_kelvin) = emitter_temperature {
        for (radius_m, field) in [
            (
                emitter.inner_radius_m(),
                "equatorial_circular_emitter.inner_temperature_kelvin",
            ),
            (
                emitter.outer_radius_m(),
                "equatorial_circular_emitter.outer_temperature_kelvin",
            ),
        ] {
            let radius_ratio = radius_m / mass_m / 6.0;
            let temperature =
                temperature_at_six_kelvin / (radius_ratio * radius_ratio.sqrt()).sqrt();
            validate_lut_temperature(temperature, field)?;
        }
    }

    let (transmittance, weighted_source_intensity, source_temperature) =
        slab.map_or((1.0, 0.0, None), |slab| {
            let transmittance = (-slab.optical_depth()).exp();
            (
                transmittance,
                slab.integrated_bolometric_emission(),
                slab.emission_temperature_kelvin(),
            )
        });
    let transmittance = if transmittance < f64::from(f32::MIN_POSITIVE) {
        0.0
    } else {
        transmittance
    };
    let source_temperature = if emitter_temperature.is_some() && weighted_source_intensity > 0.0 {
        let temperature =
            source_temperature.ok_or(GpuTraceInputError::UnresolvedSlabSourceSpectrum)?;
        validate_lut_temperature(
            temperature,
            "homogeneous_scalar_slab.emission_temperature_kelvin",
        )?;
        temperature
    } else {
        0.0
    };
    let packed = pack4(
        [
            emitter_temperature.unwrap_or(0.0),
            source_temperature,
            transmittance,
            weighted_source_intensity,
        ],
        "surface_transport",
    )?;
    let source_underflowed = weighted_source_intensity > 0.0 && packed[3] == 0.0;
    if source_underflowed || packed.iter().any(|value| value.is_subnormal()) {
        return Err(GpuTraceInputError::NotRepresentable {
            field: "surface_transport",
        });
    }
    Ok(packed)
}

fn validate_lut_temperature(
    temperature_kelvin: f64,
    field: &'static str,
) -> Result<(), GpuTraceInputError> {
    let minimum_kelvin = minimum_temperature_kelvin();
    let maximum_kelvin = maximum_temperature_kelvin();
    if (minimum_kelvin..=maximum_kelvin).contains(&temperature_kelvin) {
        Ok(())
    } else {
        Err(GpuTraceInputError::TemperatureOutsideSpectralLut {
            field,
            temperature_kelvin,
            minimum_kelvin,
            maximum_kelvin,
        })
    }
}

fn pack_surface_emitter(
    emitter: Option<EquatorialCircularEmitter>,
    mass_m: f64,
) -> Result<[f32; 4], GpuTraceInputError> {
    let Some(emitter) = emitter else {
        return Ok([0.0; 4]);
    };
    let outer_radius_over_m = emitter.outer_radius_m() / mass_m;
    if outer_radius_over_m >= f64::from(OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_OVER_M) {
        return Err(GpuTraceInputError::SurfaceOutsideEscapeBoundary {
            outer_radius_over_m,
            escape_radius_over_m: f64::from(OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_OVER_M),
        });
    }
    let packed = pack4(
        [
            emitter.inner_radius_m() / mass_m,
            outer_radius_over_m,
            emitter.intensity_at_six_m(),
            f64::from(GPU_EVENT_ARMING_BAND_OVER_M),
        ],
        "surface_emitter",
    )?;
    // A binary64 value inside the boundary can round onto it when packed for the shader.
    // Source: https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast
    if packed[1] >= OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_OVER_M {
        return Err(GpuTraceInputError::SurfaceOutsideEscapeBoundary {
            outer_radius_over_m,
            escape_radius_over_m: f64::from(OBSERVATION_BASELINE_V1_ESCAPE_RADIUS_OVER_M),
        });
    }
    let interval_collapsed = emitter.outer_radius_m() > emitter.inner_radius_m()
        && packed[1].to_bits() == packed[0].to_bits();
    let intensity_underflowed = emitter.intensity_at_six_m() > 0.0 && packed[2] == 0.0;
    if packed[0] <= 0.0
        || packed[1] < packed[0]
        || packed[2] < 0.0
        || packed[..3].iter().any(|value| value.is_subnormal())
        || interval_collapsed
        || intensity_underflowed
    {
        return Err(GpuTraceInputError::NotRepresentable {
            field: "surface_emitter",
        });
    }
    Ok(packed)
}

fn pack4(values: [f64; 4], field: &'static str) -> Result<[f32; 4], GpuTraceInputError> {
    let [Some(a), Some(b), Some(c), Some(d)] =
        values.map(|value| value.to_f32().filter(|packed| packed.is_finite()))
    else {
        return Err(GpuTraceInputError::NotRepresentable { field });
    };
    Ok([a, b, c, d])
}

#[derive(Debug, thiserror::Error)]
pub enum GpuTraceInputError {
    #[error("validated observation failed to resolve its initial ray: {0}")]
    DomainInvariant(#[source] ValidationReport),
    #[error(
        "GPU trace inputs must be normalized to observer frequency 1, got {observer_frequency}"
    )]
    NonNormalizedObserverFrequency { observer_frequency: f64 },
    #[error(
        "spacetime extremality changes from {canonical_state:?} to {gpu_extremality:?} under the GPU f32 contract"
    )]
    ExtremalityChangedByPacking {
        canonical_state: Extremality,
        gpu_extremality: Extremality,
    },
    #[error("observation field {field} is not representable by the GPU f32 contract")]
    NotRepresentable { field: &'static str },
    #[error(
        "surface outer radius {outer_radius_over_m} M must be strictly inside the GPU escape boundary {escape_radius_over_m} M"
    )]
    SurfaceOutsideEscapeBoundary {
        outer_radius_over_m: f64,
        escape_radius_over_m: f64,
    },
    #[error("a homogeneous scalar slab requires a resolved equatorial surface source")]
    ScalarSlabRequiresSurface,
    #[error("a non-zero slab source requires a blackbody spectrum for spectral GPU transport")]
    UnresolvedSlabSourceSpectrum,
    #[error(
        "{field} temperature {temperature_kelvin} K lies outside the GPU spectral LUT [{minimum_kelvin}, {maximum_kelvin}] K"
    )]
    TemperatureOutsideSpectralLut {
        field: &'static str,
        temperature_kelvin: f64,
        minimum_kelvin: f64,
        maximum_kelvin: f64,
    },
    #[error("surface footprint capture requires an equatorial surface source")]
    SurfaceFootprintRequiresSurface,
}

pub struct RayTracer {
    pipeline: wgpu::ComputePipeline,
    escape_map_node_pipeline: Option<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    dispatch: wgpu::Buffer,
    blackbody_lut: Option<wgpu::Buffer>,
    shadow_coverage: Option<ShadowCoverage>,
    plan: TracePlan,
    #[cfg(test)]
    target_kind: TraceTargetKind,
}

#[derive(Clone, Copy)]
enum TracePlan {
    AcceleratedSky,
    EquatorialBolometricSurface,
    EquatorialBlackbodySurface,
}

impl TracePlan {
    const fn resolve(observation: &Observation) -> Self {
        match observation.scene().equatorial_circular_emitter() {
            Some(emitter) if emitter.blackbody_temperature_at_six_kelvin().is_some() => {
                Self::EquatorialBlackbodySurface
            }
            Some(_) => Self::EquatorialBolometricSurface,
            None => Self::AcceleratedSky,
        }
    }

    fn scratch_bytes(self, extent: RenderExtent) -> u64 {
        match self {
            Self::AcceleratedSky => shadow_coverage_scratch_bytes(extent)
                .saturating_add(escape_map_scratch_bytes(extent)),
            Self::EquatorialBolometricSurface | Self::EquatorialBlackbodySurface => 0,
        }
    }

    const fn surface_events_enabled(self) -> f64 {
        match self {
            Self::AcceleratedSky => 0.0,
            Self::EquatorialBolometricSurface | Self::EquatorialBlackbodySurface => 1.0,
        }
    }

    const fn has_blackbody_lut(self) -> bool {
        matches!(self, Self::EquatorialBlackbodySurface)
    }
}

#[derive(Clone, Copy)]
enum TraceTargetKind {
    Presentation,
    #[cfg(test)]
    Diagnostic,
}

struct TracePipelineSpec {
    shader_source: Cow<'static, str>,
    entry_point: &'static str,
    escape_map_node_entry_point: Option<&'static str>,
    has_shadow_refinement: bool,
    plan: TracePlan,
    target_kind: TraceTargetKind,
}

pub struct TraceBatchOptions<'a> {
    escape_map_timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'a>>,
    trace_timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'a>>,
    refine_final_batch: bool,
}

impl<'a> TraceBatchOptions<'a> {
    pub const fn new(
        escape_map_timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'a>>,
        trace_timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'a>>,
        refine_final_batch: bool,
    ) -> Self {
        Self {
            escape_map_timestamp_writes,
            trace_timestamp_writes,
            refine_final_batch,
        }
    }

    #[cfg(test)]
    const fn untimed(refine_final_batch: bool) -> Self {
        Self::new(None, None, refine_final_batch)
    }
}

impl TracePlan {
    fn presentation_spec(self) -> TracePipelineSpec {
        match self {
            Self::AcceleratedSky => TracePipelineSpec {
                shader_source: accelerated_shader_source(),
                entry_point: "trace_scene_accelerated",
                escape_map_node_entry_point: Some("trace_escape_map_nodes"),
                has_shadow_refinement: true,
                plan: self,
                target_kind: TraceTargetKind::Presentation,
            },
            Self::EquatorialBolometricSurface => TracePipelineSpec {
                shader_source: bolometric_surface_shader_source(),
                entry_point: "trace_surface_scene",
                escape_map_node_entry_point: None,
                has_shadow_refinement: false,
                plan: self,
                target_kind: TraceTargetKind::Presentation,
            },
            Self::EquatorialBlackbodySurface => TracePipelineSpec {
                shader_source: spectral_surface_shader_source(),
                entry_point: "trace_spectral_surface_scene",
                escape_map_node_entry_point: None,
                has_shadow_refinement: false,
                plan: self,
                target_kind: TraceTargetKind::Presentation,
            },
        }
    }

    #[cfg(test)]
    fn capture_spec(self) -> TracePipelineSpec {
        match self {
            Self::AcceleratedSky => TracePipelineSpec {
                shader_source: trace_capture_shader_source(),
                entry_point: "capture_trace_scene",
                escape_map_node_entry_point: None,
                has_shadow_refinement: true,
                plan: self,
                target_kind: TraceTargetKind::Diagnostic,
            },
            Self::EquatorialBolometricSurface => TracePipelineSpec {
                shader_source: bolometric_surface_capture_shader_source(),
                entry_point: "capture_surface_trace_scene",
                escape_map_node_entry_point: None,
                has_shadow_refinement: false,
                plan: self,
                target_kind: TraceTargetKind::Diagnostic,
            },
            Self::EquatorialBlackbodySurface => TracePipelineSpec {
                shader_source: spectral_surface_capture_shader_source(),
                entry_point: "capture_surface_trace_scene",
                escape_map_node_entry_point: None,
                has_shadow_refinement: false,
                plan: self,
                target_kind: TraceTargetKind::Diagnostic,
            },
        }
    }

    #[cfg(test)]
    fn footprint_capture_spec(self) -> Result<TracePipelineSpec, GpuTraceInputError> {
        match self {
            Self::AcceleratedSky => Err(GpuTraceInputError::SurfaceFootprintRequiresSurface),
            Self::EquatorialBolometricSurface | Self::EquatorialBlackbodySurface => {
                Ok(TracePipelineSpec {
                    shader_source: surface_footprint_capture_shader_source(self),
                    entry_point: "capture_surface_footprint",
                    escape_map_node_entry_point: None,
                    has_shadow_refinement: false,
                    plan: self,
                    target_kind: TraceTargetKind::Diagnostic,
                })
            }
        }
    }
}

impl TraceTargetKind {
    const fn captures_records(self) -> bool {
        match self {
            Self::Presentation => false,
            #[cfg(test)]
            Self::Diagnostic => true,
        }
    }
}

fn trace_bind_group_layout_entries(
    target_kind: TraceTargetKind,
    has_escape_map: bool,
    has_blackbody_lut: bool,
) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size_of::<TraceUniforms>()),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size_of::<TraceDispatch>()),
            },
            count: None,
        },
    ];
    if target_kind.captures_records() {
        entries.extend((3..=6).map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(TRACE_RECORD_PLANE_ELEMENT_SIZE),
            },
            count: None,
        }));
    }
    if has_escape_map {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size_of::<u32>()),
            },
            count: None,
        });
    }
    if has_blackbody_lut {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(BLACKBODY_LUT_BYTE_SIZE),
            },
            count: None,
        });
    }
    entries
}

impl RayTracer {
    pub fn new(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let uniforms = TraceUniforms::from_observation(observation)?;
        let plan = TracePlan::resolve(observation);
        Ok(Self::from_uniforms(
            device,
            uniforms,
            plan.presentation_spec(),
        ))
    }

    #[cfg(test)]
    pub fn for_trace_capture(
        device: &wgpu::Device,
        observation: &Observation,
        subpixel: [f64; 2],
    ) -> Result<Self, GpuTraceInputError> {
        let mut uniforms = TraceUniforms::from_observation(observation)?;
        let packed_subpixel = pack4(
            [subpixel[0], subpixel[1], 0.0, 0.0],
            "trace_capture_subpixel",
        )?;
        uniforms.camera[2..].copy_from_slice(&packed_subpixel[..2]);
        let plan = TracePlan::resolve(observation);
        Ok(Self::from_uniforms(device, uniforms, plan.capture_spec()))
    }

    #[cfg(test)]
    pub fn for_surface_footprint_capture(
        device: &wgpu::Device,
        observation: &Observation,
        subpixel: [f64; 2],
    ) -> Result<Self, GpuTraceInputError> {
        let mut uniforms = TraceUniforms::from_observation(observation)?;
        let packed_subpixel = pack4(
            [subpixel[0], subpixel[1], 0.0, 0.0],
            "surface_footprint_subpixel",
        )?;
        uniforms.camera[2..].copy_from_slice(&packed_subpixel[..2]);
        // Source-chart differencing subtracts neighboring terminal anchors. Keep this diagnostic
        // path below the presentation step ceiling so RK4 phase error does not dominate J.
        uniforms.step_policy[0] = 0.0025;
        uniforms.step_policy[1] = 0.000_125;
        uniforms.step_policy[2] = 0.25;
        let plan = TracePlan::resolve(observation);
        Ok(Self::from_uniforms(
            device,
            uniforms,
            plan.footprint_capture_spec()?,
        ))
    }

    #[cfg(test)]
    pub fn for_accelerated_trace_capture(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let uniforms = TraceUniforms::from_observation(observation)?;
        Ok(Self::from_uniforms(
            device,
            uniforms,
            TracePipelineSpec {
                shader_source: accelerated_capture_shader_source(),
                entry_point: "capture_accelerated_trace_scene",
                escape_map_node_entry_point: Some("trace_escape_map_nodes"),
                has_shadow_refinement: true,
                plan: TracePlan::AcceleratedSky,
                target_kind: TraceTargetKind::Diagnostic,
            },
        ))
    }

    #[cfg(test)]
    pub fn for_initial_ray_capture(
        device: &wgpu::Device,
        observation: &Observation,
        subpixel: [f32; 2],
    ) -> Result<Self, GpuTraceInputError> {
        let mut uniforms = TraceUniforms::from_observation(observation)?;
        uniforms.camera[2..].copy_from_slice(&subpixel);
        Ok(Self::from_uniforms(
            device,
            uniforms,
            TracePipelineSpec {
                shader_source: Cow::Owned(format!(
                    "{}\n{INITIAL_RAY_CAPTURE_SHADER}",
                    trace_capture_shader_source()
                )),
                entry_point: "write_initial_rays",
                escape_map_node_entry_point: None,
                has_shadow_refinement: false,
                plan: TracePlan::resolve(observation),
                target_kind: TraceTargetKind::Diagnostic,
            },
        ))
    }

    #[cfg(test)]
    pub fn for_invariant_gate_capture(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let uniforms = TraceUniforms::from_observation(observation)?;
        Ok(Self::from_uniforms(
            device,
            uniforms,
            TracePipelineSpec {
                shader_source: Cow::Owned(format!(
                    "{}\n{INVARIANT_GATE_CAPTURE_SHADER}",
                    trace_capture_shader_source()
                )),
                entry_point: "write_invariant_gate_cases",
                escape_map_node_entry_point: None,
                has_shadow_refinement: false,
                plan: TracePlan::resolve(observation),
                target_kind: TraceTargetKind::Diagnostic,
            },
        ))
    }

    #[cfg(test)]
    pub fn for_event_policy_capture(
        device: &wgpu::Device,
        observation: &Observation,
    ) -> Result<Self, GpuTraceInputError> {
        let uniforms = TraceUniforms::from_observation(observation)?;
        Ok(Self::from_uniforms(
            device,
            uniforms,
            TracePipelineSpec {
                shader_source: Cow::Owned(format!(
                    "{}\n{EVENT_POLICY_CAPTURE_SHADER}",
                    trace_capture_shader_source()
                )),
                entry_point: "write_event_policy_cases",
                escape_map_node_entry_point: None,
                has_shadow_refinement: false,
                plan: TracePlan::resolve(observation),
                target_kind: TraceTargetKind::Diagnostic,
            },
        ))
    }

    fn from_uniforms(
        device: &wgpu::Device,
        uniforms: TraceUniforms,
        spec: TracePipelineSpec,
    ) -> Self {
        let TracePipelineSpec {
            shader_source,
            entry_point,
            escape_map_node_entry_point,
            has_shadow_refinement,
            plan,
            target_kind,
        } = spec;
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/util/trait.DeviceExt.html#tymethod.create_buffer_init
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU trace uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dispatch = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU trace dispatch"),
            contents: bytemuck::bytes_of(&TraceDispatch {
                tile_origin: [0; 2],
                tile_count: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let blackbody_lut = plan.has_blackbody_lut().then(|| {
            let entries = blackbody_lut();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blackbody spectral boxcar LUT"),
                contents: bytemuck::cast_slice(&entries),
                usage: wgpu::BufferUsages::STORAGE,
            })
        });
        let shadow_coverage = has_shadow_refinement.then(|| ShadowCoverage::new(device, &uniforms));
        let entries = trace_bind_group_layout_entries(
            target_kind,
            escape_map_node_entry_point.is_some(),
            blackbody_lut.is_some(),
        );
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU trace bind group layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU trace pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cartesian Kerr-Schild trace shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source),
        });
        let pipeline_constants = [("SURFACE_EVENTS_ENABLED", plan.surface_events_enabled())];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &pipeline_constants,
                ..Default::default()
            },
            cache: None,
        });
        let escape_map_node_pipeline = escape_map_node_entry_point.map(|entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &pipeline_constants,
                    ..Default::default()
                },
                cache: None,
            })
        });

        Self {
            pipeline,
            escape_map_node_pipeline,
            bind_group_layout,
            uniforms,
            dispatch,
            blackbody_lut,
            shadow_coverage,
            plan,
            #[cfg(test)]
            target_kind,
        }
    }

    pub fn scratch_bytes(&self, extent: RenderExtent) -> u64 {
        self.plan.scratch_bytes(extent)
    }

    pub(crate) const fn has_escape_map(&self) -> bool {
        self.escape_map_node_pipeline.is_some()
    }

    pub fn create_target(&self, device: &wgpu::Device, extent: RenderExtent) -> TraceImage {
        let texture_usage = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-linear HDR trace target"),
            size: wgpu::Extent3d {
                width: extent.width(),
                height: extent.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: texture_usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        #[cfg(test)]
        let record_planes = self.target_kind.captures_records().then(|| {
            let size = trace_record_plane_size(extent);
            DiagnosticPlanes {
                source_time: create_record_plane(device, "trace source and time", size),
                invariant_drift: create_record_plane(device, "trace invariant drift", size),
                metadata: create_record_plane(device, "trace metadata", size),
                event: create_record_plane(device, "trace event candidates", size),
            }
        });
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.dispatch.as_entire_binding(),
            },
        ];
        #[cfg(test)]
        let capture_entries = record_planes.as_ref().map(|planes| {
            planes
                .buffers()
                .into_iter()
                .enumerate()
                .map(|(index, buffer)| wgpu::BindGroupEntry {
                    binding: u32::try_from(index).expect("diagnostic binding index fits u32") + 3,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>()
        });
        #[cfg(not(test))]
        let capture_entries: Option<Vec<wgpu::BindGroupEntry<'_>>> = None;
        let escape_map = self.escape_map_node_pipeline.as_ref().map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("packed escape-direction map"),
                size: escape_map_scratch_bytes(extent),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let escape_map_entry = escape_map.as_ref().map(|buffer| wgpu::BindGroupEntry {
            binding: 7,
            resource: buffer.as_entire_binding(),
        });
        let blackbody_lut_entry = self
            .blackbody_lut
            .as_ref()
            .map(|buffer| wgpu::BindGroupEntry {
                binding: 8,
                resource: buffer.as_entire_binding(),
            });
        let entries = entries
            .into_iter()
            .chain(capture_entries.into_iter().flatten())
            .chain(escape_map_entry)
            .chain(blackbody_lut_entry)
            .collect::<Vec<_>>();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU trace bind group"),
            layout: &self.bind_group_layout,
            entries: &entries,
        });
        let shadow_coverage = self
            .shadow_coverage
            .as_ref()
            .map(|coverage| coverage.create_target(device, &view, extent));
        TraceImage {
            view,
            #[cfg(test)]
            record_planes,
            bind_group,
            shadow_coverage,
        }
    }

    #[cfg(test)]
    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceImage,
        tiles: TileRegion,
    ) {
        self.encode_batch(
            queue,
            encoder,
            target,
            tiles,
            TraceBatchOptions::untimed(true),
        );
    }

    #[cfg(test)]
    pub fn encode_base(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceImage,
        tiles: TileRegion,
    ) {
        self.encode_batch(
            queue,
            encoder,
            target,
            tiles,
            TraceBatchOptions::untimed(false),
        );
    }

    pub fn encode_batch(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceImage,
        tiles: TileRegion,
        options: TraceBatchOptions<'_>,
    ) {
        let TraceBatchOptions {
            escape_map_timestamp_writes,
            trace_timestamp_writes,
            refine_final_batch,
        } = options;
        self.set_tile_dispatch(queue, tiles);
        self.encode_escape_map_pass(encoder, target, tiles, escape_map_timestamp_writes);
        self.encode_trace_pass(
            encoder,
            target,
            tiles,
            trace_timestamp_writes,
            refine_final_batch,
        );
    }

    fn encode_escape_map_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceImage,
        tiles: TileRegion,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let Some(node_pipeline) = &self.escape_map_node_pipeline else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("escape-direction map pass"),
            timestamp_writes,
        });
        pass.set_pipeline(node_pipeline);
        pass.set_bind_group(0, &target.bind_group, &[]);
        let [workgroups_x, workgroups_y] = escape_map_node_workgroups(tiles);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    fn encode_trace_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &TraceImage,
        tiles: TileRegion,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
        refine_final_batch: bool,
    ) {
        let refinement = self
            .shadow_coverage
            .as_ref()
            .zip(target.shadow_coverage.as_ref())
            .filter(|(_, shadow)| refine_final_batch && tiles.finishes(shadow.extent));
        if let Some((_, shadow)) = refinement {
            ShadowCoverage::reset_control(encoder, shadow);
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GPU trace pass"),
            timestamp_writes,
        });
        pass.set_bind_group(0, &target.bind_group, &[]);
        pass.set_pipeline(&self.pipeline);
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePass.html#method.dispatch_workgroups
        let [workgroups_x, workgroups_y] = tiles.workgroups();
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        if let Some((coverage, shadow)) = refinement {
            coverage.encode(&mut pass, shadow);
        }
    }

    fn set_tile_dispatch(&self, queue: &wgpu::Queue, tiles: TileRegion) {
        let [origin_x, origin_y] = tiles.origin();
        let [workgroups_x, workgroups_y] = tiles.workgroups();
        let dispatch = TraceDispatch {
            tile_origin: [origin_x, origin_y],
            tile_count: [workgroups_x, workgroups_y],
        };
        // Small queue writes are staged immediately and execute before the following submission.
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html#method.write_buffer
        queue.write_buffer(&self.dispatch, 0, bytemuck::bytes_of(&dispatch));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileRegion {
    origin: [u32; 2],
    workgroups: [u32; 2],
}

impl TileRegion {
    #[cfg(any(test, feature = "gpu-benchmarks"))]
    pub const fn all(extent: RenderExtent) -> Self {
        Self {
            origin: [0, 0],
            workgroups: tile_grid(extent),
        }
    }

    pub const fn new(origin: [u32; 2], workgroups: [u32; 2]) -> Self {
        debug_assert!(workgroups[0] > 0 && workgroups[1] > 0);
        Self { origin, workgroups }
    }

    pub const fn origin(self) -> [u32; 2] {
        self.origin
    }

    pub const fn workgroups(self) -> [u32; 2] {
        self.workgroups
    }

    pub const fn len(self) -> u32 {
        self.workgroups[0] * self.workgroups[1]
    }

    pub const fn finishes(self, extent: RenderExtent) -> bool {
        let grid = tile_grid(extent);
        let row_contiguous =
            self.workgroups[1] == 1 || (self.origin[0] == 0 && self.workgroups[0] == grid[0]);
        row_contiguous
            && self.origin[0] + self.workgroups[0] == grid[0]
            && self.origin[1] + self.workgroups[1] == grid[1]
    }
}

pub const fn tile_grid(extent: RenderExtent) -> [u32; 2] {
    [
        extent.width().div_ceil(TRACE_WORKGROUP_WIDTH),
        extent.height().div_ceil(TRACE_WORKGROUP_HEIGHT),
    ]
}

const fn escape_map_node_workgroups(tiles: TileRegion) -> [u32; 2] {
    let [tile_columns, tile_rows] = tiles.workgroups();
    [
        (tile_columns * 2 + 1).div_ceil(TRACE_WORKGROUP_WIDTH),
        (tile_rows * 2 + 1).div_ceil(TRACE_WORKGROUP_HEIGHT),
    ]
}

pub fn escape_map_scratch_bytes(extent: RenderExtent) -> u64 {
    let columns = u64::from(extent.width().div_ceil(TRACE_WORKGROUP_WIDTH)) * 2 + 1;
    let rows = u64::from(extent.height().div_ceil(TRACE_WORKGROUP_HEIGHT)) * 2 + 1;
    columns
        .saturating_mul(rows)
        .saturating_mul(size_of::<u32>())
}

fn accelerated_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!(
        "{KERR_SCHILD_TRACE_SHADER}\n{LENSING_PREVIEW_SHADER}\n{GEODESIC_ACCELERATION_SHADER}"
    ))
}

fn bolometric_surface_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!(
        "{KERR_SCHILD_TRACE_SHADER}\n{LENSING_PREVIEW_SHADER}\n{SURFACE_TRANSPORT_SHADER}\n{SURFACE_PREVIEW_SHADER}"
    ))
}

fn spectral_surface_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!(
        "{KERR_SCHILD_TRACE_SHADER}\n{LENSING_PREVIEW_SHADER}\n{SURFACE_TRANSPORT_SHADER}\n{SPECTRAL_SURFACE_PREVIEW_SHADER}"
    ))
}

#[cfg(test)]
fn trace_capture_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!(
        "{KERR_SCHILD_TRACE_SHADER}\n{LENSING_PREVIEW_SHADER}\n{TRACE_CAPTURE_SHADER}"
    ))
}

#[cfg(test)]
fn bolometric_surface_capture_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!(
        "{KERR_SCHILD_TRACE_SHADER}\n{LENSING_PREVIEW_SHADER}\n{TRACE_CAPTURE_SHADER}\n{SURFACE_TRANSPORT_SHADER}\n{SURFACE_PREVIEW_SHADER}\n{SURFACE_TRACE_CAPTURE_SHADER}"
    ))
}

#[cfg(test)]
fn spectral_surface_capture_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!(
        "{KERR_SCHILD_TRACE_SHADER}\n{LENSING_PREVIEW_SHADER}\n{TRACE_CAPTURE_SHADER}\n{SURFACE_TRANSPORT_SHADER}\n{SPECTRAL_SURFACE_PREVIEW_SHADER}\n{SURFACE_TRACE_CAPTURE_SHADER}"
    ))
}

#[cfg(test)]
fn surface_footprint_capture_shader_source(plan: TracePlan) -> Cow<'static, str> {
    let preview = match plan {
        TracePlan::EquatorialBolometricSurface => SURFACE_PREVIEW_SHADER,
        TracePlan::EquatorialBlackbodySurface => SPECTRAL_SURFACE_PREVIEW_SHADER,
        TracePlan::AcceleratedSky => "",
    };
    Cow::Owned(format!(
        "{KERR_SCHILD_TRACE_SHADER}\n{LENSING_PREVIEW_SHADER}\n{TRACE_CAPTURE_SHADER}\n{SURFACE_TRANSPORT_SHADER}\n{preview}\n{SURFACE_FOOTPRINT_CAPTURE_SHADER}"
    ))
}

#[cfg(test)]
fn accelerated_capture_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!(
        "{KERR_SCHILD_TRACE_SHADER}\n{LENSING_PREVIEW_SHADER}\n{TRACE_CAPTURE_SHADER}\n{GEODESIC_ACCELERATION_SHADER}\n{ACCELERATED_TRACE_CAPTURE_SHADER}"
    ))
}

pub fn shadow_coverage_scratch_bytes(extent: RenderExtent) -> u64 {
    crate::shadow_coverage::scratch_bytes(extent)
}

pub struct TraceImage {
    view: wgpu::TextureView,
    #[cfg(test)]
    record_planes: Option<DiagnosticPlanes>,
    bind_group: wgpu::BindGroup,
    shadow_coverage: Option<ShadowTarget>,
}

impl TraceImage {
    pub const fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    #[cfg(test)]
    pub fn texture(&self) -> &wgpu::Texture {
        self.view.texture()
    }

    #[cfg(test)]
    pub const fn record_planes(&self) -> [&wgpu::Buffer; 4] {
        self.record_planes
            .as_ref()
            .expect("capture targets include diagnostic record planes")
            .buffers()
    }

    #[cfg(test)]
    pub const fn shadow_control(&self) -> &wgpu::Buffer {
        &self
            .shadow_coverage
            .as_ref()
            .expect("refined capture target contains shadow control")
            .control
    }
}

#[cfg(test)]
struct DiagnosticPlanes {
    source_time: wgpu::Buffer,
    invariant_drift: wgpu::Buffer,
    metadata: wgpu::Buffer,
    event: wgpu::Buffer,
}

#[cfg(test)]
impl DiagnosticPlanes {
    const fn buffers(&self) -> [&wgpu::Buffer; 4] {
        [
            &self.source_time,
            &self.invariant_drift,
            &self.metadata,
            &self.event,
        ]
    }
}

#[cfg(test)]
fn create_record_plane(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
pub fn trace_record_plane_size(extent: RenderExtent) -> u64 {
    u64::from(extent.width())
        .saturating_mul(u64::from(extent.height()))
        .saturating_mul(TRACE_RECORD_PLANE_ELEMENT_SIZE)
}

pub const fn size_of<T>() -> u64 {
    std::mem::size_of::<T>() as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TraceTermination {
    HorizonCrossing = 1,
    Escape = 2,
    SingularityGuard = 3,
    StepExhaustion = 4,
    NumericalFailure = 5,
    Uncertain = 6,
    EquatorialSurface = 7,
}

impl From<TraceTermination> for u32 {
    fn from(value: TraceTermination) -> Self {
        value as Self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown GPU trace termination discriminant {0}")]
pub struct UnknownTraceTermination(pub u32);

impl TryFrom<u32> for TraceTermination {
    type Error = UnknownTraceTermination;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::HorizonCrossing),
            2 => Ok(Self::Escape),
            3 => Ok(Self::SingularityGuard),
            4 => Ok(Self::StepExhaustion),
            5 => Ok(Self::NumericalFailure),
            6 => Ok(Self::Uncertain),
            7 => Ok(Self::EquatorialSurface),
            unknown => Err(UnknownTraceTermination(unknown)),
        }
    }
}

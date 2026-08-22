use gravlume_domain::ImageSample;
use num_traits::ToPrimitive as _;
use wgpu::util::DeviceExt as _;

use super::{TracePipeline, TracePlan, TraceTermination, UnknownTraceTermination, shader, size_of};
use crate::{
    extent::RenderExtent, scientific_capture::ScientificChannelModel, test_device::TestGpu,
};

const INSPECTION_REQUEST_BYTES: u64 = size_of::<GpuInspectionRequest>();
const INSPECTION_RECORD_BYTES: u64 = size_of::<GpuInspectionRecord>();
const KNOWN_NUMERICAL_FLAGS: u32 = 1 | 2 | 4;
const KNOWN_EVENT_CANDIDATES: u32 = 1 | 2 | 4 | 8;

const HORIZON_TAG: u32 = 0.0_f32.to_bits();
const ANALYTIC_ESCAPE_TAG: u32 = 1.0_f32.to_bits();
const SURFACE_RADIANCE_TAG: u32 = 2.0_f32.to_bits();
const SINGULARITY_FAILURE_TAG: u32 = (-3.0_f32).to_bits();
const STEP_EXHAUSTION_FAILURE_TAG: u32 = (-4.0_f32).to_bits();
const NUMERICAL_FAILURE_TAG: u32 = (-5.0_f32).to_bits();
const UNCERTAIN_FAILURE_TAG: u32 = (-6.0_f32).to_bits();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplePolarSide {
    Negative,
    Equatorial,
    Positive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleBranchKey {
    pub initial_polar_side: SamplePolarSide,
    pub radial_turnings: u32,
    pub equatorial_crossings: u32,
    pub azimuth_winding: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SampleInspectionSource {
    None,
    /// A normalized orientation used by the analytic sky preview, not a physical spectrum.
    AnalyticEscape {
        unit_direction: [f32; 3],
    },
    EquatorialSurface {
        radius_over_m: f32,
        azimuth_radians: f32,
        frequency_ratio: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SampleSceneValue {
    Horizon,
    /// Scene-linear orientation preview; these channels are not spectral radiance.
    AnalyticEscapePreview([f32; 3]),
    SurfaceRadiance([f32; 3]),
    TraceFailure {
        termination: TraceTermination,
        visible_rgb: [f32; 3],
    },
}

/// Test-only evidence from one fresh production-profile GPU trace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleInspection {
    pub channel_model: Option<ScientificChannelModel>,
    pub termination: TraceTermination,
    pub source: SampleInspectionSource,
    pub scene_value: SampleSceneValue,
    pub branch_key: Option<SampleBranchKey>,
    pub travel_time_over_m: f32,
    pub event_candidates: u32,
    pub event_residual: f32,
    pub steps: u32,
    pub numerical_flags: u32,
    pub maximum_invariant_drift: [f32; 4],
}

#[derive(Debug, thiserror::Error)]
enum SampleInspectionError {
    #[error(transparent)]
    UnknownTermination(#[from] UnknownTraceTermination),
    #[error("unknown GPU sample initial polar-side discriminant {0}")]
    UnknownPolarSide(u32),
    #[error("GPU sample inspection returned an invalid {field} record")]
    InvalidRecord { field: &'static str },
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
struct GpuInspectionRequest {
    pixel_extent: [u32; 4],
    subpixel: [f32; 4],
}

impl GpuInspectionRequest {
    fn new(sample: ImageSample, extent: RenderExtent) -> Self {
        let [pixel_x, pixel_y] = sample.pixel();
        let [subpixel_x, subpixel_y] = sample.subpixel().map(|coordinate| {
            coordinate
                .to_f32()
                .expect("validated subpixel fits binary32")
        });
        Self {
            pixel_extent: [pixel_x, pixel_y, extent.width(), extent.height()],
            subpixel: [subpixel_x, subpixel_y, 0.0, 0.0],
        }
    }
}

const _: () = assert!(std::mem::size_of::<GpuInspectionRequest>() == 32);
const _: () = assert!(std::mem::align_of::<GpuInspectionRequest>() == 16);

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
struct GpuInspectionRecord {
    // (termination, failure flags, steps, event candidates)
    metadata: [u32; 4],
    branch_key: [u32; 4],
    source_time: [f32; 4],
    scene_value: [f32; 4],
    // (event residual, reserved, reserved, reserved)
    event_diagnostics: [f32; 4],
    maximum_invariant_drift: [f32; 4],
}

// Six vec4 lanes give the storage record an exact, padding-free 16-byte layout.
// Source: https://www.w3.org/TR/WGSL/#alignment-and-size
const _: () = assert!(std::mem::size_of::<GpuInspectionRecord>() == 96);
const _: () = assert!(std::mem::align_of::<GpuInspectionRecord>() == 16);

impl TracePipeline {
    pub(crate) fn inspect_sample(
        &self,
        gpu: &TestGpu,
        extent: RenderExtent,
        sample: ImageSample,
    ) -> SampleInspection {
        let request = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sample inspection request"),
                contents: bytemuck::bytes_of(&GpuInspectionRequest::new(sample, extent)),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let record = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sample inspection record"),
            size: INSPECTION_RECORD_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sample inspection readback"),
            size: INSPECTION_RECORD_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let pipeline = create_inspection_pipeline(&gpu.device, self);
        let bind_group =
            create_inspection_bind_group(&gpu.device, self, &pipeline, &request, &record);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sample inspection encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sample inspection pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&record, 0, &readback, 0, INSPECTION_RECORD_BYTES);
        let submission = gpu.queue.submit([encoder.finish()]);
        let bytes = gpu.read_buffer(&readback, submission);
        let raw = bytemuck::pod_read_unaligned(&bytes);
        decode_record(
            raw,
            self.scientific_capture_metadata()
                .map(crate::scientific_capture::ScientificCaptureMetadata::channels),
        )
        .expect("GPU sample inspection record is valid")
    }
}

fn create_inspection_pipeline(
    device: &wgpu::Device,
    trace: &TracePipeline,
) -> wgpu::ComputePipeline {
    let shader_source = match trace.plan {
        TracePlan::AcceleratedSky => shader::analytic_sample_inspection(),
        TracePlan::EquatorialBolometricSurface => shader::bolometric_sample_inspection(),
        TracePlan::EquatorialBlackbodySurface => shader::blackbody_sample_inspection(),
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bounded sample inspection shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline_constants = [(
        "SURFACE_EVENTS_ENABLED",
        trace.plan.surface_events_enabled(),
    )];
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("inspect_sample"),
        // This one-off test pipeline never shares bind groups with another pipeline. Let wgpu
        // derive the layout, then create the sole bind group from that exact layout.
        // Source: https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePipelineDescriptor.html#structfield.layout
        layout: None,
        module: &shader,
        entry_point: Some("inspect_sample"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &pipeline_constants,
            ..Default::default()
        },
        cache: None,
    })
}

fn create_inspection_bind_group(
    device: &wgpu::Device,
    trace: &TracePipeline,
    pipeline: &wgpu::ComputePipeline,
    request: &wgpu::Buffer,
    record: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(0);
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: trace.uniforms.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: record.as_entire_binding(),
        },
    ];
    if let Some(blackbody_lut) = &trace.blackbody_lut {
        entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: blackbody_lut.as_entire_binding(),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: 9,
        resource: request.as_entire_binding(),
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sample inspection bind group"),
        layout: &layout,
        entries: &entries,
    })
}

fn decode_record(
    raw: GpuInspectionRecord,
    channel_model: Option<ScientificChannelModel>,
) -> Result<SampleInspection, SampleInspectionError> {
    if !raw
        .source_time
        .iter()
        .chain(&raw.scene_value)
        .chain(&raw.event_diagnostics)
        .chain(&raw.maximum_invariant_drift)
        .all(|value| value.is_finite())
    {
        return Err(SampleInspectionError::InvalidRecord {
            field: "non-finite value",
        });
    }
    if raw.event_diagnostics[1..] != [0.0; 3] {
        return Err(SampleInspectionError::InvalidRecord {
            field: "event diagnostics",
        });
    }

    let termination = TraceTermination::try_from(raw.metadata[0])?;
    let numerical_flags = raw.metadata[1];
    if numerical_flags & !KNOWN_NUMERICAL_FLAGS != 0 {
        return Err(SampleInspectionError::InvalidRecord {
            field: "failure flags",
        });
    }
    let event_candidates = raw.metadata[3];
    if event_candidates & !KNOWN_EVENT_CANDIDATES != 0 {
        return Err(SampleInspectionError::InvalidRecord {
            field: "event candidates",
        });
    }

    Ok(SampleInspection {
        channel_model,
        termination,
        source: decode_source(termination, raw.source_time)?,
        scene_value: decode_scene_value(termination, raw.scene_value)?,
        branch_key: decode_branch_key(termination, raw.branch_key)?,
        travel_time_over_m: raw.source_time[3],
        event_candidates,
        event_residual: raw.event_diagnostics[0],
        steps: raw.metadata[2],
        numerical_flags,
        maximum_invariant_drift: raw.maximum_invariant_drift,
    })
}

fn decode_source(
    termination: TraceTermination,
    source_time: [f32; 4],
) -> Result<SampleInspectionSource, SampleInspectionError> {
    match termination {
        TraceTermination::Escape => Ok(SampleInspectionSource::AnalyticEscape {
            unit_direction: source_time[..3].try_into().map_err(|_| {
                SampleInspectionError::InvalidRecord {
                    field: "escape source",
                }
            })?,
        }),
        TraceTermination::EquatorialSurface => {
            if source_time[0] <= 0.0 || source_time[2] <= 0.0 {
                return Err(SampleInspectionError::InvalidRecord {
                    field: "surface source",
                });
            }
            Ok(SampleInspectionSource::EquatorialSurface {
                radius_over_m: source_time[0],
                azimuth_radians: source_time[1],
                frequency_ratio: source_time[2],
            })
        }
        TraceTermination::HorizonCrossing
        | TraceTermination::SingularityGuard
        | TraceTermination::StepExhaustion
        | TraceTermination::NumericalFailure
        | TraceTermination::Uncertain => Ok(SampleInspectionSource::None),
    }
}

fn decode_branch_key(
    termination: TraceTermination,
    words: [u32; 4],
) -> Result<Option<SampleBranchKey>, SampleInspectionError> {
    if termination == TraceTermination::NumericalFailure {
        if words != [0; 4] {
            return Err(SampleInspectionError::InvalidRecord {
                field: "numerical-failure branch",
            });
        }
        return Ok(None);
    }

    let initial_polar_side = match words[3] {
        0 => SamplePolarSide::Negative,
        1 => SamplePolarSide::Equatorial,
        2 => SamplePolarSide::Positive,
        unknown => return Err(SampleInspectionError::UnknownPolarSide(unknown)),
    };
    if termination == TraceTermination::Uncertain {
        return Ok(None);
    }
    Ok(Some(SampleBranchKey {
        initial_polar_side,
        radial_turnings: words[0],
        equatorial_crossings: words[1],
        azimuth_winding: i32::from_ne_bytes(words[2].to_ne_bytes()),
    }))
}

fn decode_scene_value(
    termination: TraceTermination,
    value: [f32; 4],
) -> Result<SampleSceneValue, SampleInspectionError> {
    let rgb: [f32; 3] =
        value[..3]
            .try_into()
            .map_err(|_| SampleInspectionError::InvalidRecord {
                field: "scene value",
            })?;
    let tag = value[3].to_bits();

    if termination == TraceTermination::HorizonCrossing && tag == HORIZON_TAG {
        if rgb.map(f32::to_bits) != [0; 3] {
            return Err(SampleInspectionError::InvalidRecord {
                field: "horizon scene value",
            });
        }
        return Ok(SampleSceneValue::Horizon);
    }
    if termination == TraceTermination::Escape && tag == ANALYTIC_ESCAPE_TAG {
        return Ok(SampleSceneValue::AnalyticEscapePreview(rgb));
    }
    if termination == TraceTermination::EquatorialSurface && tag == SURFACE_RADIANCE_TAG {
        if rgb.into_iter().any(|channel| channel < 0.0) {
            return Err(SampleInspectionError::InvalidRecord {
                field: "surface radiance",
            });
        }
        return Ok(SampleSceneValue::SurfaceRadiance(rgb));
    }

    let visible_termination = match tag {
        SINGULARITY_FAILURE_TAG => TraceTermination::SingularityGuard,
        STEP_EXHAUSTION_FAILURE_TAG => TraceTermination::StepExhaustion,
        NUMERICAL_FAILURE_TAG => TraceTermination::NumericalFailure,
        UNCERTAIN_FAILURE_TAG => TraceTermination::Uncertain,
        _ => {
            return Err(SampleInspectionError::InvalidRecord { field: "scene tag" });
        }
    };
    let valid_failure = match termination {
        TraceTermination::EquatorialSurface => {
            visible_termination == TraceTermination::NumericalFailure
        }
        TraceTermination::SingularityGuard
        | TraceTermination::StepExhaustion
        | TraceTermination::NumericalFailure
        | TraceTermination::Uncertain => visible_termination == termination,
        TraceTermination::HorizonCrossing | TraceTermination::Escape => false,
    };
    if !valid_failure {
        return Err(SampleInspectionError::InvalidRecord {
            field: "termination/scene tag",
        });
    }
    Ok(SampleSceneValue::TraceFailure {
        termination: visible_termination,
        visible_rgb: rgb,
    })
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of};

    use super::{
        GpuInspectionRecord, GpuInspectionRequest, INSPECTION_RECORD_BYTES,
        INSPECTION_REQUEST_BYTES, SampleInspectionError, SamplePolarSide, decode_branch_key,
    };
    use crate::trace::TraceTermination;

    #[test]
    fn host_inspection_abi_is_a_sequence_of_aligned_vec4_lanes() {
        assert_eq!(INSPECTION_REQUEST_BYTES, 32);
        assert_eq!(INSPECTION_RECORD_BYTES, 96);
        assert_eq!(INSPECTION_REQUEST_BYTES + 2 * INSPECTION_RECORD_BYTES, 224);
        assert_eq!(align_of::<GpuInspectionRequest>(), 16);
        assert_eq!(align_of::<GpuInspectionRecord>(), 16);
        assert_eq!(offset_of!(GpuInspectionRequest, pixel_extent), 0);
        assert_eq!(offset_of!(GpuInspectionRequest, subpixel), 16);
        assert_eq!(offset_of!(GpuInspectionRecord, metadata), 0);
        assert_eq!(offset_of!(GpuInspectionRecord, branch_key), 16);
        assert_eq!(offset_of!(GpuInspectionRecord, source_time), 32);
        assert_eq!(offset_of!(GpuInspectionRecord, scene_value), 48);
        assert_eq!(offset_of!(GpuInspectionRecord, event_diagnostics), 64);
        assert_eq!(offset_of!(GpuInspectionRecord, maximum_invariant_drift), 80);
    }

    #[test]
    fn branch_decoder_preserves_exact_values_and_rejects_failure_placeholders() {
        let winding = -123_456_i32;
        let words = [
            0x1_0000,
            u32::MAX,
            u32::from_ne_bytes(winding.to_ne_bytes()),
            2,
        ];
        let branch = decode_branch_key(TraceTermination::StepExhaustion, words)
            .expect("known branch decodes")
            .expect("step exhaustion retains the committed branch");
        assert_eq!(branch.radial_turnings, 0x1_0000);
        assert_eq!(branch.equatorial_crossings, u32::MAX);
        assert_eq!(branch.azimuth_winding, winding);
        assert_eq!(branch.initial_polar_side, SamplePolarSide::Positive);

        assert_eq!(
            decode_branch_key(TraceTermination::NumericalFailure, [0; 4])
                .expect("zero failure sentinel decodes"),
            None
        );
        assert!(matches!(
            decode_branch_key(TraceTermination::NumericalFailure, words),
            Err(SampleInspectionError::InvalidRecord {
                field: "numerical-failure branch"
            })
        ));
        assert_eq!(
            decode_branch_key(TraceTermination::Uncertain, words)
                .expect("provisional uncertain branch is recognized"),
            None
        );
    }
}

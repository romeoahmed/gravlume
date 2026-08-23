use gravlume_domain::ImageSample;
use num_traits::ToPrimitive as _;
use wgpu::util::DeviceExt as _;

use super::{TracePipeline, TracePlan, TraceTermination, UnknownTraceTermination, shader, size_of};
use crate::{
    extent::RenderExtent, scientific_capture::ScientificChannelModel, test_device::TestGpu,
};

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

const _: () = {
    assert!(std::mem::size_of::<GpuInspectionRequest>() == 32);
    assert!(std::mem::align_of::<GpuInspectionRequest>() == 16);
    assert!(std::mem::offset_of!(GpuInspectionRequest, pixel_extent) == 0);
    assert!(std::mem::offset_of!(GpuInspectionRequest, subpixel) == 16);
};

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
const _: () = {
    assert!(std::mem::size_of::<GpuInspectionRecord>() == 96);
    assert!(std::mem::align_of::<GpuInspectionRecord>() == 16);
    assert!(std::mem::offset_of!(GpuInspectionRecord, metadata) == 0);
    assert!(std::mem::offset_of!(GpuInspectionRecord, branch_key) == 16);
    assert!(std::mem::offset_of!(GpuInspectionRecord, source_time) == 32);
    assert!(std::mem::offset_of!(GpuInspectionRecord, scene_value) == 48);
    assert!(std::mem::offset_of!(GpuInspectionRecord, event_diagnostics) == 64);
    assert!(std::mem::offset_of!(GpuInspectionRecord, maximum_invariant_drift) == 80);
};

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
    let [source_x, source_y, source_z, _] = source_time;
    match termination {
        TraceTermination::Escape => Ok(SampleInspectionSource::AnalyticEscape {
            unit_direction: [source_x, source_y, source_z],
        }),
        TraceTermination::EquatorialSurface => {
            if source_x <= 0.0 || source_z <= 0.0 {
                return Err(SampleInspectionError::InvalidRecord {
                    field: "surface source",
                });
            }
            Ok(SampleInspectionSource::EquatorialSurface {
                radius_over_m: source_x,
                azimuth_radians: source_y,
                frequency_ratio: source_z,
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
    let [red, green, blue, alpha] = value;
    let rgb = [red, green, blue];
    let tag = alpha.to_bits();

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
    use proptest::prelude::*;

    use super::{SampleBranchKey, SampleInspectionError, SamplePolarSide, decode_branch_key};
    use crate::trace::TraceTermination;

    fn branch_counter() -> impl Strategy<Value = u32> {
        prop_oneof![Just(0), Just(u32::MAX), any::<u32>()]
    }

    fn branch_winding() -> impl Strategy<Value = i32> {
        prop_oneof![Just(i32::MIN), Just(0), Just(i32::MAX), any::<i32>()]
    }

    #[test]
    fn numerical_failure_uses_an_explicit_zero_branch_sentinel() {
        assert_eq!(
            decode_branch_key(TraceTermination::NumericalFailure, [0; 4])
                .expect("zero failure sentinel decodes"),
            None
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn branch_decoder_preserves_arbitrary_committed_values(
            radial_turnings in branch_counter(),
            equatorial_crossings in branch_counter(),
            azimuth_winding in branch_winding(),
        ) {
            let polar_sides = [
                (0, SamplePolarSide::Negative),
                (1, SamplePolarSide::Equatorial),
                (2, SamplePolarSide::Positive),
            ];
            let committed_terminations = [
                TraceTermination::HorizonCrossing,
                TraceTermination::Escape,
                TraceTermination::SingularityGuard,
                TraceTermination::StepExhaustion,
                TraceTermination::EquatorialSurface,
            ];

            for (polar_side_word, initial_polar_side) in polar_sides {
                let words = [
                    radial_turnings,
                    equatorial_crossings,
                    u32::from_ne_bytes(azimuth_winding.to_ne_bytes()),
                    polar_side_word,
                ];
                for termination in committed_terminations {
                    let branch = decode_branch_key(termination, words)
                        .expect("known branch decodes")
                        .expect("committed termination retains its branch");
                    prop_assert_eq!(branch, SampleBranchKey {
                        initial_polar_side,
                        radial_turnings,
                        equatorial_crossings,
                        azimuth_winding,
                    });
                }
                prop_assert_eq!(
                    decode_branch_key(TraceTermination::Uncertain, words)
                        .expect("provisional uncertain branch is recognized"),
                    None
                );
            }
        }

        #[test]
        fn branch_decoder_rejects_invalid_protocol_words(
            payload in (any::<[u32; 4]>(), 0_usize..4),
            radial_turnings: u32,
            equatorial_crossings: u32,
            azimuth_winding: i32,
            unknown_side in 3_u32..=u32::MAX,
        ) {
            let (mut words, nonzero_index) = payload;
            words[nonzero_index] |= 1;

            prop_assert!(matches!(
                decode_branch_key(TraceTermination::NumericalFailure, words),
                Err(SampleInspectionError::InvalidRecord {
                    field: "numerical-failure branch"
                })
            ), "nonzero branch words {words:?} must not decode as a failure sentinel");
            let words = [
                radial_turnings,
                equatorial_crossings,
                u32::from_ne_bytes(azimuth_winding.to_ne_bytes()),
                unknown_side,
            ];

            prop_assert!(matches!(
                decode_branch_key(TraceTermination::StepExhaustion, words),
                Err(SampleInspectionError::UnknownPolarSide(value)) if value == unknown_side
            ));
        }
    }
}

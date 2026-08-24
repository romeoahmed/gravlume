use std::mem::size_of;

use gravlume_domain::ImageSample;

use super::{
    SampleBranchKey, SampleInspection, SampleInspectionError, SampleInspectionTicket,
    SamplePolarSide, SampleRetrace, SampleSurfaceEvaluation, SampleTraceDiagnostics,
    SampleTraceOutcome,
};
use crate::{
    extent::RenderExtent,
    scientific_capture::{ScientificChannelModel, ScientificTexel},
};

pub(super) const INSPECTION_REQUEST_BYTES: u64 = size_of::<GpuInspectionRequest>() as u64;
pub(super) const INSPECTION_RECORD_BYTES: u64 = size_of::<GpuInspectionRecord>() as u64;
pub(super) const PUBLISHED_TEXEL_OFFSET: u64 = INSPECTION_RECORD_BYTES;
const PUBLISHED_TEXEL_BYTES: u64 = size_of::<[u16; 4]>() as u64;
pub(super) const INSPECTION_READBACK_BYTES: u64 = PUBLISHED_TEXEL_OFFSET + PUBLISHED_TEXEL_BYTES;

const _: () = {
    assert!(INSPECTION_REQUEST_BYTES == 32);
    assert!(INSPECTION_RECORD_BYTES == 96);
    assert!(INSPECTION_READBACK_BYTES == 104);
    assert!(INSPECTION_REQUEST_BYTES + INSPECTION_RECORD_BYTES + INSPECTION_READBACK_BYTES == 232);
    assert!(PUBLISHED_TEXEL_OFFSET.is_multiple_of(PUBLISHED_TEXEL_BYTES));
};

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

impl TryFrom<u32> for TraceTermination {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::HorizonCrossing),
            2 => Ok(Self::Escape),
            3 => Ok(Self::SingularityGuard),
            4 => Ok(Self::StepExhaustion),
            5 => Ok(Self::NumericalFailure),
            6 => Ok(Self::Uncertain),
            7 => Ok(Self::EquatorialSurface),
            unknown => Err(unknown),
        }
    }
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub(super) struct GpuInspectionRequest {
    pixel_extent: [u32; 4],
    subpixel: [f32; 4],
}

impl GpuInspectionRequest {
    pub(super) const fn new(sample: ImageSample, extent: RenderExtent) -> Self {
        let [pixel_x, pixel_y] = sample.pixel();
        let [subpixel_x, subpixel_y] = binary32_subpixel(sample);
        Self {
            pixel_extent: [pixel_x, pixel_y, extent.width(), extent.height()],
            subpixel: [subpixel_x, subpixel_y, 0.0, 0.0],
        }
    }
}

const fn binary32_subpixel(sample: ImageSample) -> [f32; 2] {
    let [subpixel_x, subpixel_y] = sample.subpixel();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "ImageSample guarantees finite subpixel coordinates in [0, 1], so binary32 rounding is total"
    )]
    {
        [subpixel_x as f32, subpixel_y as f32]
    }
}

const _: () = {
    assert!(size_of::<GpuInspectionRequest>() == 32);
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
    assert!(size_of::<GpuInspectionRecord>() == 96);
    assert!(std::mem::align_of::<GpuInspectionRecord>() == 16);
    assert!(std::mem::offset_of!(GpuInspectionRecord, metadata) == 0);
    assert!(std::mem::offset_of!(GpuInspectionRecord, branch_key) == 16);
    assert!(std::mem::offset_of!(GpuInspectionRecord, source_time) == 32);
    assert!(std::mem::offset_of!(GpuInspectionRecord, scene_value) == 48);
    assert!(std::mem::offset_of!(GpuInspectionRecord, event_diagnostics) == 64);
    assert!(std::mem::offset_of!(GpuInspectionRecord, maximum_invariant_drift) == 80);
};

pub(super) fn decode_readback(
    bytes: &[u8],
    channel_model: Option<ScientificChannelModel>,
    ticket: SampleInspectionTicket,
) -> Result<SampleInspection, SampleInspectionError> {
    let record_bytes = bytes
        .get(..size_of::<GpuInspectionRecord>())
        .ok_or(SampleInspectionError::InvalidReadback)?;
    let record = bytemuck::try_pod_read_unaligned(record_bytes)
        .map_err(|_| SampleInspectionError::InvalidReadback)?;
    let texel_start = usize::try_from(PUBLISHED_TEXEL_OFFSET)
        .map_err(|_| SampleInspectionError::InvalidReadback)?;
    let texel_end = usize::try_from(PUBLISHED_TEXEL_OFFSET + PUBLISHED_TEXEL_BYTES)
        .map_err(|_| SampleInspectionError::InvalidReadback)?;
    let texel_bytes = bytes
        .get(texel_start..texel_end)
        .ok_or(SampleInspectionError::InvalidReadback)?;
    let rgba16_float_bits = std::array::from_fn(|channel| {
        let offset = channel * size_of::<u16>();
        u16::from_le_bytes([texel_bytes[offset], texel_bytes[offset + 1]])
    });
    decode_record(
        record,
        channel_model,
        ticket,
        ScientificTexel::from_rgba16_float_bits(rgba16_float_bits),
    )
}

fn decode_record(
    raw: GpuInspectionRecord,
    channel_model: Option<ScientificChannelModel>,
    ticket: SampleInspectionTicket,
    published_texel: ScientificTexel,
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

    let termination = TraceTermination::try_from(raw.metadata[0])
        .map_err(SampleInspectionError::UnknownTermination)?;
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

    let outcome = decode_outcome(
        termination,
        decode_branch_key(termination, raw.branch_key)?,
        raw.source_time,
        raw.scene_value,
        channel_model,
    )?;
    Ok(SampleInspection {
        published_texel,
        fresh_retrace: SampleRetrace {
            effective_subpixel: binary32_subpixel(ticket.sample()),
            outcome,
            diagnostics: SampleTraceDiagnostics {
                coordinate_time_delta_over_m: raw.source_time[3],
                event_candidates,
                event_residual: raw.event_diagnostics[0],
                steps: raw.metadata[2],
                numerical_flags,
                maximum_invariant_drift: raw.maximum_invariant_drift,
            },
        },
    })
}

pub(super) fn decode_branch_key(
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

fn decode_outcome(
    termination: TraceTermination,
    branch: Option<SampleBranchKey>,
    source_time: [f32; 4],
    value: [f32; 4],
    channel_model: Option<ScientificChannelModel>,
) -> Result<SampleTraceOutcome, SampleInspectionError> {
    let [source_x, source_y, source_z, _] = source_time;
    let [red, green, blue, alpha] = value;
    let rgb = [red, green, blue];
    let tag = alpha.to_bits();

    match termination {
        TraceTermination::HorizonCrossing => {
            require_scene_tag(tag, HORIZON_TAG)?;
            if rgb.map(f32::to_bits) != [0; 3] {
                return Err(SampleInspectionError::InvalidRecord {
                    field: "horizon scene value",
                });
            }
            Ok(SampleTraceOutcome::Horizon {
                branch: require_branch(branch)?,
            })
        }
        TraceTermination::Escape => {
            require_scene_tag(tag, ANALYTIC_ESCAPE_TAG)?;
            Ok(SampleTraceOutcome::Escape {
                branch: require_branch(branch)?,
                unit_direction: [source_x, source_y, source_z],
                preview_rgb: rgb,
            })
        }
        TraceTermination::EquatorialSurface => {
            if source_x <= 0.0 || source_z <= 0.0 {
                return Err(SampleInspectionError::InvalidRecord {
                    field: "surface source",
                });
            }
            let channels = channel_model.ok_or(SampleInspectionError::InvalidRecord {
                field: "surface channel model",
            })?;
            let evaluation = if tag == SURFACE_RADIANCE_TAG {
                if rgb.into_iter().any(|channel| channel < 0.0) {
                    return Err(SampleInspectionError::InvalidRecord {
                        field: "surface radiance",
                    });
                }
                SampleSurfaceEvaluation::Radiance(rgb)
            } else if tag == NUMERICAL_FAILURE_TAG {
                SampleSurfaceEvaluation::NumericalFailure { visible_rgb: rgb }
            } else {
                return Err(SampleInspectionError::InvalidRecord {
                    field: "termination/scene tag",
                });
            };
            Ok(SampleTraceOutcome::EquatorialSurface {
                branch: require_branch(branch)?,
                radius_over_m: source_x,
                azimuth_radians: source_y,
                frequency_ratio: source_z,
                channels,
                evaluation,
            })
        }
        TraceTermination::SingularityGuard => {
            require_scene_tag(tag, SINGULARITY_FAILURE_TAG)?;
            Ok(SampleTraceOutcome::SingularityGuard {
                branch: require_branch(branch)?,
                visible_rgb: rgb,
            })
        }
        TraceTermination::StepExhaustion => {
            require_scene_tag(tag, STEP_EXHAUSTION_FAILURE_TAG)?;
            Ok(SampleTraceOutcome::StepExhaustion {
                branch_prefix: require_branch(branch)?,
                visible_rgb: rgb,
            })
        }
        TraceTermination::NumericalFailure => {
            require_scene_tag(tag, NUMERICAL_FAILURE_TAG)?;
            Ok(SampleTraceOutcome::NumericalFailure { visible_rgb: rgb })
        }
        TraceTermination::Uncertain => {
            require_scene_tag(tag, UNCERTAIN_FAILURE_TAG)?;
            Ok(SampleTraceOutcome::Uncertain { visible_rgb: rgb })
        }
    }
}

fn require_branch(
    branch: Option<SampleBranchKey>,
) -> Result<SampleBranchKey, SampleInspectionError> {
    branch.ok_or(SampleInspectionError::InvalidRecord {
        field: "missing branch",
    })
}

const fn require_scene_tag(actual: u32, expected: u32) -> Result<(), SampleInspectionError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SampleInspectionError::InvalidRecord {
            field: "termination/scene tag",
        })
    }
}

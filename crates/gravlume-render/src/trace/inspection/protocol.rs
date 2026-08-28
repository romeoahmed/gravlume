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
const EVENT_CANDIDATE_SINGULARITY: u32 = 1;
const EVENT_CANDIDATE_HORIZON: u32 = 2;
const EVENT_CANDIDATE_SURFACE: u32 = 4;
const EVENT_CANDIDATE_ESCAPE: u32 = 8;
const KNOWN_EVENT_CANDIDATES: u32 = EVENT_CANDIDATE_SINGULARITY
    | EVENT_CANDIDATE_HORIZON
    | EVENT_CANDIDATE_SURFACE
    | EVENT_CANDIDATE_ESCAPE;
// WGSL `normalize` inherits the error of `x / sqrt(dot(x, x))`. This protocol sanity tolerance is
// deliberately loose relative to the named GPU corpus while still rejecting scaled or zero data.
// Source: https://www.w3.org/TR/WGSL/#floating-point-accuracy
const ESCAPE_DIRECTION_NORM_SQUARED_TOLERANCE: f64 = 1.0e-4;

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
    if bytes.len()
        != usize::try_from(INSPECTION_READBACK_BYTES)
            .map_err(|_| SampleInspectionError::InvalidReadback)?
    {
        return Err(SampleInspectionError::InvalidReadback);
    }
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
    let fresh_retrace = decode_retrace(record, channel_model, binary32_subpixel(ticket.sample()))?;
    Ok(SampleInspection {
        published_texel: ScientificTexel::from_rgba16_float_bits(rgba16_float_bits),
        fresh_retrace,
    })
}

#[cfg(test)]
pub(super) fn decode_corpus_readback(
    bytes: &[u8],
    channel_model: Option<ScientificChannelModel>,
    samples: &[ImageSample],
) -> Result<Vec<SampleRetrace>, SampleInspectionError> {
    let record_size = size_of::<GpuInspectionRecord>();
    let expected_size = record_size
        .checked_mul(samples.len())
        .ok_or(SampleInspectionError::InvalidReadback)?;
    if bytes.len() != expected_size {
        return Err(SampleInspectionError::InvalidReadback);
    }

    bytes
        .chunks_exact(record_size)
        .zip(samples)
        .map(|(record_bytes, sample)| {
            let record = bytemuck::try_pod_read_unaligned(record_bytes)
                .map_err(|_| SampleInspectionError::InvalidReadback)?;
            decode_retrace(record, channel_model, binary32_subpixel(*sample))
        })
        .collect()
}

fn decode_retrace(
    raw: GpuInspectionRecord,
    channel_model: Option<ScientificChannelModel>,
    effective_subpixel: [f32; 2],
) -> Result<SampleRetrace, SampleInspectionError> {
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
    if raw.source_time[3] < 0.0 {
        return Err(SampleInspectionError::InvalidRecord {
            field: "coordinate time delta",
        });
    }
    if raw
        .maximum_invariant_drift
        .into_iter()
        .any(|value| value < 0.0)
    {
        return Err(SampleInspectionError::InvalidRecord {
            field: "invariant drift",
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
    validate_terminal_diagnostics(termination, numerical_flags, event_candidates)?;

    let outcome = decode_outcome(
        termination,
        decode_branch_key(termination, raw.branch_key)?,
        raw.source_time,
        raw.scene_value,
        channel_model,
    )?;
    Ok(SampleRetrace {
        effective_subpixel,
        outcome,
        diagnostics: SampleTraceDiagnostics {
            coordinate_time_delta_over_m: raw.source_time[3],
            event_candidates,
            event_residual: raw.event_diagnostics[0],
            steps: raw.metadata[2],
            numerical_flags,
            maximum_invariant_drift: raw.maximum_invariant_drift,
        },
    })
}

const fn validate_terminal_diagnostics(
    termination: TraceTermination,
    numerical_flags: u32,
    event_candidates: u32,
) -> Result<(), SampleInspectionError> {
    let valid = match termination {
        TraceTermination::HorizonCrossing => {
            numerical_flags == 0 && event_candidates == EVENT_CANDIDATE_HORIZON
        }
        TraceTermination::Escape => {
            numerical_flags == 0 && event_candidates == EVENT_CANDIDATE_ESCAPE
        }
        TraceTermination::SingularityGuard => {
            numerical_flags == 0 && event_candidates == EVENT_CANDIDATE_SINGULARITY
        }
        TraceTermination::StepExhaustion => numerical_flags == 0 && event_candidates == 0,
        TraceTermination::NumericalFailure => numerical_flags != 0 && event_candidates == 0,
        TraceTermination::Uncertain => numerical_flags == 0 && event_candidates != 0,
        TraceTermination::EquatorialSurface => {
            numerical_flags == 0 && event_candidates == EVENT_CANDIDATE_SURFACE
        }
    };
    if valid {
        Ok(())
    } else {
        Err(SampleInspectionError::InvalidRecord {
            field: "termination diagnostics",
        })
    }
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
            let direction = [source_x, source_y, source_z];
            let norm_squared = direction
                .into_iter()
                .map(f64::from)
                .map(|component| component * component)
                .sum::<f64>();
            if (norm_squared - 1.0).abs() > ESCAPE_DIRECTION_NORM_SQUARED_TOLERANCE {
                return Err(SampleInspectionError::InvalidRecord {
                    field: "escape direction",
                });
            }
            Ok(SampleTraceOutcome::Escape {
                branch: require_branch(branch)?,
                unit_direction: direction,
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

#[cfg(test)]
mod tests {
    use num_traits::ToPrimitive as _;
    use proptest::prelude::*;

    use super::{
        ANALYTIC_ESCAPE_TAG, EVENT_CANDIDATE_ESCAPE, EVENT_CANDIDATE_HORIZON,
        EVENT_CANDIDATE_SINGULARITY, EVENT_CANDIDATE_SURFACE, GpuInspectionRecord,
        KNOWN_EVENT_CANDIDATES, KNOWN_NUMERICAL_FLAGS, SampleInspectionError, SampleTraceOutcome,
        TraceTermination, decode_retrace, validate_terminal_diagnostics,
    };

    fn escape_record(direction: [f32; 3]) -> GpuInspectionRecord {
        GpuInspectionRecord {
            metadata: [
                u32::from(TraceTermination::Escape),
                0,
                1,
                EVENT_CANDIDATE_ESCAPE,
            ],
            branch_key: [0, 0, 0, 1],
            source_time: [direction[0], direction[1], direction[2], 1.0],
            scene_value: [0.25, 0.5, 0.75, f32::from_bits(ANALYTIC_ESCAPE_TAG)],
            event_diagnostics: [0.0; 4],
            maximum_invariant_drift: [0.0; 4],
        }
    }

    #[test]
    fn trace_termination_discriminants_are_stable() {
        let cases = [
            (1, TraceTermination::HorizonCrossing),
            (2, TraceTermination::Escape),
            (3, TraceTermination::SingularityGuard),
            (4, TraceTermination::StepExhaustion),
            (5, TraceTermination::NumericalFailure),
            (6, TraceTermination::Uncertain),
            (7, TraceTermination::EquatorialSurface),
        ];

        for (raw, expected) in cases {
            assert_eq!(u32::from(expected), raw);
            assert_eq!(TraceTermination::try_from(raw), Ok(expected));
        }
    }

    #[test]
    fn terminal_diagnostics_match_the_shader_producer_table() {
        let terminations = [
            TraceTermination::HorizonCrossing,
            TraceTermination::Escape,
            TraceTermination::SingularityGuard,
            TraceTermination::StepExhaustion,
            TraceTermination::NumericalFailure,
            TraceTermination::Uncertain,
            TraceTermination::EquatorialSurface,
        ];

        for termination in terminations {
            for numerical_flags in 0..=KNOWN_NUMERICAL_FLAGS {
                for event_candidates in 0..=KNOWN_EVENT_CANDIDATES {
                    let expected = match termination {
                        TraceTermination::HorizonCrossing => {
                            numerical_flags == 0 && event_candidates == EVENT_CANDIDATE_HORIZON
                        }
                        TraceTermination::Escape => {
                            numerical_flags == 0 && event_candidates == EVENT_CANDIDATE_ESCAPE
                        }
                        TraceTermination::SingularityGuard => {
                            numerical_flags == 0 && event_candidates == EVENT_CANDIDATE_SINGULARITY
                        }
                        TraceTermination::StepExhaustion => {
                            numerical_flags == 0 && event_candidates == 0
                        }
                        TraceTermination::NumericalFailure => {
                            numerical_flags != 0 && event_candidates == 0
                        }
                        TraceTermination::Uncertain => {
                            numerical_flags == 0 && event_candidates != 0
                        }
                        TraceTermination::EquatorialSurface => {
                            numerical_flags == 0 && event_candidates == EVENT_CANDIDATE_SURFACE
                        }
                    };
                    assert_eq!(
                        validate_terminal_diagnostics(
                            termination,
                            numerical_flags,
                            event_candidates,
                        )
                        .is_ok(),
                        expected,
                        "{termination:?}, flags={numerical_flags}, candidates={event_candidates}"
                    );
                }
            }
        }
    }

    #[test]
    fn retrace_decoder_rejects_impossible_escape_diagnostics() {
        for (numerical_flags, event_candidates) in [
            (1, EVENT_CANDIDATE_ESCAPE),
            (0, 0),
            (0, EVENT_CANDIDATE_HORIZON),
        ] {
            let mut record = escape_record([1.0, 0.0, 0.0]);
            record.metadata[1] = numerical_flags;
            record.metadata[3] = event_candidates;
            assert!(matches!(
                decode_retrace(record, None, [0.5; 2]),
                Err(SampleInspectionError::InvalidRecord {
                    field: "termination diagnostics"
                })
            ));
        }
    }

    proptest! {
        #[test]
        fn unknown_trace_termination_discriminants_are_rejected(
            raw in prop_oneof![Just(0), Just(8), Just(u32::MAX), 9_u32..u32::MAX],
        ) {
            prop_assert!(matches!(
                TraceTermination::try_from(raw),
                Err(error) if error == raw
            ));
        }

        #[test]
        fn escape_decoder_accepts_direction_but_rejects_its_arbitrary_scale(
            components in prop::array::uniform3(-1.0_f64..=1.0),
            scale in prop_oneof![0.0_f32..=0.9, 1.1_f32..=4.0],
        ) {
            let norm = components.into_iter().map(|value| value * value).sum::<f64>().sqrt();
            prop_assume!(norm >= 0.25);
            let direction = components.map(|value| {
                (value / norm)
                    .to_f32()
                    .expect("a normalized component is representable in binary32")
            });
            let decoded = decode_retrace(escape_record(direction), None, [0.5; 2])
                .expect("a binary32-normalized direction satisfies the protocol");
            prop_assert!(
                matches!(decoded.outcome, SampleTraceOutcome::Escape { .. }),
                "a normalized direction must decode as Escape",
            );

            let scaled = direction.map(|component| component * scale);
            prop_assert!(
                matches!(
                    decode_retrace(escape_record(scaled), None, [0.5; 2]),
                    Err(SampleInspectionError::InvalidRecord { field: "escape direction" })
                ),
                "an arbitrarily scaled direction must be rejected",
            );
        }

        #[test]
        fn retrace_decoder_rejects_negative_accumulated_diagnostics(
            negative in -1.0e10_f32..=-f32::MIN_POSITIVE,
            drift_index in 0_usize..4,
        ) {
            let mut negative_time = escape_record([1.0, 0.0, 0.0]);
            negative_time.source_time[3] = negative;
            prop_assert!(
                matches!(
                    decode_retrace(negative_time, None, [0.5; 2]),
                    Err(SampleInspectionError::InvalidRecord {
                        field: "coordinate time delta"
                    })
                ),
                "negative accumulated time must be rejected",
            );

            let mut negative_drift = escape_record([1.0, 0.0, 0.0]);
            negative_drift.maximum_invariant_drift[drift_index] = negative;
            prop_assert!(
                matches!(
                    decode_retrace(negative_drift, None, [0.5; 2]),
                    Err(SampleInspectionError::InvalidRecord { field: "invariant drift" })
                ),
                "negative maximum drift must be rejected",
            );
        }
    }
}

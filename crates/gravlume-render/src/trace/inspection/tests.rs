use std::sync::mpsc;

use gravlume_domain::ImageSample;
use proptest::prelude::*;

use super::{
    SampleBranchKey, SampleInspection, SampleInspectionCompletion, SampleInspectionError,
    SampleInspectionRequestError, SampleInspectionTicket, SamplePolarSide, SampleRetrace,
    SampleTraceOutcome,
    protocol::{TraceTermination, decode_branch_key},
    slot::{PendingInspection, SampleInspectionSlot},
};
use crate::{
    error::GpuErrorScopes,
    extent::RenderExtent,
    test_device::{TestGpu, native_gpu},
    trace::TracePipeline,
};

const SURFACE_OBSERVABLE: &str =
    include_str!("../../../../gravlume-reference/fixtures/v2/kerr-surface-observable.toml");

impl TracePipeline {
    pub(crate) fn inspect_sample(
        &self,
        gpu: &TestGpu,
        extent: RenderExtent,
        sample: ImageSample,
    ) -> SampleInspection {
        let published = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-only sample inspection published texel"),
            size: wgpu::Extent3d {
                width: extent.width(),
                height: extent.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut slot = SampleInspectionSlot::new(&gpu.device, self);
        slot.request(&gpu.device, &gpu.queue, &published, extent, 1, sample)
            .expect("test inspection request is accepted");
        let fence = gpu.queue.submit([]);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(fence),
                timeout: None,
            })
            .expect("test inspection submission completes");
        match slot.wait_for_completion(Some(1)) {
            SampleInspectionCompletion::Completed { inspection, .. } => inspection,
            completion => panic!("test inspection must complete, got {completion:?}"),
        }
    }
}

fn surface_fixture() -> gravlume_reference::SurfaceObservationFixture {
    gravlume_reference::FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation")
}

fn fixture_extent(fixture: &gravlume_reference::SurfaceObservationFixture) -> RenderExtent {
    RenderExtent::new(
        fixture.observation().view().width().get(),
        fixture.observation().view().height().get(),
    )
    .expect("fixture extent is nonzero")
}

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

#[test]
fn map_failure_does_not_emit_a_secondary_unmap_validation_error() {
    let fixture = surface_fixture();
    let observation = fixture.observation();
    let extent = fixture_extent(&fixture);
    let gpu = native_gpu();
    let trace = TracePipeline::new(&gpu.device, observation)
        .expect("fixture observation enters the GPU profile");
    let mut slot = SampleInspectionSlot::new(&gpu.device, &trace);
    let ticket = SampleInspectionTicket::new(23, extent, fixture.sample());
    let (sender, receiver) = mpsc::sync_channel(1);
    sender
        .send(Err(wgpu::BufferAsyncError))
        .expect("synthetic map completion is delivered");
    slot.pending = Some(PendingInspection {
        receiver,
        ticket,
        cancelled: false,
    });
    let scopes = GpuErrorScopes::push(&gpu.device);

    let completion = slot
        .poll(Some(23))
        .expect("map failure produces one terminal event");

    assert_eq!(completion.ticket(), ticket);
    assert!(matches!(
        completion,
        SampleInspectionCompletion::Failed {
            error: SampleInspectionError::Map(_),
            ..
        }
    ));
    let secondary_error = pollster::block_on(scopes.finish());
    assert!(
        secondary_error.is_ok(),
        "the typed map failure must be the only reported error: {secondary_error:?}"
    );
}

#[test]
fn cancelled_request_drains_before_the_fixed_slot_is_reused() {
    let fixture = surface_fixture();
    let observation = fixture.observation();
    let extent = fixture_extent(&fixture);
    let gpu = native_gpu();
    let trace = TracePipeline::new(&gpu.device, observation)
        .expect("fixture observation enters the GPU profile");
    let published = published_texture(&gpu.device, extent);
    let mut slot = SampleInspectionSlot::new(&gpu.device, &trace);

    let request = slot
        .request(
            &gpu.device,
            &gpu.queue,
            &published,
            extent,
            7,
            fixture.sample(),
        )
        .expect("the fixed slot accepts its first request");
    assert!(matches!(
        slot.request(
            &gpu.device,
            &gpu.queue,
            &published,
            extent,
            7,
            fixture.sample(),
        ),
        Err(SampleInspectionRequestError::Busy)
    ));
    slot.cancel_active();
    assert!(slot.is_pending());

    let fence = gpu.queue.submit([]);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .expect("cancelled inspection submission drains");
    let completion = slot.wait_for_completion(Some(7));
    assert_eq!(completion.ticket(), request);
    assert!(matches!(
        completion,
        SampleInspectionCompletion::Cancelled { .. }
    ));
    assert!(!slot.is_pending());

    slot.request(
        &gpu.device,
        &gpu.queue,
        &published,
        extent,
        7,
        fixture.sample(),
    )
    .expect("the slot is reusable only after cancellation drains");
}

#[test]
fn completion_binds_ticket_and_fixed_retrace_method() {
    const PUBLISHED_TEXEL: [u16; 4] = [0x3c00, 0x4000, 0x4200, 0x3c00];

    let fixture = surface_fixture();
    let observation = fixture.observation();
    let extent = fixture_extent(&fixture);
    let gpu = native_gpu();
    let trace = TracePipeline::new(&gpu.device, observation)
        .expect("fixture observation enters the GPU profile");
    let published = published_texture(&gpu.device, extent);
    write_published_texel(
        &gpu.queue,
        &published,
        fixture.sample().pixel(),
        PUBLISHED_TEXEL,
    );
    let mut slot = SampleInspectionSlot::new(&gpu.device, &trace);

    let request = slot
        .request(
            &gpu.device,
            &gpu.queue,
            &published,
            extent,
            11,
            fixture.sample(),
        )
        .expect("inspection request is accepted");
    let fence = gpu.queue.submit([]);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .expect("inspection submission completes");
    let completion = slot.wait_for_completion(Some(11));
    assert_eq!(completion.ticket(), request);
    let SampleInspectionCompletion::Completed { inspection, .. } = completion else {
        panic!("completed GPU work must decode as a completed inspection, got {completion:?}");
    };

    assert_eq!(request.generation(), 11);
    assert_eq!(request.extent(), [extent.width(), extent.height()]);
    assert_eq!(request.sample(), fixture.sample());
    assert_eq!(
        SampleRetrace::METHOD_ID,
        "gpu-ks-rk4-v1/full-kerr-schild-retrace/wgsl-binary32"
    );
    assert!(matches!(
        inspection.fresh_retrace().outcome(),
        SampleTraceOutcome::EquatorialSurface { .. }
    ));
    assert_eq!(
        inspection.published_texel().rgba16_float_bits(),
        PUBLISHED_TEXEL
    );
    assert_eq!(
        inspection.published_texel().kind(),
        crate::ScientificPixelKind::AnalyticEscapePreview
    );
}

#[test]
fn publication_mismatch_discards_the_result_once_and_releases_the_slot() {
    let fixture = surface_fixture();
    let observation = fixture.observation();
    let extent = fixture_extent(&fixture);
    let gpu = native_gpu();
    let trace = TracePipeline::new(&gpu.device, observation)
        .expect("fixture observation enters the GPU profile");
    let published = published_texture(&gpu.device, extent);
    let mut slot = SampleInspectionSlot::new(&gpu.device, &trace);

    let request = slot
        .request(
            &gpu.device,
            &gpu.queue,
            &published,
            extent,
            17,
            fixture.sample(),
        )
        .expect("inspection request is accepted");
    let fence = gpu.queue.submit([]);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .expect("inspection submission completes");

    let completion = slot.wait_for_completion(Some(18));
    assert_eq!(completion.ticket(), request);
    assert!(matches!(
        completion,
        SampleInspectionCompletion::Cancelled { .. }
    ));
    assert!(slot.poll(Some(18)).is_none());
    assert!(!slot.is_pending());
}

fn published_texture(device: &wgpu::Device, extent: RenderExtent) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sample inspection published-texel fixture"),
        size: wgpu::Extent3d {
            width: extent.width(),
            height: extent.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn write_published_texel(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    pixel: [u32; 2],
    rgba16_float_bits: [u16; 4],
) {
    let mut bytes = [0_u8; wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize];
    for (channel, bits) in rgba16_float_bits.into_iter().enumerate() {
        let offset = channel * std::mem::size_of::<u16>();
        bytes[offset..offset + std::mem::size_of::<u16>()].copy_from_slice(&bits.to_le_bytes());
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: pixel[0],
                y: pixel[1],
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

proptest! {
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

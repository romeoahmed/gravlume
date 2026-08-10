use std::sync::{OnceLock, mpsc};

pub struct TestGpu {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

pub fn native_gpu() -> &'static TestGpu {
    static GPU: OnceLock<TestGpu> = OnceLock::new();
    GPU.get_or_init(|| pollster::block_on(request_native_gpu()))
}

async fn request_native_gpu() -> TestGpu {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = crate::native_backends();
    let instance = wgpu::Instance::new(descriptor);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await
        .expect("native contract tests require a hardware adapter");
    assert!(
        adapter.get_downlevel_capabilities().is_webgpu_compliant(),
        "native contract tests require a WebGPU-compliant adapter"
    );

    let missing_features = crate::capabilities::BASELINE_FEATURES - adapter.features();
    assert!(
        missing_features.is_empty(),
        "native adapter is missing contract features: {missing_features:?}"
    );
    let adapter_limits = adapter.limits();
    let required_limits = wgpu::Limits::default()
        .using_resolution(adapter_limits.clone())
        .using_alignment(adapter_limits);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("native GPU contract test device"),
            required_features: crate::capabilities::BASELINE_FEATURES,
            required_limits,
            ..Default::default()
        })
        .await
        .expect("native contract test device request succeeds");

    TestGpu { device, queue }
}

pub fn read_buffer(buffer: &wgpu::Buffer, submission: wgpu::SubmissionIndex) -> Vec<u8> {
    // map_async callbacks are driven by polling; waiting for the producing submission makes the
    // contract deterministic without a wall-clock timeout.
    // Source: https://docs.rs/wgpu/30.0.0/wgpu/struct.Buffer.html#method.map_async
    let gpu = native_gpu();
    let (sender, receiver) = mpsc::sync_channel(1);
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        let _send_result = sender.send(result);
    });
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("native GPU submission completes");
    receiver
        .recv()
        .expect("buffer map callback runs")
        .expect("buffer maps for readback");

    let mapped = buffer
        .get_mapped_range(..)
        .expect("mapped buffer range is available");
    let bytes = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    bytes
}

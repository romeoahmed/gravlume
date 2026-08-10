use std::{
    future::Future,
    sync::{Arc, mpsc},
};

#[derive(Debug, thiserror::Error)]
pub enum RenderInitError {
    #[error("failed to create the native presentation surface: {0}")]
    CreateSurface(#[from] wgpu::CreateSurfaceError),
    #[error("no adapter was available for the native surface: {0}")]
    RequestAdapter(#[from] wgpu::RequestAdapterError),
    #[error("adapter {adapter:?} does not satisfy Phase 0: {reason}")]
    UnsupportedAdapter { adapter: String, reason: String },
    #[error("surface does not satisfy the SDR presentation contract: {0}")]
    SurfaceCapabilities(String),
    #[error("failed to create the Phase 0 device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("failed to create {stage}: {source}")]
    GpuResource {
        stage: &'static str,
        #[source]
        source: wgpu::Error,
    },
    #[error("failed to install the initial window extent: {0}")]
    InitialResize(#[source] ResizeError),
}

#[derive(Debug, thiserror::Error)]
pub enum RenderRuntimeError {
    #[error("GPU timing/readback failed: {0}")]
    Timing(String),
    #[error("non-blocking GPU poll failed: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("failed to recreate a lost surface: {0}")]
    RecreateSurface(#[from] wgpu::CreateSurfaceError),
    #[error("surface capabilities changed incompatibly while recovering")]
    SurfaceCapabilitiesChanged,
    #[error("nonzero extent has no matching frame-resource bundle")]
    MissingFrameResources,
}

#[derive(Debug, thiserror::Error)]
pub enum ResizeError {
    #[error(
        "requested render extent {width}x{height} exceeds the device 2D texture limit of {max_texture_dimension_2d}"
    )]
    ExtentLimit {
        width: u32,
        height: u32,
        max_texture_dimension_2d: u32,
    },
    #[error("the surface no longer supports the active presentation configuration")]
    SurfaceCapabilitiesChanged,
    #[error("failed to {stage}: {source}")]
    GpuResource {
        stage: &'static str,
        #[source]
        source: wgpu::Error,
    },
}

impl ResizeError {
    #[must_use]
    pub const fn kind(&self) -> DeviceEventKind {
        match self {
            Self::ExtentLimit { .. } | Self::SurfaceCapabilitiesChanged => {
                DeviceEventKind::Validation
            }
            Self::GpuResource { source, .. } => device_error_kind(source),
        }
    }

    #[must_use]
    pub const fn is_fatal(&self) -> bool {
        matches!(
            self.kind(),
            DeviceEventKind::Internal | DeviceEventKind::OutOfMemory | DeviceEventKind::Lost
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceEventKind {
    Validation,
    Internal,
    OutOfMemory,
    Lost,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceEvent {
    kind: DeviceEventKind,
    message: String,
}

impl DeviceEvent {
    #[must_use]
    pub(crate) fn validation(message: impl Into<String>) -> Self {
        Self {
            kind: DeviceEventKind::Validation,
            message: message.into(),
        }
    }

    #[must_use]
    pub(crate) fn from_wgpu(stage: &'static str, error: wgpu::Error) -> Self {
        let (kind, message) = device_error_details(error);
        Self {
            kind,
            message: format!("{stage}: {message}"),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> DeviceEventKind {
        self.kind
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    #[must_use]
    pub const fn is_fatal(&self) -> bool {
        matches!(
            self.kind,
            DeviceEventKind::Internal | DeviceEventKind::OutOfMemory | DeviceEventKind::Lost
        )
    }
}

/// Captures every wgpu error category while a resource candidate is constructed.
///
/// Source: <https://docs.rs/wgpu/30.0.0/wgpu/struct.Device.html#method.push_error_scope>
pub struct GpuErrorScopes {
    internal: wgpu::ErrorScopeGuard,
    out_of_memory: wgpu::ErrorScopeGuard,
    validation: wgpu::ErrorScopeGuard,
}

impl GpuErrorScopes {
    pub(crate) fn push(device: &wgpu::Device) -> Self {
        Self {
            internal: device.push_error_scope(wgpu::ErrorFilter::Internal),
            out_of_memory: device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
            validation: device.push_error_scope(wgpu::ErrorFilter::Validation),
        }
    }

    pub(crate) fn finish(self) -> impl Future<Output = Result<(), wgpu::Error>> + Send {
        let validation = self.validation.pop();
        let out_of_memory = self.out_of_memory.pop();
        let internal = self.internal.pop();
        async move {
            let validation = validation.await;
            let out_of_memory = out_of_memory.await;
            let internal = internal.await;
            out_of_memory
                .or(internal)
                .or(validation)
                .map_or(Ok(()), Err)
        }
    }
}

pub fn scoped_gpu_operation<T>(
    device: &wgpu::Device,
    operation: impl FnOnce() -> T,
) -> Result<T, wgpu::Error> {
    let scopes = GpuErrorScopes::push(device);
    let result = operation();
    pollster::block_on(scopes.finish())?;
    Ok(result)
}

pub fn install_device_callbacks(
    device: &wgpu::Device,
) -> (mpsc::Sender<DeviceEvent>, mpsc::Receiver<DeviceEvent>) {
    let (sender, receiver) = mpsc::channel();
    let uncaptured_sender = sender.clone();
    device.on_uncaptured_error(Arc::new(move |error| {
        let (kind, message) = device_error_details(error);
        if uncaptured_sender
            .send(DeviceEvent { kind, message })
            .is_err()
        {
            tracing::debug!("device event receiver dropped");
        }
    }));
    let lost_sender = sender.clone();
    device.set_device_lost_callback(move |reason, message| {
        let message = format!("{reason:?}: {message}");
        if lost_sender
            .send(DeviceEvent {
                kind: DeviceEventKind::Lost,
                message,
            })
            .is_err()
        {
            tracing::debug!("device event receiver dropped");
        }
    });
    (sender, receiver)
}

fn device_error_details(error: wgpu::Error) -> (DeviceEventKind, String) {
    let kind = device_error_kind(&error);
    let message = match error {
        wgpu::Error::Validation { description, .. } | wgpu::Error::Internal { description, .. } => {
            description
        }
        wgpu::Error::OutOfMemory { .. } => "GPU out of memory".to_owned(),
    };
    (kind, message)
}

const fn device_error_kind(error: &wgpu::Error) -> DeviceEventKind {
    match error {
        wgpu::Error::Validation { .. } => DeviceEventKind::Validation,
        wgpu::Error::Internal { .. } => DeviceEventKind::Internal,
        wgpu::Error::OutOfMemory { .. } => DeviceEventKind::OutOfMemory,
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuErrorScopes, scoped_gpu_operation};

    #[test]
    fn initialization_error_scopes_capture_invalid_wgsl() {
        pollster::block_on(async {
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
                .expect("native adapter is available");
            let (device, _queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .expect("test device request succeeds");
            let scopes = GpuErrorScopes::push(&device);

            let _invalid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("intentionally invalid initialization shader"),
                source: wgpu::ShaderSource::Wgsl("@compute fn broken(".into()),
            });

            let error = scopes
                .finish()
                .await
                .expect_err("invalid WGSL is reported through the initialization scope");
            assert!(matches!(error, wgpu::Error::Validation { .. }));
        });
    }

    #[test]
    fn synchronous_runtime_scope_reports_invalid_resource_creation() {
        pollster::block_on(async {
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
                .expect("native adapter is available");
            let (device, _queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .expect("test device request succeeds");

            let error = scoped_gpu_operation(&device, || {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("intentionally invalid runtime shader"),
                    source: wgpu::ShaderSource::Wgsl("@compute fn broken(".into()),
                })
            })
            .expect_err("invalid runtime resource creation is reported by its local scope");

            assert!(matches!(error, wgpu::Error::Validation { .. }));
        });
    }
}

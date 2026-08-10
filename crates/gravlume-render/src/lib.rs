#![forbid(unsafe_code)]

//! Private GPU frame engine for Gravlume.

mod capabilities;
mod display;
mod engine;
mod extent;
mod gpu_error;
mod scene;
mod timing;

#[cfg(test)]
mod test_gpu;

#[doc(hidden)]
pub use engine::{FrameSkip, FrameStatus, GpuEngine, PollOutcome, RenderDiagnostics};
#[doc(hidden)]
pub use gpu_error::{
    DeviceEvent, DeviceEventKind, RenderInitError, RenderRuntimeError, ResizeError,
};

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
compile_error!("gravlume-render supports only native macOS, Windows, and Linux targets");

#[cfg(target_os = "macos")]
pub(crate) const fn native_backends() -> wgpu::Backends {
    wgpu::Backends::METAL
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
pub(crate) const fn native_backends() -> wgpu::Backends {
    wgpu::Backends::VULKAN
}

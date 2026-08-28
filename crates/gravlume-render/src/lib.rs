#![forbid(unsafe_code)]

//! Native GPU renderer for Gravlume.

mod capabilities;
mod display;
mod error;
mod extent;
mod renderer;
mod scientific_capture;
mod spectral_lut;
mod timing;
mod trace;

#[cfg(feature = "gpu-benchmarks")]
pub mod benchmark;

#[cfg(test)]
mod gpu_capture;
#[cfg(test)]
mod gpu_trace_tests;
#[cfg(test)]
mod test_device;

pub use capabilities::CapabilityError;
pub use error::{DeviceEvent, DeviceEventKind, RendererError, RendererInitError, ResizeError};
pub use renderer::{
    CurrentPublication, PresentResult, PresentSkip, Renderer, RendererDiagnostics, RendererUpdate,
};
pub use scientific_capture::{
    ScientificCapture, ScientificCaptureError, ScientificCaptureMetadata, ScientificChannelModel,
    ScientificNumericalMetadata, ScientificPixelKind, ScientificTexel,
};
pub use timing::TimingError;
pub use trace::{
    GpuTraceInputError, SampleBranchKey, SampleInspection, SampleInspectionCompletion,
    SampleInspectionError, SampleInspectionRequestError, SampleInspectionTicket, SamplePolarSide,
    SampleRetrace, SampleSurfaceEvaluation, SampleTraceDiagnostics, SampleTraceOutcome,
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

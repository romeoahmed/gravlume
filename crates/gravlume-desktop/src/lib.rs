#![forbid(unsafe_code)]

//! Native desktop lifecycle for Gravlume.

mod app;
mod launch;
mod lifecycle;

pub use app::{RunError, run};
pub use launch::{Launch, WindowPreferences};

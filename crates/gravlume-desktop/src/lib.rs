#![forbid(unsafe_code)]

//! Native desktop lifecycle for Gravlume.

mod app;
mod launch;
mod lifecycle;
mod schedule;
mod ui;

pub use app::{RunError, run};
pub use launch::{Launch, WindowPreferences};

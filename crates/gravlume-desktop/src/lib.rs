#![forbid(unsafe_code)]

//! Native desktop lifecycle for Gravlume.

mod app;
mod inspection;
mod preview;
mod schedule;
mod ui;

pub use app::{RunError, run};

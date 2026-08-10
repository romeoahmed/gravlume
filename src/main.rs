use std::error::Error;

use tracing_subscriber::{EnvFilter, filter::LevelFilter};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let filter = EnvFilter::builder()
        .with_regex(false)
        .with_default_directive(LevelFilter::INFO.into())
        .from_env()?;
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init()?;

    gravlume_desktop::run(gravlume_desktop::Launch::default())?;
    Ok(())
}

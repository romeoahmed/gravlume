#![forbid(unsafe_code)]

//! Native desktop lifecycle for Gravlume.

mod app;
mod launch;
mod lifecycle;

pub use app::{RunError, run};
pub use launch::{Launch, WindowPreferences};

#[cfg(test)]
mod tests {
    use super::{Launch, WindowPreferences};

    #[test]
    fn default_launch_has_a_valid_offline_window() {
        let launch = Launch::default();

        assert_eq!(launch.window().title(), "Gravlume");
        assert_eq!(launch.window().width(), 1280);
        assert_eq!(launch.window().height(), 720);
    }

    #[test]
    fn window_preferences_preserve_platform_requests() {
        let preferences = WindowPreferences::new("", 0, u32::MAX);

        assert_eq!(preferences.title(), "");
        assert_eq!(preferences.width(), 0);
        assert_eq!(preferences.height(), u32::MAX);
    }

    #[test]
    fn window_preferences_replace_launch_defaults() {
        let preferences = WindowPreferences::new("Scientific View", 1024, 768);
        let launch = Launch::default().with_window(preferences.clone());

        assert_eq!(launch.window(), &preferences);
    }
}

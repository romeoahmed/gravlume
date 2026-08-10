#![forbid(unsafe_code)]

//! Native desktop lifecycle for Gravlume.

mod app;
mod launch;
mod lifecycle;

pub use app::{RunError, run};
pub use launch::{Launch, WindowPreferences, WindowPreferencesError};

#[cfg(test)]
mod tests {
    use super::{Launch, WindowPreferences, WindowPreferencesError};

    #[test]
    fn default_launch_has_a_valid_offline_window() {
        let launch = Launch::default();

        assert_eq!(launch.window().title(), "Gravlume");
        assert_eq!(launch.window().width(), 1280);
        assert_eq!(launch.window().height(), 720);
    }

    #[test]
    fn window_preferences_reject_invalid_seam_values() {
        assert_eq!(
            WindowPreferences::new("", 800, 600),
            Err(WindowPreferencesError::EmptyTitle)
        );
        assert_eq!(
            WindowPreferences::new("Gravlume", 0, 600),
            Err(WindowPreferencesError::ZeroExtent)
        );
        assert_eq!(
            WindowPreferences::new("Gravlume", 20_000, 600),
            Err(WindowPreferencesError::ExtentTooLarge)
        );
    }

    #[test]
    fn validated_window_preferences_replace_launch_defaults() {
        let preferences =
            WindowPreferences::new("Scientific View", 1024, 768).expect("valid preferences");
        let launch = Launch::default().with_window(preferences.clone());

        assert_eq!(launch.window(), &preferences);
    }
}

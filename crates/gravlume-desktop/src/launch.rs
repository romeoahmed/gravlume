#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowPreferences {
    title: String,
    width: u32,
    height: u32,
}

impl WindowPreferences {
    /// Creates native-window preferences.
    #[must_use]
    pub fn new(title: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            title: title.into(),
            width,
            height,
        }
    }

    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }
}

impl Default for WindowPreferences {
    fn default() -> Self {
        Self {
            title: "Gravlume".to_owned(),
            width: 1280,
            height: 720,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Launch {
    window: WindowPreferences,
}

impl Launch {
    #[must_use]
    pub fn with_window(mut self, window: WindowPreferences) -> Self {
        self.window = window;
        self
    }

    pub(crate) const fn window(&self) -> &WindowPreferences {
        &self.window
    }
}

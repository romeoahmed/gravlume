const MAX_WINDOW_DIMENSION: u32 = 16_384;
const MAX_TITLE_BYTES: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowPreferences {
    title: String,
    width: u32,
    height: u32,
}

impl WindowPreferences {
    /// Creates validated native-window preferences.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or oversized title, a zero extent, or an extent above the
    /// desktop safety bound.
    pub fn new(
        title: impl Into<String>,
        width: u32,
        height: u32,
    ) -> Result<Self, WindowPreferencesError> {
        let title = title.into();
        if title.trim().is_empty() {
            return Err(WindowPreferencesError::EmptyTitle);
        }
        if title.len() > MAX_TITLE_BYTES {
            return Err(WindowPreferencesError::TitleTooLong);
        }
        if width == 0 || height == 0 {
            return Err(WindowPreferencesError::ZeroExtent);
        }
        if width > MAX_WINDOW_DIMENSION || height > MAX_WINDOW_DIMENSION {
            return Err(WindowPreferencesError::ExtentTooLarge);
        }
        Ok(Self {
            title,
            width,
            height,
        })
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WindowPreferencesError {
    #[error("window title must contain a non-whitespace character")]
    EmptyTitle,
    #[error("window title exceeds 128 UTF-8 bytes")]
    TitleTooLong,
    #[error("initial window extent must be nonzero")]
    ZeroExtent,
    #[error("initial window extent exceeds the 16384-pixel safety bound")]
    ExtentTooLarge,
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

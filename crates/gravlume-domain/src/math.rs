use crate::{
    ValidationIssue, ValidationIssueCode, ValidationReport, validation::validate_finite_array,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpacetimeEvent {
    txyz: [f64; 4],
}

impl SpacetimeEvent {
    /// Creates an event in canonical `(t, x, y, z)` component order.
    ///
    /// # Errors
    ///
    /// Returns a validation report when any component is non-finite.
    pub fn from_txyz(txyz: [f64; 4]) -> Result<Self, ValidationReport> {
        let mut report = ValidationReport::default();
        validate_finite_array(&mut report, txyz, "event.txyz");
        report.into_result(Self { txyz })
    }

    pub(super) const fn from_validated(txyz: [f64; 4]) -> Self {
        Self { txyz }
    }

    #[must_use]
    pub const fn to_txyz(self) -> [f64; 4] {
        self.txyz
    }

    pub(super) const fn x(self) -> f64 {
        self.txyz[1]
    }

    pub(super) const fn y(self) -> f64 {
        self.txyz[2]
    }

    pub(super) const fn z(self) -> f64 {
        self.txyz[3]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FourVector([f64; 4]);

impl FourVector {
    pub(super) const fn new(txyz: [f64; 4]) -> Self {
        Self(txyz)
    }

    pub(super) const fn to_array(self) -> [f64; 4] {
        self.0
    }

    pub(super) fn add(self, other: Self) -> Self {
        Self(std::array::from_fn(|index| self.0[index] + other.0[index]))
    }

    pub(super) fn subtract(self, other: Self) -> Self {
        Self(std::array::from_fn(|index| self.0[index] - other.0[index]))
    }

    pub(super) fn scaled(self, scalar: f64) -> Self {
        Self(self.0.map(|component| component * scalar))
    }

    pub(super) fn is_finite(self) -> bool {
        self.0.into_iter().all(f64::is_finite)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodesicState {
    components: [f64; 8],
}

impl GeodesicState {
    /// Creates a canonical Hamilton state `(t, x, y, z, p_t, p_x, p_y, p_z)`.
    ///
    /// # Errors
    ///
    /// Returns a validation report when any component is non-finite.
    pub fn new(
        position_txyz: [f64; 4],
        momentum_covariant_txyz: [f64; 4],
    ) -> Result<Self, ValidationReport> {
        let mut report = ValidationReport::default();
        validate_finite_array(&mut report, position_txyz, "geodesic_state.position_txyz");
        validate_finite_array(
            &mut report,
            momentum_covariant_txyz,
            "geodesic_state.momentum_covariant_txyz",
        );
        let components = std::array::from_fn(|index| {
            if index < 4 {
                position_txyz[index]
            } else {
                momentum_covariant_txyz[index - 4]
            }
        });
        report.into_result(Self { components })
    }

    /// Reconstructs a state produced by a numerical integrator.
    ///
    /// # Errors
    ///
    /// Returns a validation report when the numerical state is non-finite.
    pub fn from_components(components: [f64; 8]) -> Result<Self, ValidationReport> {
        if components.into_iter().all(f64::is_finite) {
            Ok(Self { components })
        } else {
            Err(ValidationReport::from_issue(ValidationIssue::error(
                ValidationIssueCode::NonFinite,
                "geodesic_state",
                "every numerical state component must be finite",
            )))
        }
    }

    pub(super) const fn from_validated(components: [f64; 8]) -> Self {
        Self { components }
    }

    #[must_use]
    pub const fn components(self) -> [f64; 8] {
        self.components
    }

    #[must_use]
    pub const fn event(self) -> SpacetimeEvent {
        SpacetimeEvent::from_validated([
            self.components[0],
            self.components[1],
            self.components[2],
            self.components[3],
        ])
    }

    #[must_use]
    pub const fn momentum_covariant_txyz(self) -> [f64; 4] {
        [
            self.components[4],
            self.components[5],
            self.components[6],
            self.components[7],
        ]
    }
}

impl ValidationReport {
    fn from_issue(issue: ValidationIssue) -> Self {
        let mut report = Self::default();
        report.push(issue);
        report
    }
}

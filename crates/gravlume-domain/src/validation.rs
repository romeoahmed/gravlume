use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValidationIssueCode {
    NonFinite,
    NonPositive,
    OutOfRange,
    DegenerateDirection,
    NonStationaryObserver,
    InternalInvariant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationSeverity {
    Error,
    Warning,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationIssue {
    code: ValidationIssueCode,
    field_path: String,
    severity: ValidationSeverity,
    explanation: String,
}

impl ValidationIssue {
    pub(super) fn error(
        code: ValidationIssueCode,
        field_path: impl Into<String>,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            code,
            field_path: field_path.into(),
            severity: ValidationSeverity::Error,
            explanation: explanation.into(),
        }
    }

    #[must_use]
    pub const fn code(&self) -> ValidationIssueCode {
        self.code
    }

    #[must_use]
    pub fn field_path(&self) -> &str {
        &self.field_path
    }

    #[must_use]
    pub const fn severity(&self) -> ValidationSeverity {
        self.severity
    }

    /// Returns a diagnostic explanation whose wording is not a stable protocol field.
    #[must_use]
    pub fn explanation(&self) -> &str {
        &self.explanation
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ValidationReport {
    issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub(super) fn from_error(
        code: ValidationIssueCode,
        field_path: impl Into<String>,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            issues: vec![ValidationIssue::error(code, field_path, explanation)],
        }
    }

    pub(super) fn push(&mut self, issue: ValidationIssue) {
        self.issues.push(issue);
    }

    pub(super) fn extend(&mut self, other: Self) {
        self.issues.extend(other.issues);
    }

    pub(super) fn into_result<T>(self, value: T) -> Result<T, Self> {
        if self.issues.is_empty() {
            Ok(value)
        } else {
            Err(self)
        }
    }

    #[must_use]
    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.issues.is_empty()
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "validation failed with {} issue(s)",
            self.issues.len()
        )
    }
}

impl std::error::Error for ValidationReport {}

pub fn validate_finite(report: &mut ValidationReport, value: f64, path: impl Into<String>) {
    if !value.is_finite() {
        report.push(ValidationIssue::error(
            ValidationIssueCode::NonFinite,
            path.into(),
            "value must be finite",
        ));
    }
}

pub fn validate_finite_array<const N: usize>(
    report: &mut ValidationReport,
    values: [f64; N],
    path: impl Into<String>,
) {
    if values.iter().any(|value| !value.is_finite()) {
        report.push(ValidationIssue::error(
            ValidationIssueCode::NonFinite,
            path.into(),
            "every component must be finite",
        ));
    }
}

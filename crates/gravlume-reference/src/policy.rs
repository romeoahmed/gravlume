pub const REGULAR_V1_ID: &str = "reference-regular-v1";
pub const STRICT_V1_ID: &str = "reference-strict-v1";
pub const V1_SINGULARITY_GUARD_D_OVER_M4_DECIMAL: &str = "9.094947017729282379150390625e-13";
const V1_SINGULARITY_GUARD_D_OVER_M4: f64 = f64::from_bits(0x3d70_0000_0000_0000);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferencePolicy {
    id: &'static str,
    position_relative_tolerance: f64,
    position_absolute_tolerance: f64,
    momentum_relative_tolerance: f64,
    momentum_absolute_tolerance: f64,
    initial_step_m: f64,
    minimum_step_m: f64,
    maximum_step_m: f64,
    safety_factor: f64,
    minimum_step_factor: f64,
    maximum_step_factor: f64,
    maximum_accepted_steps: u64,
    maximum_consecutive_rejects: u32,
    event_affine_tolerance_m: f64,
    event_tie_tolerance_m: f64,
    event_arming_band_m: f64,
    singularity_guard_d_over_m4: f64,
}

impl ReferencePolicy {
    #[must_use]
    pub const fn regular_v1() -> Self {
        Self {
            id: REGULAR_V1_ID,
            position_relative_tolerance: 2.0e-12,
            position_absolute_tolerance: 2.0e-13,
            momentum_relative_tolerance: 2.0e-12,
            momentum_absolute_tolerance: 2.0e-13,
            initial_step_m: 1.0e-3,
            minimum_step_m: 9.094_947_017_729_282e-13,
            maximum_step_m: 0.5,
            safety_factor: 0.9,
            minimum_step_factor: 0.2,
            maximum_step_factor: 5.0,
            maximum_accepted_steps: 200_000,
            maximum_consecutive_rejects: 64,
            event_affine_tolerance_m: 2.0e-11,
            event_tie_tolerance_m: 5.0e-11,
            event_arming_band_m: 1.28e-9,
            singularity_guard_d_over_m4: V1_SINGULARITY_GUARD_D_OVER_M4,
        }
    }

    #[must_use]
    pub const fn strict_v1() -> Self {
        let regular = Self::regular_v1();
        Self {
            id: STRICT_V1_ID,
            position_relative_tolerance: regular.position_relative_tolerance / 16.0,
            position_absolute_tolerance: regular.position_absolute_tolerance / 16.0,
            momentum_relative_tolerance: regular.momentum_relative_tolerance / 16.0,
            momentum_absolute_tolerance: regular.momentum_absolute_tolerance / 16.0,
            maximum_step_m: 0.25,
            maximum_accepted_steps: regular.maximum_accepted_steps * 2,
            maximum_consecutive_rejects: regular.maximum_consecutive_rejects * 2,
            event_affine_tolerance_m: regular.event_affine_tolerance_m / 4.0,
            event_tie_tolerance_m: regular.event_tie_tolerance_m / 4.0,
            ..regular
        }
    }

    #[must_use]
    pub const fn id(self) -> &'static str {
        self.id
    }

    #[must_use]
    pub const fn position_relative_tolerance(self) -> f64 {
        self.position_relative_tolerance
    }

    #[must_use]
    pub const fn position_absolute_tolerance(self) -> f64 {
        self.position_absolute_tolerance
    }

    #[must_use]
    pub const fn momentum_relative_tolerance(self) -> f64 {
        self.momentum_relative_tolerance
    }

    #[must_use]
    pub const fn momentum_absolute_tolerance(self) -> f64 {
        self.momentum_absolute_tolerance
    }

    #[must_use]
    pub const fn initial_step_m(self) -> f64 {
        self.initial_step_m
    }

    #[must_use]
    pub const fn minimum_step_m(self) -> f64 {
        self.minimum_step_m
    }

    #[must_use]
    pub const fn maximum_step_m(self) -> f64 {
        self.maximum_step_m
    }

    #[must_use]
    pub const fn maximum_accepted_steps(self) -> u64 {
        self.maximum_accepted_steps
    }

    #[must_use]
    pub const fn maximum_consecutive_rejects(self) -> u32 {
        self.maximum_consecutive_rejects
    }

    #[must_use]
    pub const fn event_affine_tolerance_m(self) -> f64 {
        self.event_affine_tolerance_m
    }

    #[must_use]
    pub const fn event_tie_tolerance_m(self) -> f64 {
        self.event_tie_tolerance_m
    }

    #[must_use]
    pub const fn event_arming_band_m(self) -> f64 {
        self.event_arming_band_m
    }

    pub(super) const fn singularity_guard_d_over_m4(self) -> f64 {
        self.singularity_guard_d_over_m4
    }

    pub(super) fn error_norm(self, start: [f64; 8], end: [f64; 8], error: [f64; 8]) -> f64 {
        start
            .into_iter()
            .zip(end)
            .zip(error)
            .enumerate()
            .map(|(index, ((start, end), error))| {
                let (relative, absolute) = if index < 4 {
                    (
                        self.position_relative_tolerance,
                        self.position_absolute_tolerance,
                    )
                } else {
                    (
                        self.momentum_relative_tolerance,
                        self.momentum_absolute_tolerance,
                    )
                };
                error.abs() / relative.mul_add(start.abs().max(end.abs()), absolute)
            })
            .fold(0.0_f64, f64::max)
    }

    pub(super) fn next_step_magnitude(self, current: f64, error_norm: f64, accepted: bool) -> f64 {
        let maximum_factor = if accepted {
            self.maximum_step_factor
        } else {
            1.0
        };
        let factor = if error_norm == 0.0 {
            maximum_factor
        } else {
            (self.safety_factor * error_norm.powf(-0.2))
                .clamp(self.minimum_step_factor, maximum_factor)
        };
        (current * factor).clamp(self.minimum_step_m, self.maximum_step_m)
    }

    #[cfg(test)]
    pub(super) const fn limited_to_one_step_for_test(mut self) -> Self {
        self.maximum_accepted_steps = 1;
        self
    }
}

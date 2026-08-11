use gravlume_domain::{GeodesicState, GeometryError, KerrNewmanSpacetime};

const STAGES: usize = 7;
const STATE_COMPONENTS: usize = 8;

const A: [[f64; STAGES]; STAGES] = [
    [0.0; STAGES],
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        0.0,
        0.0,
    ],
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ],
];

const FIFTH_ORDER_WEIGHTS: [f64; STAGES] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
];

const FOURTH_ORDER_WEIGHTS: [f64; STAGES] = [
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339_200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
];

const DENSE_COEFFICIENTS: [[f64; 4]; STAGES] = [
    [
        1.0,
        -8_048_581_381.0 / 2_820_520_608.0,
        8_663_915_743.0 / 2_820_520_608.0,
        -12_715_105_075.0 / 11_282_082_432.0,
    ],
    [0.0; 4],
    [
        0.0,
        131_558_114_200.0 / 32_700_410_799.0,
        -68_118_460_800.0 / 10_900_136_933.0,
        87_487_479_700.0 / 32_700_410_799.0,
    ],
    [
        0.0,
        -1_754_552_775.0 / 470_086_768.0,
        14_199_869_525.0 / 1_410_260_304.0,
        -10_690_763_975.0 / 1_880_347_072.0,
    ],
    [
        0.0,
        127_303_824_393.0 / 49_829_197_408.0,
        -318_862_633_887.0 / 49_829_197_408.0,
        701_980_252_875.0 / 199_316_789_632.0,
    ],
    [
        0.0,
        -282_668_133.0 / 205_662_961.0,
        2_019_193_451.0 / 616_988_883.0,
        -1_453_857_185.0 / 822_651_844.0,
    ],
    [
        0.0,
        40_617_522.0 / 29_380_423.0,
        -110_615_467.0 / 29_380_423.0,
        69_997_945.0 / 29_380_423.0,
    ],
];

pub struct StepAttempt {
    pub(super) end: [f64; STATE_COMPONENTS],
    pub(super) error: [f64; STATE_COMPONENTS],
    pub(super) end_derivative: [f64; STATE_COMPONENTS],
    pub(super) dense: DenseOutput,
}

pub struct DenseOutput {
    start: [f64; STATE_COMPONENTS],
    step: f64,
    stages: [[f64; STATE_COMPONENTS]; STAGES],
}

pub struct StepFailure {
    error: GeometryError,
    evaluations: u8,
}

impl StepFailure {
    pub(super) const fn error(&self) -> GeometryError {
        self.error
    }

    pub(super) const fn evaluations(&self) -> u8 {
        self.evaluations
    }
}

impl DenseOutput {
    pub(super) fn evaluate(&self, theta: f64) -> [f64; STATE_COMPONENTS] {
        let powers = [theta, theta * theta, theta.powi(3), theta.powi(4)];
        std::array::from_fn(|component| {
            self.step.mul_add(
                dense_derivative(&self.stages, component, powers),
                self.start[component],
            )
        })
    }

    pub(super) fn time_increment(&self, theta: f64) -> f64 {
        let powers = [theta, theta * theta, theta.powi(3), theta.powi(4)];
        self.step * dense_derivative(&self.stages, 0, powers)
    }
}

pub fn derivative(
    spacetime: KerrNewmanSpacetime,
    state: [f64; STATE_COMPONENTS],
) -> Result<[f64; STATE_COMPONENTS], GeometryError> {
    let state = GeodesicState::from_components(state).map_err(|_| GeometryError::NonFinite)?;
    spacetime.hamiltonian_rhs(state)
}

pub fn attempt_step(
    spacetime: KerrNewmanSpacetime,
    start: [f64; STATE_COMPONENTS],
    step: f64,
    start_derivative: [f64; STATE_COMPONENTS],
) -> Result<StepAttempt, StepFailure> {
    let mut stages = [[0.0; STATE_COMPONENTS]; STAGES];
    stages[0] = start_derivative;
    for stage_index in 1..STAGES {
        let stage_state = std::array::from_fn(|component| {
            let weighted = stages[..stage_index]
                .iter()
                .zip(A[stage_index])
                .map(|(stage, coefficient)| coefficient * stage[component])
                .sum::<f64>();
            step.mul_add(weighted, start[component])
        });
        stages[stage_index] = derivative(spacetime, stage_state).map_err(|error| StepFailure {
            error,
            evaluations: [0, 1, 2, 3, 4, 5, 6][stage_index],
        })?;
    }
    let end = weighted_state(start, step, &stages, FIFTH_ORDER_WEIGHTS);
    let fourth_order = weighted_state(start, step, &stages, FOURTH_ORDER_WEIGHTS);
    let mut error = std::array::from_fn(|index| end[index] - fourth_order[index]);
    error[0] = weighted_increment(step, &stages, FIFTH_ORDER_WEIGHTS, 0)
        - weighted_increment(step, &stages, FOURTH_ORDER_WEIGHTS, 0);
    Ok(StepAttempt {
        end,
        error,
        end_derivative: stages[6],
        dense: DenseOutput {
            start,
            step,
            stages,
        },
    })
}

fn dense_derivative(
    stages: &[[f64; STATE_COMPONENTS]; STAGES],
    component: usize,
    powers: [f64; 4],
) -> f64 {
    stages
        .iter()
        .zip(DENSE_COEFFICIENTS)
        .map(|(stage, coefficients)| {
            coefficients
                .into_iter()
                .zip(powers)
                .map(|(coefficient, power)| coefficient * power)
                .sum::<f64>()
                * stage[component]
        })
        .sum()
}

fn weighted_increment(
    step: f64,
    stages: &[[f64; STATE_COMPONENTS]; STAGES],
    weights: [f64; STAGES],
    component: usize,
) -> f64 {
    let weighted_derivative = stages
        .iter()
        .zip(weights)
        .map(|(stage, weight)| weight * stage[component])
        .sum::<f64>();
    step * weighted_derivative
}

fn weighted_state(
    start: [f64; STATE_COMPONENTS],
    step: f64,
    stages: &[[f64; STATE_COMPONENTS]; STAGES],
    weights: [f64; STAGES],
) -> [f64; STATE_COMPONENTS] {
    std::array::from_fn(|component| {
        let weighted = stages
            .iter()
            .zip(weights)
            .map(|(stage, weight)| weight * stage[component])
            .sum::<f64>();
        step.mul_add(weighted, start[component])
    })
}

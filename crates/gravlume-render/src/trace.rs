use gravlume_domain::{Observation, ValidationReport};

pub(crate) const TRACE_SHADER: &str = include_str!("shaders/trace.wgsl");
pub(crate) const TRACE_RECORD_SIZE: u64 = 48;

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct TraceUniforms {
    pub(crate) spacetime: [f32; 4],
    pub(crate) observer_event: [f32; 4],
    pub(crate) observer_velocity: [f32; 4],
    pub(crate) image_right: [f32; 4],
    pub(crate) image_up: [f32; 4],
    pub(crate) arrival: [f32; 4],
    pub(crate) projection_policy: [f32; 4],
    pub(crate) integration: [f32; 4],
    pub(crate) viewport: [u32; 4],
}

impl TraceUniforms {
    pub(crate) fn from_observation(observation: &Observation) -> Result<Self, TracePackError> {
        let scene = observation.scene();
        let spacetime = *scene.spacetime();
        let mass = spacetime.mass_m();
        let projection = *observation.projection();
        let frame = scene.observer_frame();
        let sample = projection
            .sample(0, 0, 0.5, 0.5)
            .map_err(TracePackError::DomainInvariant)?;
        let observer_frequency = observation
            .initial_ray(sample)
            .map_err(TracePackError::DomainInvariant)?
            .observer_frequency();
        let horizon = spacetime
            .outer_horizon_radius()
            .map_or(-1.0, |radius| radius / mass);

        Ok(Self {
            spacetime: pack4(
                [
                    1.0,
                    spacetime.spin_m() / mass,
                    spacetime.charge_m() / mass,
                    horizon,
                ],
                "spacetime",
            )?,
            observer_event: pack4(
                scene.observer_event().to_txyz().map(|value| value / mass),
                "observer_event",
            )?,
            observer_velocity: pack4(frame.four_velocity_txyz(), "observer_velocity")?,
            image_right: pack4(frame.image_right_txyz(), "image_right")?,
            image_up: pack4(frame.image_up_txyz(), "image_up")?,
            arrival: pack4(frame.arrival_direction_txyz(), "arrival")?,
            projection_policy: pack4(
                [
                    (projection.vertical_fov().radians() * 0.5).tan(),
                    200.0,
                    f64::from(f32::from_bits(0x2b80_0000)),
                    observer_frequency,
                ],
                "projection_policy",
            )?,
            integration: [0.04, 0.02, 0.5, 0.05],
            viewport: [
                projection.width().get(),
                projection.height().get(),
                2_048,
                u32::from(spacetime.outer_horizon_radius().is_some()),
            ],
        })
    }
}

fn pack4(values: [f64; 4], field: &'static str) -> Result<[f32; 4], TracePackError> {
    let packed = values.map(|value| value as f32);
    if values.into_iter().all(f64::is_finite) && packed.into_iter().all(f32::is_finite) {
        Ok(packed)
    } else {
        Err(TracePackError::NotRepresentable { field })
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum TracePackError {
    #[error("validated observation failed to resolve its initial ray: {0}")]
    DomainInvariant(#[source] ValidationReport),
    #[error("observation field {field} is not representable by the interactive f32 contract")]
    NotRepresentable { field: &'static str },
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct TraceRecord {
    pub(crate) direction_time: [f32; 4],
    pub(crate) invariant_drift: [f32; 4],
    pub(crate) metadata: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TraceTermination {
    HorizonCrossing = 1,
    Escape = 2,
    SingularityGuard = 3,
    StepExhaustion = 4,
    NumericalFailure = 5,
    Uncertain = 6,
}

impl From<TraceTermination> for u32 {
    fn from(value: TraceTermination) -> Self {
        value as Self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown interactive trace termination discriminant {0}")]
pub struct UnknownTraceTermination(pub u32);

impl TryFrom<u32> for TraceTermination {
    type Error = UnknownTraceTermination;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::HorizonCrossing),
            2 => Ok(Self::Escape),
            3 => Ok(Self::SingularityGuard),
            4 => Ok(Self::StepExhaustion),
            5 => Ok(Self::NumericalFailure),
            6 => Ok(Self::Uncertain),
            unknown => Err(UnknownTraceTermination(unknown)),
        }
    }
}

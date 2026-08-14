use std::num::NonZeroU32;

use gravlume_domain::{
    Angle, KerrNewmanSpacetime, KerrSchildChart, Observation, PerspectiveView, PhysicalScene,
    PhysicalSceneInput, StationaryObserverInput, ValidationReport,
};

#[derive(Clone, Copy)]
pub struct Preview {
    mass_m: f64,
    spin_m: f64,
    observer_radius_m: f64,
    observer_polar_angle_rad: f64,
    vertical_fov_rad: f64,
}

pub const DEFAULT_PREVIEW: Preview = Preview {
    mass_m: 1.0,
    spin_m: 0.8,
    observer_radius_m: 30.0,
    observer_polar_angle_rad: std::f64::consts::FRAC_PI_3,
    vertical_fov_rad: std::f64::consts::FRAC_PI_4,
};

impl Preview {
    pub fn observation(self, width: u32, height: u32) -> Result<Observation, ValidationReport> {
        let spacetime =
            KerrNewmanSpacetime::new(self.mass_m, self.spin_m, 0.0, KerrSchildChart::Outgoing)?;
        let observer_xyz = spacetime.oblate_to_cartesian(
            self.observer_radius_m,
            self.observer_polar_angle_rad,
            0.0,
        );
        let observer = StationaryObserverInput::new(
            [0.0, observer_xyz[0], observer_xyz[1], observer_xyz[2]],
            [0.0; 4],
            [0.0, 0.0, 1.0],
            1.0,
        );
        let scene = PhysicalScene::new(PhysicalSceneInput::new(
            self.mass_m,
            self.spin_m,
            0.0,
            KerrSchildChart::Outgoing,
            observer,
        ))?;
        let view = PerspectiveView::new(
            NonZeroU32::new(width).unwrap_or(NonZeroU32::MIN),
            NonZeroU32::new(height).unwrap_or(NonZeroU32::MIN),
            Angle::from_radians(self.vertical_fov_rad)?,
        )?;
        Ok(Observation::new(scene, view))
    }

    pub const fn spin_ratio(self) -> f64 {
        self.spin_m / self.mass_m
    }

    pub const fn observer_radius_ratio(self) -> f64 {
        self.observer_radius_m / self.mass_m
    }

    pub const fn vertical_fov_degrees(self) -> f64 {
        self.vertical_fov_rad.to_degrees()
    }
}

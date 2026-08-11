use crate::{
    GeodesicState, KerrNewmanSpacetime, ObserverFrame, ParameterState, SpacetimeEvent,
    StationaryObserverDraft, ValidationIssue, ValidationIssueCode, ValidationReport,
    ViewportProjection, ViewportSample, math::FourVector, observer::StationaryObserver,
};

#[derive(Clone, Debug)]
pub struct PhysicalSceneDraft {
    mass_m: f64,
    spin_m: f64,
    charge_m: f64,
    observer: StationaryObserverDraft,
}

impl PhysicalSceneDraft {
    #[must_use]
    pub const fn new(
        mass_m: f64,
        spin_m: f64,
        charge_m: f64,
        observer: StationaryObserverDraft,
    ) -> Self {
        Self {
            mass_m,
            spin_m,
            charge_m,
            observer,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PhysicalScene {
    spacetime: KerrNewmanSpacetime,
    observer: StationaryObserver,
}

impl PhysicalScene {
    /// Validates and atomically commits a physical scene draft.
    ///
    /// # Errors
    ///
    /// Returns structured issues without producing a partial scene.
    pub fn commit(draft: PhysicalSceneDraft) -> Result<Self, ValidationReport> {
        let PhysicalSceneDraft {
            mass_m,
            spin_m,
            charge_m,
            observer,
        } = draft;
        let spacetime = KerrNewmanSpacetime::validated_with_prefix(
            mass_m,
            spin_m,
            charge_m,
            "physical_scene.spacetime",
        );
        let observer_report = observer.validate("physical_scene.observer");
        let mut report = ValidationReport::default();
        if let Err(error) = &spacetime {
            report.extend(error.clone());
        }
        report.extend(observer_report);
        if !report.is_empty() {
            return Err(report);
        }
        let spacetime = spacetime?;
        let observer = observer.build(spacetime, "physical_scene.observer")?;
        Ok(Self {
            spacetime,
            observer,
        })
    }

    #[must_use]
    pub const fn spacetime(&self) -> &KerrNewmanSpacetime {
        &self.spacetime
    }

    #[must_use]
    pub const fn parameter_state(&self) -> ParameterState {
        self.spacetime.parameter_state()
    }

    #[must_use]
    pub fn outer_horizon_radius(&self) -> Option<f64> {
        self.spacetime.outer_horizon_radius()
    }

    #[must_use]
    pub const fn observer_event(&self) -> SpacetimeEvent {
        self.observer.event()
    }

    #[must_use]
    pub const fn observer_frame(&self) -> &ObserverFrame {
        self.observer.frame()
    }

    #[must_use]
    pub const fn observer_metric_g_tt(&self) -> f64 {
        self.observer.metric_g_tt()
    }
}

#[derive(Clone, Debug)]
pub struct Observation {
    scene: PhysicalScene,
    projection: ViewportProjection,
}

impl Observation {
    /// Binds a validated physical scene to a validated viewport projection.
    ///
    /// # Errors
    ///
    /// Returns a report if a corrupted internal frame reaches this boundary.
    pub fn new(
        scene: PhysicalScene,
        projection: ViewportProjection,
    ) -> Result<Self, ValidationReport> {
        let frame = scene.observer_frame();
        if frame.gram_residual().is_finite()
            && frame.gram_residual() <= 1.0e-12
            && frame.orientation_determinant().is_finite()
            && frame.orientation_determinant() > 0.0
        {
            Ok(Self { scene, projection })
        } else {
            let mut report = ValidationReport::default();
            report.push(ValidationIssue::error(
                ValidationIssueCode::InternalInvariant,
                "observation.scene.observer_frame",
                "observer frame is not orthonormal and positively oriented",
            ));
            Err(report)
        }
    }

    #[must_use]
    pub const fn scene(&self) -> &PhysicalScene {
        &self.scene
    }

    #[must_use]
    pub const fn projection(&self) -> &ViewportProjection {
        &self.projection
    }

    /// Maps one viewport sample to its physical future-directed photon momentum.
    ///
    /// # Errors
    ///
    /// Revalidates the sample against this observation's projection and returns every seam issue.
    pub fn initial_ray(&self, sample: ViewportSample) -> Result<InitialViewRay, ValidationReport> {
        let [sight_x, sight_y] = self.projection.sight_plane(sample)?;
        let frame = self.scene.observer.frame();
        let normalization = 1.0_f64.hypot(sight_x.hypot(sight_y)).recip();
        let sight_direction = frame
            .image_right()
            .scaled(sight_x)
            .add(frame.image_up().scaled(sight_y))
            .subtract(frame.arrival())
            .scaled(normalization);
        let arrival_direction = sight_direction.scaled(-1.0);
        let momentum_contravariant = frame
            .four_velocity()
            .add(arrival_direction)
            .scaled(self.scene.observer.measured_frequency());
        let momentum_covariant = self.scene.observer.lower(momentum_contravariant);
        let event = self.scene.observer.event();
        let mut components = [0.0; 8];
        components[..4].copy_from_slice(&event.to_txyz());
        components[4..].copy_from_slice(&momentum_covariant);
        let state = GeodesicState::from_validated(components);
        let observer_frequency = -momentum_covariant
            .into_iter()
            .zip(frame.four_velocity().to_array())
            .map(|(covector, vector)| covector * vector)
            .sum::<f64>();
        Ok(InitialViewRay {
            state,
            sight_direction,
            observer_frequency,
            normalized_null_residual: self
                .scene
                .observer
                .normalized_null_residual(momentum_contravariant),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InitialViewRay {
    state: GeodesicState,
    sight_direction: FourVector,
    observer_frequency: f64,
    normalized_null_residual: f64,
}

impl InitialViewRay {
    #[must_use]
    pub const fn state(self) -> GeodesicState {
        self.state
    }

    #[must_use]
    pub const fn sight_direction_txyz(self) -> [f64; 4] {
        self.sight_direction.to_array()
    }

    #[must_use]
    pub const fn observer_frequency(self) -> f64 {
        self.observer_frequency
    }

    #[must_use]
    pub const fn normalized_null_residual(self) -> f64 {
        self.normalized_null_residual
    }

    #[must_use]
    pub const fn is_future_directed(self) -> bool {
        self.observer_frequency.is_finite() && self.observer_frequency > 0.0
    }
}

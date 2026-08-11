use crate::{
    GeodesicState, KerrNewmanSpacetime, ObserverFrame, ParameterState, SpacetimeEvent,
    StationaryObserverDraft, ValidationIssue, ValidationIssueCode, ValidationReport,
    ViewportProjection, ViewportSample, math::FourVector, observer::StationaryObserver,
};

const INITIAL_RAY_NULL_TOLERANCE: f64 = 2.0e-12;

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
    #[must_use]
    pub const fn new(scene: PhysicalScene, projection: ViewportProjection) -> Self {
        Self { scene, projection }
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
        if !momentum_contravariant.is_finite() {
            return Err(non_finite_initial_ray());
        }
        let momentum_covariant = self.scene.observer.lower(momentum_contravariant);
        let normalized_null_residual = self
            .scene
            .observer
            .normalized_null_residual(momentum_contravariant);
        let event = self.scene.observer.event();
        let observer_frequency = -momentum_covariant
            .into_iter()
            .zip(frame.four_velocity().to_array())
            .map(|(covector, vector)| covector * vector)
            .sum::<f64>();
        if !momentum_covariant.into_iter().all(f64::is_finite)
            || !observer_frequency.is_finite()
            || !normalized_null_residual.is_finite()
        {
            return Err(non_finite_initial_ray());
        }
        if observer_frequency <= 0.0 || normalized_null_residual > INITIAL_RAY_NULL_TOLERANCE {
            return Err(invalid_initial_ray());
        }
        let mut components = [0.0; 8];
        components[..4].copy_from_slice(&event.to_txyz());
        components[4..].copy_from_slice(&momentum_covariant);
        let state = GeodesicState::from_validated(components);
        Ok(InitialViewRay {
            state,
            sight_direction,
            observer_frequency,
            normalized_null_residual,
        })
    }
}

fn non_finite_initial_ray() -> ValidationReport {
    let mut report = ValidationReport::default();
    report.push(ValidationIssue::error(
        ValidationIssueCode::NonFinite,
        "observation.initial_ray",
        "derived photon momentum and diagnostics must be finite",
    ));
    report
}

fn invalid_initial_ray() -> ValidationReport {
    let mut report = ValidationReport::default();
    report.push(ValidationIssue::error(
        ValidationIssueCode::InternalInvariant,
        "observation.initial_ray",
        "derived photon momentum must be future-directed and null within the domain budget",
    ));
    report
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
}

use crate::{
    EquatorialCircularEmitter, EquatorialEmissionModel, Extremality, GeodesicState,
    HomogeneousScalarSlab, ImageSample, KerrNewmanSpacetime, KerrSchildChart, ObserverFrame,
    PerspectiveView, ScalarSlabEmissionModel, SpacetimeEvent, StationaryObserverInput,
    ValidationIssueCode, ValidationReport, observer::StationaryObserver, state::FourVector,
};

const INITIAL_RAY_NULL_TOLERANCE: f64 = 2.0e-12;

#[derive(Clone, Debug)]
pub struct PhysicalSceneInput {
    mass_m: f64,
    spin_m: f64,
    charge_m: f64,
    chart: KerrSchildChart,
    observer: StationaryObserverInput,
}

impl PhysicalSceneInput {
    #[must_use]
    pub const fn new(
        mass_m: f64,
        spin_m: f64,
        charge_m: f64,
        chart: KerrSchildChart,
        observer: StationaryObserverInput,
    ) -> Self {
        Self {
            mass_m,
            spin_m,
            charge_m,
            chart,
            observer,
        }
    }
}

/// The complete observer-path transport applied after an equatorial surface hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SurfaceTransport {
    Vacuum,
    HomogeneousScalar(HomogeneousScalarSlab),
}

/// A validated equatorial emitter and its complete observer-path transport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EquatorialSurface {
    emitter: EquatorialCircularEmitter,
    transport: SurfaceTransport,
}

impl EquatorialSurface {
    /// Validates one emitter and its complete observer-path transport as a single value.
    ///
    /// # Errors
    ///
    /// Rejects a source/transport combination whose resolved spectral meaning is incomplete.
    pub fn new(
        emitter: EquatorialCircularEmitter,
        transport: SurfaceTransport,
    ) -> Result<Self, ValidationReport> {
        if let SurfaceTransport::HomogeneousScalar(slab) = transport
            && matches!(
                emitter.emission_model(),
                EquatorialEmissionModel::InverseCubeBlackbodyV1 { .. }
            )
            && slab.integrated_bolometric_emission() > 0.0
            && slab.emission_model() == ScalarSlabEmissionModel::NeutralBolometric
        {
            return Err(ValidationReport::from_error(
                ValidationIssueCode::IncompatibleModel,
                "equatorial_surface.transport.emission_model",
                "blackbody surface transport with nonzero emission requires a resolved spectrum",
            ));
        }
        Ok(Self { emitter, transport })
    }

    #[must_use]
    pub const fn emitter(self) -> EquatorialCircularEmitter {
        self.emitter
    }

    #[must_use]
    pub const fn transport(self) -> SurfaceTransport {
        self.transport
    }
}

/// The radiance interpretation selected by a validated physical scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SceneRadiance {
    /// Orientation-only analytic escape sky, not a spectral source.
    AnalyticSky,
    /// A resolved equatorial source with atomically validated transport.
    EquatorialSurface(EquatorialSurface),
}

#[derive(Clone, Debug)]
pub struct PhysicalScene {
    spacetime: KerrNewmanSpacetime,
    observer: StationaryObserver,
    radiance: SceneRadiance,
}

impl PhysicalScene {
    /// Validates physical scene input without producing a partial scene.
    ///
    /// # Errors
    ///
    /// Returns structured issues without producing a partial scene.
    pub fn new(input: PhysicalSceneInput) -> Result<Self, ValidationReport> {
        let PhysicalSceneInput {
            mass_m,
            spin_m,
            charge_m,
            chart,
            observer,
        } = input;
        let spacetime = KerrNewmanSpacetime::validated_with_prefix(
            mass_m,
            spin_m,
            charge_m,
            chart,
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
            radiance: SceneRadiance::AnalyticSky,
        })
    }

    #[must_use]
    pub const fn spacetime(&self) -> &KerrNewmanSpacetime {
        &self.spacetime
    }

    #[must_use]
    pub const fn extremality(&self) -> Extremality {
        self.spacetime.extremality()
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

    /// Returns an equivalent scene with one validated equatorial source chain installed.
    #[must_use]
    pub const fn with_equatorial_surface(mut self, surface: EquatorialSurface) -> Self {
        self.radiance = SceneRadiance::EquatorialSurface(surface);
        self
    }

    #[must_use]
    pub const fn radiance(&self) -> SceneRadiance {
        self.radiance
    }
}

#[derive(Clone, Debug)]
pub struct Observation {
    scene: PhysicalScene,
    view: PerspectiveView,
}

impl Observation {
    /// Binds a validated physical scene to an image-space perspective view.
    #[must_use]
    pub const fn new(scene: PhysicalScene, view: PerspectiveView) -> Self {
        Self { scene, view }
    }

    #[must_use]
    pub const fn scene(&self) -> &PhysicalScene {
        &self.scene
    }

    #[must_use]
    pub const fn view(&self) -> &PerspectiveView {
        &self.view
    }

    /// Maps one image sample to its physical future-directed photon momentum.
    ///
    /// # Errors
    ///
    /// Revalidates the sample against this observation's view and returns every seam issue.
    pub fn initial_ray(&self, sample: ImageSample) -> Result<InitialViewRay, ValidationReport> {
        let [sight_x, sight_y] = self.view.sight_plane(sample)?;
        let non_finite = || {
            ValidationReport::from_error(
                ValidationIssueCode::NonFinite,
                "observation.initial_ray",
                "derived photon momentum and diagnostics must be finite",
            )
        };
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
            return Err(non_finite());
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
            return Err(non_finite());
        }
        if observer_frequency <= 0.0 || normalized_null_residual > INITIAL_RAY_NULL_TOLERANCE {
            return Err(ValidationReport::from_error(
                ValidationIssueCode::InternalInvariant,
                "observation.initial_ray",
                "derived photon momentum must be future-directed and null within the domain budget",
            ));
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

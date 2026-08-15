use gravlume_domain::{EquatorialCircularEmitter, HomogeneousScalarSlab, Observation};
use gravlume_reference::{
    FixtureDocument, ObservationTrace, ObservationTracer, ReferenceComparison, ReferencePolicy,
    SurfaceFootprintError, SurfaceFootprintEstimate, SurfaceParity, TraceInputId,
};

const SURFACE_OBSERVABLE: &str = include_str!("../fixtures/v2/kerr-surface-observable.toml");
const TRANSPORT_FIXTURES: [&str; 4] = [
    include_str!("../fixtures/v3/kerr-blackbody-vacuum.toml"),
    include_str!("../fixtures/v3/kerr-blackbody-pure-absorption.toml"),
    include_str!("../fixtures/v3/kerr-blackbody-constant-slab.toml"),
    include_str!("../fixtures/v3/kerr-blackbody-pure-emission.toml"),
];

#[test]
fn regular_and_strict_surface_observables_close_the_vacuum_radiance_chain() {
    let fixture = FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let regular = ObservationTracer::baseline_v1()
        .trace(
            fixture
                .trace_request(ReferencePolicy::regular_v1())
                .expect("fixture sample resolves through its observation"),
        )
        .expect("the regular source model is valid at the localized hit");
    let strict = ObservationTracer::baseline_v1()
        .trace(
            fixture
                .trace_request(ReferencePolicy::strict_v1())
                .expect("fixture sample resolves through its observation"),
        )
        .expect("the strict source model is valid at the localized hit");
    let comparison = ReferenceComparison::baseline_v1(&regular, &strict)
        .expect("policy roles and surface input identity match");

    assert!(fixture.accepts(&regular));
    assert!(fixture.accepts(&strict));
    assert!(comparison.is_accepted(), "{:?}", comparison.issues());
}

#[test]
fn versioned_transport_fixtures_close_under_regular_and_strict_geometry() {
    for source in TRANSPORT_FIXTURES {
        let fixture = FixtureDocument::parse_toml(source)
            .expect("repository transport fixture parses")
            .into_surface_observation()
            .expect("transport fixture is a surface observation");
        let trace = |policy| {
            ObservationTracer::baseline_v1()
                .trace(
                    fixture
                        .trace_request(policy)
                        .expect("fixture sample resolves through its observation"),
                )
                .expect("fixture transport model is valid at the localized hit")
        };
        let regular = trace(ReferencePolicy::regular_v1());
        let strict = trace(ReferencePolicy::strict_v1());
        let comparison = ReferenceComparison::baseline_v1(&regular, &strict)
            .expect("policy roles and transport input identity match");

        assert!(
            fixture.accepts(&regular),
            "rejected regular fixture {}",
            fixture.input_id()
        );
        assert!(
            fixture.accepts(&strict),
            "rejected strict fixture {}",
            fixture.input_id()
        );
        assert!(comparison.is_accepted(), "{:?}", comparison.issues());
    }
}

#[test]
fn canonical_surface_footprint_is_resolved_only_with_one_exact_branch_key() {
    let fixture = FixtureDocument::parse_toml(TRANSPORT_FIXTURES[0])
        .expect("repository transport fixture parses")
        .into_surface_observation()
        .expect("transport fixture is a surface observation");
    let estimate = ObservationTracer::baseline_v1()
        .surface_footprint_v1(
            fixture.observation(),
            fixture.sample(),
            ReferencePolicy::regular_v1(),
        )
        .expect("the centered quarter-pixel neighborhood traces successfully");
    let SurfaceFootprintEstimate::Resolved(footprint) = estimate else {
        panic!("canonical regular source neighborhood must be branch-continuous");
    };

    assert!(
        footprint
            .jacobian_source_m_per_pixel()
            .into_iter()
            .flatten()
            .all(f64::is_finite)
    );
    let [major, minor] = footprint.singular_values_m_per_pixel();
    assert!(major >= minor && minor > 0.0);
    assert_ne!(footprint.parity(), SurfaceParity::Degenerate);
    assert!(footprint.branch_key().equatorial_crossings() > 0);
}

#[test]
fn surface_footprint_rejects_a_neighborhood_outside_one_pixel() {
    let fixture = FixtureDocument::parse_toml(TRANSPORT_FIXTURES[0])
        .expect("repository transport fixture parses")
        .into_surface_observation()
        .expect("transport fixture is a surface observation");
    let [pixel_x, pixel_y] = fixture.sample().pixel();
    let sample = fixture
        .observation()
        .view()
        .sample(pixel_x, pixel_y, 0.125, 0.5)
        .expect("the off-center sample is valid");
    let error = ObservationTracer::baseline_v1()
        .surface_footprint_v1(fixture.observation(), sample, ReferencePolicy::regular_v1())
        .expect_err("a centered quarter-pixel stencil would cross the pixel boundary");

    assert!(matches!(
        error,
        SurfaceFootprintError::NeighborhoodOutsidePixel
    ));
}

#[test]
fn optical_depth_slab_closes_vacuum_absorption_and_constant_source_limits() {
    let fixture = FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let vacuum = trace_modified_surface(&fixture, None, "surface-vacuum");
    let vacuum_observable = vacuum
        .surface_observable()
        .expect("the canonical sample hits the source");

    let optical_depth = 0.75;
    let absorption = trace_modified_surface(
        &fixture,
        Some(
            HomogeneousScalarSlab::pure_absorption_v1(optical_depth)
                .expect("pure absorption is valid"),
        ),
        "surface-pure-absorption",
    );
    let absorption_observable = absorption
        .surface_observable()
        .expect("the absorbed ray still has a source observable");
    let expected_absorption =
        vacuum_observable.observed_bolometric_intensity() * (-optical_depth).exp();
    assert!(
        (absorption_observable.observed_bolometric_intensity() - expected_absorption).abs()
            <= 8.0 * f64::EPSILON
    );

    let source_function = 0.125;
    let slab = HomogeneousScalarSlab::constant_bolometric_v1(optical_depth, source_function)
        .expect("constant slab is valid");
    let transported = trace_modified_surface(&fixture, Some(slab), "surface-constant-slab");
    let transported = transported
        .surface_observable()
        .expect("the transported ray has a source observable");
    let transmittance = (-optical_depth).exp();
    let expected = vacuum_observable
        .observed_bolometric_intensity()
        .mul_add(transmittance, source_function * -(-optical_depth).exp_m1());
    assert!((transported.observed_bolometric_intensity() - expected).abs() <= 8.0 * f64::EPSILON);
    assert_eq!(
        transported.optical_depth().to_bits(),
        optical_depth.to_bits()
    );
    assert_eq!(
        transported.vacuum_observed_bolometric_intensity().to_bits(),
        vacuum_observable.observed_bolometric_intensity().to_bits()
    );
}

#[test]
fn optically_thin_slab_uses_a_non_cancelling_emission_weight() {
    let fixture = FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let optical_depth = f64::EPSILON;
    let source_function = 1.0e16;
    let slab = HomogeneousScalarSlab::constant_bolometric_v1(optical_depth, source_function)
        .expect("thin slab is valid");
    let outcome = trace_modified_surface(&fixture, Some(slab), "surface-thin-slab");
    let observable = outcome
        .surface_observable()
        .expect("the transported ray has a source observable");
    let expected = observable.vacuum_observed_bolometric_intensity().mul_add(
        (-optical_depth).exp(),
        source_function * -(-optical_depth).exp_m1(),
    );

    assert!(expected > 1.0);
    assert_eq!(
        observable.observed_bolometric_intensity().to_bits(),
        expected.to_bits()
    );
}

#[test]
fn zero_absorption_preserves_the_finite_pure_emission_limit() {
    let fixture = FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let integrated_emission = 0.375;
    let slab = HomogeneousScalarSlab::pure_emission_bolometric_v1(integrated_emission)
        .expect("pure emission is valid without dividing by absorption");
    let outcome = trace_modified_surface(&fixture, Some(slab), "surface-pure-emission");
    let observable = outcome
        .surface_observable()
        .expect("the pure-emission ray has a source observable");

    assert_eq!(observable.optical_depth().to_bits(), 0.0_f64.to_bits());
    let expected = observable.vacuum_observed_bolometric_intensity() + integrated_emission;
    assert_eq!(
        observable.observed_bolometric_intensity().to_bits(),
        expected.to_bits()
    );
}

#[test]
fn blackbody_surface_shifts_temperature_and_produces_named_spectral_bands() {
    let fixture = FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let emitter = EquatorialCircularEmitter::inverse_cube_blackbody_v1(6.0, 20.0, 1.0, 6_000.0)
        .expect("blackbody surface is valid");
    let observation = Observation::new(
        fixture
            .observation()
            .scene()
            .clone()
            .with_equatorial_circular_emitter(emitter),
        *fixture.observation().view(),
    );
    let outcome = ObservationTracer::baseline_v1()
        .trace(
            ObservationTrace::new(
                TraceInputId::new("surface-blackbody"),
                &observation,
                fixture.sample(),
                ReferencePolicy::regular_v1(),
            )
            .expect("fixture sample resolves"),
        )
        .expect("blackbody transport is valid");
    let observable = outcome
        .surface_observable()
        .expect("the canonical sample hits the blackbody surface");
    let radius_ratio = observable.source_anchor().radius_m() / 6.0;
    let emitted_temperature = 6_000.0 / (radius_ratio * radius_ratio.sqrt()).sqrt();
    let observed_temperature = emitted_temperature * observable.frequency_ratio().value();

    assert_eq!(
        observable.emitted_temperature_kelvin(),
        Some(emitted_temperature)
    );
    assert_eq!(
        observable.vacuum_observed_temperature_kelvin(),
        Some(observed_temperature)
    );
    let bands = observable
        .observed_spectral_band_intensities()
        .expect("blackbody transport resolves the versioned RGB boxcar instrument");
    assert!(
        bands
            .into_iter()
            .all(|value| value.is_finite() && value >= 0.0)
    );
    assert!(bands.into_iter().sum::<f64>() < observable.observed_bolometric_intensity());
}

fn trace_modified_surface(
    fixture: &gravlume_reference::SurfaceObservationFixture,
    slab: Option<HomogeneousScalarSlab>,
    id: &'static str,
) -> gravlume_reference::ReferenceOutcome {
    let mut scene = fixture.observation().scene().clone();
    if let Some(slab) = slab {
        scene = scene.with_homogeneous_scalar_slab(slab);
    }
    let observation = Observation::new(scene, *fixture.observation().view());
    ObservationTracer::baseline_v1()
        .trace(
            ObservationTrace::new(
                TraceInputId::new(id),
                &observation,
                fixture.sample(),
                ReferencePolicy::regular_v1(),
            )
            .expect("fixture sample resolves"),
        )
        .expect("surface transport is valid")
}

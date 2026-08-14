use gravlume_reference::{
    EquatorialCircularSurface, FixtureDocument, FixtureError, ObservationTracer,
    ReferenceComparison, ReferencePolicy,
};

const SURFACE_OBSERVABLE: &str = include_str!("../fixtures/v2/kerr-surface-observable.toml");

#[test]
fn equatorial_circular_surface_rejects_invalid_radial_intervals() {
    for (inner_radius_m, outer_radius_m) in [
        (f64::NAN, 20.0),
        (6.0, f64::INFINITY),
        (0.0, 20.0),
        (20.0, 6.0),
    ] {
        assert!(EquatorialCircularSurface::new(inner_radius_m, outer_radius_m).is_err());
    }
}

#[test]
fn surface_fixture_schema_is_strict_versioned_and_canonical() {
    let with_unknown = SURFACE_OBSERVABLE.replacen(
        "schema_version = 2",
        "schema_version = 2\nundeclared = true",
        1,
    );
    let unsupported = SURFACE_OBSERVABLE.replacen("schema_version = 2", "schema_version = 3", 1);
    let drifted = SURFACE_OBSERVABLE.replacen(
        "frequency_ratio = \"0.953264138194626409\"",
        "frequency_ratio = \"0.95\"",
        1,
    );

    assert!(matches!(
        FixtureDocument::parse_toml(&with_unknown),
        Err(FixtureError::Toml(_))
    ));
    assert!(matches!(
        FixtureDocument::parse_toml(&unsupported),
        Err(FixtureError::UnsupportedSchema(3))
    ));
    assert!(matches!(
        FixtureDocument::parse_toml(&drifted),
        Err(FixtureError::PresetMismatch { .. })
    ));
}

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
                .expect("fixture request remains valid"),
        )
        .expect("the regular source model is valid at the localized hit");
    let strict = ObservationTracer::baseline_v1()
        .trace(
            fixture
                .trace_request(ReferencePolicy::strict_v1())
                .expect("fixture request remains valid"),
        )
        .expect("the strict source model is valid at the localized hit");
    let comparison = ReferenceComparison::baseline_v1(&regular, &strict)
        .expect("policy roles and surface input identity match");

    assert!(fixture.expected().accepts(&regular));
    assert!(fixture.expected().accepts(&strict));
    assert!(comparison.is_accepted(), "{:?}", comparison.issues());
    assert!(comparison.source_anchor_distance_m().is_some());
    assert!(comparison.frequency_ratio_relative_error().is_some());

    let observable = regular
        .surface_observable()
        .expect("surface termination carries its physical observable");
    let anchor = observable
        .source_anchor()
        .as_equatorial_surface()
        .expect("the configured source is an equatorial surface");
    assert!((6.0..=20.0).contains(&anchor.radius_m()));
    assert!(anchor.azimuth_rad().is_finite());
    assert!(observable.frequency_ratio().value() > 0.0);

    let emitted_bolometric_intensity = (anchor.radius_m() / 6.0).powi(-3);
    let observed_bolometric_intensity = observable
        .vacuum_observed_bolometric_intensity(emitted_bolometric_intensity)
        .expect("the validation fixture radiance is finite and non-negative");
    let expected = observable.frequency_ratio().value().powi(4) * emitted_bolometric_intensity;
    assert_eq!(observed_bolometric_intensity.to_bits(), expected.to_bits());
    for invalid in [f64::NAN, f64::INFINITY, -f64::MIN_POSITIVE] {
        assert!(
            observable
                .vacuum_observed_bolometric_intensity(invalid)
                .is_err()
        );
    }
}

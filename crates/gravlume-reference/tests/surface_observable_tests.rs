use gravlume_reference::{
    FixtureDocument, ObservationTracer, ReferenceComparison, ReferencePolicy,
    SurfaceFootprintError, SurfaceFootprintEstimate, SurfaceParity,
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

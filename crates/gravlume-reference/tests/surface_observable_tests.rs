use gravlume_reference::{
    FixtureDocument, ObservationTracer, ReferenceComparison, ReferencePolicy,
};

const SURFACE_OBSERVABLE: &str = include_str!("../fixtures/v2/kerr-surface-observable.toml");

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

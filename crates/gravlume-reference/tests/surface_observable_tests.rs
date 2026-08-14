use gravlume_reference::{
    EquatorialCircularSurface, FixtureDocument, ObservationTracer, ReferenceComparison,
    ReferencePolicy,
};
use proptest::prelude::*;

const SURFACE_OBSERVABLE: &str = include_str!("../fixtures/v2/kerr-surface-observable.toml");

proptest! {
    #[test]
    fn equatorial_circular_surface_accepts_exactly_its_documented_domain(
        inner_radius_m in proptest::num::f64::ANY,
        outer_radius_m in proptest::num::f64::ANY,
    ) {
        let is_valid = inner_radius_m.is_finite()
            && outer_radius_m.is_finite()
            && inner_radius_m > 0.0
            && outer_radius_m >= inner_radius_m;

        prop_assert_eq!(
            EquatorialCircularSurface::new(inner_radius_m, outer_radius_m).is_ok(),
            is_valid
        );
    }
}

#[test]
fn regular_and_strict_surface_observables_close_the_vacuum_radiance_chain() {
    let fixture = FixtureDocument::parse_toml(SURFACE_OBSERVABLE)
        .expect("repository surface fixture parses")
        .into_surface_observation()
        .expect("fixture is a surface observation");
    let regular = ObservationTracer::baseline_v1()
        .trace(fixture.trace_request(ReferencePolicy::regular_v1()))
        .expect("the regular source model is valid at the localized hit");
    let strict = ObservationTracer::baseline_v1()
        .trace(fixture.trace_request(ReferencePolicy::strict_v1()))
        .expect("the strict source model is valid at the localized hit");
    let comparison = ReferenceComparison::baseline_v1(&regular, &strict)
        .expect("policy roles and surface input identity match");

    assert!(fixture.accepts(&regular));
    assert!(fixture.accepts(&strict));
    assert!(comparison.is_accepted(), "{:?}", comparison.issues());
}

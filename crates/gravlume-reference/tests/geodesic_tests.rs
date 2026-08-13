use gravlume_domain::{GeodesicState, KerrNewmanSpacetime, KerrSchildChart};
use gravlume_reference::{
    AffineDirection, EventConfiguration, GeodesicConfigurationError, GeodesicTrace, GeodesicTracer,
    ReferencePolicy, Termination, TraceInputId,
};

#[test]
fn reference_policy_ids_are_stable_and_strict_refines_every_bound() {
    let regular = ReferencePolicy::regular_v1();
    let strict = ReferencePolicy::strict_v1();

    assert_eq!(regular.id(), "reference-regular-v1");
    assert_eq!(strict.id(), "reference-strict-v1");

    for (regular_tolerance, strict_tolerance) in [
        (
            regular.position_relative_tolerance(),
            strict.position_relative_tolerance(),
        ),
        (
            regular.position_absolute_tolerance(),
            strict.position_absolute_tolerance(),
        ),
        (
            regular.momentum_relative_tolerance(),
            strict.momentum_relative_tolerance(),
        ),
        (
            regular.momentum_absolute_tolerance(),
            strict.momentum_absolute_tolerance(),
        ),
        (
            regular.event_affine_tolerance_m(),
            strict.event_affine_tolerance_m(),
        ),
        (
            regular.event_tie_tolerance_m(),
            strict.event_tie_tolerance_m(),
        ),
    ] {
        assert!(strict_tolerance < regular_tolerance);
    }
    assert!(strict.maximum_step_m() < regular.maximum_step_m());
    assert!(strict.maximum_accepted_steps() > regular.maximum_accepted_steps());
    assert!(strict.maximum_consecutive_rejects() > regular.maximum_consecutive_rejects());
}

#[test]
fn v1_reference_seam_rejects_a_non_normalized_mass_scale() {
    for mass_m in [2.0, 1.0_f64.next_up()] {
        let spacetime = KerrNewmanSpacetime::new(mass_m, 0.0, 0.0, KerrSchildChart::Ingoing)
            .expect("spacetime is valid");
        let error = GeodesicTracer::new(
            spacetime,
            ReferencePolicy::regular_v1(),
            EventConfiguration::horizon_only(),
        )
        .expect_err("v1 requires exact M = 1");

        assert_eq!(error, GeodesicConfigurationError::NonNormalizedMass);
    }
}

#[test]
fn weak_field_scattering_converges_to_the_leading_four_m_over_b_deflection() {
    let mass_m = 1.0_f64;
    let boundary_radius_m = 1_000.0_f64;
    let impact_parameter_m = 50.0_f64;
    let tangential_momentum = impact_parameter_m / boundary_radius_m;
    let scalar_f = 2.0 * mass_m / boundary_radius_m;
    let quadratic = scalar_f.mul_add(
        scalar_f,
        -(1.0 - scalar_f) * tangential_momentum.mul_add(tangential_momentum, -1.0 - scalar_f),
    );
    let radial_momentum = (scalar_f - quadratic.sqrt()) / (1.0 - scalar_f);
    let state = GeodesicState::new(
        [0.0, boundary_radius_m, 0.0, 0.0],
        [-1.0, radial_momentum, tangential_momentum, 0.0],
    )
    .expect("constructed state is finite");
    let spacetime = KerrNewmanSpacetime::new(mass_m, 0.0, 0.0, KerrSchildChart::Ingoing)
        .expect("spacetime is valid");
    let events =
        EventConfiguration::with_escape_radius(boundary_radius_m).expect("escape surface is valid");
    let outcome = GeodesicTracer::new(spacetime, ReferencePolicy::regular_v1(), events)
        .expect("mass is normalized")
        .trace(GeodesicTrace::new(
            TraceInputId::new("weak-field-scattering"),
            state,
            AffineDirection::Positive,
        ));

    assert_eq!(outcome.termination(), Termination::Escape);
    let flat_boundary_angle = 2.0_f64.mul_add(
        -(impact_parameter_m / boundary_radius_m).asin(),
        std::f64::consts::PI,
    );
    let deflection = outcome.azimuth_advance_rad() - flat_boundary_angle;
    let leading_order = 4.0 * mass_m / impact_parameter_m;
    assert!((deflection - leading_order).abs() / leading_order < 0.08);
}

#[test]
fn equatorial_surface_is_localized_as_a_distinct_terminal_event() {
    let spacetime = KerrNewmanSpacetime::new(1.0, 0.0, 0.0, KerrSchildChart::Ingoing)
        .expect("spacetime is valid");
    let state = GeodesicState::new([0.0, 50.0, 0.0, 1.0], [-1.0, -0.9, 0.0, -0.2])
        .expect("state is finite");
    let events = EventConfiguration::horizon_only()
        .with_equatorial_surface(5.0, 60.0)
        .expect("surface is valid");
    let outcome = GeodesicTracer::new(spacetime, ReferencePolicy::regular_v1(), events)
        .expect("mass is normalized")
        .trace(GeodesicTrace::new(
            TraceInputId::new("equatorial-surface"),
            state,
            AffineDirection::Positive,
        ));

    assert_eq!(outcome.termination(), Termination::EquatorialSurface);
    assert!(outcome.state().components()[3].abs() < 2.0e-11);
}

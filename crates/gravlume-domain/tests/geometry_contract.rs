use gravlume_domain::{
    GeometryError, KerrNewmanSpacetime, ParameterState, SpacetimeEvent, ValidationIssueCode,
};

#[test]
fn parameter_state_and_horizon_are_classified_without_clamping() {
    let subextremal = KerrNewmanSpacetime::new(1.0, 0.8, 0.0).expect("parameters are valid");
    let extremal = KerrNewmanSpacetime::new(1.0, 0.8, 0.6).expect("parameters are valid");
    let superextremal = KerrNewmanSpacetime::new(1.0, 1.0, 0.5).expect("parameters are valid");

    assert_eq!(subextremal.parameter_state(), ParameterState::Subextremal);
    assert_eq!(extremal.parameter_state(), ParameterState::Extremal);
    assert_eq!(
        superextremal.parameter_state(),
        ParameterState::Superextremal
    );
    assert!((subextremal.outer_horizon_radius().expect("horizon exists") - 1.6).abs() < 2.0e-15);
    assert_eq!(extremal.outer_horizon_radius(), Some(1.0));
    assert_eq!(superextremal.outer_horizon_radius(), None);
}

#[test]
fn finite_extreme_scales_preserve_extremality_and_horizon_classification() {
    for mass_m in [f64::MAX / 2.0, f64::MIN_POSITIVE] {
        let extremal =
            KerrNewmanSpacetime::new(mass_m, mass_m, 0.0).expect("parameters are finite");

        assert_eq!(extremal.parameter_state(), ParameterState::Extremal);
        assert_eq!(extremal.outer_horizon_radius(), Some(mass_m));
    }
}

#[test]
fn validated_spacetime_rejects_an_unrepresentable_outer_horizon() {
    let report = KerrNewmanSpacetime::new(f64::MAX, 0.0, 0.0)
        .expect_err("validated geometry must have a finite outer horizon");

    assert!(report.issues().iter().any(|issue| {
        issue.code() == ValidationIssueCode::NonFinite
            && issue.field_path() == "spacetime.outer_horizon_radius_m"
    }));
}

#[test]
fn oblate_radius_and_rank_one_metric_inverse_satisfy_the_algebra_contract() {
    let spacetime = KerrNewmanSpacetime::new(1.0, 0.8, 0.2).expect("parameters are valid");

    for event in [
        SpacetimeEvent::from_txyz([0.0, 30.0, 2.0, 5.0]).expect("event is finite"),
        SpacetimeEvent::from_txyz([1.0, 0.0, 0.0, 10.0]).expect("event is finite"),
        SpacetimeEvent::from_txyz([2.0, 20.0, -4.0, 0.0]).expect("event is finite"),
    ] {
        let radius = spacetime.radius(event).expect("point is outside the ring");
        let [_, x, y, z] = event.to_txyz();
        let transverse_squared = y.mul_add(y, x * x);
        let radial_denominator = 0.8_f64.mul_add(0.8, radius * radius);
        let identity = transverse_squared / radial_denominator + z * z / radius.powi(2);

        assert!((identity - 1.0).abs() < 2.0e-13);
        assert!(
            spacetime
                .metric_inverse_residual(event)
                .expect("metric is finite")
                < 2.0e-12
        );
    }
}

#[test]
fn metric_inverse_residual_is_term_normalized_under_strong_cancellation() {
    let spacetime = KerrNewmanSpacetime::new(1.0, 0.0, 0.0).expect("parameters are valid");
    let event = SpacetimeEvent::from_txyz([0.0, 1.0e-4, 2.0e-4, 3.0e-4]).expect("event is finite");
    let residual = spacetime
        .metric_inverse_residual(event)
        .expect("metric is finite");

    assert!(residual < 2.0e-12, "normalized residual was {residual:e}");
}

#[test]
fn metric_inverse_residual_preserves_representable_values_when_term_norm_overflows() {
    let spacetime = KerrNewmanSpacetime::new(5.0e113, 0.0, 0.0).expect("parameters are valid");
    let event = SpacetimeEvent::from_txyz([0.0, 1.0e-40, 0.0, 0.0]).expect("event is finite");
    let residual = spacetime
        .metric_inverse_residual(event)
        .expect("scaled metric contraction is finite");

    assert!(residual.is_finite() && residual > 0.0 && residual < f64::MIN_POSITIVE);
}

#[test]
fn schwarzschild_limit_is_spherical_and_stationary() {
    let spacetime = KerrNewmanSpacetime::new(2.0, 0.0, 0.0).expect("parameters are valid");
    let event = SpacetimeEvent::from_txyz([7.0, 3.0, 4.0, 12.0]).expect("event is finite");

    assert!((spacetime.radius(event).expect("radius exists") - 13.0).abs() < f64::EPSILON);
    assert!(
        (spacetime
            .metric_component_tt(event)
            .expect("metric is finite")
            + 9.0 / 13.0)
            .abs()
            < 2.0e-15
    );
}

#[test]
fn representable_far_field_coordinates_do_not_overflow_internal_squares() {
    let spacetime = KerrNewmanSpacetime::new(1.0, 0.0, 0.0).expect("parameters are valid");

    for coordinate in [1.0e100, 1.0e200] {
        let event =
            SpacetimeEvent::from_txyz([0.0, coordinate, 0.0, 0.0]).expect("event is finite");

        assert_eq!(
            spacetime.radius(event),
            Ok(coordinate),
            "radius failed at {coordinate:e}"
        );
        assert_eq!(
            spacetime.metric_component_tt(event),
            Ok(-1.0),
            "metric failed at {coordinate:e}"
        );
        assert_eq!(
            spacetime.singularity_guard_residual(event, 1.0),
            Ok(f64::MAX),
            "guard side failed at {coordinate:e}"
        );
    }
}

#[test]
fn reissner_nordstrom_and_minkowski_limits_match_closed_form_g_tt() {
    let event = SpacetimeEvent::from_txyz([0.0, 10.0, 0.0, 0.0]).expect("event is finite");
    let reissner_nordstrom = KerrNewmanSpacetime::new(1.0, 0.0, 0.6).expect("parameters are valid");
    let expected_g_tt = -1.0 + 2.0 / 10.0 - 0.6_f64.powi(2) / 10.0_f64.powi(2);
    assert!(
        (reissner_nordstrom
            .metric_component_tt(event)
            .expect("metric is finite")
            - expected_g_tt)
            .abs()
            < 4.0 * f64::EPSILON
    );

    let minkowski_limit =
        KerrNewmanSpacetime::new(1.0e-12, 0.0, 0.0).expect("positive mass is valid");
    assert!(
        (minkowski_limit
            .metric_component_tt(event)
            .expect("metric is finite")
            + 1.0)
            .abs()
            < 2.1e-13
    );
    assert!(
        minkowski_limit
            .metric_inverse_residual(event)
            .expect("metric is finite")
            < 4.0 * f64::EPSILON
    );
}

#[test]
fn ring_singularity_and_nonnegative_radius_branch_disk_are_distinct_failures() {
    let spacetime = KerrNewmanSpacetime::new(1.0, 0.8, 0.0).expect("parameters are valid");
    let ring = SpacetimeEvent::from_txyz([0.0, 0.8, 0.0, 0.0]).expect("event is finite");
    let branch_disk = SpacetimeEvent::from_txyz([0.0, 0.0, 0.0, 0.0]).expect("event is finite");

    assert_eq!(spacetime.radius(ring), Err(GeometryError::RingSingularity));
    assert_eq!(
        spacetime.radius(branch_disk),
        Err(GeometryError::ChartBoundary)
    );

    let point_just_inside_ring =
        SpacetimeEvent::from_txyz([0.0, 0.8_f64.next_down(), 0.0, 0.0]).expect("event is finite");
    assert_eq!(
        spacetime.radius(point_just_inside_ring),
        Err(GeometryError::ChartBoundary)
    );
}

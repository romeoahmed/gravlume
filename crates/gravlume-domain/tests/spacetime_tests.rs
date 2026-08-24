use approx::{abs_diff_eq, assert_abs_diff_eq};
use gravlume_domain::{
    Extremality, GeometryError, KerrNewmanSpacetime, KerrSchildChart, SpacetimeEvent,
    ValidationIssueCode,
};
use proptest::prelude::*;

#[test]
fn extremality_and_horizon_are_classified_without_clamping() {
    let subextremal = KerrNewmanSpacetime::new(1.0, 0.8, 0.0, KerrSchildChart::Ingoing)
        .expect("parameters are valid");
    let extremal = KerrNewmanSpacetime::new(1.0, 1.0, 0.0, KerrSchildChart::Ingoing)
        .expect("parameters are valid");
    let superextremal = KerrNewmanSpacetime::new(1.0, 1.0, 0.5, KerrSchildChart::Ingoing)
        .expect("parameters are valid");

    assert_eq!(subextremal.extremality(), Extremality::Subextremal);
    assert_eq!(extremal.extremality(), Extremality::Extremal);
    assert_eq!(superextremal.extremality(), Extremality::Superextremal);
    assert_abs_diff_eq!(
        subextremal.outer_horizon_radius().expect("horizon exists"),
        1.6,
        epsilon = 2.0e-15
    );
    assert_eq!(extremal.outer_horizon_radius(), Some(1.0));
    assert_eq!(superextremal.outer_horizon_radius(), None);
}

#[test]
fn finite_extreme_scales_preserve_extremality_and_horizon_classification() {
    for mass_m in [f64::MAX / 2.0, f64::MIN_POSITIVE] {
        let extremal = KerrNewmanSpacetime::new(mass_m, mass_m, 0.0, KerrSchildChart::Ingoing)
            .expect("parameters are finite");

        assert_eq!(extremal.extremality(), Extremality::Extremal);
        assert_eq!(extremal.outer_horizon_radius(), Some(mass_m));
    }
}

#[test]
fn near_extremal_state_uses_the_exact_binary64_parameter_values() {
    let spin_m = 1.0_f64.next_down();
    let charge_m = 2.0_f64.powi(-26);
    let superextremal = KerrNewmanSpacetime::new(1.0, spin_m, charge_m, KerrSchildChart::Ingoing)
        .expect("finite parameters are valid");
    let subextremal = KerrNewmanSpacetime::new(1.0, spin_m, 0.0, KerrSchildChart::Ingoing)
        .expect("finite parameters are valid");

    assert_eq!(superextremal.extremality(), Extremality::Superextremal);
    assert_eq!(superextremal.outer_horizon_radius(), None);
    assert_eq!(subextremal.extremality(), Extremality::Subextremal);
    assert_eq!(
        subextremal.outer_horizon_radius(),
        Some(2.0_f64.powi(-26) + 1.0)
    );
}

#[test]
fn validated_spacetime_rejects_an_unrepresentable_outer_horizon() {
    let report = KerrNewmanSpacetime::new(f64::MAX, 0.0, 0.0, KerrSchildChart::Ingoing)
        .expect_err("validated geometry must have a finite outer horizon");

    assert!(report.issues().iter().any(|issue| {
        issue.code() == ValidationIssueCode::NonFinite
            && issue.field_path() == "spacetime.outer_horizon_radius_m"
    }));
}

proptest! {
    #[test]
    fn oblate_radius_and_rank_one_metric_inverse_satisfy_the_algebra_contract(
        mass in 0.5_f64..4.0,
        spin_fraction in -0.9_f64..0.9,
        charge_fraction in -0.2_f64..0.2,
        radius_fraction in 3.0_f64..30.0,
        polar in 0.1_f64..(std::f64::consts::PI - 0.1),
        azimuth in -std::f64::consts::PI..std::f64::consts::PI,
        outgoing in any::<bool>(),
    ) {
        let spin = mass * spin_fraction;
        let charge = mass * charge_fraction;
        let radius = mass * radius_fraction;
        let coordinates = if outgoing {
            KerrSchildChart::Outgoing
        } else {
            KerrSchildChart::Ingoing
        };
        let spacetime = KerrNewmanSpacetime::new(mass, spin, charge, coordinates)
            .expect("strategy remains strictly subextremal");
        let [x, y, z] = spacetime.oblate_to_cartesian(radius, polar, azimuth);
        let event = SpacetimeEvent::from_txyz([0.0, x, y, z]).expect("generated event is finite");
        let radius = spacetime.radius(event).expect("point is outside the ring");
        let [_, x, y, z] = event.to_txyz();
        let transverse_squared = y.mul_add(y, x * x);
        let radial_denominator = spin.mul_add(spin, radius * radius);
        let identity = transverse_squared / radial_denominator + z * z / radius.powi(2);

        prop_assert!(abs_diff_eq!(
            radius / (mass * radius_fraction),
            1.0,
            epsilon = 4.0e-14
        ));
        prop_assert!(abs_diff_eq!(identity, 1.0, epsilon = 2.0e-13));
        prop_assert!(
            spacetime
                .metric_inverse_residual(event)
                .expect("metric is finite")
                < 2.0e-12
        );
    }
}

#[test]
fn kerr_schild_branch_reverses_the_radial_direction_and_oblate_twist() {
    let ingoing = KerrNewmanSpacetime::new(1.0, 0.8, 0.0, KerrSchildChart::Ingoing)
        .expect("parameters are valid");
    let outgoing = KerrNewmanSpacetime::new(1.0, 0.8, 0.0, KerrSchildChart::Outgoing)
        .expect("parameters are valid");
    let state = gravlume_domain::GeodesicState::new([0.0, 10.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0])
        .expect("state is finite");
    let ingoing_rhs = ingoing.hamiltonian_rhs(state).expect("geometry is regular");
    let outgoing_rhs = outgoing
        .hamiltonian_rhs(state)
        .expect("geometry is regular");

    assert_abs_diff_eq!(
        ingoing_rhs[1],
        -outgoing_rhs[1],
        epsilon = 4.0 * f64::EPSILON
    );

    let polar = std::f64::consts::FRAC_PI_3;
    let ingoing_position = ingoing.oblate_to_cartesian(30.0, polar, 0.0);
    let outgoing_position = outgoing.oblate_to_cartesian(30.0, polar, 0.0);
    assert_eq!(
        ingoing_position[0].to_bits(),
        outgoing_position[0].to_bits()
    );
    assert_abs_diff_eq!(
        ingoing_position[1],
        -outgoing_position[1],
        epsilon = 4.0 * f64::EPSILON
    );
    assert_eq!(
        ingoing_position[2].to_bits(),
        outgoing_position[2].to_bits()
    );
}

#[test]
fn metric_inverse_residual_is_term_normalized_under_strong_cancellation() {
    let spacetime = KerrNewmanSpacetime::new(1.0, 0.0, 0.0, KerrSchildChart::Ingoing)
        .expect("parameters are valid");
    let event = SpacetimeEvent::from_txyz([0.0, 1.0e-4, 2.0e-4, 3.0e-4]).expect("event is finite");
    let residual = spacetime
        .metric_inverse_residual(event)
        .expect("metric is finite");

    assert_abs_diff_eq!(residual, 0.0, epsilon = 2.0e-12);
}

#[test]
fn metric_inverse_residual_preserves_representable_values_when_term_norm_overflows() {
    let spacetime = KerrNewmanSpacetime::new(5.0e113, 0.0, 0.0, KerrSchildChart::Ingoing)
        .expect("parameters are valid");
    let event = SpacetimeEvent::from_txyz([0.0, 1.0e-40, 0.0, 0.0]).expect("event is finite");
    let residual = spacetime
        .metric_inverse_residual(event)
        .expect("scaled metric contraction is finite");

    assert!(residual.is_finite() && residual > 0.0 && residual < f64::MIN_POSITIVE);
}

#[test]
fn schwarzschild_limit_is_spherical_and_stationary() {
    let spacetime = KerrNewmanSpacetime::new(2.0, 0.0, 0.0, KerrSchildChart::Ingoing)
        .expect("parameters are valid");
    let event = SpacetimeEvent::from_txyz([7.0, 3.0, 4.0, 12.0]).expect("event is finite");

    assert_abs_diff_eq!(
        spacetime.radius(event).expect("radius exists"),
        13.0,
        epsilon = f64::EPSILON
    );
    assert_abs_diff_eq!(
        spacetime
            .metric_component_tt(event)
            .expect("metric is finite"),
        -9.0 / 13.0,
        epsilon = 2.0e-15
    );
}

proptest! {
    #[test]
    fn representable_far_field_coordinates_do_not_overflow_internal_squares(exponent in 80_i32..=300) {
        let spacetime = KerrNewmanSpacetime::new(1.0, 0.0, 0.0, KerrSchildChart::Ingoing)
            .expect("parameters are valid");
        let coordinate = 10.0_f64.powi(exponent);
        let event =
            SpacetimeEvent::from_txyz([0.0, coordinate, 0.0, 0.0]).expect("event is finite");

        prop_assert_eq!(
            spacetime.radius(event),
            Ok(coordinate),
            "radius failed at {:e}", coordinate
        );
        prop_assert_eq!(
            spacetime.metric_component_tt(event),
            Ok(-1.0),
            "metric failed at {:e}", coordinate
        );
        prop_assert_eq!(
            spacetime.singularity_guard_residual(event, 1.0),
            Ok(f64::MAX),
            "guard side failed at {:e}", coordinate
        );
    }
}

proptest! {
    #[test]
    fn representable_axis_radius_does_not_require_a_representable_square(exponent in -300_i32..=-160) {
        let spacetime = KerrNewmanSpacetime::new(1.0, 1.0, 0.0, KerrSchildChart::Ingoing)
            .expect("parameters are valid");
        let coordinate = 10.0_f64.powi(exponent);
        let event = SpacetimeEvent::from_txyz([0.0, 0.0, 0.0, coordinate])
            .expect("event is finite");

        prop_assert_eq!(spacetime.radius(event), Ok(coordinate));
        prop_assert_eq!(spacetime.metric_component_tt(event), Ok(-1.0));
    }
}

#[test]
fn reissner_nordstrom_and_minkowski_limits_match_closed_form_g_tt() {
    let event = SpacetimeEvent::from_txyz([0.0, 10.0, 0.0, 0.0]).expect("event is finite");
    let reissner_nordstrom = KerrNewmanSpacetime::new(1.0, 0.0, 0.6, KerrSchildChart::Ingoing)
        .expect("parameters are valid");
    let expected_g_tt = -1.0 + 2.0 / 10.0 - 0.6_f64.powi(2) / 10.0_f64.powi(2);
    assert_abs_diff_eq!(
        reissner_nordstrom
            .metric_component_tt(event)
            .expect("metric is finite"),
        expected_g_tt,
        epsilon = 4.0 * f64::EPSILON
    );

    let minkowski_limit = KerrNewmanSpacetime::new(1.0e-12, 0.0, 0.0, KerrSchildChart::Ingoing)
        .expect("positive mass is valid");
    assert_abs_diff_eq!(
        minkowski_limit
            .metric_component_tt(event)
            .expect("metric is finite"),
        -1.0,
        epsilon = 2.1e-13
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
    let spacetime = KerrNewmanSpacetime::new(1.0, 0.8, 0.0, KerrSchildChart::Ingoing)
        .expect("parameters are valid");
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

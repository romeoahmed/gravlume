use std::{f64::consts::FRAC_PI_4, num::NonZeroU32};

use gravlume_domain::{
    Angle, Observation, ParameterState, PhysicalScene, PhysicalSceneDraft, StationaryObserverDraft,
    ValidationIssueCode, ViewportProjection,
};

const OBSERVER_POSITION: [f64; 4] = [0.0, 25.980_762_113_533_16, 0.692_820_323_027_550_9, 15.0];

fn default_scene() -> PhysicalScene {
    scene_with_frequency(1.0)
}

fn scene_with_frequency(measured_frequency: f64) -> PhysicalScene {
    PhysicalScene::commit(PhysicalSceneDraft::new(
        1.0,
        0.8,
        0.0,
        StationaryObserverDraft::new(
            OBSERVER_POSITION,
            [0.0; 4],
            [0.0, 0.0, 1.0],
            measured_frequency,
        ),
    ))
    .expect("the versioned default scene is valid")
}

fn default_projection() -> ViewportProjection {
    ViewportProjection::perspective(
        NonZeroU32::new(1280).expect("width is nonzero"),
        NonZeroU32::new(720).expect("height is nonzero"),
        Angle::from_radians(FRAC_PI_4).expect("angle is finite"),
    )
    .expect("the versioned projection is valid")
}

#[test]
fn invalid_scene_reports_stable_codes_and_field_paths() {
    let result = PhysicalScene::commit(PhysicalSceneDraft::new(
        0.0,
        f64::NAN,
        0.0,
        StationaryObserverDraft::new([0.0; 4], [0.0; 4], [0.0, 0.0, 1.0], 0.0),
    ));

    let report = result.expect_err("invalid seam input is rejected transactionally");
    assert!(report.issues().iter().any(|issue| {
        issue.code() == ValidationIssueCode::NonPositive
            && issue.field_path() == "physical_scene.spacetime.mass_m"
    }));
    assert!(report.issues().iter().any(|issue| {
        issue.code() == ValidationIssueCode::NonFinite
            && issue.field_path() == "physical_scene.spacetime.spin_m"
    }));
    assert!(report.issues().iter().any(|issue| {
        issue.code() == ValidationIssueCode::NonPositive
            && issue.field_path() == "physical_scene.observer.measured_frequency"
    }));
}

#[test]
fn default_kerr_observer_matches_the_versioned_contract() {
    let scene = default_scene();

    assert_eq!(scene.parameter_state(), ParameterState::Subextremal);
    assert!(
        (scene
            .spacetime()
            .outer_horizon_radius()
            .expect("horizon exists")
            - 1.6)
            .abs()
            < 2.0e-15
    );
    assert!((scene.observer_metric_g_tt() + 0.933_345_183_078_563_8).abs() < 2.0e-15);
    assert!(scene.observer_frame().gram_residual() < 2.0e-12);
    assert!(scene.observer_frame().orientation_determinant() > 0.0);
}

#[test]
fn observer_gram_residual_is_term_normalized_near_the_stationary_limit() {
    let spin = 0.8_f64;
    let radius = 2.0_f64 + 1.0e-8;
    let x = radius.mul_add(radius, spin * spin).sqrt();
    let scene = PhysicalScene::commit(PhysicalSceneDraft::new(
        1.0,
        spin,
        0.0,
        StationaryObserverDraft::new([0.0, x, 0.0, 0.0], [0.0; 4], [0.0, 0.0, 1.0], 1.0),
    ))
    .expect("a finite stationary observer remains representable near g_tt = 0");

    assert!(scene.observer_frame().gram_residual() < 2.0e-12);
}

#[test]
fn viewport_samples_produce_future_directed_null_rays() {
    let scene = default_scene();
    let projection = default_projection();
    let observation = Observation::new(scene, projection);

    for (x, y, offset_x, offset_y) in [
        (640, 360, 0.5, 0.5),
        (0, 0, 0.0, 0.0),
        (1279, 719, 1.0, 1.0),
        (317, 509, 0.25, 0.75),
    ] {
        let sample = observation
            .projection()
            .sample(x, y, offset_x, offset_y)
            .expect("sample is in bounds");
        let ray = observation
            .initial_ray(sample)
            .expect("sample remains valid for the observation projection");

        assert!(ray.normalized_null_residual() < 2.0e-12);
        assert!((ray.observer_frequency() - 1.0).abs() < 2.0e-12);
        assert!(ray.observer_frequency() > 0.0);
    }
}

#[test]
fn viewport_rejects_pixels_and_subpixels_outside_the_seam() {
    let projection = ViewportProjection::perspective(
        NonZeroU32::new(3).expect("width is nonzero"),
        NonZeroU32::new(2).expect("height is nonzero"),
        Angle::from_radians(FRAC_PI_4).expect("angle is finite"),
    )
    .expect("projection is valid");

    assert!(projection.sample(3, 0, 0.5, 0.5).is_err());
    assert!(projection.sample(0, 2, 0.5, 0.5).is_err());
    assert!(projection.sample(0, 0, -f64::EPSILON, 0.5).is_err());
    assert!(projection.sample(0, 0, 0.5, 1.0 + f64::EPSILON).is_err());
}

#[test]
fn initial_ray_resolves_a_sample_against_the_observation_projection() {
    let observation_projection = default_projection();
    let foreign_projection = ViewportProjection::perspective(
        NonZeroU32::new(1280).expect("width is nonzero"),
        NonZeroU32::new(720).expect("height is nonzero"),
        Angle::from_radians(2.0 * FRAC_PI_4).expect("angle is finite"),
    )
    .expect("projection is valid");
    let observation = Observation::new(default_scene(), observation_projection);
    let foreign_sample = foreign_projection
        .sample(100, 200, 0.25, 0.75)
        .expect("sample is valid for the foreign projection");
    let local_sample = observation_projection
        .sample(100, 200, 0.25, 0.75)
        .expect("sample is valid for the observation projection");

    assert_eq!(
        observation
            .initial_ray(foreign_sample)
            .expect("coordinates are valid for the observation projection")
            .state(),
        observation
            .initial_ray(local_sample)
            .expect("coordinates are valid for the observation projection")
            .state()
    );
}

#[test]
fn normalized_initial_null_residual_is_stable_under_frequency_scaling() {
    let projection = default_projection();
    let observation = Observation::new(scene_with_frequency(1.0e200), projection);
    let sample = projection
        .sample(317, 509, 0.25, 0.75)
        .expect("sample is valid");

    assert!(
        observation
            .initial_ray(sample)
            .expect("sample remains valid for the observation projection")
            .normalized_null_residual()
            < 2.0e-12
    );
}

#[test]
fn initial_ray_rejects_non_finite_derived_momentum() {
    let projection = default_projection();
    let observation = Observation::new(scene_with_frequency(f64::MAX), projection);
    let sample = projection
        .sample(317, 509, 0.25, 0.75)
        .expect("sample is valid");

    let report = observation
        .initial_ray(sample)
        .expect_err("overflowing derived momentum is rejected at the ray seam");

    assert!(report.issues().iter().any(|issue| {
        issue.code() == ValidationIssueCode::NonFinite
            && issue.field_path() == "observation.initial_ray"
    }));
}

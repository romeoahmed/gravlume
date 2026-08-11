use std::{f64::consts::FRAC_PI_4, num::NonZeroU32};

use gravlume_domain::{
    Angle, Observation, ParameterState, PhysicalScene, PhysicalSceneDraft, StationaryObserverDraft,
    ValidationIssueCode, ViewportProjection,
};

const OBSERVER_POSITION: [f64; 4] = [0.0, 25.980_762_113_533_16, 0.692_820_323_027_550_9, 15.0];

fn default_scene() -> PhysicalScene {
    PhysicalScene::commit(PhysicalSceneDraft::new(
        1.0,
        0.8,
        0.0,
        StationaryObserverDraft::new(OBSERVER_POSITION, [0.0; 4], [0.0, 0.0, 1.0], 1.0),
    ))
    .expect("the versioned default scene is valid")
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
    assert!((scene.outer_horizon_radius().expect("horizon exists") - 1.6).abs() < 2.0e-15);
    assert!((scene.observer_metric_g_tt() + 0.933_345_183_078_563_8).abs() < 2.0e-15);
    assert!(scene.observer_frame().gram_residual() < 2.0e-12);
    assert!(scene.observer_frame().orientation_determinant() > 0.0);
}

#[test]
fn viewport_samples_produce_future_directed_null_rays() {
    let scene = default_scene();
    let projection = ViewportProjection::perspective(
        NonZeroU32::new(1280).expect("width is nonzero"),
        NonZeroU32::new(720).expect("height is nonzero"),
        Angle::from_radians(FRAC_PI_4).expect("angle is finite"),
    )
    .expect("the versioned projection is valid");
    let observation = Observation::new(scene, projection).expect("observation invariants hold");

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
        let ray = observation.initial_ray(sample);

        assert!(ray.normalized_null_residual() < 2.0e-12);
        assert!((ray.observer_frequency() - 1.0).abs() < 2.0e-12);
        assert!(ray.is_future_directed());
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

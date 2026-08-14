use std::{f64::consts::FRAC_PI_4, num::NonZeroU32};

use gravlume_domain::{
    Angle, KerrSchildChart, Observation, PerspectiveView, PhysicalScene, PhysicalSceneInput,
    StationaryObserverInput, ValidationIssueCode,
};
use proptest::prelude::*;

const OBSERVER_POSITION: [f64; 4] = [0.0, 25.980_762_113_533_16, 0.692_820_323_027_550_9, 15.0];

fn default_scene() -> PhysicalScene {
    scene_with_frequency(1.0)
}

fn scene_with_frequency(measured_frequency: f64) -> PhysicalScene {
    PhysicalScene::new(PhysicalSceneInput::new(
        1.0,
        0.8,
        0.0,
        KerrSchildChart::Ingoing,
        StationaryObserverInput::new(
            OBSERVER_POSITION,
            [0.0; 4],
            [0.0, 0.0, 1.0],
            measured_frequency,
        ),
    ))
    .expect("the versioned default scene is valid")
}

fn default_view() -> PerspectiveView {
    PerspectiveView::new(
        NonZeroU32::new(1280).expect("width is nonzero"),
        NonZeroU32::new(720).expect("height is nonzero"),
        Angle::from_radians(FRAC_PI_4).expect("angle is finite"),
    )
    .expect("the versioned view is valid")
}

#[test]
fn invalid_scene_reports_stable_codes_and_field_paths() {
    let result = PhysicalScene::new(PhysicalSceneInput::new(
        0.0,
        f64::NAN,
        0.0,
        KerrSchildChart::Ingoing,
        StationaryObserverInput::new([0.0; 4], [0.0; 4], [0.0, 0.0, 1.0], 0.0),
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
fn observer_gram_residual_is_term_normalized_near_the_stationary_limit() {
    let spin = 0.8_f64;
    let radius = 2.0_f64 + 1.0e-8;
    let x = radius.mul_add(radius, spin * spin).sqrt();
    let scene = PhysicalScene::new(PhysicalSceneInput::new(
        1.0,
        spin,
        0.0,
        KerrSchildChart::Ingoing,
        StationaryObserverInput::new([0.0, x, 0.0, 0.0], [0.0; 4], [0.0, 0.0, 1.0], 1.0),
    ))
    .expect("a finite stationary observer remains representable near g_tt = 0");

    assert!(scene.observer_frame().gram_residual() < 2.0e-12);
}

fn image_sample() -> impl Strategy<Value = (u32, u32, u32, u32, f64, f64)> {
    (1_u32..=2_048, 1_u32..=2_048).prop_flat_map(|(width, height)| {
        (
            Just(width),
            Just(height),
            0..width,
            0..height,
            0.0_f64..=1.0,
            0.0_f64..=1.0,
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn image_samples_produce_future_directed_null_rays(
        (width, height, x, y, offset_x, offset_y) in image_sample(),
    ) {
        let view = PerspectiveView::new(
            NonZeroU32::new(width).expect("generated width is nonzero"),
            NonZeroU32::new(height).expect("generated height is nonzero"),
            Angle::from_radians(FRAC_PI_4).expect("angle is finite"),
        )
        .expect("generated view is valid");
        let observation = Observation::new(default_scene(), view);
        let sample = observation
            .view()
            .sample(x, y, offset_x, offset_y)
            .expect("strategy generates an in-bounds sample");
        let ray = observation
            .initial_ray(sample)
            .expect("sample remains valid for the observation view");

        prop_assert!(ray.normalized_null_residual() < 2.0e-12);
        prop_assert!((ray.observer_frequency() - 1.0).abs() < 2.0e-12);
        prop_assert!(ray.observer_frequency() > 0.0);
    }
}

#[test]
fn image_view_rejects_pixels_and_subpixels_outside_the_seam() {
    let view = PerspectiveView::new(
        NonZeroU32::new(3).expect("width is nonzero"),
        NonZeroU32::new(2).expect("height is nonzero"),
        Angle::from_radians(FRAC_PI_4).expect("angle is finite"),
    )
    .expect("view is valid");

    assert!(view.sample(3, 0, 0.5, 0.5).is_err());
    assert!(view.sample(0, 2, 0.5, 0.5).is_err());
    assert!(view.sample(0, 0, -f64::EPSILON, 0.5).is_err());
    assert!(view.sample(0, 0, 0.5, 1.0 + f64::EPSILON).is_err());
}

proptest! {
    #[test]
    fn normalized_initial_null_residual_is_stable_under_frequency_scaling(exponent in -150_i32..=150) {
        let view = default_view();
        let observation = Observation::new(scene_with_frequency(10.0_f64.powi(exponent)), view);
        let sample = view
            .sample(317, 509, 0.25, 0.75)
            .expect("sample is valid");

        prop_assert!(
            observation
                .initial_ray(sample)
                .expect("finite frequency produces a finite ray")
                .normalized_null_residual()
                < 2.0e-12
        );
    }
}

#[test]
fn initial_ray_rejects_non_finite_derived_momentum() {
    let view = default_view();
    let observation = Observation::new(scene_with_frequency(f64::MAX), view);
    let sample = view.sample(317, 509, 0.25, 0.75).expect("sample is valid");

    let report = observation
        .initial_ray(sample)
        .expect_err("overflowing derived momentum is rejected at the ray seam");

    assert!(report.issues().iter().any(|issue| {
        issue.code() == ValidationIssueCode::NonFinite
            && issue.field_path() == "observation.initial_ray"
    }));
}

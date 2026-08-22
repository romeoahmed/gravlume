use gravlume_domain::{ImageSample, Observation, ValidationReport};

use crate::{
    ObservationTrace, ObservationTraceError, ObservationTracer, ReferenceOutcome, ReferencePolicy,
    ReferenceTerminal, SourceAnchor, SurfaceObservable, TraceBranchKey, TraceInputId,
    surface::wrapped_angle_difference,
};

const DIFFERENCE_OFFSET_PIXELS: f64 = 0.25;
const DIFFERENCE_SPAN_PIXELS: f64 = 2.0 * DIFFERENCE_OFFSET_PIXELS;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SurfaceFootprintEstimate {
    Resolved(SurfaceFootprint),
    Discontinuity,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceFootprint {
    source_anchor: SourceAnchor,
    branch_key: TraceBranchKey,
    jacobian_source_m_per_pixel: [[f64; 2]; 2],
    singular_values_m_per_pixel: [f64; 2],
    parity: SurfaceParity,
}

impl SurfaceFootprint {
    #[must_use]
    pub const fn source_anchor(self) -> SourceAnchor {
        self.source_anchor
    }

    #[must_use]
    pub const fn branch_key(self) -> TraceBranchKey {
        self.branch_key
    }

    /// Returns rows `(radial, local azimuthal arc)` and columns `(screen x, screen y)`.
    #[must_use]
    pub const fn jacobian_source_m_per_pixel(self) -> [[f64; 2]; 2] {
        self.jacobian_source_m_per_pixel
    }

    /// Returns major then minor singular value in source metres per image pixel.
    #[must_use]
    pub const fn singular_values_m_per_pixel(self) -> [f64; 2] {
        self.singular_values_m_per_pixel
    }

    #[must_use]
    pub const fn parity(self) -> SurfaceParity {
        self.parity
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceParity {
    Positive,
    Negative,
    Degenerate,
}

impl ObservationTracer {
    /// Estimates the local equatorial-source Jacobian with real quarter-pixel traces.
    ///
    /// The estimate is returned only when the center and all four neighbors have an unambiguous
    /// equatorial termination and exactly the same discrete branch key. A semantic mismatch is a
    /// successful `Discontinuity` result, never a large cross-branch derivative.
    ///
    /// # Errors
    ///
    /// Rejects samples whose subpixel position cannot support a centered neighborhood, a center
    /// ray that does not resolve a surface, or trace/setup failures.
    pub fn surface_footprint_v1(
        &self,
        observation: &Observation,
        sample: ImageSample,
        policy: ReferencePolicy,
    ) -> Result<SurfaceFootprintEstimate, SurfaceFootprintError> {
        let [subpixel_x, subpixel_y] = sample.subpixel();
        let supported_subpixel = DIFFERENCE_OFFSET_PIXELS..=1.0 - DIFFERENCE_OFFSET_PIXELS;
        if !supported_subpixel.contains(&subpixel_x) || !supported_subpixel.contains(&subpixel_y) {
            return Err(SurfaceFootprintError::NeighborhoodOutsidePixel);
        }
        let [
            center_outcome,
            left_outcome,
            right_outcome,
            up_outcome,
            down_outcome,
        ] = trace_surface_neighborhood(*self, observation, sample, policy)?;
        let center = resolved_surface(&center_outcome)
            .ok_or(SurfaceFootprintError::CenterSurfaceUnavailable)?;
        let [Some(left), Some(right), Some(up), Some(down)] = [
            resolved_surface(&left_outcome),
            resolved_surface(&right_outcome),
            resolved_surface(&up_outcome),
            resolved_surface(&down_outcome),
        ] else {
            return Ok(SurfaceFootprintEstimate::Discontinuity);
        };
        let neighbors = [left, right, up, down];
        if neighbors
            .iter()
            .any(|neighbor| neighbor.branch_key != center.branch_key)
        {
            return Ok(SurfaceFootprintEstimate::Discontinuity);
        }
        let center_anchor = center.observable.source_anchor();
        let center_radius = center_anchor.radius_m();
        let local_arc = |observable: SurfaceObservable| {
            center_radius
                * wrapped_angle_difference(
                    observable.source_anchor().azimuth_rad(),
                    center_anchor.azimuth_rad(),
                )
        };
        let jacobian = [
            [
                (right.observable.source_anchor().radius_m()
                    - left.observable.source_anchor().radius_m())
                    / DIFFERENCE_SPAN_PIXELS,
                (down.observable.source_anchor().radius_m()
                    - up.observable.source_anchor().radius_m())
                    / DIFFERENCE_SPAN_PIXELS,
            ],
            [
                (local_arc(right.observable) - local_arc(left.observable)) / DIFFERENCE_SPAN_PIXELS,
                (local_arc(down.observable) - local_arc(up.observable)) / DIFFERENCE_SPAN_PIXELS,
            ],
        ];
        let metrics =
            footprint_metrics(jacobian).ok_or(SurfaceFootprintError::NonFiniteJacobian)?;
        let determinant = metrics.determinant;
        let determinant_floor = 64.0 * f64::EPSILON * metrics.squared_norm;
        let parity = if determinant > determinant_floor {
            SurfaceParity::Positive
        } else if determinant < -determinant_floor {
            SurfaceParity::Negative
        } else {
            SurfaceParity::Degenerate
        };
        Ok(SurfaceFootprintEstimate::Resolved(SurfaceFootprint {
            source_anchor: center_anchor,
            branch_key: center.branch_key,
            jacobian_source_m_per_pixel: jacobian,
            singular_values_m_per_pixel: metrics.singular_values,
            parity,
        }))
    }
}

fn trace_surface_neighborhood(
    tracer: ObservationTracer,
    observation: &Observation,
    center: ImageSample,
    policy: ReferencePolicy,
) -> Result<[ReferenceOutcome; 5], SurfaceFootprintError> {
    let [pixel_x, pixel_y] = center.pixel();
    let [subpixel_x, subpixel_y] = center.subpixel();
    let neighbor = |offset_x: f64, offset_y: f64| {
        observation
            .view()
            .sample(
                pixel_x,
                pixel_y,
                subpixel_x + offset_x,
                subpixel_y + offset_y,
            )
            .map_err(SurfaceFootprintError::InvalidSample)
    };
    let samples = [
        center,
        neighbor(-DIFFERENCE_OFFSET_PIXELS, 0.0)?,
        neighbor(DIFFERENCE_OFFSET_PIXELS, 0.0)?,
        neighbor(0.0, -DIFFERENCE_OFFSET_PIXELS)?,
        neighbor(0.0, DIFFERENCE_OFFSET_PIXELS)?,
    ];
    let trace = |sample, label| {
        let request = ObservationTrace::new(
            TraceInputId::new(format!("surface-footprint-v1-{label}")),
            observation,
            sample,
            policy,
        )
        .map_err(SurfaceFootprintError::InvalidSample)?;
        tracer.trace(request).map_err(SurfaceFootprintError::Trace)
    };
    let [center, left, right, up, down] = samples;
    Ok([
        trace(center, "center")?,
        trace(left, "left")?,
        trace(right, "right")?,
        trace(up, "up")?,
        trace(down, "down")?,
    ])
}

#[derive(Clone, Copy)]
struct FootprintMetrics {
    determinant: f64,
    squared_norm: f64,
    singular_values: [f64; 2],
}

fn footprint_metrics(jacobian: [[f64; 2]; 2]) -> Option<FootprintMetrics> {
    if !jacobian.into_iter().flatten().all(f64::is_finite) {
        return None;
    }
    let determinant = jacobian[0][0].mul_add(jacobian[1][1], -jacobian[0][1] * jacobian[1][0]);
    let squared_norm = jacobian
        .into_iter()
        .flatten()
        .map(|value| value * value)
        .sum::<f64>();
    let discriminant = squared_norm
        .mul_add(squared_norm, -4.0 * determinant * determinant)
        .max(0.0)
        .sqrt();
    let major = squared_norm.midpoint(discriminant).sqrt();
    // `sqrt((norm² - discriminant) / 2)` loses the minor axis near a critical curve.
    // For a 2x2 matrix, sigma_major * sigma_minor = |determinant|.
    let minor = if major == 0.0 {
        0.0
    } else {
        determinant.abs() / major
    };
    if !major.is_finite() || !minor.is_finite() {
        return None;
    }
    Some(FootprintMetrics {
        determinant,
        squared_norm,
        singular_values: [major, minor],
    })
}

#[derive(Clone, Copy)]
struct ResolvedSurface {
    observable: SurfaceObservable,
    branch_key: TraceBranchKey,
}

const fn resolved_surface(outcome: &ReferenceOutcome) -> Option<ResolvedSurface> {
    match outcome.terminal() {
        ReferenceTerminal::ObservedSurface { event, observable } if !event.is_ambiguous() => {
            Some(ResolvedSurface {
                observable: *observable,
                branch_key: outcome.branch_key(),
            })
        }
        _ => None,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SurfaceFootprintError {
    #[error("surface footprint v1 requires subpixel coordinates in [0.25, 0.75]")]
    NeighborhoodOutsidePixel,
    #[error("surface footprint sample validation failed: {0}")]
    InvalidSample(#[source] ValidationReport),
    #[error("surface footprint trace failed: {0}")]
    Trace(#[source] ObservationTraceError),
    #[error("the footprint center did not resolve an unambiguous equatorial surface")]
    CenterSurfaceUnavailable,
    #[error("surface footprint produced a non-finite source Jacobian")]
    NonFiniteJacobian,
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::footprint_metrics;

    proptest! {
        #[test]
        fn determinant_quotient_preserves_every_binary_singular_axis(
            minor_exponent in prop_oneof![
                Just(-1074),
                Just(-1022),
                Just(-1),
                Just(0),
                -1074_i32..=0,
            ],
        ) {
            let expected_minor = 2.0_f64.powi(minor_exponent);
            let jacobian = [[1.0_f64, 0.0], [0.0, expected_minor]];
            let metrics = footprint_metrics(jacobian).expect("finite matrix has finite metrics");
            let [major, minor] = metrics.singular_values;

            prop_assert_eq!(major.to_bits(), 1.0_f64.to_bits());
            prop_assert_eq!(minor.to_bits(), expected_minor.to_bits());
        }
    }
}

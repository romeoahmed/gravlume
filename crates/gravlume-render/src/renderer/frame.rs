use num_traits::ToPrimitive as _;

use crate::{
    display::{DisplayPipeline, DisplayTarget, PublishedScene, ScenePresentation},
    error::ResizeError,
    extent::RenderExtent,
    ray_tracer::{
        RayTracer, TileRegion, TraceImage, direction_reconstruction_scratch_bytes,
        shadow_coverage_scratch_bytes, tile_grid,
    },
};

const MAXIMUM_NATIVE_TRACE_PIXELS: u64 = 3_840 * 2_160;
const HDR_BYTES_PER_PIXEL: u64 = 8;
const UI_BYTES_PER_PIXEL: u64 = 4;
const MAXIMUM_CORE_RESOURCE_BYTES: u64 = 256 * 1024 * 1024;
const INITIAL_TILES_PER_BATCH: u32 = 512;
const TARGET_BATCH_MS: f64 = 32.0;
const MAXIMUM_BATCH_MS: f64 = 50.0;
const MAXIMUM_BATCH_SCALE: f64 = 1.5;
const MINIMUM_BATCH_SCALE: f64 = 0.5;

pub struct FrameResources {
    candidate: Option<TraceCandidate>,
    display: DisplayTarget,
    presentation: ScenePresentation,
    extent: RenderExtent,
    completed_batches: u32,
    total_compute_ms: f64,
    maximum_batch_ms: f64,
}

struct TraceCandidate {
    trace: TraceImage,
    completed_presentation: ScenePresentation,
    progress: TraceProgress,
}

pub struct CompletedCandidate {
    view: wgpu::TextureView,
    presentation: ScenePresentation,
}

#[derive(Debug)]
struct TraceProgress {
    grid: [u32; 2],
    total_tiles: u32,
    next_tile: u32,
    tiles_per_batch: u32,
    maximum_batch_tiles: u32,
    maximum_dispatch_dimension: u32,
    in_flight: Option<TileRegion>,
    completed_batches: u32,
    total_compute_ms: f64,
    maximum_batch_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceSubmission {
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoreResourcePlan {
    published: u64,
    installed: FrameResourceFootprint,
    replacement: FrameResourceFootprint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrameResourceFootprint {
    ui: u64,
    candidate: u64,
    trace_scratch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TraceCompletion {
    Stale,
    Pending,
    Ready,
}

#[derive(Clone, Copy, Debug)]
pub struct TraceProgressDiagnostics {
    completed_tiles: u32,
    total_tiles: u32,
    completed_batches: u32,
    total_compute_ms: f64,
    maximum_batch_ms: f64,
}

impl TraceProgressDiagnostics {
    pub fn completion(self) -> f64 {
        f64::from(self.completed_tiles) / f64::from(self.total_tiles)
    }

    pub const fn completed_batches(self) -> u32 {
        self.completed_batches
    }

    pub const fn total_compute_ms(self) -> f64 {
        self.total_compute_ms
    }

    pub const fn maximum_batch_ms(self) -> f64 {
        self.maximum_batch_ms
    }
}

impl FrameResourceFootprint {
    const EMPTY: Self = Self {
        ui: 0,
        candidate: 0,
        trace_scratch: 0,
    };

    const fn display_only(extent: RenderExtent) -> Self {
        Self {
            ui: extent_pixels(extent),
            candidate: 0,
            trace_scratch: 0,
        }
    }

    fn tracing(extent: RenderExtent) -> Self {
        Self {
            ui: extent_pixels(extent),
            candidate: extent_pixels(extent),
            trace_scratch: shadow_coverage_scratch_bytes(extent)
                .saturating_add(direction_reconstruction_scratch_bytes(extent)),
        }
    }

    const fn required_bytes(self) -> u64 {
        self.ui
            .saturating_mul(UI_BYTES_PER_PIXEL)
            .saturating_add(self.candidate.saturating_mul(HDR_BYTES_PER_PIXEL))
            .saturating_add(self.trace_scratch)
    }
}

impl CoreResourcePlan {
    pub fn without_installed_frame(published: RenderExtent, replacement: RenderExtent) -> Self {
        Self {
            published: extent_pixels(published),
            installed: FrameResourceFootprint::EMPTY,
            replacement: FrameResourceFootprint::tracing(replacement),
        }
    }

    fn rebuild(
        published: RenderExtent,
        installed: FrameResourceFootprint,
        replacement: RenderExtent,
    ) -> Self {
        Self {
            published: extent_pixels(published),
            installed,
            replacement: FrameResourceFootprint::tracing(replacement),
        }
    }

    const fn required_bytes(self) -> u64 {
        self.published
            .saturating_mul(HDR_BYTES_PER_PIXEL)
            .saturating_add(self.installed.required_bytes())
            .saturating_add(self.replacement.required_bytes())
    }
}

impl TraceSubmission {
    pub const fn new(generation: u64) -> Self {
        Self { generation }
    }
}

impl CompletedCandidate {
    pub fn into_parts(self) -> (wgpu::TextureView, ScenePresentation) {
        (self.view, self.presentation)
    }
}

impl FrameResources {
    pub fn new(
        device: &wgpu::Device,
        trace: &RayTracer,
        display: &DisplayPipeline,
        published: &PublishedScene,
        extent: RenderExtent,
    ) -> Self {
        let display_target = DisplayPipeline::create_target(device, extent);
        let presentation = display.bind_scene(device, published.view(), &display_target);
        let candidate = Self::create_candidate(device, trace, display, &display_target, extent);
        Self {
            candidate: Some(candidate),
            display: display_target,
            presentation,
            extent,
            completed_batches: 0,
            total_compute_ms: 0.0,
            maximum_batch_ms: 0.0,
        }
    }

    fn create_candidate(
        device: &wgpu::Device,
        trace: &RayTracer,
        display: &DisplayPipeline,
        display_target: &DisplayTarget,
        extent: RenderExtent,
    ) -> TraceCandidate {
        let trace = trace.create_target(device, extent);
        let completed_presentation = display.bind_scene(device, trace.view(), display_target);
        TraceCandidate {
            trace,
            completed_presentation,
            progress: TraceProgress::new(
                extent,
                device.limits().max_compute_workgroups_per_dimension,
            ),
        }
    }

    pub const fn ui_view(&self) -> &wgpu::TextureView {
        self.display.ui_view()
    }

    pub const fn presentation(&self) -> &ScenePresentation {
        &self.presentation
    }

    pub fn install_presentation(&mut self, presentation: ScenePresentation) {
        self.presentation = presentation;
    }

    pub const fn extent(&self) -> RenderExtent {
        self.extent
    }

    pub fn candidate_trace(&self) -> Option<&TraceImage> {
        self.candidate.as_ref().map(|candidate| &candidate.trace)
    }

    pub fn next_batch(&self) -> Option<TileRegion> {
        self.candidate.as_ref()?.progress.next_batch()
    }

    pub fn submitted(&mut self, batch: TileRegion) {
        if let Some(candidate) = self.candidate.as_mut() {
            candidate.progress.submitted(batch);
        }
    }

    pub fn complete_submission(
        &mut self,
        submission: TraceSubmission,
        current_generation: u64,
        compute_ms: f64,
    ) -> Option<CompletedCandidate> {
        let completion = self
            .candidate
            .as_mut()
            .map_or(TraceCompletion::Stale, |candidate| {
                candidate
                    .progress
                    .complete_submission(submission, current_generation, compute_ms)
            });
        if completion == TraceCompletion::Stale {
            return None;
        }
        self.completed_batches += 1;
        if compute_ms.is_finite() {
            self.total_compute_ms += compute_ms;
            self.maximum_batch_ms = self.maximum_batch_ms.max(compute_ms);
        }
        if completion != TraceCompletion::Ready {
            return None;
        }
        let candidate = self.candidate.take()?;
        Some(CompletedCandidate {
            view: candidate.trace.view().clone(),
            presentation: candidate.completed_presentation,
        })
    }

    pub fn diagnostics(&self) -> TraceProgressDiagnostics {
        let (completed_tiles, total_tiles) = self.candidate.as_ref().map_or_else(
            || {
                let [columns, rows] = tile_grid(self.extent);
                let tiles = columns * rows;
                (tiles, tiles)
            },
            |candidate| {
                let diagnostics = candidate.progress.diagnostics();
                (diagnostics.completed_tiles, diagnostics.total_tiles)
            },
        );
        TraceProgressDiagnostics {
            completed_tiles,
            total_tiles,
            completed_batches: self.completed_batches,
            total_compute_ms: self.total_compute_ms,
            maximum_batch_ms: self.maximum_batch_ms,
        }
    }

    fn resource_footprint(&self) -> FrameResourceFootprint {
        if self.candidate.is_some() {
            FrameResourceFootprint::tracing(self.extent)
        } else {
            FrameResourceFootprint::display_only(self.extent)
        }
    }

    pub fn rebuild_plan(
        &self,
        published: RenderExtent,
        replacement: RenderExtent,
    ) -> CoreResourcePlan {
        CoreResourcePlan::rebuild(published, self.resource_footprint(), replacement)
    }
}

impl TraceProgress {
    const fn new(extent: RenderExtent, maximum_dispatch_dimension: u32) -> Self {
        debug_assert!(maximum_dispatch_dimension > 0);
        let grid = tile_grid(extent);
        let total_tiles = grid[0] * grid[1];
        let maximum_batch_tiles = if grid[0] > maximum_dispatch_dimension {
            maximum_dispatch_dimension
        } else {
            grid[0].saturating_mul(if grid[1] < maximum_dispatch_dimension {
                grid[1]
            } else {
                maximum_dispatch_dimension
            })
        };
        Self {
            grid,
            total_tiles,
            next_tile: 0,
            tiles_per_batch: if INITIAL_TILES_PER_BATCH < maximum_batch_tiles {
                INITIAL_TILES_PER_BATCH
            } else {
                maximum_batch_tiles
            },
            maximum_batch_tiles,
            maximum_dispatch_dimension,
            in_flight: None,
            completed_batches: 0,
            total_compute_ms: 0.0,
            maximum_batch_ms: 0.0,
        }
    }

    fn next_batch(&self) -> Option<TileRegion> {
        if self.in_flight.is_some() || self.next_tile == self.total_tiles {
            return None;
        }
        let tile_x = self.next_tile % self.grid[0];
        let tile_y = self.next_tile / self.grid[0];
        let remaining_tiles = self.total_tiles - self.next_tile;
        let budget = self.tiles_per_batch.min(remaining_tiles);
        let remaining_columns = self.grid[0] - tile_x;
        let workgroups_x = budget
            .min(remaining_columns)
            .min(self.maximum_dispatch_dimension);
        let workgroups_y = if tile_x == 0 && workgroups_x == self.grid[0] && budget >= self.grid[0]
        {
            (budget / self.grid[0])
                .min(self.grid[1] - tile_y)
                .min(self.maximum_dispatch_dimension)
        } else {
            1
        };
        Some(TileRegion::new(
            [tile_x, tile_y],
            [workgroups_x, workgroups_y],
        ))
    }

    fn submitted(&mut self, batch: TileRegion) {
        let origin = batch.origin();
        debug_assert_eq!(origin[1] * self.grid[0] + origin[0], self.next_tile);
        debug_assert!(self.in_flight.is_none());
        self.next_tile += batch.len();
        self.in_flight = Some(batch);
    }

    fn completed(&mut self, compute_ms: f64) {
        let Some(batch) = self.in_flight.take() else {
            return;
        };
        self.completed_batches += 1;
        if compute_ms.is_finite() {
            self.total_compute_ms += compute_ms;
            self.maximum_batch_ms = self.maximum_batch_ms.max(compute_ms);
        }
        if self.next_tile == self.total_tiles || !compute_ms.is_finite() || compute_ms <= 0.0 {
            return;
        }

        let scale = if compute_ms > MAXIMUM_BATCH_MS {
            MINIMUM_BATCH_SCALE
        } else {
            (TARGET_BATCH_MS / compute_ms).clamp(MINIMUM_BATCH_SCALE, MAXIMUM_BATCH_SCALE)
        };
        let scaled = (f64::from(batch.len()) * scale).round().clamp(
            1.0,
            f64::from(self.total_tiles.min(self.maximum_batch_tiles)),
        );
        self.tiles_per_batch = scaled.to_u32().unwrap_or(self.total_tiles);
    }

    fn complete_submission(
        &mut self,
        submission: TraceSubmission,
        current_generation: u64,
        compute_ms: f64,
    ) -> TraceCompletion {
        if submission.generation != current_generation {
            return TraceCompletion::Stale;
        }
        self.completed(compute_ms);
        if self.is_complete() {
            TraceCompletion::Ready
        } else {
            TraceCompletion::Pending
        }
    }

    const fn is_complete(&self) -> bool {
        self.next_tile == self.total_tiles && self.in_flight.is_none()
    }

    const fn diagnostics(&self) -> TraceProgressDiagnostics {
        let completed_tiles = match self.in_flight {
            Some(batch) => self.next_tile - batch.len(),
            None => self.next_tile,
        };
        TraceProgressDiagnostics {
            completed_tiles,
            total_tiles: self.total_tiles,
            completed_batches: self.completed_batches,
            total_compute_ms: self.total_compute_ms,
            maximum_batch_ms: self.maximum_batch_ms,
        }
    }
}

pub const fn validate_extent(
    extent: RenderExtent,
    limits: &wgpu::Limits,
    resource_plan: CoreResourcePlan,
) -> Result<(), ResizeError> {
    if extent.width() > limits.max_texture_dimension_2d
        || extent.height() > limits.max_texture_dimension_2d
    {
        return Err(ResizeError::ExtentLimit {
            width: extent.width(),
            height: extent.height(),
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
        });
    }
    if extent_pixels(extent) > MAXIMUM_NATIVE_TRACE_PIXELS {
        return Err(ResizeError::NativePixelBudget {
            width: extent.width(),
            height: extent.height(),
            maximum_pixels: MAXIMUM_NATIVE_TRACE_PIXELS,
        });
    }
    let required_bytes = resource_plan.required_bytes();
    if required_bytes > MAXIMUM_CORE_RESOURCE_BYTES {
        return Err(ResizeError::FrameResourceBudget {
            width: extent.width(),
            height: extent.height(),
            required_bytes,
            maximum_bytes: MAXIMUM_CORE_RESOURCE_BYTES,
        });
    }
    Ok(())
}

const fn extent_pixels(extent: RenderExtent) -> u64 {
    extent.width() as u64 * extent.height() as u64
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn trace_progress_covers_each_tile_once_with_bounded_batches(
            width in 1_u32..=257,
            height in 1_u32..=257,
            maximum_dispatch_dimension in 1_u32..=16,
        ) {
            let extent = RenderExtent::new(width, height).expect("generated extent is nonzero");
            let mut progress = TraceProgress::new(extent, maximum_dispatch_dimension);
            let [tile_columns, tile_rows] = tile_grid(extent);
            let mut covered =
                vec![false; usize::try_from(tile_columns * tile_rows).expect("small grid")];

            while let Some(batch) = progress.next_batch() {
                let [origin_x, origin_y] = batch.origin();
                let [workgroups_x, workgroups_y] = batch.workgroups();
                prop_assert!(workgroups_x <= maximum_dispatch_dimension);
                prop_assert!(workgroups_y <= maximum_dispatch_dimension);
                for tile_y in origin_y..origin_y + workgroups_y {
                    for tile_x in origin_x..origin_x + workgroups_x {
                        prop_assert!(tile_x < tile_columns && tile_y < tile_rows);
                        let index = usize::try_from(tile_y * tile_columns + tile_x)
                            .expect("small grid index");
                        prop_assert!(!covered[index], "tile ({tile_x}, {tile_y}) was repeated");
                        covered[index] = true;
                    }
                }
                progress.submitted(batch);
                prop_assert!(progress.next_batch().is_none(), "one batch stays in flight");
                progress.completed(TARGET_BATCH_MS);
            }

            prop_assert!(covered.into_iter().all(|tile| tile));
            prop_assert_eq!(progress.next_tile, progress.total_tiles);
            prop_assert!(progress.in_flight.is_none());
            prop_assert!(progress.next_batch().is_none(), "complete traces are reused");
        }
    }

    #[test]
    fn publication_gate_requires_the_complete_current_generation() {
        let extent = RenderExtent::new(4_097, 9).expect("extent is nonzero");
        let mut progress = TraceProgress::new(
            extent,
            wgpu::Limits::default().max_compute_workgroups_per_dimension,
        );
        let submission = TraceSubmission::new(7);
        let mut covered_tiles = 0;

        while let Some(batch) = progress.next_batch() {
            progress.submitted(batch);
            covered_tiles += batch.len();
            let expected = if covered_tiles == progress.total_tiles {
                TraceCompletion::Ready
            } else {
                TraceCompletion::Pending
            };
            assert_eq!(batch.finishes(extent), expected == TraceCompletion::Ready);
            assert_eq!(
                progress.complete_submission(submission, 7, TARGET_BATCH_MS),
                expected
            );
        }

        let stale_extent = RenderExtent::new(1, 1).expect("extent is nonzero");
        let mut stale = TraceProgress::new(
            stale_extent,
            wgpu::Limits::default().max_compute_workgroups_per_dimension,
        );
        let batch = stale
            .next_batch()
            .expect("one tile requires one submission");
        stale.submitted(batch);
        assert_eq!(
            stale.complete_submission(submission, 8, TARGET_BATCH_MS),
            TraceCompletion::Stale
        );
    }

    #[test]
    fn core_resource_budget_accounts_for_transactional_4k_rebuild() {
        let limits = wgpu::Limits::default();
        let extent = RenderExtent::new(3_840, 2_160).expect("extent is nonzero");
        let initial = CoreResourcePlan::without_installed_frame(RenderExtent::ONE, extent);
        let active_rebuild =
            CoreResourcePlan::rebuild(extent, FrameResourceFootprint::tracing(extent), extent);
        let completed_rebuild =
            CoreResourcePlan::rebuild(extent, FrameResourceFootprint::display_only(extent), extent);
        let cold_rebuild = CoreResourcePlan::rebuild(
            RenderExtent::ONE,
            FrameResourceFootprint::tracing(extent),
            extent,
        );
        assert_eq!(extent_pixels(extent), MAXIMUM_NATIVE_TRACE_PIXELS);
        assert!(initial.required_bytes() <= MAXIMUM_CORE_RESOURCE_BYTES);
        assert!(validate_extent(extent, &limits, initial).is_ok());
        assert!(validate_extent(extent, &limits, cold_rebuild).is_ok());
        assert!(active_rebuild.required_bytes() > MAXIMUM_CORE_RESOURCE_BYTES);
        assert!(matches!(
            validate_extent(extent, &limits, active_rebuild),
            Err(ResizeError::FrameResourceBudget { .. })
        ));
        assert!(validate_extent(extent, &limits, completed_rebuild).is_ok());
    }

    #[test]
    fn resize_rejects_pixels_beyond_the_native_4k_policy() {
        let limits = wgpu::Limits::default();
        let extent = RenderExtent::new(3_840, 2_161).expect("extent is nonzero");

        assert!(matches!(
            validate_extent(
                extent,
                &limits,
                CoreResourcePlan::without_installed_frame(RenderExtent::ONE, extent),
            ),
            Err(ResizeError::NativePixelBudget {
                width: 3_840,
                height: 2_161,
                maximum_pixels: MAXIMUM_NATIVE_TRACE_PIXELS,
            })
        ));
    }

    #[test]
    fn trace_batches_respect_the_device_dispatch_dimension_at_4k() {
        let extent = RenderExtent::new(3_840, 2_160).expect("extent is nonzero");
        let maximum_dispatch_dimension = 512;
        let mut progress = TraceProgress::new(extent, maximum_dispatch_dimension);
        let [tile_columns, tile_rows] = tile_grid(extent);
        let mut covered_tiles = 0;

        while let Some(batch) = progress.next_batch() {
            let [workgroups_x, workgroups_y] = batch.workgroups();
            assert!(workgroups_x <= maximum_dispatch_dimension);
            assert!(workgroups_y <= maximum_dispatch_dimension);
            let [origin_x, origin_y] = batch.origin();
            assert_eq!(origin_y * tile_columns + origin_x, covered_tiles);
            progress.submitted(batch);
            covered_tiles += batch.len();
            progress.completed(f64::MIN_POSITIVE);
        }

        assert_eq!(covered_tiles, tile_columns * tile_rows);
    }

    #[test]
    fn resize_rejects_each_excess_texture_dimension() {
        let limits = wgpu::Limits::default();
        let maximum = limits.max_texture_dimension_2d;
        let too_wide = RenderExtent::new(maximum + 1, 1).expect("extent is nonzero");
        let too_tall = RenderExtent::new(1, maximum + 1).expect("extent is nonzero");

        assert!(matches!(
            validate_extent(
                too_wide,
                &limits,
                CoreResourcePlan::without_installed_frame(RenderExtent::ONE, too_wide),
            ),
            Err(ResizeError::ExtentLimit {
                width,
                height: 1,
                max_texture_dimension_2d,
            }) if width == maximum + 1 && max_texture_dimension_2d == maximum
        ));
        assert!(matches!(
            validate_extent(
                too_tall,
                &limits,
                CoreResourcePlan::without_installed_frame(RenderExtent::ONE, too_tall),
            ),
            Err(ResizeError::ExtentLimit {
                width: 1,
                height,
                max_texture_dimension_2d,
            }) if height == maximum + 1 && max_texture_dimension_2d == maximum
        ));
    }
}

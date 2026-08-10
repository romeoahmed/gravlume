# Repository Guidelines

## Project Structure & Module Organization

Gravlume is a Rust 2024 workspace. The root package (`src/main.rs`) starts the application. `crates/gravlume-desktop` owns the winit/egui lifecycle, while `crates/gravlume-render` owns wgpu capabilities, resources, passes, timing, and device errors. Keep WGSL beside its consumer in `crates/gravlume-render/src/shaders/`. Unit and GPU contract tests live beside implementation under `#[cfg(test)]`; versioned scientific inputs belong in `tests/fixtures/vN/`. Treat `docs/architecture.md`, `docs/validation.md`, and `docs/platform.md` as implementation contracts, not aspirational notes.

## Build, Test, and Development Commands

- `cargo fmt --all -- --check` verifies rustfmt output.
- `cargo test --workspace --all-targets --locked` runs the complete workspace suite; native GPU tests require a usable Metal or Vulkan adapter.
- `cargo clippy --workspace --all-targets --locked -- -D warnings` enforces the repository lint policy.
- `GRAVLUME_SMOKE_ONCE=1 cargo run --locked` opens the native stack, presents a frame, completes timing readback, and exits.
- `cargo tree -e features` audits feature closure after dependency changes.

Sandboxed agents must request elevated execution for Cargo commands rather than weakening or skipping verification.

## Coding Style & Naming Conventions

Use stable Rust 1.97, edition 2024, and rustfmt defaults (four-space indentation). Name modules, functions, and variables `snake_case`; types and traits use `UpperCamelCase`; constants use `SCREAMING_SNAKE_CASE`. Workspace Clippy groups `all`, `pedantic`, and `nursery` are enabled. Fix warnings at the root cause. Do not use `#[allow]`; reserve `#[expect(..., reason = "...")]` for a verified false positive. Avoid `unwrap`/`expect` in production paths; propagate typed errors. Add dependencies only to their direct consumer and specify at least `major.minor`; keep `Cargo.lock` committed.

## Testing Guidelines

Write deterministic tests around observable behavior, lifecycle transitions, numerical contracts, and GPU resource semantics. Use descriptive names such as `updates_are_transactional_and_generation_based`. Avoid assertions that freeze prose, private implementation, or speculative performance. Version fixture schemas instead of changing existing meanings. Add tests only when they protect a meaningful contract or reproduce a defect.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit style: `feat(render): ...`, `fix(desktop): ...`, `test(render): ...`, `docs: ...`. Keep commits atomic and explain intent. Pull requests should summarize behavior and contract changes, list verification commands, note tested OS/backend/adapter, link relevant issues, and include screenshots for visible UI changes. Update affected documentation whenever public behavior, platform support, or validation rules change.

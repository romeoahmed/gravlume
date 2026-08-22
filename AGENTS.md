# Repository Instructions

These instructions apply to the entire workspace. More specific `AGENTS.md` or `AGENTS.override.md` files take precedence within their directories.

## Sources of truth

- `Cargo.toml` and `Cargo.lock` define toolchain, dependency, and feature facts.
- `docs/product.md`, `docs/physics.md`, `docs/validation.md`, `docs/architecture.md`, and `docs/platform.md` are normative contracts.
- `docs/reference-implementation.md` and `docs/gpu-renderer.md` describe evidence that exists now.
- `docs/research/` records experiments and decisions; it never defines production behavior by itself.

Do not copy a contract into several documents. Keep one authoritative statement and link to it.

## Workspace boundaries

Gravlume is a Rust 2024 native desktop workspace:

- `gravlume-domain`: validated values and independent `f64` domain mathematics;
- `gravlume-reference`: CPU reference integration, fixtures, and comparisons;
- `gravlume-native-display`: the narrow audited native display-state interface;
- `gravlume-render`: wgpu resources, WGSL pipelines, publication, timing, and GPU errors;
- `gravlume-desktop`: the winit/egui composition root;
- root package: process entry point only.

Keep WGSL beside its Rust consumer in `crates/gravlume-render/src/shaders/`. Do not introduce a solver trait, render graph, compatibility layer, or public seam until a second real consumer exists.

## Change workflow

1. Read the applicable contract and inspect `git status` before editing.
2. Preserve unrelated staged and unstaged work; never overwrite user changes.
3. Prefer root-cause changes at the smallest stable seam. Breaking changes are allowed when they make the model clearer and all consumers migrate together.
4. Update the authoritative documentation in the same change as behavior or contract changes.
5. Run verification proportional to risk; Cargo commands require elevated sandbox execution.

Required workspace checks:

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

Use `GRAVLUME_SMOKE_ONCE=1 cargo run --locked` for native lifecycle, surface, publication, and timing changes. Run `cargo tree -e features` after dependency or target-feature changes.

## Rust and WGSL

- Use the Rust version and edition declared by the workspace `Cargo.toml`, rustfmt defaults, and
  workspace lints.
- Fix warnings at the root cause. Do not use `#[allow]`; use `#[expect(..., reason = "...")]` only for a verified lint condition.
- Avoid `unwrap` and `expect` in production. Preserve typed errors and their source chains.
- Keep domain types private and validated; use explicit `#[repr(C)]` DTOs at GPU boundaries.
- Treat WGSL floating-point, layout, resource-usage, and synchronization rules as contracts. Do not assume backend-specific ordering, fusion, subgroup width, or IEEE behavior beyond the specification.
- `unsafe` is denied workspace-wide. Native display modules may use narrowly scoped, documented `#[expect(unsafe_code)]` because they are the audited FFI boundary; do not spread unsafe into callers.

## Tests and scientific evidence

- Test observable behavior, lifecycle transitions, numerical budgets, ABI, and GPU resource semantics.
- Prefer deterministic examples for named scientific cases and property tests for algebraic domains and state transitions.
- Do not freeze prose, private helpers, pass counts, generated formatting, or speculative performance.
- Version scientific fixtures under `crates/gravlume-reference/fixtures/vN/`; never change an existing schema or profile meaning in place.
- CPU/GPU agreement is necessary but not sufficient. Preserve independent equations, high-precision checks, convergence evidence, and explicit applicability limits.
- Benchmark only a correctness-approved production workload. Record revision, platform, adapter, backend, scene, extent, profile, warm-up, sample method, and statistic.

## Documentation

- Write user and project documentation in concise Chinese; keep identifiers, API names, and established mathematical terms in their canonical form.
- Begin each document by stating its purpose and authority. Put current evidence, future design, and historical research in separate documents.
- Define each technical fact once. Replace duplicated formulas, thresholds, versions, and performance numbers with links to the authoritative location.
- Use descriptive headings and relative links. Avoid fragile line-number links, temporary paths, phase names, and status claims without evidence.
- Keep comments and source-level documentation in precise English; explain intent or invariants, not obvious syntax.

## Git

Follow Conventional Commits (`feat(render): ...`, `fix(desktop): ...`, `refactor(workspace): ...`, `test(render): ...`, `docs: ...`). Keep commits atomic, include `Signed-off-by` with `git commit -s`, and never commit or stage unless the user asks. Before committing, review the staged diff and rerun the relevant checks.

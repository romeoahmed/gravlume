# Native platform contract

Gravlume is a native Rust 2024 desktop application. The supported presentation paths are Metal on
macOS and Vulkan on Windows and Linux/Wayland. D3D12, GLES, X11, WebGPU-in-a-browser, and WebGL are
outside this contract.

This document describes the implementation that exists in the workspace. Experimental algorithms
and unimplemented GPU variants belong under `docs/research/`, not in this contract.

## 1. Toolchain and dependency boundaries

The workspace uses Rust 1.97 and edition 2024. `Cargo.toml` and `Cargo.lock` are authoritative for
the complete version and feature closure; the table records the direct technology boundaries.

| Responsibility | Direct crates | Boundary |
|---|---|---|
| window and UI | `winit 0.30`, `egui 0.36`, `egui-winit 0.36`, `egui-wgpu 0.36` | `gravlume-desktop` owns the event loop and UI; `gravlume-render` owns GPU rendering |
| GPU | `wgpu 30.0` | WGSL pipelines, surface negotiation, resource ownership, and GPU timing |
| native display state | `objc2`/`objc2-app-kit 0.3`, `windows 0.62`, `wayland-client 0.31`, `wayland-protocols 0.32` | private, audited platform modules behind `gravlume-native-display`'s safe API |
| mathematics and ABI | `glam 0.33`, `bytemuck 1.25`, `num-traits 0.2` | `glam` is implementation math; GPU DTOs use explicit `#[repr(C)]` layouts rather than domain types |
| fixtures and reference work | `serde 1.0`, `toml 1.1`, `rayon 1.12` | strict versioned fixture parsing and a bounded reference-computation pool |
| errors and diagnostics | `thiserror 2.0`, `tracing 0.1`, `tracing-subscriber 0.3`, `pollster 1.0` | one subscriber in the executable; typed errors at crate boundaries |
| verification | `proptest 1.11`, optional `criterion 0.8.2` | deterministic contract/property tests; Criterion is enabled only for the explicit GPU benchmark target |

Platform-specific Cargo features are additive overlays on the same workspace dependency: Metal is
enabled only on macOS, Vulkan only on Windows/Linux, and winit's Wayland features only on Linux.
Dependencies belong to their direct consumer. Changes to this graph require `cargo tree -e features`
and confirmation that X11 and unintended GPU backends remain absent.

## 2. Native desktop baseline

| Target | Backend | Release evidence |
|---|---|---|
| latest stable macOS | Metal | native surface, EDR state changes, headless compute, lifecycle tests, and smoke run |
| Windows 11 build 22621 or newer | Vulkan | named OS/adapter/driver, HDR toggle, cross-monitor movement, lifecycle tests, and smoke run |
| Linux desktop | Vulkan + Wayland | named distribution/compositor/adapter/driver, color-management-v1 feedback, lifecycle tests, and smoke run |

A candidate adapter must be WebGPU-compliant, non-software, expose `TIMESTAMP_QUERY`, support the
required `rgba16float` usages, and satisfy the project limits. Device creation requests exactly the
features consumed by production; it never requests `Features::all()` or copies adapter-wide buffer
maxima into the project allocation policy. The renderer currently requires only
[`TIMESTAMP_QUERY`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWebGPU.html#associatedconstant.TIMESTAMP_QUERY).

Failure to satisfy the baseline is an unsupported platform, not permission to change the physical
model, precision, resource semantics, or validation thresholds.

## 3. HDR state and presentation

HDR is resolved from two independent facts:

1. **Transport capability:** `SurfaceCapabilities::format_capabilities` must advertise the exact
   `Rgba16Float + ExtendedSrgbLinear` pair. The color space means native linear scRGB; wgpu does not
   encode a non-sRGB render target for the application.
2. **Live display state:** `gravlume-native-display` reports a typed `DynamicRange` snapshot and
   change notification for the display currently carrying the window.

Only the intersection enables HDR. A missing pair, explicit SDR state, system suppression, invalid
headroom, unsupported native integration, or a failed/unknown state query produces an SDR surface
with a typed reason. This is a deliberate fail-closed product policy: wgpu's all-`None`
`DisplayHdrInfo` means “unknown,” not “SDR,” so unknown native state is never silently reinterpreted
as a positive HDR decision.

The scene remains linear in `Rgba16Float`. The final pass tone-maps scene radiance, maps the
normalized scene/UI composite to the platform reference white in linear light, and writes either:

- linear values to `ExtendedSrgbLinear` HDR; or
- the encoding expected by the selected SDR surface format.

The surface format or an FP16 intermediate alone does not constitute end-to-end HDR.

### 3.1 macOS

The main-thread AppKit monitor reads the window's current `NSScreen`. A potential EDR component
value above `1.0` establishes display capability; the current value supplies live tone-map headroom
and may legitimately fall back to `1.0`. `NSWindowDidChangeScreenNotification`,
`NSApplicationDidChangeScreenParametersNotification`, and begin/end HDR suppression notifications
only mark output state dirty; the main thread then re-reads the screen.

This follows Apple's distinction between
[`maximumPotentialExtendedDynamicRangeColorComponentValue`](https://developer.apple.com/documentation/appkit/nsscreen/maximumpotentialextendeddynamicrangecolorcomponentvalue)
and the live
[`maximumExtendedDynamicRangeColorComponentValue`](https://developer.apple.com/documentation/appkit/nsscreen/maximumextendeddynamicrangecolorcomponentvalue).

### 3.2 Windows

Windows uses inbox WinRT `Windows.Graphics.Display.DisplayInformation`, not Windows App SDK and not
an application-level DXGI display query. On Windows 11 build 22621+, a minimal private projection of
`IDisplayInformationStaticsInterop::GetForWindow` creates and caches the object for the top-level
HWND. The owning UI thread has a `Windows.System.DispatcherQueue`; the monitor subscribes to
`AdvancedColorInfoChanged`, removes the token before window teardown, and lets dispatcher shutdown
finish while the winit message loop is still alive.

`CurrentAdvancedColorKind == HighDynamicRange` enables the native half of the HDR decision.
`MaxLuminanceInNits / SdrWhiteLevelInNits` supplies headroom and
`SdrWhiteLevelInNits / 80` supplies the linear scRGB UI-white scale. Microsoft requires caching the
window-bound object and listening for changes as the window moves or display parameters change:
[`GetForWindow`](https://learn.microsoft.com/en-us/windows/win32/api/windows.graphics.display.interop/nf-windows-graphics-display-interop-idisplayinformationstaticsinterop-getforwindow).
The 80-nit scRGB unit and linear UI adjustment follow Microsoft's
[Advanced Color guidance](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range).

### 3.3 Linux/Wayland

Linux builds Wayland only. The monitor creates a non-owning guest `wayland-client` connection from
the winit raw handles, binds `wp_color_manager_v1` version 2–3, and obtains the surface's preferred
parametric image description. It installs a new snapshot only after the description and information
events complete. Missing protocol support, an old protocol version, a non-parametric description,
incomplete luminance fields, or a dispatch failure remain typed unknown states and select SDR.

The application creates only read-only surface feedback. Vulkan WSI continues to own the
color-managed surface/presentation encoding. winit remains the sole socket reader; the guest queue
only dispatches already-read events during `about_to_wait`, with a distant wake guard because winit
0.30 can otherwise discard a wake that contains only guest-queue events.

The protocol contract is
[`color-management-v1`](https://wayland.app/protocols/color-management-v1), including version-2
`preferred_changed2` and `get_preferred_parametric` semantics.

### 3.4 Verification status

The macOS/Metal path has native runtime coverage in this workspace. Windows and Linux code paths
have source/API review and target-specific compile coverage where available; those do not replace
real HDR toggle, hotplug, cross-monitor, compositor, and driver testing on their target systems.

## 4. WGSL and GPU resource semantics

The handwritten shader modules live beside their Rust consumers in
`crates/gravlume-render/src/shaders/`. Rust assembles the production and diagnostic shader sources;
wgpu's locked WGSL frontend validates them during pipeline creation, and GPU contract tests create
and execute the relevant entry points. There is no WESL generator, checked-in generated shader, or
direct Naga dependency in the workspace.

Shader behavior follows the [WGSL specification](https://www.w3.org/TR/WGSL/):

- core arithmetic is `f32`; code does not assume `f64`, implementation-specific subgroup behavior,
  or stronger NaN/subnormal/fusion guarantees than WGSL provides;
- workgroup barriers are reached in uniform control flow and synchronize only the workgroup scope;
- storage atomics provide the queue/index operations that explicitly require atomicity;
- resource access modes match the declared binding types and texture formats;
- host DTO alignment, size, and field order match WGSL's memory-layout rules.

On the final trace batch, the selective shadow-coverage pipeline deliberately uses three dispatches
in one compute pass: base trace, edge classification, and edge refinement. WebGPU defines each
compute dispatch as a separate usage scope, so the candidate texture may be a sampled texture in
classification and a write-only storage texture in refinement without an illegal same-dispatch
alias. Classification uses an atomic counter and writes each accepted pixel index once; refinement
writes each listed pixel once. It does not rely on a dispatch-wide barrier, because WGSL exposes no
such primitive. See the official
[WebGPU resource-usage and synchronization rules](https://gpuweb.github.io/gpuweb/#resource-usages).

Rust domain types are not GPU ABI types. Uniform/storage DTOs have explicit `#[repr(C)]` layouts,
fixed integer discriminants, checked narrowing from binary64 domain inputs, and contract tests for
size/alignment and observable GPU behavior.

## 5. Release evidence

| Layer | Required evidence |
|---|---|
| dependency | committed lockfile; target backend/Wayland feature closure; no unintended protocol/GPU stack duplicates |
| capability | required feature/limit/format usages and structured rejection reason |
| shader | all production/diagnostic entry points create successfully; host/WGSL ABI contracts hold |
| headless GPU | odd extents, workgroup boundaries, multi-batch execution, shadow refinement, resize, and readback |
| numerical | termination/branch agreement and continuous observables within `docs/validation.md` budgets |
| performance | fixed revision, OS, adapter, driver, power mode, scene, extent, warm-up, and raw GPU timestamps |
| native output | HDR/SDR transitions, cross-display movement, surface recovery, and one complete presented generation |

One adapter probe is not a release matrix. A wgpu, winit, platform-binding, or protocol upgrade is a
coordinated change: re-audit official API semantics, Cargo features, WGSL validation, native display
lifecycle, and the three supported platform matrices.

# 架构与实现合同

本文只描述仓库中存在的模块、接口和生命周期不变量。未来算法放在[渲染设计](rendering.md)，实验过程放在 [`research/`](research/)；两者不能冒充当前 API。

## 1. 依赖方向

```text
gravlume (binary)
└─ gravlume-desktop
   ├─ gravlume-domain
   ├─ gravlume-native-display
   └─ gravlume-render
      ├─ gravlume-domain
      └─ gravlume-native-display

gravlume-reference
└─ gravlume-domain
```

- `domain`：validated `f64` 领域值、Kerr–Newman 几何、观察者与初始光线；不依赖 GPU/UI/serde。
- `reference`：独立 CPU oracle、事件定位、fixture 和比较；不依赖 renderer。
- `native-display`：AppKit、inbox WinRT 与 Wayland color-management 的窄安全边界；只导出值类型和监视器生命周期。
- `render`：wgpu tracing、方向重建、阴影覆盖率、HDR/SDR 输出、资源与错误语义。
- `desktop`：唯一拥有 winit/egui 生命周期的组合根。

没有 `research` crate、通用 render graph、可替换 solver trait、WESL 生成层或 `xtask`。出现第二个真实消费者前不预建抽象。

## 2. 领域接口

```rust
pub struct PhysicalSceneInput { /* 未验证的调用输入 */ }
pub struct PhysicalScene { /* validated scene */ }
pub struct PerspectiveView { /* validated extent + vertical FOV */ }
pub struct ImageSample { /* pixel + subpixel，无派生 sight state */ }
pub struct Observation { /* PhysicalScene + PerspectiveView */ }

impl PhysicalScene {
    pub fn new(input: PhysicalSceneInput) -> Result<Self, ValidationReport>;
}

impl PerspectiveView {
    pub fn new(
        width: NonZeroU32,
        height: NonZeroU32,
        vertical_fov: Angle,
    ) -> Result<Self, ValidationReport>;
}
```

`Input` 明确表示尚未验证的数据；成功构造后的领域值字段私有且始终满足不变量。`Observation::initial_ray` 会针对自己的 view 重新验证 `ImageSample`，建立 future-directed/null/frequency 合同。持久化 fixture DTO 与领域类型分离；fixture 字段名和 schema 版本不能随内部重命名漂移。

`KerrSchildChart` 表示 ingoing/outgoing chart，不是“坐标数据”；`Extremality` 表示 subextremal/extremal/superextremal 分类，不再使用含义模糊的 `ParameterState`。

## 3. Reference 接口

Reference 分成两个有意不同的 seam：

- `ObservationTracer + ObservationTrace`：从 validated `Observation` 发起规范化 backward trace；
- `GeodesicTracer + GeodesicTrace`：fixture、收敛研究和批处理直接使用的低层 canonical-state tracer。

```rust
let request = ObservationTrace::new(id, &observation, sample, policy)?;
let outcome = ObservationTracer::baseline_v1().trace(request)?;

let tracer = GeodesicTracer::new(spacetime, policy, events)?;
let outcome = tracer.trace(GeodesicTrace::new(id, state, direction));
```

数值失败、步数耗尽与 non-convergence 是 typed `ReferenceOutcome`，不是 panic。只有输入归一化或配置错误走 `Err`。CPU 与 WGSL 独立实现核心方程，避免同一符号错误同时污染 oracle 与被测对象。

## 4. Renderer 接口与内部模块

跨 crate 接口只保留桌面层实际消费的操作：

```rust
pub struct Renderer;

impl Renderer {
    pub async fn new(
        window: Arc<Window>,
        observation: &Observation,
        dynamic_range: DynamicRange,
    ) -> Result<Self, RendererInitError>;

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), ResizeError>;
    pub fn advance_trace(&mut self) -> Result<(), RendererError>;
    pub fn poll(&mut self) -> Result<RendererUpdate, RendererError>;
    pub fn present(/* egui output */) -> Result<PresentResult, RendererError>;
}
```

`advance_trace` 自行判断是否可推进并安全 no-op；调用者不需要先查询 `trace_can_advance`，也不观察 Submitted/Idle 这类实现状态。`PresentResult` 只区分已提交与可恢复的 `PresentSkip`。`RendererUpdate` 汇总已发布 generation、已完成 presentation 与 device events。

`gravlume-render/src` 的职责：

| 模块 | 所有权 |
|---|---|
| `renderer.rs` | surface/device/queue、extent generation、submission、原子发布和 presentation |
| `renderer/frame.rs` | frame bundle、trace batch、自适应预算和事务式资源准入 |
| `ray_tracer.rs` | Observation→GPU DTO、主追迹与方向重建 pipeline、trace image |
| `shadow_coverage.rs` | 边缘分类/refinement pipeline 与 scratch target |
| `display.rs` | scene/UI 最终线性合成与 surface encoding |
| `capabilities.rs` | adapter gate 与纯 surface-output resolver |
| `timing.rs` | 有界 timestamp query/readback 状态 |
| `error.rs` | error scopes、device callbacks 与 typed errors |
| `extent.rs` | 非零 extent 与 generation 转移 |
| `benchmark.rs` | 仅在 `gpu-benchmarks` feature 下暴露给 Cargo benchmark crate 的 seam |

领域状态与稳健浮点算子分别位于 `domain/state.rs`、`domain/numerics.rs`；极值分类使用的精确 binary64 算术是 `domain/spacetime/exact_binary.rs` 的私有实现。Reference 的公共 fixture 接口保留在 `fixture.rs`，v1 wire DTO 和 profile 规则位于 `fixture/v1.rs`。这些子模块使用 Rust 2018+ 的 `module.rs + module/child.rs` 布局，不引入同名 `mod.rs`。[Rust 模块文件](https://doc.rust-lang.org/stable/book/ch07-05-separating-modules-into-different-files.html)

WGSL 与消费它的 renderer 放在 `src/shaders/`：`trace.wgsl` 是基线积分器，`direction_reconstruction.wgsl` 是保守 tile accelerator，`shadow_coverage.wgsl` 只解决 capture/escape 边界覆盖率，`display.wgsl` 负责输出。文件名描述数学职责，不使用 roadmap 阶段名。

## 5. GPU 计算与发布

每个新 extent generation 创建隐藏的 native-resolution candidate：

```text
direction stencil nodes
  -> tile resolve (reconstruct or full KS fallback)
  -> final-batch shadow edge classify/refine
  -> timestamp readback
  -> generation check
  -> promote candidate texture view
  -> one redraw/present
```

- direction reconstruction 用共享 4-pixel node grid 和每 tile `3×3` stencil；只有全部 Escape 且方向误差 gate 通过才重建，其余逐像素 Cartesian Kerr–Schild fallback。
- pure-Kerr interval Bernstein certificate 只在严格支持域证明 capture 时跳过积分；任何不确定都 fallback。
- shadow coverage 先从不可变 candidate alpha 读取 Horizon/Escape tag，再在独立 dispatch 写回四个 rotated-grid 子样本的线性平均；不在一个 dispatch 中制造邻域读写竞争。
- incomplete candidate 永不进入 display bind group；过期 generation 的 completion 只能回收资源，不能发布。

上一张完整 FP16 scene 跨 resize 保留并 aspect-fit。compute batch 不 acquire surface、不运行 egui、不 present；因此不会把隐藏的空间批次“扫描”给用户。

## 6. Surface 与事件循环

桌面层遵循 winit 0.30 `ApplicationHandler`：首次 `resumed` 创建 window/renderer，重复 resume/suspend 必须幂等；所有窗口事件先交给 egui；`RedrawRequested` 才 present；`about_to_wait` 只做非阻塞 GPU poll、合并 resize 和安排 deadline。[winit `ApplicationHandler`](https://docs.rs/winit/0.30.13/winit/application/trait.ApplicationHandler.html)

wgpu surface 不变量：

- zero extent 不 configure/acquire；
- acquired texture 在 present/drop 前不重配；
- `Suboptimal` 完成本帧后重配，`Outdated` 重配，`Lost` 重建，`Timeout/Occluded` 跳过；
- `Surface::configure` 在已配置 surface 上等待 GPU idle，因此 live resize 必须合并，不能为每个原始事件同步重建。[wgpu `Surface`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Surface.html)

egui 顺序为 `on_window_event → take_egui_input → Context::run → handle_platform_output`。纹理先安装，render 编码后、submit 前后按 egui-wgpu 合同释放；scene 与 UI 分开渲染，最终在线性空间合成。[egui-winit `State`](https://docs.rs/egui-winit/0.36.1/egui_winit/struct.State.html)

## 7. HDR 与平台边界

`native-display::DynamicRange` 是平台状态的唯一 DTO；renderer 直接消费它，不再维护同构映射类型。只有可靠 HDR 状态、有限且有效的 headroom/reference-white，以及 surface 广告的精确 `Rgba16Float + ExtendedSrgbLinear` pair 同时满足时启用 HDR。其余情况带 `SdrReason` fail-closed 到 SDR。

原生对象与 unsafe 只存在于三个带审核说明的 `native-display/platform/*.rs` 模块。render 不导入 AppKit、WinRT、Wayland 或 backend-private wgpu 类型。native-only 支持域是 macOS、Windows 和 Wayland Linux；其他 target 在编译期拒绝，不提供无消费者的兼容垫片。

## 8. GPU ABI 与错误

GPU DTO 使用 `#[repr(C)]` 标量数组、显式 padding 与 `bytemuck::Pod`；领域类型和 `glam` 不直接上传。WGSL binding、access、format、enum discriminant 与 Rust size/offset 由 Naga/contract tests 验证。当前没有上传 `glam` 类型，因此不启用 `glam/bytemuck`。

错误按处理者分类：

- `ValidationReport`：调用者可修正的领域输入；
- `ResizeError`：事务式 rebuild 拒绝或资源失败；
- `PresentSkip`：可恢复的 surface 状态；
- `DeviceEvent`：异步 validation/OOM/lost/internal；
- `RendererInitError` / `RendererError`：不能在当前操作内恢复的 renderer 错误。

production 路径不使用 `unwrap/expect`，错误保留 typed source chain；UI/日志边界才转文本。

## 9. 内存、benchmark 与验证

production 不常驻每像素科学 records。candidate 为 `8 B/pixel` HDR、UI 为 `4 B/pixel`，另加按 extent 精确计算的 direction reconstruction 与 shadow scratch；published scene 单独为 `8 B/pixel`。三份 `16 B/pixel` record plane 只在 small-extent GPU contract capture 中创建。

4K worst transactional plan 在分配前按 published + installed + replacement 逐项计算，并受 256 MiB core gate 与 3840×2160 pixel policy 约束。surface images、driver heap/alignment 不在该逻辑账本内，不能把 gate 冒充实测显存峰值。

Criterion benchmark 位于 Cargo 独立 `benches/` crate；Cargo 官方规定它只能调用 library public API，因此 feature-gated `gravlume_render::benchmark::register` 是有意且最小的 benchmark seam，不使用 `#[doc(hidden)]` 假装私有。[Cargo benchmark targets](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#benchmarks)

合并门槛：

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

测试保护 observable、生命周期、数值预算与 GPU 资源语义；不冻结错误文案、私有 helper、pass 数量或未经证实的性能阈值。性能结论必须记录 target/OS/adapter/backend/profile/scene/extent/样本与统计方法。

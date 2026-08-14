# 架构与实现合同

本文定义仓库现有模块的职责、接口语义和生命周期不变量。它不复制 public Rust 签名，也不收录未来算法：签名以源码/rustdoc 为准，候选设计见[渲染设计](rendering.md)，实验过程见[研究记录](research/)。

## 依赖方向

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

| crate                     | Interface responsibility                                            | 禁止反向渗透                          |
| ------------------------- | ------------------------------------------------------------------- | ------------------------------------- |
| `gravlume-domain`         | validated `f64` values、时空、observer、view 与 initial ray         | 不依赖 GPU、UI、serde 或 fixture      |
| `gravlume-reference`      | 独立 CPU oracle、event、fixture、batch 与 comparison                | 不依赖 renderer 或 WGSL 实现          |
| `gravlume-native-display` | safe `DynamicRange` snapshot 与 monitor lifecycle                   | 不配置 wgpu surface，不暴露原生对象   |
| `gravlume-render`         | GPU trace、frame resources、publication、display、timing 与错误语义 | 不拥有 event loop，不导入平台原生接口 |
| `gravlume-desktop`        | winit/egui 组合根、输入、调度与用户可见状态                         | 不实现物理或 GPU resource policy      |
| root binary               | 日志与启动                                                          | 不承载业务模块                        |

没有 `research` crate、通用 render graph、可替换 solver trait、WESL 生成层或 `xtask`。只有出现第二个真实消费者时才增加 seam。

## Domain 与 reference seams

`Input` 后缀表示未验证调用输入。构造成功后的领域值字段私有并持续满足不变量；fixture DTO 与领域类型分离，wire 字段和 schema version 不随内部重命名漂移。

核心 domain 流程：

```text
PhysicalSceneInput -> PhysicalScene
                          + EquatorialCircularEmitter
                          + PerspectiveView -> Observation
Observation + ImageSample -> InitialViewRay
```

`Observation::initial_ray` 针对自己的 view 验证 sample，并建立 future-directed、null 与 frequency 合同。`KerrSchildChart` 表示 chart convention；`Extremality` 表示 subextremal/extremal/superextremal 分类。

`EquatorialCircularEmitter` 是 domain-owned validated source；它原子携带 inclusive radial
interval 与 `inverse-cube-bolometric-v1` 的 $I_6$，但不携带 solver、pipeline 或 display policy。
Reference 保留两个有意不同的接口：

- `ObservationTracer`：从 validated `Observation` 的 Physical Scene 读取 emitter，并一次性构造规范化 backward trace、Source Anchor、Frequency Ratio、emitted intensity 与 vacuum-observed intensity；
- `GeodesicTracer`：fixture、收敛研究和批处理使用 canonical state。

`GeodesicTracer` 不猜测 canonical state 的 observer frequency，也不把所有 equatorial event 解释成 emitter。source event 只在能够证明 $M=\omega_{\rm obs}=1$ 的 Observation seam 安装；Physical Scene 无 emitter 时 sky outcome 不携带 source observable。

输入或配置错误、无 timelike circular emitter 与非法 radiance 返回 typed `Err`；数值失败、步数耗尽与 non-convergence 是 `ReferenceOutcome`，不是 panic。CPU 与 WGSL 保持独立计算图，避免同一个符号错误同时污染 oracle 和被测对象。

## Renderer interface

桌面层只需要：创建 renderer、提交最新 extent、推进隐藏 trace、非阻塞 poll、present、suspend/resume、更新 display state 与读取只读 diagnostics。调用者不观察内部 batch、query 或 pipeline 状态。

接口语义：

- `resize` 是事务式请求；失败时保留上一组可用资源；
- `advance_trace` 自行判断是否可推进，不要求先查询内部状态；
- `poll` 汇总 publication、presentation completion 与 device events；
- `present` 只报告已提交或可恢复的 surface skip；
- `suspend/resume` 幂等，并保留上一张完整 scene；
- `update_output` 只换 display contract，不使 geometry generation 失效。

## Renderer modules

| 模块                 | 所有权                                                                                    |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `renderer.rs`        | instance/surface/device/queue、extent generation、submission、publication 与 presentation |
| `renderer/frame.rs`  | frame bundle、trace scheduling、resource admission 与事务式 rebuild                       |
| `ray_tracer.rs`      | Observation→GPU DTO、private sealed TracePlan、plan-specific pipeline/scratch 与 candidate image |
| `shadow_coverage.rs` | capture/escape 边缘分类、选择性 subpixel refinement 与 scratch                            |
| `display.rs`         | scene/UI 线性合成、tone mapping 与 surface encoding                                       |
| `capabilities.rs`    | adapter baseline 与纯 surface-output resolver                                             |
| `timing.rs`          | 有界 timestamp query、readback 与 submission lifecycle                                    |
| `error.rs`           | error scopes、device callbacks 与 typed renderer errors                                   |
| `extent.rs`          | non-zero extent、paused state 与 generation transition                                    |
| `benchmark.rs`       | `gpu-benchmarks` feature 下供 Cargo bench target 使用的最小 seam                          |
| `gpu_capture.rs`、`gpu_trace_tests.rs`、`test_device.rs` | `cfg(test)` 下的原生 GPU fixture、readback 与合同测试 |

WGSL 位于 `src/shaders/`：

| shader                       | 职责                                                      |
| ---------------------------- | --------------------------------------------------------- |
| `kerr_schild_trace.wgsl`     | 精确 Cartesian Kerr–Schild integration 与 observables     |
| `geodesic_acceleration.wgsl` | interval capture、escape-direction map 与完整 KS fallback |
| `lensing_preview.wgsl`       | termination/direction 到 scene-linear preview             |
| `surface_preview.wgsl`       | equatorial Source Anchor/Frequency Ratio 的直接 bolometric transport |
| `surface_trace_capture.wgsl` | test-only surface GeometricSample serialization           |
| `shadow_coverage.wgsl`       | shadow boundary classification 与 selective refinement    |
| `display.wgsl`               | scene/UI composite 与 HDR/SDR output mapping              |
| `*_capture.wgsl`             | test-only scientific readback entry points                |

文件名描述数学或渲染职责，不使用 roadmap 阶段名。Rust 负责组合生产与 capture shader source；仓库不维护生成的 WGSL 副本。

## GPU trace 与 publication

每个 active extent generation 拥有隐藏的原生分辨率 candidate：

```text
private TracePlan
  -> sky: escape-map -> reconstruct/full-KS trace -> final-batch shadow refine
  -> surface: full-KS trace -> immediate bolometric transport
  -> plan-sized timestamp readback
  -> generation check
  -> promote candidate texture view
  -> request one presentation
```

- escape-direction map 在共享 4-pixel node grid 上追踪；每个 `8×8` tile 读取 `3×3` stencil，只有 branch 一致且方向误差通过才重建；
- interval Bernstein certificate 只在严格支持域证明 capture，无结论时执行完整 KS；
- shadow classification 从不可变 candidate 读取 alpha tag，refinement 在后续 dispatch 写回真实 subpixel 平均；
- incomplete candidate 永不进入 display bind group；stale completion 只能回收资源；
- candidate 完成后直接提升 texture view，不做同尺寸 publication copy。

上一张完整 FP16 scene 跨 resize 保留并 aspect-fit。compute batch 不 acquire surface、不运行 egui、不 present，因此隐藏批次不会以扫描或低分辨率过渡暴露给用户。

## Event loop 与 surface

桌面层使用 winit 0.30 `ApplicationHandler`：

- `resumed` 创建或恢复 window、display monitor 与 renderer；重复 resume/suspend 幂等；
- window event 先交给 egui，再处理应用语义；
- 只有 `RedrawRequested` 执行 presentation；
- `about_to_wait` 做非阻塞 GPU/native-display poll、合并 resize 和安排下一 deadline。

Surface 不变量：

- zero extent 不 configure 或 acquire；
- acquired texture 在 present/drop 前不重配；
- `Suboptimal` 完成本帧后重配，`Outdated` 强制重配，`Lost` 重建；`Timeout`/`Occluded` 跳过并等待恢复信号；
- live resize 合并最新 physical extent，避免为每个原始事件同步等待 GPU idle；
- suspend 期间不分配 replacement；恢复 surface 后读取一次最新 inner size。

这些语义来自 [`ApplicationHandler`](https://docs.rs/winit/0.30.13/winit/application/trait.ApplicationHandler.html)与 [`Surface::configure`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Surface.html#method.configure)。

## HDR 与 native display seam

`gravlume-native-display::DynamicRange` 是平台状态的唯一跨 crate DTO。Renderer 只在以下事实同时成立时选择 HDR：

1. native monitor 给出可靠 HDR 状态与有效 headroom/reference white；
2. surface 广告精确的 `Rgba16Float + ExtendedSrgbLinear` pair。

否则 resolver 选择 SDR 并保留 `SdrReason`。native object 与 `unsafe` 只存在于三个 `native-display/src/platform/*.rs` 模块；renderer 不导入 AppKit、WinRT、Wayland 或 backend-private wgpu 类型。完整平台语义见[平台合同](platform.md)。

## GPU ABI 与错误

GPU DTO 使用 `#[repr(C)]` 标量数组、显式 padding 与 `bytemuck::Pod`。领域类型和 `glam` 不直接上传；因此当前不启用 `glam/bytemuck`。Host/WGSL size、offset、binding、access、format 与 discriminant 由 ABI/GPU contracts 验证。

错误按处理者分类：

| 类型                                  | 处理者                                          |
| ------------------------------------- | ----------------------------------------------- |
| `ValidationReport`                    | 修正领域输入                                    |
| `GpuTraceInputError`                  | 修正不可表示或不满足 GPU profile 的 Observation |
| `ResizeError`                         | 处理事务式 rebuild 拒绝或资源失败               |
| `PresentSkip`                         | 等待可恢复 surface 状态                         |
| `DeviceEvent`                         | 展示异步 validation/OOM/lost/internal 诊断      |
| `RendererInitError` / `RendererError` | 终止当前无法恢复的 renderer 操作                |

Production 不使用 `unwrap/expect` 恢复错误；typed source chain 只在 UI/日志接口转换为文本。

## 内存与资源预算

Production 不常驻每像素科学 records：

- candidate HDR：`8 B/pixel`；
- UI target：`4 B/pixel`；
- published scene：`8 B/pixel`；
- escape map 与 shadow scratch：按 extent 精确计算；
- 四个 `16 B/pixel` scientific planes：只在 small-extent test capture 中创建。

分配 replacement 前，renderer 计算 published + installed + replacement 的完整 core plan，并同时执行 3840×2160 pixel policy 与 256 MiB core-resource gate。Surface images、driver heap 和 alignment 不在该逻辑账本内；这项 gate 不是实测显存峰值。

## 验证与 benchmark

测试保护 observable、生命周期、数值预算、ABI 与 GPU resource semantics；不冻结错误文案、private helper、pass 数量或未经证明的性能阈值。当前覆盖见 [GPU 证据](gpu-renderer.md)。

Cargo `benches/` target 只能使用 library public interface，因此 feature-gated `gravlume_render::benchmark::register` 是一个有真实第二消费者的窄 seam。[Cargo benchmark targets](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#benchmarks) Benchmark 方法与历史数据只在[研究记录](research/gpu-benchmark-methodology.md)维护。

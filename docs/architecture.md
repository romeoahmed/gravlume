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
EquatorialCircularEmitter + SurfaceTransport -> EquatorialSurface
PhysicalScene + EquatorialSurface + PerspectiveView -> Observation
Observation + ImageSample -> InitialViewRay
```

`Observation::initial_ray` 针对自己的 view 验证 sample，并建立 future-directed、null 与 frequency 合同。`KerrSchildChart` 表示 chart convention；`Extremality` 表示 subextremal/extremal/superextremal 分类。

`EquatorialCircularEmitter` 是 domain-owned validated source；它原子携带 inclusive radial
interval 与 $I_6$，并由 `EquatorialEmissionModel` 明确区分 `inverse-cube-bolometric-v1` 和携带
$T_6$ 的 `inverse-cube-blackbody-v1`，不靠 nullable temperature 推断类型，也不携带 solver、
pipeline 或 display policy。`SurfaceTransport::{Vacuum, HomogeneousScalar}` 明确表达完整 observer
path；`EquatorialSurface` 原子绑定 emitter 与 transport，因而不存在 slab-only scene。Blackbody
surface 的非零 slab emission 必须在 `EquatorialSurface::new` 处解析为
`ScalarSlabEmissionModel::BlackbodyV1`。`HomogeneousScalarSlab` 只保存总 optical depth、integrated
emission 与显式 emission model；它不伪装成空间变化的 volume medium。
Reference 保留两个有意不同的接口：

- `ObservationTracer`：从 validated `Observation` 读取 `SceneRadiance`，并一次性构造规范化 backward
  trace、branch key、Source Anchor、Frequency Ratio、vacuum/final bolometric intensity 与可选
  blackbody bands；另以五条真实射线提供 branch-checked surface footprint；
- `GeodesicTracer`：fixture、收敛研究和批处理使用 canonical state。

`GeodesicTracer` 不猜测 canonical state 的 observer frequency，也不把所有 equatorial event 解释成 emitter。source event 只在能够证明 $M=\omega_{\rm obs}=1$ 的 Observation seam 安装；Physical Scene 无 emitter 时 sky outcome 不携带 source observable。

输入或配置错误、无 timelike circular emitter 与非法 radiance 返回 typed `Err`；数值失败、步数耗尽与 non-convergence 是 `ReferenceOutcome`，不是 panic。`ReferenceTerminal` 把 event、escape direction 与 surface observable 放在各自 terminal variant 内；event termination 必有 `LocalizedEvent`，纯 geodesic surface 与 resolved observation surface 是不同 variant。CPU 与 WGSL 保持独立计算图，避免同一个符号错误同时污染 oracle 和被测对象。

## Renderer interface

桌面层只需要：创建 renderer、提交最新 extent、推进隐藏 trace、非阻塞 poll、present、按需请求一个 sample inspection、suspend/resume、更新 display state 与读取只读 diagnostics。调用者不观察内部 batch、query 或 pipeline 状态。

接口语义：

- `resize` 是事务式请求；失败时保留上一组可用资源；
- `advance_trace` 自行判断是否可推进，不要求先查询内部状态；
- `poll` 汇总 publication、presentation completion、sample-inspection completion 与 device events；
- `present` 只报告已提交或可恢复的 surface skip；
- `suspend/resume` 幂等，并保留上一张完整 scene；
- `update_output` 只换 display contract，不使 geometry generation 失效。
- `request_sample_inspection` 只接纳 active extent 与 published generation/extent 完全一致时的 validated
  `ImageSample`；zero extent 或 retained publication 与 active extent 不一致时拒绝。Renderer 捕获
  published generation、extent 与 sample 并返回 ticket；caller 不提供 identity、solver 或 GPU handle。
  固定槽位最多容纳一个 pending request。
- resize/suspend 只能逻辑取消已提交 inspection；槽位必须等 submission、mapping 和 mapped view 全部
  drain 后才能复用。每个 ticket 经 `poll` 恰好产生一次 completed、cancelled 或 typed failed completion。
- completed inspection 把目标 generation 的实际 `Rgba16Float` texel 与 fresh full-KS/WGSL-binary32
  retrace 分开返回。两者的 producer、精度与 refinement 路径不同，接口不声明数值或 bit identity；
  terminal-specific 类型负责排除无意义的 source、branch 与 radiance 组合。
- `capture_scene_linear` 是显式阻塞的 export 操作，只读取已原子发布的 surface generation；它返回
  `ScientificTexel` slice 与 bolometric/final-spectral/LUT 模型误差 metadata，不经过 display 或
  UI。每个 texel 把 `Rgba16Float` binary16 words 与 alpha-tag classification 绑定在同一接口；只有
  `SurfaceRadiance` 可读取 physical RGB。Metadata 直接复用 validated `EquatorialSurface`，vacuum
  由 `SurfaceTransport::Vacuum` 显式表达。该接口导出最终 radiance 与 texel kind，不导出每像素
  source anchor、branch、frequency ratio、travel time 或 event diagnostics；这些结构化证据由上述
  单样本 production inspection 提供，但它不是持久 artifact container 或整帧 record plane。

## Renderer modules

| 模块                                                     | 所有权                                                                                       |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `renderer.rs`                                            | instance/surface/device/queue、extent generation、submission、publication 与 presentation    |
| `renderer/frame.rs`                                      | frame bundle、trace scheduling、resource admission 与事务式 rebuild                          |
| `trace.rs`、`trace/input.rs`、`trace/shader.rs`          | private sealed `TracePlan` 与 pipeline/scratch、受检 GPU ABI packing、唯一有序 WGSL 组合入口 |
| `trace/shadow_coverage.rs`                               | capture/escape 边缘分类、选择性 subpixel refinement 与 scratch                               |
| `trace/inspection.rs`                                    | public typed evidence、ticket、completion 与 error interface                                 |
| `trace/inspection/protocol.rs`                           | 固定 host/WGSL ABI、readback bytes 与 strict typed decoder                                   |
| `trace/inspection/kernel.rs`                             | production/corpus 共用的 private compute pipeline、binding 与 ordered dispatch               |
| `trace/inspection/slot.rs`                               | production 单槽 request、generation/cancel/map GPU lifecycle                                 |
| `trace/inspection/corpus.rs`                             | `cfg(test)` sparse batch 的 limit admission、linear readback 与 ordered decode               |
| `spectral_lut.rs`                                        | versioned Planck boxcar LUT 的独立 host generator 与固定布局                                 |
| `scientific_capture.rs`                                  | 已发布 surface texture 的显式 readback、texel kind、解释 metadata 与 numerical budgets       |
| `display.rs`                                             | scene/UI 线性合成、tone mapping 与 surface encoding                                          |
| `capabilities.rs`                                        | adapter baseline 与纯 surface-output resolver                                                |
| `timing.rs`                                              | 有界 timestamp query、readback 与 submission lifecycle                                       |
| `error.rs`                                               | error scopes、device callbacks 与 typed renderer errors                                      |
| `extent.rs`                                              | non-zero extent、paused state 与 generation transition                                       |
| `benchmark.rs`                                           | `gpu-benchmarks` feature 下供 Cargo bench target 使用的最小 seam                             |
| `gpu_capture.rs`、`gpu_trace_tests.rs`、`test_device.rs` | `cfg(test)` 下的原生 GPU fixture、readback 与合同测试                                        |

WGSL 位于 `src/shaders/`：

| shader                            | 职责                                                                       |
| --------------------------------- | -------------------------------------------------------------------------- |
| `trace_protocol.wgsl`             | host-shareable ABI、trace 状态、termination 与公共数值 helper              |
| `kerr_schild_dynamics.wgsl`       | Cartesian Kerr–Schild geometry、Hamilton RHS 与 RK4                        |
| `geodesic_events.wgsl`            | dense event localization、invariant、observable 与 branch evidence         |
| `geodesic_integration.wgsl`       | per-ray integration state machine 与完整 KS entry points                   |
| `lensing_preview.wgsl`            | termination/direction 到 scene-linear preview                              |
| `analytic_sky_preview.wgsl`       | 完整 KS 结果到 analytic-sky presentation entry point                       |
| `surface_transport.wgsl`          | inverse-cube/slab transport、范围安全缩放与三 band 向量运算                |
| `bolometric_surface_preview.wgsl` | equatorial source 的直接 bolometric transport                              |
| `blackbody_surface_preview.wgsl`  | blackbody LUT、三 boxcar bands 与 spectral slab transport                  |
| `surface_trace_capture.wgsl`      | test-only surface GeometricSample serialization                            |
| `surface_footprint_capture.wgsl`  | test-only branch-checked source-chart finite difference                    |
| `sample_inspection.wgsl`          | runtime-sized inspection kernel；production `N=1`，tests 使用 sparse batch |
| `*_sample_inspection.wgsl`        | plan-specific scene-value adapter                                          |
| `shadow_coverage.wgsl`            | shadow boundary classification 与 selective refinement                     |
| `display.wgsl`                    | scene/UI composite 与 HDR/SDR output mapping                               |
| `*_capture.wgsl`                  | test-only scientific readback entry points                                 |

文件名描述数学或渲染职责，不使用 roadmap 阶段名。上述四个 trace core fragment 不是可独立编译的
shader module；`trace/shader.rs` 是生产、shadow 与 test capture source 顺序的唯一所有者。仓库不维护
生成的 WGSL 副本。

## GPU trace 与 publication

每个 active extent generation 拥有隐藏的原生分辨率 candidate：

```text
private TracePlan
  -> sky: full-KS trace -> final-batch shadow refine
  -> bolometric surface: full-KS trace -> immediate g^4 + slab transport
  -> blackbody surface: full-KS trace -> gT + LUT bands + slab transport
  -> trace timestamp resolve/copy/map-on-submit + bound generation
  -> promote candidate texture view
  -> request one presentation
```

- 三种 plan 都从同一完整 Cartesian Kerr–Schild trace 形成 terminal、方向、travel time 与 diagnostics；
- shadow classification 从不可变 candidate 读取 alpha tag，refinement 在后续 dispatch 写回真实 subpixel 平均；
- incomplete candidate 永不进入 display bind group；stale completion 只能回收资源；
- candidate 完成后直接提升 texture view，不做同尺寸 publication copy。
- [`CommandEncoder::map_buffer_on_submit`](https://docs.rs/wgpu/30.0.1/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit) 把 timestamp copy 与 mapping 绑定到 producing encoder；readback
  completion 携带自己的 extent generation，不另维护可错配的 parallel submission slot。
- surface RGB 只有 alpha tag `2.0` 才是 metadata 所述 physical radiance；escape 的 `1.0` 是解析方向
  preview、horizon 是零、负 alpha 是 failure。Display 忽略 scene alpha，scientific capture 必须分类。

可选 inspection 是 publication 的只读旁路消费者：同一 encoder 产生 fresh record、复制 request
绑定的一个 published texel，并用 `map_buffer_on_submit` 把 readback 绑定到该 submission。它不修改
candidate、published texture 或默认 frame resource plan；精确 ABI、copy 顺序与 Metal 证据见
[按需单样本检查决策](research/on-demand-sample-inspection.md#gpu-protocol-与资源证据)。

上一张完整 FP16 scene 跨 resize 保留并 aspect-fit。compute batch 不 acquire surface、不运行 egui、不 present，因此隐藏批次不会以扫描或低分辨率过渡暴露给用户。

## Event loop 与 surface

桌面层使用 winit 0.30 `ApplicationHandler`：

- `resumed` 创建或恢复 window、display monitor 与 renderer；重复 resume/suspend 幂等；
- window event 先交给 egui，再处理应用语义；
- 未被 egui 消费的左键释放只在 physical client extent 与 current publication extent 相等时，把有效
  physical cursor 映射为中心 `ImageSample`，并展示同代 published texel 与 retrace；新 publication
  使旧 generation 结果失效。只有 no-op resize 仍匹配该 extent 时，retained publication 才能结束
  viewport wait；
- 只有 `RedrawRequested` 执行 presentation；
- `about_to_wait` 做非阻塞 GPU/native-display poll；`DesktopSchedule` 统一拥有 live resize、repaint、
  GPU poll 与 native monitor deadline，并返回唯一下一次 wake。

Surface 不变量：

- zero extent 不 configure 或 acquire；
- surface configuration 与 physical client extent 必须匹配；尺寸不一致时 presentation 行为由平台
  决定，因此事务式 resize 拒绝后不 acquire/present、不继续 trace，也不允许用 retained publication
  检查新 viewport；
- acquired texture 在 present/drop 前不重配；
- `Suboptimal` 完成本帧后重配，`Outdated` 强制重配，`Lost` 重建；`Timeout`/`Occluded` 跳过并等待恢复信号；
- live resize 合并最新 physical extent，避免为每个原始事件同步等待 GPU idle；`resize_ready` 只等待
  trace/presentation 的 size-dependent owner，不等待只读取 retained publication 的 inspection。Resize
  立即进入 `ViewportChanging`，随后逻辑取消 inspection；旧 ticket completion 不得覆盖该 UI 状态；
- suspend 期间不分配 replacement；恢复 surface 后读取一次最新 inner size。

这些语义来自 [`ApplicationHandler`](https://docs.rs/winit/0.30.13/winit/application/trait.ApplicationHandler.html)、[`Window::inner_size`](https://docs.rs/winit/0.30.13/winit/window/struct.Window.html#method.inner_size) 与 [`SurfaceConfiguration`](https://docs.rs/wgpu/30.0.1/wgpu/type.SurfaceConfiguration.html#structfield.width)。

## HDR 与 native display seam

`gravlume-native-display::DynamicRange` 是平台状态的唯一跨 crate DTO。Renderer 只在以下事实同时成立时选择 HDR：

1. native monitor 给出可靠 HDR 状态与有效 headroom/reference white；
2. surface 广告精确的 `Rgba16Float + ExtendedSrgbLinear` pair。

否则 resolver 选择 SDR 并保留 `SdrReason`。native object 与 `unsafe` 只存在于三个 `native-display/src/platform/*.rs` 模块；renderer 不导入 AppKit、WinRT、Wayland 或 backend-private wgpu 类型。完整平台语义见[平台合同](platform.md)。

## GPU ABI 与错误

GPU DTO 使用 `#[repr(C)]` 标量数组、显式 padding 与 `bytemuck::Pod`。领域类型和 `glam` 不直接上传；因此当前不启用 `glam/bytemuck`。`TraceUniforms` 由连续 `[f32; 4]`/`vec4<f32>` block 组成，blackbody plan 另绑定固定布局的 read-only storage LUT；精确 size、offset、binding、access、format 与 discriminant 以 host/shader source 和 ABI/GPU contracts 为准。

错误按处理者分类：

| 类型                                  | 处理者                                          |
| ------------------------------------- | ----------------------------------------------- |
| `ValidationReport`                    | 修正领域输入                                    |
| `GpuTraceInputError`                  | 修正不可表示或不满足 GPU profile 的 Observation |
| `ResizeError`                         | 处理事务式 rebuild 拒绝或资源失败               |
| `SampleInspectionRequestError`        | 展示无 current publication、越界或固定槽 Busy   |
| `SampleInspectionCompletion`          | 展示完成/取消；失败保留 typed readback source   |
| `PresentSkip`                         | 等待可恢复 surface 状态                         |
| `DeviceEvent`                         | 展示异步 validation/OOM/lost/internal 诊断      |
| `RendererInitError` / `RendererError` | 终止当前无法恢复的 renderer 操作                |

Production 不使用 `unwrap/expect` 恢复错误；typed source chain 只在 UI/日志接口转换为文本。

## 内存与资源预算

Production 不常驻每像素科学 records：

- candidate HDR：`8 B/pixel`；
- UI target：`4 B/pixel`；
- published scene：`8 B/pixel`；
- sky plan 的 shadow scratch：按 extent 精确计算；
- blackbody plan 的固定 spectral LUT；
- sample inspection 固定 request `32 B`、record `96 B`、readback `104 B`，合计最多 `232 B`
  logical buffer；1×1 texel copy 提供 `256 B` row pitch，但最后一行只占实际 `8 B` texel，readback
  不为未使用的下一行分配 padding；
- 四个 `16 B/pixel` scientific planes：只在 test-only diagnostic capture 中创建。通常测试使用
  small extent；版本化 surface/radiance 与 footprint witness 为保持 canonical ray identity，会在
  原始 extent 上临时分配 planes，但只 dispatch 目标 tile，并只回读目标 pixel 的四个 `16 B` record
  lane 与一个对齐后的 `8 B` texel；它们不进入 production resource plan。

显式 scientific export 临时分配 padded readback buffer，完成 map 后即释放；它不属于 steady frame
resources。WebGPU texel copy 对正常有限值保证数值等价，但允许重新编码 zero、subnormal 与
non-finite representation；export 因而让每个 `ScientificTexel` 同时提供 channel bit words 与
semantic kind，不声称所有异常 bit pattern 跨 backend 原样保存。[WebGPU texel copies](https://www.w3.org/TR/webgpu/#texel-copies)

分配 replacement 前，renderer 计算 published scene、installed frame 与 replacement frame 的 extent-scaled core plan，并同时执行 3840×2160 pixel policy 与 256 MiB frame-resource gate。固定 pipeline buffer/LUT、surface images、driver heap 和 alignment 不在该逻辑账本内；这项 gate 不是实测显存峰值。

## 验证与 benchmark

测试保护 observable、生命周期、数值预算、ABI 与 GPU resource semantics；不冻结错误文案、private helper、pass 数量或未经证明的性能阈值。浮点 observable 按具名 absolute/relative budget 比较；值的二进制身份本身是合同时，如 artifact identity、ABI field 或 exact representability，才使用 exact-bit equality。当前覆盖见 [GPU 证据](gpu-renderer.md)。

Cargo `benches/` target 只能使用 library public interface，因此 feature-gated `gravlume_render::benchmark::register` 是一个有真实第二消费者的窄 seam。[Cargo benchmark targets](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#benchmarks) Benchmark 方法与历史数据只在[研究记录](research/gpu-benchmark-methodology.md)维护。

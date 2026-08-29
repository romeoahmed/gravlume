# GPU Renderer 实现与证据

本文是当前 GPU 路径的证据清单：记录仓库已经实现、自动验证和明确未外推的能力。连续模型、误差预算、资源所有权和平台语义分别以[数学物理](physics.md)、[验证合同](validation.md)、[架构合同](architecture.md)和[平台合同](platform.md)为准。

## 当前数据流

```text
validated Observation
  -> checked f64-to-f32 packing
  -> private sealed TracePlan
  -> hidden native-resolution candidate
       sky: full KS + shadow refine
       bolometric surface: full KS + local GeometricSample + immediate g^4/slab transport
       blackbody surface: full KS + gT + spectral LUT/slab transport
  -> timestamp + generation check
  -> atomic texture-view publication
  -> optional outputs before display
       full-frame tagged scientific readback
       one-pixel published texel + structured full-KS inspection
  -> linear scene/UI composition
  -> HDR/scRGB or SDR surface
```

## 已实现

### 输入与 ABI

- `Renderer::new` 只接受 validated `Observation`。Emitter 与 path transport 已由
  `EquatorialSurface::new` 原子验证；slab-only scene 或 unresolved blackbody slab source 在 domain seam
  返回 `ValidationReport`。Host 再按 $M$ 无量纲化并受检转换为 binary32；不可表示字段、未归一化
  observer frequency、f32 packing 后改变 extremality 分类、压扁非空 source interval、把正
  intensity、slab emission 或 transmittance 下溢为零/落入 subnormal、source 未严格落在 numerical
  escape boundary 内或 blackbody radial temperature 超出 LUT，返回 `GpuTraceInputError`。
- Shader 初始 coordinate time 固定为零；GPU 累计相对 coordinate-time duration，因此共同平移 observer/target 时间原点不会改变 observable。Host 在 binary32 转换前判定 initial polar side，并把离散值写入 `observer.x`；该 lane 不再伪装成未使用的 coordinate time。
- `TraceUniforms` 与 dispatch DTO 使用自有 `#[repr(C)]` 标量数组。逐 batch uniform 只传 shader 实际读取的 `tile_origin: vec2<u32>`；workgroup 数量直接是 `dispatch_workgroups` 的命令参数，不在 buffer 中重复。Event thresholds 填充既有 `vec4` lane；四类 event 以固定 `vec4<f32>` fraction 槽位表达，termination 由槽位映射。Blackbody plan 独占 versioned read-only spectral LUT。精确 size、offset、stride 与 binding 由 host/shader source 和 ABI tests 定义；默认画面只包含实际运行需要的 uniform、dispatch 与 plan-specific scratch，不创建 extent-scaled record plane。Production inspection 只增加一个与 extent 无关的固定单槽 record/readback，资源上限以[架构合同](architecture.md#内存与资源预算)为准。
- Termination discriminant 固定为 horizon、escape、singularity guard、step exhaustion、numerical failure、uncertain 与 equatorial surface，并有 checked host/WGSL mapping。可证明的 determinate sample 另携带 initial polar side、radial/equatorial crossing counts 与 signed azimuth winding 的 exact branch key；numerical failure 与 `Uncertain` 都不输出确定 branch。
- Renderer 只匹配一次 `SceneRadiance`，同一 compiled input 同时生成 `TraceUniforms`、private sealed
  `TracePlan` 与 scientific metadata；不存在三个消费者各自解释 Observation 的漂移。WGSL pipeline
  override 固化 surface-event capability。Surface plan 在 shader、bindings 与 target 上不含 shadow
  refinement scratch；caller 不选择 solver 或 accelerator。

### 数值基线

- WGSL core 按 protocol、Kerr–Schild dynamics、event/observable 与 integration state machine 拆分，
  Rust 在 pipeline creation 前按唯一固定顺序组合；它不引入运行期 indirection 或持久化中间状态。
  shader 独立实现 binary32 outgoing Cartesian Kerr–Schild geometry 与 negative-affine classical RK4。
  每 ray 的动态状态是 $(\mathbf x,\mathbf p)$，$E=-p_t$ 为构造常量；relative time 与 spatial derivative
  一起用 `vec4` 做 RK accumulation。
- Geometry 复用 discriminant-root $\Sigma$、$1/r$ 与 $1/(r^2+a^2)$；Hamilton force 只计算 contracted principal-null Jacobian。Carter diagnostic 使用不借 $H=0$、无 axis seam 的 Cartesian 表达式。
- Ordinary accepted step 复用 exact endpoint geometry/RHS 供 event、invariant 与下一步 $k_1$ 使用；它没有把 classical RK4 的 $k_4$ 错当成 FSAL。
- Event 保留 endpoint bracket。每个 armed crossed guard 独立定位后选择 affine traversal 上最早 candidate；tie 以具单位 affine distance 判定，全部 candidates 按稳定 bit order 保留，ambiguity 独立记录并降为 `Uncertain`。Equatorial crossing 始终计入 branch，surface arming 只决定它能否成为 terminal。仅当 Bézier derivative controls 证明 guard cubic 单调且 derivative 有条件时，执行固定六次 safeguarded Newton；否则保留 chord fraction。Radial turning 在同一 cubic dense state 上重算 geometry/RHS 并二分 bracket；若 terminal fraction 落入 bracket，则离散顺序降为 `Uncertain`。Branch key 只提交严格先于 terminal 的 turning/crossing。Travel time、source coordinate、event residual 与 drift 均来自同一个 localized state。
- 四项 recorded invariant 任一超过 GPU profile budget，就把确定终止降为 `Uncertain`。Radicand、denominator、finite 与 singularity guards 都产生 machine-readable failure。

完整公式、符号验证和 binary32 边界见 [KS RK4 约化记录](research/kerr-schild-rk4-reduction.md)。

### 当前求解与轮廓 refinement

- **Full KS：** analytic sky 与 surface plan 都逐像素执行完整 Cartesian Kerr–Schild trace，并由同一
  localized state 提供 terminal、方向、travel time 与 diagnostics。
- **Shadow coverage：** 最后一批完成后，先读取不可变 alpha branch tag 分类 capture/escape 边缘，再以四个真实 rotated-grid subpixel rays 覆盖边界。非边缘像素保持原结果，不用颜色 blur 伪造物理 coverage。

Escape-direction map 与 interval capture 在合同复核中一并撤出 production：前者重建时把 travel time
写成零，后者直接 capture 时不计算 time integral；原测试只比较 terminal/direction，不能证明
[验证合同](validation.md#53-gpu-renderer-agreement)要求的 `1e-3 M` travel-time gate。Numerical fixed-step
Mino candidate 也已因 accepted ray 的 travel-time 反例删除。实验与否决证据只在
[加速研究账本](research/gpu-geodesic-acceleration.md)和 [Mino 决策记录](research/mino-step-selection.md)维护。

### Thin surface transport 与 scientific capture

- Surface event 使用 $z=0$ 双向 crossing；dense-localized radius 必须位于 source inclusive interval。
- `gpu-ks-rk4-v2` 按[验证合同定义的 source-edge refinement policy](validation.md#53-gpu-renderer-agreement)，从同一 initial state 对 base trace 准入的 surface ray 完整重追；只有重追结果进入 radiance 与 inspection。该局部分支不共享 queue、workgroup memory 或中间 trajectory，仍由一个 invocation 独占一条 ray。
- Localized state 独立求 Kerr–Newman prograde circular emitter、oblate chart azimuth 与 $g=\nu_{\rm obs}/\nu_{\rm em}$；非法 orbit/frequency 产生 visible numerical failure。
- `GeometricSample` 只在 invocation-local function value 中存在；event ambiguity 从 candidate bitset 派生，不复制进该值。Production 不创建 G-buffer；neutral plan 将 $I_{\rm em}=I_6(r/6M)^{-3}$ 经 $g^4$ 与解析 slab 后直接写 `RGBA16F` candidate。完整乘积先按 `frexp` 分离 significand/exponent，只有最终 exponent 已证明可表示时才用 `ldexp` 物化；不会因为未输出的 vacuum 中间值溢出而拒绝最终可表示的 radiance。
- Blackbody plan 使用 $T_{\rm obs}=gT_6(r/6M)^{-3/4}$，在固定 $\log_2T$ LUT 中插值 observer-frame `600–700/500–600/400–500 nm` boxcar 的 $\log_2$ fractions。三 channel 是具名 band-integrated intensity，不是 CIE/sRGB，也不覆盖剩余 bolometric power；shader 以 invocation-local `vec3` 一次分解、缩放和有界求和三个 band，将 fraction 的 normal significand/整数 exponent 与 intensity、$g^4$、径向 dilution 和 transmittance 一起累计，直到完整 radiance 的 exponent 已知后才用 vector `ldexp` 物化并交给 `RGBA16F` rounding。Storage LUT 仍使用 16-byte stride 的 `vec4`，没有把 local vector 写法误当作 12-byte host ABI。该路径不把 standalone subnormal fraction 传给 WGSL built-in，也不因未缩放 fraction 的 FTZ 静默丢失可表示 radiance。
- `HomogeneousScalarSlab` 预先保存 $\tau$ 与稳定计算的 integrated emission；GPU 执行 $I_{out}=I_{vac}e^{-\tau}+E$。非零 spectral slab source 必须带自己的 blackbody temperature；neutral bolometric source 不被猜成 spectrum。它是 path-integrated 解析边界，不是 arbitrary volume integration。
- `Renderer::capture_scene_linear` 显式等待 copy/map，读取已发布、tone-map/UI 之前的 scene。
  Surface alpha tag `2.0` 才表示 metadata 所述 radiance；escape tag `1.0` 仍是 analytic
  orientation preview，zero 是 horizon，negative tag 是 trace failure。API 返回
  `ScientificTexel` slice；raw RGBA binary16 words、texel kind 与只对 `SurfaceRadiance` 开放的 RGB
  projection 不能发生索引错配。Metadata 原子携带 source/transport/channel，以及
  [验证合同](validation.md#53-gpu-renderer-agreement)定义的 normal-channel relative budgets、
  `RGBA16F` minimum-normal floor 与 LUT 分项预算；subnormal 的跨 backend 解释由同一合同限定。它只
  导出最终 radiance、texel kind 与整次 capture 的解释 metadata，不导出逐像素 source anchor、branch、
  $g$、travel time 或 event/invariant records；这些逐样本证据不混入整帧 capture，而由下述
  production inspection 提供。
- 该 source 只声称运动学 circular thin surface 与 diluted blackbody；不声称 orbit radial stability、Novikov–Thorne/Page–Thorne disk 或完整 GRRT。

### Production 有界 sample inspection

- `Renderer::request_sample_inspection` 接受 validated `ImageSample`，捕获当前 published
  generation/extent 并返回 ticket。执行方法固定为 `SampleRetrace::METHOD_ID` 所标识的 full
  Kerr–Schild RK4/WGSL binary32；caller 不选择 solver、accelerator 或尚不存在的 quality policy。
- Private 单槽复用 production uniform、surface policy、blackbody LUT 与 plan-specific scene-value
  function，并以 one-element storage binding 复用 test-only corpus 的同一 private kernel；这不增加
  production batch interface。它只允许一个 pending request，并在 resize/suspend 后保持 Busy 直到已提交
  work 和 mapping drain；mapping failure 保留 typed source，只有成功 mapping 才在 mapped view drop 后
  `unmap`。
- Completion 分开携带实际 published `Rgba16Float` texel 与 fresh retrace。严格 decoder 要求 readback
  恰为 `104 B`，并拒绝非单位 Escape direction、未知 bit 与不可能的 terminal/flags/event-candidate
  组合；它只形成
  terminal 合法的 source/scene/branch/channel 组合；`NumericalFailure`/`Uncertain` 没有 branch，step
  exhaustion 只有 branch prefix。三种 plan 的 GPU tests 与 production lifecycle tests 覆盖 ABI、非法
  branch protocol、单次消费、cancel-drain、generation mismatch 和两类像素证据分离。精确 protocol、
  资源和 `8×8 + lane 0` Metal 反例见[采用决策](research/on-demand-sample-inspection.md)与[历史基线](research/bounded-sample-inspection.md)。
- `gravlume-desktop` 是首个 consumer：physical client、active extent 与 publication extent 完全一致时，
  画面点击显示同代 texel、typed retrace 与 diagnostics；新 publication、resize 或 suspend 使旧结果
  失效，旧 completion 不能覆盖 `ViewportChanging`。

### Publication 与 display

- 每个 extent generation 创建隐藏 candidate。Compute batch 不 acquire surface、不运行 egui、不
  present；timestamp resolve/copy 与 mapping 由
  [`map_buffer_on_submit`](https://docs.rs/wgpu/30.0.1/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit)
  绑定到同一 encoder，callback completion 自带 submission generation；匹配当前 generation 后 candidate
  view 才成为 published scene。
- Resize 继续显示上一张完整 scene，并按 aspect-fit 处理比例变化；没有整图 publication copy、低分辨率阶段或 tile 扫描。
- Scene 保持 extended-linear sRGB。egui 先画到透明 gamma-encoded premultiplied target，final pass 解码到 linear-premultiplied 后合成。
- HDR 只在 native state 与精确 `Rgba16Float + ExtendedSrgbLinear` surface pair 同时可靠时启用；否则带原因选择 SDR。解析天空只用于方向、HDR 和 failure visibility，不是物理 source model。

## 自动化证据

`cargo test -p gravlume-render --all-targets --locked` 当前覆盖：

| 层                        | 合同                                                                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| packing/ABI               | termination round-trip、uniform size/offset、production binding/access/format、Naga parse/validation、event candidate/ambiguity capture                      |
| normalization             | 物理等价质量尺度产生相同 dimensionless record；时间原点平移不改变 observable                                                                                 |
| initial ray               | center/corners/jitter 的 CPU/WGSL angular、null 与 frequency budgets                                                                                         |
| solver                    | 默认 Kerr matrix 的 termination、escape direction、event residual、travel time、四项 invariant drift、affine tie 与 surface arming                           |
| surface                   | outer-edge 九点有独立 BL/Mino → CPU regular/strict → fresh binary32 链；其中 canonical v2 `(640,16)` 继续闭合到最终 `RGBA16F` |
| scalar/spectral transport | 四个 v3 fixture 的 vacuum、absorption、constant slab、pure emission；完整 `f32` blackbody bands、normal/subnormal `RGBA16F` representation 与 LUT budgets                     |
| branch/footprint          | 四个 Schwarzschild/Kerr/Kerr–Newman profile 的分层 surface terminal/branch-key exact gate；五条真实 quarter-pixel ray 的 parity 与 CPU/GPU Jacobian max-norm |
| scientific export         | bound texel words/kind、physical RGB gating、row unpadding 与解释 metadata                                                                                   |
| sample inspection         | fixed ABI/resource cap、strict decode、published texel/retrace 分离、Busy/cancel-drain/generation mismatch 与三种 plan 的真实 GPU record                     |
| dispatch                  | odd extent、workgroup boundary、multi-batch 与 single-dispatch equality、65 个乱序 sample 的 runtime-array partial workgroup、device workgroup-dimension cap |
| solver/refinement         | full-KS terminal/direction/travel-time gate；shadow refinement 只改真实边缘 coverage                                                                          |
| coverage                  | branch-edge detection、四样本 fractional coverage、reset/order 与非边缘稳定性                                                                                |
| resources                 | 4K pixel boundary、cold/active/completed transactional plan 与 synthetic oversized plan 的分配前 typed rejection                                             |
| lifecycle/display         | publication generation、resize、HDR/SDR resolver、linear composition 与 native smoke                                                                         |

GPU tests 需要可用 Metal 或 Vulkan adapter。CPU 与 GPU 使用不同精度、状态和积分器；agreement 只证明受测样本满足预算，不构成独立物理证明。

## 适用域与限制

- Fresh binary32 continuous-observable comparison 已覆盖 canonical exterior Kerr surface sample 及同一
  positive-spin observation 上跨 outer source edge 的固定 stencil；扩展 Schwarzschild、负自旋与
  Kerr–Newman matrix 目前只准入 terminal/branch exactness。负自旋 `(62,7)` 虽已有独立 120/180 位
  source/phase/transfer 研究证书，但尚未被 Reference regular/strict 与 fresh GPU structured comparison
  消费。其他非 canonical source/time phase 尚未满足同一预算，详见
  [负自旋研究记录](research/negative-spin-continuous-witness.md)与
  [reconstruction 研究记录](research/radiative-transfer-and-source-reconstruction.md#57-surface-full-ks-的-binary32-phase-边界)。
- Near-critical/high-winding 的独立 BL/Mino pair 已有 research-only 证书，但 GPU/reference
  structured ladder 尚未闭合；near-axis 与 near-extreme 也仍缺对应 GPU/reference evidence。
- RK4 v2 只对具名 source-edge band 条件重追；当前没有 near-critical、axis、near-extreme 或一般
  `Uncertain` ray 的第二种 science-quality GPU policy。
- CPU surface footprint 与 test-only GPU ordinary-region 证据已经存在，但 production 不持久化 source/Jacobian map，也没有 branch-aware reconstruction、multi-image/near-critical ladder 或 texture filtering consumer。
- Scalar slab 只覆盖 homogeneous path-integrated analytic operator；没有空间变化 volume coefficient、ordered checkpoints、scattering、slow-light 或 polarization。
- Scientific capture 是最终 radiance 的整帧内存 readback API，不是稳定的磁盘 container；production sample inspection 是 live process 内的一次性证据，也不是持久 artifact interface。异常 texture representation 只按 WebGPU 数值等价合同处理，不承诺 NaN/subnormal bit preservation。
- Production inspection 只覆盖一个 sample 与 presentation `gpu-ks-rk4-v2` policy；test-only ordered
  batch 不构成 bounded-region production interface。固定 `(640,12..20)` source-edge 九点均有独立
  BL/Mino terminal/branch/continuous witness，并通过 CPU regular/strict 与同一 ordered fresh binary32
  comparison；其中与 canonical v2 重合的 `(640,16)` 继续闭合到最终 `RGBA16F`。整个 stencil 仍没有
  统一的最终 texture gate，其余 strata 同样未闭合。精确边界见
  [连续字段 corpus 记录](research/continuous-field-corpus.md)，因此路线图的质量基线仍开放。
- Shadow coverage 只处理 capture/escape silhouette，不处理 Escape/escape caustic、source winding 或通用 texture footprint。
- Windows 与 Wayland 尚无具名目标设备的 runtime HDR/lifecycle 发布矩阵。
- 项目没有 60 FPS 声明，也没有把逻辑资源账本称为 driver 显存峰值。

## 性能证据

Batching 只控制 watchdog 与事件循环响应；当前减少工作的 production 优化是 outgoing chart、endpoint
reuse 与 KS 代数约化，shadow refinement 则只在最终批次为真实边缘增加必要样本。所有历史 A/B、累计
结果、adapter、extent 与统计方法集中在[GPU geodesic 加速账本](research/gpu-geodesic-acceleration.md)，
永久 benchmark 的测量边界见[基准方法](research/gpu-benchmark-methodology.md)。本页不维护第二份数字。

## 主要来源

- [WGSL specification](https://www.w3.org/TR/WGSL/)：floating-point、layout、built-ins 与 execution semantics；
- [Bozzola, Chan & Paschalidis 2023](https://doi.org/10.1103/PhysRevD.108.084004)：backward ray 的 horizon-penetrating chart 选择；
- [wgpu 30 documentation](https://docs.rs/wgpu/30.0.1/wgpu/)：limits、bindings、dispatch、timestamp、surface 与 HDR color spaces；
- [Rust numeric casts](https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast)、[type layout](https://doc.rust-lang.org/stable/reference/type-layout.html)与 [`bytemuck::Pod`](https://docs.rs/bytemuck/1.25.2/bytemuck/trait.Pod.html)：host narrowing 与 byte contract。

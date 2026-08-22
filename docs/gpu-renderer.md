# GPU Renderer 实现与证据

本文是当前 GPU 路径的证据清单：记录仓库已经实现、自动验证和明确未外推的能力。连续模型、误差预算、资源所有权和平台语义分别以[数学物理](physics.md)、[验证合同](validation.md)、[架构合同](architecture.md)和[平台合同](platform.md)为准。

## 当前数据流

```text
validated Observation
  -> checked f64-to-f32 packing
  -> private sealed TracePlan
  -> hidden native-resolution candidate
       sky: escape-map + reconstruction/full-KS fallback + shadow refine
       bolometric surface: full KS + local GeometricSample + immediate g^4/slab transport
       blackbody surface: full KS + gT + spectral LUT/slab transport
  -> timestamp + generation check
  -> atomic texture-view publication
  -> optional tagged scientific readback before display
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
- `TraceUniforms` 与 dispatch DTO 使用自有 `#[repr(C)]` 标量数组。Event thresholds 填充既有 `vec4` lane，当前 uniform 为 11 个连续 16-byte block、共 176 byte；四类 event 以固定 `vec4<f32>` fraction 槽位表达，termination 由槽位映射。Blackbody plan 独占 binding 8 的 4097-entry read-only `array<vec4<f32>>`；WGSL array stride 为 16 byte。默认画面只包含实际运行需要的 uniform、dispatch 与 plan-specific scratch；extent-scaled record planes 与定长单样本 record 都只在 test capture 创建。
- Termination discriminant 固定为 horizon、escape、singularity guard、step exhaustion、numerical failure、uncertain 与 equatorial surface，并有 checked host/WGSL mapping。可证明的 determinate sample 另携带 initial polar side、radial/equatorial crossing counts 与 signed azimuth winding 的 exact branch key；numerical failure 与 `Uncertain` 都不输出确定 branch。
- Renderer 只匹配一次 `SceneRadiance`，同一 compiled input 同时生成 `TraceUniforms`、private sealed
  `TracePlan` 与 scientific metadata；不存在三个消费者各自解释 Observation 的漂移。WGSL pipeline
  override 固化 surface-event capability。Surface plan 在 shader、bindings、timing 与 target 上都不含
  escape map 或 shadow refinement scratch；caller 不选择 accelerator。

### 数值基线

- WGSL core 按 protocol、Kerr–Schild dynamics、event/observable 与 integration state machine 拆分，
  Rust 在 pipeline creation 前按唯一固定顺序组合；它不引入运行期 indirection 或持久化中间状态。
  shader 独立实现 binary32 outgoing Cartesian Kerr–Schild geometry 与 negative-affine classical RK4。
  每 ray 的动态状态是 $(\mathbf x,\mathbf p)$，$E=-p_t$ 为构造常量；relative time 与 spatial derivative
  一起用 `vec4` 做 RK accumulation。
- Geometry 复用 discriminant-root $\Sigma$、$1/r$ 与 $1/(r^2+a^2)$；Hamilton force 只计算 contracted principal-null Jacobian。Carter diagnostic 使用不借 $H=0$、无 axis seam 的 Cartesian 表达式。
- Ordinary accepted step 复用 exact endpoint geometry/RHS 供 event、invariant 与下一步 $k_1$ 使用；它没有把 classical RK4 的 $k_4$ 错当成 FSAL。
- Event 保留 endpoint bracket。每个 armed crossed guard 独立定位后选择 affine traversal 上最早 candidate；tie 以具单位 affine distance 判定，全部 candidates 按稳定 bit order 保留，ambiguity 独立记录并降为 `Uncertain`。Surface 从 profile arming band 外进入后才允许 crossing。仅当 Bézier derivative controls 证明 guard cubic 单调且 derivative 有条件时，执行固定六次 safeguarded Newton；否则保留 chord fraction。Radial turning 在同一 cubic dense state 上重算 geometry/RHS 并二分 bracket；若 terminal fraction 落入 bracket，则离散顺序降为 `Uncertain`。Branch key 只提交严格先于 terminal 的 turning/crossing。Travel time、source coordinate、event residual 与 drift 均来自同一个 localized state。
- 四项 recorded invariant 任一超过 GPU profile budget，就把确定终止降为 `Uncertain`。Radicand、denominator、finite 与 singularity guards 都产生 machine-readable failure。

完整公式、符号验证和 binary32 边界见 [KS RK4 约化记录](research/kerr-schild-rk4-reduction.md)。

### 保守加速

- **Escape-direction map：** 在共享 4-pixel grid 上追踪，按 `8×8` tile 读取 `3×3` stencil；branch/condition 通过时重建归一化 Escape direction，否则逐像素执行完整 KS。
- **Interval capture：** 严格支持域内，用向外扩张的 Bernstein interval 证明 radial potential 无 turning point后直接分类 Horizon；near-axis、near-extreme、超出参数 envelope 或任何不确定性都 fallback。
- **Shadow coverage：** 最后一批完成后，先读取不可变 alpha branch tag 分类 capture/escape 边缘，再以四个真实 rotated-grid subpixel rays 覆盖边界。非边缘像素保持原结果，不用颜色 blur 伪造物理 coverage。

Numerical fixed-step Mino candidate 已因 accepted ray 的 travel-time 反例从 production 删除。性能与否决证据只在[加速研究账本](research/gpu-geodesic-acceleration.md)和 [Mino 决策记录](research/mino-step-selection.md)维护。

### Thin surface transport 与 scientific capture

- Surface event 使用 $z=0$ 双向 crossing；dense-localized radius 必须位于 source inclusive interval。
- Localized state 独立求 Kerr–Newman prograde circular emitter、oblate chart azimuth 与 $g=\nu_{\rm obs}/\nu_{\rm em}$；非法 orbit/frequency 产生 visible numerical failure。
- `GeometricSample` 只在 invocation-local function value 中存在；event ambiguity 从 candidate bitset 派生，不复制进该值。Production 不创建 G-buffer；neutral plan 将 $I_{\rm em}=I_6(r/6M)^{-3}$ 经 $g^4$ 与解析 slab 后直接写 `RGBA16F` candidate。完整乘积先按 `frexp` 分离 significand/exponent，只有最终 exponent 已证明可表示时才用 `ldexp` 物化；不会因为未输出的 vacuum 中间值溢出而拒绝最终可表示的 radiance。
- Blackbody plan 使用 $T_{\rm obs}=gT_6(r/6M)^{-3/4}$，在固定 $\log_2T$ LUT 中插值 observer-frame `600–700/500–600/400–500 nm` boxcar 的 $\log_2$ fractions。三 channel 是具名 band-integrated intensity，不是 CIE/sRGB，也不覆盖剩余 bolometric power；shader 以 invocation-local `vec3` 一次分解、缩放和有界求和三个 band，将 fraction 的 normal significand/整数 exponent 与 intensity、$g^4$、径向 dilution 和 transmittance 一起累计，直到完整 radiance 的 exponent 已知后才用 vector `ldexp` 物化并交给 `RGBA16F` rounding。Storage LUT 仍使用 16-byte stride 的 `vec4`，没有把 local vector 写法误当作 12-byte host ABI。该路径不把 standalone subnormal fraction 传给 WGSL built-in，也不因未缩放 fraction 的 FTZ 静默丢失可表示 radiance。
- `HomogeneousScalarSlab` 预先保存 $\tau$ 与稳定计算的 integrated emission；GPU 执行 $I_{out}=I_{vac}e^{-\tau}+E$。非零 spectral slab source 必须带自己的 blackbody temperature；neutral bolometric source 不被猜成 spectrum。它是 path-integrated 解析边界，不是 arbitrary volume integration。
- `Renderer::capture_scene_linear` 显式等待 copy/map，读取已发布、tone-map/UI 之前的 scene。
  Surface alpha tag `2.0` 才表示 metadata 所述 radiance；escape tag `1.0` 仍是 analytic
  orientation preview，zero 是 horizon，negative tag 是 trace failure。API 返回
  `ScientificTexel` slice；raw RGBA binary16 words、texel kind 与只对 `SurfaceRadiance` 开放的 RGB
  projection 不能发生索引错配。Metadata 原子携带 source/transport/channel，以及 bolometric
  `2e-3`、final spectral `4e-3` 与 LUT 分项误差预算。它只导出最终 radiance、texel kind 与整次
  capture 的解释 metadata，不导出逐像素 source anchor、branch、$g$、travel time 或 event/invariant
  records；这些逐样本证据当前只由下述 test-only capture 提供，不混入整帧 capture，也不构成
  production interface。
- 该 source 只声称运动学 circular thin surface 与 diluted blackbody；不声称 orbit radial stability、Novikov–Thorne/Page–Thorne disk 或完整 GRRT。

### Test-only 有界 sample evidence

- `cfg(test)` inspection module 的唯一 Interface 是对 validated `ImageSample` 执行一次同步、plan-matched full-KS retrace 并返回 typed record。它复用 `TracePipeline` 的 uniform、surface policy、blackbody LUT 与 channel model；production `Renderer` 不创建这套 pipeline 或 buffer，也没有公开 inspection interface。Generation、request identity、queue、取消和 supersession 都留给未来真实 consumer 定义，不在测试 helper 中预演。
- Request、storage result 与 staging readback分别为 32、96、96 byte，合计 224 logical bytes，与 viewport extent 无关。Host/WGSL 分别只使用两个和六个 16-byte `vec4` lane；同一编译单元的短生命周期 record 不携带 version、producer/domain 或 host echo compatibility fields。确定终止的两个 branch count、signed winding 与 initial polar side 各占完整 `u32` lane；numerical failure 的全零 lane 是 unavailable sentinel，`Uncertain` 的 provisional counters 也不构成 exact branch，host 对两者都返回 `None`。
- Test helper 提交一个 workgroup。Shader 保持与正确性基线相同的 `8×8` workgroup specialization，但只有 `local_invocation_index == 0` 进入 sequential RK/event solver，其余 lane 在建立 ray state 前返回，因此每个请求只重算一条 ray。`@workgroup_size(1)` 在当前 Metal 实测中丢失 travel/drift/branch 返回字段，已被反证；过程与恢复条件见[有界单样本审计](research/bounded-sample-inspection.md)。这不是 SIMD 性能声明。
- Result 在 tone map、display encoding 与 UI 之前调用同一 plan-specific scene-value function，并保存完整 f32 RGBA tag；只有 Surface terminal 的 `SurfaceRadiance` 按 channel model 解释为物理输出，Escape RGB 明确是 orientation preview。96-byte record copy 到 `MAP_READ | COPY_DST` staging 后，复用测试设备的 `map_async`、submission-bound `Device::poll(Wait)`、mapped-range read 与 `unmap`。该证据验证候选 layout 与 observable，不声明已发布 `RGBA16F` texel 的 bitwise history，也不冻结 future consumer interface。

### Publication 与 display

- 每个 extent generation 创建隐藏 candidate。Compute batch 不 acquire surface、不运行 egui、不
  present；timestamp resolve/copy 与 mapping 由
  [`map_buffer_on_submit`](https://docs.rs/wgpu/30.0.0/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit)
  绑定到同一 encoder，callback completion 自带 submission generation；匹配当前 generation 后 candidate
  view 才成为 published scene。
- Resize 继续显示上一张完整 scene，并按 aspect-fit 处理比例变化；没有整图 publication copy、低分辨率阶段或 tile 扫描。
- Scene 保持 extended-linear sRGB。egui 先画到透明 gamma-encoded premultiplied target，final pass 解码到 linear-premultiplied 后合成。
- HDR 只在 native state 与精确 `Rgba16Float + ExtendedSrgbLinear` surface pair 同时可靠时启用；否则带原因选择 SDR。解析天空只用于方向、HDR 和 failure visibility，不是物理 source model。

## 自动化证据

`cargo test -p gravlume-render --all-targets --locked` 当前覆盖：

| 层                | 合同                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------ |
| packing/ABI       | termination round-trip、uniform size/offset、production binding/access/format、Naga parse/validation、event candidate/ambiguity capture |
| normalization     | 物理等价质量尺度产生相同 dimensionless record；时间原点平移不改变 observable                                 |
| initial ray       | center/corners/jitter 的 CPU/WGSL angular、null 与 frequency budgets                                         |
| solver            | 默认 Kerr matrix 的 termination、escape direction、event residual、travel time、四项 invariant drift、affine tie 与 surface arming |
| surface           | canonical v2 fixture 的 event position、oblate anchor、Frequency Ratio、travel time 与 `RGBA16F` radiance    |
| scalar/spectral transport | 四个 v3 fixture 的 vacuum、absorption、constant slab、pure emission、blackbody bands 与 LUT budgets  |
| branch/footprint  | 四个 Schwarzschild/Kerr/Kerr–Newman profile 的分层 surface terminal/branch-key exact gate；五条真实 quarter-pixel ray 的 parity 与 CPU/GPU Jacobian max-norm |
| scientific export | bound texel words/kind、physical RGB gating、row unpadding 与解释 metadata                                |
| sample inspection prototype | 32/96-byte aligned layout、确定终止的 `u32` branch、indeterminate unavailable；三种 plan 的真实 GPU record |
| dispatch          | odd extent、workgroup boundary、multi-batch 与 single-dispatch equality、device workgroup-dimension cap      |
| acceleration      | escape-map 与 full baseline branch/direction gate；Kerr/KN interval capture 的支持域与 conservative fallback |
| coverage          | branch-edge detection、四样本 fractional coverage、reset/order 与非边缘稳定性                                |
| resources         | 4K pixel boundary、cold/completed/worst transactional plan 与分配前 typed rejection                          |
| lifecycle/display | publication generation、resize、HDR/SDR resolver、linear composition 与 native smoke                         |

GPU tests 需要可用 Metal 或 Vulkan adapter。CPU 与 GPU 使用不同精度、状态和积分器；agreement 只证明受测样本满足预算，不构成独立物理证明。

## 适用域与限制

- Regular continuous-observable matrix 仍以 canonical exterior Kerr surface sample 为主；扩展
  Schwarzschild、正/负自旋与 Kerr–Newman matrix 目前只准入 terminal/branch exactness。默认
  presentation RK4 的非 canonical source/time phase 尚未满足同一预算，详见
  [reconstruction 研究记录](research/radiative-transfer-and-source-reconstruction.md#57-surface-full-ks-的-binary32-phase-边界)。
  KN accelerator equality 也只覆盖严格亚极端的具名样本，不是完整 charge sweep。
- Near-critical、高绕转、near-axis 与 near-extreme 的 GPU/reference ladder 尚未闭合。
- RK4 使用固定 radius-scaled step policy；当前没有 `Uncertain` ray 的第二遍更高精度追迹。
- CPU surface footprint 与 test-only GPU ordinary-region 证据已经存在，但 production 不持久化 source/Jacobian map，也没有 branch-aware reconstruction、multi-image/near-critical ladder 或 texture filtering consumer。
- Scalar slab 只覆盖 homogeneous path-integrated analytic operator；没有空间变化 volume coefficient、ordered checkpoints、scattering、slow-light 或 polarization。
- Scientific capture 是最终 radiance 的整帧内存 readback API，不是稳定的磁盘 container；逐样本 path evidence 当前只有 test-only prototype，尚无 production artifact interface。异常 texture representation 只按 WebGPU 数值等价合同处理，不承诺 NaN/subnormal bit preservation。
- Test-only inspection prototype 只覆盖一个 sample 与 presentation `gpu-ks-rk4-v1` policy；没有真实 production consumer、bounded-region batch、inspection UI、trajectory/checkpoint、event bracket width、Jacobi/parity、独立 high-precision certificate 或第二 science-quality GPU policy。Canonical surface/analytic/blackbody 已接纳，但 source edge、critical curve 两侧、higher-order winding 与正负 spin 的连续字段 corpus 尚未闭合。
- Shadow coverage 只处理 capture/escape silhouette，不处理 Escape/escape caustic、source winding 或通用 texture footprint。
- Windows 与 Wayland 尚无具名目标设备的 runtime HDR/lifecycle 发布矩阵。
- 项目没有 60 FPS 声明，也没有把逻辑资源账本称为 driver 显存峰值。

## 性能证据

Batching 只控制 watchdog 与事件循环响应；真正减少工作的是 outgoing chart、endpoint reuse、KS 代数约化、escape-direction map 与 interval capture。所有历史 A/B、累计结果、adapter、extent 与统计方法集中在[GPU geodesic 加速账本](research/gpu-geodesic-acceleration.md)，永久 benchmark 的测量边界见[基准方法](research/gpu-benchmark-methodology.md)。本页不维护第二份数字。

## 主要来源

- [WGSL specification](https://www.w3.org/TR/WGSL/)：floating-point、layout、built-ins 与 execution semantics；
- [Bozzola, Chan & Paschalidis 2023](https://doi.org/10.1103/PhysRevD.108.084004)：backward ray 的 horizon-penetrating chart 选择；
- [wgpu 30 documentation](https://docs.rs/wgpu/30.0.0/wgpu/)：limits、bindings、dispatch、timestamp、surface 与 HDR color spaces；
- [Rust numeric casts](https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast)、[type layout](https://doc.rust-lang.org/stable/reference/type-layout.html)与 [`bytemuck::Pod`](https://docs.rs/bytemuck/1.25.2/bytemuck/trait.Pod.html)：host narrowing 与 byte contract。

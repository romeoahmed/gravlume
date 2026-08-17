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

- `Renderer::new` 只接受 validated `Observation`。Host 按 $M$ 无量纲化，并受检转换为 binary32；不可表示字段、未归一化 observer frequency、f32 packing 后改变 extremality 分类、压扁非空 source interval、把正 intensity、slab emission 或 transmittance 下溢为零/落入 subnormal、source 未严格落在 numerical escape boundary 内、blackbody radial temperature 超出 LUT，或 slab/source spectrum 不闭合，都会返回 `GpuTraceInputError`。
- Shader 初始 coordinate time 固定为零；GPU 累计相对 coordinate-time duration，因此共同平移 observer/target 时间原点不会改变 observable。
- `TraceUniforms` 与 dispatch DTO 使用自有 `#[repr(C)]` 标量数组。Event thresholds 填充既有 `vec4` lane，当前 uniform 为 11 个连续 16-byte block、共 176 byte；四类 event 以固定 `vec4<f32>` fraction 槽位表达，termination 由槽位映射。Blackbody plan 独占 binding 8 的 4097-entry read-only `array<vec4<f32>>`；WGSL array stride 为 16 byte。Production ABI 只包含实际运行需要的 uniform、dispatch 与 plan-specific scratch；四个 diagnostic record planes 仅由 test capture 创建。
- Termination discriminant 固定为 horizon、escape、singularity guard、step exhaustion、numerical failure、uncertain 与 equatorial surface，并有 checked host/WGSL mapping。Surface sample 另携带 initial polar side、radial/equatorial crossing counts 与 signed azimuth winding 的 exact branch key。
- Renderer 从 Physical Scene 解析 private sealed `TracePlan`，并用 WGSL pipeline override 固化 surface-event capability。Surface plan 在 shader、bindings、timing 与 target 上都不含 escape map 或 shadow refinement scratch；caller 不选择 accelerator。

### 数值基线

- WGSL 独立实现 binary32 outgoing Cartesian Kerr–Schild geometry 与 negative-affine classical RK4。每 ray 的动态状态是 $(\mathbf x,\mathbf p)$，$E=-p_t$ 为构造常量；relative time 与 spatial derivative 一起用 `vec4` 做 RK accumulation。
- Geometry 复用 discriminant-root $\Sigma$、$1/r$ 与 $1/(r^2+a^2)$；Hamilton force 只计算 contracted principal-null Jacobian。Carter diagnostic 使用不借 $H=0$、无 axis seam 的 Cartesian 表达式。
- Ordinary accepted step 复用 exact endpoint geometry/RHS 供 event、invariant 与下一步 $k_1$ 使用；它没有把 classical RK4 的 $k_4$ 错当成 FSAL。
- Event 保留 endpoint bracket。每个 armed crossed guard 独立定位后选择 affine traversal 上最早 candidate；tie 以具单位 affine distance 判定，全部 candidates 按稳定 bit order 保留，ambiguity 独立记录并降为 `Uncertain`。Surface 从 profile arming band 外进入后才允许 crossing。仅当 Bézier derivative controls 证明 guard cubic 单调且 derivative 有条件时，执行固定六次 safeguarded Newton；否则保留 chord fraction。Travel time、source coordinate、event residual 与 drift 均来自同一个 localized state。
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
- `GeometricSample` 只在 invocation-local function value 中存在；event ambiguity 从 candidate bitset 派生，不复制进该值。Production 不创建 G-buffer；neutral plan 将 $I_{\rm em}=I_6(r/6M)^{-3}$ 经 $g^4$ 与解析 slab 后直接写 `RGBA16F` candidate。
- Blackbody plan 使用 $T_{\rm obs}=gT_6(r/6M)^{-3/4}$，在固定 $\log_2T$ LUT 中插值 observer-frame `600–700/500–600/400–500 nm` boxcar fractions。三 channel 是具名 band-integrated intensity，不是 CIE/sRGB，也不覆盖剩余 bolometric power。
- `HomogeneousScalarSlab` 预先保存 $\tau$ 与稳定计算的 integrated emission；GPU 执行 $I_{out}=I_{vac}e^{-\tau}+E$。非零 spectral slab source 必须带自己的 blackbody temperature；neutral bolometric source 不被猜成 spectrum。它是 path-integrated 解析边界，不是 arbitrary volume integration。
- `Renderer::capture_scene_linear` 显式等待 copy/map，读取已发布、tone-map/UI 之前的 scene。
  Surface alpha tag `2.0` 才表示 metadata 所述 radiance；escape tag `1.0` 仍是 analytic
  orientation preview，zero 是 horizon，negative tag 是 trace failure。API 返回 RGBA binary16
  words、texel kind、source/transport/channel，以及 bolometric `2e-3`、final spectral `4e-3` 与
  LUT 分项误差 metadata，避免调用者把 preview RGB 当成光谱或把 LUT 预算误当成最终预算。
- 该 source 只声称运动学 circular thin surface 与 diluted blackbody；不声称 orbit radial stability、Novikov–Thorne/Page–Thorne disk 或完整 GRRT。

### Publication 与 display

- 每个 extent generation 创建隐藏 candidate。Compute batch 不 acquire surface、不运行 egui、不 present；最后 timestamp/readback 完成且 generation 匹配后，candidate view 才成为 published scene。
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
| scientific export | raw normal binary16 values、row unpadding、surface/sky/horizon/failure texel kind 与解释 metadata             |
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
- Scientific capture 是内存 readback API，不是稳定的磁盘 container；它给出声明预算与 texel kind，不是每像素独立误差 certificate。异常 texture representation 只按 WebGPU 数值等价合同处理，不承诺 NaN/subnormal bit preservation。
- Shadow coverage 只处理 capture/escape silhouette，不处理 Escape/escape caustic、source winding 或通用 texture footprint。
- Windows 与 Wayland 尚无具名目标设备的 runtime HDR/lifecycle 发布矩阵。
- 项目没有 60 FPS 声明，也没有把逻辑资源账本称为 driver 显存峰值。

## 性能证据

Batching 只控制 watchdog 与事件循环响应；真正减少工作的是 outgoing chart、endpoint reuse、KS 代数约化、escape-direction map 与 interval capture。所有历史 A/B、累计结果、adapter、extent 与统计方法集中在[GPU geodesic 加速账本](research/gpu-geodesic-acceleration.md)，永久 benchmark 的测量边界见[基准方法](research/gpu-benchmark-methodology.md)。本页不维护第二份数字。

## 主要来源

- [WGSL specification](https://www.w3.org/TR/WGSL/)：floating-point、layout、built-ins 与 execution semantics；
- [Bozzola, Chan & Paschalidis 2023](https://doi.org/10.1103/PhysRevD.108.084004)：backward ray 的 horizon-penetrating chart 选择；
- [wgpu 30 documentation](https://docs.rs/wgpu/30.0.0/wgpu/)：limits、bindings、dispatch、timestamp、surface 与 HDR color spaces；
- [Rust type layout](https://doc.rust-lang.org/stable/reference/type-layout.html)与 [`bytemuck::Pod`](https://docs.rs/bytemuck/1.25.2/bytemuck/trait.Pod.html)：host byte contract。

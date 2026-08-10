# 渲染算法、适用边界与研究门槛

本文把黑洞图像拆成 geometry、transport、sampling/reconstruction 和 display 四类问题。外部论文或作者实现只支持其明确的 spacetime、source、observer、precision 和 hardware domain；所有方法在进入 Gravlume 前都要通过本项目 fixture 与帧时测量。

## 1. 开发结论

1. **生产顺序是 Schwarzschild → Kerr → Kerr 专用加速；Kerr–Newman 是 research geometry。** Kerr 是解析轨迹、transfer map 和观测解释的主干，显著净电荷没有默认产品优先级所需的天体证据。
2. **GPU Cartesian Kerr–Schild `f32` tracer 是数值基线。** CPU `f64` reference 与之独立；analytic/LUT 不能成为唯一路径。
3. **先保存几何语义，再谈重建。** termination、branch/parity、source anchor、travel time、frequency ratio 和 source-space footprint 是 adaptive/filter/temporal 的输入。
4. **表面 transfer map 可高价值复用，体传输必须谨慎。** scalar volume 和 Stokes transport 需要沿有序路径重积；不能把一张源坐标纹理冒充所有介质。
5. **fast-light/slow-light 是 source 数据语义。** slow-light 需要 emission time、snapshot revision、插值与 out-of-domain policy。
6. **active-ray compaction 和 temporal 必须用实测证明价值。** ODE 分支多不自动证明 wavefront/indirect queue 或普通 TAA 在 Vulkan/Metal 净赚。
7. **WGSL `f32` 的误差合同按 observable 分层。** finite 与 invariant drift 不能单独保证 near-critical ray 的 branch/angle 正确。

## 2. 中间结果合同

### 2.1 Geometry sample

geometry 阶段输出结构化字段，而不是直接输出最终颜色：

```text
GeometricSample
  - termination + numerical flags
  - image branch + parity/winding class
  - source kind + source anchor
  - observer/source frequency ratio
  - affine/coordinate travel time where defined
  - source-space Jacobian or conservative footprint
  - steps + event/invariant diagnostics
  - geometry generation + solver/domain tag
```

surface disk 可以把 source anchor 表为带 winding/side 的局部坐标；escape 表为有 seam-aware chart 的天空方向；volume 不能只保存单点，它至少需要有序 path checkpoints 或可重建的路径状态。

`branch` 不是 UI 标签。只要 source map 在临界曲线、caustic、遮挡或 winding 处不连续，相邻像素就可能属于不同 branch；spatial/temporal filter 必须先比较 branch/status，再比较连续坐标。

### 2.2 Path observation 与 display 分离

```text
Trace -> GeometricSample
      -> surface | scalar-volume | research-Stokes transport
      -> invariant radiance + optical depth + transport flags
      -> reconstruct/history
      -> scene-linear HDR
      -> display transform/gamut encoding
      -> egui/diagnostic overlays
```

geometry accelerator 只替换 Trace，不能绕过 sample contract、diagnostics 或 validation。Appearance effect 从 scene-linear result 开始，不回写 geodesic 或 emission。

## 3. Geometry 路线

### 3.1 数值基线

首选 Cartesian Kerr–Schild Hamilton/一阶系统：

- 任意外部 observer，避开 Boyer–Lindquist horizon/pole coordinate singularity；
- 可扩到 Kerr–Newman；
- metric/derivative 每步较贵，但状态与 event 语义通用；
- backward trace 的 ingoing/outgoing chart 必须与传播/数值方向验证。

GPU 候选先比较固定 RK4 + 几何/量化 step 与少量有界 embedded tier。每 ray 的完全自适应 DP5(4) 容易产生 accept/reject 发散；它优先属于 CPU reference。选择基于 GPU milliseconds 对 observable error 曲线，不基于“阶数更高”的名字。

### 3.2 Schwarzschild 专用路径

球对称允许低维 deflection/beam LUT。每个 lookup 仍携带：

- input domain 与 normalization；
- escape/capture branch；
- interpolation error bound；
- source footprint；
- producer revision/fingerprint；
- out-of-domain 时使用通用数值路径。

LUT 不覆盖 dynamic volume 或 arbitrary spacetime；它是 `TracePlan` 的 sealed variant。低维 deflection/beam LUT 与星空 footprint 可对照 [Bruneton 2020](https://arxiv.org/abs/2010.08735)，但适用域与误差必须由本项目 fixture 重新建立。

### 3.3 Kerr 专用路径

Carter separability、Carlson elliptic forms、analytic geodesics，以及 [AART](https://doi.org/10.1103/PhysRevD.107.043030)/[Krang](https://doi.org/10.21105/joss.07273) 一类 transfer functions 适合 exterior Kerr。优先用途：

1. CPU oracle 与 fixture；
2. 固定 observer/薄赤道源 transfer map；
3. photon-ring/lensing-band 专用高分辨率图；
4. 经 bake-off 后的 interactive accelerator。

输入域必须写清 observer near/far、source surface、allowed roots/turning points、horizon policy 与 coordinate time。analytic code 仍要测试 branch classification、root degeneracy 和 `f32` condition；“闭式”不等于数值稳定。

### 3.4 Kerr–Newman

Kerr–Newman exterior 有 Mino-time 显式解，可用于 CPU semi-analytic reference 或固定参数 LUT。[Wang–Lee–Lin 2022](https://arxiv.org/abs/2208.11906) 但 WGSL 内同时实现完整 elliptic functions、root branches、任意近场 observer 与 horizon crossing 风险很高。

产品策略：domain/reference 保留 $q_e$ 和 sub/extreme/superextreme 分类；interactive 通用路径用 Kerr–Schild；analytic KN 只有独立 fixture、误差与产品场景证明价值后再实现。

## 4. Exterior Mino-time candidate

解析/半解析 Kerr 路径是 accelerator，不是跨后端数值基线。为比较一个不依赖完整椭圆函数的 exterior candidate，取 $M=E=1$、$b=L_z/E$、$\eta=\mathcal Q/E^2$、$\mu=\cos\theta$、electric charge $e=q_e/M$：

\[
\Delta=r^2-2r+a^2+e^2,
\quad P=r^2+a^2-ab,
\quad A=(b-a)^2+\eta,
\]

\[
R(r)=P^2-\Delta A,
\quad
U(\mu)=\eta+(a^2-\eta-b^2)\mu^2-a^2\mu^4,
\]

\[
r'=v_r,
\quad v_r'=2rP-(r-1)A,
\]

\[
\mu'=v_\mu,
\quad v_\mu'=(a^2-\eta-b^2)\mu-2a^2\mu^3.
\]

二阶势状态可自然穿过 turning point，并监测 $v_r^2-R$、$v_\mu^2-U$。但 $\Delta$ 分母仍使它只适合 exterior。

为改善远场条件数，令 $u=1/r,w=du/d\lambda,c=a^2-ab,d=a^2+e^2$：

\[
V(u)=w^2=(1+cu^2)^2-(u^2-2u^3+du^4)A,
\]

\[
u'=w,
\qquad
w'=(2c-A)u+3Au^2+2(c^2-dA)u^3.
\]

reciprocal state 没有消除临界轨道误差放大，也没有解决 horizon 分母；它必须携带 domain、branch 与 refine，并在域外选择 Kerr–Schild 数值路径。

### 4.1 `f32` 反例

Schwarzschild、$r_0=50M$、固定 RK4 的探索性复算观察到：raw-$r$ polynomial state 在 `f32` 中可能因 $r^4-b^2r^2+2b^2r$、$v_r^2-R$ 和大量小步相消，把应 escape 的 ray 错判 capture。该结果支持暂不采用 raw-$r$ state，但在完整初值、update/event/termination、浮点模式、高精度 oracle 和 machine-readable checkpoints 固化为 fixture 前，只是候选反例 `[X]`，不是对所有 raw-radius 方法的普遍证明。

reciprocal-$u$ 明显改善，但近 critical impact parameter 仍观察到精度地板。下表使用 equatorial Schwarzschild、$M=E=1$、$r_0=50M$，以 `float32` RK4 积分 $(u,w,\phi)$，返回 $u_0$ 时线性定位事件；角度基准由 80 位积分用 $r=r_{\rm turn}+s^2$ 消去 turning-point 端点奇性得到。表中数字保留为候选 fixture 的验收输入 `[X]`；只有补齐[验证合同的 numerical-conditioning metadata](validation.md#2-最低-fixture-矩阵)后才升级为 `[N]`。

| case | affine step | angle absolute error | max `abs(w²-V)` | steps |
|---|---:|---:|---:|---:|
| $b=6$ | 0.02 / 0.01 / 0.005 / 0.001 | `2.071e-4 / 4.470e-6 / 7.754e-7 / 1.985e-5` | `1.252e-6 / 2.086e-7 / 5.662e-7 / 1.505e-6` | 39 / 78 / 155 / 771 |
| $b=\sqrt{27}+10^{-3}$ | 0.02 / 0.01 / 0.005 / 0.001 | `2.886e-4 / 5.509e-4 / 2.540e-4 / 1.246e-3` | `5.364e-7 / 4.387e-7 / 4.768e-7 / 1.520e-6` | 107 / 214 / 427 / 2134 |

这组候选数据表明减小 step 未必单调改善角度，constraint drift 也不能单独预测 branch/angle error。因此 Exterior Mino 只参加同一 observable budget 下的 bake-off。`[A][X]`

## 5. Surface、volume 与时间

### 5.1 可复用关系

| 输出消费者 | 可复用 geometry | 必须重做 |
|---|---|---|
| static sky/surface | branch、source anchor、frequency ratio、travel time、Jacobian | source radiance 与 display 可独立变化 |
| dynamic surface fast-light | 同 geometry；source revision 参与 transport key | 当前 snapshot 的 emission |
| scalar volume fast-light | geodesic checkpoints 可复用 | 沿有序 path 的 emission/absorption 积分 |
| slow-light volume | geometry + travel time/checkpoint | 按 retarded time 取 snapshot/interpolate，再积 transport |
| polarized medium | geometry + transported basis checkpoints（若合同一致） | Stokes transfer 的有序矩阵演化 |

表面 transfer map 是高收益资产，因为 source evaluation 可在 map 后独立完成。volume 的 integrand 沿整条 ray 变化；缓存最终颜色或单点 source coordinate 不能代表另一 medium。

### 5.2 Fast-light contract

fast-light request 固定：

- `snapshot_id` 与 coordinate time；
- spatial interpolation scheme；
- source units/normalization；
- out-of-domain policy；
- geometry/transport generation。

所有 path sample 从同一 snapshot 求 medium。它是近似模型，不是“实时”同义词。

### 5.3 Slow-light contract

沿 trace 得到 emission coordinate time $t_{\rm em}=t_{\rm obs}-\Delta t$（符号依 canonical convention，在 reference fixture 固定）。source provider 必须声明：

- snapshots 的 time coordinate、ordering 和 revision；
- nearest/linear/higher-order interpolation；
- periodic、clamp、vacuum 或 hard-error 的 out-of-range policy；
- chunk/cache budget 和缺失 snapshot 行为；
- retarded-time key 对 temporal history 的兼容规则。

geometry 与 transport 可分开，但 geometry sample 必须保留 travel time/checkpoints；否则 slow-light 无法后来补上。[Blacklight](https://github.com/c-white/blacklight) 的 fast/slow-light 工作流可作独立实现对照，不提供本项目的数据或性能保证。

### 5.4 Forward scattering

deterministic Backward Trace 从 viewport sample 找 source，适合图像形成。Monte Carlo forward packets 适合 scattering/energy deposition，但其随机 sample、queue、variance 和终止模型不同。两者共享 domain/medium definitions 与 artifact schema，不共享一个最低公分母 GPU pass。

若 forward packets 必须向同一 pixel/bin splat，可单独测试 `SHADER_FLOAT32_ATOMIC` variant；它只解决冲突写入，不解决高争用、方差或浮点求和次序。接纳条件是在 Vulkan 与 Metal 上都优于分层归约，并在固定 seed 下满足相对于确定性 reference 的误差预算。Backward Trace、per-pixel transport、active queue 和主诊断归约均不依赖该 feature。

## 6. Source-space footprint

### 6.1 为什么屏幕梯度不够

tone-mapped RGB 平滑不表示 lens map 平滑：暗 photon ring、不同 winding 的相同颜色、cubemap seam 或 zero-emissivity 区域都可能隐藏几何不连续。footprint 定义在 source coordinates。

对屏幕坐标 $s=(s_x,s_y)$ 与局部连续 source chart $y(s)$，估计

\[
J=\frac{\partial y}{\partial s}
\approx
\begin{bmatrix}
[y(s+\delta_x)-y(s-\delta_x)]/(2\delta_x) &
[y(s+\delta_y)-y(s-\delta_y)]/(2\delta_y)
\end{bmatrix}.
\]

从 $J$ 的 singular values/ellipse 推导 texture LOD、anisotropy 与 refine signal。差分只在 termination、branch、source chart、parity 和 generation 相同的邻域有效；否则标为 discontinuity，而不是跨边界求一个巨大梯度。

escape sky 使用 seam-aware spherical/cubemap differential；surface disk 使用局部 tangent chart；volume footprint 还取决于沿路径的 beam/medium scale，不能由最终单点 Jacobian 完整表达。

### 6.2 Ray cone 的有界用途

一阶 cone/differential 适合：

- environment-map LOD 与 anisotropic filtering；
- ordinary image region 的 texture footprint；
- coarse tile 的保守 refine bound。

在 caustic/Jacobian singularity、branch split、multiple images 与强曲率长路径处，一阶模型失效。此时增加真实 samples、分 branch 或执行完整 trace；不能把 cone 无限扩张后平均不同物理像。

### 6.3 Footprint 验收

比较三种输入：tone-mapped edge、geometry/status edge、source Jacobian + branch discontinuity。场景至少含 Schwarzschild 高频 sky、Kerr critical curve、薄盘多像和 cubemap seam。以 full-resolution trace 的 source coordinate、termination、branch 和 filtered radiance 为 reference；只有 source method 在同预算下降低 worst-case error 才进入主线。

## 7. Adaptive sampling 与 active-ray execution

### 7.1 两层 adaptive

**Image-space adaptive** 决定哪些像素/tiles 需要真实 trace；**per-ray adaptive integration** 决定一条 ray 的 step。前者减少 ray 数，后者改变单 ray 成本与误差；它们有不同诊断，不能把“adaptive”当一个开关。

image classifier 至少合并：

- termination/branch/parity/source kind 不一致；
- source anchor/Jacobian/footprint disagreement；
- frequency ratio、optical depth、travel time 和 invariant diagnostic；
- radiance gradient（补充信号，不是唯一信号）；
- confidence/error estimate。

### 7.2 Execution routes

按顺序做三条可比较路径：

1. full-screen fixed dispatch，inactive lane early return；
2. coarse + full-screen refine second dispatch；
3. prefix/atomic active queue + indirect dispatch。

每条路径在 Metal/Vulkan 比较总 GPU time、active ratio、queue-build time 和 field error。register pressure、memory traffic 只有在具名 vendor profiler/offline compiler 提供可复现采集方法时才作为附加证据。只有第 3 条在目标 active ratio 分布上稳定获益才保留；wgpu 能编码 indirect dispatch 不证明它对该 workload 划算。

### 7.3 发散对策

- scalar surface、scalar volume、Stokes 分为不同 pipeline；是否降低寄存器压力只由后端编译产物或 profiler 证明；
- termination 与 cost bucket 先统计，再决定是否按粗粒度排序；
- step size 量化为少量 tier 可改善 coherence，但必须量化误差；
- 避免一开始按大量细分类全局排序；queue 构建本身可能比浪费 lane 更贵；
- coarse/refine 使用同一 solver/domain，避免 spatial seam；
- step exhaustion 是显式 outcome，不能让长 ray 无限拖住 workgroup。

## 8. Temporal correspondence 与 dynamic resolution

### 8.1 为什么普通 motion vector 不成立

黑洞 pixel 对 source 的 mapping 是多值且可不连续。observer 小移动可令一个 pixel 跨 critical curve、改变 winding/branch 或从 disk 切到 sky；一个二维 screen motion vector 不能描述全部 correspondence。FSR/TAA 的命名和通用输入不构成适用性证明。

### 8.2 History key

history acceptance 至少要求：

```text
geometry generation
transport generation
termination + image branch + parity
source kind + source anchor/chart
observer correspondence
retarded time + snapshot revision when dynamic
resolved trace semantics
```

continuous comparison还要检查 source displacement、depth/travel-time、frequency ratio、Jacobian/footprint 与 local confidence。任一 semantic key 不同直接拒绝，不以“小权重继续混色”。

### 8.3 验证顺序

1. **Stationary accumulation**：scene 与 observer 完全不变，只积累 jittered samples；
2. **Source-space reprojection**：static sky/surface，保存 source anchor/branch 并在新 frame 反查；
3. **Moving observer**：加入 observer event/frame correspondence 与 disocclusion；
4. **Dynamic source**：加入 emission time/snapshot revision；
5. **Volume**：只有 path/retarded-time contract 支持时再尝试。

每阶段记录 accept ratio、false-accept、ghost duration、critical-curve worst-case 和实际 GPU savings。accept ratio 高但 branch 错误仍是失败。

### 8.4 Dynamic resolution

display/history extent 可固定为 presentation size，geometry/transport internal extent 由 Quality Policy 控制。resolution 改变后 footprint 与 reconstruction key 必须反映新 sample spacing；history 不因物理纹理大小相同就自动相容。

controller 使用 GPU pass time 的延迟测量、上/下阈值、hysteresis、最小驻留帧和有限 scale steps。`TIMESTAMP_QUERY` 是发布基线；CPU frame time 不得冒充 GPU trace time。

## 9. WGSL `f32` 数值合同

### 9.1 规范边界

WGSL core 不提供 `f64`。subnormal、NaN/Inf 传播、重结合和 fused operation 不应按某一 CPU IEEE 直觉写成跨后端科学合同；项目也不依赖 `fast-math` 或一个极小 absolute epsilon。[WGSL floating point](https://www.w3.org/TR/WGSL/#floating-point-evaluation)

WGSL 没有可作为完整防线的 core `isFinite` 工作流。实现先在运算前检查 denominator、radicand、index、range 与状态尺度；NaN self-compare/范围检查只作额外诊断，不让 undefined/indeterminate 后果扩散。

### 9.2 具名策略

- 所有距离先除以 $M$，状态维持 $O(1)$；
- 选条件较好的 state，如 reciprocal radius，而非在远场计算大多项式相消；
- 正 quadratic root 使用稳定公式；
- horizon/extremal 的 $\Delta$ 用参数状态选择 factored/nonnegative form；
- event 定位使用 bracket/residual，不用“刚好跨过就算命中”；
- 临界 classifier 同时看 step-halving disagreement、potential barrier margin、branch/termination 与 reference band；
- `fma` 只作为经 observable 验证的数值变换，不假定跨 Metal/Vulkan 有性能收益，也不替代代数重写；
- NaN、overflow、step exhaustion 输出 typed diagnostics。

### 9.3 分层 error budget

`ObservableErrorBudget` 至少分为：

1. canonical `f64` → GPU `f32` packing；
2. integration 与 event localization；
3. spatial sampling/reconstruction；
4. temporal reuse；
5. transport；
6. display/capture。

只有最终颜色误差会把两种错误 branch 的相似颜色判为“正确”。报告必须保留 termination、source anchor、branch、frequency ratio、travel time 与 invariant fields。

## 10. Transport 与 polarization tiers

能力依次增加：

1. vacuum sky/surface + frequency ratio；
2. scalar emission/absorption + optical depth/error；
3. Kerr vacuum polarization；
4. low-resolution Stokes `IQUV` + Faraday；
5. Kerr–Newman polarization、slow-light plasma 与 scattering research。

每个 tier 使用独立 pipeline/trace contract，避免无关逻辑进入同一 shader；实际寄存器与性能收益必须测量。Kerr Penrose–Walker path 必须和直接 parallel-transport ODE 比较；full Stokes 使用 analytic slab、basis convention、coefficient sign/unit 和独立代码对照。改变 medium、basis 或 retarded-time semantics 会使 transport generation 换代。

## 11. 发行前最低算法矩阵

| 能力 | 必须比较 | 未通过时 |
|---|---|---|
| GPU trace | CPU reference 的 termination/source/frequency/error | 降低 extent/提高 refine；不能隐藏 failure |
| spatial reconstruction | full-resolution fields，不只 PSNR | 扩大真实 trace 区域 |
| temporal | no-history reference + branch-aware fields | 限制 stationary 或关闭 |
| dynamic resolution | fixed-resolution quality/performance | 固定分辨率 |
| LUT/analytic | Cartesian KS in-domain/out-of-domain | 不进入 resolved plan |
| scalar volume | analytic slab + step/tolerance sequence | 限制 surface/vacuum |
| polarization | direct parallel transport/analytic slab/independent code | 不发行该能力 |

任何 accelerator 只有在适用域可检查、错误可观察，并相对数值基线满足同一 observable budget 时才能进入 resolved plan。完整实施顺序与阶段退出条件见[路线图](roadmap.md)。

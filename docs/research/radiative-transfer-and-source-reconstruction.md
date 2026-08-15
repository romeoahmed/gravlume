# 辐射传输、source-space 重建与解析加速证据

> **状态：研究记录，非 production 契约。** 本文在提交 `6503334c1fbd` 上核验标量辐射传输、Kerr/Kerr–Newman 赤道圆轨道 source、WGSL 数据布局、branch-aware footprint 与 Carlson 椭圆积分路线。它只记录一手证据、可否证假设和建议的验收方法；已采用的稳定语义仍以 [`physics.md`](../physics.md)、[`rendering.md`](../rendering.md) 与 [`validation.md`](../validation.md) 为准，任何新结论在进入实现前必须回写对应规范。

## 结论

1. 基线提交已经完成 equatorial source anchor、frequency ratio、surface full-KS trace、sealed
   `TracePlan` 与即时 bolometric $g^4I_{\rm em}$；当时的下一条真实缺口是带 optical depth 的标量
   emission/absorption，而不是再次设计 surface observable。
2. $I_\nu/\nu^3$ 是 invariant intensity；无碰撞传播给 spectral $g^3$，只有对频率积分后才得到 bolometric $g^4$。后续 transport 应以正 optical depth 和 transmittance 累积，不能对普通 RGB 做“频移”。
3. 当前 Kerr–Newman 圆轨道公式与 timelike existence check 有一手文献支持，但 existence 不等于 radial stability。若 source 只表示运动学薄面，当前支持域可保留；若称为稳定吸积盘，必须另加 ISCO/effective-potential gate。
4. 基线 host-shared ABI 使用 `vec4`/`[f32; 4]`，当时的 160-byte uniform 与 16-byte record
   element 都符合 WGSL 布局。`vec3<f32>` 的 size 是 12 byte、alignment 是 16 byte，放进 array
   后 stride 是 16 byte；它不能用来获得 12 B/pixel 的 host ABI。
5. footprint 必须对 source map 而非 tone-mapped RGB 求导，并在完全相同的 termination/source chart/turning or winding branch 上才有效。critical curve、caustic、chart seam 与 branch change 应触发真实重采样，不能跨界插值。
6. Carlson symmetric forms 适合具名 root topology 内的 terminal accelerator 或 CPU oracle，不是完整 geodesic solver。root classification、turning count、积分反演、$t/\phi$ 极点项和 source event 仍是必要工作；near-degenerate、near-axis、near-extreme 与 unsupported branch 必须回退 Cartesian Kerr–Schild。

> **后续采用记录：** scalar slab、diluted-blackbody boxcar bands、v3 fixtures、最小 branch key、
> 五射线 surface footprint 与 tagged scientific capture 已按本记录的保守边界落地。当前契约与证据见
> [`physics.md`](../physics.md#7-frequency-与-radiative-transfer)、[`validation.md`](../validation.md#32-surface-observable)、
> [`reference-implementation.md`](../reference-implementation.md)和 [`gpu-renderer.md`](../gpu-renderer.md)；
> production reconstruction 与 Carlson accelerator 仍未实现。

## 1. 问题、方法与当前差距

本轮检查以下可否证问题：

- invariant transfer 的 coefficient、方向与 $g^3/g^4$ 是否自洽，哪些解析 fixture 能发现符号、频率和小 optical-depth 错误；
- 中性 test particle 的 Kerr/Kerr–Newman 赤道圆轨道公式在哪些域成立，现有检查是否误把轨道存在性当成稳定性；
- WGSL/WebGPU 的 host-shared 布局和并行执行实际保证了什么，哪些只是 Metal/Vulkan/GPU vendor 的性能经验；
- source-space differential 至少需要保存哪些离散 branch key 与连续 observable；
- Carlson 路线在哪些受限域能够成为 conservative accelerator，而不重复已否决的 fixed-step Mino 外推。

方法是对当前 Rust/WGSL、v2 fixture 和规范做静态核对，再与原始论文、W3C 标准及作者代码交叉检查。本轮没有修改 production、没有运行 benchmark，也没有生成新的性能数字；下文的 fixture 是待实现的验证设计，不是已通过证据。

配套的 80 位 SymPy/mpmath 复算脚本是
[`verify_scalar_transport.py`](scripts/verify_scalar_transport.py)。它验证 invariant/blackbody
恒等式、slab 极限与 partition、Planck normalization、binary64 thin-limit cancellation、v3
expected，以及 4097-entry LUT 的 midpoint 误差；运行方式为：

```text
uv run --isolated --project docs/research/scripts --locked \
  python -B docs/research/scripts/verify_scalar_transport.py
```

| 能力 | 基线提交中的证据 | 本轮确认的缺口 |
| --- | --- | --- |
| source anchor、$g$、bolometric source | CPU [`surface.rs`](../../crates/gravlume-reference/src/surface.rs)、[v2 fixture](../../crates/gravlume-reference/fixtures/v2/kerr-surface-observable.toml) | fixture 只覆盖 vacuum surface，不覆盖 absorption 或 spectral bins |
| GPU surface | [`kerr_schild_trace.wgsl`](../../crates/gravlume-render/src/shaders/kerr_schild_trace.wgsl) 产生 invocation-local sample；[`surface_preview.wgsl`](../../crates/gravlume-render/src/shaders/surface_preview.wgsl) 即时做 $g^4$ | 没有 optical depth、emissivity/absorptivity sampling、branch/parity/Jacobian |
| execution seam | [`ray_tracer.rs`](../../crates/gravlume-render/src/ray_tracer.rs) 的 `TracePlan` 是私有 sealed 分支 | 不需要为尚不存在的第二 consumer 提前公开 solver trait |
| appearance/reconstruction | [`rendering.md`](../rendering.md) 已定义 trace → transport → reconstruct 的方向 | production 仍只持久化 `RGBA16F`；source-space reconstruction 尚未实现 |
| analytic acceleration | [GPU acceleration ledger](gpu-geodesic-acceleration.md) 已否决 fixed-step numerical Mino | 尚无 root-aware Carlson implementation 或 accepted-domain classifier |

## 2. Invariant radiative transfer

### 2.1 变量与方向

[Lindquist 1966](https://doi.org/10.1016/0003-4916%2866%2990207-7) 建立了 curved-spacetime kinetic transfer；[Younsi、Wu 与 Fuerst 2012](https://arxiv.org/abs/1207.4234) 从 covariant Boltzmann equation 给出适合 ray tracing 的形式。对由局部流体四速度 $u^\mu$ 测得的正频率 $\nu=-k_\mu u^\mu$，定义

\[
\mathcal I=\frac{I_\nu}{\nu^3},\qquad
\mathcal E=\frac{j_\nu}{\nu^2},\qquad
\mathcal A=\nu\alpha_\nu .
\]

沿 future-directed photon affine parameter，局部方程可写成

\[
\frac{d\mathcal I}{d\lambda}=\mathcal E-\mathcal A\mathcal I.
\]

这组式子只用于记录推导来源；production 的 frequency convention 和 observable 仍由 [`physics.md`](../physics.md#7-frequency-与-radiative-transfer) 定义。实现时必须固定 $k^\mu$ 的 normalization、local frequency 和 affine orientation；不能只把 `step` 的符号塞进 $\Delta\tau$。对 backward tracing，最不易出错的状态是从 observer 向 source 累积

\[
\Delta\tau=\int \alpha_\nu\nu\,|d\lambda|\ge0,
\qquad T\leftarrow T e^{-\Delta\tau},
\]

并让距 observer 已有 optical depth $\tau$ 的 segment 贡献乘以当前 $T=e^{-\tau}$。rejected integrator step 不得提交 $\tau$、emission 或 travel time。

若一个 homogeneous segment 按光传播方向更新，解析式是

\[
I_{\rm out}=I_{\rm in}e^{-\Delta\tau}
  +S_\nu\left(1-e^{-\Delta\tau}\right),
\qquad S_\nu=\frac{j_\nu}{\alpha_\nu}.
\]

数值实现应计算 `-expm1(-delta_tau)`。`alpha == 0` 必须走 pure-emission limit，不能先形成 `j / alpha`；大 $\Delta\tau$ 则应自然趋向 $S_\nu$，而不是用任意 clamp 掩盖负值或非有限 coefficient。[Younsi et al. 的 formal solution](https://doi.org/10.1051/0004-6361/201219599) 直接支持这三个极限。

### 2.2 $g^3$、$g^4$ 与 spectrum

无碰撞段保持 $\mathcal I$，所以

\[
I_{\nu,{\rm obs}}(\nu_{\rm obs})
=g^3I_{\nu,{\rm em}}(\nu_{\rm obs}/g).
\]

把 $d\nu_{\rm obs}=g\,d\nu_{\rm em}$ 一并换元后才有 $I_{\rm obs}=g^4I_{\rm em}$。因此当前 [`surface_preview.wgsl`](../../crates/gravlume-render/src/shaders/surface_preview.wgsl) 对明确标成 bolometric 的 source 连乘四次 $g$ 是正确的 baseline；将同一操作用于 spectral bin 或普通 RGB 则没有物理意义。对 Planck spectrum，上式等价于 $T_{\rm obs}=gT_{\rm em}$，可作为独立于“代码也乘四次”的验证。[Younsi et al. 2012](https://arxiv.org/abs/1207.4234) 给出 invariant transfer；[RAPTOR](https://doi.org/10.1051/0004-6361/201732149) 的原始实现论文也使用 $\nu=-k\cdot u$ 和 invariant emission coefficient。

沿 volume ray，$j_\nu$ 与 $\alpha_\nu$ 必须在每个 accepted sample 的 local fluid frequency 上求值。一个固定的 observer spectral bin 会沿路径对应不同 local frequency；这正是 spectral LUT 需要频率坐标而 RGB 不足以替代它的原因。

### 2.3 最小解析 fixture

| fixture | 构造 | 必须比较的 observable / 可发现错误 |
| --- | --- | --- |
| `rt-vacuum-redshift` | vacuum 中给定解析 $g$ 与非平坦 compact spectrum | 每个 bin 比较 $g^3I_\nu(\nu/g)$；高精度积分比较 $g^4$，发现漏掉频率换元 |
| `rt-blackbody-shift` | 两个 $g<1$、$g>1$，同一 emitter temperature | spectrum 等于温度 $gT$ 的 Planck curve，积分同时满足 $g^4$ |
| `rt-pure-absorption` | Minkowski/static slab，$j=0$ | $I_{\rm out}=I_{\rm in}e^{-\tau}$；覆盖 $\tau=0,2^{-20},10^{-3},1,20$ |
| `rt-constant-slab` | $g=1$，常量正 $j,\alpha$ | 与 analytic slab 逐值比较；thin limit 由 `expm1` 保真，thick limit 趋近 $S_\nu$ |
| `rt-pure-emission` | $\alpha=0,j>0$ | 与 path length 成线性；发现 `j/alpha` 和错误 early return |
| `rt-partition-order` | 同一 slab 分为 1、2、17 个 segment；另以 backward accumulator 计算 | terminal intensity 与 transmittance 同解，发现 segment 顺序、重复提交 rejected step 和 signed-$\tau$ 错误 |
| `rt-invalid-coefficient` | 负值、NaN、overflowing source | typed rejection/diagnostic，不允许 clamp 成可见 radiance |

CPU oracle 应用 binary64 或更高精度直接算 analytic expression；GPU 比较 scene-linear spectral/bolometric 值和 $\tau$，不比较 tone-mapped RGB。随机 property test 可生成非负 $I,j,\alpha,L$ 并检查 positivity、partition invariance 和 monotonic transmittance，但命名 fixture 仍需固定上述边界。

## 3. Kerr/Kerr–Newman 赤道圆轨道 emitter

### 3.1 适用域

当前 [`physics.md`](../physics.md#8-equatorial-circular-emitter-与-disk-边界) 中的 $\Omega_\pm$、timelike normalization、frequency ratio 与 specific energy 和 [Pugliese、Quevedo、Ruffini 2013](https://arxiv.org/abs/1303.6250) 对**中性、无自旋 test particle** 的 Kerr–Newman equatorial circular geodesic 一致。它不覆盖带电 emitter 的 Lorentz force、pressure-supported/MHD flow、self-force 或 finite-thickness disk。

现有实现逐 hit 检查 $Mr-q_e^2\ge0$、圆轨道 denominator 为正和四速度 timelike；这足以拒绝不存在或已到 null limit 的 circular state。随后还必须要求该 ray 的 emitted frequency $-p\cdot u>0$。这些条件可以说明“此处存在正频 timelike circular emitter”，不能说明该 orbit 对径向扰动稳定。

对单位质量中性粒子，独立稳定性 oracle 可从 Carter radial potential 开始：

\[
R(r)=\left[E(r^2+a^2)-aL_z\right]^2
-\Delta\left[r^2+(L_z-aE)^2\right],
\qquad
\Delta=r^2-2Mr+a^2+q_e^2.
\]

circular orbit 满足 $R=R'=0$，marginally stable orbit 还满足 $R''=0$；稳定侧需按同一 potential convention 检查 second variation，而不是复用 timelike denominator。Pugliese et al. 分别给出 black-hole 与 naked-singularity 区域的 circular/stable classification；不能把一个区域的 root ordering 外推到另一区域。[Pugliese et al. 2013](https://doi.org/10.1103/PhysRevD.88.024042)

因此有两个清晰的产品解释：

- 若 `EquatorialCircularEmitter` 只是给定 velocity law 的运动学薄面，保留存在性和 timelike checks，并明确 orbit 可能 unstable；
- 若后续对象称为 stable accretion disk，则 source radial domain 必须与单独求得的 ISCO/stability region 相交。若还声明三维 perturbative stability，应再验证 vertical epicyclic mode，而不只 radial potential。

Schwarzschild 的 photon circular limit 是 $3M$、ISCO 是 $6M$；Kerr 的 prograde/retrograde ISCO 由 [Bardeen、Press 与 Teukolsky 1972](https://doi.org/10.1086/151796) 给出。按该闭式以 $a/M=0.8$ 复算，prograde ISCO 约为 $2.9066M$，所以当前默认 $[6M,20M]$ surface 位于这个 **pure-Kerr baseline** 的 stable side；这不能证明所有允许的 charge/extremality 参数也稳定。

### 3.2 near-extreme 与数值限制

当 horizon、photon orbit 与 ISCO 的 Boyer–Lindquist coordinate radii 在 extremal limit 聚合时，它们并不因此成为同一个几何 orbit；[Jacobson 2011](https://arxiv.org/abs/1107.5081) 说明该极限依赖所取 spacelike slice。数值上，圆轨道 denominator 和 $-u^2$ 同时变小时，$u^t$ 与 $g$ 会强烈放大输入误差；把微小负 radicand clamp 为零会把 timelike/null branch 混在一起。

near-extreme fixture 应记录无量纲 condition signals，而非预先宣布一个十进制 epsilon：

- $\Delta/r^2$、$D/r^2$、$-u^2$ 与相邻 radial roots 的 separation；
- CPU binary64/高精度与 GPU binary32 的 $\Omega,u^t,E,g$ agreement；
- horizon、null circular limit 和 marginal stability root 的两侧样本；
- 正/负 spin、subextreme 边界，以及当前 API 若允许的 extreme/superextreme 分类。

production threshold 应由 observable error 与 false-accept sweep 反推。任一 condition 不确定就返回 typed invalid/fallback；不得通过 clamp 或强制选择 prograde sign 继续。

### 3.3 建议 fixture

1. `orbit-schwarzschild`: $r=3M$ 两侧验证 null denominator，$r=6M$ 两侧用 $R''$ 验证 marginal stability，$r=10M$ 交叉比较 metric normalization、$R=R'=0$ 与解析 $\Omega$。
2. `orbit-kerr-spin-pair`: 同一 $|a|$ 的 prograde/retrograde 分支，比较 Bardeen–Press–Teukolsky ISCO、$u^t$、specific energy 和正 emitted frequency。
3. `orbit-kn-domain`: 由 Pugliese classification 选取一个 black-hole regular、一个 near-null、一个 algebraically invalid $Mr<q_e^2$ 和一个明确的另一 extremality class；expected result 显式区分 unsupported、existent-unstable 与 existent-stable。
4. `orbit-near-extreme-ladder`: $1-a^2/M^2-q_e^2/M^2$ 逐级缩小，在每个 critical root 两侧采样；高精度 oracle 给 branch 与 residual，GPU 只能 accepted-agree 或 conservative fallback。
5. `orbit-ray-frequency`: 对每个合法 orbit 构造 future-directed photon，使 $-p\cdot u$ 分别 regular、接近零和非正；直接验证 $g$ 的 domain，而不只验证轨道自身。

## 4. WGSL/WebGPU 布局与 GPU 并行边界

### 4.1 Host-shared layout

[WGSL alignment and size rules](https://www.w3.org/TR/WGSL/#alignment-and-size) 给出下列关键事实：

| WGSL type | alignment | size | `storage` natural array stride |
| --- | ---: | ---: | ---: |
| `f32` / `u32` | 4 | 4 | 4 |
| `vec2<f32>` | 8 | 8 | 8 |
| `vec3<f32>` | 16 | 12 | 16 |
| `vec4<f32>` | 16 | 16 | 16 |

structure member offset 必须向该 member alignment 取整，array stride 是 element size 向 element alignment 取整。`uniform` address space 对 array/structure 还有 16-byte 级的额外约束；`storage` 使用其 host-shareable natural layout，详见 [address-space layout constraints](https://www.w3.org/TR/WGSL/#address-space-layout-constraints)。dynamic uniform binding offset 的 `minUniformBufferOffsetAlignment` 是另一项 device limit，不能反过来推导 structure size。[WebGPU limits](https://www.w3.org/TR/webgpu/#limits)

基线提交中的 [`TraceUniforms`](../../crates/gravlume-render/src/ray_tracer.rs) 是十个 `[f32; 4]`，
与当时 shader 的十个 `vec4<f32>` 一一对应，总计 160 byte；`TraceDispatch` 的两个 `vec2<u32>`
总计 16 byte。采用 scalar transport 后 production ABI 增加一个具名 `surface_transport` block；当前
176-byte 合同以 [GPU 证据](../gpu-renderer.md)为准。test-only record plane 每 element 16 byte。
这些布局是清晰、可审计的 ABI。shader 内 invocation-local `GeometricSample` 不是 host buffer
layout，不需要为了 Rust packing 把其 `vec3` 拆掉；反过来，也不能把 local `vec3` 的写法当成
“持久化时只需 12 byte”的证据。

若 reconstruction 将来确实需要 semantic map，优先按 consumer 拆成可选的 16-byte plane/texture，例如 discrete key、source/time/$g$、footprint/quality；不应把完整 diagnostic `GeometricSample` 原样常驻。AoS、SoA、`RGBA16F` 或量化 key 的选择必须同时通过：

- Rust/WGSL size、alignment、offset 的 compile-time/validation test；
- representative extent 的 admission 和 steady/rebuild memory accounting；
- 解码后的 source、$g$、time、Jacobian 和 branch exactness/error gate；
- Metal 与 Vulkan 上的 bandwidth/timestamp A/B。

### 4.2 Workgroup、branch 与 vector type

WebGPU 要求 `workgroup_size` 各维及乘积不超过 requested device limits；当前 8×8 即 64 invocations，低于 WebGPU guaranteed `maxComputeInvocationsPerWorkgroup` 基线，且与二维相邻 pixel store 相容。[WebGPU compute limits](https://www.w3.org/TR/webgpu/#limits) AMD 的 [RDNA performance guide](https://gpuopen.com/learn/rdna-performance-guide/) 也把 8×8/64-thread tile 和 coherent memory access 列为常见起点，但那只是 vendor heuristic，不是 Metal/Vulkan/WGSL contract。

同一个 subgroup/SIMD group 内的 divergent path 通常降低 lane 利用率；这是 [RDNA guide](https://gpuopen.com/learn/rdna-performance-guide/) 的性能经验，不是 WGSL 语义。具体 subgroup width 和 occupancy 由 backend/device/pipeline 决定；[Apple threadgroup guidance](https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups) 与 [Vulkan subgroup rules](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#limits-subgroup) 都不支持在可移植 WGSL 中硬编码“总是 32 lanes”。因此：

- surface/volume/Stokes 等物理 tier 继续使用独立 pipeline；
- 对 geodesic step-count divergence 的优化只在 histogram 和 paired A/B 后进行，不能因代码中出现 `if` 就改成 branchless；
- WGSL [`select`](https://www.w3.org/TR/WGSL/#select-builtin) 没有 short-circuit 语义，不能用它藏住 invalid division、sqrt 或 out-of-bounds access；domain guard 应先成立；
- `vec4` 是明确的语言与存储类型，不承诺编译成一次四宽 hardware instruction。vector expression 可保留可读性与 ABI 一致性，性能结论仍需 backend profile。

当前一 invocation 一 pixel、8×8 tile、局部 `GeometricSample`、只写 production `RGBA16F` 的结构与这些边界一致。现阶段更大的风险是增加未被 consumer 使用的 per-pixel state，而不是缺少另一个向量 wrapper。

## 5. Source-space footprint 与 branch-aware reconstruction

### 5.1 需要求导的映射

[Igehy 1999](https://graphics.stanford.edu/papers/trd/) 的 ray differentials 用邻近 primary ray 的导数估计 texture footprint；[DNGR](https://arxiv.org/abs/1502.03808) 则直接积分 geodesic deviation，以 elliptical ray bundle 处理 Kerr lensing 的过滤和 critical curves。这两项证据共同支持对

\[
J=\frac{\partial s}{\partial(x,y)}
\]

求导，其中 $(x,y)$ 是 screen coordinate，$s$ 是 source chart，而不是对最终 RGB 求导。给定 pixel covariance $C_p$，局部 source footprint 可写成 $C_s=JC_pJ^T$；singular values 给出各向异性轴，$\operatorname{sign}(\det J)$ 给出 regular region 的 parity。接近 $\det J=0$ 时 parity 不稳定且映射穿过 critical curve，应标为 refine/discontinuous，而不是把退化或跨 branch 跳变的 stencil 解释成可过滤 ellipse。

disk 的 $(r,\phi)$ 必须先做 periodic unwrap，再在局部 tangent chart 中 differencing；sky 要用 spherical tangent/cubemap seam-aware chart。volume emission 的 footprint 随路径和 medium scale 演化，terminal 单点 Jacobian 不足以代表整条 beam。

### 5.2 离散 semantic key 先于连续差分

[AART](https://arxiv.org/abs/2211.07469) 把不同 lensing band 分开构造，并在 transfer function 中保存 source $r,t,\phi$ 和 emitted radial-momentum sign；其[作者代码](https://github.com/iAART/aart)展示了 branch information 不是从相似颜色可靠恢复的。DNGR 也以 ray-bundle/caustic 结构说明不同 image branch 不能无条件插值。

一个可测试的最小候选 key/observable 是：

| 类别 | 候选字段 | 用途 |
| --- | --- | --- |
| exact key | termination、source kind/chart、producer/generation | 阻止 horizon/sky/surface、旧 generation 或不同 chart 混合 |
| branch key | radial/polar momentum sign、turning count、equatorial crossing/lensing order、winding | 区分同一 source anchor 的多像路径；具体最小集由反例缩减 |
| orientation | parity 或可稳定复算 parity 的 sign | 阻止跨 critical curve 重建 |
| continuous | unwrapped source coordinate、travel/emission time、$g$ | source lookup、slow-light 与 transport |
| differential | $J$ 或 conservative ellipse/singular values | anisotropic filtering 与 adaptive sampling |
| confidence | event residual、Jacobian consistency、near-critical/near-seam flag | conservative accept/refine |

当前 shader 的 `source_coordinates = (r, phi, g)` 和 termination 已经是有用起点，但没有 turning/winding/parity/Jacobian；shadow alpha 只覆盖 Horizon/Escape silhouette，不能推出 surface branch continuity。

### 5.3 最小可执行验证

1. 对每个中心 pixel 追 `center, ±x, ±y` 五条 full rays；仅当所有 exact/branch key 相同且 chart 可 unwrap 时做 central difference。任一 key 不同，expected result 是 `refine`，不是任意 Jacobian。
2. 在 ordinary region 比较 finite-difference $J$ 与更小 step、九点 stencil 或 CPU geodesic-deviation oracle；同时比较 singular values、parity 和 source displacement，不只比较 determinant。
3. 固定至少四类命名 scene：smooth disk、不同 winding 的 higher-order image、critical curve 两侧、sky/cubemap seam。再加 high-frequency source texture 检验 filtered scene-linear radiance。
4. reconstruction 与高倍率 full-trace/supersample oracle 比较 termination/branch exactness、source coordinate、$g$、time、radiance 和 temporal stability。PSNR/SSIM 只能补充画质，不能替代 branch gate。
5. 对每个 accepted reconstruction 保存“为何可插值”的 condition signal；classifier mutation test 必须证明去掉任一 key 或 seam check 会出现固定反例。

先用 reconstruction pass 的邻近 full samples 建立该证据，比立即在每条 geodesic 中积分 geodesic-deviation tensor 更小。只有 finite difference 的额外 ray cost 或 near-critical failure 已被量化，才有理由把 analytic differential 加入 hot loop。

## 6. Carlson/椭圆积分作为受限 accelerator

### 6.1 它解决什么，不解决什么

[Carlson 1979](https://doi.org/10.1007/BF01396491) 的 symmetric elliptic integrals 通过 duplication 把 arguments 逐步拉近，主要使用算术和 square root；[NIST DLMF §19.36](https://dlmf.nist.gov/19.36) 给出算法与误差界。$R_F,R_C,R_D,R_J$ 可统一处理 Kerr geodesic 中第一、第二、第三类 elliptic terms，避免为每种 Legendre normal form 各写一套 special function。

但 Carlson evaluator 的输入不是相机 ray 本身。[Dexter 与 Agol 2009](https://arxiv.org/abs/0903.0620) 的 semi-analytic Kerr solver 仍要分类 radial/polar roots、处理 turning point 和 complex-root case，并为 $t,\phi$ 计算带 pole 的第三类项。[Gralla 与 Lupsasca 2020](https://arxiv.org/abs/1910.12881) 进一步给出 manifestly real 的 root classes 和 trajectory inversion；其修正版范围是 non-extremal Kerr exterior。[Wang、Lee 与 Lin 2022](https://arxiv.org/abs/2208.11906) 将显式 Mino-time trajectory 扩展到 Kerr–Newman exterior，但增加的 charge/root classes 仍需独立验证。

所以完整 terminal accelerator 至少包含：

1. 从项目 tetrad/KS initial covector 构造并复核 $E,L_z,Q$；
2. conservative quartic root topology 与 radial/polar sign/turning count classifier；
3. Carlson/等价 manifestly-real integral 及其 inversion；
4. $t,\phi$ 的 pole/log term、chart seam 与 event ordering；
5. horizon/escape/surface terminal localization、source anchor、travel time 和 $g$；
6. terminal state 回到 outgoing Cartesian KS 后的独立 null/invariant/event residual；
7. 任一 uncertain condition 排入现有 full-KS fallback。

这也解释了为何 Carlson terminal map 不能自动服务 arbitrary volume transport：volume 需要沿 ray 的 ordered checkpoints、local frequency 和 medium sample。除非 analytic solver 另行暴露并验证可反演的 path sampler，否则它只应声称 terminal sky/surface 能力。

### 6.2 第一版应封闭的定义域

建议先把 candidate 收窄为：pure Kerr、明确 subextreme、observer 与 terminal 都在 exterior、off-axis、具名且 roots 充分分离的 topology，以及当前 first surface/escape event 能由同一 branch 无歧义定位。以下情况直接 full-KS：

- repeated/near-repeated roots、critical curve 邻域或 classifier residual 不足；
- near-axis，polar amplitude/azimuth formula 条件差；
- near-extreme、接近 horizon 的 $\Delta$ pole 或 BL↔outgoing-KS transform 条件差；
- unsupported turning/crossing count、volume checkpoint、inside-horizon continuation；
- Kerr–Newman，直到 pure-Kerr accepted set 已证明 correctness 和端到端收益，并为 charge topology 建立独立 fixture。

Carlson duplication 本身较规整，不代表前后的 root sorting、case selection 和 inversion 在 GPU 上无 divergence。一个固定 observer/spacetime 的 transfer map bake，或“classify/compact → analytic pass → KS fallback”的两阶段结构，可能比把所有 case 塞进单个 megakernel 更适合 GPU；是否更快只能比较 analytic + queue + fallback 的总 timestamp。[现有 acceleration ledger](gpu-geodesic-acceleration.md) 已给出相同的 conservative-fallback 原则。

### 6.3 准入 fixture 与 oracle

| gate | oracle / observable | 失败含义 |
| --- | --- | --- |
| special function | 80+ bit $R_F/R_C/R_D/R_J$ 与 direct quadrature、DLMF identities；覆盖零参数、尺度变换、接近相等 arguments | evaluator 不得进入 accepted candidate |
| roots/topology | 高精度 quartic roots、potential residual、root separation、turning sequence | uncertain 必须 fallback；false accept 为 blocker |
| terminal | 与独立 binary64 full-KS 比 termination、source anchor、travel time、$g$、event residual | potential identity 通过也不能替代 terminal observable |
| critical ladder | radial/polar double-root 两侧、positive/negative spin、near-axis、near-extreme | accepted error 或 branch flip 为 blocker；fallback 是预期成功 |
| classifier mutation | 删除一个 sign/root-separation/domain check 后必须触发固定反例 | 没有 witness 的 guard 需重新论证，但不能凭性能直删 |
| GPU A/B | 相同 accepted set 和最终 observable 下，统计 analytic + compaction + fallback 总时间 | “closed form”或低 fallback ratio 本身不是收益证据 |

当前 fixed-step Mino candidate 已因 accepted ray 的 travel-time 反例被拒绝，见 [`mino-step-selection.md`](mino-step-selection.md)。Carlson 路线的恢复条件不是继续扫一个更小 fixed step，而是用 root-aware integral 封闭 terminal phase，并在上述 gates 下证明 conservative acceptance。

## 7. 研究决策与恢复条件

| 候选 | 本轮决策 | 进入 production / 重开条件 |
| --- | --- | --- |
| scalar invariant emission/absorption | **优先补齐**，先 CPU analytic fixture，再 GPU | 规范固定 coefficient、orientation、frequency sampling；第 2.3 节 fixture 和 CPU/GPU agreement 通过 |
| 对普通 RGB 直接应用 spectral redshift | **拒绝** | 引入具名 spectrum/bandpass 与频率轴后才可讨论；bolometric 继续只用 $g^4$ |
| 把 timelike circular existence 当 stable disk | **拒绝** | 独立 effective-potential/epicyclic gate 和 ISCO fixture 通过，且产品确实需要 stable disk 语义 |
| 持久化完整 `GeometricSample` G-buffer | **暂不采用** | branch-aware reconstruction 证明有 consumer，并通过 layout、显存、误差与 Metal/Vulkan A/B |
| source-space finite-difference footprint | **下一阶段可实验** | exact semantic key、seam/critical fallback 和 supersampled oracle 先落地 |
| 所有 Kerr/KN ray 全量改成 Carlson | **拒绝** | 不存在 unsupported/ill-conditioned accepted ray，或始终保留可观测上等价的 KS fallback；端到端收益成立 |
| pure-Kerr exterior Carlson terminal accelerator | **保留为受限候选** | root-aware classifier、80+ bit oracle、KS observable gate、false-accept sweep 与总 GPU A/B 全部通过 |

推荐的实现顺序是：标量 invariant transport 与解析 slab fixture → spectral/blackbody fixture → 最小 branch key 与 source-space finite-difference footprint → branch-aware reconstruction → pure-Kerr root-aware Carlson CPU oracle/transfer-map candidate。每一步只引入当下 consumer 需要的字段，并继续让 full-KS 定义 unsupported domain 的行为。

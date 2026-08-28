# 辐射传输、source-space 重建与解析加速证据

本文以提交 `6503334c1fbd` 为研究基线，核验标量辐射传输、Kerr/Kerr–Newman 赤道圆轨道 source、WGSL 数据布局、branch-aware footprint 与 Carlson 路线，并记录后续哪些结论被采用。它只保存一手证据、可否证假设、实验结果和恢复条件，不定义 production 行为。

**状态：混合研究记录。** 已采用语义以[数学物理](../physics.md)、[渲染设计](../rendering.md)、[验证合同](../validation.md)与实现证据为准；任何新结论进入实现前必须先回写其唯一权威文档。

## 结论

1. 基线提交已经完成 equatorial source anchor、frequency ratio、surface full-KS trace、sealed
   `TracePlan` 与即时 bolometric $g^4I_{\rm em}$；当时的下一条真实缺口是带 optical depth 的标量
   emission/absorption，而不是再次设计 surface observable。
2. $I_\nu/\nu^3$ 是 invariant intensity；无碰撞传播给 spectral $g^3$，只有对频率积分后才得到 bolometric $g^4$。已采用的 scalar operator 以非负 optical depth 和 transmittance 表达；未来 volume transport 仍必须保持这一方向语义，不能对普通 RGB 做“频移”。
3. 当前 Kerr–Newman 圆轨道公式与 timelike existence check 有一手文献支持，但 existence 不等于 radial stability。若 source 只表示运动学薄面，当前支持域可保留；若称为稳定吸积盘，必须另加 ISCO/effective-potential gate。
4. 基线 host-shared ABI 使用 `vec4`/`[f32; 4]`，当时的 160-byte uniform 与 16-byte record
   element 都符合 WGSL 布局。`vec3<f32>` 的 size 是 12 byte、alignment 是 16 byte，放进 array
   后 stride 是 16 byte；它不能用来获得 12 B/pixel 的 host ABI。
5. footprint 必须对 source map 而非 tone-mapped RGB 求导，并在完全相同的 termination/source chart/turning or winding branch 上才有效。critical curve、caustic、chart seam 与 branch change 应触发真实重采样，不能跨界插值。
6. Carlson symmetric forms 适合具名 root topology 内的 terminal accelerator 或 CPU oracle，不是完整 geodesic solver。root classification、turning count、积分反演、$t/\phi$ 极点项和 source event 仍是必要工作；near-degenerate、near-axis、near-extreme 与 unsupported branch 必须回退 Cartesian Kerr–Schild。

> **后续采用记录：** scalar slab、diluted-blackbody boxcar bands、v3 fixtures、最小 branch key、
> 五射线 surface footprint、tagged scene-linear radiance capture 与有界单样本 record 已按保守边界落地；后续又采用固定单槽 production ticket/completion seam、lifecycle cancel-drain 与 desktop consumer。当前契约与证据见
> [`physics.md`](../physics.md#7-frequency-与-radiative-transfer)、[`validation.md`](../validation.md#32-surface-observable)、
> [`reference-implementation.md`](../reference-implementation.md)、[`gpu-renderer.md`](../gpu-renderer.md)和[production inspection 决策](on-demand-sample-inspection.md)；
> 连续字段质量基线、production reconstruction 与 Carlson accelerator 仍未实现。

## 1. 问题、方法与当前差距

本轮检查以下可否证问题：

- invariant transfer 的 coefficient、方向与 $g^3/g^4$ 是否自洽，哪些解析 fixture 能发现符号、频率和小 optical-depth 错误；
- 中性 test particle 的 Kerr/Kerr–Newman 赤道圆轨道公式在哪些域成立，现有检查是否误把轨道存在性当成稳定性；
- WGSL/WebGPU 的 host-shared 布局和并行执行实际保证了什么，哪些只是 Metal/Vulkan/GPU vendor 的性能经验；
- source-space differential 至少需要保存哪些离散 branch key 与连续 observable；
- Carlson 路线在哪些受限域能够成为 conservative accelerator，而不重复已否决的 fixed-step Mino 外推。

方法是对研究基线的 Rust/WGSL、v2 fixture 和规范做静态核对，再与原始论文、W3C 标准及作者代码交叉检查。
一手资料核对阶段没有修改 production；其后的 reconstruction 候选实验及 benchmark 单列在第 5.6 节，
最终没有保留候选代码。后续落地状态以本节证据表和第 2.3 节矩阵为准，不从最初的建议语气推断当前能力。

配套的 80 位 SymPy/mpmath 复算实现是
[`scalar_transport.py`](scripts/src/gravlume_research/checks/scalar_transport.py)。它验证 invariant/blackbody
恒等式、slab 极限与 partition、Planck normalization、binary64 thin-limit cancellation 与
4097-entry LUT 的 midpoint 误差，并生成 v3 expected 的独立 oracle；运行方式为：

```text
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research scalar-transport
```

高精度物理常数以十进制 source 保存，并在 `workdps(80)` 生效后才构造为 `mpf`；否则
module import 会先按 mpmath 默认精度舍入，后续提高 precision 不能恢复丢失位。该行为由
[mpmath precision management](https://mpmath.org/doc/1.3.0/general.html#precision-management)
约束，并由一个具名 6000 K red-band oracle 保留最小回归测试。旧构造在该 case 引入
$6.21612160649749\times10^{-20}$ 的 absolute error，远大于 `1e-70` oracle gate；影响范围是四份
v3 fixture 的 spectral expected，geometry、bolometric、输入与 tolerance 均未改变。

```text
uv run --isolated --project docs/research/scripts --locked \
  pytest docs/research/scripts/tests/test_scalar_transport.py
```

依赖解析、升级、完整测试与 lint 命令见[统一 Python 研究工具链](python-research-tooling.md)。

修正只更新同一 `surface-transport-v1` 物理 profile 的高精度 spectral expected；schema、
profile meaning、输入与 tolerance 均未改变，符合[验证合同的 oracle 勘误条件](../validation.md#6-fixture-envelope)。

| 能力                                  | 当前采用证据                                                                                                                                                                                                                                                                                                                                                                                                                                               | 未闭合边界                                                                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| source、frequency 与 scalar transport | CPU [`surface.rs`](../../crates/gravlume-reference/src/surface.rs)、[v2 neutral fixture](../../crates/gravlume-reference/fixtures/v2/kerr-surface-observable.toml)与 [v3 spectral/slab fixtures](../../crates/gravlume-reference/fixtures/v3/)                                                                                                                                                                                                                   | 更广 source geometry、空间变化 coefficient、scattering 与 polarization                                               |
| GPU surface                           | [`geodesic_integration.wgsl`](../../crates/gravlume-render/src/shaders/geodesic_integration.wgsl) 产生 invocation-local sample；plan-specific surface shader 完成 bolometric 或 spectral/slab transport；production 单槽 inspection 已有完整 branch key 与 published-texel separation，test capture 另有 finite-difference Jacobian                                                                                                                          | production 没有 semantic/footprint map、branch-aware reconstruction 或 multi-image/near-critical convergence evidence |
| execution seam                        | [`trace.rs`](../../crates/gravlume-render/src/trace.rs) 的 `TracePlan` 是私有 sealed 分支                                                                                                                                                                                                                                                                                                                                                                   | 不需要为尚不存在的第二 consumer 提前公开 solver trait                                                                 |
| appearance/reconstruction             | [渲染设计](../rendering.md)定义 trace → transport → reconstruct 的准入边界                                                                                                                                                                                                                                                                                                                                                                                | production 仍只持久化 `RGBA16F`；source-space reconstruction 尚未实现                                                 |
| analytic acceleration                 | [GPU acceleration ledger](gpu-geodesic-acceleration.md) 已否决 fixed-step numerical Mino                                                                                                                                                                                                                                                                                                                                                                    | 尚无 root-aware Carlson implementation 或 accepted-domain classifier                                                  |

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

把 $d\nu_{\rm obs}=g\,d\nu_{\rm em}$ 一并换元后才有 $I_{\rm obs}=g^4I_{\rm em}$。因此当前 [`bolometric_surface_preview.wgsl`](../../crates/gravlume-render/src/shaders/bolometric_surface_preview.wgsl) 对明确标成 bolometric 的 source 连乘四次 $g$ 是正确的 baseline；将同一操作用于 spectral bin 或普通 RGB 则没有物理意义。对 Planck spectrum，上式等价于 $T_{\rm obs}=gT_{\rm em}$，可作为独立于“代码也乘四次”的验证。[Younsi et al. 2012](https://arxiv.org/abs/1207.4234) 给出 invariant transfer；[RAPTOR](https://doi.org/10.1051/0004-6361/201732149) 的原始实现论文也使用 $\nu=-k\cdot u$ 和 invariant emission coefficient。

沿 volume ray，$j_\nu$ 与 $\alpha_\nu$ 必须在每个 accepted sample 的 local fluid frequency 上求值。一个固定的 observer spectral bin 会沿路径对应不同 local frequency；这正是 spectral LUT 需要频率坐标而 RGB 不足以替代它的原因。

### 2.3 最小解析验证矩阵

下表是验证 case，不是稳定 artifact ID。采用状态只说明当前仓库覆盖了该数学命题；精确 fixture、profile 与 tolerance 仍以[验证合同](../validation.md)和版本化 TOML 为准。

| Case                    | 构造                                                               | 必须比较的 observable / 可发现错误                                                                        | 状态                                      |
| ----------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| vacuum spectral shift   | vacuum 中给定解析 $g$ 与非平坦 compact spectrum                    | 每个 bin 比较 $g^3I_\nu(\nu/g)$；高精度积分比较 $g^4$，发现漏掉频率换元                                   | 一般 compact spectrum 尚未实现            |
| blackbody shift         | $g<1$ 的 canonical surface 与独立 Planck oracle                    | spectrum 等于温度 $gT$ 的 Planck curve，积分同时满足 $g^4$                                                | 已由 symbolic identity 与 v3 fixture 覆盖 |
| pure absorption         | static homogeneous slab，$j=0$                                     | $I_{\rm out}=I_{\rm in}e^{-\tau}$；覆盖 zero、thin、regular 与 high optical depth                         | 已由 v3 fixture 与边界测试覆盖            |
| constant slab           | $g=1$，常量正 source function                                      | 与 analytic slab 逐值比较；thin limit 由 `expm1` 保真，thick limit 趋近 source                             | 已由 v3 fixture 与 property test 覆盖     |
| pure emission           | $\alpha=0,j>0$                                                     | 与 integrated emission 成线性；发现 `j/alpha` 和错误 early return                                        | 已由 v3 fixture 覆盖                      |
| partition/order         | 同一 homogeneous operator 分段组合                                 | terminal intensity 与 transmittance 同解，发现顺序、重复提交与 signed-$\tau$ 错误                         | operator 已覆盖；volume checkpoints 未实现 |
| invalid coefficient     | 负值、NaN、overflowing source                                      | typed rejection/diagnostic，不允许 clamp 成可见 radiance                                                  | domain/property tests 已覆盖              |

CPU oracle 使用 binary64 或更高精度直接计算 analytic expression；GPU 比较 scene-linear spectral/bolometric 值，不比较 tone-mapped RGB。Property tests 覆盖连续代数域，具名 fixture 保留 canonical scientific case；两者不相互替代。

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

### 3.3 尚未采用的 stability fixture

以下是把运动学 circular emitter 扩展为稳定轨道声明时需要的新证据；当前 `EquatorialCircularEmitter` 不作该声明，这些 case 尚未进入 versioned fixture。

1. `orbit-schwarzschild`: $r=3M$ 两侧验证 null denominator，$r=6M$ 两侧用 $R''$ 验证 marginal stability，$r=10M$ 交叉比较 metric normalization、$R=R'=0$ 与解析 $\Omega$。
2. `orbit-kerr-spin-pair`: 同一 $|a|$ 的 prograde/retrograde 分支，比较 Bardeen–Press–Teukolsky ISCO、$u^t$、specific energy 和正 emitted frequency。
3. `orbit-kn-domain`: 由 Pugliese classification 选取一个 black-hole regular、一个 near-null、一个 algebraically invalid $Mr<q_e^2$ 和一个明确的另一 extremality class；expected result 显式区分 unsupported、existent-unstable 与 existent-stable。
4. `orbit-near-extreme-ladder`: $1-a^2/M^2-q_e^2/M^2$ 逐级缩小，在每个 critical root 两侧采样；高精度 oracle 给 branch 与 residual，GPU 只能 accepted-agree 或 conservative fallback。
5. `orbit-ray-frequency`: 对每个合法 orbit 构造 future-directed photon，使 $-p\cdot u$ 分别 regular、接近零和非正；直接验证 $g$ 的 domain，而不只验证轨道自身。

## 4. WGSL/WebGPU 布局与 GPU 并行边界

### 4.1 Host-shared layout

[WGSL alignment and size rules](https://www.w3.org/TR/WGSL/#alignment-and-size) 给出下列关键事实：

| WGSL type     | alignment | size | `storage` natural array stride |
| ------------- | --------: | ---: | -----------------------------: |
| `f32` / `u32` |         4 |    4 |                              4 |
| `vec2<f32>`   |         8 |    8 |                              8 |
| `vec3<f32>`   |        16 |   12 |                             16 |
| `vec4<f32>`   |        16 |   16 |                             16 |

structure member offset 必须向该 member alignment 取整，array stride 是 element size 向 element alignment 取整。`uniform` address space 对 array/structure 还有 16-byte 级的额外约束；`storage` 使用其 host-shareable natural layout，详见 [address-space layout constraints](https://www.w3.org/TR/WGSL/#address-space-layout-constraints)。dynamic uniform binding offset 的 `minUniformBufferOffsetAlignment` 是另一项 device limit，不能反过来推导 structure size。[WebGPU limits](https://www.w3.org/TR/webgpu/#limits)

研究基线的 `TraceUniforms` 是十个 `[f32; 4]`，与当时 shader 的十个 `vec4<f32>` 一一对应，
总计 160 byte；`TraceDispatch` 的两个 `vec2<u32>`
总计 16 byte。采用 scalar transport 后 production ABI 增加一个具名 `surface_transport` block；当前
合同以 host/shader source 和 [GPU 证据](../gpu-renderer.md)为准。test-only record plane 每 element 16 byte。
这些布局是清晰、可审计的 ABI。shader 内 invocation-local `GeometricSample` 不是 host buffer
layout，不需要为了 Rust packing 把其 `vec3` 拆掉；反过来，也不能把 local `vec3` 的写法当成
“持久化时只需 12 byte”的证据。

若 reconstruction 将来确实需要 semantic map，优先按 consumer 拆成可选的 16-byte plane/texture，例如 discrete key、source/time/$g$、footprint/quality；不应把完整 diagnostic `GeometricSample` 原样常驻。AoS、SoA、`RGBA16F` 或量化 key 的选择必须同时通过：

- Rust/WGSL size、alignment、offset 的 compile-time/validation test；
- representative extent 的 admission 和 steady/rebuild memory accounting；
- 解码后的 source、$g$、time、Jacobian 和 branch exactness/error gate；
- Metal 与 Vulkan 上的 bandwidth/timestamp A/B。

研究基线使用 `wgpu 30.0.0`；下列计算采用该版本
[`Limits::defaults`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Limits.html) 的相关值。当前锁定版本以
[`Cargo.lock`](../../Cargo.lock) 为准，依赖升级后必须重新核对实际 requested limits。基线值是：每 shader
stage 最多 8 个 storage-buffer binding、单 binding 最多 128 MiB、单 buffer 最多 256 MiB、
workgroup storage 16 KiB、每 workgroup 256 invocations、每 dispatch dimension 65535
workgroups；device 只能使用创建时实际请求到的 limits，不能因 adapter 报告更高值就默认可用。由此可直接算出：
3840×2160 的一个 16 B/pixel plane 是 132,710,400 B（126.5625 MiB），距默认单 binding
上限只剩 1.4375 MiB；4096×2160 的同类 plane 是 135 MiB，已经越界；三个同尺寸 plane
若合并进一个 UHD buffer 则是 379.6875 MiB，也超过默认 `maxBufferSize`。所以“每个 semantic plane 都是
`array<vec4<_>>` 且全分辨率常驻”不是 portable design：resource admission 必须按 requested device
limits 与实际 extent 决定使用 coarse nodes、tile/split binding、texture 或拒绝该 mode，不能在
allocation 失败后静默降精度。`minStorageBufferOffsetAlignment = 256` 只约束 dynamic binding
offset，不会把 WGSL element stride 改成 256 byte。

### 4.2 Workgroup、branch 与 vector type

WebGPU 要求 `workgroup_size` 各维及乘积不超过 requested device limits；当前 8×8 即 64 invocations，低于 WebGPU guaranteed `maxComputeInvocationsPerWorkgroup` 基线，且与二维相邻 pixel store 相容。[WebGPU compute limits](https://www.w3.org/TR/webgpu/#limits) AMD 的 [RDNA performance guide](https://gpuopen.com/learn/rdna-performance-guide/) 也把 8×8/64-thread tile 和 coherent memory access 列为常见起点，但那只是 vendor heuristic，不是 Metal/Vulkan/WGSL contract。

同一个 subgroup/SIMD group 内的 divergent path 通常降低 lane 利用率；这是 [RDNA guide](https://gpuopen.com/learn/rdna-performance-guide/) 的性能经验，不是 WGSL 语义。具体 subgroup width 和 occupancy 由 backend/device/pipeline 决定；[Apple threadgroup guidance](https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups) 与 [Vulkan subgroup rules](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#limits-subgroup) 都不支持在可移植 WGSL 中硬编码“总是 32 lanes”。因此：

- surface/volume/Stokes 等物理 tier 继续使用独立 pipeline；
- 对 geodesic step-count divergence 的优化只在 histogram 和 paired A/B 后进行，不能因代码中出现 `if` 就改成 branchless；
- WGSL [`select`](https://www.w3.org/TR/WGSL/#select-builtin) 没有 short-circuit 语义，不能用它藏住 invalid division、sqrt 或 out-of-bounds access；domain guard 应先成立；
- `vec4` 是明确的语言与存储类型，不承诺编译成一次四宽 hardware instruction。vector expression 可保留可读性与 ABI 一致性，性能结论仍需 backend profile。

现行一 invocation 一 pixel、8×8 tile、局部 `GeometricSample`、只写 production `RGBA16F` 的结构仍落在上述研究基线范围内。若 reconstruction 为 tile halo 增加 `var<workgroup>` 或改变 dispatch，必须按当前 requested limits 重新 admission，不能把历史默认值提升为长期合同。现阶段更大的风险是增加未被 consumer 使用的 per-pixel state，而不是缺少另一个向量 wrapper。

### 4.3 Workgroup barrier 与跨 dispatch 可见性

WGSL 的内存模型采用 Vulkan memory model；所有 synchronization built-in 的 execution scope
和 memory scope 都是 `Workgroup`。`workgroupBarrier()` 是 `AcquireRelease + WorkgroupMemory`，
`storageBarrier()` 是 `AcquireRelease + UniformMemory`，且所有 barrier 必须位于 compute shader
的 uniform control flow。[WGSL memory model](https://www.w3.org/TR/WGSL/#memory-model)
[WGSL synchronization built-ins](https://www.w3.org/TR/WGSL/#barrier-builtin-functions) 因而它们只能在
**同一 workgroup** 内发布对应 address space 的写入；二者都不是跨 workgroup 或全 dispatch
barrier。当前 [`geodesic_acceleration.wgsl`](../../crates/gravlume-render/src/shaders/geodesic_acceleration.wgsl)
的两次 `workgroupBarrier()` 用于共享
`var<workgroup>` stencil，且所有 invocation 都会到达，属于规范覆盖的用法。

WGSL storage atomic 的 ordering 是 `Relaxed`：只对同一 atomic memory location 的 atomic
operations 给出 ordering，不会给 payload 的 non-atomic store 或另一 atomic location 建立发布关系。
[WGSL atomic built-ins](https://www.w3.org/TR/WGSL/#atomic-builtin-functions) 所以 atomic append
counter 可以分配 queue slot，但不能让另一个 workgroup 在**同一 dispatch** 中把“counter 已增加”
当作 payload 已发布或全局第一阶段已结束的信号。

跨 workgroup 的 producer/consumer 应编码成有序的不同 dispatch（或 pass），而不是在 WGSL 中模拟
grid barrier。WebGPU 把 compute pass 中的每个 `dispatchWorkgroups` 定义为一个独立 usage scope；
一个 scope 内的 operations 可并发执行。[WebGPU synchronization](https://gpuweb.github.io/gpuweb/#synchronization)
`wgpu 30` 对正常 binding 自动插入 resource barrier；
[`ComputePass::transition_resources`](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePass.html#method.transition_resources)
明确是 native-only interoperability 高级 API，并说明其 semantics/granularity 与 binding 的自动 barrier
相同。`wgpu-hal 30` 还保证同一 queue 上 command buffer 按提交顺序执行、后者可观察前者结果。
[`wgpu-hal Queue`](https://wgpu.rs/doc/wgpu_hal/trait.Queue.html) 因此正常 reconstruction 路径应让
trace dispatch 写 map、后续 classify/reconstruct dispatch 读 map；这项 GPU→GPU dependency 不要求
CPU wait，也不要求在两个 dispatch 之间调用 WGSL `storageBarrier()`。显式 transition 只在绕过普通
binding tracking 的 native interop 场景才有理由出现。

## 5. Source-space footprint 与 branch-aware reconstruction

### 5.1 需要求导的映射

[Igehy 1999](https://graphics.stanford.edu/papers/trd/) 的 ray differentials 用邻近 primary ray 的导数估计 texture footprint；[DNGR](https://arxiv.org/html/1502.03808) 则直接积分 geodesic deviation，以 elliptical ray bundle 处理 Kerr lensing 的过滤和 critical curves。这两项证据共同支持先对 source map 求导，但 production consumer 真正需要的是它所查询坐标 $q=U(s)$ 的 Jacobian：

\[
J_q=\frac{\partial (U\circ s)}{\partial(x,y)}.
\]

其中 $(x,y)$ 是 screen coordinate，$s$ 是几何 source chart，$U$ 才是 texture UV、sky tangent
coordinate 或其他 consumer lookup transform。给定 pixel covariance $C_p$，局部 lookup footprint 是
$C_q=J_qC_pJ_q^T$；singular values 的数值依赖坐标单位与 chart metric，不能把 test-only
$(r,r_c\Delta\phi)$ Jacobian 不经 transform 就当成 texture-UV gradient。
[Heckbert 1989](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1989/5504.html) 对 arbitrary mapping
的 space-variant resampling 给出同一条“先明确 mapping，再构造 filter”边界。

在固定 oriented charts 的 regular region，$\operatorname{sign}(\det J_q)$ 给出 orientation/parity，而
critical curve 是
**精确满足** $\det J_q=0$ 的 locus；这不是“$|\det J_q|$ 较小就已证明一个有限 pixel 穿过
critical curve”。[Daněk 与 Heyrovský 2015](https://arxiv.org/abs/1501.02722) 将 Jacobian 零等值线定义为
critical curve；[DNGR](https://arxiv.org/html/1502.03808) 展示 caustic crossing 时 image pair 在
critical curve 上产生或湮灭。小 determinant 只说明局部映射 ill-conditioned。可执行 gate 应在
determinant 的离散化/舍入 uncertainty interval 包含零、邻点 parity 改变、branch key 改变或
nonlinear consistency 超预算时 `refine`；只有 analytic bound 或更密采样真正 bracket 零点时，才能
声称 footprint crosses a critical curve。

disk 的 $(r,\phi)$ 必须先做 periodic unwrap，再在局部 tangent chart 中 differencing；sky 要用 spherical tangent/cubemap seam-aware chart。volume emission 的 footprint 随路径和 medium scale 演化，terminal 单点 Jacobian 不足以代表整条 beam。

过滤的对象也不能止于 source texel。最终 pixel observable 是 footprint 内 **scene-linear observed
radiance** 的积分，schematically

\[
\bar L_p=\int_P W_p(x,y)\,L_{\rm obs}(s(x,y),g(x,y),\tau(x,y),\ldots)\,dx\,dy.
\]

因此“先平均 source，再乘中心 ray 的 $g^4$/transmittance/visibility”一般不等于上式；DNGR 的
formal filtering 同时包含 beam、filter 与 occlusion，并明确指出只在 beam centre 评估 visibility
会在穿过 shadow 时给出错误硬边。[DNGR Appendix A.3](https://arxiv.org/html/1502.03808) 候选实现要么
在 footprint samples 上重算 transport 后平均最终 radiance，要么重建 source、$g$、time、$\tau$ 等
连续量并证明它们在 accepted footprint 内的 variation 均低于预算。tone-mapped RGB 与 discrete
branch tag 都不得作为插值对象。

### 5.2 离散 semantic key 先于连续差分

[AART](https://arxiv.org/html/2211.07469) 把不同 lensing band 分开构造，并在 transfer function 中保存 source $r,t,\phi$；其[作者代码](https://github.com/iAART/aart)还保存 emitted radial-momentum sign，展示 branch information 不是从相似颜色可靠恢复的。DNGR 也以 ray-bundle/caustic 结构说明不同 image branch 不能无条件插值。

一个可测试的最小候选 key/observable 是：

| 类别         | 候选字段                                                                              | 用途                                                      |
| ------------ | ------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| exact key    | termination、source kind/chart、producer/generation                                   | 阻止 horizon/sky/surface、旧 generation 或不同 chart 混合 |
| branch key   | radial/polar momentum sign、turning count、equatorial crossing/lensing order、winding | 区分同一 source anchor 的多像路径；具体最小集由反例缩减   |
| orientation  | parity 或可稳定复算 parity 的 sign                                                    | 阻止跨 critical curve 重建                                |
| continuous   | unwrapped source coordinate、travel/emission time、$g$                                | source lookup、slow-light 与 transport                    |
| differential | $J_q$ 或 conservative ellipse/singular values                                         | anisotropic filtering 与 adaptive sampling                |
| confidence   | event residual、Jacobian consistency、near-critical/near-seam flag                    | conservative accept/refine                                |

当前 [`surface_footprint_capture.wgsl`](../../crates/gravlume-render/src/shaders/surface_footprint_capture.wgsl)
已经用中心与真实 `±0.25 pixel` 四邻 ray、完整 production branch key、periodic $\phi$ unwrap 和
$(r,r_c\Delta\phi)$ central difference 产生 test-only Jacobian/parity；对应准入由
[`validation.md`](../validation.md#53-gpu-renderer-agreement) 定义。这修正了“当前完全没有 branch/Jacobian”的旧判断，
但不改变 production 缺口：production 仍没有持久化 semantic/footprint map、tile classifier、source
texture consumer 或 branch-aware reconstruct/fallback pass。shadow alpha 也仍只覆盖
Horizon/Escape silhouette，不能推出 surface branch continuity。

### 5.3 局部近似、filter 与 refinement 边界

五射线 exact-key equality 是必要条件，不是连续性的证明：有限 sparse stencil 仍可能在两采样点之间
漏掉 branch boundary、critical curve 或 source-domain edge。DNGR 明确说明 elliptical beam filter 假设
终端 ellipse 足够小、相邻 pixel 间 beam shape 变化不大；极端情况会出现 distortion、flicker 和 aliasing，
其处理是增加每 pixel beam 并 resample。[DNGR filtering](https://arxiv.org/html/1502.03808)
AART 同样指出 unresolved source/image feature 会被漏掉或经 interpolation 产生 spurious feature。
[AART resolution requirements](https://arxiv.org/html/2211.07469) 因而 accepted tile 至少还需通过：

- center/neighbor 的完整 exact/branch key 相同，且 periodic/chart transform 唯一；
- half-step、九点 stencil 或 child-tile 对 $J_q$、source displacement、$g$/time/transport variation 的
  nonlinear consistency 在预算内；
- determinant uncertainty interval 不含零，且样本 parity 不变；
- source-domain/shadow/chart seam 没有被 footprint support 触及；不能只检查 sample center。

若 source 是 filterable texture，WGSL
[`textureSampleGrad`](https://www.w3.org/TR/WGSL/#texturesamplegrad) 接受显式 `ddx`/`ddy`，没有
implicit derivative 的 fragment-only 限制，因此 compute reconstruction 可以把 $J_q$ 的两列直接作为
lookup gradient；显式 EWA 或 bounded supersampling 可用
[Heckbert 1989](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1989/5504.html) 作为独立 oracle。
WebGPU `maxAnisotropy > 1` 会启用平台支持的 anisotropic filtering，但实际值会被 clamp，精确过滤行为
是 implementation-dependent，且 min/mag/mipmap filter 必须全为 `linear`。
[WebGPU sampler](https://gpuweb.github.io/gpuweb/#dom-gpusamplerdescriptor-maxanisotropy) 因而
`textureSampleGrad + hardware anisotropy` 是性能/画质候选，不是可跨 Metal/Vulkan 固定的科学算法；
不能用它替代 scene-linear radiance oracle。

AART 还在 pure-Kerr、distant-observer、equatorial-source 的适用域内给出更强的 scale boundary：
第 $n$ 个 higher-order image 的 feature 尺度约按 $e^{-\gamma n}$ 缩小，image grid 需相应加密；
该 scaling 只是启发式，论文最终以所有 source/image
resolution 加倍后的 observable convergence 验证。[AART lensing-band resolution](https://arxiv.org/html/2211.07469)
该指数律不能直接外推到 Kerr–Newman 或任意 observer/source geometry，但足以否决“固定 tile/node
spacing 对所有 winding/lensing order 都无条件充分”；classifier 必须按 branch/order refine，或把
未解析的高阶 image 保守地送回 full trace。

先用 reconstruction pass 的邻近 full samples 建立该证据，比立即在每条 geodesic 中积分 geodesic-deviation tensor 更小。只有 finite difference 的额外 ray cost 或 near-critical failure 已被量化，才有理由把 analytic differential 加入 hot loop。

### 5.4 Production 候选数据流

在不引入公开 solver trait 或 render graph 的前提下，最小可证伪候选可以保持 private staged pipeline：

1. **Trace nodes：** 在 coarse/adaptive nodes 记录 packed exact branch key、continuous source/time/$g$、
   transport/quality signals；不要先承诺全分辨率多 plane。
2. **Classify tiles：** 带 halo 读取 nodes，做 exact-key、periodic unwrap、determinant uncertainty、
   curvature/half-step、source-domain 与 transport-variation gates；输出 accepted、refine 或 full-trace。
3. **Reconstruct/transport：** 仅对 accepted tile 在 consumer lookup chart 中构造 $J_q$、采样 source，
   并在 scene-linear domain 评价/积分 transport；refine/full-trace queue 在后续 dispatch 处理。

若用 atomic counter compact refine queue，producer dispatch 可以用 atomic 分配 slot 后写 payload，但
consumer 必须是第 4.3 节所述的后续 dispatch/pass；不得在同一 dispatch 的其他 workgroup 中读取。
coarse grid、AoS/SoA、packed key、8×8/halo、buffer/texture、dispatch fusion 或拆分都只是性能假设。
它们要在 correctness-approved workload 上比较总 trace + classify + reconstruct + fallback timestamp 和
resident/rebuild memory，不能把较少 dispatch、较高 accept ratio 或“连续内存”本身当作收益证据。

### 5.5 规范保证、假设与可证伪验收

| 结论                                                                                | 性质                                           | 不能外推的内容                                                         |
| ----------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------- |
| host-shared alignment/stride、requested limits 与每 dispatch 一个 usage scope       | WGSL/WebGPU/`wgpu 30` 保证                     | 不说明 bandwidth/residency，也不保证全分辨率 `vec4` planes 合理        |
| barrier 只同步同 workgroup；storage atomic 为 relaxed                               | WGSL 保证                                      | 不存在 portable same-dispatch grid barrier 或 counter→payload 发布协议 |
| 有序 dispatch/pass + 普通 binding tracking 形成 GPU producer/consumer 路径          | WebGPU usage-scope + `wgpu 30`/`wgpu-hal` 保证 | 不代表 backend 会 fusion，也不需要 CPU wait/手写 transition            |
| `textureSampleGrad` 接受显式 gradient；anisotropy 精确行为 implementation-dependent | WGSL/WebGPU 保证                               | 不保证与 explicit EWA、Metal 与 Vulkan 逐值一致                        |
| finite-difference ellipse 在 smooth regular branch 可近似 footprint                 | Igehy/DNGR 支持的局部模型                      | exact key 相同仍不能证明有限 tile 内无未采样 discontinuity             |
| coarse nodes、SoA、packed key、8×8 halo、hardware anisotropy 更快                   | 待测性能假设                                   | 只能由端到端 paired benchmark 接受或否决                               |

建议把下一轮 admission 固定为以下可复现 gates：

1. **ABI/limits：** Rust/WGSL offset、size、stride 与 canary round-trip 全等；在 requested limits 下分别
   对 3840×2160、4096×2160 做 resource plan，越限必须 tile/split/texture 或 typed reject；branch
   pack/unpack bit-exact。
2. **同步：** producer 写确定性 payload/counter，consumer 分别放在同 compute pass 的下一 dispatch、
   下一 pass、同 submit 的下一 command buffer；在 odd extent、workgroup boundary 与 multi-batch 上做
   checksum，Metal/Vulkan 重复运行均与 CPU expected 相等且无 CPU wait。counter→payload 的同 dispatch
   variant 即使偶然通过也不得成为 acceptance evidence；静态检查保证不存在这种 consumer。每个 shader
   barrier 都由 WGSL validation 证明处于 uniform control flow。
3. **classifier：** 命名 corpus 覆盖 smooth disk、disk/source edge、periodic/cubemap seam、至少两个
   winding/higher-order branch、critical curve 两侧；accepted pixel 必须全部通过更密 full-ray oracle 的
   branch exactness、source/$g$/time/transport variation 与 Jacobian/radiance budgets，任一 false accept
   都是 blocker。删除每项 gate 的 mutation 必须暴露固定 witness。
4. **scale convergence：** node spacing/finite-difference step 逐次减半并把 source/image resolution
   加倍；按 branch/lensing order 比 accepted mask、fallback mask、scene-linear radiance 与 temporal
   stability，直到预先固定的预算内收敛。不能用同一 sparse stencil 自证。
5. **filter/radiance：** high-frequency/checkerboard source、sharp disk/shadow edge、steep $g$ 与 optical
   depth gradient 分别比较 `textureSampleGrad`、explicit EWA/bounded sampling 与 4×/8× full-trace pixel
   integral；hardware anisotropy 只有在 Metal/Vulkan 都满足同一 observable budget 时才能进入该 profile。
6. **性能：** 在同一 correctness-approved scene/extent/profile 下记录总 GPU timestamp、fallback ratio、
   steady/rebuild memory，并与 all-full-KS baseline paired A/B；只有最终 observable 同预算且总时间/显存
   达到预先阈值，才接受任一 layout、tile 或 fusion 假设。

### 5.6 两个 reconstruction 候选的拒绝证据

在 `8145b24` 基线上做过两个未提交候选，均已从工作区删除。测试机为 Apple M5、Metal、integrated
GPU；固定 workload 是 1280×720 bolometric surface，命令为：

```text
cargo bench -p gravlume-render --bench trace_gpu \
  --features gpu-benchmarks --locked -- --noplot
```

两个候选都用独立 prepass 在不同 dispatch 生成 2-pixel-spacing semantic map，map node 是 8 个
`u32`（source $r,\phi,g,t$、packed branch/termination、winding、step count、event residual）。map node
直接替代同 pixel 的 full trace；reconstruction 只在 exact branch 相同、event phase 可接受且 parity
不退化时发生，其他 pixel 走原 full Cartesian KS。该布局在 1280×720 占 7,404,832 byte；即使它未触及
单 binding limit，端到端时间仍是决定性 gate。

1. **额外 edge validator 候选：** 对 cell edge 再发 full rays，并用它们校正 center。所有 accepted
   center 对独立 binary64 CPU reference 的最大误差为 source `4.076e-5 M`、$g$ relative
   `2.641e-6`、travel time `4.313e-5 M`，但 GPU 时间为
   `[36.094, 36.469, 36.850] ms`。correctness 通过不能抵消重复 geodesic 的成本。
2. **无额外 ray 的 stencil 候选：** 简单 bilinear 曾接受 30,863 个 center；1,517 个确定性分层、
   rejection-boundary 和 full-GPU-discrepancy 样本的 CPU oracle 给出最大 source `3.174e-2 M`、
   travel time `3.582e-2 M`、spectral budget ratio `4.395`，因此直接 false-accept。改为 3×3
   same-branch/same-step-phase、bilinear/biquadratic consistency 后只接受 3,735 个 center；1,389 个同类
   CPU 样本通过 source `1e-3 M`、$g$ relative `2.5e-4`、time `1e-3 M` 与 FP16 scene-linear
   spectral budget，但 GPU 时间仍为 `[32.233, 32.574, 32.908] ms`。

删除候选后，同机立即复测的 all-full-KS 基线为 `[19.765, 19.975, 20.191] ms`；Criterion 给出的
候选相对基线回归约 63%。这里的 CPU 分层样本只能支持“该候选未在样本中 false-accept”，不能替代第
5.5 节要求的按 branch/order 全域收敛；性能结果则已经足以否决候选，无需用更昂贵 oracle 为其建立
production 准入证据。

因此不能靠放宽 certificate 或把 validator 融进同一 megakernel 恢复该设计。重开条件是提出能显著减少
总 geodesic 数的 coarse/adaptive node、stationary transfer-map amortization 或等价方案，并重新通过
independent CPU/supersample correctness、Metal/Vulkan 总时间与 resource admission；不得只报告 accept
ratio、dispatch 数或相对另一个已失败候选的改善。

### 5.7 Surface full-KS 的 binary32 phase 边界

reconstruction 失败分析还暴露出一个与 interpolation 无关的边界：production radius-scaled RK4 的
默认 step scale `0.1` 只在 canonical v2/v3 pixel 通过连续 observable gate，不能据此声称整个 surface
frame 都满足同一预算。新增的 64×36 branch matrix 在 Schwarzschild、positive/negative-spin Kerr 与
Kerr–Newman 四个场景各分层抽取最多 24 个 surface pixel；terminal 与 radial/equatorial/winding key
全部和独立 CPU reference 精确一致。该测试只准入离散语义，不把连续误差隐藏在“branch 正确”之下。

同一矩阵的未提交连续量扫描以 `[source M, g relative, time M, radiance relative]` 记录最大值。默认
Schwarzschild profile 为 `[2.482e-2, 2.539e-4, 3.161e-2, 2.175e-3]`，超过 source、time 与
radiance gate。仅缩小 step 没有形成可采用解：scale `0.04` 的 source 已降到 `3.965e-3 M`，但 time
仍是 `5.473e-3 M`；`0.01` 的 time 为 `1.205e-3 M`；`0.008` 在 Schwarzschild 达到
`8.454e-4 M`，但 positive-spin-wide 仍为 `1.745e-3 M`；继续到 `0.005` 又因 binary32 trajectory
roundoff 退化为 `2.269e-3 M`。对正 coordinate-time increments 使用 compensated summation 没有改变
结果，说明主误差来自 trajectory/event phase，而不是最终标量累加。

这些 step 候选与 compensated hot-loop 均已删除，production 保持已基准化的 `0.1` policy；当前连续
GPU 证据仍严格限于 canonical v2/v3 fixture。恢复条件不是放宽 `1e-3 M`，而是更高阶/自适应且
binary32-stable 的 integrator、受限 analytic terminal solver 或显式 science-quality profile，并同时
通过这组矩阵、near-critical/branch-order corpus 和端到端性能测试。

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

| gate                | oracle / observable                                                                                       | 失败含义                                                      |
| ------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| special function    | 80+ bit $R_F/R_C/R_D/R_J$ 与 direct quadrature、DLMF identities；覆盖零参数、尺度变换、接近相等 arguments | evaluator 不得进入 accepted candidate                         |
| roots/topology      | 高精度 quartic roots、potential residual、root separation、turning sequence                               | uncertain 必须 fallback；false accept 为 blocker              |
| terminal            | 与独立 binary64 full-KS 比 termination、source anchor、travel time、$g$、event residual                   | potential identity 通过也不能替代 terminal observable         |
| critical ladder     | radial/polar double-root 两侧、positive/negative spin、near-axis、near-extreme                            | accepted error 或 branch flip 为 blocker；fallback 是预期成功 |
| classifier mutation | 删除一个 sign/root-separation/domain check 后必须触发固定反例                                             | 没有 witness 的 guard 需重新论证，但不能凭性能直删            |
| GPU A/B             | 相同 accepted set 和最终 observable 下，统计 analytic + compaction + fallback 总时间                      | “closed form”或低 fallback ratio 本身不是收益证据             |

当前 fixed-step Mino candidate 已因 accepted ray 的 travel-time 反例被拒绝，见 [`mino-step-selection.md`](mino-step-selection.md)。Carlson 路线的恢复条件不是继续扫一个更小 fixed step，而是用 root-aware integral 封闭 terminal phase，并在上述 gates 下证明 conservative acceptance。

## 7. 研究决策与恢复条件

| 候选                                               | 本轮决策                                          | 进入 production / 重开条件                                                                                                                                                                                     |
| -------------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| scalar invariant emission/absorption               | **已采用为 homogeneous path-integrated operator** | 空间变化 coefficient、ordered checkpoints 与通用 volume transport 仍需新的合同和独立 evidence                                                                                                                  |
| 对普通 RGB 直接应用 spectral redshift              | **拒绝**                                          | 引入具名 spectrum/bandpass 与频率轴后才可讨论；bolometric 继续只用 $g^4$                                                                                                                                       |
| 把 timelike circular existence 当 stable disk      | **拒绝**                                          | 独立 effective-potential/epicyclic gate 和 ISCO fixture 通过，且产品确实需要 stable disk 语义                                                                                                                  |
| 持久化完整 `GeometricSample` G-buffer              | **暂不采用**                                      | branch-aware reconstruction 证明有 consumer，并通过 layout、显存、误差与 Metal/Vulkan A/B                                                                                                                      |
| test-only source-space finite-difference footprint | **已有 ordinary-region 证据，不等于 production**  | 继续保留完整 branch key 与 CPU/GPU Jacobian gate；不得将五射线 equality 解释成全 tile 连续性证明                                                                                                               |
| branch-aware production reconstruction             | **当前 2-pixel map 候选已实验并拒绝**             | 换用能显著减少总 geodesic 的 coarse/adaptive 或 stationary-amortized 方案，再通过 requested-limit admission、跨 dispatch producer/consumer、按 branch/order 收敛、scene-linear supersample oracle 与端到端 A/B |
| 所有 Kerr/KN ray 全量改成 Carlson                  | **拒绝**                                          | 不存在 unsupported/ill-conditioned accepted ray，或始终保留可观测上等价的 KS fallback；端到端收益成立                                                                                                          |
| pure-Kerr exterior Carlson terminal accelerator    | **保留为受限候选**                                | root-aware classifier、80+ bit oracle、KS observable gate、false-accept sweep 与总 GPU A/B 全部通过                                                                                                            |

后续采用记录显示，标量 invariant transport、spectral/blackbody fixture、最小 branch key、production
单样本 inspection 与 test-only finite-difference footprint 已完成；第 5.6 节的 production reconstruction
候选未通过端到端 A/B，所以当前生产路径仍保持 full KS。下一项工作应在已通过的单样本 seam 上补更广
continuous-field quality baseline，并实现独立的 interactive/science-quality execution policy 支持域，而不放宽 observable budget；只有真实 filterable source 或 history
consumer 和基线同时存在后，才从新的 coarse/adaptive 或 stationary-amortized transfer-map 设计重开。
pure-Kerr root-aware Carlson 可以先建立 CPU oracle，但不是该产品闭环的前置条件。每一步只引入当下
consumer 需要的字段，并继续让 full-KS 定义 unsupported domain 的行为。

# 验证合同

本文定义 CPU reference 方法、最低 fixture 矩阵和首个版本化验收 profile。阈值是持续验收合同；当前 Rust reference 已覆盖的证据和仍未闭合的 oracle 梯级单独记录在 [Reference 实现与证据](reference-implementation.md)，不得把局部通过外推为整个矩阵已完成。

`baseline-v1` 是 Validation Profile；它引用 `reference-regular-v1` 与 `reference-strict-v1` 两个 Reference Policy。任一标识用于 artifact 后都不可原地改义；修正输入、阈值或枚举必须产生新 ID，并保留旧 fixture 的读取或明确拒绝策略。

## 1. CPU reference 方法

reference 固定使用 Dormand–Prince 5(4) 的七次求值 FSAL pair。令 $k_i=f(\lambda+c_i h,y+h\sum_{j<i}a_{ij}k_j)$；tableau 为：

|  $c_i$ | 非零 $a_{ij}$                                     |
| -----: | ------------------------------------------------- |
|    $0$ | —                                                 |
|  $1/5$ | $1/5$                                             |
| $3/10$ | $3/40,9/40$                                       |
|  $4/5$ | $44/45,-56/15,32/9$                               |
|  $8/9$ | $19372/6561,-25360/2187,64448/6561,-212/729$      |
|    $1$ | $9017/3168,-355/33,46732/5247,49/176,-5103/18656$ |
|    $1$ | $35/384,0,500/1113,125/192,-2187/6784,11/84$      |

第五阶与 embedded 第四阶 weights：

\[
b^{(5)}=\left(\frac{35}{384},0,\frac{500}{1113},\frac{125}{192},
-\frac{2187}{6784},\frac{11}{84},0\right),
\]

\[
b^{(4)}=\left(\frac{5179}{57600},0,\frac{7571}{16695},
\frac{393}{640},-\frac{92097}{339200},\frac{187}{2100},\frac1{40}\right).
\]

误差 $e=y^{(5)}-y^{(4)}$ 对每个 state group 使用 component-wise infinity norm：

\[
E_g=\max_{j\in g}
\frac{|e_j|}{\mathrm{atol}_g+\mathrm{rtol}_g\max(|y_j|,|y_j^{(5)}|)},
\qquad E=\max_gE_g.
\]

$E\le1$ 才提交 step。accepted step 的新 magnitude 为

\[
|h_{\rm new}|=|h|\operatorname{clamp}
\left(f_{\min},f_{\max},0.9E^{-1/5}\right),
\]

rejected step 令 $f_{\max}=1$；$E=0$ 使用最大 growth。Backward Trace 保持 $h<0$，controller 只改变 magnitude。$h_{\min},h_{\max}$、group tolerance、reject/step 上限和显式 initial step 都来自 versioned Reference Policy。

accepted step 上的 quartic dense output 定义为

\[
y(\theta)=y_n+h\sum_{i=1}^7 k_i\sum_{j=1}^4P_{ij}\theta^j,
\qquad 0\le\theta\le1,
\]

\[
P=\begin{pmatrix}
1&-8048581381/2820520608&8663915743/2820520608&-12715105075/11282082432\\
0&0&0&0\\
0&131558114200/32700410799&-68118460800/10900136933&87487479700/32700410799\\
0&-1754552775/470086768&14199869525/1410260304&-10690763975/1880347072\\
0&127303824393/49829197408&-318862633887/49829197408&701980252875/199316789632\\
0&-282668133/205662961&2019193451/616988883&-1453857185/822651844\\
0&40617522/29380423&-110615467/29380423&69997945/29380423
\end{pmatrix}.
\]

精确复算确认 tableau row sums、两组 weights、一阶 test equation 的 5/4 阶展开，以及 dense output 的两端值/导数和四阶内部展开。[A] 原始 pair 见 [Dormand–Prince 1980](https://doi.org/10.1016/0771-050X%2880%2990013-3)，连续延拓见 [Shampine 1986](https://doi.org/10.1090/S0025-5718-1986-0815836-3)；[SciPy RK45](https://github.com/scipy/scipy/blob/v1.18.0/scipy/integrate/_ivp/rk.py) 是同系数的维护实现对照。[P]

实现还必须满足：

- state groups（position、momentum、polarization/transport）有独立 `atol + rtol * scale`；
- embedded error 先按 group 归一化，再取最大，不把大小不同的坐标平均；
- 只有 accepted step 提交状态和 transport side effect；rejected step 不计物理 step；
- safety、growth bounds、`h_min/h_max`、max rejects/steps 全部进入 policy/artifact；
- dense output 负责 event bracket/localization；
- sample 外层可用专用 Rayon pool，单条轨迹保持确定顺序，结果按 input index 排列。

CPU `f64` 不是绝对 ground truth。reference ladder 至少包含：

1. exact algebra/special limits；
2. 80-bit 或更高精度 spot check；
3. tolerance/step-halving convergence；
4. 独立 chart/state representation；
5. published trajectory 或作者实现；
6. GPU field agreement。

## 2. 最低 fixture 矩阵

| 类别                   | 必须覆盖                                                                                                                                                       |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| algebra                | radius quadratic、$l^2=0$、metric inverse、horizon roots、special limits                                                                                       |
| observer               | Minkowski tetrad、boost、ergoregion rejection、Fermi–Walker analytic cases                                                                                     |
| Schwarzschild          | weak deflection $4M/b$、photon sphere $3M$、shadow $\sqrt{27}M$、near-critical escape/capture pairs                                                            |
| Kerr                   | prograde/retrograde equatorial paths、axis/pole、published trajectories、Carter drift                                                                          |
| Kerr–Newman            | $q_e\to0$、RN、extremal/superextremal classification、exterior orbit samples                                                                                   |
| events                 | escape/horizon/disk/singularity/step exhaustion、same-step competing event                                                                                     |
| radiation              | vacuum, pure absorption, constant slab, $g^3/g^4$, blackbody temperature shift                                                                                 |
| polarization           | orthonormal screen basis、gauge transform、parallel transport、EVPA                                                                                            |
| numerical conditioning | raw-radius 与 reciprocal-radius candidate reproducer：完整初值、step/update/event/termination、浮点模式、checkpoints、high-precision oracle 与 expected branch |

每个 fixture 写明 schema、coordinate/chart、signature、mass normalization、initial observer/frame、photon orientation、solver/precision、expected observable、tolerance 和 source。没有这些 metadata 的图像不是科学 fixture。

第 3–7 节定义首个 machine-readable schema、Reference Policy、默认 Observation 与具名阈值。

## 3. `kerr-exterior-observation-v1`

首个纵向闭环不依赖外部资产：geometry 阶段使用解析方向天空；transport 阶段增加一个规定发射率的薄表面。所有长度先以 $M$ 无量纲化。

| 字段           | v1 值                                                                                     |
| -------------- | ----------------------------------------------------------------------------------------- |
| spacetime      | ingoing Cartesian Kerr–Schild，$M=1,a=0.8,q_e=0$                                          |
| extremality    | subextremal，$r_+=1.6M$                                                                   |
| observer event | $t=0$；oblate $(r,\theta,\phi)=(30M,\pi/3,0)$                                             |
| observer state | stationary $u=\partial_t/\sqrt{-g_{tt}}$；仅因 $g_{tt}<0$ 有效                            |
| target/up      | target 为原点；up hint 为 spin $+z$                                                       |
| viewport       | perspective，$1280\times720$，vertical FOV $45^\circ$，左上原点                           |
| sample         | pixel index + subpixel offset；无 jitter 时 $(\delta_x,\delta_y)=(0.5,0.5)$               |
| photon scale   | $\omega_{\rm obs}=1$                                                                      |
| outer event    | $R_{\rm esc}=200M$；finite-boundary 误差单独计入                                          |
| display        | scene-linear capture；有限且非负的 $x$ 在 exposure 0 后逐通道作 $x/(1+x)$ 与 SDR transfer |

默认 ingoing oblate-to-Cartesian 变换得到

\[
(x,y,z)=(25.98076211353316,
0.6928203230275509,15)M.
\]

负 radiance 或 non-finite 不进入该 display function：预览显示 diagnostic；scene-linear capture 保留 raw channels、texel kind 与 failure classification。path observation fields 不属于这项 display capture 合同，任何输出都不能先 clamp 再解释为物理结果。

80 位复算恢复 $r=30M$，并给出

\[
g_{tt}=-0.933345183078563810878066121578.
\]

按[数学物理合同第 5.1 节](physics.md#51-viewport-sample-与初始光线)构造的 $(u,e_R,e_U,e_A)$ Gram residual 小于 $3\times10^{-80}$，orientation determinant 为 $+1$；中心与四角初始 Photon Momentum 的 null/frequency residual 小于 $10^{-70}$。[N] 这些高精度数值只证明连续构造自洽，不替代未来 Rust/WGSL fixture。

### 3.1 Geometry scene

Physical Scene 无 emitter 时 GPU 使用无 emission/absorption 的 analytic sky。sky radiance 只是定位方向与 seam 的测试图，不解释为光谱；方向主轴与不同空间频率必须可从 Source Anchor 复原。几何比较使用 termination、escape direction、travel time 和 invariant，不使用 tone-mapped RGB 作为 oracle。默认桌面 scene 在同一 analytic escape environment 前安装第 3.2 节的 emitter。

### 3.2 Surface observable

reference surface fixture 增加 equatorial surface：

- $r\in[6M,20M]$，prograde circular emitter；
- comoving bolometric intensity $I_{\rm em}(r)=I_0(r/6M)^{-3}$；
- vacuum between surface and observer，无 absorption；
- observed bolometric intensity $I_{\rm obs}=g^4I_{\rm em}$。

该 emitter domain 是 Physical Scene；`baseline-v1` trace 的 applicability 另要求
$r_{\rm out}<R_{\rm esc}=200M$。触碰或越过数值 escape boundary 的合法 Physical Scene 必须在
CPU/GPU trace seam 返回 typed error，不能静默先终止为 Escape。

这是规定的 neutral surface-radiance fixture，不声称 Novikov–Thorne/Page–Thorne accretion
physics；不得为让默认图更漂亮而改变这一合同。

版本化 artifact 位于 [`fixtures/v2/kerr-surface-observable.toml`](../crates/gravlume-reference/fixtures/v2/kerr-surface-observable.toml)，引用 v1 canonical Observation 而不复制 scene 参数。它固定 source、image sample、emission profile、strict expected observable 与 tolerance；v1 schema/profile 不原地扩展。CPU regular/strict reference 先独立闭环；GPU full Cartesian Kerr–Schild test 再按本页 GPU gate 比较 termination、Source Anchor、Frequency Ratio、travel time 与最终 `RGBA16F` radiance。

`surface-transport-v1` 使用独立 schema v3，保持同一 geometry sample，另固定：

- diluted-blackbody source：$T_6=6000\,\mathrm K$ 与数学物理合同的 $T(r)$；
- observer-frame `visible-boxcar-v1` 三 bands；
- vacuum、$\tau=0.75$ pure absorption、$\tau=0.35$ constant blackbody source，以及
  $\tau=0$ pure blackbody emission 四种解析边界；
- emitted/vacuum/final bolometric intensity、emitted/observer temperature、optical depth 与三 band
  final intensity。

四份 canonical artifact 位于 [`fixtures/v3/`](../crates/gravlume-reference/fixtures/v3/)。几何
expected 来自 strict reference convergence；transfer identity、Planck normalization、band integral
与 cancellation-sensitive limit 由 80 位独立计算生成。CPU regular/strict 均须接受 artifact；GPU
比较最终 `RGBA16F` scene-linear bands，不能经过 tone mapping。

## 4. Reference Policy

`reference-regular-v1` 使用第 1 节的 DP5(4)，所有状态在 pack 前保持 `f64`。输入已经以 $M=\omega_{\rm obs}=1$ 归一化；直接尺度输入 $M$ 必须精确为 1，派生的 tetrad contraction $\omega_{\rm obs}$ 按具名浮点预算验证。

| policy field                 |                                     v1 值 |
| ---------------------------- | ----------------------------------------: |
| position/time `rtol`         |                                   `2e-12` |
| position/time `atol`         |                                   `2e-13` |
| covariant momentum `rtol`    |                                   `2e-12` |
| covariant momentum `atol`    |                                   `2e-13` |
| initial step magnitude       |                                  `1e-3 M` |
| minimum step magnitude       |                                $2^{-40}M$ |
| maximum step magnitude       |                                   `0.5 M` |
| safety / shrink / growth     |                         `0.9 / 0.2 / 5.0` |
| max accepted steps           |                                  `200000` |
| max consecutive rejects      |                                      `64` |
| dense-event affine tolerance |                                 `2e-11 M` |
| event tie tolerance          |                                 `5e-11 M` |
| event arming band            | `1.28e-9` normalized event-function units |
| singularity guard            |                $(r^4+a^2z^2)/M^4=2^{-40}$ |

`reference-strict-v1` 把所有 `rtol/atol` 除以 16、maximum step 降为 `0.25 M`、event tolerance 除以 4、step/reject 上限加倍。regular fixture 必须同时运行两个 policy；“baseline 成功”而 strict 改变 branch/termination 时，baseline 失败，不选择更好看的结果。

起点位于 event surface 时 event 初始 unarmed。horizon/singularity 在 $F>b_{\rm arm}$、escape 在
$F<-b_{\rm arm}$、双向 equatorial surface 在 $|F|>b_{\rm arm}$ 后 armed；一旦 armed 不再因返回
band 而清除。fixture 声明的初始 armed 状态必须由初始 canonical state、event function 和 arming
band 推导并一致；未安装对应 event 时不得携带该声明。Reference Outcome 总是记录
accepted/rejected steps、实际 min/max step、event bracket/residual 和触发的资源上限。

## 5. 验收预算

### 5.1 Algebra 与初始数据

| observable                                        | CPU `f64` | WGSL `f32` |
| ------------------------------------------------- | --------: | ---------: |
| normalized radius polynomial residual             |   `2e-13` |     `3e-5` |
| normalized $l^2$ / metric-inverse max residual    |   `2e-12` |     `5e-5` |
| Observer Frame max Gram residual                  |   `2e-12` |     `8e-5` |
| Viewport initial-ray angular CPU/GPU disagreement |         — | `2e-6 rad` |
| normalized initial null/frequency residual        |   `2e-12` |     `8e-5` |

Residual 的 denominator 必须是量纲匹配的 term norm 与 `1` 的最大值，不以另一个可能同时错误的实现作归一化。exact axis、equator、horizon-near 和 far-field 分桶报告，不只给全局平均。

### 5.2 Reference agreement

regular fixture 的 baseline/strict comparison 必须满足：

| observable                                   |             v1 gate |
| -------------------------------------------- | ------------------: |
| termination                                  | exact enum equality |
| surface branch key                           |      exact equality |
| escape-direction angle                       |          `2e-9 rad` |
| localized event position                     |            `2e-9 M` |
| Frequency Ratio relative error               |              `2e-9` |
| Source Anchor surface coordinate             |            `2e-9 M` |
| travel-time absolute error                   |            `2e-8 M` |
| normalized null/$E$/$L_z$/$\mathcal Q$ drift |         each `5e-9` |

escape direction 是 localized escape state 上 Hamilton RHS 空间分量按实际 affine traversal 符号取向后的单位 coordinate direction；它不是 terminal position 的径向单位向量。方向无法求值时 comparison 失败，不能省略该 gate。

equatorial Source Anchor 以数学物理合同定义的 $(r,\phi_s)$ 比较。surface-coordinate distance 固定为

\[
\operatorname{hypot}\!\left(\Delta r,\frac{r_1+r_2}{2}\operatorname{wrap}(\Delta\phi_s)\right),
\]

并使用上表的 `2e-9 M` gate；任一 surface outcome 缺少 anchor 或 Frequency Ratio 时 comparison 失败，不能把 `None/None` 当作一致。

near-critical fixture 不套一个全局 angle tolerance。它必须给成对的 escape/capture 或两侧 branch 样本、distance-to-critical 标签和独立高精度 observable；discrete classification 必须正确，continuous tolerance 由 fixture 自身给出。

surface branch key v1 固定为 initial polar side、已提交 radial turning count、已提交 equatorial
crossing count 与 signed azimuth winding。若 terminal 位于一步内部，只提交 terminal fraction 之前的
crossing；rejected step 不改变 key。regular/strict key 不同即 comparison 失败，即使最终 source
coordinates 接近。

### 5.3 GPU renderer agreement

下列是当前 GPU 实现的初始、可否证门槛 `[X]`；实现数据只能通过新 profile 调整，不能原地放宽：

`gpu-ks-rk4-v1` 以 $M$ 无量纲化，固定 radial step scale `0.1`、step range
`[0.005M,8M]`、最多 `2048` steps、event tie tolerance $2^{-17}M$ 与 equatorial surface
arming band $2^{-12}M$。`gpu-ks-rk4-v2` 保留该 base policy 与全部 v1 gate；若无歧义 base
surface terminal 的 Source Anchor 距任一 source radial edge 不超过 `0.25M`，则从同一 canonical
initial state 重追，使用 radial step scale `0.0025`、step range `[0.000125M,0.25M]` 和相同
`2048` step 上限，只以第二次完整结果作为 v2 outcome。该条件重追属于新 method identity，不改变
v1 profile 的含义；重追若不能给出满足同一 gate 的确定结果，仍须返回 typed uncertainty/failure。

两个 profile 的 tie 都比较 localized fractions 的差乘当前 affine step magnitude；candidate 以
singularity → horizon → emitter → escape bit order 保存，ambiguity 独立保存。arming 是 per-ray
sticky state，只在已提交 endpoint 离开 band 后置位。Equatorial crossing 属于 branch observable，
不因 scene 未安装 surface 而停止计数；arming 只控制 crossing 能否成为 surface terminal。

| observable                                            |                                             v1/v2 gate |
| ----------------------------------------------------- | -----------------------------------------------------: |
| regular termination / branch                          |                               exact reference equality |
| regular escape/source angular error                   | ≤ `0.35` pixel footprint；本 viewport 为 `3.82e-4 rad` |
| regular travel-time absolute error                    |                                               `1e-3 M` |
| recorded normalized null/$E$/$L_z$/$\mathcal Q$ drift |                                            each `0.05` |
| Frequency Ratio relative error                        |                                                 `2e-3` |
| surface event position                                |                                               `5e-3 M` |
| bolometric surface `RGBA16F` relative error           |                                                 `2e-3` |
| final spectral surface band relative error            |                                                 `4e-3` |
| numerical failure on regular matrix                   |                                                    `0` |
| stale history after generation/resize/cut             |                                   `0` accepted samples |

GPU universal path 动态积分六个 canonical spatial variables，并把 per-ray $E=-p_t$ 作为构造上
不变的常量；coordinate time 用同一 RK stages 的相对 increment 累计，因此共同平移 observer
coordinate-time origin 不得改变任何 trace observable。GPU Carter drift 使用不借助 $H=0$ 的
全域 Cartesian 形式，axis 不设 epsilon seam；CPU reference 故意保留完整八维 RHS、full null
Jacobian 与独立 trigonometric Carter evaluator。两边不同计算图的一致性才构成验证，不能把同一
优化公式复制到 oracle 后称为独立通过。

event localization 保留 endpoint bracket 与 priority。只有 guard cubic 的三个 Bézier derivative
control values 证明单调、且当前 derivative 高于 binary32 conditioning floor，才用固定六次
safeguarded Newton；否则使用旧 chord fraction。测试 gate 比较 localized state 上重新求值的真实
event residual，而不是只比较 cubic polynomial residual。

落入 near-critical uncertainty band 的 sample 只有经过满足同一预算的 refine 才能输出确定 branch；没有这样的第二遍求解时必须保持 `Uncertain`，数值失败则输出 `NumericalFailure`。null/Carter drift 是诊断与 classifier 输入，不可单独替代 observable agreement。

`visible-boxcar-v1` 的 4097-entry LUT 以 $\log_2T\in[-8,24]$、每 octave 128 intervals 保存并线性
插值 $\log_2$ band fraction。shader 将 fraction 的 significand/exponent 与 bolometric intensity、
几何 transport 合并，只有完整 radiance 的 exponent 已知后才物化线性值并交给 `RGBA16F`
rounding；不得先把微小 fraction 截断为零。相对 80 位 Planck oracle，全部 midpoint 重建 fraction 的
absolute error 必须不超过
`3e-6`；当 expected fraction 至少为 `1e-6` 时 relative error 不超过 `2e-3`。最终 spectral fixture
同时包含 geometry/transport 与 FP16 rounding，因此使用上表 `4e-3` gate，而不是把 LUT 单项预算
误作总误差。

surface-footprint GPU capture 是 test-only 证据路径：中心与真实 `±0.25 pixel` 四邻域必须全部为
无歧义 surface terminal，且完整 branch key exact equality，才输出 source chart
$(r,r_c\Delta\phi)$ 的 central-difference $J$ 与 parity。canonical ordinary sample 使用更细的
diagnostic step policy，GPU/CPU Jacobian 的 matrix max-norm relative error gate 为 `3e-3`；任一
semantic mismatch 输出 discontinuity。该证据不等于 production reconstruction 已实现。

离散 surface branch matrix 另覆盖 Schwarzschild、正/负自旋 Kerr 与 Kerr–Newman 四个 64×36
profile，每个 profile 对最多 24 个分层 surface pixel 要求 CPU/GPU terminal、radial turnings、
equatorial crossings 与 signed winding exact equality。该 matrix 不改变上表连续预算，也不把尚未通过
同预算的非 canonical source/time 样本登记为连续 agreement evidence。

presentation accelerator 不另设宽松容差。任何 analytic/Mino candidate 的 accepted ray 都必须通过同一 termination/direction/travel-time gate；potential/reciprocal constraint 只是额外 condition signal，不能代替 observable。已拒绝的 fixed-step Mino candidate 正是因为高分辨率 accepted ray 越过 travel-time budget。后续 elliptic/Carlson variant 至少覆盖正/负 spin、近场高绕转、critical 两侧、near-axis、near-extreme 与 unsupported-domain fallback；parameter sweep 只属于研究 artifact，不进入常规测试。

## 6. Fixture envelope

fixture 使用 UTF-8 TOML，未知字段和未知 enum 一律拒绝。`schema_version` 管结构，`profile` 管阈值；两者不能互相替代。输入/期望的高精度十进制以字符串保存，解析器必须明确转换精度，不能先经 `f64` 再声称 80 位来源；固定 canonical preset 的十进制原文是规范 artifact，必须在舍入到 `f64` 前精确核对。

v1 保存 geometry/Observation，v2 保存 neutral surface observable，v3 保存 blackbody/slab transport；
旧 schema 的字段与 profile 含义不原地扩展。每个 parser 只接受仓库内具名 canonical preset，修改
任一 expected、producer 或 applicability 字段都必须以新 artifact identity/version 进入。

```toml
schema_version = 1
profile = "baseline-v1"
id = "schwarzschild-scatter-b6-v1"
kind = "geodesic"
evidence = "numeric"

[producer]
method = "80-digit direct-r and reciprocal-u quadrature"

[spacetime]
family = "kerr-newman"
chart = "ingoing-cartesian-kerr-schild"
mass_m = "1"
spin_m = "0"
charge_m = "0"

[initial]
position_txyz_m = ["0", "50", "0", "0"]
momentum_covariant = ["-1", "...", "0.12", "0"]
affine_direction = "positive"

[expected]
termination = "escape"
turning_radius_m = "..."
azimuth_advance_rad = "..."

[tolerance]
turning_radius_abs_m = "5e-11"
azimuth_advance_abs_rad = "5e-10"
```

必需字段：

- envelope：schema/profile/id/kind/evidence；
- producer：方法、precision、独立来源或推导；
- convention：chart、signature、component order、mass/frequency normalization；
- initial：完整状态、integration direction、event arming；
- expected：typed terminal 与具名 continuous observables；
- tolerance：每个 continuous observable 独立 abs/rel 规则；
- applicability：regular/near-critical、参数域和不适用条件。

持久化 enum 名称是 versioned protocol，不由 UI label 或 `strum` iteration order 生成。NaN/Inf、负零和未声明单位在 input seam 拒绝；同一 logical label 若绑定到不同 canonical bits，comparison 返回 identity collision，不能进入数值验收。artifact 额外记录 producer revision、policy、dtype、adapter/backend、shader digest 与实际 resource counters。

## 7. 80 位 Schwarzschild 基准

三个 fixture 均取 binary64 bits 精确的 $M=E=1$、equatorial、$r_0=50M$ 和 future-directed inbound state，沿正 affine parameter 积分；直接输入不得用 epsilon 接受另一 affine 尺度。$b>b_c=\sqrt{27}M$ 的两条 ray 在 exterior turning point 后返回 $r=50M$；$b<b_c$ 的配对 ray 穿过 $r_+=2M$。radial potential 是

\[
R(r)=r^4-b^2r^2+2b^2r.
\]

对 scattering ray，分别用 $r=r_{\rm turn}+s^2$ 与 $u=1/r=u_{\rm turn}-s^2$ 消除端点平方根奇性；两种 80 位积分的 $\Delta\phi$ 差为 $5.62\times10^{-41}$ 和 $9.05\times10^{-41}$。capture ray 的 direct-$r$/reciprocal-$u$ 积分相差 $2.90\times10^{-55}$。[N]

|                 $b$ | route                       |                            $r_{\rm turn}/M$ |                                $\Delta\phi$ |
| ------------------: | --------------------------- | ------------------------------------------: | ------------------------------------------: |
|                 $6$ | $50M\to r_{\rm turn}\to50M$ | `4.453363193811354931623303616376185063746` | `4.620418726881239063416504280864374022974` |
| $\sqrt{27}+10^{-3}$ | $50M\to r_{\rm turn}\to50M$ | `3.034497655706080454343638050694227224458` | `11.08947404451632010196559671422843740259` |
| $\sqrt{27}-10^{-3}$ | $50M\to r_+=2M$             |                                           — | `9.875337813686692580530011972416581225191` |

对应 ingoing Cartesian Kerr–Schild 初始 covector 以 80 位十进制存档，并直接代入 $g^{\mu\nu}p_\mu p_\nu$；按文件中的实际十进制重算，absolute residual 依次为 $4.91\times10^{-81}$、$3.17\times10^{-81}$ 与 $3.66\times10^{-81}$。[fixture files](../crates/gravlume-reference/fixtures/v1/) 是独立、可复算的数据合同。[N]

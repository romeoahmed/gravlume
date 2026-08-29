# Source-edge 九点 corpus 高精度 BL/Mino witness

本文记录 `kerr-exterior-observation-v1` 上跨 outer source edge 的固定九点 corpus 的独立高精度见证、可复算方法、失败边界与后续候选。它只是一份 research decision/evidence record，不定义 production interface、Validation Profile、fixture schema、GPU method identity 或支持域；这些合同仍分别以[数学物理](../physics.md)、[验证合同](../validation.md)、[路线图](../roadmap.md)和[连续字段 corpus 记录](continuous-field-corpus.md)为准。

**状态：固定九点 corpus 的 semantic witness 已采用；统一 texture gate、持久 artifact 与其他 strata 未闭合。** 当前方法认证 center-subpixel `(640,12..20)`：`y=12,13` 为 Escape，`y=14..20` 为 ordinary surface，且 `(640,16)` 与 v2 canonical case 重合。它不能授权其他 observation、Kerr–Newman、near-critical/axis/extreme case、production BL/Mino solver 或 WGSL fast path。

## 1. 可否证假设

首切片检验以下命题：

1. 从规范十进制 Observation 独立重建的 future-directed Photon Momentum，可在 ingoing Cartesian Kerr–Schild 与 Boyer–Lindquist canonical covector 之间保持 $E,L_z,\mathcal Q$ 和 null constraint；
2. pure Kerr separated radial/polar potentials 能给出与 Cartesian KS reference 相同的 discrete path identity，而不是只恢复接近的 terminal position；
3. 对七个 simple-root surface case，分 turning segment 的 100+ decimal-digit quadrature 能独立恢复 Source Anchor、Frequency Ratio、KS coordinate-time duration 与 surface radiance；
4. 对两个 outer-edge 外侧 case，第一次 equatorial crossing 的 signed radial margin、Escape 与下一 crossing 的 Mino-order margin、localized KS position、negative-affine traversal direction 和 travel time 能在同一 separated graph 中闭合；
5. 120/180 decimal-digit 重算、turning-root conditioning 与两种 KS chart primitive evaluation 能把数值误差压到 validation gate 之下；
6. 若上述任一 discrete identity、root/event order、constraint 或 precision-doubling gate 失败，该 case 必须保持 unsupported，不能用 CPU/GPU agreement 替代。

Hamilton–Jacobi separability 与第四常数来自 [Carter 1968](https://doi.org/10.1103/PhysRev.174.1559)；affine/Mino reparameterization 来自 [Mino 2003](https://doi.org/10.1103/PhysRevD.67.084027)；real Kerr null-geodesic root topology 与分段积分对照 [Gralla–Lupsasca 2020](https://doi.org/10.1103/PhysRevD.101.044032)。项目的物理自旋、chart handedness 和 covector transform 已由 [KS↔BL/Mino seam](kerr-schild-mino-map.md) 单独封闭。

## 2. 与 Cartesian KS reference 的独立性

可复算实现为 [`bl_mino.py`](scripts/src/gravlume_research/checks/bl_mino.py)。它：

- 不导入 `gravlume-domain`、`gravlume-reference` 或 renderer；
- 不读取 Rust reference outcome、GPU record、texture 或 fixture expected；
- 从 `M=1,a=0.8,q_e=0`、observer oblate event、stationary observer、target/up、viewport/FOV 与 pixel/subpixel 十进制输入重新构造 KS metric、Observer Frame 和 camera covector；
- 用 one-form invariance 独立转到 BL covector，再形成 $E=-p_t$、$b=L_z/E$ 与 $\eta=\mathcal Q/E^2$；
- 在 separated BL/Mino graph 中求 terminal；只在最后把 endpoint observable 转回 ingoing KS chart；
- 用 mpmath arbitrary precision、precision doubling 和等价 evaluation 自己给出误差证据。

Observer Frame 的 image-right complement 虽可从不同 coordinate seed 构造，但在给定 $u$、sight、up 与 orientation 后只有一个一维正向解；因此脚本不共享 Rust 的浮点 Gram–Schmidt 中间值。

## 3. Separated dynamics 与 branch

令

\[
P=r^2+a^2-ab,\qquad A=(b-a)^2+\eta,
\]

\[
R(r)=P^2-\Delta A,
\qquad
U(\mu)=\eta+(a^2-\eta-b^2)\mu^2-a^2\mu^4,
\]

其中 $\mu=\cos\theta$、$\Delta=r^2-2r+a^2$。Energy-rescaled Mino parameter 满足

\[
\left(\frac{dr}{d\tau}\right)^2=R,
\qquad
\left(\frac{d\mu}{d\tau}\right)^2=U.
\]

脚本先从 initial Hamilton tangent 决定 radial/polar sign，再分类 quartic roots；不从 backward traversal、position 或 `signum(0)` 猜 branch。固定 corpus 的 exact discrete identity 为：

| 字段                                 | `(640,12..13)` outside | `(640,14..20)` inside |
| ------------------------------------ | --------------------- | ---------------------- |
| terminal                             | Escape                | surface                |
| initial polar side                   | `positive`            | `positive`             |
| radial turnings                      | `1`                   | `1`                    |
| polar turnings                       | `1`                   | `1`                    |
| equatorial crossings before terminal | `1`                   | `0`                    |
| signed azimuth winding               | `0`                   | `0`                    |

Terminal surface crossing 本身不计入“before terminal”的 crossing count，与 Reference branch-key 提交语义一致。Polar turning 是独立 witness 保存的 path identity；当前 Rust branch key 没有该字段。

## 4. Turning regularization

直接把 `1/sqrt(R)` 或 `1/sqrt(U)` 交给 generic quadrature，会在 simple turning endpoint 形成可积但条件不良的平方根奇点；把 working precision 调高并不能消除表达式先发生的 `0/0` 消减。

### 4.1 Polar segment

对当前 root topology，

\[
U(\mu)=a^2(\mu_+^2-\mu^2)(\mu^2-\mu_-^2),
\qquad \mu_-^2<0.
\]

每个到 turning 的 segment 使用 $\mu=\mu_+\sin\chi$。Jacobian 精确消去 $\sqrt{\mu_+^2-\mu^2}$，得到端点有限的 integrand。Surface case 累计 Observer→turn→equator；Escape case 再以同一 regularized variable 累计 equator→negative-$\mu$ endpoint，均保留一个 polar turning。

### 4.2 Radial segment

先用 quartic roots 找到 observer 下方最近 exterior simple root $r_t$，再 refine $R(r_t)=0$。Synthetic division 直接形成

\[
Q(r)=\frac{R(r)}{r-r_t},
\]

避免在 $r\approx r_t$ 时相减近等大项。令 $r=r_t+s^2$ 后，

\[
\frac{dr}{\sqrt{R(r)}}=\frac{2\,ds}{\sqrt{Q(r_t+s^2)}},
\]

端点同样有限。Source→turn 与 turn→observer 两段相加，保留一个 radial turning。Simple-root conditioning 以 $|R'(r_t)|$ 和 $|U'(\mu_+)|$ 单独报告；它们不是 potential 的正最小值，因为 potential 在真实 turning 处必须为零。

mpmath 官方文档说明 `quad` 的默认 tanh-sinh 对 endpoint singularity 通常更稳，并明确建议通过提高 precision 检验此类积分；本实现仍先解析 regularize，再使用 quadrature，而不把 quadrature rule 当作 root certificate。[mpmath quadrature](https://mpmath.org/doc/1.3.0/calculus/integration.html)

## 5. Phase、chart 与 source observable

BL phase 使用 separated integrals：

\[
\frac{dt_B}{d\tau}
=\frac{(r^2+a^2)P}{\Delta}+a(b-a)+a^2\mu^2,
\]

\[
\frac{d\phi_B}{d\tau}
=\frac{aP}{\Delta}+\frac{b}{1-\mu^2}-a.
\]

Ingoing KS endpoint relation 为

\[
dt_s=dt_B+\frac{2r}{\Delta}dr,
\qquad
d\phi_s=d\phi_B+\frac a\Delta dr.
\]

Chart shift 只依赖 endpoints；radial turning 的往返部分精确抵消。脚本同时以直接 quadrature 和 subextremal horizon-log primitive 计算这两个 shift，并报告 normalized disagreement。Source Frequency Ratio 和 bolometric radiance 按[数学物理合同](../physics.md#7-frequency-与-radiative-transfer)独立计算，不从 fixture RGB 或 GPU output 反推。

九点 corpus 把每个 terminal radial value 映射为相对 $20M$ outer edge 的 signed margin，并要求随 `pixel_y=12..20` 严格递增；`y=13/14` 的相邻负/正 margin 给出分类 bracket。两个 outside case 还必须证明 crossing 位于 $20M$ 之外，且 $R_{esc}=200M$ 先于下一次 equatorial crossing；随后把 BL endpoint tangent 转为 ingoing Cartesian KS，并按 reference 的 negative-affine traversal orientation 归一化 Escape direction。Position、direction 与 travel time 分别验收，不用 preview RGB 替代。

## 6. 认证条件与 120/180 十进制位结果

研究环境要求 Python 3.14 或更高版本；精确依赖版本以 [`pyproject.toml`](scripts/pyproject.toml)与 [`uv.lock`](scripts/uv.lock) 为准，不把一次复算所用的 Python patch 版本写成算法合同。每次结果必须先满足以下认证条件：

- terminal、initial side、turning/crossing counts 与 winding 精确等于第 3 节的对应 discrete identity；
- 所有连续字段为 real finite；surface source 位于 $[6M,20M]$，Escape position 位于 $R_{esc}$、direction normalized/outward，edge 与 event-order margin 符号正确；有物理正号要求的 observable 与 conditioning 严格为正；
- initial null、Mino constraint 与 chart primitive 的 normalized residual 均满足

\[
\rho < 10^{-(p-15)},
\]

其中 $p$ 是 working decimal digits，15 位作为 guard。任一条件失败都不给出通过证书。mpmath precision context、finite 检查与数值比较的实现依据集中在[一手来源](#11-一手来源)，私有 Python record、函数名与测试结构不属于研究结论。

### 6.1 Corpus precision certificate

120 与 180 decimal-digit 两次完整重算先各自通过 exact corpus identity、signed-margin ordering、event order、finite/physical-value、turning-root 与 residual gate，再逐 case 比较全部 source/transfer/phase fields、Escape vector lanes、constants of motion 和 turning derivatives。整个 corpus 的最大 normalized delta 为 `3.85612445201e-94`，超过要求的 80 stable decimal digits。

180 位运行的 terminal radial field、outer-edge signed margin 与 travel time 如下；表格只显示供审阅的有效位，CLI 会输出主要 source/escape/transfer/phase observable 的 110 位文本：

| pixel | terminal | crossing/source radius $/M$ | signed margin $/M$ | travel time $/M$ |
| ----- | -------- | --------------------------: | -----------------: | ----------------: |
| 12    | Escape   | 20.164713144839971847       | -0.164713144839971847 | 238.406047718117012 |
| 13    | Escape   | 20.035241577257517150       | -0.035241577257517150 | 238.438694378676361 |
| 14    | surface  | 19.906414902636657675       | 0.093585097363342325  | 55.111445736567960  |
| 15    | surface  | 19.778228798382417869       | 0.221771201617582131  | 55.006599298066207  |
| 16    | surface  | 19.650678984603292402       | 0.349321015396707598  | 54.902474247630054  |
| 17    | surface  | 19.523761223635148545       | 0.476238776364851455  | 54.799067017429487  |
| 18    | surface  | 19.397471319572138440       | 0.602528680427861560  | 54.696374090040828  |
| 19    | surface  | 19.271805117804512272       | 0.728194882195487728  | 54.594391998135353  |
| 20    | surface  | 19.146758504563225500       | 0.853241495436774500  | 54.493117324177979  |

七个 surface case 的其余 consumer observable 为：

| pixel | source azimuth / rad | frequency ratio | emitted intensity | observed intensity |
| ----- | -------------------: | --------------: | ----------------: | -----------------: |
| 14 | 3.088172652067336683 | 0.954336623855338749 | 0.027382594561449430 | 0.022713337755283022 |
| 15 | 3.087666778945727741 | 0.953802558846997944 | 0.027918466604424057 | 0.023106038586166092 |
| 16 | 3.087156262423669198 | 0.953264138194622858 | 0.028465647567239848 | 0.023505748696197129 |
| 17 | 3.086641041843746459 | 0.952721311199448859 | 0.029024402523646663 | 0.023912600497345078 |
| 18 | 3.086121055522226600 | 0.952174026402790559 | 0.029595003517611926 | 0.024326729051185183 |
| 19 | 3.085596240727705795 | 0.951622231572182167 | 0.030177729768427544 | 0.024748272123592492 |
| 20 | 3.085066533659228475 | 0.951065873687221144 | 0.030772867882523797 | 0.025177370240523024 |

两个 Escape case 的 endpoint fields 与 competing-event margin 为：

```text
y=12 position=(-170.743537420571297,1.337744245785817,-104.140872592375216)
     direction=(-0.822251516308215,0.005874201091297,-0.569093962092711)
     escape_before_next_crossing_mino_margin=0.224619765654128162
y=13 position=(-170.447402756461085,1.369244882783221,-104.624437497465033)
     direction=(-0.820715680321072,0.006023904198190,-0.571305071440235)
     escape_before_next_crossing_mino_margin=0.224931067328566682
```

每个 case 还输出 $E,b,\eta$、radial/polar simple-root derivative 与 initial-null/Mino/chart residual。所有 180 位 residual 均通过 $10^{-165}$ 门槛；它们是方程与条件性诊断，不替代上表的 observable comparison。

`polyroots` 的官方合同指出 multiple/ill-conditioned roots 需要额外 precision 与 convergence study；脚本因此把 root classification、simple-root derivative 和完整 precision doubling 分开保存，而不把一次 `polyroots` return 当作充分证明。[mpmath polynomial roots](https://mpmath.org/doc/1.3.0/calculus/polynomials.html) `findroot(..., verify=True)` 只验证求得点的 residual，仍不替代 bracket/root-topology 证据。[mpmath root finding](https://mpmath.org/doc/1.3.0/calculus/optimization.html)

## 7. 消费者与证据链

[`surface_observable_tests.rs`](../../crates/gravlume-reference/tests/surface_observable_tests.rs) 将上述 expectation 舍入到 binary64，并要求 `reference-regular-v1` 与 `reference-strict-v1` 同时满足 exact branch、Escape position/direction/time、surface source/frequency/time/intensity gates。Source Anchor 只应用[验证合同定义的二维 wrapped surface-distance gate](../validation.md#52-reference-agreement)，不把 radial/azimuth component tolerance 分开后同时放行。它不改变 v2 fixture 的 schema、profile、producer 字段或旧 expected。

统一 CLI 直接复算固定九点 corpus，并在构造证书时验证 precision、discrete identity、event
margin、unit/outward direction 与 residual threshold。该模块不发布通用 scientific API；连续量使用显式
semantic gate，验证不冻结私有 record、辅助函数或打印格式。

Renderer 已有 canonical v2 的 fresh binary32 fields 与最终 `RGBA16F` gate；ordered corpus 对全部九点逐项比较 fresh binary32 terminal/branch/continuous fields。因此当前证据链是：

```text
independent BL/Mino 120/180 digits
  -> Cartesian KS f64 regular/strict
  -> outgoing Cartesian KS WGSL binary32
  -> RGBA16F texture (canonical only)
```

CPU test 与 GPU test 仍是两个 consumer；GPU 没有读取 Python 输出或 BL equations。除 canonical v2 外，corpus 尚无统一的最终 `RGBA16F` texture gate。没有 WGSL、buffer ABI、workgroup、dispatch 或 publication 改动：现有 runtime-array kernel 已满足 host-shareable `vec4` layout、每 invocation 独占 record 与 partial-workgroup guard，本次科学增量不以未测 AoS→SoA、subgroup 或 workgroup-size 改写混入证据提交。

## 8. 后续候选

| 候选 | Interface 与实现 | 优点 | 风险/准入条件 |
| ---- | --------------- | ---- | ------------- |
| 分段 high-precision quadrature corpus generator | 输入一个版本化 canonical case，输出 typed branch、observable、bounds 与 provenance；内部按 root topology 分 turning segment | 方程独立、容易逐 stratum 扩展、适合先建立 evidence | 每种 topology 都要单独 regularize；near-degenerate root 可能昂贵或 unsupported；在首个持久 consumer 前不冻结 schema |
| Manifestly-real elliptic/Carlson terminal solver | pure-Kerr classified root topology 到 terminal observable 的小 interface；KS 为 fallback adapter | 可避免长 phase accumulation，未来可能成为 CPU oracle/GPU accelerator | 闭式存在不证明 WGSL `f32` 稳定；必须先过 axis/extreme/degenerate/root-branch 与 phase certificate，再做 Metal/Vulkan Pareto gate |

**决策：** 采用第一个候选扩展独立 evidence；第二个候选保持研究方向。当前 fixed-step reciprocal-Mino 已被 travel-time 反例否决，不能借本次高精度成功恢复；见 [Mino step selection](mino-step-selection.md)。

## 9. 不支持域与恢复条件

当前方法只接受固定 `(640,12..20)` center-subpixel corpus，并拒绝：

- 不是 canonical ingoing pure Kerr Observation 的输入；
- initial sign、root topology 或 event order 不符合本记录的单 radial/polar turning case；
- radial/polar root 不是可分离的 simple real root；
- surface crossing 不在 `[6M,20M]`，或 Escape case 无法证明 first crossing 在外缘之外且 escape 早于下一 crossing，或 horizon/singularity/event competition 可能更早；
- axis、near-axis、extreme/near-extreme、multiple/near-multiple roots；
- precision doubling 不能保留至少 80 个 normalized decimal digits，或任一 normalized delta 非 finite；
- sample 不是上述 canonical viewport 的九个具名整数 pixel；
- terminal/branch 不精确匹配对应具名 case；
- 任一连续字段非 finite、surface/escape/event-order margin 非法、物理正号非法，或 null/Mino/chart primitive residual 达不到 $p-15$ 位。

扩展一个新 stratum 的恢复条件是：从规范十进制输入独立重建，保存 exact discrete identity，给出 root/event signed margin，至少做 120/180 位重算，并让 regular/strict 与 GPU（若声称 GPU 支持）分别通过自己的 observable gate。不能通过放宽一个 RGB max error、只看 invariant drift 或复制 Cartesian KS equations 进入 witness 来恢复。

## 10. 复算命令

正式 witness：

```text
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research bl-mino-surface
```

依赖解析、升级、完整测试与 lint 命令见[统一 Python 研究工具链](python-research-tooling.md)。

对应 Rust consumer test：

```text
cargo test -p gravlume-reference --test surface_observable_tests --locked \
  independent_bl_mino_witness
```

对应 ordered GPU corpus test：

```text
cargo test -p gravlume-render --lib --locked \
  gpu_trace_tests::surface::ordered_gpu_surface_edge_corpus_matches_reference_fields -- --exact
```

## 11. 一手来源

- [Carter, *Global Structure of the Kerr Family of Gravitational Fields* (1968)](https://doi.org/10.1103/PhysRev.174.1559)：Hamilton–Jacobi separability、第四常数与 quadratures；
- [Mino, *Perturbative Approach to an Orbital Evolution around a Supermassive Black Hole* (2003)](https://doi.org/10.1103/PhysRevD.67.084027)：Mino parameter；
- [Gralla & Lupsasca, *Null geodesics of the Kerr exterior* (2020)](https://doi.org/10.1103/PhysRevD.101.044032)：real root topology、turning segments 与 Kerr null-geodesic integrals；
- [mpmath precision management](https://mpmath.org/doc/1.3.0/general.html#precision-management)、[quadrature](https://mpmath.org/doc/1.3.0/calculus/integration.html)、[root finding](https://mpmath.org/doc/1.3.0/calculus/optimization.html)、[polynomial roots](https://mpmath.org/doc/1.3.0/calculus/polynomials.html)与[finite/comparison utilities](https://mpmath.org/doc/1.3.0/general.html)：arbitrary-precision context、quadrature、root 与认证边界；

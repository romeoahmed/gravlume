# Canonical surface 与 outer-edge pair 高精度 BL/Mino witness

本文记录 `kerr-exterior-observation-v1` 上 canonical surface ray 与相邻 outer-source-edge pair 的独立高精度见证、可复算方法、失败边界与后续候选。它只是一份 research decision/evidence record，不定义 production interface、Validation Profile、fixture schema、GPU method identity 或支持域；这些合同仍分别以[数学物理](../physics.md)、[验证合同](../validation.md)、[路线图](../roadmap.md)和[连续字段 corpus 记录](continuous-field-corpus.md)为准。

**状态：canonical 与相邻 outer-edge pair 已采用；完整 corpus 与 artifact 未闭合。** 当前方法认证 `(640,16)` ordinary surface，以及 `(640,13)` Escape / `(640,14)` surface 的 center-subpixel pair。它不能授权完整九点 stencil、其他 observation、Kerr–Newman、near-critical/axis/extreme case、production BL/Mino solver 或 WGSL fast path。

## 1. 可否证假设

首切片检验以下命题：

1. 从规范十进制 Observation 独立重建的 future-directed Photon Momentum，可在 ingoing Cartesian Kerr–Schild 与 Boyer–Lindquist canonical covector 之间保持 $E,L_z,\mathcal Q$ 和 null constraint；
2. pure Kerr separated radial/polar potentials 能给出与 Cartesian KS reference 相同的 discrete path identity，而不是只恢复接近的 terminal position；
3. 对 simple-root surface case，分 turning segment 的 100+ decimal-digit quadrature 能独立恢复 Source Anchor、Frequency Ratio、KS coordinate-time duration 与 surface radiance；
4. 对 outer-edge 外侧 case，第一次 equatorial crossing 的 signed radial margin、Escape 与下一 crossing 的 Mino-order margin、localized KS position、negative-affine traversal direction 和 travel time 能在同一 separated graph 中闭合；
5. 120/180 decimal-digit 重算、turning-root conditioning 与两种 KS chart primitive evaluation 能把数值误差压到 validation gate 之下；
6. 若上述任一 discrete identity、root/event order、constraint 或 precision-doubling gate 失败，该 case 必须保持 unsupported，不能用 CPU/GPU agreement 替代。

Hamilton–Jacobi separability 与第四常数来自 [Carter 1968](https://doi.org/10.1103/PhysRev.174.1559)；affine/Mino reparameterization 来自 [Mino 2003](https://doi.org/10.1103/PhysRevD.67.084027)；real Kerr null-geodesic root topology 与分段积分对照 [Gralla–Lupsasca 2020](https://doi.org/10.1103/PhysRevD.101.044032)。项目的物理自旋、chart handedness 和 covector transform 已由 [KS↔BL/Mino seam](kerr-schild-mino-map.md) 单独封闭。

## 2. 与 Cartesian KS reference 的独立性

可复算实现为 [`verify_bl_mino_surface_witness.py`](scripts/verify_bl_mino_surface_witness.py)。它：

- 不导入 `gravlume-domain`、`gravlume-reference` 或 renderer；
- 不读取 Rust reference outcome、GPU record、texture 或 fixture expected；
- 从 `M=1,a=0.8,q_e=0`、observer oblate event、stationary observer、target/up、viewport/FOV 与 pixel/subpixel 十进制输入重新构造 KS metric、Observer Frame 和 camera covector；
- 用 one-form invariance 独立转到 BL covector，再形成 $E=-p_t$、$b=L_z/E$ 与 $\eta=\mathcal Q/E^2$；
- 在 separated BL/Mino graph 中求 terminal；只在最后把 endpoint observable 转回 ingoing KS chart；
- 用 mpmath 1.3 arbitrary precision、precision doubling 和等价 evaluation 自己给出误差证据。

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

脚本先从 initial Hamilton tangent 决定 radial/polar sign，再分类 quartic roots；不从 backward traversal、position 或 `signum(0)` 猜 branch。三个具名 case 的 exact discrete identity 为：

| 字段                                 | `(640,16)` ordinary | `(640,13)` outside | `(640,14)` inside |
| ------------------------------------ | ------------------- | ------------------ | ----------------- |
| terminal                             | surface             | Escape             | surface           |
| initial polar side                   | `positive`          | `positive`         | `positive`        |
| radial turnings                      | `1`                 | `1`                | `1`               |
| polar turnings                       | `1`                 | `1`                | `1`               |
| equatorial crossings before terminal | `0`                 | `1`                | `0`               |
| signed azimuth winding               | `0`                 | `0`                | `0`               |

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

Outer-edge pair 另把第一次 equatorial crossing 解在 source outer radius 两侧。Outside case 必须证明 crossing 位于 $20M$ 之外，且 $R_{esc}=200M$ 先于下一次 equatorial crossing；随后把 BL endpoint tangent 转为 ingoing Cartesian KS，并按 reference 的 negative-affine traversal orientation 归一化 Escape direction。Position、direction 与 travel time 分别验收，不用 preview RGB 替代。

## 6. 认证条件与 120/180 十进制位结果

研究环境要求 Python 3.10 或更高版本；精确依赖版本以 [`pyproject.toml`](scripts/pyproject.toml)与 [`uv.lock`](scripts/uv.lock) 为准，不把一次复算所用的 Python patch 版本写成算法合同。每次结果必须先满足以下认证条件：

- terminal、initial side、turning/crossing counts 与 winding 精确等于第 3 节的对应 discrete identity；
- 所有连续字段为 real finite；surface source 位于 $[6M,20M]$，Escape position 位于 $R_{esc}$、direction normalized/outward，edge 与 event-order margin 符号正确；有物理正号要求的 observable 与 conditioning 严格为正；
- initial null、Mino constraint 与 chart primitive 的 normalized residual 均满足

\[
\rho < 10^{-(p-15)},
\]

其中 $p$ 是 working decimal digits，15 位作为 guard。任一条件失败都不给出通过证书。mpmath precision context、finite 检查与数值比较的实现依据集中在[一手来源](#11-一手来源)，私有 Python record、函数名与测试结构不属于研究结论。

### 6.1 Canonical surface

120 与 180 decimal-digit 两次完整重算分别先通过同一个 exact canonical identity seam，再比较 source/transfer/phase、constants of motion 及 radial/polar turning derivatives；所有 normalized delta 先要求 finite。最大值为：

```text
2.84198169412e-116
```

180 位运行保留的主要输出如下；显示位数少于 working precision，但远高于 binary64：

```text
source_radius_m=19.650678984603292401979974641605230299041623565323313117936548513382249668946276890048582220807888696663401564
source_azimuth_rad=3.0871562624236691978088921179317878660145706182501492328189946911311491495070169991747041709053874960130102662
frequency_ratio=0.95326413819462285789527508811632595633667388809241237167378175163595970034762335101213042362744948362044209166
travel_time_m=54.902474247630053841777238380553092247364224368644429474406867691088764431638153286622246820521818044574301422
emitted_bolometric_intensity=0.028465647567239848463048432243762537365965395227154871892176970966564958642363544152492872077320420261682581790
observed_bolometric_intensity=0.023505748696197128508784777489376183681874993000970528536248652660575945459339356874036049870753785112414387513
energy=0.96609791588563310122334760012182709671623780961475185717111104084186357155978342318687688893224572025401175692
impact_parameter=-0.097077403034846176173748571716234534524033968006103986458653222851983509233574937910244025566519010090379433865
carter_parameter=130.33773934131547879531639059882714720394238555932693290625932484496191771867535611756462700122611718710644936
```

Conditioning 与 constraint diagnostics：

```text
radial_turning_derivative=1902.5746415344517763
polar_turning_derivative=261.96471835640187990
initial_null_residual=2.99768672025e-181
mino_constraint_residual=1.44183089292e-180
chart_primitive_residual=7.22975959531e-181
```

### 6.2 Outer-edge pair

Pair certificate 同时重算 outside/inside 的全部 semantic fields、Escape position/direction lanes、source/transfer/phase 与 conditioning。最大 normalized delta 为 `3.85612445201e-94`，超过要求的 80 stable decimal digits。180 位主要结果：

```text
outside=(640,13) terminal=escape crossings=1
first_crossing_radius_m=20.0352415772575171500624302418
outer_edge_signed_margin_m=-0.0352415772575171500624302418
escape_position_xyz_m=(-170.44740275646108545,1.36924488278322070,-104.62443749746503348)
escape_direction_xyz=(-0.82071568032107155,0.00602390419818969,-0.57130507144023473)
travel_time_m=238.43869437867636128
escape_before_next_crossing_mino_margin=0.22493106732856668234

inside=(640,14) terminal=equatorial-surface crossings=0
source_radius_m=19.9064149026366576745254702082
outer_edge_signed_margin_m=0.0935850973633423254745297918
source_azimuth_rad=3.0881726520673366834001641677
frequency_ratio=0.9543366238553387492793786680
travel_time_m=55.1114457365679603363977638103
observed_bolometric_intensity=0.0227133377552830216082509940
```

`polyroots` 的官方合同指出 multiple/ill-conditioned roots 需要额外 precision 与 convergence study；脚本因此把 root classification、simple-root derivative 和完整 precision doubling 分开保存，而不把一次 `polyroots` return 当作充分证明。[mpmath polynomial roots](https://mpmath.org/doc/1.3.0/calculus/polynomials.html) `findroot(..., verify=True)` 只验证求得点的 residual，仍不替代 bracket/root-topology 证据。[mpmath root finding](https://mpmath.org/doc/1.3.0/calculus/optimization.html)

## 7. 消费者与证据链

[`surface_observable_tests.rs`](../../crates/gravlume-reference/tests/surface_observable_tests.rs) 将上述 expectation 舍入到 binary64，并要求 `reference-regular-v1` 与 `reference-strict-v1` 同时满足 exact branch、Escape position/direction/time、surface source/frequency/time/intensity gates。Source Anchor 只应用[验证合同定义的二维 wrapped surface-distance gate](../validation.md#52-reference-agreement)，不把 radial/azimuth component tolerance 分开后同时放行。它不改变 v2 fixture 的 schema、profile、producer 字段或旧 expected。

Research tests 保护公开的认证 seam：canonical deterministic oracle、固定 source-edge pair、非法 sample/precision、discrete identity、event margin、unit/outward direction 与 residual threshold。连续量使用显式 semantic gate；测试不冻结私有 record、辅助函数或生成格式。

Renderer 已有 canonical v2 的 fresh binary32 fields 与最终 `RGBA16F` gate；ordered corpus 也包含相邻 pair，并逐项比较 fresh binary32 terminal/branch/continuous fields。因此当前证据链是：

```text
independent BL/Mino 120/180 digits
  -> Cartesian KS f64 regular/strict
  -> outgoing Cartesian KS WGSL binary32
  -> RGBA16F texture (canonical only)
```

CPU test 与 GPU test 仍是两个 consumer；GPU 没有读取 Python 输出或 BL equations。Pair 尚无统一的最终 `RGBA16F` texture gate。没有 WGSL、buffer ABI、workgroup、dispatch 或 publication 改动：现有 runtime-array kernel 已满足 host-shareable `vec4` layout、每 invocation 独占 record 与 partial-workgroup guard，本次科学增量不以未测 AoS→SoA、subgroup 或 workgroup-size 改写混入证据提交。

## 8. 后续候选

| 候选 | Interface 与实现 | 优点 | 风险/准入条件 |
| ---- | --------------- | ---- | ------------- |
| 分段 high-precision quadrature corpus generator | 输入一个版本化 canonical case，输出 typed branch、observable、bounds 与 provenance；内部按 root topology 分 turning segment | 方程独立、容易逐 stratum 扩展、适合先建立 evidence | 每种 topology 都要单独 regularize；near-degenerate root 可能昂贵或 unsupported；在首个持久 consumer 前不冻结 schema |
| Manifestly-real elliptic/Carlson terminal solver | pure-Kerr classified root topology 到 terminal observable 的小 interface；KS 为 fallback adapter | 可避免长 phase accumulation，未来可能成为 CPU oracle/GPU accelerator | 闭式存在不证明 WGSL `f32` 稳定；必须先过 axis/extreme/degenerate/root-branch 与 phase certificate，再做 Metal/Vulkan Pareto gate |

**决策：** 采用第一个候选扩展独立 evidence；第二个候选保持研究方向。当前 fixed-step reciprocal-Mino 已被 travel-time 反例否决，不能借本次高精度成功恢复；见 [Mino step selection](mino-step-selection.md)。

## 9. 不支持域与恢复条件

当前方法只接受 `(640,16)` ordinary surface 与固定 `(640,13)/(640,14)` pair，并拒绝：

- 不是 canonical ingoing pure Kerr Observation 的输入；
- initial sign、root topology 或 event order 不符合本记录的单 radial/polar turning case；
- radial/polar root 不是可分离的 simple real root；
- surface crossing 不在 `[6M,20M]`，或 Escape case 无法证明 first crossing 在外缘之外且 escape 早于下一 crossing，或 horizon/singularity/event competition 可能更早；
- axis、near-axis、extreme/near-extreme、multiple/near-multiple roots；
- precision doubling 不能保留至少 80 个 normalized decimal digits，或任一 normalized delta 非 finite；
- sample 不是上述 canonical viewport 的具名整数 pixel/pair；
- terminal/branch 不精确匹配对应具名 case；
- 任一连续字段非 finite、surface/escape/event-order margin 非法、物理正号非法，或 null/Mino/chart primitive residual 达不到 $p-15$ 位。

扩展一个新 stratum 的恢复条件是：从规范十进制输入独立重建，保存 exact discrete identity，给出 root/event signed margin，至少做 120/180 位重算，并让 regular/strict 与 GPU（若声称 GPU 支持）分别通过自己的 observable gate。不能通过放宽一个 RGB max error、只看 invariant drift 或复制 Cartesian KS equations 进入 witness 来恢复。

## 10. 复算命令

正式 witness：

```text
uv run --isolated --project docs/research/scripts --locked \
  python -B docs/research/scripts/verify_bl_mino_surface_witness.py
```

Research module 行为测试：

```text
uv run --isolated --project docs/research/scripts --locked \
  python -B -m unittest discover -s docs/research/scripts \
  -p 'test_bl_mino_surface_witness.py'
```

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
- [mpmath 1.3 precision management](https://mpmath.org/doc/1.3.0/general.html#precision-management)、[quadrature](https://mpmath.org/doc/1.3.0/calculus/integration.html)、[root finding](https://mpmath.org/doc/1.3.0/calculus/optimization.html)、[polynomial roots](https://mpmath.org/doc/1.3.0/calculus/polynomials.html)与[finite/comparison utilities](https://mpmath.org/doc/1.3.0/general.html)：arbitrary-precision context、quadrature、root 与认证边界；
- [Hypothesis `@given`](https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.given)：research boundary 的生成式性质测试与反例缩减。

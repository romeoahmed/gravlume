# Canonical surface 的 high-precision BL/Mino witness

本文记录 `kerr-exterior-observation-v1` 上一个 canonical surface ray 的独立高精度见证、可复算方法、失败边界与后续候选。它只是一份 research decision/evidence record，不定义 production interface、Validation Profile、fixture schema、GPU method identity 或支持域；这些合同仍分别以[数学物理](../physics.md)、[验证合同](../validation.md)、[路线图](../roadmap.md)和[连续字段 corpus 记录](continuous-field-corpus.md)为准。

**状态：canonical 首切片已采用；完整 corpus 与 artifact 未闭合。** 当前方法只认证 `(640,16,0.5,0.5)`。它不能授权九点 source-edge stencil、其他 observation、Kerr–Newman、near-critical/axis/extreme case、production BL/Mino solver 或 WGSL fast path。

## 1. 可否证假设

首切片检验以下命题：

1. 从规范十进制 Observation 独立重建的 future-directed Photon Momentum，可在 ingoing Cartesian Kerr–Schild 与 Boyer–Lindquist canonical covector 之间保持 $E,L_z,\mathcal Q$ 和 null constraint；
2. pure Kerr separated radial/polar potentials 能给出与 Cartesian KS reference 相同的 discrete path identity，而不是只恢复接近的 terminal position；
3. 对该 ordinary、simple-root case，分 turning segment 的 100+ decimal-digit quadrature 能独立恢复 Source Anchor、Frequency Ratio、KS coordinate-time duration 与 surface radiance；
4. 120/180 decimal-digit 重算、turning-root conditioning 与两种 KS chart primitive evaluation 能把数值误差压到 validation gate 之下；
5. 若上述任一 discrete identity、root topology、constraint 或 precision-doubling gate 失败，该 case 必须保持 unsupported，不能用 CPU/GPU agreement 替代。

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

脚本先从 initial Hamilton tangent 决定 radial/polar sign，再分类 quartic roots；不从 backward traversal、position 或 `signum(0)` 猜 branch。Canonical case 的 exact discrete identity 为：

| 字段                                  | 结果                 |
| ------------------------------------- | -------------------- |
| terminal                              | `equatorial-surface` |
| initial polar side                    | `positive`           |
| radial turnings                       | `1`                  |
| polar turnings                        | `1`                  |
| equatorial crossings before terminal  | `0`                  |
| signed azimuth winding                | `0`                  |

Terminal surface crossing 本身不计入“before terminal”的 crossing count，与 Reference branch-key 提交语义一致。Polar turning 是独立 witness 保存的 path identity；当前 Rust branch key 没有该字段。

## 4. Turning regularization

直接把 `1/sqrt(R)` 或 `1/sqrt(U)` 交给 generic quadrature，会在 simple turning endpoint 形成可积但条件不良的平方根奇点；把 working precision 调高并不能消除表达式先发生的 `0/0` 消减。

### 4.1 Polar segment

对当前 root topology，

\[
U(\mu)=a^2(\mu_+^2-\mu^2)(\mu^2-\mu_-^2),
\qquad \mu_-^2<0.
\]

每个到 turning 的 segment 使用 $\mu=\mu_+\sin\chi$。Jacobian 精确消去 $\sqrt{\mu_+^2-\mu^2}$，得到端点有限的 integrand。Observer→turn 与 equator→turn 分别积分后相加，保留一个 polar turning。

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

## 6. 120/180 位结果

锁定环境为 Python 3.14.7、mpmath 1.3.0；依赖以 [`uv.lock`](scripts/uv.lock) 为准。`SurfaceWitness` 是认证边界，而不是原始数值容器：生成的每个实例都通过 `__post_init__` 要求 terminal、initial side、turning/crossing counts 与 winding 的 runtime type/value 精确等于本 case 的 canonical discrete identity；所有连续字段为 real finite；terminal source 位于 $[6M,20M]$；具物理正号的 observable/conditioning 为正；并要求三个 normalized equation residual 满足

\[
\rho < 10^{-(p-15)},
\]

其中 $p$ 是本次 working decimal digits，15 位作为 guard。任何检查失败都直接返回 `UnsupportedWitnessError`，不会生成 certificate 或打印 `RESULT=PASS`。mpmath 官方合同说明 `workdps` 只临时改变并恢复 decimal precision；finite classification 必须显式使用 `isfinite`（[mpmath precision context 与 `isfinite`](https://mpmath.org/doc/1.3.0/general.html)）。Python 对 NaN 的有序比较恒为 false，因此不能让 `max` 或普通阈值比较隐式承担 NaN 检查（[Python value comparisons](https://docs.python.org/3/reference/expressions.html#value-comparisons)）。不可变 dataclass 的 generated `__init__` 会调用 `__post_init__`，`dataclasses.replace` 也保持这一不变量；但 dataclass 不执行 annotation 的 runtime type check（[Python dataclasses](https://docs.python.org/3.10/library/dataclasses.html#dataclasses.dataclass)），且 `bool` 是 `int` 的子类（[Python boolean type](https://docs.python.org/3.10/library/stdtypes.html#boolean-type-bool)），所以离散 identity 同时检查 exact type 与 value，不能只依赖 `==`。

120 与 180 decimal-digit 两次完整重算分别先通过同一个 exact canonical identity boundary，再比较 source/transfer/phase、constants of motion 及 radial/polar turning derivatives；所有 normalized delta 先要求 finite。最大值为：

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

`compute_canonical_surface_witness` 的外部 seam 还要求两个 pixel coordinate 的 runtime type 精确为 `int`，先拒绝 `bool`、float 数值别名及其他对象，再检查 viewport 与 canonical sample。这样 `(640.0,16.0)` 不会因 Python 数值相等规则绕过整数 sample interface。

`polyroots` 的官方合同指出 multiple/ill-conditioned roots 需要额外 precision 与 convergence study；脚本因此把 root classification、simple-root derivative 和完整 precision doubling 分开保存，而不把一次 `polyroots` return 当作充分证明。[mpmath polynomial roots](https://mpmath.org/doc/1.3.0/calculus/polynomials.html) `findroot(..., verify=True)` 只验证求得点的 residual，仍不替代 bracket/root-topology 证据。[mpmath root finding](https://mpmath.org/doc/1.3.0/calculus/optimization.html)

## 7. Test consumer 与证据链

[`surface_observable_tests.rs`](../../crates/gravlume-reference/tests/surface_observable_tests.rs) 将上述 expectation 舍入到 binary64，并要求 `reference-regular-v1` 与 `reference-strict-v1` 同时满足 discrete identity、source、frequency、KS time 和 intensity gates。Source Anchor 只应用[验证合同定义的二维 wrapped surface-distance gate](../validation.md#52-reference-agreement)，不把 radial/azimuth component tolerance 分开后同时放行。它不改变 v2 fixture 的 schema、profile、producer 字段或旧 expected。

Research module test 保留具名 deterministic oracle，并用 Hypothesis 性质覆盖非 canonical 整数 sample、非整数 coordinate、非法 precision、离散 identity 变异与 residual certification boundary；不再为 viewport 内外各保存一个重复样例，也不重复断言成功构造已经保证的 post-init 条件。Hypothesis 的 `@given` 可直接装饰 unittest method，并负责生成与缩减反例（[Hypothesis `@given`](https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.given)）。具名 `mpf` observable 使用显式 `rel_eps=0`、`abs_eps=gate` 的 [`mpmath.almosteq`](https://mpmath.org/doc/1.3.0/general.html#almosteq)，不会退回 working-precision 默认容差。

Renderer 已有 canonical v2 的 fresh binary32 fields 与最终 `RGBA16F` gate。因此当前纵向链是：

```text
independent BL/Mino 120/180 digits
  -> Cartesian KS f64 regular/strict
  -> outgoing Cartesian KS WGSL binary32
  -> RGBA16F texture
```

这条链只覆盖一个重合 sample。CPU test 与 GPU test 仍是两个 consumer；GPU 没有读取 Python 输出或 BL equations。没有 WGSL、buffer ABI、workgroup、dispatch 或 publication 改动，因此本记录不提出新的 GPU layout/vectorization 声明。

## 8. 两个后续候选

| 候选 | Interface 与实现 | 优点 | 风险/准入条件 |
| ---- | --------------- | ---- | ------------- |
| 分段 high-precision quadrature corpus generator | 输入一个版本化 canonical case，输出 typed branch、observable、bounds 与 provenance；内部按 root topology 分 turning segment | 方程独立、容易逐 stratum 扩展、适合先建立 evidence | 每种 topology 都要单独 regularize；near-degenerate root 可能昂贵或 unsupported；在首个持久 consumer 前不冻结 schema |
| Manifestly-real elliptic/Carlson terminal solver | pure-Kerr classified root topology 到 terminal observable 的小 interface；KS 为 fallback adapter | 可避免长 phase accumulation，未来可能成为 CPU oracle/GPU accelerator | 闭式存在不证明 WGSL `f32` 稳定；必须先过 axis/extreme/degenerate/root-branch 与 phase certificate，再做 Metal/Vulkan Pareto gate |

**决策：** 采用第一个候选扩展独立 evidence；第二个候选保持研究方向。当前 fixed-step reciprocal-Mino 已被 travel-time 反例否决，不能借本次高精度成功恢复；见 [Mino step selection](mino-step-selection.md)。

## 9. Unsupported 与恢复条件

当前 function 在外部 seam typed-reject 所有非 `(640,16)` sample，以及：

- 不是 canonical ingoing pure Kerr Observation 的输入；
- initial sign、root topology 或 event order 不符合本记录的单 radial/polar turning case；
- radial/polar root 不是可分离的 simple real root；
- source crossing 不在 `[6M,20M]`，或 horizon/escape/singularity/event competition 可能更早；
- axis、near-axis、extreme/near-extreme、multiple/near-multiple roots；
- precision doubling 不能保留至少 80 个 normalized decimal digits，或任一 normalized delta 非 finite；
- pixel coordinate 的 runtime type 不是精确 `int`，或整数 sample 不是 `(640,16)`；
- terminal/branch 的 runtime type 或 value 不精确匹配本 canonical case；
- 任一连续字段非 finite、source 越出 surface domain、物理正号非法，或 null/Mino/chart primitive residual 达不到 $p-15$ 位。

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
  canonical_surface_matches_the_independent_bl_mino_witness -- --exact
```

## 11. 一手来源

- [Carter, *Global Structure of the Kerr Family of Gravitational Fields* (1968)](https://doi.org/10.1103/PhysRev.174.1559)：Hamilton–Jacobi separability、第四常数与 quadratures；
- [Mino, *Perturbative Approach to an Orbital Evolution around a Supermassive Black Hole* (2003)](https://doi.org/10.1103/PhysRevD.67.084027)：Mino parameter；
- [Gralla & Lupsasca, *Null geodesics of the Kerr exterior* (2020)](https://doi.org/10.1103/PhysRevD.101.044032)：real root topology、turning segments 与 Kerr null-geodesic integrals；
- [mpmath 1.3 quadrature](https://mpmath.org/doc/1.3.0/calculus/integration.html)、[root finding](https://mpmath.org/doc/1.3.0/calculus/optimization.html)、[polynomial roots](https://mpmath.org/doc/1.3.0/calculus/polynomials.html)与[precision/finite/almosteq utilities](https://mpmath.org/doc/1.3.0/general.html)：本 research tool 的 arbitrary-precision numerical contracts；
- [Hypothesis `@given`](https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.given)：research boundary 的生成式性质测试与反例缩减；
- [Python 3.10 dataclasses](https://docs.python.org/3.10/library/dataclasses.html#post-init-processing)与[value comparisons](https://docs.python.org/3/reference/expressions.html#value-comparisons)：认证对象构造不变量与 NaN rejection；
- Rust [`f64::midpoint`](https://doc.rust-lang.org/stable/std/primitive.f64.html#method.midpoint)与 [`f64::hypot`](https://doc.rust-lang.org/stable/std/primitive.f64.html#method.hypot)：binary64 Source Anchor test 的 midpoint 与 Euclidean distance primitive。

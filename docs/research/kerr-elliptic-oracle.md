# Kerr root topology 与 elliptic oracle

本文定义 pure Kerr null radial quartic 的最小研究合同：如何建立可复算的
root-topology 证书，以及如何用 `RF`、`RC`、`RD`、`RJ` 建立不依赖生产求解器的高精度
special-function oracle。本文是研究决策记录，不定义 production 支持域、Rust/WGSL API、
fixture schema 或质量阈值；这些权威合同仍分别以[数学物理](../physics.md)、
[验证](../validation.md)、[渲染准入](../rendering.md)和[路线图](../roadmap.md)为准。

**状态：受限 pure-Kerr 研究层已闭合。** 锁定环境中的 private oracle 以 exact rational root
isolation、Carlson defining integrals、minimal identities 与 120/180 位复算关闭 pure-Kerr
nonextremal、分离裕量明确的研究切片；不能据此宣称 WGSL `f32`、真实 GPU、近极端、
principal-value/complex Carlson 或完整 terminal solver 已闭合。研究实现与测试见
[`_topology.py`](scripts/src/gravlume_research/checks/kerr_elliptic/_topology.py)、
[`_carlson.py`](scripts/src/gravlume_research/checks/kerr_elliptic/_carlson.py)和
[`test_kerr_elliptic.py`](scripts/tests/test_kerr_elliptic.py)。

## 1. 职责与实现边界

当前 production shader 仍是 Cartesian Kerr–Schild RK4；它没有 quartic root classifier 或
Carlson special functions，见
[`geodesic_integration.wgsl`](../../crates/gravlume-render/src/shaders/geodesic_integration.wgsl)。
[GPU 加速研究](gpu-geodesic-acceleration.md)已把 root-aware elliptic/Carlson terminal solver
列为 fixed-step Mino 失败后的下一候选，但没有把候选写成 production 事实。

[`bl_mino`](scripts/src/gravlume_research/checks/bl_mino/) proof package 已能为命名
surface/capture pair 以 `mp.polyroots`、exterior stationary minimum 和独立 quadrature 证明
separatrix 两侧的 terminal、event order 与 continuous observables；120/180 位证书记录在
[Kerr observable corpus](kerr-observable-corpus.md)。它的“虚部小于
`10^(-(dps-30))` 即视为 real”和“只数 observer 与 outer horizon 之间的 real roots”是该具名
witness 的内部判据，不是 Gralla–Lupsasca 全局 topology classifier。把它直接复用为 accelerator
admission seam 会漏掉 complex-root class、root/horizon coincidence、initial radial branch 与
turning sequence。尤其现有 capture witness 的“零个 exterior root”只能证明它位于
outer critical curve 的 capture 一侧，尚不足以在 II、III、IV 中命名一个 global class。

该 oracle 需要两份相互独立的证据：

1. `R(r)` 的 topology/branch 证书必须在有裕量的 generic case 给出稳定分类，并把所有
   degeneracy 明确送入 fallback；
2. Carlson oracle 必须同时通过定义积分、exact identities 和 precision doubling，不能只与
   同一库的另一条 special-function 调用比较。

实现仍全部位于 `docs/research/scripts/`，没有进入 Cargo dependency closure，没有新增 Rust/WGSL
solver、public trait、render graph、fixture schema 或 production support seam。

## 2. Kerr radial topology 的最小合同

### 2.1 适用方程与先决条件

对 `E != 0` 的 null geodesic，令 `lambda=L_z/E`、`eta=Q/E^2`，则

\[
R(r)=(r^2+a^2-a\lambda)^2-\Delta\left[\eta+(\lambda-a)^2\right],
\qquad \Delta=r^2-2Mr+a^2 .
\]

Gralla–Lupsasca 在 `0<a<M` 的 Kerr exterior 中证明 polar admissibility 蕴含
`zeta=eta+(lambda-a)^2 >= 0`，并把 `zeta=0` 单列为 principal null congruence；其余 root
taxonomy 假定 `zeta>0`、`E != 0`，extremal limit 也不在论文证明域内
（[原论文 Eq. 2、5、7、73--77](https://arxiv.org/pdf/1910.12881)）。所以 classifier 的输入
不能只有四个 quartic coefficients；至少还要先证明 `0<|a|<M`、`E!=0`、polar motion
允许，并保留初始 `r` 与 radial/polar momentum sign。

在 separated equations 中，变换 `(a,lambda)->(-a,-lambda)` 保持 `R` 与 `Theta` 不变，可由
[原论文 Eq. 7--8](https://arxiv.org/pdf/1910.12881)直接代数验证。它只允许 topology oracle
把负 spin 规范化到正 spin；KS/BL chart、azimuth、emitter motion 与最终 observable 仍需
项目已有 physical-spin seam 的独立证据，不能由这个代数对称性代替。

### 2.2 Generic classes 与实际 motion branch

对上述适用域，Gralla–Lupsasca 得到四种 generic root structure，并证明两个 horizon 之间
没有 real root；outer critical curve `C+` 是 `r3=r4>r+` 的 double-root boundary
（[原论文 Sec. IV.B](https://arxiv.org/pdf/1910.12881)）：

| class | root order | exterior allowed range | oracle 决策 |
| --- | --- | --- | --- |
| I | `r1<r2<r-<r+<r3<r4` | Ia: `r+<r<r3`; Ib: `r4<r<infinity` | 先只接纳分离的 Ib；Ia 分类但 fallback |
| II | `r1<r2<r3<r4<r-<r+` | `r+<r<infinity`，fly-in/out | 分类，但 terminal reduction fallback |
| III | `r1<r2<r-<r+`，`r3=conj(r4)` | `r+<r<infinity`，fly-in/out | 分类但 complex-pair reduction fallback |
| IV | 两对 complex-conjugate roots | exterior 全域 | 分类但 complex-pair reduction fallback |

Class I 仍不是一条唯一轨迹：初始 radius 位于 `r3` 下方还是 `r4` 上方决定 Ia/Ib，初始
radial sign 与 turn count 再决定被积分的 path。论文因此针对四种 substitution class 分别使用
initial position、initial momentum signs 与 turning count，而不是从 unordered roots 猜终点
（[原论文 Sec. VI 与 App. B](https://arxiv.org/pdf/1910.12881)）。首个可实现切片应只接受
**Ib、`r_initial>r4`、四个 simple real roots 都有分离裕量、至多一次已知 radial turn、全部
请求端点位于 exterior**；这已经覆盖现有 higher-order surface case 的 radial 形态，又不必
同时证明其余 substitution 与 branch reduction。

Compère、Liu、Long 对 exterior nonextremal Kerr 的更广 timelike taxonomy 找到四种 generic
simple-root structure 和由 horizon coincidence、zero root、double root 产生的八种
non-generic structure，并且还用 polar、time、azimuthal constraints 判定运动是否允许；null
分类只是其 limit（[原论文摘要与 DOI](https://doi.org/10.1103/PhysRevD.105.024075)）。这支持
两个边界：不能用 quartic root count 代替 physical admissibility，也不能把 timelike 的完整
taxonomy 原样移植成 Gravlume 的 null contract。

### 2.3 私有 classifier record 与独立检查

实现使用一个 Python-private exact-rational topology case，不新增 Rust trait 或持久 schema。每个 case
保存以下可观察量：

- canonical rational inputs：`M,a,E,lambda,eta,r_initial` 与 initial momentum signs；
- `r+`、`r-`、quartic coefficients，以及 120/180 位分别重建的 real roots、multiplicity 与
  degree-derived nonreal-root count；
- topology ID、initial allowed interval、ordered exterior turn sequence；
- 每个 real root 的 sign-changing bracket、normalized `R(root)`、`R'(root)`；
- four-real case 的 Vieta residual、conservative minimum bracket separation、
  root-to-horizon/initial gaps；
- isolated stationary points 与 `R` 的 signed extrema，用另一条路径复核 real-root count；
- polar admissibility、所有 denominator/domain predicates 与最小 normalized margin。

对 real-coefficient quartic，先隔离 derivative cubic 的 real stationary points，再用区间端点的
`R` 符号与 `R(+/-infinity)>0` 计数 simple real roots；每个被接纳的 real root 必须另有
sign-changing bracket。这样 topology 不依赖 `mp.polyroots` 的“微小虚部即 real”启发式。
Complex roots 只由“degree 减去 exact-isolated real-root multiplicity”计数；real coefficients 保证
nonreal roots 成 conjugate pairs。证书刻意不保存一个依赖 branch/tolerance 的 complex approximation，
也不使用 complex-root 值做 Carlson reduction。Root class 在 coefficient space 的边界处会改变，
因此任何 sign、order 或 separation margin 未显著大于 120/180 重建差异的 case 都必须
fallback，而不是硬选一侧；critical double root 可继续作为 boundary witness，但不是 accepted
fast-path sample（double-root 角色见
[Gralla–Lupsasca Sec. IV.B](https://arxiv.org/pdf/1910.12881)）。

最低限度的 RED mutations 是：

| mutation | 必须失败的观察量 |
| --- | --- |
| 把 Ib 的 `r3/r4` 交换 | order、Vieta 或 allowed interval |
| 删除一个 complex conjugate | coefficient/Vieta residual |
| 把 exact critical double root 标为 simple accepted | pair separation 与 `R'` margin |
| 保留 roots 但翻转 initial radial sign | ordered turn/event identity |
| 只按 exterior-root count 把 II/III/IV 合并成一个 accepted formula | topology-specific reduction/domain tag |
| 把 `a<0` 规范化却不同时翻转 `lambda` | separated-potential identity |

## 3. Carlson oracle 的最小数学核

### 3.1 Accepted 定义域

Carlson 的定义为

\[
R_F(x,y,z)=\frac12\int_0^\infty
\frac{dt}{\sqrt{(t+x)(t+y)(t+z)}},
\]

\[
R_J(x,y,z,p)=\frac32\int_0^\infty
\frac{dt}{(t+p)\sqrt{(t+x)(t+y)(t+z)}},
\quad R_C(x,y)=R_F(x,y,y),\quad R_D(x,y,z)=R_J(x,y,z,z).
\]

定义、cut-plane branch 和 zero restrictions 见
[Carlson 1995 Eq. 1--4](https://arxiv.org/pdf/math/9409227)与
[DLMF §19.16](https://dlmf.nist.gov/19.16)。为了让 oracle 的“独立积分”与 branch 判据都
清楚，支持域应严格缩为：

| function | accepted arguments |
| --- | --- |
| `RF` | `x,y,z>=0`，至多一个为零 |
| `RC` | `x>=0, y>0` |
| `RD` | `x,y>=0`，至多一个为零，`z>0` |
| `RJ` | `x,y,z>=0`，至多一个为零，`p>0` |

`p<0` 或 `RC` 的第二参数为负需要 Cauchy principal value；complex arguments 又需要一致的
square-root branch。它们虽在数学定义域内，却不属于当前 accepted 域
（[Carlson 1995 Eq. 2--4](https://arxiv.org/pdf/math/9409227)）。`p=0` 不是 `RJ` 定义域，且
正 `x,y,z` 下 `p->0+` 时发散，见
[DLMF 19.20.7](https://dlmf.nist.gov/19.20.E7)。

### 3.2 每个函数只需一组能区分错误的 identities

当前 oracle 从同一 canonical decimal inputs 独立计算 transformed defining integral。定义积分采用
`t=(u/(1-u))^2`、`u in [0,1]` 并分段 quadrature，同时显式处理 `t=0` 的单 zero 与
`t=infinity` 尾部；这条路径不调用 `mp.elliprf/rc/rd/rj`。`mpmath` special functions 只作第三方
triangulation；未来 production 候选可以使用 Carlson duplication + deviation polynomial，但必须
单独给出 rounding 与 termination 证书。`mpmath` 官方文档明确说明一般计算不保证
correct rounding，困难输入可能需要额外 guard digits
（[`mpmath` precision 文档](https://mpmath.org/doc/1.3.0/technical.html)）。

最小 identity set 如下；它覆盖 definition、scale、argument wiring 与 duplication correction，
不需要把整章 DLMF 冻成测试：

1. **定义积分**：四个函数都与上述独立 transformed quadrature 比较。
2. **homogeneity**：`RF/RC` 的 degree 为 `-1/2`，`RD/RJ` 为 `-3/2`
   （[DLMF 19.16.11](https://dlmf.nist.gov/19.16.E11)）。使用 binary 与非 binary scale ladder，
   可发现漏 normalization 或错误 degree。
3. **symmetry/degeneracy**：`RF` 对三个参数全对称；`RJ` 对前三个参数全对称；`RD` 只对
   `x,y` 对称；并验证 `RC=RF(x,y,y)`、`RD=RJ(x,y,z,z)`。定义直接给出这些性质，`RD`
   的有限 domain 见 [DLMF 19.16.5--6](https://dlmf.nist.gov/19.16)。
4. **diagonal 与 exact anchors**：`RF(x,x,x)=RC(x,x)=1/sqrt(x)`、
   `RD(x,x,x)=RJ(x,x,x,x)=x^(-3/2)`；再保留 `RC(0,1/4)=pi`、
   `RC(9/4,2)=log(2)` 等 Carlson 原文 check values
   （[Carlson 1995 Sec. 5](https://arxiv.org/pdf/math/9409227)；
   [DLMF §19.20](https://dlmf.nist.gov/19.20)）。
5. **exact duplication**：令
   `L=sqrt(x)sqrt(y)+sqrt(y)sqrt(z)+sqrt(z)sqrt(x)`，验证 `RF` 的
   [DLMF 19.26.18](https://dlmf.nist.gov/19.26.E18)、`RD` 的带 additive term
   [19.26.20](https://dlmf.nist.gov/19.26.E20)、`RJ` 的 `3 RC(alpha^2,beta^2)` correction
   [19.26.22--23](https://dlmf.nist.gov/19.26.E22)以及 `RC` 的
   [19.26.25](https://dlmf.nist.gov/19.26.E25)。DLMF 的算法分析说明一次 duplication 把
   argument differences 缩小四倍，见 [§19.36(i)](https://dlmf.nist.gov/19.36.i)。
6. **cross identities**：验证
   `RD(x,y,z)+RD(y,z,x)+RD(z,x,y)=3/sqrt(x*y*z)`（Carlson 1995 Eq. 54）以及
   `RJ(x,y,y,p)=3[RC(x,y)-RC(x,p)]/(p-y)`（远离 `p=y` 才使用，
   [DLMF 19.20.8](https://dlmf.nist.gov/19.20.E8)）。后者在 `p~y` 是差分消去测试，不应反过来
   当稳定实现公式。
7. **positive-domain bounds**：对适用 case 再验证 DLMF 的 `RF/RJ` 上下界；它不是精度
   oracle，却能在进入昂贵 quadrature 前发现 sign、argument permutation 或 normalization
   错误（[DLMF §19.24](https://dlmf.nist.gov/19.24)）。

Legendre `F/E/Pi` conversion 可作额外 cross-check，但如果两侧最终调用同一 `mpmath`
Carlson kernel，它不构成独立证据。最小集合保留 transformed defining integral，正是为了不让
library self-agreement 冒充 oracle independence。

### 3.3 Conditioning 与 typed fallback

| condition | 数值风险 | 处理与证据 |
| --- | --- | --- |
| 参数整体过大/过小 | 中间乘积 overflow/underflow | 先用 homogeneity 归一化，scale ladder 复核 |
| equal/near-equal arguments | deviation polynomial wiring、过早停止 | diagonal exact case + separation ladder + duplication residual |
| 一个允许的 zero | complete-integral endpoint | transformed defining integral + Carlson/DLMF zero anchors |
| `p->0+` | `RJ` 发散、dynamic range 增长 | 有限 positive ladder；达到 declared condition limit 后 typed fallback |
| `p<0` | principal-value transform 与 cancellation | 拒绝；Carlson 展示 transform 在 `RJ` 过零附近丢失 significant figures（[Sec. 5](https://arxiv.org/pdf/math/9409227)） |
| complex argument/branch | square-root branch 选择会改变结果 | 拒绝；DLMF 要求由 reduction 产生的首步 square roots 保持原 branch，不能一律取 principal root（[§19.36(i)](https://dlmf.nist.gov/19.36.i)） |
| iteration cap/non-finite correction | 无可证误差或 domain 逸出 | 返回 typed unsupported/ill-conditioned，不返回“最佳猜测” |
| 多个准确 Carlson 值作相消线性组合 | 单函数 relative error 不界定 terminal observable | guard digits 下直接复算组合并保存 cancellation ratio；必要时退回 quadrature |
| Kerr roots 合并或接近 horizon/pole | topology 跳变；BL `t,phi` 项可能分别发散 | root classifier fallback；沿用项目已证明的 KS/BL finite combination，不把 Carlson 当正则化证明 |

Carlson 1995 给出的 truncation relative-error bound 明确假设目标误差 `r` 远大于 machine
precision，因此 roundoff 相对 approximation error 可忽略
（[原文算法假设](https://arxiv.org/pdf/math/9409227)）。论文还指出 complex `RJ/RD` correction
可能相消，使所述 relative bound 不再严格。当前 defining-integral oracle 不实现该 truncation
algorithm，因此只保存 definition/identity residual 和 precision-doubling delta；未来 duplication
候选还必须另外保存 truncation estimate，且不能把它直接标成 total error bound。

## 4. Mutation 与闭合证据

Deterministic mutations 必须区分以下错误：near-double root 无裕量却被接纳、initial radial sign
选错 turn sequence、homogeneity degree 错误、`RD` additive term 缺失、`RJ/RC` correction 缺失、
把 `RD` 当作 full-permutation symmetric、未定义 principal-value policy 却接纳 negative `p`、
catastrophic cancellation、丢失 sign-changing bracket、用 imaginary tolerance 计数 nonreal roots，
以及在 classifier 前接纳 domain boundary。测试只冻结这些 observable behaviors，不冻结 helper、
quadrature 分段、iteration count 或测试名。

当前 corpus 包含：

- Carlson 原文 exact/numerical anchors；
- equal、near-equal、one-zero、wide-scale 与 `p->0+` ladders；
- 从命名 Gralla Ib quartic reduction 生成的 positive-argument cases；
- exact spherical double root 与 `10^-100` rational perturbation 的 near-double fallback boundary；
- II、III、IV、`E=0`、Schwarzschild `a=0`、principal null congruence、near-extremal、
  negative-spin canonicalization 与 principal-value/complex arguments 的 typed-rejection cases。

受限 pure-Kerr 研究层按以下证据条件闭合：

1. 所有 accepted topology cases 在 120/180 decimal digits 从 canonical inputs 重建，class、
   root order、turn sequence 不变；normalized continuous deltas 统一满足
   [通用证书门槛](kerr-observable-corpus.md#通用证书门槛)；
2. 每个 accepted real root 有独立 sign bracket，所有 topology/domain margin 都大于相应
   precision delta，Vieta、potential、derivative 与 identity residual 都通过同一位数预算；
3. 每个 Carlson case 同时通过 transformed definition integral、适用 exact identities、
   120/180 precision doubling 与 `mpmath` secondary triangulation；formal certificate 在构造时
   门禁所有 PASS residual，并把 `RD(x,y,z) != RD(z,y,x)` 保留为有裕量的 negative control，
   因而端到端命令不能依赖 pytest 才发现稳定但错误的结果；
4. 上述 mutations 都 RED，所有 out-of-domain/ill-conditioned cases 都稳定返回 typed fallback；
5. 输出仍是 research-private record；不新增 solver trait、render graph、public Carlson API 或
   production support claim。

锁定环境的正式 witness：

```bash
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research kerr-elliptic
```

锁定研究环境的结果为：

| 证书量 | 结果 |
| --- | ---: |
| topology 120/180 maximum normalized delta | `3.49889380552e-142` |
| topology false acceptance | `0` |
| Carlson definition residual | `1.12562807914e-203` |
| Carlson identity residual | `2.35538393693e-181` |
| Carlson 120/180 maximum normalized delta | `1.13505268495e-149` |
| Carlson `RD` x-y symmetry residual | `0.0` |
| Carlson `RD` full-permutation negative-control delta | `0.748822334264` |
| Kerr Ib radial reduction residual | `1.28580213625e-182` |

Class I 的 separated Ib inward/outward 与 negative-spin canonical equivalent 只标记为
`eligible-for-further-analysis`。Ia、II、III、IV、exact/near double 仍分别带 typed fallback reason。
这些数字是该锁定研究环境的复算证据，不是 runtime tolerance。

这只证明“高精度 oracle 足以发现目标错误”，不证明 Carlson reduction 已覆盖全部 Kerr
terminal integrals，也不证明生产 fast path 比现有 KS baseline 更快或更准。

## 5. 不能外推到 WGSL `f32` 的边界

WGSL runtime `f32` 是 IEEE-754 binary32 value set；标准没有 concrete `f64` shader type，
`AbstractFloat` 只参与 shader/pipeline creation-time expression typing
（[WGSL abstract numeric types](https://www.w3.org/TR/WGSL/#abstract-numeric-types)、
[floating-point types](https://www.w3.org/TR/WGSL/#floating-point-types)）。即使只在 positive
real 域，以下差异也阻止“120/180 位通过 => WGSL 正确”的推理：

- WGSL 不规定 rounding mode，中间结果可向上或向下 round；部分运算输入/输出可
  flush-to-zero（[§15.7.2--4](https://www.w3.org/TR/WGSL/#floating-point-evaluation)）；
- implementation 可以 reassociate，并在满足标准精度条件时 fuse expressions，改变
  cancellation 与 overflow 路径（[§15.7.5](https://www.w3.org/TR/WGSL/#floating-point-reassociation)）；
- Carlson 1995 的 truncation proof 假设 roundoff 可忽略，恰好不提供 binary32 total-error
  certificate；
- root topology 在 double-root boundary 不连续，binary32 输入量化足以把一个低裕量 sample
  移到 separatrix 另一侧；
- WGSL [numeric built-ins](https://www.w3.org/TR/WGSL/#numeric-builtin-functions) 没有
  `RF/RC/RD/RJ`；任何 shader 实现的 iteration cap、branch、normalization、register pressure
  与 divergent fallback 都是新的 production evidence。

因此后续 WGSL 工作至少需要：逐操作覆盖标准允许 rounding/FTZ/reassociation 的 f32 model、
accepted-domain 零 false-accept classifier、真实 Metal/Vulkan 的 special-function 与完整 semantic
terminal comparison，以及 correctness-approved workload 的 timing。Host/WGSL record 应保持
显式 `repr(C)`/host-shareable DTO，并让每个 invocation 独立写自己的 result；这些 layout、并行与
GPU execution tests 都不能由 Python oracle 替代
（[WGSL host-shareable types 与 memory layout](https://www.w3.org/TR/WGSL/#host-shareable-types)，
[compute invocations](https://www.w3.org/TR/WGSL/#compute-shader-execution)）。

## 6. 采用与不采用

采用：

- Gralla–Lupsasca I--IV 作为 pure-Kerr null generic topology vocabulary；
- accepted slice 只含有分离裕量的 Ib exterior segment，其余 class 仍被正确分类并 fallback；
- 正实参数 `RF/RC/RD/RJ`，以 transformed defining integral、minimal exact identities 与
  precision doubling 构成 high-precision oracle；
- critical pair、complex/PV、principal congruence、horizon coincidence 与 near-extremal 作为
  mandatory rejection/boundary corpus；
- 所有 seam 保持 Python-private，等第二个真实 consumer 和 GPU evidence 出现后再设计接口。

不采用：

- 把 `mp.polyroots` 的 imaginary tolerance 或 quartic discriminant 单独当 topology proof；
- 只凭 exterior-root count 合并 II/III/IV，或忽略 initial sign/turn count；
- 只比较 `mpmath` special functions 与自身 Legendre conversion；
- 支持 negative-`p` principal value 或 complex Carlson branches；
- 把 Carlson truncation bound、高精度证书或 CPU/GPU agreement 外推为 WGSL binary32
  correctness、统一 `RGBA16F` 质量、跨平台稳定性或性能结论。

## 7. 一手依据

- B. C. Carlson, *Numerical computation of real or complex elliptic integrals*, Numer.
  Algorithms 10 (1995)：[author manuscript](https://arxiv.org/pdf/math/9409227)；
- NIST Digital Library of Mathematical Functions：
  [definitions §19.16](https://dlmf.nist.gov/19.16)、
  [special cases §19.20](https://dlmf.nist.gov/19.20)、
  [duplication §19.26](https://dlmf.nist.gov/19.26)、
  [computation §19.36](https://dlmf.nist.gov/19.36)；
- S. E. Gralla and A. Lupsasca, *The Null Geodesics of the Kerr Exterior*：
  [Phys. Rev. D 101, 044032](https://doi.org/10.1103/PhysRevD.101.044032)，
  [corrected arXiv v3](https://arxiv.org/abs/1910.12881)；
- G. Compère, Y. Liu and J. Long, *Classification of radial Kerr geodesic motion*：
  [Phys. Rev. D 105, 024075](https://doi.org/10.1103/PhysRevD.105.024075)，
  [arXiv v2](https://arxiv.org/abs/2106.03141)；
- W3C, [WebGPU Shading Language](https://www.w3.org/TR/WGSL/)；
- `mpmath` maintainers, [elliptic functions](https://mpmath.org/doc/1.3.0/functions/elliptic.html)
  与 [precision/representation](https://mpmath.org/doc/1.3.0/technical.html)。

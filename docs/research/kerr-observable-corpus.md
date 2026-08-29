# Kerr observable corpus 与高精度证书

本文汇总 pure-Kerr 连续 observable 的具名研究 corpus、独立 BL/Mino 证书和保守
conditioning 证书。它只记录可复算的研究证据；产品支持域、验收预算和当前实现分别以
[路线图](../roadmap.md)、[验证合同](../validation.md)、[Reference 证据](../reference-implementation.md)
与 [GPU 证据](../gpu-renderer.md)为准。

**状态：表中具名 research strata 已闭合，工程层仍开放。** Positive-spin outer-source-edge 九点已经
形成“独立高精度 → CPU regular/strict → fresh WGSL binary32”证据链；critical/higher-order、
surface/capture boundary 与 negative-spin case 只有独立研究证书；near-axis、near-extreme、
near-horizon 和 near-double-root 只认证保守 fallback。除 canonical `(640,16)` 外，尚无统一
`RGBA16F` texture gate，也没有持久 artifact、第二质量方法或 production analytic solver。

## 证据矩阵

| Stratum | 固定输入 | 独立研究证据 | Rust / GPU 消费 | 仍未证明 |
| --- | --- | --- | --- | --- |
| outer source edge | `kerr-exterior-observation-v1`，`(640,12..20)` | separated BL/Mino，120/180 digits | CPU regular/strict 与 ordered fresh-binary32 全覆盖；`(640,16)` 另有 texture gate | 其余八点 texture、持久 artifact、邻域支持域 |
| critical / surface-capture | `kerr-critical-outgoing-v1`，`(33,10)` 与 `(33,11)` | double-root、event order、higher-order/winding 证书 | 尚无对应 structured Rust/GPU comparison | binary32 conditioning、texture 与 production routing |
| negative spin | `kerr-negative-spin-outgoing-v1`，`(62,7)` | signed spin/chart/emitter、source/phase/transfer 证书 | 现有 profile 只比较 terminal/branch | continuous Rust/GPU agreement |
| ill-conditioned support | near-axis、near-horizon、near-extreme、near-double-root named cases | exact rational + outward binary32 interval | production 尚未消费 classifier | accepted observable；regular control 也不等于 supported |

各行必须独立闭合。一个 stratum 的 root topology、precision 或 GPU 通过不能外推到另一行。

## 通用证书门槛

### 独立计算图

[`bl_mino`](scripts/src/gravlume_research/checks/bl_mino/) proof package 从 canonical 十进制输入重新
构造 observer event、tetrad、camera covector、constants of motion、roots 与 quadrature。它不导入
Rust crate，不读取 Reference outcome、GPU record、texture 或 fixture expected；只在 terminal 后把
BL/Mino observable 映回项目的 Kerr–Schild chart。坐标、physical-spin 与 covector convention 由
[KS ↔ BL/Mino seam](kerr-schild-mino-map.md)单独验证。

对 $b=L_z/E$、$\eta=\mathcal Q/E^2$、$\mu=\cos\theta$，独立路径使用

\[
R(r)=\left(r^2+a^2-ab\right)^2-\Delta\left[(b-a)^2+\eta\right],
\]

\[
U(\mu)=\eta+(a^2-\eta-b^2)\mu^2-a^2\mu^4.
\]

Root topology、initial radial/polar sign、turning/crossing count、signed winding 与 event order
先按离散 identity 验收，再比较 source/escape、frequency ratio、coordinate-time duration、radiance
与 diagnostics。最终坐标接近、RGB 接近或 invariant drift 很小都不能替代 path identity。

Simple turning endpoint 先解析 regularize：polar segment 使用 $\mu=\mu_+\sin\chi$，radial segment
使用 $r=r_t+s^2$ 并通过 synthetic division 形成 $R(r)/(r-r_t)$。Quadrature 因而不直接承受
平方根 endpoint singularity；提高 `mp.dps` 不是对错误 integrand 的补救。

### 数值认证

每个 accepted high-precision case 都必须：

1. 在 120 与 180 decimal digits 下从十进制输入完整重建，不复用低精度 roots；
2. 保持 terminal、root class、turning/crossing、signed winding 与 event order exact；
3. 要求每个 semantic scalar/vector lane real finite，且
   $|x_{120}-x_{180}|/\max(1,|x_{180}|)<10^{-80}$；
4. 要求 initial-null、Mino constraint 与 chart primitive normalized residual 小于
   $10^{15-p}$，其中 $p$ 是 working precision；
5. 证明 edge、event-order、root-separation 与 physical-sign margin 大于独立误差界和 consumer gate
   的保守合成值；无法证明时返回 unsupported/fallback，不放宽 tolerance；
6. 用 deterministic mutation 翻转至少一项 spin、root/turning branch、event order、phase/winding 或
   support margin，并确认 validator 拒绝。

`gravlume_research._precision` 只实现这组 finite、normalized-delta 与 digit-budget gate；物理方程、
case identity 和 signed margins 仍由所属 proof module 负责。

## Outer-source-edge 九点

固定 Observation 是 $M=1$、$a=+0.8M$、pure Kerr、ingoing Cartesian Kerr–Schild、
`1280×720` viewport、vacuum inverse-cube bolometric equatorial surface $r\in[6M,20M]$。
Center-subpixel `x=640, y=12..20` 的 exact identity 为：initial polar side `positive`、一个 radial
turning、一个 polar turning、winding `0`；Escape case 在 terminal 前有一次 equatorial crossing，
Surface case 为零次。

120/180-digit corpus 的 maximum normalized delta 是 `3.85612445201e-94`。180-digit 摘要如下；
signed margin 定义为 $20M-r_{crossing/source}$：

| Pixel y | Terminal | crossing/source radius / M | signed margin / M | travel time / M |
| ---: | --- | ---: | ---: | ---: |
| 12 | Escape | `20.164713144839971847` | `-0.164713144839971847` | `238.406047718117012` |
| 13 | Escape | `20.035241577257517150` | `-0.035241577257517150` | `238.438694378676361` |
| 14 | Surface | `19.906414902636657675` | `+0.093585097363342325` | `55.111445736567960` |
| 15 | Surface | `19.778228798382417869` | `+0.221771201617582131` | `55.006599298066207` |
| 16 | Surface | `19.650678984603292402` | `+0.349321015396707598` | `54.902474247630054` |
| 17 | Surface | `19.523761223635148545` | `+0.476238776364851455` | `54.799067017429487` |
| 18 | Surface | `19.397471319572138440` | `+0.602528680427861560` | `54.696374090040828` |
| 19 | Surface | `19.271805117804512272` | `+0.728194882195487728` | `54.594391998135353` |
| 20 | Surface | `19.146758504563225500` | `+0.853241495436774500` | `54.493117324177979` |

Margins 严格递增并只在 `13/14` 之间变号。两个 Escape case 还证明 escape sphere 先于下一次
equatorial crossing；七个 Surface case 分别认证 wrapped source azimuth、$g$、emitted/observed
bolometric intensity 与 chart travel time。完整高精度值由 proof module 输出，舍入后的独立 expectation
由 [`surface_observable_tests.rs`](../../crates/gravlume-reference/tests/surface_observable_tests.rs)消费。

## Critical 与 surface/capture pair

同一 independent graph 求得 critical sample
`y_c=10.7224312643914645870867` 和 double-root radius
`2.86515214245146438937563M`；normalized $R/R'$ residual 分别为
`3.6953213e-181` 与 `5.0417417e-182`。两侧具名结果为：

| Sample | Radial class | Terminal | radial / polar turnings | prior crossings | winding | terminal-after-first margin / Mino time |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `(33,10)` | two exterior roots | Surface | `1 / 2` | `1` | `+1` | `0.581286720264356734` |
| `(33,11)` | no exterior root | Horizon | `0 / 1` | `1` | `0` | `0.253747275109561552` |

两条 first crossing 都在 `6M` 内缘以内；surface ray 在 radial turning 后的第二次 crossing 命中
emitter，capture ray 则先到 horizon。Surface ray 的 $g$、travel time 与 observed bolometric intensity
分别为 `0.836608171867888381`、`61.2333640179788339M`、`0.130650998765441454`；120/180-digit
maximum normalized delta 为 `1.054956268933375e-110`。该证书保存 unwrapped phase，因此能发现
wrapped angle 相同但 winding 错误的回归。

## Negative-spin case

固定 case 使用 outgoing Cartesian Kerr–Schild、$a=-0.8M$、$r_o=12M$、$64\times36$ viewport、
pixel `(62,7)` 与 emitter branch `s=-1`。它是 ordinary Surface：radial/polar turnings `1/1`、
prior crossings `0`、winding `0`、two-exterior-simple-roots。180-digit 结果为：

| Source radius / M | wrapped azimuth | frequency ratio | travel time / M | observed intensity |
| ---: | ---: | ---: | ---: | ---: |
| `8.88056287724239486588` | `2.20926108917431712116` | `1.18144227618558749399` | `21.2187054680351506744` | `0.600872461017906537023` |

120/180-digit maximum normalized delta 为 `2.86108784907e-97`。Mutations 分别翻转 physical spin、
emitter branch 与 next-crossing margin，均被拒绝。该结果只关闭 independent continuous witness；
现有 negative-spin GPU profile 仍只声明 terminal/branch evidence。

## Conditioning fallback

未来 BL/Mino/Carlson accelerator 的 support report 使用 exact rational inputs 与逐操作 outward-rounded
binary32 intervals。五个正 margin 是 axis denominator $\sin^2\theta$、extremality gap
$M^2-a^2$、horizon-root separation squared、exterior horizon polynomial $\Delta(r)$ 和 minimum radial-root
separation。每项只有在 lower bound 与 exact margin 都严格超过 error envelope 时才能继续；任一项
不可证即 fallback。

| Case | 决定性证书 | Decision |
| --- | --- | --- |
| regular control | minimum margin `0.64`；maximum envelope `1.11007690430e-5`；五项均严格分离 | eligible for further analysis |
| near axis | axis denominator `1.5e-70 <= 1.17549435082e-38` | fallback |
| near horizon | horizon polynomial `2.4e-60 <= 1.13248836442e-6` | fallback |
| near extremality | extremality gap `3.0e-60 <= 5.36441859822e-7`；root separation squared `1.2e-59 <= 2.14576789404e-6` | fallback |
| near radial degeneracy | root separation `1.5e-30 <= 7.15255794148e-7` | fallback |

Named fallback corpus 的 false acceptance 为零；120/180-digit serialization delta 为
`7.16855763179e-122`。Regular control 通过只说明这五个最低 guards 没有拒绝它，不证明任何 terminal
observable 或 WGSL implementation 已正确。

## Rust 与 WGSL 消费边界

Test-only ordered corpus 复用 production 单槽 inspection 的 private kernel、96-byte record 与 strict
decoder；它不是 production batch API。对 $N$ 个 samples，logical buffers 是 `32N` request、`96N`
record 和 `96N` readback，host 在分配前检查 binding/buffer/dispatch limits。Runtime-sized array 的
元素数由 effective binding size 决定；dispatch 是 `ceil(N/64),1,1`，每个 active invocation 独占一条
ray 和一个 record，不使用共享写入、atomic 或 barrier。

Discrete words、reserved zeros 与 explicit bitcasts 使用 exact equality；continuous fields 按
[验证合同](../validation.md#5-验收预算)的各自预算比较。WGSL 允许受限的 subnormal flush、reassociation
与 fusion，`RGBA16F` 写入还会量化，因此 fresh binary32 record 不能替代最终 texture evidence。

当前证据链是：

```text
source edge: independent BL/Mino -> CPU regular/strict -> fresh WGSL binary32
canonical (640,16): above -> final RGBA16F
critical / negative spin: independent BL/Mino only
ill-conditioned cases: conservative research fallback only
```

## 复算

从仓库根目录执行：

```bash
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research bl-mino-surface
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research kerr-support
uv run --isolated --project docs/research/scripts --locked \
  pytest docs/research/scripts/tests/test_bl_mino.py \
    docs/research/scripts/tests/test_kerr_support.py
```

对应 source-edge consumers：

```bash
cargo test -p gravlume-reference --test surface_observable_tests --locked \
  source_edge_corpus_matches_the_independent_bl_mino_witness -- --exact
cargo test -p gravlume-render --lib --locked \
  gpu_trace_tests::surface::ordered_gpu_surface_edge_corpus_matches_reference_fields -- --exact
```

完整 Python 环境与 gate 见[研究工具链](python-research-tooling.md)。pytest 不替代两个端到端
scientific checks。

## 适用域与恢复条件

本记录不授权 production BL/Mino/Carlson solver、Kerr–Newman observable、near-axis/extreme accepted
result、volume checkpoint、polarization、持久 artifact 或新的 quality profile。扩展新 stratum 必须从
canonical 十进制输入重建，保存 exact path identity 与 signed margins，满足本页通用证书门槛；若要
声明 Reference/GPU 支持，还必须在同一 immutable input identity 上逐 observable 消费，并对最终 texture
producer 另设量化 gate。

Root degeneracy、axis chart、event competition 或 precision gate 失败时只能保持 unsupported/fallback。
Public corpus API、solver trait、full-frame semantic plane 和 artifact schema 只有在真实 production consumer
出现后才能提案。

## 一手来源

- [Carter 1968](https://doi.org/10.1103/PhysRev.174.1559)：Hamilton–Jacobi separability、第四守恒量与 quadrature；
- [Mino 2003](https://doi.org/10.1103/PhysRevD.67.084027)：radial/polar reparameterization；
- [Gralla & Lupsasca 2020](https://doi.org/10.1103/PhysRevD.101.044032)：Kerr exterior root topology 与 manifestly-real integrals；
- [mpmath documentation](https://mpmath.org/doc/1.3.0/)：precision、quadrature 与 root-finding 边界；
- [WGSL specification](https://www.w3.org/TR/WGSL/)：host-shareable layout、runtime arrays、workgroup synchronization 与 floating-point semantics。

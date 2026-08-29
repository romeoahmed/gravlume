# Kerr critical curve 与 surface/capture 高精度证书

本文记录一个 pure Kerr、outgoing Kerr–Schild observation 的相邻
surface/capture pair，以及其中 higher-order surface、signed winding 和 radial
double-root separatrix 的独立 BL/Mino 高精度证书。本文是可复算研究证据，不定义
production 支持域、GPU quality policy、fixture/schema 或 public API；这些合同仍分别以
[数学物理](../physics.md)、[验证](../validation.md)、[渲染准入](../rendering.md)和
[路线图](../roadmap.md)为准。

**状态：P1 surface/capture 与 P2 critical/higher-order 的纯研究层已闭合。** Rust
reference、WGSL binary32、最终 `RGBA16F`、真实 Metal/Vulkan 和产品支持域仍未由本文闭合。

## 1. 可证伪问题与固定输入

目标不是寻找一张“看起来临界”的图，而是从同一不可变 observation 证明相邻 sample
分别位于 radial separatrix 两侧，并保存能发现 terminal、event order、phase 与 winding
回归的连续 observable。

固定输入为：

| 字段 | 值 |
| --- | --- |
| spacetime | pure Kerr，`M=1`、physical spin `a=0.8M` |
| chart | outgoing Kerr–Schild，`s=-1` |
| observer | `r=30M`、`θ=π/3`、chart azimuth `0`，stationary observer |
| viewport | `64×36`、vertical FOV `π/4`、center subpixel |
| surface | equatorial inclusive interval `[6M,20M]` |
| named samples | `(33,10)` 与 `(33,11)` |

研究脚本从这些输入独立重建 Kerr–Schild metric、observer tetrad、photon covector、
`E`、`ξ=L_z/E` 与 `η=Q/E²`；它不导入 Rust trajectory、GPU record 或扫描结果。

## 2. 分离方程与 horizon 正则化

以

\[
P(r)=r^2+a^2-a\xi,\qquad
K=(\xi-a)^2+\eta
\]

写 pure-Kerr radial potential：

\[
R(r)=P(r)^2-\Delta(r)K,\qquad
\Delta=r^2-2Mr+a^2.
\]

每个 sample 的 exterior local minimum 由 `R'(r_min)=0` 独立求解，并以
`R(r_min)` 的符号分类：负值给出两个 exterior simple roots 与 scattering branch，正值给出
无 exterior root 的 monotonic capture branch。两 sample 之间再对 sample-center `y` 求解
`R(r_min(y),y)=0`；同时验证 `R'=0`、`R''>0`，因此 root class 是
`exterior-double-root`，而不是仅凭像素 terminal 推测 critical curve。

Capture case 使用 outgoing chart，并采用
[KS/BL seam §3.3](kerr-schild-mino-map.md#33-outgoing-capture-的-horizon-finite-radial-combination)
中由 SymPy exact `cancel` 证明的 horizon-finite time/azimuth combined primitive。这避免分别
计算两个在 `r_+` 发散、但差值有限的 primitive。Polar motion 则按 simple turning endpoint
做变量替换；每个 equatorial crossing、turning 与 terminal 的 Mino duration 显式分段累计，
不由最终坐标反推 branch。

## 3. 180 位证书结果

Critical point 为：

| observable | 180-digit 重建摘要 |
| --- | ---: |
| critical sample `y_c` | `10.7224312643914645870867` |
| double-root radius | `2.86515214245146438937563 M` |
| normalized `R` residual | `3.6953213e-181` |
| normalized `R'` residual | `5.0417417e-182` |
| `R''` | `45.1235406767685266371909` |

两侧 discrete identity 与 signed classification 为：

| sample | `y-y_c` | `R(r_min)` | exterior roots | terminal | radial/polar turnings | crossings before terminal | winding |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `(33,10)` | `-0.222431264391464587` | `-5.33932970970068354` | `2` | Equatorial Surface | `1 / 2` | `1` | `+1` |
| `(33,11)` | `+0.777568735608535413` | `+12.1140801603194982` | `0` | Horizon | `0 / 1` | `1` | `0` |

Event order 由绝对 Mino 时刻及其差值共同保存：

| sample | first equatorial `λ_M` | terminal `λ_M` | signed terminal-after-first margin | first-crossing radius | terminal radius |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(33,10)` | `0.482422908499503382` | `1.06370962876386012` | `0.581286720264356734` | `3.46682062959464423 M` | `9.32129978253847620 M` |
| `(33,11)` | `0.550028508498244261` | `0.803775783607805813` | `0.253747275109561552` | `2.52447422544180659 M` | `r_+=1.6 M` |

两条 first crossing 都严格位于 `6M` 内缘以内，因此不是 surface event。Scattering case
在 radial turning 后的第二次 equatorial crossing 命中 surface；capture case 在第一次
non-surface crossing 后先抵达 horizon。

连续 phase/transport 摘要为：

| sample | unwrapped / wrapped chart azimuth | `g` | chart travel time | observed bolometric intensity |
| --- | --- | ---: | ---: | ---: |
| `(33,10)` | `5.49040860557681712 / -0.792776701602769357` | `0.836608171867888381` | `61.2333640179788339 M` | `0.130650998765441454` |
| `(33,11)` | `3.18345860557315797 / -3.09972670160642851` | 不适用 | `38.7687668382615638 M` | 不适用 |

Winding 由 unwrapped chart phase、observer/terminal oblate twist 与 canonical cycle 一起求得；
wrapped angle 本身不足以区分 image sequence。

## 4. Precision doubling 与 mutation RED

120 与 180 decimal digits 两次都从固定 observation 重新构造 tetrad、constants、roots 和
全部 quadrature，不复用低精度 root。对 critical point、signed distances/classification、
event times/margins、source/horizon position、wrapped/unwrapped phase、travel time、transfer
和 turning diagnostics 逐 lane 比较，maximum normalized delta 为
`1.054956268933375e-110`，小于预注册的 `1e-80` 门槛。

此外，`kerr-schild-map` scientific witness 对 outgoing capture 的有限 time/azimuth
combination 运行 exact SymPy identity；pytest 单独调用同一 proof seam，防止 CLI 漏接。

Deterministic mutation tests 证明证书会拒绝：

- 改变 `critical_sample_y` 却保留旧 signed distances；
- 改变 horizon-after-crossing margin 却保留两个绝对 Mino 时刻；
- 给 higher-order unwrapped phase 增加 `2π` 却保留旧 winding；
- 改变 terminal stratum、root count、turning/crossing count 或 signed classification。

这组测试观察语义恒等式，不冻结 quadrature helper、求根迭代次数或 CLI 文本格式。

## 5. 决策与边界

本证书关闭[路线图连续字段工作](../roadmap.md#连续字段证据与质量政策)中以下纯研究缺口：

- surface/capture competing-event pair 的独立 event-order witness；
- critical curve 两侧的 signed distance、double-root 与 root-topology witness；
- 一个有两次 polar turning、一次先前 equatorial crossing 且 winding `+1` 的 higher-order
  surface continuous witness。

它不关闭结构化 Rust/GPU comparison、binary32 conditioning、最终 texture quantization、统一
quality method、artifact 或跨平台执行。尤其不能把 arbitrary-precision 稳定性外推为 WGSL
`f32` 稳定性；WGSL 允许规定范围内的 reassociation、fusion 与 flush-to-zero，且 `vec4` ABI/并行
dispatch 的既有 test seam 仍需由真实 GPU consumer 单独验收。

## 6. 复算命令

从仓库根目录执行：

```bash
/Users/victor/.local/bin/uv run --isolated \
  --project docs/research/scripts --locked \
  gravlume-research bl-mino-surface

/Users/victor/.local/bin/uv run --isolated \
  --project docs/research/scripts --locked \
  pytest docs/research/scripts/tests/test_bl_mino.py
```

完整 Python gate 仍以[研究工具链](python-research-tooling.md)列出的全部 scientific witness、
pytest 与 Ruff 命令为准。

## 7. 一手依据

- Carter 的 Hamilton–Jacobi separation：
  [Phys. Rev. 174, 1559 (1968)](https://doi.org/10.1103/PhysRev.174.1559)；
- Mino parameter：
  [Phys. Rev. D 67, 084027 (2003)](https://doi.org/10.1103/PhysRevD.67.084027)；
- Kerr null-geodesic root classification：
  [Gralla–Lupsasca, Phys. Rev. D 101, 044032 (2020)](https://doi.org/10.1103/PhysRevD.101.044032)；
- arbitrary-precision quadrature、verified root finding 与 polynomial conditioning：
  [`mpmath` 1.3.0 文档](https://mpmath.org/doc/1.3.0/)；
- host-shareable layout、runtime arrays、floating-point 与 memory semantics：
  [W3C WGSL](https://www.w3.org/TR/WGSL/)。

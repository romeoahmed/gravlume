# 负自旋连续字段证书

本文记录一个固定 pure-Kerr 负自旋 surface ray 的独立 BL/Mino 高精度证书，权威范围仅限该具名输入的数学与数值研究证据。产品行为、支持域和验收预算仍分别以[数学物理](../physics.md)、[验证合同](../validation.md)、[Reference 证据](../reference-implementation.md)和[GPU 证据](../gpu-renderer.md)为准；本文不定义 production solver、public interface、versioned fixture 或 GPU 支持。

## 1. 研究问题与输入

[路线图](../roadmap.md#连续字段证据与质量政策)要求正负 Kerr spin 都有能发现 branch、phase 或 radiance 回归的独立 witness。现有 `negative-spin-near` profile 只比较 Reference/GPU terminal 与 branch（[`surface.rs`](../../crates/gravlume-render/src/gpu_trace_tests/surface.rs)），没有独立连续 source/phase/transfer 证据。

本证书预注册其中一个远离 source edge 和 radial separatrix 的 sample，只读取 profile 的十进制输入，不读取任何 Rust trace 或 GPU 输出：

| 项 | 固定值 |
| --- | --- |
| spacetime | pure Kerr，$M=1$，physical spin $a=-0.8M$ |
| chart | outgoing Cartesian Kerr–Schild |
| observer | $r_o=12M$，$\theta_o=\pi/3$，chart azimuth $0$，stationary，$\nu_o=1$ |
| view | $64\times36$，vertical FOV $\pi/4$ |
| sample | pixel `(62, 7)`，center subpixel `(0.5, 0.5)` |
| emitter | equatorial circular branch $s=-1$，$6M\le r\le20M$，inverse-cube bolometric source，vacuum |

observer tetrad、photon covector、Boyer–Lindquist constants、roots、turning quadratures 和 transfer 都由 [`bl_mino.py`](scripts/src/gravlume_research/checks/bl_mino.py) 从这些输入重建。该 private in-process module 是测试 seam；没有第二个 production consumer，因此没有引入 solver trait 或 public adapter。

## 2. 方程与符号约定

Carter 的 Hamilton–Jacobi 分离给出第四常数和 radial/polar quadrature（[Carter 1968](https://doi.org/10.1103/PhysRev.174.1559)）；Mino parameter 把两者解耦（[Mino 2003](https://doi.org/10.1103/PhysRevD.67.084027)）。对归一化常数 $\xi=L_z/E$、$\eta=Q/E^2$，脚本直接验证

$$
R(r)=\left(r^2+a^2-a\xi\right)^2
-\Delta\left[(\xi-a)^2+\eta\right],
\qquad
\Delta=r^2-2Mr+a^2,
$$

$$
\Theta_\mu(\mu)=
\eta+(a^2-\eta-\xi^2)\mu^2-a^2\mu^4.
$$

root topology 与 manifestly real quadrature 的适用边界采用 Gralla–Lupsasca 对 Kerr exterior 的分类（[Gralla–Lupsasca 2020](https://doi.org/10.1103/PhysRevD.101.044032)）。polar potential 只依赖 $a^2$，但 radial potential、chart map、azimuth phase 和 emitter frequency 保留 signed $a$；turning regularization 中出现的是 $|a|$，不能把整个 case 当作正自旋复本。

equatorial circular emitter 使用与 Reference 相同的 signed branch 方程（[`surface.rs`](../../crates/gravlume-reference/src/surface.rs)）：

$$
\Omega_s=\frac{s\sqrt{Mr}}{r^2+s a\sqrt{Mr}},\qquad s=-1.
$$

它是 Kerr circular-orbit 公式的 signed 表达；原始轨道公式见 Bardeen、Press 与 Teukolsky（[1972](https://doi.org/10.1086/151796)）。本例 $a<0$ 且 $s=-1$，因此 $\Omega_s<0$。stationary observer 已归一化为 $\nu_o=1$，所以

$$
g=\frac{\nu_o}{\nu_e}
=\frac{1}{u^t E(1-\Omega_s\xi)},
\qquad
I_o=g^4 I_e,
\qquad
I_e=\left(\frac{r_s}{6M}\right)^{-3}.
$$

## 3. 具名结果

锁定环境为 Python 3.14、`mpmath 1.3.0`；`mpmath` 的 arbitrary-precision quadrature、root finding 和 polynomial conditioning 行为见[官方文档](https://mpmath.org/doc/1.3.0/)。以下结果来自 180 decimal digits 的完整重建，表中保留足够数字用于人工审阅：

| 字段 | 证书值 |
| --- | --- |
| terminal / branch | `equatorial-surface`; initial polar `positive`; radial/polar turns `1/1`; prior crossings `0`; winding `0` |
| radial roots | `two-exterior-simple-roots`; count `2` |
| stationary radius / barrier | `5.01273223624256160948M`; `-385.759943552513808389` |
| outer radial turning | `6.74848843631938849164M`，距 horizon `5.14848843631938849164M` |
| source Mino duration | `0.267742403810439436390` |
| observer→radial-turn duration | `0.154488593994636010454` |
| source after radial-turn margin | `0.113253809815803425936` |
| next crossing after source margin | `0.402698639240499561899` |
| source radius | `8.88056287724239486588M` |
| source inner / outer margin | `2.88056287724239486588M` / `11.1194371227576051341M` |
| source azimuth, unwrapped / wrapped | `2.20926108917431712116` / `2.20926108917431712116` rad |
| emitter angular velocity | `-0.0366779731763039523299 M^-1` |
| frequency ratio $g$ | `1.18144227618558749399` |
| coordinate-time duration | `21.2187054680351506744M` |
| emitted / observed intensity | `0.308412712127873614280` / `0.600872461017906537023` |
| $E,\xi,\eta$ | `0.912972241081606810934`, `-6.40729511819284033334`, `20.1296148462242169353` |

180 位重建的 normalized residual 为：initial null `1.3991e-181`、separated Mino constraints `5.9632e-181`、chart primitive `1.2050e-180`。它们均小于预注册的 $10^{15-p}=10^{-165}$ gate。

## 4. 精度与证伪能力

证书在 120 和 180 decimal digits 下从十进制输入完整重建，不复用低精度 roots。所有 semantic scalar 的 maximum normalized delta 为

$$
2.86108784907\times10^{-97}<10^{-80}.
$$

这是一项 convergence witness，不是 interval-arithmetic proof；root count、turning signs、terminal、crossing count 和 winding 仍使用 exact identity 验收。

TDD 的第一轮 RED 因 `_negative_spin_surface_witness` 不存在而在 collection 阶段失败；GREEN 后 mutation test 分别把 physical spin 改为 `+0.8M`、把 emitter branch 改为 `+1`，两者都被 `_UnsupportedWitnessError` 拒绝。第二轮 RED 要求尚不存在的 120/180 位 certificate；GREEN 后同时锁住固定 pixel、root topology、事件顺序、signed phase 与 transfer 字段。审阅阶段再用一个仍为正的 stale next-crossing margin 形成 RED，validator 随后从 signed $(a,\xi,\eta)$ 重新构造 polar/radial durations 并拒绝该 mutation。测试没有冻结 quadrature 分段或 private helper。

## 5. 复算

```bash
/Users/victor/.local/bin/uv run --isolated \
  --project docs/research/scripts --locked \
  gravlume-research bl-mino-surface

/Users/victor/.local/bin/uv run --isolated \
  --project docs/research/scripts --locked \
  pytest docs/research/scripts/tests/test_bl_mino.py
```

两条命令必须分别得到 `RESULT=PASS` 和完整测试通过；pytest 不能替代 scientific witness。

## 6. 证据边界

本记录闭合的只有路线图审计中的 **P0 negative-spin independent continuous-field research layer**。它证明该具名 ray 的 signed spin、chart、emitter branch、source phase、transfer 和 root/event identity 可以被独立复算；它不证明：

- Rust Reference 或 fresh WGSL producer 已逐 continuous field 与此证书 agreement；
- binary32/WGSL 在附近区域稳定，或 negative-spin profile 已扩大为 production support domain；
- near-axis、near-extreme、Kerr–Newman、volume checkpoint 或 polarization 可复用该证书；
- CPU/GPU agreement、截图或小 invariant drift 足以代替独立数学证据。

后续工程必须在同一 immutable observation identity 上比较 terminal、branch、source、phase、time、frequency、radiance 和 diagnostics；任何 unsupported 或 conditioning 不可证明的 case 仍须保守 fallback。

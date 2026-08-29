# 近轴/近极值保守 fallback 证书

本文记录未来 Kerr 专用解析/半解析 accelerator 在 near-axis、near-horizon、near-extreme 与 near-double-root 输入上的保守 conditioning 证书。其权威范围仅限 research-only exact/binary32 condition report；产品支持域、production routing、WGSL solver 与验收预算仍分别以[渲染准入](../rendering.md)、[验证合同](../validation.md)、[Reference 证据](../reference-implementation.md)和[GPU 证据](../gpu-renderer.md)为准。

**状态：路线图审计 P3 的纯研究层已闭合。** 所有预注册病态输入都得到 `fallback`，named corpus 的 false acceptance 为零；regular control 只标记为 `eligible-for-further-analysis`，不标记为 supported。

## 1. 问题与停止边界

现有 production 路径是全域 Cartesian Kerr–Schild WGSL RK4；exact axis 已有独立解析分支，host 也会拒绝 binary32 packing 改变 extremality 分类的输入（[`kerr_schild_dynamics.wgsl`](../../crates/gravlume-render/src/shaders/kerr_schild_dynamics.wgsl)、[`trace/input.rs`](../../crates/gravlume-render/src/trace/input.rs)）。未来 BL/Mino/Carlson accelerator 仍会面对 axis chart、horizon roots、radial root topology 与 near-critical conditioning，不能因为高精度恒等式成立就默认有限精度可用。

本证书只回答一个单向问题：**具名 margin 是否已经落入一个 outward-rounded binary32 error envelope，因而必须 fallback？** 它不尝试证明 regular case 的完整 shader accuracy，也不修改 Rust/WGSL routing。

## 2. 五个正 margin

对 subextremal pure Kerr 的 exact rational 输入，私有 classifier 同时计算：

$$
u=\sin^2\theta,
\qquad
e=M^2-a^2,
\qquad
h^2=(r_+-r_-)^2=4e,
$$

$$
\Delta(r)=(r-M)^2-e,
\qquad
s_r=\min_{i\ne j}|r_i-r_j|.
$$

这里 $u$ 是 axis chart denominator，$e$ 区分 sub/extreme/superextreme，$h^2$ 避免在证书图中引入 `sqrt`，$\Delta>0$ 与已知 $r>M$ 一起给出 exterior horizon separation，$s_r$ 约束 radial-root degeneracy。Kerr horizon 与 $\Delta$ 的规范定义见[数学物理合同](../physics.md#3-horizonergoregion-与观察域)。

每个 exact margin $m>0$ 都有一个严格包含它的 binary32 区间 $[l,u]$，并保存

$$
B=\max(|m-l|,|u-m|).
$$

guard 只有在 $l>0$ 且 $m>B$ 时才获证；等于 error bound 也拒绝。五项任一不可证，整个 report 即为 `fallback`，没有默认接受或经验 epsilon。

## 3. Arithmetic envelope

[`_interval_f32.py`](scripts/src/gravlume_research/_interval_f32.py) 是 Kerr capture 与本证书的共同 research-only 深模块：

1. exact rational input 先被相邻 binary32 值 outward bracket；
2. `add/sub/mul/square` 按预注册计算图逐步求上下界；
3. 每一步再次向相邻 binary32 扩张，并覆盖允许的 subnormal flush-to-zero；
4. 每个结果都反查 exact `Fraction` 必须位于区间内。

WGSL `f32` 是 IEEE-754 binary32，但运行时不指定单一 rounding mode；规范还允许规定范围内的 flush-to-zero、reassociation 与 fusion（[W3C WGSL floating-point evaluation](https://www.w3.org/TR/WGSL/#floating-point-evaluation)、[accuracy](https://www.w3.org/TR/WGSL/#floating-point-accuracy)、[reassociation and fusion](https://www.w3.org/TR/WGSL/#reassociation-and-fusion)）。因此该 envelope 只对应显式顺序的候选 guard graph。病态 case 被拒绝是可采用的负证据；regular control 通过只说明它没有被这组最低 guard 排除，仍需完整 algorithm、observable 与 backend 证书。

## 4. 具名结果

锁定环境中的 [`kerr_support.py`](scripts/src/gravlume_research/checks/kerr_support.py) 得到：

| case | exact margin | absolute error bound | decision / failed guard |
| --- | ---: | ---: | --- |
| `regular-control` | 最小为 $u=0.75$ | 对应 $5.96046447754\times10^{-8}$ | `eligible-for-further-analysis`; none |
| `near-axis` | $u=1.5\times10^{-70}$ | $1.17549435082\times10^{-38}$ | `fallback`; axis denominator |
| `near-horizon` | $\Delta=2.4\times10^{-60}+2.25\times10^{-120}$ | $1.13248836442\times10^{-6}$ | `fallback`; horizon polynomial |
| `near-extremality` | $e=3.0\times10^{-60}-2.25\times10^{-120}$ | $5.36441859822\times10^{-7}$ | `fallback`; extremality gap |
| `near-extremality` | $h^2=1.2\times10^{-59}-9.0\times10^{-120}$ | $2.14576789404\times10^{-6}$ | `fallback`; horizon-root separation squared |
| `near-radial-degeneracy` | $s_r=1.5\times10^{-30}$ | $7.15255794148\times10^{-7}$ | `fallback`; radial-root separation |

`false_acceptance_count=0` 只量化这四个预注册 fallback label；它不外推到任意参数空间或未来 production classifier。

## 5. 精度与证伪能力

输入、物理 margin 与 binary32 endpoint 都保存为 exact `Fraction`。为验证报告序列化和复算没有先经默认精度丢位，全部 margin、endpoint 与 error bound 又在 120/180 decimal digits 下独立转换；maximum normalized delta 为

$$
7.16855763179\times10^{-122}<10^{-80}.
$$

这不是用高精度替代 interval proof：分类本身由 exact containment 与严格不等式决定。

TDD 的第一轮 RED 因 `gravlume_research.checks.kerr_support` 尚不存在而在 collection 阶段失败。GREEN 后，具名测试要求四类病态原因分别拒绝；mutation test 把 regular axis 的 `absolute_error_bound` 精确改成其 margin，严格分类立即变为 `fallback`。共享 `IntervalF32` 的原有 Hypothesis corpus 同时复跑 20,000 个 primitive 和 5,000 个 Bernstein case，防止提取深模块改变既有 enclosure 语义。

## 6. 复算

```bash
/Users/victor/.local/bin/uv run --isolated \
  --project docs/research/scripts --locked \
  gravlume-research kerr-support

/Users/victor/.local/bin/uv run --isolated \
  --project docs/research/scripts --locked \
  pytest docs/research/scripts/tests/test_kerr_support.py \
    docs/research/scripts/tests/test_kerr_capture.py
```

第一条必须输出五个具名 decision、`false_acceptance_count=0` 与 `RESULT=PASS`；pytest 不能替代端到端 scientific witness。

## 7. 证据边界

本记录闭合的只有路线图审计 **P3 conservative fallback certificate**。它不证明：

- 任意 `eligible-for-further-analysis` case 已满足 terminal/source/time/frequency/radiance budget；
- WGSL 可安全运行 Carlson functions、root classifier 或 BL chart，或当前 GPU path 已消费这些 guard；
- axis/horizon/extreme case 的高精度渲染已经存在；
- 一个显式运算图的 interval 能覆盖未固定的 reassociation、资源、ABI、dispatch 或 backend 行为。

未来 production classifier 必须复制其实际 arithmetic graph 的完整 outward proof，并在任何 margin、domain 或 operation accuracy 不可证时 typed fallback 到现有 Cartesian Kerr–Schild 路径。

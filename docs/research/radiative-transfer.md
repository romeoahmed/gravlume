# 标量辐射传输证据

本文保存 scalar surface transport 的独立 oracle、fixture 勘误和适用边界。它不重复 production
方程：frequency、$g^3/g^4$、homogeneous slab、blackbody bands 与 circular emitter 的权威定义在
[数学物理合同](../physics.md#7-frequency-与-radiative-transfer)，fixture 与 tolerance 在
[验证合同](../validation.md)。当前实现状态分别见 [Reference 证据](../reference-implementation.md)
和 [GPU 证据](../gpu-renderer.md)。

**状态：具名 scalar boundary models 已采用；空间变化 volume、scattering、stable-disk 声明与
polarization 未实现。** 本记录只说明现有合同为何有独立证据，以及哪些外推仍被拒绝。

## 证据矩阵

| Case | 独立检查 | 当前消费 | 未覆盖域 |
| --- | --- | --- | --- |
| vacuum bolometric | spectral-to-bolometric symbolic identity | v2 fixture、CPU/GPU fields、canonical texture gate | arbitrary spectrum |
| blackbody shift | Planck scaling、normalization 与 band integral | v3 fixtures、4097-entry LUT oracle | 三个具名 boxcar 之外的 color model |
| pure absorption | analytic exponential limit | v3 fixture 与 property tests | spatial coefficient |
| constant slab | analytic source-function solution、thin/thick limits | v3 fixture 与 CPU/GPU agreement | ordered volume checkpoints |
| pure emission | zero-absorption limit | v3 fixture | emission field |
| invalid coefficients | negative/non-finite rejection | domain/property tests | 不用 clamp 构造结果 |

CPU oracle 使用 binary64 或 arbitrary precision analytic expressions；GPU gate 比较 scene-linear
physical channels，不比较 tone-mapped RGB。Named fixtures 约束 canonical cases，property tests 约束
代数域；两者不能互相替代。

Circular emitter 只验证[数学物理合同定义的运动学与声明边界](../physics.md#8-equatorial-circular-emitter-与-disk-边界)。
它不提供 radial/vertical stability 证据，也不授权 Novikov–Thorne/Page–Thorne disk。

## 高精度 oracle 与 fixture 勘误

[`scalar_transport.py`](scripts/src/gravlume_research/checks/scalar_transport.py) 在 80 decimal digits
下复算 invariant/blackbody identities、slab limits 与 partition、Planck normalization、binary64
thin-limit cancellation 和 LUT midpoint error，并生成 v3 spectral expected。

旧实现先按 mpmath 默认 precision 创建十进制常量，再提高 working precision；丢失的位数无法恢复。
6000 K red-band witness 因而出现 `6.21612160649749e-20` absolute error，远大于独立
`1e-70` oracle gate。修正只更新四份 v3 artifact 的 spectral expected；schema、profile、canonical
inputs、tolerance、geometry 与 bolometric expected 均未改变，符合
[fixture 勘误合同](../validation.md#6-fixture-envelope)。

复算：

```bash
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research scalar-transport
uv run --isolated --project docs/research/scripts --locked \
  pytest docs/research/scripts/tests/test_scalar_transport.py
```

## 不支持域与重开条件

- Spatially varying volume 必须先定义 ordered checkpoints、local fluid frequency、accepted-step
  commit 和独立 analytic/convergence cases；total-$\tau$ slab 不是实现证明。
- Scattering 与 Stokes/Faraday 需要新的 state、source term、basis/normalization 和验证合同。
- 普通 RGB 只有引入具名 spectral/bandpass interpretation 后才能参与 physical frequency shift。
- Stable-disk 声明必须先把 orbit stability 与 source-domain gate 写入权威合同并由真实 consumer 消费。

## 一手来源

- [Lindquist 1966](https://doi.org/10.1016/0003-4916%2866%2990207-7)：relativistic kinetic transfer；
- [Younsi, Wu & Fuerst 2012](https://doi.org/10.1051/0004-6361/201219599)：covariant transfer 与 formal solution；
- [Bardeen, Press & Teukolsky 1972](https://doi.org/10.1086/151796)：Kerr circular orbit 与 ISCO；
- [Pugliese, Quevedo & Ruffini 2013](https://doi.org/10.1103/PhysRevD.88.024042)：Kerr–Newman circular/stability domain；
- [mpmath precision management](https://mpmath.org/doc/1.3.0/general.html#precision-management)：高精度常量构造边界。

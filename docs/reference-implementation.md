# Reference 实现与证据

本文是 `gravlume-domain` 与 `gravlume-reference` 的当前证据清单。连续模型和验收阈值分别由[数学物理合同](physics.md)与[验证合同](validation.md)定义；本页不复制 profile 数值，也不把 `f64` 结果称为绝对 ground truth。

## 已实现

| 领域        | 当前实现                                                                                                                      |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| validation  | `PhysicalSceneInput → PhysicalScene → Observation` 原子验证；稳定协议是 issue code 与 field path                              |
| view ray    | `Observation::initial_ray` 独占 top-left pixel/subpixel 到 future-directed Photon Momentum 的映射                             |
| spacetime   | canonical `(t,x,y,z,p_t,p_x,p_y,p_z)` `f64` Cartesian Kerr–Schild Hamilton system                                             |
| parameters  | 对实际 binary64 bit pattern 精确判定 extremality；axis geometry 使用解析极限                                                  |
| integration | 七次求值、FSAL 的 Dormand–Prince 5(4)，按 position/momentum group 归一化误差                                                  |
| events      | accepted-step quartic dense output；只在 bracket 内定位 horizon、escape、equatorial surface 与 singularity guard              |
| outcomes    | typed terminal、counters、step range、event bracket/residual、invariant drift、turning radius、travel time 与 azimuth advance |
| fixtures    | 严格 v1 TOML、1 MiB 上限、unknown-field rejection、版本化 identity/profile 与高精度十进制字符串                               |
| comparison  | 先验证 input/profile identity，再比较 regular/strict observable；配置错误与数值 issue 分离                                    |
| batch       | 有界专用 Rayon pool；单轨迹顺序确定，输出保持 input order                                                                     |

Backward Trace 使用负 affine traversal，不改写物理 momentum。coordinate duration 从 dense/local step increment 累计，不由两个绝对时间相减。step/reject exhaustion 和 numerical failure 不伪装成物理 terminal。

## 自动化证据

`cargo test --workspace --all-targets --locked` 当前覆盖：

- oblate radius、null principal covector、rank-one inverse 与 Minkowski/Schwarzschild/RN/Kerr/KN 特殊极限；
- extremal/superextremal binary64 分类、horizon、`g_tt` 与 stationary observer 适用域；
- Observer Frame Gram/orientation、view sample 与 frequency-scale-invariant initial ray；
- weak-field `4M/b`、regular `b=6` 80 位 fixture 和 `sqrt(27)±10^-3` near-critical branch；
- regular/strict identity、termination、event position、escape direction、travel time 与 invariant drift gates；
- horizon、escape、equatorial surface、singularity guard、step exhaustion 与 same-step ambiguity；
- 默认 Kerr Observation 的非临界收敛、negative-affine turning localization 与 batch ordering。

这些测试不需要 GPU。fixture producer、80 位 observable 和 tolerance 由 [`fixtures/v1`](../crates/gravlume-reference/fixtures/v1/) 保存。

## 适用域

- trajectory fixture 主要覆盖 equatorial Schwarzschild 和默认 exterior Kerr Observation 的非临界样本；
- published Kerr/Kerr–Newman trajectory、独立 chart/state representation、near-axis Killing-tensor overlap 与更广参数扫描尚未闭合；
- finite escape sphere 是数值边界，不能解释为无穷远精确 observable；
- renderer 从同一 validated `Observation` 独立构造 GPU initial ray，不消费 CPU trajectory；
- near-critical GPU agreement、source anchor、frequency ratio 与 transport 仍属于后续证据。

历史 raw-radius/reciprocal-radius `f32` 条件性实验已移入[研究记录](research/mino-step-selection.md)，不属于 reference fixture 合同。

## 主要来源

- [Dormand–Prince 1980](https://doi.org/10.1016/0771-050X%2880%2990013-3) 与 [Shampine 1986](https://doi.org/10.1090/S0025-5718-1986-0815836-3)；
- [SciPy RK45 source](https://github.com/scipy/scipy/blob/v1.18.0/scipy/integrate/_ivp/rk.py) 作为维护实现的系数对照；
- [Brent 的 bracketed root 文献页](https://maths-people.anu.edu.au/~brent/pub/pub006.html)；
- [Serde container attributes](https://serde.rs/container-attrs.html)、[`toml::from_str`](https://docs.rs/toml/1.1.4/toml/fn.from_str.html)与 [`rayon::ThreadPool`](https://docs.rs/rayon/1.12.0/rayon/struct.ThreadPool.html)。

精确依赖版本仍以 `Cargo.lock` 为准。

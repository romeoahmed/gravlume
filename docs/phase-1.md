# Phase 1 实现与证据

本文记录当前 `gravlume-domain` 与 `gravlume-reference` 的实现边界。连续公式和阈值仍分别以[数学物理合同](physics.md)与[验证合同](validation.md)为准；这里不复制或改写 profile。

## 已实现闭环

- `PhysicalSceneDraft → PhysicalScene → Observation` 是原子 validation seam；稳定字段为 issue code、field path 与 severity。Observer Frame 保存 Gram residual、orientation determinant 和 up-axis fallback 诊断。
- `ViewportSample` 只保存 pixel/subpixel coordinates；`Observation::initial_ray` 针对自己的 projection 重新验证并独占 top-left sample 到 future-directed Photon Momentum 的 CPU 映射。`ReferenceRequest` 在绑定 Observation 时解析 initial ray，`ReferenceInstrument` 只通过该接口构造 backward trace，负 affine traversal 不改写物理 momentum。
- `KerrNewmanSpacetime` 使用 canonical `(t,x,y,z,p_t,p_x,p_y,p_z)` `f64` 状态与闭式 Cartesian Kerr–Schild Hamilton RHS；metric/radius denominator 失败是 typed error，不 clamp 方程。
- `ReferenceTracer` 在 seam 强制 v1 的 `M=1` 归一化，`ReferenceInstrument` 额外强制 `omega_obs=1`；随后使用七次求值、FSAL 的 Dormand–Prince 5(4)，按 position/momentum group 归一化误差。拒步不提交 state/event side effect。
- accepted step 保存 quartic dense output；horizon、escape、equatorial surface 与 singularity guard 只在 bracket 内二分定位。同一步 candidate 在 tie tolerance 内全部保留，并按 `singularity → horizon → emitter → escape` 排序。
- outcome 分开记录绑定完整有效输入的 identity、terminal、accepted/rejected/RHS counters、实际 min/max step、event bracket/residual、null/E/Lz/Carter drift、dense-localized turning radius、Hamilton RHS terminal traversal direction、非负 coordinate travel duration 与 azimuth advance。turning point 按 affine traversal direction 检测并使用 dense output 定位；无转向的 capture 不以最小采样半径冒充 turning point，step/reject exhaustion 和 numerical failure 不伪装成物理 terminal。
- v1 TOML 使用 `deny_unknown_fields` 并限制为 1 MiB；未知字段/enum/schema/profile、与 profile 不一致的固定 event 值、NaN/Inf、负零与非法物理值在 seam 拒绝。80 位十进制保留为字符串到解析 seam，运行时明确转换为 `f64`，不声称保留 80 位算术。
- `ReferenceComparison::baseline_v1` 在计算预算前验证 regular/strict policy roles 与 input ID；配置错误返回 `ComparisonError`，只有有效配对才产生数值 `ComparisonIssue`。
- `ReferenceBatch` 建立最多 256 worker 的专用 Rayon pool；单条轨迹顺序确定，indexed parallel collect 保持 input order。

## 当前自动化证据

`cargo test --workspace --all-targets --locked` 当前覆盖：

- oblate radius identity、rank-one inverse、Schwarzschild、Reissner–Nordström、Kerr–Newman、extremal/superextremal 与趋近 Minkowski 的特殊极限；
- 默认 Kerr stationary observer 的 horizon、`g_tt`、frame Gram/orientation，以及跨 projection sample 重新解析和 frequency-scale-invariant viewport null/frequency seam；
- weak-field Schwarzschild deflection 对 leading `4M/b`、regular `b=6` 80 位 fixture、`sqrt(27)±10^-3` near-critical escape/capture 分类；
- regular/strict 的 policy/input identity、termination、event position、escape direction、travel time 与 invariant drift comparison gate；
- horizon、escape、equatorial surface、singularity guard、step exhaustion 和 same-step ambiguity；
- 默认 Kerr Observation 的非临界 backward ray regular/strict 收敛、negative-affine turning localization，以及 Rayon batch 顺序。

这些测试不需要 GPU。`f64` 结果不是绝对 ground truth；fixture producer、80 位 observable 和每项 tolerance 仍由 `tests/fixtures/v1` 保存。

## 适用域与未外推项

- 当前 trajectory fixture 只覆盖 equatorial Schwarzschild 和默认 exterior Kerr Observation 的非临界样本。published Kerr/Kerr–Newman trajectory、独立 chart/state representation、near-axis Killing-tensor overlap 与更广参数扫描仍是 reference ladder 的扩展项。
- raw-radius/reciprocal-radius `f32` 条件性数据仍按[渲染研究](rendering.md)标记为候选 `[X]`；本实现没有把缺少完整 metadata 的表升级成 machine-readable scientific fixture。
- finite escape sphere 是数值边界；当前 comparison 报告边界上的方向、位置和 travel time，不把它解释为无穷远精确 observable。
- renderer 尚未消费这些 CPU 结果；GPU agreement、sky/source anchor、频率比与画面属于 Phase 2/3。

## 实现来源

- Dormand–Prince pair：原始论文 [Dormand–Prince 1980](https://doi.org/10.1016/0771-050X%2880%2990013-3)，continuous extension [Shampine 1986](https://doi.org/10.1090/S0025-5718-1986-0815836-3)，系数交叉检查 [SciPy 1.18.0 RK45 source](https://github.com/scipy/scipy/blob/v1.18.0/scipy/integrate/_ivp/rk.py)。
- bracketed root guarantee：[Brent 1971/1973 作者书目页](https://maths-people.anu.edu.au/~brent/pub/pub006.html)；当前实现选择合同允许的 safeguarded bisection。
- serialization seam：[Serde container attributes](https://serde.rs/container-attrs.html) 与 [`toml::from_str` 1.1.4](https://docs.rs/toml/1.1.4/toml/fn.from_str.html)。
- batch execution：[`rayon::ThreadPool` 1.12.0](https://docs.rs/rayon/1.12.0/rayon/struct.ThreadPool.html)；vector operations：[`glam::DVec3` 0.33.3](https://docs.rs/glam/0.33.3/glam/f64/struct.DVec3.html)。依赖闭包的最终事实以 `Cargo.lock` 为准。

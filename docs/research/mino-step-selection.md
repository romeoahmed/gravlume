# 数值 Mino candidate：研究结论

本文保存 reciprocal-Mino RK4 candidate 的正面结果、最小反例和否决理由，不定义当前 solver；当前 baseline 与未完成方向分别见 [GPU 证据](../gpu-renderer.md)和[路线图](../roadmap.md)。

**状态：已拒绝。** 数值 reciprocal-Mino RK4 candidate 已从 production 删除。它曾在低分辨率矩阵和本机 GPU timing 上表现很好，但更高分辨率独立 reference 暴露出超过正式 observable budget 的 terminal phase error。本文保留这些证据，避免把一次失败误写成“从未有价值”，也避免再次把局部 gate 当成全域证明。

坐标/physical-spin 的 exact seam 见 [KS–BL/Mino seam](kerr-schild-mino-map.md)；RK4/Hermite
局部阶数的可复现实现为
[`mino_step.py`](scripts/src/gravlume_research/checks/mino_step.py)。

## 1. 候选为何有吸引力

pure Kerr exterior 的 Hamilton–Jacobi 方程可分离。令
$u=1/r$、$\mu=\cos\theta$、$b=L_z/E$、$\eta=\mathcal Q/E^2$、
$c=a^2-ab$、$\zeta=(b-a)^2+\eta$，energy-rescaled Mino state 满足

\[
u'=w,
\qquad
w'=(2c-\zeta)u+3\zeta u^2+2(c^2-a^2\zeta)u^3,
\]

\[
\mu'=q,
\qquad
q'=(a^2-\eta-b^2)\mu-2a^2\mu^3.
\]

这套 polynomial RHS 避免每个 RK stage 重建 Cartesian Kerr–Schild radius、metric、principal
null vector 与空间导数。形式化脚本证明 classical RK4 对该三次系统的 local defect 从 $h^5$
开始，cubic Hermite interior defect 从 $h^4$ 开始。因此在固定平滑轨迹上，步长因子 $f$ 的
理论主序为

\[
W(f)=\Theta(f^{-1}),
\qquad
e_{global}(f)=O(f^4).
\]

这解释了候选为何能快，也说明更大 $f$ 在 truncation envelope 上单调更差。它不证明 binary32
terminal observable 逐点单调：最后一步相位、turning branch、event localization 与 roundoff
都会改变误差常数。

形式化命令：

```text
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research mino-step
```

该检查只证明局部数值模型，不证明 supported domain 或完整 renderer correctness；环境与完整 gate 见[统一 Python 研究工具链](python-research-tooling.md)。

## 2. 曾经通过的证据

corrected outgoing-KS/BL seam 后，受限候选使用 pure Kerr、off-axis、subextremal exterior、
finite/turning/reciprocal-constraint 与 winding gates，任何不确定性从原始 ray 回退完整 KS。
低分辨率 strict-DP5(4) matrix 导出的经验 envelope 曾支持 `219/256`；默认 1280×720 本机
实验接受 `775,399/921,600` pixels（`84.136%`），相对 interval capture + KS 的历史
256-pair timing 为 `-35.768%`，95% CI `[-36.390%, -35.189%]`。

这些结果是重要的算法信号：可分离 polynomial dynamics 可能显著减少 GPU 工作，不能因为最终
候选被拒绝就丢弃。但它们只覆盖已采样 phase/profile，并未建立连续 viewport 的误差上界。

## 3. 否决证据

常规测试扩展到 `320×180` 后，独立 Cartesian KS reference 在 pixel `(175, 51)` 复现了正式
失败：候选 travel-time absolute error 为约 `2.661354e-3 M`，超过
`1e-3 M` contract。此前 factor 扫描还出现过 escape-direction error 约
`4.438e-4 rad`，超过 `3.82e-4 rad` contract。

关键根因不是一次 GPU 抖动，而是验收模型不充分：

- reciprocal potential residual 约束 energy surface，不约束累计 azimuth/time phase；
- winding cutoff 是经验域，不是对 terminal phase error 的严格上界；
- 从稀疏 factor sample 拟合 $C f^4$ 只给受测轨迹 envelope，不能外推到新的 pixel lattice；
- 更小 fixed factor 会连续增加工作，却仍不能证明临界/turning 邻域的全域误差。

因此继续寻找一个“最佳 fixed factor”只是在移动未证明边界。高分辨率测试是真实、最小、
mutation-sensitive 的反例；production numerical-Mino WGSL、pipeline constants、benchmark
variants 和专用测试均已删除。通用 Cartesian KS 仍是基线；interval Bernstein capture 后续因没有
产生 travel-time observable 而与 escape map 一并撤出 production，历史 A/B 不再构成当前路径授权。

## 4. 下一条数学路线

下一步不应重做 fixed-step sweep，而应研究 pure-Kerr terminal 的解析/半解析 elliptic solver：

1. 用 Gralla–Lupsasca v3 的 radial/polar root topology 给出明确支持域与 branch；
2. 用 Carlson symmetric elliptic integrals或等价的 manifestly-real 形式计算 terminal
   direction/time，避免几十步 phase accumulation；
3. exact/CAS 验证项目 physical-spin、outgoing KS/BL Jacobian、初值常数与 event reconstruction；
4. near-multiple roots、axis、near-extreme、horizon denominator 或任何 condition 不确定时回退 KS；
5. 先以 high-precision CPU oracle 覆盖每个 accepted pixel，再做 WGSL/f32 和 GPU Pareto gate。

解析 terminal solver 仍不是通用 path sampler：未来 volume/slow-light 需要 trajectory
checkpoints，带电、extreme/superextreme 与 unsupported topology 也仍需要独立实现或 KS。

主要来源：

- [Mino 2003](https://arxiv.org/abs/gr-qc/0302075)
- [Gralla–Lupsasca v3](https://arxiv.org/abs/1910.12881)
- [Dexter–Agol 2009](https://arxiv.org/abs/0903.0620)
- [Wang–Lee–Lin 2022](https://arxiv.org/abs/2208.11906)

最终结论不是“Mino parameter 无用”，而是：**当前数值 fixed-step Mino productization 缺少全域
phase-error certificate；可分离结构应转化为解析/半解析 terminal accelerator，而不是继续以
GPU factor sweep 代替数学边界。**

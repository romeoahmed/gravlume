# 研究决策记录

本目录保存可复算的假设、方法、证据和取舍，不定义 production interface。已采用的事实必须回写对应规范/证据文档；已拒绝的候选保留最小反例和恢复条件，避免重复试错。

## 决策索引

| 记录                                                 | 状态                       | Production 影响                                                                |
| ---------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------ |
| [完整帧原子发布](atomic-frame-publication.md)        | 已采用                     | 隐藏 candidate、完整 generation 发布；可见 tile/低分辨率阶段已拒绝             |
| [GPU geodesic 加速](gpu-geodesic-acceleration.md)    | 混合账本                   | escape map、interval capture、KS 约化已采用；其他候选逐项记录                  |
| [Kerr–Schild RK4 约化](kerr-schild-rk4-reduction.md) | 已采用，step policy 仍研究 | reduced Hamiltonian、Carter、Hermite event 与 KN certificate 已进入 production |
| [Kerr–Schild ↔ Mino seam](kerr-schild-mino-map.md)   | 数学 seam 已采用           | physical-spin/chart 修复已进入 domain/WGSL                                     |
| [数值 Mino step 选择](mino-step-selection.md)        | 已拒绝                     | fixed-step candidate 因高分辨率 travel-time 反例删除                           |
| [辐射传输与 source 重建](radiative-transfer-and-source-reconstruction.md) | 混合决策 | scalar slab、spectral fixture 与 footprint 证据已采用；path inspection/reconstruction/Carlson 待证 |
| [原生 HDR 输出](native-hdr-output.md)                | 已采用                     | native display state、extended-linear HDR 与 typed SDR fallback                |
| [GPU benchmark 方法](gpu-benchmark-methodology.md)   | 当前方法                   | 只测固定 production workload；临时 variant 不形成永久接口                      |

## 可复算脚本

[`scripts/`](scripts/) 使用锁定的 uv/SymPy/Hypothesis 环境复核符号恒等式、binary32 interval、chart seam 与数值模型。脚本只生成研究证据，不进入 Cargo runtime dependency closure。

运行方式和精度假设必须写在对应记录中；脚本通过只证明它声明的命题，不自动授权 production 支持域。

## 记录格式

每份记录至少包含：

1. **状态：** 已采用、仍实验、已拒绝或混合账本；
2. **问题：** 要验证的可否证假设和定义域；
3. **方法：** 符号/数值/性能输入、precision、observable 与工具版本；
4. **结果：** 正反证据，不只给最优数字；
5. **决策：** 对 production 的影响；
6. **恢复条件：** 被拒绝方案需要什么新证据才可重开。

## 维护规则

- 新证据改变决策时更新原记录，不另写互相矛盾的“最终版”；
- 性能结果绑定 revision、平台、adapter、backend、scene、extent、profile、样本设计和 correctness gate；
- 临时绝对路径、源文件行号、已删除 variant 和 benchmark-only interface 不得成为持久结论；
- 原始实验可保留，但摘要必须明确区分历史 baseline 与当前 production；
- 研究记录可以较长，规范文档只保留采用后的稳定语义和链接。

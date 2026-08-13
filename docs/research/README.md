# Research decision records

`research/` 保存可复算的实验与设计决策，不定义 production API。已经采用的事实必须回写主题合同；被拒绝的候选保留最小反例和恢复它所需的新证据，避免重复试错。

| 记录 | 状态 | Production 影响 |
|---|---|---|
| [完整帧发布](atomic-frame-publication.md) | 已采用 | 隐藏 candidate、完整 generation 原子发布；低分辨率阶梯已拒绝 |
| [GPU geodesic 加速](gpu-geodesic-acceleration.md) | 混合决策账本 | direction reconstruction、interval capture、endpoint reuse 等已采用；其余按 ledger |
| [原生 GPU benchmark](gpu-benchmark-methodology.md) | 当前方法 | Criterion 驱动固定 production workload；临时 variant 不留永久 API |
| [Kerr–Schild/Mino 映射](kerr-schild-mino-map.md) | 数学 seam 已采用 | physical-spin/chart 修复已进入 domain/WGSL；fixed-step Mino 未获授权 |
| [Mino step 选择](mino-step-selection.md) | 已拒绝 | 高分辨率 travel-time 反例否决 numerical fixed-step candidate |
| [原生 HDR 输出](native-hdr-output.md) | 已采用 | shared display-state DTO、extended-linear HDR 与 typed SDR fallback |

可复算脚本位于 [`scripts/`](scripts/)。脚本是研究证据，不进入运行时 Cargo feature closure。

维护规则：

- 每份记录首段必须给出 `已采用`、`仍实验` 或 `已拒绝` 状态；
- 性能数字必须说明 scene、extent、adapter/backend、统计方法和 correctness gate；
- 临时绝对路径、行号链接、过期类型名和已删除 shader variant 不得成为结论依据；
- 新证据改变决策时更新原记录，不另写一份互相矛盾的“最终版”。

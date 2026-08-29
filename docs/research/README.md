# 研究记录索引

本目录保存可复算的研究证据与决策，不定义 production 行为。当前实现与合同分别以
[Reference 证据](../reference-implementation.md)、[GPU 证据](../gpu-renderer.md)和
[文档地图](../README.md)列出的规范为准；未完成工作只由[路线图](../roadmap.md)排序。

## 已采用的工程决策

- [完整帧原子发布](atomic-frame-publication.md)：隐藏 candidate、完整 generation 发布，拒绝可见
  tile 或低分辨率阶段。
- [Production 单样本检查](sample-inspection.md)：固定单槽、ticket/completion、cancel-drain、
  published texel 与 fresh retrace 分离，并保留 Metal protocol 反例。
- [原生 HDR 输出](native-hdr-output.md)：native display state、extended-linear HDR 与 typed SDR
  fallback。
- [GPU benchmark 方法](gpu-benchmark-methodology.md)：只测 correctness-approved production
  workload，不把临时 variant 变成永久接口。

## 数学与数值证据

- [Kerr observable corpus](kerr-observable-corpus.md)：source-edge、critical、negative-spin 与
  ill-conditioned fallback 的分层高精度证书，以及 Rust/WGSL 消费边界。
- [Kerr–Schild RK4 约化](kerr-schild-rk4-reduction.md)：reduced Hamiltonian、Carter invariant 与
  Hermite event localization；step policy 仍是开放研究。
- [Kerr–Schild ↔ BL/Mino seam](kerr-schild-mino-map.md)：physical-spin、chart、covector 与 tangent
  的 exact mapping。
- [Kerr root topology 与 elliptic oracle](kerr-elliptic-oracle.md)：exact root isolation、
  positive-real Carlson forms 与 Class Ib reduction；不授权 production solver。
- [Scalar radiative transfer](radiative-transfer.md)：invariant transfer、analytic slab、spectral
  fixture 与 equatorial emitter 的声明边界。

## 候选与否决账本

- [GPU geodesic 加速](gpu-geodesic-acceleration.md)：Mino、map、interval、workgroup 与 wavefront
  候选的正反证据；当前 production 仍使用 Cartesian Kerr–Schild。
- [Source-space reconstruction](source-reconstruction.md)：已拒绝的 2-pixel candidate、最小
  semantic key、准入 gate 与重开条件。

## 研究工具链

[统一 Python 研究工具链](python-research-tooling.md)定义 Python 3.14、锁定 `uv` 环境、单一 CLI、
八个 end-to-end scientific checks、pytest/Hypothesis 与 Ruff。实现位于 [`scripts/`](scripts/)，
不进入 Cargo runtime dependency closure；一项检查通过只证明它自己的命题。

## 维护规则

- 新证据改变决策时更新所属记录；不保留重复的路线图审计或“最终版”快照。
- 已拒绝候选只保留最小反例、根因和可证伪的重开条件。
- 性能数字必须绑定 revision、平台、adapter、backend、scene、extent、profile、样本设计和
  correctness gate。
- 历史 baseline 与当前 production 必须分开；临时路径、行号和已删除 API 不得成为合同。
- 同一公式、阈值或状态只在一个权威位置定义，其他记录使用相对链接。

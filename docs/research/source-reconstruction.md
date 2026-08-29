# Source-space reconstruction 决策

本文保存 branch-aware source reconstruction 的设计约束、一组已拒绝候选、binary32 phase 边界和
重开条件。它不定义 production pipeline；当前 geometry/transport、资源所有权和后续顺序分别以
[渲染设计](../rendering.md)、[架构合同](../architecture.md)、[GPU 证据](../gpu-renderer.md)和
[路线图](../roadmap.md)为准。

**状态：production reconstruction 未采用。** Test-only five-ray footprint 已证明 ordinary-region
finite difference 可行，但没有真实 filterable source/history consumer；固定 2-pixel map 候选既有
false acceptance，也在收紧 classifier 后产生约 63% GPU 回归。当前 production 保持 full Cartesian
Kerr–Schild trace 与 selective shadow coverage。

## 问题与最低语义

Reconstruction 不能对 tone-mapped RGB 求导。对 image coordinate $s$ 与局部 source chart $y(s)$，
候选 footprint 是 $J=\partial y/\partial s$；只有邻域先满足以下 exact key 相容时，连续差分才有意义：

- termination、source kind 与 ambiguity；
- image branch、initial polar side、turning/crossing counts 与 signed winding；
- source chart/seam、scene/transport revision 与 generation；
- near-critical、caustic、parity-degenerate 与 unsupported classifier 均为 false。

Accepted region 再比较 source displacement、$g$、travel time、transport variation、Jacobian/parity 和
scene-linear radiance。任一 key 不同或 uncertainty 不足，必须增加真实 samples 或 full trace；不能跨
critical curve、winding、source edge 或 chart seam 插值。

逻辑 `GeometricSample` 不要求 full-frame G-buffer。只有真实 consumer 固定 lookup、sampling、color/
spectral interpretation 和 error budget 后，才物化其最小 fields。

## 可接受的私有数据流

未来候选可保持三段私有 pipeline：

1. trace coarse/adaptive nodes，记录 packed exact key 与必要的 source/time/$g$/quality fields；
2. 带 halo 分类，检查 key、periodic unwrap、Jacobian/curvature、source-domain 与 transport variation，
   输出 accepted/refine/full-trace；
3. accepted region 在 consumer chart 中采样并评价完整 transport，refine/full-trace queue 放到后续
   dispatch/pass。

Coarse spacing、AoS/SoA、packed key、workgroup shape、buffer/texture 和 pass 数都只是性能假设。跨
workgroup producer/consumer 必须由有序的不同 dispatch/pass 建立；WGSL barrier 只有 Workgroup scope，
不能模拟 grid barrier。

## 准入门槛

1. **Consumer：** versioned sampling/source chart/revision/color interpretation 的真实 source 或
   stationary-history consumer；synthetic fixture 不算第二消费者。
2. **Oracle：** full-resolution direct trace 或 independent supersampling；discrete key 先 exact，
   continuous footprint 与 final scene-linear radiance 再按具名预算比较。
3. **Corpus：** source edge、critical 两侧、caustic、多像/higher-order、高频 source 与 seam；accepted
   false acceptance 必须为零。
4. **Scale convergence：** node spacing 与 finite-difference step 逐次减半、source/image resolution
   加倍；不能让同一 sparse stencil 自证。
5. **Transport：** 每个 filter sample 计算自己的 $g$、band fraction 与 optical depth；不能用中心
   sample 的 transport 代表整个 footprint。
6. **Lifecycle：** scene/view/source/quality/generation/cut 不相容时拒绝 history；stale publish 为零。
7. **GPU：** requested-limit admission、Metal/Vulkan total trace + classify + reconstruct + fallback time、
   resource peak 与 publication latency 形成更优 Pareto 点。

## 已拒绝的 2-pixel map

两个未提交候选基于 revision `8145b24`，Apple M5/Metal，1280×720 bolometric surface。它们用锁定的
`trace_gpu` Criterion workload 测量，并在删除 candidate 后立即复测 all-full-KS baseline。独立 prepass
在 2-pixel spacing 生成 8×`u32` semantic nodes；branch/phase/parity 不安全的 pixel 回退 full KS。
Map 占 `7,404,832` bytes。

| Candidate | Correctness result | GPU duration | Decision |
| --- | --- | --- | --- |
| extra edge validator | accepted centers 对 CPU 的 max source `4.076e-5M`、$g$ rel `2.641e-6`、time `4.313e-5M` | `[36.094,36.469,36.850] ms` | 正确但重复 rays，拒绝 |
| unvalidated bilinear | 1,517 samples 中 source `3.174e-2M`、time `3.582e-2M`、spectral budget ratio `4.395` | 未进入准入 | false acceptance，拒绝 |
| tightened 3×3 classifier | 1,389 samples 通过 source `1e-3M`、$g$ rel `2.5e-4`、time `1e-3M` 与 FP16 spectral gate；只接受 3,735 centers | `[32.233,32.574,32.908] ms` | 安全域过小且慢，拒绝 |

同机 all-full-KS 基线为 `[19.765,19.975,20.191] ms`；最终候选相对回归约 63%。该历史 artifact
没有保存当前 benchmark contract 要求的全部环境 metadata，不能作为现行或跨平台性能基线；配对回归
仍足以拒绝这个固定设计，无需为无性能前景的候选继续扩建昂贵 oracle。融合 pass、放宽 gate 或只报告
accept ratio 都不能恢复它。

## Surface binary32 phase 边界

拒绝分析同时证明 production step scale `0.1` 的 continuous agreement 不能从 canonical v2/v3 pixel
外推到整帧。64×36 的 Schwarzschild、positive/negative Kerr 与 Kerr–Newman matrix 只通过 exact
terminal/branch；默认 Schwarzschild continuous max `[source M, g relative, time M, radiance relative]`
为 `[2.482e-2,2.539e-4,3.161e-2,2.175e-3]`，超过多项预算。

单纯缩小 fixed step 没有给出稳定解：

- scale `0.04` 的 source 降到 `3.965e-3M`，time 仍为 `5.473e-3M`；
- `0.008` 在 Schwarzschild 达到 `8.454e-4M`，positive-spin-wide 仍为 `1.745e-3M`；
- `0.005` 因 binary32 trajectory roundoff 退化为 `2.269e-3M`；
- compensated time summation 不改变结果，根因是 trajectory/event phase，不是最终标量累加。

这些临时 step/hot-loop variants 已删除。恢复条件是 binary32-stable higher-order/adaptive integrator、
受限 analytic terminal solver 或显式 science-quality profile，并同时通过 full observable matrix、
near-critical corpus 和端到端性能测试；不得放宽 `1e-3M` travel-time gate。

## 重开条件

只有真实 consumer 出现后，才从能显著减少总 geodesic 数的 coarse/adaptive、stationary-amortized
transfer map 或其他新设计重开。新的方案必须重新满足本页全部准入门槛；历史 2-pixel implementation
不是待调参 backlog。Source footprint evidence、较少 dispatch、较高 accept ratio 或漂亮截图都不能
单独授权 production reconstruction。

## 一手来源

- [Igehy 1999](https://doi.org/10.1145/311535.311555)：ray differentials；
- [James et al. 2015](https://doi.org/10.1088/0264-9381/32/6/065001)：black-hole lensing beam / footprint；
- [WGSL synchronization](https://www.w3.org/TR/WGSL/#barrier-builtin-functions)：barrier scope 与 uniform-control-flow contract；
- [WebGPU resource usages](https://www.w3.org/TR/webgpu/#resource-usages)：有序 dispatch/pass 的 resource usage scopes。

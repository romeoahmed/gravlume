# GPU 测地线加速决策账本

本文是 GPU geometry 优化的历史性能与正确性账本，不定义当前 pipeline、资源或支持域；采用状态必须回到 [GPU 证据](../gpu-renderer.md)、[架构合同](../architecture.md)和实际源码核对。

**状态：混合决策账本。** 各实验是否已实现以第 7 节 ledger 与链接的当前合同为准。研究基线为仓库提交 `9dbdb71c0ad325c7c78ca518cbdff528daa2f2fe`，目标是找到能显著降低完整原生分辨率画面延迟、同时保留物理与数值合同的算法路线。本文延续[完整帧原子发布研究](atomic-frame-publication.md)：计算可以分批，画面仍只发布完整 generation；不重新引入可见低分辨率阶段或扫描式 reveal。

## 结论

1. 研究起点的默认 1280×720 trace 累计 GPU 时间曾为 `80.5–192.2 ms`，说明只调 batch 不足以接近 33.3/16.7 ms 目标。此后两次 32-pair 位置对称 A/B 中，escape-direction map 稳定改善 `27.923–37.941%`，interval capture 再改善 `5.946–9.156%`，证明减少真实 KS rays 是当前有效主线；这些增量仍不足以构成跨平台 60 FPS 声明。[当前证据](../gpu-renderer.md#适用域与限制) [资源合同](../architecture.md#内存与资源预算)
2. **受限 numerical-Mino 快路径已经否决**：它曾在默认 1280×720 接受 `84.136%` pixels，并相对 interval capture + KS 改善 `35.768%`；但扩大 reference lattice 后出现越过正式 travel-time budget 的 accepted ray。该结果证明可分离结构值得继续利用，却不授权 fixed-step implementation。当前 production 顺序是 Kerr/Kerr–Newman interval capture → Cartesian KS。[候选结论](mino-step-selection.md)
3. 在现有 Kerr–Schild 路径内，Bogacki–Shampine 3(2) FSAL pair 值得比完整 Dormand–Prince 5(4) 更早试验：首次 4 次、此后每 accepted step 3 次新 RHS，并自带 embedded estimate；但它是三阶方法，是否真的减少总 RHS 只能由 observable gate 和 GPU A/B 决定。[Bogacki–Shampine 原文](https://doi.org/10.1016/0893-9659%2889%2990079-7) [Dormand–Prince 原文](https://doi.org/10.1016/0771-050X%2880%2990013-3)
4. active-ray wavefront/compaction 不是第一步。它可能回收长尾 inactive lanes，但每轮必须把状态写回显存；在得到 step 分位数和 active-ratio 曲线以前，不能证明收益大于 state traffic 与多 pass 开销。若进入实验，应把完整 RK step 融在单 kernel 内，只在多个 step 的 chunk 边界 compact；不要把四个 RK stage 拆成四个 kernel。[Laine–Karras–Aila 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf) [PBRT v4 wavefront 取舍](https://www.pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Mapping_Path_Tracing_to_the_GPU)
5. 截图中的“脏”主要不能先归因于测地线噪声。研究基线的 `analytic_sky` 把经纬周期中央约 90–92% 设为高亮，两项取 `max` 后约 99.2% 天空被额外加上 `2.0` radiance。初次反转 mask 后，实际截图仍显示薄线在临界曲线处被放大为发白的环；当前实现因此进一步换成无 seam、不过曝的低阶球面多项式与局部轴向色标。每像素一个中心样本仍没有 branch-aware coverage，因此真正的 horizon/escape 轮廓抗锯齿是后续 appearance/filtering 问题，不能靠放宽积分误差或模糊几何掩盖。[preview shader](../../crates/gravlume-render/src/shaders/lensing_preview.wgsl)

已完成的顺序是：production/capture ABI 瘦身 → 2D/global coherence → interval capture → outgoing-KS/BL exact seam → numerical-Mino 实验并否决 → branch-aware shadow coverage。下一步算法主线是完整解析/半解析 Kerr terminal solver；只有 aggregate long-tail 证据支持时才做 wavefront/subgroup。BS3(2) fixed-step 探针仍未证明价值。

## 1. 基线环境与当前成本模型

研究基线的 lockfile 当时固定 `wgpu/wgpu-core/wgpu-hal/wgpu-types/Naga 30.0.0`、`bytemuck 1.25.2`、`glam 0.33.3`、`egui/egui-wgpu 0.36.1`、`pollster 1.0.1` 与 `winit 0.30.13`；当前依赖以 [`Cargo.lock`](../../Cargo.lock) 为准。macOS 只启用 wgpu Metal，Windows/Linux 只启用 Vulkan；当前 device 只请求 `TIMESTAMP_QUERY`。[workspace manifest](../../Cargo.toml) [render manifest](../../crates/gravlume-render/Cargo.toml) [capability baseline](../../crates/gravlume-render/src/capabilities.rs)

研究基线的 production shader 是一 invocation 一 pixel 的 megakernel：8×8 workgroup、最多 2048 个 radius-scaled fixed RK4 step，普通 accepted step 复用 exact endpoint 后仍有 4 次 geometry/RHS 求值；当时最终写 `rgba16float` HDR 与三个 16 B/pixel record plane。[current solver](../../crates/gravlume-render/src/shaders/geodesic_integration.wgsl)

默认场景的开发数据从平均/最坏 `882/2048` step 降到了 `61/132`，说明 outgoing Kerr–Schild chart、几何 step 与 endpoint reuse 已经拿掉了最明显的冗余；剩余时间更接近“每步数学本身 × 全屏像素数”，而不是发布调度问题。[实现证据](../gpu-renderer.md#适用域与限制) 坐标选择仍不可随意回退：backward ray 的方向与 horizon-penetrating chart 会改变数值适定性，相关分析见 Bozzola、Chan 与 Paschalidis 的[原论文](https://arxiv.org/abs/2310.02321)。

### 1.1 生产路径有一个独立的显存根因

历史实现只在 test build 暴露三个 record buffer handle，却仍在 production 创建、持有并写入对应资源，因此 `#[cfg(test)]` 没有消除成本。当前 diagnostic capture 已与 production ABI 分离：[trace pipeline](../../crates/gravlume-render/src/trace.rs) 与 [production solver](../../crates/gravlume-render/src/shaders/geodesic_integration.wgsl)。

研究基线的 frame accounting 是：record planes `48 B` + candidate HDR `8 B` + published HDR `8 B` + UI `4 B` = `68 B/pixel`。因而 2560×1440 是 `250,675,200` bytes（约 239.1 MiB），两阶段 rebuild 约 478.1 MiB。若 production presentation 不分配 full diagnostic planes，steady candidate+published+UI 核心变为 `20 B/pixel`，同一 extent 是 `73,728,000` bytes（约 70.3 MiB），下降 70.6%；当前实现又以 texture-view promotion 消除了 candidate 与同尺寸 published copy 在新 generation 内的重复。

实现结果：production ABI 现在只有 trace uniform、HDR storage texture、dispatch uniform 与 packed escape-direction map；当前四个 record plane 由 test-only capture shader/target 注入。新 generation 分配 candidate HDR、UI、global node map 与 shadow edge queue；上一张完整 scene 独立持有，完成后直接提升 candidate texture view，不再分配同尺寸 published copy。4K 两类 scratch 合计 `2,271,620 B/candidate`；初次 generation 为 `101,804,428 B`，上一张已完成 4K scene 时的同尺寸 rebuild 为 `201,337,220 B`，均通过 256 MiB admission。仍有旧 4K candidate 在追迹时再申请同尺寸 replacement 的真实 plan 是 `269,964,040 B`，因此会在分配前拒绝。这关闭了截图中的 host budget rejection，同时没有把 surface drawable、driver heap 或 alignment 伪装成已测总显存。

建议把资源合同拆成两个明确产品，而不是让测试字段支配每帧资源：

- `PresentationTrace`：只产生画面所需的 HDR 或 compact semantic map，加 O(1) 大小的 termination/step/drift histogram；
- `DiagnosticCapture`：按显式请求分配三个 f32/u32 plane，继续服务 GPU 合同、科学诊断和 readback；
- 若未来 source 重建确实需要 direction/branch，应给 presentation 定义自己的最小、误差受检格式，不能把 48 B/pixel 测试记录原样常驻。

这项主要修复 resize/VRAM，而非承诺 6× 算力提升；但它给 fallback queue、semantic map 或 wavefront state 留出真实预算，也减少 terminal store traffic。完整 diagnostic capture 仍是验证工具，不能删除其语义。

### 1.2 先补可归因统计

在更换积分器或执行模型前，GPU 端至少需要小型 `atomic<u32>` histogram，而不是依赖全屏 record readback：

- accepted/rejected step 与 RHS/geometry evaluation count；
- step count `p50/p90/p95/p99/max` 的 bucket；
- 每个 step/chunk 的 active-ray ratio；
- termination/uncertain/fallback 分布；
- 四项 drift 与 event residual bucket；
- 每个 pass 的 timestamp、fallback queue build time 和 indirect trace time。

这些统计只需整数 atomics，不需要 `SHADER_FLOAT32_ATOMIC`；float observable 先按固定边界映射到 u32 bucket。没有 active-ratio 曲线，就无法区分“算法每条 ray 太贵”和“少量长 ray 拖住 SIMT”两类根因。

## 2. 最大的算法机会：利用 Kerr/Kerr–Newman 可分离性

### 2.1 数值 Mino 实验为何失败，以及为何转向解析路线

Kerr 的 Hamilton–Jacobi 方程可分离。[Mino 的原始工作](https://arxiv.org/abs/gr-qc/0302075)引入的重参数化把 radial 与 polar motion 解耦；Gralla–Lupsasca 系统分类 radial/angular potential roots，并给出 manifestly real 的 Legendre elliptic integral 与 Jacobi-function 参数曲线。[Gralla–Lupsasca v3](https://arxiv.org/abs/1910.12881) 的 v3 明确修正了发表后发现的错误，且论文范围是非极端 Kerr exterior，因此实现不能从旧版公式抄写，也不能把 extremal/inside-horizon 外推成已证明域。

Kerr–Newman 也有 Mino-time radial/angular potentials 与显式 elliptic/Jacobi 解，Wang、Lee 与 Lin 给出了 root 分类和可指定初值的 manifestly real 解。[Kerr–Newman 原论文](https://arxiv.org/abs/2208.11906) 这证明“解析/半解析路线存在”，不证明一份 binary32 WGSL 实现对当前近场 observer、outgoing Kerr–Schild event 和所有 extremality 都正确。

完整解析实现每个像素需要：从近场 tetrad 初值构造 `E,Lz,Q`、分类 quartic roots 与 turning branches、求 elliptic functions、累计 azimuth/time integral、再转换回项目的 outgoing Cartesian Kerr–Schild observable。WGSL 标准 built-ins 不提供 elliptic integrals/Jacobi functions，必须自行实现并验证。[WGSL built-in functions](https://www.w3.org/TR/WGSL/#builtin-functions) 已有 Kerr analytic 实现也说明了边界：Krang 面向 GPU/differentiability，但第一方 README 明确把 observer 限制在 asymptotic infinity；这不是当前 `r_obs=30M` 的 drop-in solver。[Krang source/README](https://github.com/dominic-chang/Krang.jl) Dexter–Agol 的 semi-analytic 方法以 Carlson elliptic integrals化简 Kerr photon orbit，是很好的 CPU/oracle 对照，但其存在同样不等于 WGSL/f32 合同已经成立。[Dexter–Agol 2009](https://arxiv.org/abs/0903.0620)

数值实验曾用 `(u=1/r, u', μ=cosθ, μ')` polynomial RHS 自然越过 turning point，并数值积分 `φ,t`。它确实显著减少 geometry 工作，但 high-resolution reference 证明 potential constraints 不能界定累计 terminal phase。继续收紧一个全局 fixed factor 只会线性增加工作，并不补上证明缺口。因此下一候选直接使用具名 root topology 与 elliptic/Carlson terminal integrals；Cartesian KS 继续承担任何不确定域。

### 2.2 必须封闭的域

未来 analytic/Mino-derived variant 只接受同时满足下列条件的像素：

- pure Kerr、明确 subextremal、observer 与当前段都在 exterior；
- 从 validated tetrad/KS covector 得到的 `E,Lz,Q`、radial/polar sign 与 potential constraints 全部有限且落在安全区间；
- root/turning topology 不接近具名退化阈值，reciprocal state 约束在每个 checkpoint 内满足预算；
- horizon/escape event、source direction 与 travel time 可从同一 localized state/积分提交；
- Boyer–Lindquist/Mino 到 outgoing Kerr–Schild 的 `t,φ` 变换在 horizon bracket 前保持受控。

任何条件不满足都进入 KS queue，不允许让快路径猜 branch。Gralla–Lupsasca 的 `t,φ` integrals 含 `Δ` 分母，而当前 outgoing Kerr–Schild baseline 正是为了 backward horizon crossing 的正则性；因此 Mino 路径可以从 exterior 一侧定位 horizon event，却不能越过 BL coordinate singularity 后继续假装和 KS state 等价。[Kerr formulas](https://arxiv.org/pdf/1910.12881) [坐标分析](https://arxiv.org/abs/2310.02321)

可分离 solver 把 `E,Lz,Q` 当常量，因此“它们没有被更新”不能冒充数值正确性。accepted terminal 必须重建 outgoing KS position/covector，再用独立的现有公式复算 null/`E,Lz,Q` agreement；同时记录 `v_r²-R`、`v_μ²-U`、root separation/condition bucket。任何一项超过安全预算就回退，不能把构造出来的零 drift 当作 confidence。

legacy Mino prototype 曾因 outgoing chart 把 oblate twist 留在 `+a` 而产生最坏 `0.194 rad` escape-direction error；exact SymPy legacy RED 与 corrected Kerr–Newman metric/covector/tangent GREEN 随后封闭了 Gate A/B。修正后的 numerical candidate 在低分辨率 Gate C/D 与 GPU timestamps 上表现很好，却在 `320×180` reference 产生约 `2.661354e-3 M` travel-time error，超过 `1e-3 M` contract。完整正反证据见 [KS–BL seam](kerr-schild-mino-map.md) 与 [Mino candidate conclusion](mino-step-selection.md)。

### 2.3 推荐的 GPU 组合，而不是全量替换

```text
all pixels
    │
    ▼
conservative fast solver ── accepted ──► terminal/appearance output
    │
    └── uncertain pixel indices
              │
              ▼
      compact + indirect args
              │
              ▼
      validated KS RK4 fallback ───────► overwrite those pixels
              │
              ▼
          atomic full-frame publish
```

快路径只排队 `u32 pixel_index`，fallback 从相机初值重算，所以最坏队列是 4 B/pixel（双队列也仅 8 B/pixel），不需要保存几十到上百字节的 ODE continuation state。`wgpu 30` 已提供 [`ComputePass::dispatch_workgroups_indirect`](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePass.html#method.dispatch_workgroups_indirect)，参数布局由 [`util::DispatchIndirectArgs`](https://docs.rs/wgpu/30.0.0/wgpu/util/struct.DispatchIndirectArgs.html) 定义；argument buffer 使用 `STORAGE | INDIRECT`，fast pass、args finalize、fallback pass 可按同一 queue 顺序编码，不做 CPU hot-path readback。

这种结构也适用于 BS3(2) coarse solver 或未来解析 solver：快路径的错误估计/condition classifier 负责**保守拒绝**，KS 负责最终正确性。最终 gate 看 fast+queue+fallback 总时间和最终 observable；低 fallback ratio 本身不是正确性指标，错误地接受临界 ray 才是失败。

### 2.4 完整解析与预计算的适用边界

- **完整 Kerr elliptic solver**：常数级每 ray 工作最可能带来数量级改善，也是数值 Mino phase failure 后的优先路线。先只做 terminal sky/surface map，不把它包装成能提供 future transport volume checkpoints 的通用 path API。
- **Kerr–Newman analytic**：论文给出了 exterior 解，但 charge、extreme/superextreme 分类和更多 radial topology 扩大验证面；保留通用 KS baseline，等 pure-Kerr variant 有明确收益后再扩展。[Wang–Lee–Lin 2022](https://arxiv.org/abs/2208.11906)
- **AART 非均匀网格**：它利用 Kerr integrability 和为 photon-ring lensing 定制的 adaptive grid，论文应用是 equatorial source 与长基线 visibility；适合高分辨率 photon-ring 科学产品，不是当前任意近场相机/全天空交互画面的通用替代。[AART 原论文](https://arxiv.org/abs/2211.07469)
- **Schwarzschild LUT/beam map**：Bruneton 用预计算表实现 non-rotating black hole 的 constant-time-per-pixel beam tracing 与过滤；这是 `a=0,q=0` sealed specialization 的强候选，但不能帮助默认 `a=0.8`。[Bruneton 2020](https://arxiv.org/abs/2010.08735)
- **固定场景 transfer map**：当 spacetime/observer 不变而只改变 sky、surface appearance 时，缓存 direction/branch/travel-time map 可完全跳过 geodesic；相机、FOV、extent lattice 或 spacetime 改变时必须以 generation key 失效。它是产品层复用，不是伪装成通用 solver 的 LUT。

### 2.5 当前演示与未来交互的适配

| workload                                      | 最合适路线                                                        | 不应外推                                                |
| --------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------- |
| 当前固定 `a=0.8`、近场 observer、analytic sky | interval capture + KS；研究 pure-Kerr elliptic terminal fast path | AART equatorial grid、Schwarzschild LUT                 |
| 只改 sky/appearance/HDR                       | 复用 terminal map，重跑 appearance pass                           | 重新追完整 geodesic                                     |
| 未来移动 camera/FOV/spin                      | 每 generation 重跑 analytic + KS fallback                         | 固定 observer LUT/旧 transfer map                       |
| 未来 Kerr–Newman/超极端参数                   | 通用 KS baseline；analytic KN 独立受限 variant                    | 把 pure-Kerr root classifier 扩大命名即复用             |
| 未来 surface                                  | terminal map 加 source anchor/branch/frequency                    | 只缓存 tone-mapped RGB                                  |
| 未来 volume/slow-light                        | `PathSampler`/checkpoints 与 retarded time                        | 只有 terminal elliptic result 却声称支持 path transport |

因此当前“scene 不变时不重复追迹”应与“参数变化后快速得到新完整 generation”分开计时；前者是 cache hit，后者才检验 solver 是否达到交互预算。

## 3. Embedded/adaptive RK：先比较成本模型，再谈高阶

当前普通 accepted RK4 step 是 4 次 exact geometry/RHS evaluation。以默认平均 61 step 粗算，每 ray 约 244 次 evaluation，不含 terminal extra work。

Bogacki–Shampine 3(2) 是四-stage、FSAL 的 embedded pair：第一次 accepted step 4 次 evaluation，之后理想情况下 3 次新 evaluation，并以同一 stages 给出二阶误差估计。[原始论文](https://doi.org/10.1016/0893-9659%2889%2990079-7) 若没有 reject，它即使平均 step 数增加到约 81 才与 `4×61` evaluation 持平；因此它比“直接搬 CPU reference”更值得先试。不过其主解只有三阶，event/escape angle 是否能在较少总 evaluation 下满足门槛完全未知。

Dormand–Prince 5(4) 首步 7 stages、FSAL 后每 accepted step 6 次新 evaluation。[原始论文](https://doi.org/10.1016/0771-050X%2880%2990013-3) 它要把平均 accepted steps 降到约 40 以下才仅在 evaluation count 上追平当前 61-step RK4，还未计 error norm、reject/retry、更多 live stage state 和 SIMT 分歧。CPU reference 选择 DP5(4) 是为了独立、高精度 oracle，不构成 GPU 的默认选择。[reference contract](../reference-implementation.md)

GPU ODE 文献证明一 thread 一 independent system 与 adaptive Cash–Karp 可行，同时明确指出复杂 per-system flow 会导致严重 thread divergence。[Niemeyer–Sung](https://arxiv.org/abs/1611.02274) GPU ODE 实现也常选择不保存中间轨迹、只在事件/observable 点提取结果，这与 Gravlume 的 terminal/diagnostic 语义一致。[Hegedűs](https://arxiv.org/abs/1810.03931) 但这些结果没有使用本项目的 metric、backend、误差预算，不能移植论文中的速度数字。

建议的 BS3(2) 实验合同：

1. 误差 norm 分开缩放 position/momentum，并以 observable bake-off 标定，不把 invariant drift 当作 angle/branch error 的替代；
2. step 只落到少量 `base × 2^k` tier，retry 次数有硬上限；失败后排入 KS baseline，而不是强行接受；
3. rejected attempt 不提交 travel time、event side 或 maximum drift；
4. accepted endpoint derivative保持 FSAL 复用；event 只在 bracket 内用与该 pair 一致的 dense/endpoint Hermite state 定位；
5. 统计 accepted/rejected steps、实际 RHS count、tier 分布和每个 subgroup/workgroup 的 divergence。

若 BS3(2) 不能同时降低 total GPU time 与 observable error，就删除 variant；不因为“自适应更现代”而保留。

本轮先做了一个受控的 fixed-step 主解探针，而没有把它伪装成完整 embedded variant：base step `0.1` 在 regular fixture 的 travel-time gate 上达到 `1.187e-3 M`，已经超过 `1e-3 M`；缩到 `0.05` 后合同通过，但增加的 step 数没有在噪声较大的 Metal smoke 样本中证明总时间收益。该探针已删除。它不否定带二阶误差估计、量化 tier 与有界 retry/fallback 的完整 E2，但证明“把 RK4 tableau 直接换成三阶公式”不能作为生产优化。

## 4. Symplectic/Hamiltonian 方法为什么暂不进入交互主线

当前 Cartesian Kerr–Schild Hamiltonian 非可分离。FANTASY 用双份 phase space 把任意非可分离 Hamiltonian 变为显式 symplectic scheme，并提供二阶/四阶组合；代价正是状态翻倍与更多 subflow。[FANTASY 原论文](https://arxiv.org/abs/2010.02237) 另一类工作通过 time transformation 为 Kerr/Kerr–Newman 类 Hamiltonian寻找多项可积 split，目标主要是长时间 orbit integration。[Wu et al. 2022](https://arxiv.org/abs/2210.13185)

这类方法的优势是长期相空间结构与有界能量误差；当前默认 ray 平均约 61 step 后就到 horizon/escape，验收对象则是 termination、source angle、travel time 与 event position。symplectic 不自动保证这些 finite-time observables，更不自动解决 event localization；终止插值本身也不再是原 symplectic map。对短 ray，双状态、更多 stages 与寄存器压力很可能比长期守恒收益更大。

因此只在以下证据同时出现时重新开启：near-critical/多绕行 ray 占有可观比例、现有/embedded RK 的主要失败是长期 phase drift、且 Mino/analytic path 无法覆盖目标 domain。即使进入，也先做 CPU/GPU research variant，不替换 KS baseline。

## 5. GPU 执行架构：coherence、wavefront、persistent work 与 subgroup

### 5.1 二维像素布局已进入 production

历史实现虽然声明 `@workgroup_size(8,8,1)`，却把 pixel range 映射到 workgroup y 维，再以 `global_id.y * 8 + global_id.x` 线性化；一个 workgroup 实际覆盖 64 个连续 linear pixels，通常是 64×1 屏幕条带。黑洞 shadow/critical curve 是二维曲线，因此真正的 8×8 tile 可以提高同一 workgroup 内的 cost/branch coherence。

同进程 timestamp-query A/B 曾以旧 linear strip 为 benchmark-only baseline；Apple M5/Metal 的位置对称实验给出约 4% 的小幅但可重复收益，因此 production 采用真正的 8×8 屏幕 tile，并以有界 tile 矩形分批。历史 baseline 与 paired harness 在作出决策后均已删除；永久 benchmark 只测当前 production pipeline。该结果仍是单一 adapter/backend 证据，Vulkan 需要独立复测，不能从 Apple 固定跨平台收益。

不继续无界枚举 shape。只有后续 profile 显示 workgroup mapping 仍是主要成本时，才在相同 64 threads 下有限比较 16×4、32×2；每个 variant 仍必须通过 paired CI、顺序偏差与逐 pixel observable gate。Morton order 或 128-thread tile 只有 step heatmap/occupancy 证据支持时才重新开启。

完成 escape-direction map、interval capture 与 RK 热循环削减后，又按上述边界复测了 16×4：物理 workgroup 保持 64 lanes，并以 linear local index 映射回完全相同的 8×8 logical tile。921,600 个 branch 全等；第一次 32 对为 `-0.626%`、CI `[-1.286%, -0.002%]`，但约 `0.788%` half-order bias 已大于点估计，独立第二次反向为 `+0.414%`、CI `[-0.665%, +1.971%]`。因此保留 8×8，并移除 shape override/额外 pipeline；没有证据支持继续枚举 32×2。
Metal 原生 pipeline 明确把 [`threadExecutionWidth`](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth) 与 [`maxTotalThreadsPerThreadgroup`](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/maxtotalthreadsperthreadgroup) 作为 pipeline/device-dependent 属性；在 wgpu 抽象层只能依据 adapter limits/subgroup properties 与 timestamp A/B，不能硬编码一个“Apple SIMD 宽度”。

### 5.2 Global shared transfer map 已进入 production

第一版 workgroup-local transfer 让每个 8×8 tile 独立追 `(0,4,8)²` 九点。它通过全像素 branch/direction gate，并在 Apple M5/Metal 上取得 `-18.809%`、95% CI `[-22.941%, -14.456%]`，但相邻 tile 重复计算边界 node，且九条长 trace 会让其他 lane 等待。它作为可否证中间基线有价值，不再是 production 数据流。

当前 production 先用独立 escape-map pass 在全局 4-pixel grid 上追踪每个共享 node 一次，把 Horizon/Escape tag 与 octahedral-quantized Escape direction 压成一个 `u32`；trace pass 再让每个 8×8 tile 读自己的 `3×3` stencil。九点必须全部 Escape，且四边中点与中心相对归一化整 tile corner direction 的 chord residual 不超过 `min(半像素角, 3.0e-4 rad)`，整 tile 才以同一 3×3 stencil 的四个 4×4 子格做连续分片双线性 direction reconstruction。其他 tile 执行原始逐像素 KS，并复用 tile 内四个 node。两个 pass 位于同一 command buffer 的独立 compute usage scope；同步依据是 wgpu 的资源转换，不是 backend 私有 barrier。

1280×720 全像素合同要求 FP16 terminal tag 全等，并对所有 reconstructed Escape 比较未 tone-map 的 direction。Apple M5/Metal 两次 32 对位置对称 `ABBA/BAAB` A/B 相对 workgroup-local 路径分别为 `-37.941%`、CI `[-41.091%, -34.885%]` 与 `-27.923%`、CI `[-32.350%, -23.353%]`；初始 `2.0e-4` 准入重建 `621,084` 个 Escape pixels。把内部 residual 门限预注册为 `2.5e-4 rad` 后，相对原门限的 16/32 对分别改善 `10.432%`（CI `[-16.384%, -5.848%]`）和 `6.551%`（CI `[-11.095%, -2.015%]`），直接 full-KS gate 重建 `653,604` pixels，最大 chord² `9.251701e-8`。在相同 3×3 nodes 上改用分片重建，把同一 accepted set 的最大 full-KS chord² 降到 `2.320720e-8`；利用该余量把内部 gate 一次性扩大到仍低于最终合同的 `3.0e-4 rad`，重建 `677,064` pixels，最大 chord² `2.348202e-8`，并相对 2.5e-4 整 tile 双线性再改善 `4.775%`、CI `[-5.683%, -3.957%]`。全部方案均保持 921,600 个 branch 全等，最终预算是 `3.82e-4 rad` 对应的 `1.45924e-7` chord²；spin/near-field/far-field/near-axis 参数矩阵通过，因此 production 采用 3.0e-4 分片方案。16×16 tile 用同一 gate 的 32 对实验为 `-0.221%`、CI `[-4.717%, +4.762%]`，故恢复 8×8。这里保留 4% 历史布局收益、19% local transfer、28–38% global map、6–10% 初次门限改进与 4.8% 分片改进的理由相同：置信区间、重复方向、正确性和复杂度一起判断，不设机械倍数门槛。

global map 稳定后又单独测试“8×8 workgroup 不变、四个 workgroup 共享一个 16×16 coarse transfer cell”，避免把旧 256-lane workgroup 的否决机械外推。它通过全像素 branch gate，最大 direction chord² `4.365467e-8` 仍在预算内，但 16 对相对 8×8 cell **慢 `29.126%`**、CI `[+24.004%, +34.100%]`；更稀 node 带来的节省被更低 reconstruction acceptance 与 fallback 成本压过。候选已完全移除，production 不增加 cell-size override。

另一个 benchmark-only `16→8` 层级原型先追 8-pixel coarse nodes、分类 16×16 macro，只为不稳定 macro 补追 4-pixel nodes。第一版 full-grid selective dispatch 保持全屏 branch 全等与 direction gate，但两边启用同一 interval capture 后，16 对相对当前 map 仅 `-0.649%`、CI `[-1.982%, +0.781%]`。第二版把每个不稳定 macro 的 16 个非 coarse nodes 压成一个 coherent 4×4 workgroup，并用确定 ownership 消除 shared-node 重复；正确性仍通过，却在 16 对中确定回退 `+8.798%`、CI `[+5.324%, +12.344%]`。空 lane 不是唯一瓶颈，额外分类/dispatch、16-thread workgroup occupancy 与 ownership 分支已经让层级 node 生成失去收益证据；两版均完整移除，不再升级 atomic/indirect queue。

### 5.3 Interval Kerr capture fast path

对纯 Kerr，径向势可写成

\[
R(r)=E^2r^4+\left[-2Ea(L-aE)-((L-aE)^2+Q)\right]r^2
+2((L-aE)^2+Q)r-a^2Q.
\]

若 backward ray 初始向内，且能证明 `R(r)>0` 覆盖 `[r_+,r_obs]`，就不存在外部 radial turning point，可直接判定 Horizon。实现不求四次方程的根，而是把区间分成 12 段；每段 degree-4 power coefficients 转为 Bernstein coefficients，只有全部向外扩张 interval 的下界严格为正才接受。`IntervalF32` 以 `vec2<lower,upper>` 表示，在每个 WGSL `+/-/*` 后通过 bit reinterpretation 建立优化屏障并向外扩一 ULP，near-zero/FTZ 扩到最小 normal；`E/Lz/Q/r_obs` 再用相对 `2^-12` envelope 包住 host→shader 与初值/invariant seam。Kerr–Newman 只在 constant coefficient 增加 exact `-q_e²[(Lz-aE)²+Q]` interval。任一非有限、near-extreme、`|a|>0.9`、`r_obs>256M`、near-axis、径向导数 margin 不足或区间条件失败都回退完整 KS。[WGSL floating-point evaluation](https://www.w3.org/TR/WGSL/#floating-point-evaluation)

[可复现实现](scripts/src/gravlume_research/checks/kerr_capture.py)用 SymPy exact algebra 验证 Kerr quartic 与 quartic→Bernstein 恒等式；pytest 中的 Hypothesis 原生 binary32 strategy 生成、收缩 strict-f32 primitive/Bernstein enclosure 反例，显式 examples 固定 signed zero、subnormal、normal 边界与最大 finite。检查另覆盖 packed-horizon seam 和 mutation witnesses。属性测试补充数值实现，不替代符号证明；版本与执行政策见[统一 Python 研究工具链](python-research-tooling.md)。[Hypothesis `@given`](https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.given) [32-bit float strategy](https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#hypothesis.strategies.floats)

```text
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research kerr-capture
```

脚本明确不把工程化 physical-input envelope 夸大成连续域形式证明。GPU oracle 另对 default、负自旋、`r_obs=5M`、远场、near-extreme 与 near-axis profile 做全像素 branch/direction 比较；后两者必须零接受。一次真实 near-extreme false accept 正是由该矩阵发现，随后收紧为 explicit fallback。

默认 1280×720 接受 `56,937/921,600` pixels（`6.178%`）。在已经采用 global map 后，Apple M5/Metal 两次 32 对增量分别为 `-9.156%`、CI `[-10.733%, -7.707%]` 与 `-5.946%`、CI `[-9.573%, -3.652%]`，branch tags 全等，Escape direction 完全不变。这个 shortcut 只服务 presentation Horizon/color；科学 record capture 仍执行完整 KS。它没有 2× 收益，但实现局部、单向保守、fallback 完整且两次稳定改善远高于其维护成本，因此保留。

分段数只做一次有界反例实验：`12→8` 不扩大接受域且全像素 gate 仍过，但覆盖从 `56,937` 降到 `51,881`（`5.629%`），16 对增量只剩 `-2.562%`、CI `[-4.423%, -0.299%]`；恢复 12 段后的相邻 16 对为 `-7.472%`、CI `[-10.448%, -5.225%]`。因此 production 固定 12 段，不增加 runtime override，也不继续枚举 4/16/24 段。

另一个只会 early-reject、不扩大接受域的候选在 interval 构造前检查 `((L-aE)^2+Q)/E^2≤64`。它保持 `56,937` 接受像素与全部 oracle gate，但 16 对增量只有 `-4.112%`、CI `[-5.280%, -2.654%]`，低于相邻无筛选 12 段的 `-7.472%`；额外 scalar 分支/计算没有被跳过的 interval 工作抵消，故候选已移除。

把 `interval_scale` 从通用 interval multiply 特化为按 scalar 符号只乘两个端点，也通过 strict-f32/SymPy 与 GPU oracle；但 16 对 capture 增量为 `-4.328%`、CI `[-4.790%, -3.833%]`，仍弱于相邻原实现 `-7.472%`。在没有直接 old/new capture 对照证明净改善前，不用源码级乘法计数替代实测，特化已移除。

把 12 个 Bernstein 段从单调 horizon→observer 顺序改成两端交替检查，接受域与 921,600 个 branch 完全不变；但 16 对相对原顺序只有 `+0.152%`、CI `[-1.110%, +1.412%]`。非 capture ray 的实际失败段分布不足以偿还 index 选择，override、额外 pipeline 与实验 API 均已移除。

单 ray 热循环里，`geometry_at` 过去在 RK2/3/4 中间 stages 也计算 `singularity_measure`，但 RHS 只消费 metric 与空间导数；guard 只在初态、每个 committed endpoint 和 localized terminal state 读取。production 现在只在这些可观察位置计算 guard，中间 stage 跳过。全屏 branch/direction gate 通过，32 对相对相同 transfer+capture baseline 改善 `3.148%`、CI `[-5.713%, -1.390%]`；没有新增状态、资源或 public seam，因此这项小而稳定的收益获准保留。相反，完全关闭 presentation 的 travel-time/metadata 记账只得到 `+0.160%`、CI `[-3.496%, +4.635%]`，候选已移除，科学与展示路径继续共享同一完整 observable policy。

### 5.4 为什么不直接把 RK4 拆成 wavefront

wavefront 的确定收益来自让每个 kernel 以更一致的工作开始并隔离 register-heavy 阶段；确定代价是 kernel 间所有 live state 都要读写 global memory。PBRT 的实现因此会融合部分阶段减少 memory traffic，而不是把算法机械地拆到最细。[PBRT v4](https://www.pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Mapping_Path_Tracing_to_the_GPU) Laine、Karras 与 Aila 也同时量化了 divergence/register 优势和每 path 212-byte global state/queue 成本，说明“wavefront”不是免费开关。[原论文](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)

Gravlume 若每 `K` 个完整 RK step 暂停，需要持久化至少 position/momentum、初始 invariants、maximum drift、travel time、step/flags；若还保存 endpoint RHS 来保留当前复用，状态更大。全屏 indexed state 很容易重新吃掉 production capture planes 刚释放的上百 MiB。持久化完整 `Geometry` 尤其错误：它含 radius、metric intermediates 与 gradients，应该在 resume 重算或通过实测决定只缓存 RHS。[Geometry definition](../../crates/gravlume-render/src/shaders/trace_protocol.wgsl)

只有当 histogram 显示明显长尾（例如 p95/p50 与 late active ratio 足够高）时，才实现 `K=8/16/32` 三个 fused-step chunk 候选：

1. chunk 内保持完整 RK tableau、event 与 invariant 逻辑；
2. chunk 末只 compact unfinished indices；
3. 下一轮用 indirect dispatch，不做 CPU count readback；
4. timestamp 分开记录 integrate、compact 和 args build；
5. 总 state bytes/pixel、queue worst case 与 rebuild peak 一并纳入 gate。

如果 compaction 总时间没有超过 full-screen inactive early-return 的收益，就保留简单路径。`wgpu` 能发 indirect dispatch 只证明 API 可表达，不证明 workload 合算。

### 5.5 Persistent threads 的有限适用性

CUDA ray-traversal 研究用 persistent threads/动态 work distribution 改善硬件利用率；其结果建立在特定 CUDA warp 与硬件上。[Aila–Laine 2009](https://research.nvidia.com/publication/2009-08_understanding-efficiency-ray-traversal-gpus) 在 portable WGSL 中让每个 invocation 完整追一条 ray 后从 global atomic 再取一条，并不能保证回收同一 SIMT group 内尚未 reconverge 的 masked lanes；同时单个全帧 persistent dispatch 又违背当前明确的 batch/watchdog 时间边界。

因此不把“永久 worker pool”作为主线。更干净的 GPU work distribution 是上一节的两遍结构：快路径只产出困难像素 index，第二遍对 compact queue 做 indirect KS。它只有有限 pass 数、无 continuation state，而且对 unsupported/uncertain domain 有物理含义。

### 5.6 Subgroup 只优化 queue，不优化方程

WGSL subgroup ballot、exclusive add、elect 可以让一个 subgroup 先做局部 prefix，再以一次 global atomic 预留连续 queue 区间；它不会减少 geodesic RHS，也不会自动修复不同 ray step count。[WGSL subgroup built-ins](https://www.w3.org/TR/WGSL/#subgroup-builtin-functions)

锁定版本有必须遵守的实现边界：

- `wgpu-types 30` 的 `SUBGROUP` 覆盖 compute/fragment，`SUBGROUP_BARRIER` 另需 `SUBGROUP`；当前 queue ballot/scan 没有 subgroup memory dependency，不应顺手请求 barrier。[locked feature source](https://docs.rs/crate/wgpu-types/30.0.0/source/src/features.rs)
- subgroup width 不是 32 常量；Vulkan 的 `subgroupSize` 是 implementation-dependent power of two。[Vulkan subgroup properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceSubgroupProperties.html) `AdapterInfo::subgroup_min_size/max_size` 只能记录 capability range：锁定的 wgpu-hal 30 Metal backend 将其报告为典型 `4..64` 桶，并不暴露当前 pipeline 的 `threadExecutionWidth`，所以实际 occupancy 仍要用 Metal profiler 与 timestamp A/B 确认。[locked adapter source](https://docs.rs/crate/wgpu-types/30.0.0/source/src/adapter.rs) [locked Metal backend](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/metal/mod.rs)
- Metal backend 只在 `supports_simd_scoped_operations` 时报告 subgroup/barrier；Vulkan backend 检查所需 stages/operation bits，并在启用 subgroup 时允许 varying subgroup size。[locked Metal source](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/metal/adapter.rs) [locked Vulkan source](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/vulkan/adapter.rs)
- 标准 WGSL 已有 `enable subgroups`，但 Naga 30 的 parser 把该 directive 标记为 unimplemented。若重启这一 research variant，必须在独立实验中请求 device feature、按锁定 parser 的实际能力省略 directive，并用同版 Naga parse/validate；它不属于 production platform contract。升级 wgpu/Naga 后重新核对，不能永久固化这个例外。[locked Naga source](https://docs.rs/crate/naga/30.0.0/source/src/front/wgsl/parse/directive/enable_extension.rs)

实现顺序应是：普通 workgroup prefix + atomic baseline，确认 queue 本身值得存在，再增加 subgroup variant。若 subgroup 版在 Metal/Vulkan 任一目标上未越过预设收益，就只保留 baseline。

## 6. Ray differentials、coherence 与“脏”画面

Igehy 的 ray differentials 用 image-plane derivatives 近似相邻 ray footprint，以局部 filtering 代替纯 supersampling。[Igehy 1999](https://graphics.stanford.edu/papers/trd/) DNGR 更直接相关：它为近场、任意运动 Kerr camera 积分 elliptical ray bundles/geodesic deviation，并把 bundle filtering 作为获得平滑、无 flicker IMAX 画面的关键。[James et al. 2015](https://arxiv.org/abs/1502.03808)

但这不是免费的 trace 加速：

- geodesic deviation 需要额外 bundle/Jacobian state 和更高阶 metric 信息；当前 shader 只有 Hamilton RHS 所需导数；
- 用相邻有限差分代替 deviation 至少增加 2–4 条 ray；
- critical curve、不同 termination/branch 与 caustic 上一阶 footprint 本来就失效，必须真实 sample/refine；
- 它首先解决 source filtering/temporal quality，只有在能可靠跳过像素时才成为性能方法。

当前更便宜的诊断顺序是：

1. 已把高频 `analytic_sky` 换成无 seam 的平滑低频方向图；实图中“脏点”消失且 CPU/GPU direction gate 不变，确认主要问题在 appearance sampling；
2. 保存或临时生成 compact `direction + termination/branch` map，在后 pass 用相邻 source directions 估计 angular footprint；只在 semantic key 一致时过滤；
3. 在 branch split/高 Jacobian tile 强制真实 samples，不跨 horizon/escape 或多像分支插值；
4. 最后才比较 analytic deviation bundle 与 finite-difference rays。

`dpdx/dpdy/fwidth` 只定义在 fragment shader 的协作 quad 中，当前 compute trace 不能直接对 source direction 调用它们；若要用硬件导数，应把 compact semantic map 交给后续 fragment pass，否则就在 compute 后 pass 显式读取同 branch 邻居。[WGSL derivatives](https://www.w3.org/TR/WGSL/#derivative-builtin-functions)

### 6.1 Shadow coverage 的最小原生管线

截图中的黑色轮廓是 horizon-capture 与 escape 的屏幕分界，不是把事件视界当作普通几何表面投影后得到的多边形边。因此第一版抗锯齿应使用 branch-aware shadow coverage，不对 tone-mapped RGB 做 FXAA：

1. base trace 继续对每个像素追一条中心光线，并把 `Rgba16Float.a` 定义为 compact discriminator：Horizon `0`、Escape `1`、其他 termination 用负的可精确表示整数；display 只消费 RGB，不把 alpha 解释为普通透明度。
2. 全屏 `classify_shadow_edges` 把同一 HDR view 作为非过滤 `texture_2d<f32>` 读取；当中心是 Horizon/Escape，且上下左右至少一个合法邻居属于另一 branch 时，通过一次 `atomicAdd` 把该 pixel index 追加到 bounded `u32` list。每 pixel 只有一个 classify invocation，所以不需要另外的 bitset 去重。
3. list capacity 取 `min(width * height, 8 * (width + height))`。计数器可以超过 capacity，但 list store 必须先检查 slot。第二个 dispatch 固定启动 `ceil(capacity / 64)` 个 workgroup；每个 invocation 先读最终计数，若 `count > capacity` 或自身 index 超出 count 就返回。因此极端不连续画面会完整保留 base，不会只抗锯齿半圈，也不需要第三个 finalize pipeline 或 indirect-dispatch control buffer。
4. `refine_shadow_edges` 对 list 中每个 pixel 追四条质心在像素中心的 rotated-grid subray，并在 scene-linear HDR 中平均。只有四条全为 Horizon/Escape 才 `textureStore`；任何 subray 为 Uncertain/数值失败时不写该 texel，原有 base 值自然保留，不需要为了回退而读取 write-only target。成功 refine 后 alpha 是四个 subray 的 escape coverage `0, 0.25, …, 1`。

项目只发布原生 Metal/Vulkan，不以 WebGPU 可移植性为设计目标。`texture_storage_2d<rgba16float, read_write>` 因此不能仅因属于 native extension 就被排除：锁定 wgpu 30 可在请求 `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` 后使用它，当前 Apple M5/Metal probe 也实际报告并成功 dispatch 了 `STORAGE_READ_WRITE`。但它仍不能把 classification 和 refinement 合成一个正确的 in-place dispatch：一个 invocation 读取邻居 branch 时，另一个 invocation 可能已覆写同一 texel；read-write capability 不提供跨 workgroup 全局 barrier，这会形成未排序的读写冲突。两阶段管线是同步语义要求，不是 Web fallback。第一版仍使用 sampled `textureLoad` classification 与 write-only `textureStore` refinement，因为它无需新增 feature、在既定 Metal/Vulkan 后端都有明确资源转换，并且 native extension 对本算法没有减少 dispatch。未来只有真实 A/B 证明 native read-write variant 有收益且数据流已消除竞态时才请求该 feature。[wgpu native format feature](https://docs.rs/wgpu/30.0.0/wgpu/struct.Features.html#associatedconstant.TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) [Apple resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization) [Vulkan synchronization](https://docs.vulkan.org/spec/latest/chapters/synchronization.html) [locked Metal format caps](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/metal/adapter.rs) [locked Vulkan format caps](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/vulkan/adapter.rs)

base、classify 和 refine 可在同一 command buffer 的独立 compute pass 中，也可在同一 compute pass 内切换 pipeline 后连续 dispatch。关键事实不是浏览器支持，而是锁定的 native wgpu-core 30 在每个 compute dispatch 前建立独立 resource-usage scope 并插入所需 backend barrier，因此同一 texture 可按 `storage-write → sampled-read → storage-write` 转换，不需要第二张 FP16 target。[locked compute usage scopes](https://docs.rs/crate/wgpu-core/30.0.0/source/src/command/compute.rs)

scratch 只有一个 atomic count 和 bounded `u32` edge list。refine 将两者作为 read-only storage；classification 将其作为 read-write storage。固定上限 refine dispatch 在 4K 也只有 48,000 个廉价 invocation（750 个 64-lane workgroup），相对 8.3M classification pixels 与昂贵 subray trace 很小。只有 timestamp A/B 证明这些空 lane 或固定 dispatch 成为瓶颈，才升级到 indirect/finalize；不能先为未测得的开销增加 pipeline、buffer usage 与验证复杂度。

将两个 postpass 追加到最后一个 base batch 的 submission，然后再 resolve timestamp/copy readback。map completion 因而同时证明 base 和 coverage 全部完成；现有 generation 校验只在此后提升 candidate，不需要第二个 CPU/GPU 往返或新的可见中间态。Sky plan 使用四个 query tick：一对包围 escape-map pass，一对包围含 classify/refine 的 trace pass；surface plan 没有空 escape pass，只用一对 tick 包围 trace。各实际 pass 的 duration 相加后驱动 batch budget。只有具名 benchmark/profile 需要更细地定位子阶段。publication correctness 依据是 queue 命令顺序与 generation，不是 timestamp 值。[compute pass timestamps](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePassTimestampWrites.html)

4K shadow capacity 是 `48,000` indices：按 16-byte 最小 binding size 计算为 `192,016 B`。global node map 的 4K grid 是 `961×541` 个 `u32`，即 `2,079,604 B`；每个 tracing candidate 的 scratch 合计 `2,271,620 B`。资源计划不再用固定 bytes/pixel 猜 lifecycle：初次 4K generation 为 `101,804,428 B`，completed 4K scene 上重建为 `201,337,220 B`，cold rebuild 为 `203,608,848 B`，都通过 `256 MiB`。若旧 4K candidate 仍 active，同尺寸 replacement 峰值为 `269,964,040 B`，会被 typed admission 拒绝。这仍是不包含 driver heap/alignment 的项目 core-resource admission，不声称是实测总显存。

一个可维护的接口应区分 `TerminalMap`、`PathSampler` 与 `DiagnosticCapture`：sky/surface appearance 只消费 terminal direction/branch/time，volume 需要有序 path checkpoints，科学验证需要 f32 records。不能为了让一份 analytic terminal solver 看似通用而伪造 volume path，也不能为了测试 capture 让 presentation 常驻全部记录。

## 7. 候选实验优先级

| 顺序 | 实验                                                                                  | 主要假设                                                                                    | 保留门槛                                                                                                                                               | 主要风险                                                                           |
| ---: | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
|   E0 | production/capture 资源拆分（已实现）；GPU histogram（待做）                          | 48 B/pixel 记录是 resize/VRAM 根因，统计可替代 production readback                          | 1440p 新 generation 核心约 42.2 MiB；完整 diagnostic 仍通过；trace 无显著回退                                                                          | 未来语义字段被误删                                                                 |
|   E1 | 64×1 vs 8×8 paired A/B（Metal 已采用；Vulkan 待复测）                                 | 二维 cost coherence 可减少 masked lanes                                                     | exact-output gate；Metal 稳定约 4% 的明确小收益获准进入 production                                                                                     | backend 特异、Vulkan 可能不复现                                                    |
|  E1a | 16×4 physical workgroup / 同一 8×8 logical tile（已拒绝）                             | 相同 64 lanes 下可能改善 occupancy                                                          | 两次 32 对为 `-0.626%`（CI 勉强排零）与 `+0.414%`（CI 跨零）；branch 全等                                                                              | 收益小于顺序偏差且不能复现                                                         |
|  E1b | 8×8 workgroup-local 3×3 Escape transfer + exact KS fallback（历史基线）               | 稳定天空区域可减少逐像素 ODE                                                                | 三场景全像素 branch/direction gate；Metal `-18.809%`，95% CI `[-22.941%, -14.456%]`                                                                    | 相邻 tile 重复 node；SIMT 等待；已被 E1d 取代                                      |
|  E1c | 16×16 cooperative tile，保持同一 3×3 stencil/门槛/fallback（已拒绝）                  | 更大的协作 tile 可摊薄 stencil ray                                                          | 三场景全像素 gate 通过；32 对 Metal A/B 为 `-0.221%`，95% CI `[-4.717%, +4.762%]`，落在噪声内，候选已移除                                              | 256-lane 空闲与更低 acceptance 抵消理论 ray reduction                              |
| E1c2 | 8×8 workgroup + 16×16 global coarse cell（已拒绝）                                    | 保留 occupancy，只减少 global node 密度                                                     | branch/direction gate 通过；16 对 Metal 为 `+29.126%`，CI `[+24.004%, +34.100%]`                                                                       | reconstruction acceptance 下降，fallback 成本远超 node 节省                        |
|  E1d | 4px packed global node map + 8×8 resolve/fallback（已采用；Vulkan 待复测）            | 跨 tile 共享 stencil node 并缩短 workgroup 内长 trace 等待                                  | 1280×720 全像素 gate；两次 Metal run 改善 `27.923–37.941%`，各自 CI 排除零                                                                             | 两 pass 与约 2.0 MiB/4K map；经验 stencil 仍需 exact fallback                      |
|  E1e | 12-segment interval Bernstein Kerr capture（已采用支持域；Vulkan 待复测）             | 严格正径向势可直接证明无 turning point                                                      | 默认接受 6.178%；相对 E1d 两次 Metal run 改善 `5.946–9.156%`，各自 CI 排除零；全像素 oracle gate                                                       | 只判 capture；physical-input envelope 是受测 seam；near-extreme/axis 必须 fallback |
|  E1f | 8-segment interval capture（已拒绝）                                                  | 更少 coefficient checks 可能抵消覆盖损失                                                    | 全像素 gate 通过；覆盖降至 5.629%，16 对改善仅 2.562%；恢复 12 段为 7.472%                                                                             | 较宽 segment 使 interval lower bound 更难保持严格为正                              |
|  E1g | normalized separation early-reject（已拒绝）                                          | 高 impact-parameter ray 可跳过 interval coefficient work                                    | 覆盖与 gate 不变；16 对改善 4.112%，弱于无筛选 7.472%                                                                                                  | scalar guard 自身成本与 divergence 抵消节省                                        |
|  E1h | sign-specialized `interval_scale`（已拒绝）                                           | 两个 endpoint multiply 可替代通用四乘法/minmax                                              | formal/oracle gate 通过；16 对改善 4.328%，弱于相邻原实现 7.472%                                                                                       | 源码 operation count 没转成可归因 GPU 收益                                         |
| E1h2 | Bernstein segment 两端交替 early-out（已拒绝）                                        | 非 capture ray 可在更有区分力的 segment 更早失败                                            | 921,600 branch 全等；16 对 `+0.152%`、CI `[-1.110%, +1.412%]`                                                                                          | 失败段分布没有证据，index 选择抵消 early-out                                       |
|  E1i | `16→8` coarse classify + selective fine dispatch（已拒绝）                            | stable macro 跳过 4px fine nodes                                                            | full-grid 16 对 `-0.649%`、CI 跨零；coherent 4×4 fine workgroup 16 对 `+8.798%`、CI `[+5.324%, +12.344%]`；全像素 gate 均通过                          | dispatch/occupancy/ownership 成本；不再升级 queue                                  |
|  E1j | committed-only singularity guard（已采用）                                            | RK intermediate RHS 不消费 guard observable                                                 | full gate；32 对 `-3.148%`、CI `[-5.713%, -1.390%]`                                                                                                    | 必须继续在 every committed/localized endpoint 计算                                 |
|  E1k | presentation metadata/travel-time disable（已拒绝）                                   | 未显示字段可减轻寄存器/ALU                                                                  | 32 对 `+0.160%`、CI `[-3.496%, +4.635%]`                                                                                                               | 无稳定收益，且分裂 scientific/presentation policy                                  |
|  E1l | exact KS RHS factorization（已采用；Vulkan 待复测）                                   | Hamilton force 只需 contracted null Jacobian，不需三个完整 derivative vectors               | SymPy exact；完整 render/GPU gate；研究基线 Naga 30.0.0 不再生成动态 loop/array；Metal 1280×720 从 vector-Jacobian `21.951 ms` 降到 `14.446–14.826 ms` | binary32 非 bitwise；收益只在 Apple M5/Metal 实测                                  |
|  E1m | direct `Sigma`/gradient + cached reciprocal（已采用）                                 | `Sigma=root`、direct gradient 与 factored `grad f` 都是 exact identity；root 形式定义域更大 | SymPy exact；完整 render/GPU gate；旧局部 A/B 的 `1–2.5%` 波动不再否定更短依赖图                                                                       | binary32 非 bitwise；CPU oracle 保留独立 residual reconstruction                   |
|  E1n | 6D phase、`vec4<t,x,y,z>` RHS、全域 Carter、Hermite event（已采用）                   | stationarity 精确固定 per-ray `E`；Cartesian Carter 无 axis seam；单调 cubic guard 为四阶   | 持久化 SymPy；coordinate-time translation、axis/Schwarzschild、GPU/reference/event residual gate 全过                                                  | near-tangent/非单调 guard 回 chord；step policy 未随之放宽                         |
|  E1o | Kerr–Newman interval capture（已采用受限域）                                          | KN radial quartic 只在常数项增加 `-q_e²[(Lz-aE)²+Q]`                                        | SymPy exact；1280×720 KN full-KS branch/direction 全像素等价                                                                                           | 仅严格亚极端 neutral photon；near-extreme/axis/不确定全部 fallback                 |
|   E2 | BS3(2) FSAL + 量化 step + bounded fallback                                            | embedded estimate 能以更少 RHS 达到相同 observable                                          | 全流程 p50/p95 至少 20%；最终 gate 全过                                                                                                                | 三阶需更多 step、reject 分歧                                                       |
|   E3 | pure-Kerr exterior reciprocal-Mino numerical fast path + inline KS fallback（已拒绝） | polynomial separable RHS 能消除主要 geometry 成本                                           | 默认接受 84.136%；256 对 `-35.768%`，但 320×180 accepted ray travel time 越过合同                                                                      | constraint/winding gate 未界定 terminal phase；实现已删除                          |
|   E4 | 完整 Kerr elliptic terminal solver + 同一 fallback                                    | root-aware special-function 求值避免几十步 phase accumulation                               | 高精度全 accepted-pixel oracle；相对 interval+KS 有实质收益；复杂度有 fixture 覆盖                                                                     | roots/special functions/近场初值复杂                                               |
|   E5 | K-step fused wavefront + compaction                                                   | measured long tail 足以覆盖 state traffic                                                   | integrate+compact+args 总 p50/p95 至少 20%，内存仍过项目 gate                                                                                          | 显存、bandwidth、多 pass                                                           |
|   E6 | subgroup queue build                                                                  | global atomic contention 已成为可测瓶颈                                                     | 各支持 backend 稳定改善、CI 排除零且代码成本受控；baseline variant 保留                                                                                | 宽度/工具链差异                                                                    |
|   E7 | branch-aware source footprint/reconstruction                                          | 当前视觉问题主要是 sampling；稳定区可少 trace                                               | source/filter quality 与 temporal gate 通过，且总 GPU time下降                                                                                         | caustic false interpolation                                                        |

保留门槛按风险与成本分层，不使用统一百分比：布局或局部、可完整 fallback 的 accelerator，只要位置对称 CI 排除零、收益跨复测稳定、正确性 gate 全过且代码成本低，约 4% 也值得保留；引入新物理 domain、special functions 或大规模持久状态的 E3/E4/E5 则需要显著更高收益来偿还验证与维护面。5% noise threshold 只标记单次 run 的可疑区间，不是硬淘汰线；若 run-to-run noise 更大，先扩大样本并使用交错配对，而不是按一次最好结果决策。

### 明确延后或否决

- 无界、每 ray 连续 step-size 的 DP5(4) megakernel：在本 shader 中先验 stage cost与 divergence 不利，除非 E2 数据推翻成本模型；
- 把四个 RK stage 分 kernel：必然增加每 stage global state traffic；
- 全帧 persistent dispatch：不符合 batch responsiveness，且 portable WGSL 不提供 CUDA warp 调度保证；
- symplectic 直接替换：当前短终止轨迹没有证明其长期结构优势能转成 observable/time 收益；
- neural trajectory surrogate 直接给物理结果：没有 near-critical branch 的可证明 worst-case bound；最多做 conservative classifier/hint，错误或低置信必须回到已验证 solver；
- 直接把 AART/Krang 当近场通用实现：两者的一手来源都限定了更窄的问题域。

## 8. 正确性门槛

所有 accelerator 最终输出沿用[validation contract](../validation.md)，不能另设更宽松的“fast mode”：

- regular fixture termination 完全一致；regular domain 最终 NumericalFailure/StepExhaustion/未回退 Uncertain 为 0；
- escape/source direction ≤ `3.82e-4 rad`（当前 0.35 pixel）；
- travel time ≤ `1e-3 M`；
- null、E、Lz、Carter 四项 recorded drift 各自 ≤ `0.05`；
- Frequency Ratio relative error ≤ `2e-3`（当目标 profile 产生该 observable）；
- surface event position/residual ≤ `5e-3 M`；
- near-critical paired rays 不得给出错误但“确定”的 branch；安全带内可以回退，最终结果仍必须通过；
- generation invalidation 后 stale publish 为 0。

### 8.1 解析/Mino 形式化检查

1. 用 CAS 在 exact rational/symbolic 表达下从 Hamilton–Jacobi potentials 推导项目符号约定；逐项验证 `u=1/r` 变换、二阶势 RHS、`E,Lz,Q` 与 outgoing KS/BL covector Jacobian。
2. 对 root discriminant/topology 使用 binary64/高精度 oracle 划分安全区；near-multiple roots 生成成对 fixture，证明 classifier 只会保守回退，不会跨类接受。
3. 以 80-bit 以上数值在 Mino/elliptic 与独立 f64 DP5(4) KS reference 之间比较 checkpoint、turning count、terminal event、direction 与 `t,φ`；不是只比较守恒量。
4. 变形关系至少覆盖质量尺度、`q_e → -q_e`（neutral photon metric 不变）、spin/image reflection、共同 coordinate-time 平移和 step-halving convergence。
5. subextreme、near-extreme、exact-extreme、superextreme、near-axis、equatorial、近 horizon observer 与远场分别标注 supported/fallback；不以一个默认 Kerr scene 外推。

### 8.2 GPU 数值与跨后端检查

- 当前 macOS/Metal 可做完整运行 gate；Vulkan 在未运行真实 adapter 前只能完成 compile/Naga/layout 检查，文档必须明确“未运行”，不能把 shader 编译当数值验证；
- 比较 observable tolerance，不要求 Metal/Vulkan bit identity；WGSL 对 floating-point evaluation、重关联和有限数学有自己的允许范围。[WGSL floating-point evaluation](https://www.w3.org/TR/WGSL/#floating-point-evaluation)
- 每个 fast-path accepted pixel 都记录 condition/error bucket；测试 capture 能反查为什么接受。fallback 是 machine-readable outcome，不是静默修图；
- event localization 只在 bracket 内；同一步多 event 保留现有 priority/tie policy；reject 不产生任何已提交副作用。

建议 fixture 场景矩阵：Minkowski/Schwarzschild；默认 `a=0.8,r_obs=30M,FOV=45°`；`a=±0.99`；near-axis；Kerr–Newman charge sweep；sub/extreme/superextreme；critical impact parameter 两侧；720p/1080p/1440p 与 odd extents。性能只跑 representative 子集，物理 gate 跑小而高价值的 deterministic samples。

## 9. 性能测量合同

每份 A/B artifact 固定 target、OS、adapter、driver、power mode、release profile、scene fingerprint、extent、shader/feature variant、warm-up 与样本数，并至少记录：

- invalidation → complete publish 的 CPU wall latency；
- trace、fast path、queue/compact、fallback、appearance、publication 各 pass GPU p50/p95；
- ms/megapixel、rays/s、accepted/rejected steps、RHS 与 geometry evaluations/ray；
- step `p50/p90/p95/p99/max` 与 active ratio by step/chunk；
- fallback ratio 及其按 root/event/condition 原因分解；
- termination distribution 与所有 correctness error quantiles/max；
- steady bytes/pixel、fallback/state queue worst case、resize/rebuild peak；
- 只有具名 vendor profiler/offline compiler 才能报告 register、occupancy 或 memory traffic；源码推测不能作为完成证据。

旧 GPU ray tracer 的绝对数字不能外推：GRay 与 Odyssey 都展示了 GPU 上大规模单精度 geodesic/RK 可达很高吞吐，但它们使用旧 NVIDIA CUDA 硬件、不同坐标/状态/精度和每 photon-step 指标。[GRay](https://arxiv.org/abs/1303.5057) [Odyssey](https://arxiv.org/abs/1601.02063) 本项目只把它们当架构可行性证据，所有决策以锁定 wgpu 30 的 Metal/Vulkan total-frame A/B 为准。

## 10. 最终建议

短期不要再花主要精力微调 batch。production 中未消费的 48 B/pixel capture planes 已移除，真正 8×8 布局、global packed transfer、selective shadow coverage 与 supported-domain interval capture 已分别通过 A/B 和数值 gate。下一步先补 aggregate accepted/fallback/step telemetry，并在 Vulkan 目标机复测相同 shader 与 gate；不要因为本地 Metal 数字良好就加入 backend-specific 产品分支。

outgoing-KS/BL physical-spin seam Gate A/B 已通过；restricted numerical-Mino Gate C/D 已否决。生产以 interval capture + 完整 KS 收束，下一项数学主线是 root-aware elliptic/Carlson terminal solver，而不是重做 fixed-factor scan。只有 step histogram 证明剩余长尾明显时才付出 wavefront state 的显存/带宽成本。ray differentials/branch-aware footprint 独立解决更一般的 source aliasing；当前 shadow coverage 只解决 Horizon/Escape 轮廓，不能代替 geodesic correctness。

这条路线把“前沿算法”约束为可失败、可回退、可测量的 accelerator，而不是另起一套无法与现有物理合同对齐的快图模式。

# GPU 测地线加速决策账本

本文是 GPU geometry 优化的历史性能与正确性账本，不定义当前 pipeline、资源或支持域；采用状态必须回到 [GPU 证据](../gpu-renderer.md)、[架构合同](../architecture.md)和实际源码核对。

**状态：混合决策账本。** 各实验是否已实现以第 7 节 ledger 与链接的当前合同为准。研究基线为仓库提交 `9dbdb71c0ad325c7c78ca518cbdff528daa2f2fe`，目标是找到能显著降低完整原生分辨率画面延迟、同时保留物理与数值合同的算法路线。本文延续[完整帧原子发布研究](atomic-frame-publication.md)：计算可以分批，画面仍只发布完整 generation；不重新引入可见低分辨率阶段或扫描式 reveal。

**当前纠正：** escape-direction map reconstruction 把
`GeometricSample.travel_time` 写成零，interval capture 也没有积分 coordinate time；旧全屏 gate 只比较
terminal 与 direction，漏掉验证合同的 `1e-3 M` travel-time observable。两条路径及专用测试现已删除，
当前 analytic sky 与 surface 都执行完整 Cartesian KS。下文 A/B 数字仍是历史性能证据，不再表示采用。

## 结论

1. 研究起点的默认 1280×720 trace 累计 GPU 时间曾为 `80.5–192.2 ms`。escape-direction map 与 interval capture 的历史 A/B 分别显示 `27.923–37.941%` 和 `5.946–9.156%` 的局部收益，但 gate 漏掉 travel time，因此这些数字不能授权 production，也不构成跨平台 60 FPS 声明。[当前证据](../gpu-renderer.md#适用域与限制) [资源合同](../architecture.md#内存与资源预算)
2. **三条不完整 fast path 均已否决或撤出**：numerical-Mino 在扩大 lattice 后越过 travel-time budget；escape map 与 interval capture 根本没有产生可信 travel time。可分离结构仍值得研究，但当前 production 直接执行 Cartesian KS。完整根因见[数值 Mino 结论](#21-数值-mino-实验为何失败以及为何转向解析路线)。
3. 在现有 Kerr–Schild 路径内，Bogacki–Shampine 3(2) FSAL pair 值得比完整 Dormand–Prince 5(4) 更早试验：首次 4 次、此后每 accepted step 3 次新 RHS，并自带 embedded estimate；但它是三阶方法，是否真的减少总 RHS 只能由 observable gate 和 GPU A/B 决定。[Bogacki–Shampine 原文](https://doi.org/10.1016/0893-9659%2889%2990079-7) [Dormand–Prince 原文](https://doi.org/10.1016/0771-050X%2880%2990013-3)
4. active-ray wavefront/compaction 不是第一步。它可能回收长尾 inactive lanes，但每轮必须把状态写回显存；在得到 step 分位数和 active-ratio 曲线以前，不能证明收益大于 state traffic 与多 pass 开销。若进入实验，应把完整 RK step 融在单 kernel 内，只在多个 step 的 chunk 边界 compact；不要把四个 RK stage 拆成四个 kernel。[Laine–Karras–Aila 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf) [PBRT v4 wavefront 取舍](https://www.pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Mapping_Path_Tracing_to_the_GPU)
5. 截图中的“脏”主要不能先归因于测地线噪声。研究基线的 `analytic_sky` 把经纬周期中央约 90–92% 设为高亮，两项取 `max` 后约 99.2% 天空被额外加上 `2.0` radiance。初次反转 mask 后，实际截图仍显示薄线在临界曲线处被放大为发白的环；当前实现因此进一步换成无 seam、不过曝的低阶球面多项式与局部轴向色标。每像素一个中心样本仍没有 branch-aware coverage，因此真正的 horizon/escape 轮廓抗锯齿是后续 appearance/filtering 问题，不能靠放宽积分误差或模糊几何掩盖。[preview shader](../../crates/gravlume-render/src/shaders/lensing_preview.wgsl)

已完成的顺序是：production/capture ABI 瘦身 → 2D/global coherence 与 interval 实验 → outgoing-KS/BL exact seam → numerical-Mino 否决 → branch-aware shadow coverage → 完整 observable 复核后撤出 map/interval。下一步算法主线是完整解析/半解析 Kerr terminal solver；只有 aggregate long-tail 证据支持时才做 wavefront/subgroup。BS3(2) fixed-step 探针仍未证明价值。

## 1. 研究基线与成本根因

研究基线的 lockfile 当时固定 `wgpu/wgpu-core/wgpu-hal/wgpu-types/Naga 30.0.0`、`bytemuck 1.25.2`、`glam 0.33.3`、`egui/egui-wgpu 0.36.1`、`pollster 1.0.1` 与 `winit 0.30.13`；当前依赖以 [`Cargo.lock`](../../Cargo.lock) 为准。macOS 只启用 wgpu Metal，Windows/Linux 只启用 Vulkan；当前 device 只请求 `TIMESTAMP_QUERY`。[workspace manifest](../../Cargo.toml) [render manifest](../../crates/gravlume-render/Cargo.toml) [capability baseline](../../crates/gravlume-render/src/capabilities.rs)

研究基线的 production shader 是一 invocation 一 pixel 的 megakernel：8×8 workgroup、最多 2048 个 radius-scaled fixed RK4 step，普通 accepted step 复用 exact endpoint 后仍有 4 次 geometry/RHS 求值；当时最终写 `rgba16float` HDR 与三个 16 B/pixel record plane。[current solver](../../crates/gravlume-render/src/shaders/geodesic_integration.wgsl)

默认场景的开发数据从平均/最坏 `882/2048` step 降到了 `61/132`，说明 outgoing Kerr–Schild chart、几何 step 与 endpoint reuse 已经拿掉了最明显的冗余；剩余时间更接近“每步数学本身 × 全屏像素数”，而不是发布调度问题。[实现证据](../gpu-renderer.md#适用域与限制) 坐标选择仍不可随意回退：backward ray 的方向与 horizon-penetrating chart 会改变数值适定性，相关分析见 Bozzola、Chan 与 Paschalidis 的[原论文](https://arxiv.org/abs/2310.02321)。

### 1.1 生产路径有一个独立的显存根因

历史实现只在 test build 暴露三个 record buffer handle，却仍在 production 创建、持有并写入对应资源，因此 `#[cfg(test)]` 没有消除成本。当前 diagnostic capture 已与 production ABI 分离：[trace pipeline](../../crates/gravlume-render/src/trace.rs) 与 [production solver](../../crates/gravlume-render/src/shaders/geodesic_integration.wgsl)。

研究基线的 frame accounting 是：record planes `48 B` + candidate HDR `8 B` + published HDR `8 B` + UI `4 B` = `68 B/pixel`。因而 2560×1440 是 `250,675,200` bytes（约 239.1 MiB），两阶段 rebuild 约 478.1 MiB。若 production presentation 不分配 full diagnostic planes，steady candidate+published+UI 核心变为 `20 B/pixel`，同一 extent 是 `73,728,000` bytes（约 70.3 MiB），下降 70.6%；当前实现又以 texture-view promotion 消除了 candidate 与同尺寸 published copy 在新 generation 内的重复。

当前 production ABI 只有 trace uniform、HDR storage texture 与 dispatch uniform；四个 record plane 仅由 test-only capture 注入。新 generation 分配 candidate HDR、UI 与 shadow edge queue，上一张完整 scene 独立持有，完成后直接提升 candidate texture view。删除 global map 后，3840×2160 的 cold、active 与 completed transactional rebuild 都通过 256 MiB 逻辑 gate；synthetic oversized plan 仍在分配前返回 typed rejection。该账本不包含 surface drawable、driver heap 或 alignment，也不冒充实测总显存。

采用的资源决策是把 presentation 与 diagnostic capture 分开，而不是让测试字段支配每帧资源：

- production trace 只创建画面与既定 refinement 实际消费的资源；
- test-only diagnostic capture 按显式请求临时创建完整 scientific planes 并做有界 readback；
- 若未来 source 重建确实需要 direction/branch，应按真实 consumer 定义最小、误差受检格式，不能把历史 `48 B/pixel` 测试记录原样常驻。

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

已删除的 reciprocal-Mino candidate 使用 `(u=1/r, u', μ=cosθ, μ')` 的 cubic polynomial RHS
自然越过 turning point，并数值积分 `φ,t`。锁定的
[`mino_step.py`](scripts/src/gravlume_research/checks/mino_step.py)形式化检查证明 classical RK4 local
defect 从 $h^5$ 开始、cubic Hermite interior defect 从 $h^4$ 开始；在固定平滑轨迹上，步长因子
$f$ 因而只有 $W(f)=\Theta(f^{-1})$ 与 $e_{global}(f)=O(f^4)$ 的局部模型，不能外推为
binary32 terminal observable 的全域证书。

候选曾通过低分辨率 strict-DP5(4) matrix 的 `219/256` cases；默认 `1280×720` 本机实验接受
`775,399/921,600` pixels（`84.136%`），历史 256-pair timing 相对 interval capture + KS 为
`-35.768%`，95% CI `[-36.390%, -35.189%]`。这证明可分离 polynomial dynamics 有真实性能信号，
但只覆盖受测 phase/profile。

`320×180` 扩展在 pixel `(175,51)` 找到最小正式反例：travel-time absolute error 约
`2.661354e-3 M`，超过 `1e-3 M` budget；更早 factor sweep 还出现约 `4.438e-4 rad` 的
escape-direction error，超过 `3.82e-4 rad` budget。Potential residual 只约束 energy surface，
winding cutoff 与稀疏 $Cf^4$ fit 都不能界定累计 azimuth/time phase。继续收紧全局 fixed factor
只会线性增加工作，并不补上证明缺口；production WGSL、pipeline constants、benchmark variants
与专用 tests 因而全部删除。

该候选只有在新的 phase-error certificate 能先验覆盖完整 accepted domain 时才可重开；增加抽样或
缩小经验步长不够。下一候选直接使用具名 root topology 与 elliptic/Carlson terminal integrals，
Cartesian KS 继续承担任何不确定域。复算命令与环境见[统一 Python 研究工具链](python-research-tooling.md)。

### 2.2 必须封闭的域

未来 analytic/Mino-derived variant 只接受同时满足下列条件的像素：

- pure Kerr、明确 subextremal、observer 与当前段都在 exterior；
- 从 validated tetrad/KS covector 得到的 `E,Lz,Q`、radial/polar sign 与 potential constraints 全部有限且落在安全区间；
- root/turning topology 不接近具名退化阈值，reciprocal state 约束在每个 checkpoint 内满足预算；
- horizon/escape event、source direction 与 travel time 可从同一 localized state/积分提交；
- Boyer–Lindquist/Mino 到 outgoing Kerr–Schild 的 `t,φ` 变换在 horizon bracket 前保持受控。

任何条件不满足都进入 KS queue，不允许让快路径猜 branch。Gralla–Lupsasca 的 `t,φ` integrals 含 `Δ` 分母，而当前 outgoing Kerr–Schild baseline 正是为了 backward horizon crossing 的正则性；因此 Mino 路径可以从 exterior 一侧定位 horizon event，却不能越过 BL coordinate singularity 后继续假装和 KS state 等价。[Kerr formulas](https://arxiv.org/pdf/1910.12881) [坐标分析](https://arxiv.org/abs/2310.02321)

可分离 solver 把 `E,Lz,Q` 当常量，因此“它们没有被更新”不能冒充数值正确性。accepted terminal 必须重建 outgoing KS position/covector，再用独立的现有公式复算 null/`E,Lz,Q` agreement；同时记录 `v_r²-R`、`v_μ²-U`、root separation/condition bucket。任何一项超过安全预算就回退，不能把构造出来的零 drift 当作 confidence。

legacy Mino prototype 曾因 outgoing chart 把 oblate twist 留在 `+a` 而产生最坏 `0.194 rad`
escape-direction error；exact SymPy legacy RED 与 corrected Kerr–Newman metric/covector/tangent GREEN
随后封闭了 chart/spin seam。修正后的 numerical candidate 仍被本节的 high-resolution
travel-time 反例否决。映射证据见 [KS–BL seam](kerr-schild-mino-map.md)。

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

| workload                                      | 最合适路线                                                   | 不应外推                                                |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| 当前固定 `a=0.8`、近场 observer、analytic sky | full Cartesian KS；研究 pure-Kerr elliptic terminal fast path | AART equatorial grid、Schwarzschild LUT                 |
| 只改 sky/appearance/HDR                       | 复用 terminal map，重跑 appearance pass                      | 重新追完整 geodesic                                     |
| 未来移动 camera/FOV/spin                      | 每 generation 重跑 analytic + KS fallback                    | 固定 observer LUT/旧 transfer map                       |
| 未来 Kerr–Newman/超极端参数                   | 通用 KS baseline；analytic KN 独立受限 variant               | 把 pure-Kerr root classifier 扩大命名即复用             |
| 未来 surface                                  | terminal map 加 source anchor/branch/frequency               | 只缓存 tone-mapped RGB                                  |
| 未来 volume/slow-light                        | ordered checkpoints 与 retarded time                         | 只有 terminal elliptic result 却声称支持 path transport |

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

一个已删除的 fixed-step 主解探针在 base step `0.1` 时越过 regular fixture 的 travel-time gate；
缩到 `0.05` 后合同通过，但增加的 step 数没有在 Metal smoke samples 中证明总时间收益。它不否定带
二阶误差估计、量化 tier 与有界 retry/fallback 的完整 embedded candidate，但证明“直接替换 RK4
tableau”不能作为 production 优化。

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

### 5.2 Global shared transfer map 历史实验（已撤出）

Workgroup-local baseline 让每个 8×8 tile 独立追 `3×3` nodes；它通过历史 branch/direction gate，
Apple M5/Metal A/B 为 `-18.809%`，95% CI `[-22.941%,-14.456%]`，但相邻 tile 重复边界 nodes。
Global variant 改为 4-pixel grid 上每个 node 只追一次，再按 exact branch key 与 conservative direction
residual 接纳 `3×3` stencil reconstruction。两个 pass 由 command ordering/resource transition 建立
可见性，不依赖 backend-private barrier。

Global variant 相对 local baseline 的两次位置对称 A/B 为 `-37.941%` 与 `-27.923%`，且当时的
terminal/direction gate 通过；后来发现它把 `GeometricSample.travel_time` 写成零，因此整条路径撤出。
旧 threshold sweep、accepted-count 调参与临时 shader overrides 不再保留为路线图。

两个结构反例仍有复用价值：保持 8×8 workgroup 但共享 16×16 coarse cell 时慢 `29.126%`，95% CI
`[+24.004%,+34.100%]`；`16→8` hierarchical variants 分别得到跨零的 `-0.649%` 和明确回退的
`+8.798%`。更稀 nodes 没有偿还 acceptance loss、额外 dispatch、ownership 与 occupancy 成本，
相关实现均已删除。

### 5.3 Interval Kerr capture 历史实验（已撤出）

对纯 Kerr，径向势可写成

\[
R(r)=E^2r^4+\left[-2Ea(L-aE)-((L-aE)^2+Q)\right]r^2
+2((L-aE)^2+Q)r-a^2Q.
\]

若 backward ray 初始向内，且能证明 `R(r)>0` 覆盖 `[r_+,r_obs]`，就不存在外部 radial turning point。被撤出的 prototype 不求四次方程的根，而是把区间分成 12 段；每段 degree-4 power coefficients 转为 Bernstein coefficients，只有全部向外扩张 interval 的下界严格为正才接受。`IntervalF32` 以 `vec2<lower,upper>` 表示，在每个 WGSL `+/-/*` 后通过 bit reinterpretation 建立优化屏障并向外扩一 ULP，near-zero/FTZ 扩到最小 normal；`E/Lz/Q/r_obs` 再用相对 `2^-12` envelope 包住 host→shader 与初值/invariant seam。Kerr–Newman 只在 constant coefficient 增加 exact `-q_e²[(Lz-aE)²+Q]` interval。任一非有限、near-extreme、`|a|>0.9`、`r_obs>256M`、near-axis、径向导数 margin 不足或区间条件失败都回退完整 KS。[WGSL floating-point evaluation](https://www.w3.org/TR/WGSL/#floating-point-evaluation)

[可复现实现](scripts/src/gravlume_research/checks/kerr_capture.py)用 SymPy exact algebra 验证 Kerr quartic 与 quartic→Bernstein 恒等式；pytest 中的 Hypothesis 原生 binary32 strategy 生成、收缩 strict-f32 primitive/Bernstein enclosure 反例，显式 examples 固定 signed zero、subnormal、normal 边界与最大 finite。检查另覆盖 packed-horizon seam 和 mutation witnesses。属性测试补充数值实现，不替代符号证明；版本与执行政策见[统一 Python 研究工具链](python-research-tooling.md)。[Hypothesis `@given`](https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.given) [32-bit float strategy](https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#hypothesis.strategies.floats)

```text
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research kerr-capture
```

脚本明确不把工程化 physical-input envelope 夸大成连续域形式证明。GPU oracle 另对 default、负自旋、`r_obs=5M`、远场、near-extreme 与 near-axis profile 做全像素 branch/direction 比较；后两者必须零接受。一次真实 near-extreme false accept 正是由该矩阵发现，随后收紧为 explicit fallback。

默认 1280×720 接受 `56,937/921,600` pixels（`6.178%`）。在已经采用 global map 后，Apple M5/Metal 两次 32 对增量分别为 `-9.156%`、CI `[-10.733%, -7.707%]` 与 `-5.946%`、CI `[-9.573%, -3.652%]`，branch tags 全等，Escape direction 完全不变。这个 shortcut 只服务当时的 presentation Horizon/color；科学 record capture 仍执行完整 KS。它没有计算 travel time，因此即使 branch gate 与性能 A/B 通过也不能满足完整 terminal observable，现已撤出。

四个 bounded microvariants 都未改善完整 candidate：`12→8` segments 降低覆盖并削弱收益；
normalized-separation early reject 与 sign-specialized interval scale 的额外分支没有偿还 interval work；
交替 segment order 的 CI 跨零。它们只证明源码 operation count 不能替代 paired GPU evidence，
对应 overrides、pipelines 与实验 API 均已删除。

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

将两个 postpass 追加到最后一个 base batch 的 submission，然后再 resolve timestamp/copy readback。map completion 因而同时证明 base 和 coverage 全部完成；generation 校验只在此后提升 candidate，不需要第二个 CPU/GPU 往返或新的可见中间态。当前 `GpuTimings` 只用一对 query tick 包围 trace pass，不为内部 dispatch 数量建立测试合同。publication correctness 依据是 queue 命令顺序与 generation，不是 timestamp 值。[compute pass timestamps](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePassTimestampWrites.html)

4K shadow capacity 是 `48,000` indices：按 16-byte 最小 binding size 计算为 `192,016 B`。删除 global node map 后，每个 sky candidate 只保留该 shadow scratch；cold、active 与 completed 4K transactional rebuild 都通过 `256 MiB`。资源计划仍在分配前拒绝 synthetic oversized plan；这只是不包含 driver heap/alignment 的项目 core-resource admission，不声称是实测总显存。

这些实验确认三种消费边界必须分开：sky/surface appearance 只消费 terminal
direction/branch/time，volume 需要有序 path checkpoints，科学验证需要 f32 records。不能为了让
analytic terminal solver 看似通用而伪造 volume path，也不能为了测试 capture 让 presentation 常驻全部记录。

## 7. 决策摘要

| 决定 | 候选 | 依据 |
| --- | --- | --- |
| 已采用 | production/capture 资源拆分、真正的 8×8 tile、exact KS RHS/Sigma 约化、6D phase、Cartesian Carter、Hermite event、committed-only guard、shadow coverage | 独立代数/observable gate；局部优化不扩大 public seam |
| 已撤出 | global transfer map、interval Kerr/Kerr–Newman capture | 历史 branch/direction 与 A/B 通过，但没有可信 travel time |
| 已拒绝 | numerical fixed-step Mino、16×4/16×16/hierarchical layouts、interval microvariants、metadata/time disable、fixed-step BS3 probe | terminal-phase 反例、收益不可复现或完整 candidate 回归 |
| 仅在新证据下重开 | embedded BS3、elliptic terminal solver、wavefront/compaction、subgroup queue、source reconstruction | 分别需要 observable error model、high-precision topology/terminal oracle、long-tail telemetry、queue bottleneck 或真实 source consumer |

路线排序只由[能力路线](../roadmap.md)维护。本账本保留候选的正反证据和重开条件，不再给已删除
variants 编号或维护调参 backlog。

### 明确延后或否决

- 无界、每 ray 连续 step-size 的 DP5(4) megakernel：在本 shader 中先验 stage cost与 divergence 不利，除非 E2 数据推翻成本模型；
- 把四个 RK stage 分 kernel：必然增加每 stage global state traffic；
- 全帧 persistent dispatch：不符合 batch responsiveness，且 portable WGSL 不提供 CUDA warp 调度保证；
- symplectic 直接替换：当前短终止轨迹没有证明其长期结构优势能转成 observable/time 收益；
- neural trajectory surrogate 直接给物理结果：没有 near-critical branch 的可证明 worst-case bound；最多做 conservative classifier/hint，错误或低置信必须回到已验证 solver；
- 直接把 AART/Krang 当近场通用实现：两者的一手来源都限定了更窄的问题域。

## 8. 准入与验证边界

所有 accelerator 沿用[验证合同](../validation.md#5-验收预算)的完整 observable budgets 和
[渲染设计](../rendering.md)的 support/fallback 原则，不能另设宽松 fast mode。Candidate-specific
evidence 还必须覆盖：

- exact/CAS convention、root/topology 与 state reduction；
- independent high-precision terminal、phase、turning 与 near-degenerate rejection；
- accepted-domain classifier 零 false acceptance，fallback 原因 machine-readable；
- Metal/Vulkan 各自的 real-adapter layout、floating-point、observable 与 total-workload evidence。

Metal/Vulkan 不要求 bit identity，但每个 backend 都必须满足同一 observable contract；shader compile
或 Naga validation 不能替代 runtime evidence。WGSL 允许的 rounding、FTZ、reassociation 与 fusion
见 [floating-point evaluation](https://www.w3.org/TR/WGSL/#floating-point-evaluation)。性能 artifact
统一遵循 [GPU benchmark 方法](gpu-benchmark-methodology.md)，本页不维护第二份测量合同。

当前 production 以完整 Cartesian Kerr–Schild 收束，并保留 8×8 layout、exact algebraic reduction
与 selective shadow coverage。Map、interval 和 numerical-Mino variants 已撤出；elliptic、wavefront、
subgroup 与 reconstruction 只有满足各自重开条件后才能形成新 candidate。它们必须是可失败、可回退、
可测量的 accelerator，不能另建无法与现有物理合同对齐的快图模式。

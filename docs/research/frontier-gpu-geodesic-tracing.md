# 前沿 GPU 测地线追迹研究：可验证的 Kerr/Kerr–Newman 加速路线

状态：研究决策记录，不代表已实现能力。研究基线为仓库提交
`9dbdb71c0ad325c7c78ca518cbdff528daa2f2fe`，目标是找到能显著降低完整原生分辨率画面延迟、同时保留物理与数值合同的算法路线。本文延续[完整帧原子发布研究](atomic-progressive-trace.md)：计算可以分批，画面仍只发布完整 generation；不重新引入可见低分辨率阶段或扫描式 reveal。

## 结论

1. 当前默认 1280×720 trace 的累计 GPU 时间样本为 `80.5–192.2 ms`，离 33.3 ms 交互目标约 `2.4–5.8×`、离 16.7 ms 约 `4.8–11.5×`。只调 batch、workgroup 或提交方式不足以填平这个数量级；主线必须减少每条光线的昂贵 `geometry_at`/RHS 求值。[当前证据](../interactive-trace.md#适用域与未外推项) [性能合同](../architecture.md#11-内存与性能预算)
2. 最值得先做的算法实验是一个**封闭的 exterior Kerr 快路径**：先比较二阶势形式的 Mino-time 数值积分，再比较完整椭圆函数解；任何 root topology、近临界条件数、坐标变换或误差检查不确定的像素，都只把 pixel index 写入 GPU 队列，由现有 Cartesian Kerr–Schild RK4 从初值重算并覆盖结果。这样保留经过验证的基线，不需要为所有像素持久化大 ODE 状态。
3. 在现有 Kerr–Schild 路径内，Bogacki–Shampine 3(2) FSAL pair 值得比完整 Dormand–Prince 5(4) 更早试验：首次 4 次、此后每 accepted step 3 次新 RHS，并自带 embedded estimate；但它是三阶方法，是否真的减少总 RHS 只能由 observable gate 和 GPU A/B 决定。[Bogacki–Shampine 原文](https://doi.org/10.1016/0893-9659%2889%2990079-7) [Dormand–Prince 原文](https://doi.org/10.1016/0771-050X%2880%2990013-3)
4. active-ray wavefront/compaction 不是第一步。它可能回收长尾 inactive lanes，但每轮必须把状态写回显存；在得到 step 分位数和 active-ratio 曲线以前，不能证明收益大于 state traffic 与多 pass 开销。若进入实验，应把完整 RK step 融在单 kernel 内，只在多个 step 的 chunk 边界 compact；不要把四个 RK stage 拆成四个 kernel。[Laine–Karras–Aila 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf) [PBRT v4 wavefront 取舍](https://www.pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Mapping_Path_Tracing_to_the_GPU)
5. 截图中的“脏”主要不能先归因于测地线噪声。研究基线的 `analytic_sky` 把经纬周期中央约 90–92% 设为高亮，两项取 `max` 后约 99.2% 天空被额外加上 `2.0` radiance。初次反转 mask 后，实际截图仍显示薄线在临界曲线处被放大为发白的环；当前实现因此进一步换成无 seam、不过曝的低阶球面多项式与局部轴向色标。每像素一个中心样本仍没有 branch-aware coverage，因此真正的 horizon/escape 轮廓抗锯齿是后续 appearance/filtering 问题，不能靠放宽积分误差或模糊几何掩盖。[shader](../../crates/gravlume-render/src/shaders/trace.wgsl)

推荐顺序是：**补齐 GPU aggregate 统计 → 真正的 2D coherence A/B → 建立 outgoing-KS/BL 零步变换恒等式 → 再重启 exterior Kerr Mino 快路径 → 解析 Kerr bake-off → 有证据后才做 wavefront/subgroup → branch-aware footprint 与重建**。production/capture ABI 瘦身已经完成；BS3(2) fixed-step 探针与本轮 Mino 原型均因量化 gate 未通过而没有合入。

## 1. 锁定环境与当前成本模型

[`Cargo.lock`](../../Cargo.lock) 锁定 `wgpu/wgpu-core/wgpu-hal/wgpu-types/Naga 30.0.0`、`bytemuck 1.25.2`、`glam 0.33.3`、`egui/egui-wgpu 0.36.1`、`pollster 1.0.1` 与 `winit 0.30.13`。macOS 只启用 wgpu Metal，Windows/Linux 只启用 Vulkan；当前 device 只请求 `TIMESTAMP_QUERY`。[workspace manifest](../../Cargo.toml) [render manifest](../../crates/gravlume-render/Cargo.toml) [capability baseline](../../crates/gravlume-render/src/capabilities.rs#L1-L14)

研究基线的 production shader 是一 invocation 一 pixel 的 megakernel：8×8 workgroup、最多 2048 个 radius-scaled fixed RK4 step，普通 accepted step 复用 exact endpoint 后仍有 4 次 geometry/RHS 求值；当时最终写 `rgba16float` HDR 与三个 16 B/pixel record plane。[shader state/bindings](../../crates/gravlume-render/src/shaders/trace.wgsl#L1-L87) [RK4](../../crates/gravlume-render/src/shaders/trace.wgsl#L405-L432) [main loop](../../crates/gravlume-render/src/shaders/trace.wgsl#L636-L900)

默认场景的开发数据从平均/最坏 `882/2048` step 降到了 `61/132`，说明 outgoing Kerr–Schild chart、几何 step 与 endpoint reuse 已经拿掉了最明显的冗余；剩余时间更接近“每步数学本身 × 全屏像素数”，而不是发布调度问题。[实现证据](../interactive-trace.md#适用域与未外推项) 坐标选择仍不可随意回退：backward ray 的方向与 horizon-penetrating chart 会改变数值适定性，相关分析见 Bozzola、Chan 与 Paschalidis 的[原论文](https://arxiv.org/abs/2310.02321)。

### 1.1 生产路径有一个独立的显存根因

Rust `TraceTarget` 的三个 buffer handle 字段只在测试 build 可见，但 bind group 与 WGSL bindings 在 production 仍无条件创建、持有和写入，所以 `#[cfg(test)]` 没有消除资源。[target allocation](../../crates/gravlume-render/src/trace.rs#L329-L371) [production stores](../../crates/gravlume-render/src/shaders/trace.wgsl#L607-L632)

研究基线的 frame accounting 是：record planes `48 B` + candidate HDR `8 B` + published HDR `8 B` + UI `4 B` = `68 B/pixel`。因而 2560×1440 是 `250,675,200` bytes（约 239.1 MiB），两阶段 rebuild 约 478.1 MiB。若 production presentation 不分配 full diagnostic planes，steady candidate+published+UI 核心变为 `20 B/pixel`，同一 extent 是 `73,728,000` bytes（约 70.3 MiB），下降 70.6%；当前实现又以 texture-view promotion 消除了 candidate 与同尺寸 published copy 在新 generation 内的重复。

实现结果：production ABI 现在只有 trace uniform、HDR storage texture 与 dispatch uniform，三个 record plane 由 test-only diagnostic shader/target 注入。新 generation 只分配 candidate HDR 与 UI，即 `12 B/pixel`；上一张完整 scene 独立持有，完成后直接提升 candidate texture view，不再分配同尺寸 published copy。4K 新 generation 为 `99,532,800` bytes（约 94.9 MiB）；同尺寸 transactional rebuild 的 conservative core plan 为 `32 B/pixel = 265,420,800` bytes（253.125 MiB），仍通过 256 MiB admission。这关闭了截图中的 host budget rejection，同时没有把 surface drawable、driver heap 或 alignment 伪装成已测总显存。

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

### 2.1 为什么先做数值 Mino，而不是直接移植完整解析代码

Kerr 的 Hamilton–Jacobi 方程可分离。[Mino 的原始工作](https://arxiv.org/abs/gr-qc/0302075)引入的重参数化把 radial 与 polar motion 解耦；Gralla–Lupsasca 系统分类 radial/angular potential roots，并给出 manifestly real 的 Legendre elliptic integral 与 Jacobi-function 参数曲线。[Gralla–Lupsasca v3](https://arxiv.org/abs/1910.12881) 的 v3 明确修正了发表后发现的错误，且论文范围是非极端 Kerr exterior，因此实现不能从旧版公式抄写，也不能把 extremal/inside-horizon 外推成已证明域。

Kerr–Newman 也有 Mino-time radial/angular potentials 与显式 elliptic/Jacobi 解，Wang、Lee 与 Lin 给出了 root 分类和可指定初值的 manifestly real 解。[Kerr–Newman 原论文](https://arxiv.org/abs/2208.11906) 这证明“解析/半解析路线存在”，不证明一份 binary32 WGSL 实现对当前近场 observer、outgoing Kerr–Schild event 和所有 parameter state 都正确。

完整解析实现每个像素需要：从近场 tetrad 初值构造 `E,Lz,Q`、分类 quartic roots 与 turning branches、求 elliptic functions、累计 azimuth/time integral、再转换回项目的 outgoing Cartesian Kerr–Schild observable。WGSL 标准 built-ins 不提供 elliptic integrals/Jacobi functions，必须自行实现并验证。[WGSL built-in functions](https://www.w3.org/TR/WGSL/#builtin-functions) 已有 Kerr analytic 实现也说明了边界：Krang 面向 GPU/differentiability，但第一方 README 明确把 observer 限制在 asymptotic infinity；这不是当前 `r_obs=30M` 的 drop-in solver。[Krang source/README](https://github.com/dominic-chang/Krang.jl) Dexter–Agol 的 semi-analytic 方法以 Carlson elliptic integrals化简 Kerr photon orbit，是很好的 CPU/oracle 对照，但其存在同样不等于 WGSL/f32 合同已经成立。[Dexter–Agol 2009](https://arxiv.org/abs/0903.0620)

因此首个 accelerator 应沿用仓库已有的[二阶势候选](../rendering.md#4-exterior-mino-time-candidate)：用 `(u=1/r, u', μ=cosθ, μ')` 的 polynomial RHS 自然越过 turning point，数值积分 `φ,t`，避免每个 RK stage 重建 Cartesian Kerr–Schild radius、principal null vector 及其三个空间导数。它仍是 ODE，但 RHS 规模远小于当前 `Geometry`，可先回答“可分离性在真实 GPU 上值多少”。

### 2.2 必须封闭的域

第一版 Mino variant 只接受同时满足下列条件的像素：

- pure Kerr、明确 subextremal、observer 与当前段都在 exterior；
- 从 validated tetrad/KS covector 得到的 `E,Lz,Q`、radial/polar sign 与 potential constraints 全部有限且落在安全区间；
- root/turning topology 不接近具名退化阈值，reciprocal state 约束在每个 checkpoint 内满足预算；
- horizon/escape event、source direction 与 travel time 可从同一 localized state/积分提交；
- Boyer–Lindquist/Mino 到 outgoing Kerr–Schild 的 `t,φ` 变换在 horizon bracket 前保持受控。

任何条件不满足都进入 KS queue，不允许让快路径猜 branch。Gralla–Lupsasca 的 `t,φ` integrals 含 `Δ` 分母，而当前 outgoing Kerr–Schild baseline 正是为了 backward horizon crossing 的正则性；因此 Mino 路径可以从 exterior 一侧定位 horizon event，却不能越过 BL coordinate singularity 后继续假装和 KS state 等价。[Kerr formulas](https://arxiv.org/pdf/1910.12881) [坐标分析](https://arxiv.org/abs/2310.02321)

可分离 solver 把 `E,Lz,Q` 当常量，因此“它们没有被更新”不能冒充数值正确性。accepted terminal 必须重建 outgoing KS position/covector，再用独立的现有公式复算 null/`E,Lz,Q` agreement；同时记录 `v_r²-R`、`v_μ²-U`、root separation/condition bucket。任何一项超过安全预算就回退，不能把构造出来的零 drift 当作 confidence。

本轮数值 Mino 原型因此判定为 **no-go**，且未进入 WGSL：240 个稀疏默认场景样本在保守步长下 termination 全部一致，travel-time 最大误差约 `2.25e-4 M`，但 outgoing Kerr–Schild 与 Boyer–Lindquist 的 azimuth/spin convention 尚未闭合，最坏 escape direction 误差约 `0.194 rad`，远超 `3.82e-4 rad` 合同；near-axis 样本还出现 `μ` 越界。更快步长约有 9/240 个失败，保守步长又升至约 240 steps，无法证明 GPU 收益。下一次实现前必须先以 f64/f32 零步 identity 证明六维 Mino/BL 初态能重建项目四维 outgoing-KS 导数，再重跑 branch/angle/GPU timestamp gate；在这之前移植论文公式会制造错误但“确定”的画面。

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

- **完整 Kerr elliptic solver**：常数级每 ray 工作最可能带来数量级改善，值得在数值 Mino 映射、坐标与 branch fixtures 稳定后实现。先只做 terminal sky/surface map，不把它包装成能提供 Phase 3 volume checkpoints 的通用 path API。
- **Kerr–Newman analytic**：论文给出了 exterior 解，但 charge、extreme/superextreme 分类和更多 radial topology 扩大验证面；保留通用 KS baseline，等 pure-Kerr variant 有明确收益后再扩展。[Wang–Lee–Lin 2022](https://arxiv.org/abs/2208.11906)
- **AART 非均匀网格**：它利用 Kerr integrability 和为 photon-ring lensing 定制的 adaptive grid，论文应用是 equatorial source 与长基线 visibility；适合高分辨率 photon-ring 科学产品，不是当前任意近场相机/全天空交互画面的通用替代。[AART 原论文](https://arxiv.org/abs/2211.07469)
- **Schwarzschild LUT/beam map**：Bruneton 用预计算表实现 non-rotating black hole 的 constant-time-per-pixel beam tracing 与过滤；这是 `a=0,q=0` sealed specialization 的强候选，但不能帮助默认 `a=0.8`。[Bruneton 2020](https://arxiv.org/abs/2010.08735)
- **固定场景 transfer map**：当 spacetime/observer 不变而只改变 sky、surface appearance 时，缓存 direction/branch/travel-time map 可完全跳过 geodesic；相机、FOV、extent lattice 或 spacetime 改变时必须以 generation key 失效。它是产品层复用，不是伪装成通用 solver 的 LUT。

### 2.5 当前演示与未来交互的适配

| workload | 最合适路线 | 不应外推 |
|---|---|---|
| 当前固定 `a=0.8`、近场 observer、analytic sky | pure-Kerr Mino/elliptic terminal fast path；完整后缓存 terminal map | AART equatorial grid、Schwarzschild LUT |
| 只改 sky/appearance/HDR | 复用 terminal map，重跑 appearance pass | 重新追完整 geodesic |
| 未来移动 camera/FOV/spin | 每 generation 重跑 per-ray Mino/analytic + KS fallback | 固定 observer LUT/旧 transfer map |
| 未来 Kerr–Newman/超极端参数 | 通用 KS baseline；analytic KN 独立受限 variant | 把 pure-Kerr root classifier 扩大命名即复用 |
| 未来 surface | terminal map 加 source anchor/branch/frequency | 只缓存 tone-mapped RGB |
| 未来 volume/slow-light | `PathSampler`/checkpoints 与 retarded time | 只有 terminal elliptic result 却声称支持 path transport |

因此当前“scene 不变时不重复追迹”应与“参数变化后快速得到新完整 generation”分开计时；前者是 cache hit，后者才检验 solver 是否达到 interactive。

## 3. Embedded/adaptive RK：先比较成本模型，再谈高阶

当前普通 accepted RK4 step 是 4 次 exact geometry/RHS evaluation。以默认平均 61 step 粗算，每 ray 约 244 次 evaluation，不含 terminal extra work。

Bogacki–Shampine 3(2) 是四-stage、FSAL 的 embedded pair：第一次 accepted step 4 次 evaluation，之后理想情况下 3 次新 evaluation，并以同一 stages 给出二阶误差估计。[原始论文](https://doi.org/10.1016/0893-9659%2889%2990079-7) 若没有 reject，它即使平均 step 数增加到约 81 才与 `4×61` evaluation 持平；因此它比“直接搬 CPU reference”更值得先试。不过其主解只有三阶，event/escape angle 是否能在较少总 evaluation 下满足门槛完全未知。

Dormand–Prince 5(4) 首步 7 stages、FSAL 后每 accepted step 6 次新 evaluation。[原始论文](https://doi.org/10.1016/0771-050X%2880%2990013-3) 它要把平均 accepted steps 降到约 40 以下才仅在 evaluation count 上追平当前 61-step RK4，还未计 error norm、reject/retry、更多 live stage state 和 SIMT 分歧。CPU reference 选择 DP5(4) 是为了独立、高精度 oracle，不构成 GPU interactive 的默认选择。[reference contract](../reference-implementation.md)

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

### 5.1 先测真正的像素布局

虽然 shader 声明 `@workgroup_size(8,8,1)`，production dispatch 把 pixel range 映射到 workgroup y 维，shader 再以 `global_id.y * 8 + global_id.x` 线性化；一个 workgroup 实际覆盖 64 个连续 linear pixels，通常是 64×1 屏幕条带，而不是 8×8 屏幕 tile。[dispatch](../../crates/gravlume-render/src/trace.rs) [shader index](../../crates/gravlume-render/src/shaders/trace.wgsl)

黑洞 shadow/critical curve 是二维曲线，真正的 8×8 tile 可能比 64×1 条带有更相近的 step/branch；也可能因地址计算或 backend occupancy 变慢。第一项低风险 A/B 应比较：

- 64-thread linear strip；
- 8×8 screen tile；
- 16×8/128-thread tile（只在 limits 允许时）；
- 可选 tile 内 Morton order，但只在 step heatmap 证明二维 locality 有价值后测试。

每个 variant 必须产生逐 pixel 相同 observable；Metal 与 Vulkan 分别测，不能从 Apple 的结果固定跨平台 workgroup。
Metal 原生 pipeline 明确把 [`threadExecutionWidth`](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth) 与 [`maxTotalThreadsPerThreadgroup`](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/maxtotalthreadsperthreadgroup) 作为 pipeline/device-dependent 属性；在 wgpu 抽象层只能依据 adapter limits/subgroup properties 与 timestamp A/B，不能硬编码一个“Apple SIMD 宽度”。

### 5.2 为什么不直接把 RK4 拆成 wavefront

wavefront 的确定收益来自让每个 kernel 以更一致的工作开始并隔离 register-heavy 阶段；确定代价是 kernel 间所有 live state 都要读写 global memory。PBRT 的实现因此会融合部分阶段减少 memory traffic，而不是把算法机械地拆到最细。[PBRT v4](https://www.pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Mapping_Path_Tracing_to_the_GPU) Laine、Karras 与 Aila 也同时量化了 divergence/register 优势和每 path 212-byte global state/queue 成本，说明“wavefront”不是免费开关。[原论文](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)

Gravlume 若每 `K` 个完整 RK step 暂停，需要持久化至少 position/momentum、初始 invariants、maximum drift、travel time、step/flags；若还保存 endpoint RHS 来保留当前复用，状态更大。全屏 indexed state 很容易重新吃掉生产 record planes 刚释放的上百 MiB。持久化完整 `Geometry` 尤其错误：它含 radius/gradients、principal null vectors 与三组 derivatives，应该在 resume 重算或通过实测决定只缓存 RHS。[Geometry definition](../../crates/gravlume-render/src/shaders/trace.wgsl#L14-L38)

只有当 histogram 显示明显长尾（例如 p95/p50 与 late active ratio 足够高）时，才实现 `K=8/16/32` 三个 fused-step chunk 候选：

1. chunk 内保持完整 RK tableau、event 与 invariant 逻辑；
2. chunk 末只 compact unfinished indices；
3. 下一轮用 indirect dispatch，不做 CPU count readback；
4. timestamp 分开记录 integrate、compact 和 args build；
5. 总 state bytes/pixel、queue worst case 与 rebuild peak 一并纳入 gate。

如果 compaction 总时间没有超过 full-screen inactive early-return 的收益，就保留简单路径。`wgpu` 能发 indirect dispatch 只证明 API 可表达，不证明 workload 合算。

### 5.3 Persistent threads 的有限适用性

CUDA ray-traversal 研究用 persistent threads/动态 work distribution 改善硬件利用率；其结果建立在特定 CUDA warp 与硬件上。[Aila–Laine 2009](https://research.nvidia.com/publication/2009-08_understanding-efficiency-ray-traversal-gpus) 在 portable WGSL 中让每个 invocation 完整追一条 ray 后从 global atomic 再取一条，并不能保证回收同一 SIMT group 内尚未 reconverge 的 masked lanes；同时单个全帧 persistent dispatch 又违背当前明确的 batch/watchdog 时间边界。

因此不把“永久 worker pool”作为主线。更干净的 GPU work distribution 是上一节的两遍结构：快路径只产出困难像素 index，第二遍对 compact queue 做 indirect KS。它只有有限 pass 数、无 continuation state，而且对 unsupported/uncertain domain 有物理含义。

### 5.4 Subgroup 只优化 queue，不优化方程

WGSL subgroup ballot、exclusive add、elect 可以让一个 subgroup 先做局部 prefix，再以一次 global atomic 预留连续 queue 区间；它不会减少 geodesic RHS，也不会自动修复不同 ray step count。[WGSL subgroup built-ins](https://www.w3.org/TR/WGSL/#subgroup-builtin-functions)

锁定版本有必须遵守的实现边界：

- `wgpu-types 30` 的 `SUBGROUP` 覆盖 compute/fragment，`SUBGROUP_BARRIER` 另需 `SUBGROUP`；当前 queue ballot/scan 没有 subgroup memory dependency，不应顺手请求 barrier。[locked feature source](https://docs.rs/crate/wgpu-types/30.0.0/source/src/features.rs)
- subgroup width 不是 32 常量；Vulkan 的 `subgroupSize` 是 implementation-dependent power of two。[Vulkan subgroup properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceSubgroupProperties.html) `AdapterInfo::subgroup_min_size/max_size` 只能记录 capability range：锁定的 wgpu-hal 30 Metal backend 将其报告为典型 `4..64` 桶，并不暴露当前 pipeline 的 `threadExecutionWidth`，所以实际 occupancy 仍要用 Metal profiler 与 timestamp A/B 确认。[locked adapter source](https://docs.rs/crate/wgpu-types/30.0.0/source/src/adapter.rs) [locked Metal backend](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/metal/mod.rs)
- Metal backend 只在 `supports_simd_scoped_operations` 时报告 subgroup/barrier；Vulkan backend 检查所需 stages/operation bits，并在启用 subgroup 时允许 varying subgroup size。[locked Metal source](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/metal/adapter.rs) [locked Vulkan source](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/vulkan/adapter.rs)
- 标准 WGSL 已有 `enable subgroups`，但 Naga 30 的 parser 把该 directive 标记为 unimplemented；本仓库锁定 variant 必须按现有 platform contract 省略 directive、请求 device feature，并用 Naga 30 parse/validate。升级 wgpu/Naga 后重新核对，不能永久固化这个例外。[locked Naga source](https://docs.rs/crate/naga/30.0.0/source/src/front/wgsl/parse/directive/enable_extension.rs) [仓库平台合同](../platform.md#52-可选加速-variant)

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

一个可维护的接口应区分 `TerminalMap`、`PathSampler` 与 `DiagnosticCapture`：sky/surface appearance 只消费 terminal direction/branch/time，volume 需要有序 path checkpoints，科学验证需要 f32 records。不能为了让一份 analytic terminal solver 看似通用而伪造 volume path，也不能为了测试 capture 让 presentation 常驻全部记录。

## 7. 候选实验优先级

| 顺序 | 实验 | 主要假设 | 保留门槛 | 主要风险 |
|---:|---|---|---|---|
| E0 | production/capture 资源拆分（已实现）；GPU histogram（待做） | 48 B/pixel 记录是 resize/VRAM 根因，统计可替代 production readback | 1440p 新 generation 核心约 42.2 MiB；完整 diagnostic 仍通过；trace 无显著回退 | 未来语义字段被误删 |
| E1 | 64×1、8×8、16×8 workgroup/layout A/B | 二维 cost coherence 可减少 masked lanes | Metal/Vulkan 各自 p50/p95 至少 5%，结果逐 pixel 等价 | backend 特异、收益落入噪声 |
| E2 | BS3(2) FSAL + 量化 step + bounded fallback | embedded estimate 能以更少 RHS 达到相同 observable | 全流程 p50/p95 至少 20%；最终 gate 全过 | 三阶需更多 step、reject 分歧 |
| E3 | pure-Kerr exterior Mino numerical fast path + KS indirect fallback | polynomial separable RHS 能消除主要 geometry 成本 | 默认场景至少 2×；参数矩阵几何平均至少 1.5×；无 overconfident wrong branch | f32 near-critical floor、horizon/time transform |
| E4 | 完整 Kerr elliptic terminal solver + 同一 fallback | 常数级 special-function 求值优于几十步 ODE | 相对 E3 仍有实质收益，或提供 E3 做不到的稳定性；复杂度有 fixture 覆盖 | roots/special functions/近场初值复杂 |
| E5 | K-step fused wavefront + compaction | measured long tail 足以覆盖 state traffic | integrate+compact+args 总 p50/p95 至少 20%，内存仍过项目 gate | 显存、bandwidth、多 pass |
| E6 | subgroup queue build | global atomic contention 已成为可测瓶颈 | 各支持 backend 至少 5%，baseline variant 保留 | 宽度/工具链差异 |
| E7 | branch-aware source footprint/reconstruction | 当前视觉问题主要是 sampling；稳定区可少 trace | source/filter quality 与 temporal gate 通过，且总 GPU time下降 | caustic false interpolation |

新 solver（E3/E4）要求更高收益，是因为它引入新的物理 domain、公式与长期维护面；5% 的偶然提升不足以支付复杂度。阈值应在运行前固定，若实际 run-to-run noise 更大，则先扩大样本而不是下调门槛。

### 明确延后或否决

- 无界、每 ray 连续 step-size 的 DP5(4) megakernel：在本 shader 中先验 stage cost与 divergence 不利，除非 E2 数据推翻成本模型；
- 把四个 RK stage 分 kernel：必然增加每 stage global state traffic；
- 全帧 persistent dispatch：不符合 batch responsiveness，且 portable WGSL 不提供 CUDA warp 调度保证；
- symplectic 直接替换：当前短终止轨迹没有证明其长期结构优势能转成 observable/time 收益；
- neural trajectory surrogate 直接给物理结果：没有 near-critical branch 的可证明 worst-case bound；最多做 conservative classifier/hint，错误或低置信必须回到已验证 solver；
- 直接把 AART/Krang 当近场通用实现：两者的一手来源都限定了更窄的问题域。

## 8. 正确性门槛

所有 accelerator 最终输出沿用[interactive agreement](../validation.md#53-interactive-agreement)，不能另设更宽松的“fast mode”：

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

短期不要再花主要精力微调 batch。production 中未消费的 48 B/pixel capture planes 已移除；下一步先补 aggregate telemetry，再用真正 2D tile 建立干净 baseline。BS3(2) fixed-step 探针已证明简单 tableau 替换不够，Mino 原型也已在方向合同上失败；应先封闭 outgoing-KS/BL coordinate seam，再让任何受限快路径通过 GPU index queue 回到现有 KS RK4，最终仍原子发布完整原生分辨率帧。

若 Mino 快路径在默认场景达不到至少 2×，完整 elliptic WGSL 实现大概率不值得立即承担；若达到，则以同一 fixtures 和 fallback classifier 推进 analytic Kerr。只有 step histogram 证明长尾明显时才付出 wavefront state 的显存/带宽成本。ray differentials/branch-aware footprint 独立解决画面 aliasing，并为未来 image-space adaptive 提供依据，但不能代替 geodesic correctness。

这条路线把“前沿算法”约束为可失败、可回退、可测量的 accelerator，而不是另起一套无法与现有物理合同对齐的快图模式。

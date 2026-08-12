# 完整帧发布与交互式 geodesic trace 研究

状态：历史研究结论。隐藏 candidate 与原子发布已采用；完整低分辨率阶梯经实际观感评估后被否决，生产实现只发布原生分辨率完整帧。后续算法 bake-off 仍是研究候选。范围锁定为 Rust 1.97、wgpu/wgpu-core/wgpu-types/Naga 30.0.0、Metal/Vulkan 与当前 Phase 2 Cartesian Kerr–Schild WGSL tracer。

## 结论

从上到下显示尚未完成的 trace 是发布模型错误，不是可接受的 progressive rendering。计算可以分批，**可见性必须以完整 candidate generation 为单位**。当前最合适的主线是：

1. display 始终采样一个只含完整画面的 `published HDR`；
2. compute 分批写不可见的 `candidate trace target`；
3. candidate 的全部像素完成且 generation 仍有效时，等待最后一批 timestamp readback 后直接提升 candidate texture view，重绑 display 并 present；画面只发生一次完整帧切换，不再执行整图 publication pass；
4. 研究阶段曾考虑以**完整低分辨率帧**缩短反馈，但实测仍产生明显的分辨率跳变，现已淘汰；生产路径从空白/上一完整 generation 直接原子切换到原生分辨率完整帧；
5. 调度只解决 watchdog 和观感。真正的性能工作应先去除每步重复的 geometry/RHS 求值，再以固定 RK4、少量有界 embedded tier、事件局部细化和 active-ray wavefront 做可归因 A/B；不应直接把 CPU 式无界 DP5(4) accept/reject loop 搬进每条 GPU ray。

采用的原生分辨率原子发布方案符合项目对 generation、timestamp 和 stale history 的已有约束：[architecture §3](../architecture.md#3-snapshot-generation-与事务提交)、[architecture §8–11](../architecture.md#8-frame-graph-与资源生命周期)、[validation interactive gate](../validation.md#53-interactive-agreement)。

## 1. 已观察到的根因

研究开始时，`FrameResources` 只有一个 `TraceTarget`，display bind group 直接绑定该 target 的 HDR view；每个 redraw 写下一段像素后立即运行 display，因此部分写入天然可见。当前实现已按本文结论拆分 hidden candidate 与 complete published scene；历史根因保留在此供决策追溯。把线性 pixel offset 改成棋盘、Morton 或随机顺序只能改变伪影形状，不能使画面成为完整帧。

shader 当前每条 ray 最多执行 2048 个 fixed-step RK4 iteration；RK4 本身每步调用 4 次 `hamilton_rhs`：[trace.wgsl `MAXIMUM_STEPS`](../../crates/gravlume-render/src/shaders/trace.wgsl#L68)、[`rk4_step`](../../crates/gravlume-render/src/shaders/trace.wgsl#L405-L432)、[main loop](../../crates/gravlume-render/src/shaders/trace.wgsl#L633-L672)。这给出每像素最多 8192 个 RK stage RHS，仅 1280×720 就是 75.5 亿个 stage RHS 的理论 ceiling；这还没有计入 invariant 与 event 路径的额外调用。

普通非终止 iteration 的源码级成本更高：

- `start_geometry`：1 次 `geometry_at`；
- RK4：4 次 RHS，而每次 RHS 内部各做 1 次 `geometry_at`；
- `next_geometry`：1 次 `geometry_at`；
- `invariants(end)`：自身 1 次 geometry，再调用 RHS 而增加 1 次 geometry。

因此是每步 5 次 RHS、8 次 geometry 求值。终止步还重新求 start/end RHS、localized geometry/RHS/invariants：[trace.wgsl event path](../../crates/gravlume-render/src/shaders/trace.wgsl#L728-L804)。`geometry_at` 又构造 Kerr–Schild radius、metric scalar 与三个空间导数，不是廉价 getter：[trace.wgsl geometry](../../crates/gravlume-render/src/shaders/trace.wgsl#L163-L310)。在没有后端编译物和 profiler 证据前不能断言编译器会跨这些控制流消除重复计算。

所以问题分成两层：

- **发布层**：partial candidate 被 display 直接采样，造成扫描感；
- **算法层**：每像素工作量本身很大，而且 ray 的终止步数不同，造成长 dispatch 与 SIMT masked-lane 浪费。

前者必须无条件修正；后者必须测量后逐项优化。

## 2. wgpu/WebGPU 能否做“隐藏计算、完整后发布”

### 2.1 Queue 顺序足以提供发布栅栏

wgpu-hal 30 明确规定：同一 `Queue` 上，前一次 submission 的全部命令在后一次开始前完成且结果可见；同一 submission 中 command buffer 也按输入顺序执行并相互可见：[locked wgpu-hal queue contract](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/lib.rs)。WebGPU 也把 draw/copy/compute 放在单一 queue timeline 上，并由 `submit` 安排执行：[WebGPU timelines](https://www.w3.org/TR/webgpu/#programming-model-timelines)、[`GPUQueue.submit`](https://www.w3.org/TR/webgpu/#dom-gpuqueue-submit)、[synchronization](https://www.w3.org/TR/webgpu/#programming-model-synchronization)。

wgpu-core 为 submission 分配严格递增 index、提交后登记 lifetime tracker，并依据资源 usage 插入 transition：[locked wgpu-core queue source](https://docs.rs/crate/wgpu-core/30.0.0/source/src/device/queue.rs)。

由此可以安全编码：

```text
compute final candidate batch
  -> validate generation after GPU completion
  -> promote candidate texture view
  -> display the promoted complete scene
  -> present
```

它们可以在一个 command buffer 中，也可以按该顺序提交到同一 queue。无需 `Device::poll(Wait)`，也无需 CPU 等待 callback 才让 display 看到结果。GPU 实现可以在不改变可观察结果的前提下重排物理执行；timestamp 也可能观察到重叠，因此正确性依据是 queue/usage 的结果顺序，而不是“硬件绝不并行”这一假设：[locked wgpu timestamp semantics](https://docs.rs/crate/wgpu-types/30.0.0/source/src/lib.rs)。

隐藏 candidate 内部仍可按任意 tile 顺序计算。WGSL 的 `global_invocation_id` 给出当前 compute grid invocation 位置，`textureStore` 只写指定 texel；加入 checked tile/pixel offset 即可使用 Morton、棋盘或 cost-aware scheduling，而不会改变 publish 语义：[WGSL builtin values](https://www.w3.org/TR/WGSL/#builtin-values)、[WGSL `textureStore`](https://www.w3.org/TR/WGSL/#textureStore-builtin)。locked Naga 30 把该 builtin 解析为 `GlobalInvocationId`，并把 store lowering 为带坐标的 `ImageStore`：[Naga WGSL conversion](https://docs.rs/crate/naga/30.0.0/source/src/front/wgsl/parse/conv.rs)、[Naga image store lowering](https://docs.rs/crate/naga/30.0.0/source/src/front/wgsl/lower/mod.rs)。这里应称 `candidate/back texture`；wgpu 语境中的 staging 通常指 upload/readback buffer，容易误导资源用途。

### 2.2 实现采用 texture-view promotion，不复制完整画面

研究早期建议过 copy-to-published；实现复核后采用更省内存和带宽的 texture-view promotion。锁定 wgpu 30 的 `TextureView` 可克隆并拥有 parent texture；candidate 完成且 generation 匹配时，把其 view 提升为 published scene，再安装预创建的 scene/UI bind group。partial candidate 从未绑定到 presentation，提升也不复制 texel。[locked `TextureView`](https://docs.rs/crate/wgpu/30.0.0/source/src/api/texture_view.rs) [locked `BindGroup`](https://docs.rs/crate/wgpu/30.0.0/source/src/api/bind_group.rs)

这也使当前 generation 不再同时拥有 candidate 与同尺寸 published copy。研究基线与实现后的账本为：

| 资源模型 | 字节/pixel | 2560×1440 核心资源 |
|---|---:|---:|
| 研究基线 candidate(records+HDR)+UI | 60 | 约 211 MiB |
| 已实现 candidate HDR+UI | 12 | 约 42 MiB |
| 新 candidate+UI 与同尺寸上一 complete scene 共存 | 20 | 约 70 MiB |

实现模型低于 architecture 的核心预算；resize 时保留上一 complete scene，并让新 generation 只分配 candidate+UI。若连续 resize 使旧 candidate 仍在 flight，wgpu lifetime tracker 仍会延迟释放底层资源，因此实际 peak 仍须测量，且旧 generation 绝不 publish 为新 generation。[architecture memory budget](../architecture.md#11-内存与性能预算)

`BindGroup` 没有修改既有 entry 的 API，创建时绑定具体 view；每个 view 又持有其 parent texture clone：[wgpu `BindGroup`](https://docs.rs/wgpu/30.0.0/wgpu/struct.BindGroup.html)、[locked `Texture::create_view`](https://docs.rs/crate/wgpu/30.0.0/source/src/api/texture.rs)、[locked `TextureView`](https://docs.rs/crate/wgpu/30.0.0/source/src/api/texture_view.rs)。因此 resize 时预建“旧 scene + 新 UI”和“candidate + 新 UI”两份 bind group；完成点只交换已验证 handle，不尝试改写 bind group。

提交中的 command tracker 持有被引用资源，直到相应 submission 完成才释放：[locked lifetime tracker](https://docs.rs/crate/wgpu-core/30.0.0/source/src/device/life.rs)。generation 失效后可以 drop Rust handle，让 tracker 延迟回收；不要对仍可能 in-flight 的 texture 主动调用 `destroy`。wgpu 的 texture 文档也区分“handle 可 drop”和“底层资源在 GPU 使用时不可销毁”：[locked texture safety](https://docs.rs/crate/wgpu/30.0.0/source/src/api/texture.rs)。

### 2.3 Timestamp 与 backpressure

当前项目用 compute pass begin/end timestamps、resolve buffer、readback buffer 和非阻塞 `Device::poll(Poll)`，方向正确：[timing.rs](../../crates/gravlume-render/src/timing.rs#L69-L179)。`TIMESTAMP_QUERY` 支持 pass timestamp writes，结果需乘 `Queue::get_timestamp_period`：[wgpu compute pass timestamps](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePassDescriptor.html#structfield.timestamp_writes)、[wgpu timestamp period](https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html#method.get_timestamp_period)。Apple GPU 不应依赖 inside-encoder/inside-pass 手写 timestamp 的额外 feature；locked feature table 只给 Apple GPU 保证 baseline pass descriptor 路径：[locked timestamp features](https://docs.rs/crate/wgpu-types/30.0.0/source/src/features.rs)。

可移植 readback 必须保持 `QUERY_RESOLVE | COPY_SRC` buffer 与 `MAP_READ | COPY_DST` buffer 分离，因为默认 WebGPU map usage 组合受限：[locked buffer usages](https://docs.rs/crate/wgpu-types/30.0.0/source/src/buffer.rs)。`map_async` 只会在 GPU 不再使用 buffer 后完成，buffer mapped 时 GPU 不能使用它；callback 需要 submit 或 poll 推进：[wgpu `map_async`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Buffer.html#method.map_async)、[locked mapping contract](https://docs.rs/crate/wgpu/30.0.0/source/src/api/buffer.rs)。

当前单 readback slot 把“下一 batch 能否提交”绑在 map 完成上，天然提供强 backpressure，但也把 throughput 与 telemetry latency 耦合。后续若允许多个 candidate batch in flight，应使用 2–3 slot timestamp ring；ring 满时丢 telemetry sample或停止追加 compute，不能 `PollType::Wait` 阻塞 UI。`Queue::on_submitted_work_done` 是此前全部 queue work 的粗粒度栅栏，适合 shutdown/drain，不适合每 batch 交互节拍：[wgpu queue callback](https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html#method.on_submitted_work_done)、[locked callback contract](https://docs.rs/crate/wgpu/30.0.0/source/src/api/queue.rs)。

GPU timestamp 只测查询点之间的 queue time，不能代替交互延迟。还应以 CPU `Instant` 记录 `generation invalidated -> first complete publish submitted` 与 `-> final complete publish submitted`。这与项目现有证据政策一致：[architecture diagnostics](../architecture.md#10-diagnostics可复现性能与错误)、[rendering dynamic resolution](../rendering.md#84-dynamic-resolution)。

## 3. 三种完整帧策略比较

### A. 隐藏全分辨率 candidate，完成后原子 publish

**语义**：所有 batch 只写 candidate；display 重复显示 last-complete published frame；最后一个 batch 完成并通过 generation gate 后直接提升 candidate texture view，再显示新 published frame。

**优点**：

- 不改变 sample lattice、数值结果或最终分辨率；
- 每个 dispatch 可控制在 watchdog/帧预算内；
- published texture 不含 record planes，内存比双完整 target 小；
- generation token 能明确拒绝 resize/cut/scene change 后的旧 candidate。

**缺点**：

- 第一次画面和大改场景仍要等完整 full-resolution trace；只修复扫描感，不降低总计算量；
- 旧 front 与当前 UI 参数可能短暂不一致，必须显示明确的“正在计算 generation N”，不能把它当新 snapshot；
- transactional rebuild 同时持有 published、installed candidate/UI 与 replacement candidate/UI；必须按 `32 B/pixel` worst live core plan admission，不能只报新 generation 的 `12 B/pixel`。

**判断**：已采用。它保证视觉正确性；交互延迟必须靠追迹算法本身降低，不能以可见的分辨率退化掩盖。

### B. 完整低分辨率帧后逐级提升（已否决）

**语义**：使用有限 resolution ladder，例如 `1/4 linear -> 1/2 -> native`；每一级都完整 trace 到隐藏 candidate，完成后才 publish。下一等级开始前，屏幕持续显示上一完整等级。

**优点**：

- time-to-first-complete-frame 随 pixel count 下降；不会出现扫描半成品；
- resolution ladder 离散，便于 timestamp hysteresis、内存 admission 和跨 Metal/Vulkan 复现；
- 与仓库允许 internal extent 由 Quality Policy 控制的方向一致：[rendering dynamic resolution](../rendering.md#84-dynamic-resolution)。

**缺点**：

- 需要 display/reconstruction 支持 internal extent 与 presentation extent 不同；当前 `display.wgsl` 直接用 surface pixel 做 `textureLoad`，假定尺寸相同：[display.wgsl](../../crates/gravlume-render/src/shaders/display.wgsl#L48-L68)；
- 简单双线性颜色上采样会跨 horizon/escape、branch 与 cubemap seam，违反“先保存几何语义”的合同。最低限度应按 termination/source direction discontinuity 做 edge-aware reconstruction，不同 semantic key 不插值：[rendering source footprint](../rendering.md#6-source-footprint-jacobian-与-differential-ray)、[history key](../rendering.md#82-history-key)；
- 分辨率变化改变 sample spacing，必须进入 resolved trace/history key；不能只把低分纹理拉伸后声称与 native 等价。

**判断**：实测分辨率跳变仍明显，且会引入 reconstruction 与 sample-lattice 语义；不进入生产路径。本节保留为被否决方案的决策记录。

### C. 交错/空间填充，并在每 pass 重建完整画面

**语义**：每 pass 计算蓝噪声、棋盘或空间填充曲线选择的 samples，再从当前完整 sample set 重建一张新的隐藏 full-frame candidate，完成重建后 publish；原始未计算 texel本身永不直接显示。

**优点**：

- 空间覆盖比线性 scan 均匀；每次 publish 仍是完整画面；
- 若稳定区域可从 source-space footprint 可靠重建，可能比纯 resolution ladder 更早恢复细节。

**缺点**：

- “交错”本身没有信息补全能力；没有可靠 reconstruction 时只是把扫描伪影变成噪声/棋盘；
- 每 pass 的 full-frame reconstruction、sample mask 与 semantic key 增加 bandwidth 和内存；
- critical curve、多像、branch split 处的一阶重建失效，必须真实 refine：[rendering failure boundary](../rendering.md#64-failure-boundary)；
- 当前 Phase 2 尚无 Jacobian/branch/parity/source anchor 全套字段，过早实现会把 Phase 4 复杂度拉进基础 tracer。

**判断**：不作为根因修复。只有未来独立研究证明 field-aware reconstruction 在同一误差预算下优于原生追迹，才进入实验 variant。

## 4. 算法根因与候选优化

### 4.1 第一优先：复用 exact endpoint geometry/RHS

当前 invariant 检查已经为每个 accepted endpoint 计算了 exact `hamilton_rhs(end)`。这个 derivative 正好是下一 RK4 step 的 `k1`；虽然 classical RK4 不是 FSAL method，**这里不是复用 RK4 的 `k4`**，而是复用额外计算的 exact endpoint RHS，因此数学上仍是原 RK4。进一步把 `geometry_at` 作为值传给 `hamilton_rhs_from_geometry` 与 `invariants_from_geometry_rhs`，可让 next event、invariant、next step 共用同一 endpoint geometry/RHS。

预期源码级普通步成本可从“5 RHS + 8 geometry”降到“4 RHS algebra + 4 geometry”：3 个新的 RK stage，加 1 个 exact endpoint RHS；初始 state 也已由 initial invariant 路径提供 exact RHS。终止路径同样无需再次求 start/end RHS，只需 localized point 的 geometry/RHS/invariant。

这是保持 step policy、RK tableau、event crossing 与 observable contract 不变的低风险候选，但仍必须通过后端编译物或 GPU timestamp A/B 证明编译器没有已经消除这些求值。保留条件：固定 scene/extent 下 trace GPU p50/p95 的改善超过 run-to-run noise，且 termination、direction、travel time、四项 drift 与 event residual 全部通过现有 gate。

### 4.2 第二优先：事件只在 bracket 内局部细化

当前先按 endpoint event value 线性估计 fraction，再对 state 做一次 cubic Hermite evaluation；没有在 dense curve 上继续 bracket：[trace.wgsl localization](../../crates/gravlume-render/src/shaders/trace.wgsl#L728-L804)。可让普通步继续较大，只对已发生 sign crossing 的终止步执行固定上限的 dense bisection/secant hybrid。每次局部化只评估 dense state 与 event geometry，不重新做完整 RK step；最后只在 accepted localized state 求一次 RHS/invariants。

bracket 必须保持在 `[0,1]`，同一步多个 event 要保留 priority/tie policy，不能为速度在 bracket 外 extrapolate。CPU reference 已有 dense bracket+bisection 的可验证实现：[reference tracer localization](../../crates/gravlume-reference/src/tracer.rs#L595-L630)，其科学来源是 [Brent 1971/1973 原文](https://maths-people.anu.edu.au/~brent/pub/pub006.html)。GPU interactive 不需要复制 reference 的 `2e-11 M` tolerance；它只需满足自身 `5e-3 M` event position gate，并报告 iteration/residual。

这一项的价值是允许后续放宽远离 event 的 step，而不牺牲终止位置。只增加局部化迭代而不改变 outer step，不会显著降低总 trace time，应视为 correctness enabler，不应单独包装成性能优化。

### 4.3 有界 embedded tier，而非每 ray 无界 DP5(4)

Dormand–Prince 5(4) 每个首次 step 7 stages，FSAL 后每 accepted step 6 个新 RHS；embedded error 控制可能 reject 并重试。原始 pair 及 dense extension见 [Dormand–Prince 1980](https://doi.org/10.1016/0771-050X%2880%2990013-3) 与 [Shampine 1986](https://www.ams.org/journals/mcom/1986-46-173/S0025-5718-1986-0815836-3/S0025-5718-1986-0815836-3.pdf)。本仓库 reference 的 locked implementation也明确体现 7 stages、FSAL 后 6 evaluations、reject 不提交 state/event：[integrator.rs](../../crates/gravlume-reference/src/integrator.rs#L3-L58)、[attempt/accept loop](../../crates/gravlume-reference/src/tracer.rs#L121-L205)。

完整 DP5(4) 只有在平均 accepted step 足够大、足以抵消比复用后 RK4 更高的 stage/register/control-flow 成本时才可能获益。GPU geodesic 先例证明 adaptive RK 可行，但不证明适合本 shader/误差合同：Odyssey 使用 fifth-order adaptive RK，并按每 photon/step 报告性能；其数值状态、坐标、硬件和精度都不同：[Odyssey 原论文](https://arxiv.org/abs/1601.02063)。

更适合先测试的 GPU variant 是：

- 仅允许少量量化 step tier，避免相邻 lanes 完全不同的连续 step sequence；
- 用 embedded estimate 或 invariant trend 选择“接受 coarse / 进入 refine”，每个 dispatch 的 retry 上限固定；
- regular rays 走 lean baseline；near-critical、event-near 或 drift-growing rays进入第二 pipeline；预算仍不足则输出 `Uncertain`，不能给错误确定 branch；
- 比较 fixed RK4、低 stage embedded 3(2)、DP5(4) 三者的 GPU ms 对 field error 曲线，而不是只比较方法阶数。

仓库 rendering contract 已明确提出“固定 RK4 + 几何/量化 step 与少量有界 embedded tier”，并警告完全自适应 DP5(4) 的 accept/reject 发散：[rendering integrator policy](../rendering.md#2-state-chart-与-integrator-候选)。独立 GPU ODE 研究也把 adaptive per-system control 的 thread divergence 识别为主要性能变量：[Niemeyer & Sung 2017 原文](https://arxiv.org/abs/1611.02274)。

### 4.4 Active-ray wavefront/compaction 只在 profile 后引入

当前一个 invocation 把一条 ray 从 observer 跑到 termination；同一 workgroup 内有的 ray 很早 capture，有的 ray接近 2048 steps。SIMT 对分支路径屏蔽不参与 lanes，长尾 ray 会让其他 lanes空转。NVIDIA 的原始 wavefront 研究指出 divergence 与大 kernel register pressure 都会降低利用率，而拆成 specialized kernels 也有 state memory、queue management 和 launch 开销：[Laine–Karras–Aila 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)。GRay 则证明大量独立 null geodesics 可用 stream-processing GPU integrator并行，但其报告的每 photon/step 数字不能直接外推到 wgpu/WGSL：[GRay 原论文](https://arxiv.org/abs/1303.5057)。

对本项目，wavefront 形态是每次 dispatch 只推进固定数量 steps，把 active state 写 storage buffer；后续 pass 对 active queue 继续积分，terminal rays 写最终 record。它自然限制单 dispatch duration，也可能提高同阶段 coherence。但它会增加至少 position、momentum、travel time、drift、step/event 状态的持久带宽，并需要 queue build/compaction。

进入实现前必须先记录 steps p50/p90/p95/p99/max、termination 分布和 active ratio。若 full-screen second dispatch + early return 已足够，不能因为 wgpu 支持 indirect dispatch 就引入 atomic/prefix queue；仓库已有相同 gate：[rendering queue strategies](../rendering.md#72-queue-与-dispatch-策略)、[architecture GPU execution](../architecture.md#7-gpu-执行模型)。

## 5. 推荐实施顺序与验收

### Stage 0：恢复完整帧语义

- candidate 与 published HDR 分离；任何 incomplete candidate 永不被 display bind group 引用；
- 最后 batch 完成且 generation 仍有效时提升 texture view；一个 surface frame 中所有可见像素来自同一完整 generation；
- resize/scene invalidation 携带 generation token，旧 completion 只释放资源，不 publish；
- 初始无 front 时显示确定的 loading/neutral frame；若保留旧 front，UI 明确标出它是 last-complete generation。

必须测试的是 state-machine observable：incomplete 不 publish、complete 恰好 publish 一次、resize/cut 使旧 completion失效、in-flight 资源不被主动 destroy。不要钉死 tile 顺序或 batch 数。

### Stage 1：完整低分辨率首帧（已淘汰）

- 使用有限 resolution ladder，每一级独立完成后 publish；
- internal extent/sample spacing/quality tier 进入 generation 与 diagnostics；
- 初始可用 nearest/semantic edge-preserving reconstruction 做实验，但不得跨 termination/branch 插值；
- timestamp controller 有上下阈值、hysteresis、最小驻留帧，不能按单帧噪声来回跳档。

该 stage 未进入最终实现：观感评估否决了分辨率阶梯，生产路径保留原生分辨率原子发布，并转向消除坐标屏障、重新标定 geometric step 与复用 endpoint 求值。

### Stage 2：去除重复 geometry/RHS

- 只改变数据流复用，不同时改变 integrator；
- 对同一 baseline 进行 before/after GPU timestamp、RHS/geometry counter 与 numerical matrix；
- 若改善处于噪声内，撤回而不是保留复杂度。

### Stage 3：事件局部细化与 integrator bake-off

- 先用 fixed upper-bound dense bracket refinement 解耦 event residual 与普通 step；
- 再独立比较 fixed RK4、量化 embedded tier、DP5(4)；
- 只有在同一 observable error budget 下显著改善 error–time curve 的 variant 才保留。

### Stage 4：根据长尾证据决定 wavefront/compaction

- 若 p99/max steps 与 p50 差距大且 vendor profiler 显示 lane under-utilization，再比较 megakernel、full-screen step wave、compacted queue；
- 若 queue build/state bandwidth 抵消收益，保留较简单路径并记录被撤回实验。

## 6. 必须采集的指标

| 维度 | 指标 | Gate/用途 |
|---|---|---|
| 视觉发布 | incomplete candidate 被 present 次数 | 必须为 0 |
| generation | resize/cut/scene change 后旧 candidate publish 次数 | 必须为 0 |
| 交互 | invalidation 到 first complete publish、final native publish 的 CPU wall time | p50/p95，分开报告 |
| dispatch | 每 batch GPU ms、p50/p95/max、dispatch count | 验证长 dispatch/watchdog 预算 |
| 总成本 | trace/reconstruct/display GPU ms | 不能只把成本移到 reconstruction |
| 算法 | RHS/geometry evaluations、accepted/rejected/refine steps | 解释时间变化 |
| 发散 | steps p50/p90/p95/p99/max、active ratio、termination 分布 | 决定是否 compaction |
| 正确性 | termination、escape angle、travel time、四项 drift、event residual | 现有 validation gates 全部通过 |
| 重建 | source/termination/branch discontinuity worst-case，不只 PSNR | 防止低分辨率跨物理边界 |
| 内存 | steady、resize/invalidation peak，含 in-flight old generation | 1440p core <256 MiB，总 peak 目标 <512 MiB |

每份 artifact 固定 OS、adapter、driver、power mode、build profile、scene fingerprint、internal/presentation extent、warm-up、样本数和 p50/p95；这是项目现有 performance evidence contract：[architecture performance evidence](../architecture.md#11-内存与性能预算)、[platform release evidence](../platform.md#7-发布证据)。

实验账本应同时记录保留与撤回项，避免把多个变化混成一个无法归因的数字：

| Variant | Baseline -> result | 数值 gate | Verdict | 原因 |
|---|---|---|---|---|
| endpoint geometry/RHS reuse | 待测 | 待测 | 待定 | 单变量实验 |
| dense event refinement | 待测 | 待测 | 待定 | correctness enabler 与 total time 分开报告 |
| quantized embedded tier | 待测 | 待测 | 待定 | 与 optimized RK4 同场比较 |
| active-ray compaction | 待测 | 待测 | 待定 | 包含 queue-build/state bandwidth |

## 7. 明确淘汰的捷径

- **直接显示线性 tile、棋盘或随机 partial writes**：都不是完整帧，只改变伪影；淘汰。
- **每次 redraw 重跑全屏 tracer**：静态 generation 重复计算；淘汰。
- **用 `Device::poll(Wait)` 等 candidate 完成**：阻塞 event loop，违反 frame lifecycle；淘汰。[wgpu `Device::poll`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Device.html#method.poll)
- **双完整 TraceTarget ping-pong**：1440p 核心约 408 MiB，resize 峰值失控；除非未来 records 也必须双缓冲且有新内存预算，否则淘汰。
- **把简单颜色 bilinear/temporal upscale 当作物理正确 reconstruction**：会跨 termination/branch/critical curve；淘汰。
- **未经 profile 直接上 full DP5(4) 或 compacted indirect queue**：可能增加 stages、register pressure、发散或 queue overhead；只保留为 bake-off variant。
- **以 CPU frame time 代替 GPU trace time，或只报一次 smoke wall time**：不能归因；淘汰。

## 8. 来源说明

框架事实只采用 wgpu 30 官方 API、WebGPU/WGSL 规范和本地 locked crates 源码。gfx-rs 的 [wgpu wiki](https://github.com/gfx-rs/wgpu/wiki) 作为 first-party 导航检查过，但没有用非规范 wiki 文字覆盖具体 30.0.0 API/源码合同。数值与 GPU 执行判断只引用论文原文及仓库既有验证合同；不同论文的硬件、坐标、精度和 workload 不直接当作本项目性能承诺。

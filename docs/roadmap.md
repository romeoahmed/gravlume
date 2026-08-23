# 能力路线与退出条件

本文是尚未完成能力的依赖路线：只规定下一项工作的前置条件、交付物、退出条件与明确非目标。当前实现事实由 [Reference 证据](reference-implementation.md)和 [GPU 证据](gpu-renderer.md)维护，历史实验与被否决候选只保留在[研究记录](research/README.md)。

## 当前边界

- equatorial surface 的 Source Anchor、Frequency Ratio、vacuum/homogeneous-slab transport 与固定 observer-frame spectral bands 已由 `surface-transport-v1` fixtures 闭合；exact branch key 与 ordinary-region footprint 另有独立 CPU/GPU 证据。这些只完成当前具名的 surface physical-transport slice，不等于完整观测仪器或完整 GRRT。
- private sealed `TracePlan` 已按真实消费者选择 producer；surface plan 使用 invocation-local `GeometricSample` 和 full Kerr–Schild trace。引入这两个 seam、为 surface 禁用只回答 sky/capture 的 accelerator，均已是当前事实，不再列为后续任务。
- production renderer 直接形成 scene-linear radiance；scientific capture 负责整帧最终 radiance。Renderer 已提供固定单槽 sample inspection，桌面点击是首个真实消费者；它把实际 published texel 与 fresh full-KS record 分开，并闭合 request/cancel/poll 的 generation 与资源语义。连续字段 corpus、独立 high-precision evidence 与第二质量政策仍未闭合。
- 两个固定二像素 semantic-map 原型均未达到正确性与性能的共同门槛。当前没有可接纳的 production reconstruction、temporal reuse 或全帧 transfer map。

这些事实的细节和适用域不得复制到本文件；以实现证据和[验证合同](validation.md)为准。

## 已解析的依赖顺序

```text
有界样本审计 + 连续字段证据
              ↓
具名质量政策与支持域
              ↓
真实 source/history consumer
              ↓
branch-aware reconstruction + stationary reuse
              ↓
交互、资产、平台与发行闭环
```

Kerr 解析加速是旁路优化：它可以先建立 CPU oracle，但不能绕过上述 observable ladder，也不是 surface 产品闭环的前置条件。空间变化 volume、polarization 与 scattering 继续作为独立研究方向。

## 有界样本审计与质量基线

这是下一项 resolved work。目标不是持久化全分辨率 G-buffer，而是让用户和验证工具能够对选定像素或小区域取得与画面同代、同 profile 的结构化物理证据。

当前已完成 production 单样本 seam、桌面点击消费者、固定资源上限以及 canonical bolometric、blackbody、analytic GPU 接纳。本节仍保持开放，直到 source edge、critical/higher-order branch、正负自旋连续字段 corpus 与具名质量政策达到下列退出条件；当前 UI 不等于科学支持域已经扩大。

**交付：**

- 有界、按需的 sample inspection，至少携带 observation/generation/profile identity、producer/domain tag、实际 published `Rgba16Float` texel、fresh full-KS evaluated scene value、typed termination、在 terminal 可证明时的 exact branch key、source kind/anchor、frequency ratio、travel time 以及 event/invariant diagnostics；两类像素证据不得混同，`NumericalFailure` 与未经满足预算 refine 的 `Uncertain` 必须把 branch 显式标为 unavailable；
- 同一不可变 observation 上的 CPU/GPU 结构化比较，不经过 tone map、display encoding 或 UI 合成；
- 覆盖 ordinary surface、source edge、surface/capture boundary、不同 winding/higher-order branch、critical curve 两侧与正负 Kerr spin 的具名连续字段 corpus；
- 把 interactive execution 与 science-quality policy 分开声明。两者对 accepted physical result 使用同一 observable budget；低成本路径只能收窄支持域、refine/fallback 或显式返回 uncertainty，不能另设更宽容差。

**退出条件：**

- inspection 的资源上限、读取范围、generation 一致性、取消和错误语义明确；增加 record sink 不静默改变 solver/step policy，另一次 science-quality trace 必须使用不同 request identity；默认画面不承担隐藏的全帧诊断存储；
- discrete branch 与 continuous source anchor、$g$、travel time、radiance 分别通过独立 reference、high-precision case 或收敛证据；
- 所有正式接纳域满足[验证合同](validation.md)中的 observable budget；不满足的域被收窄、分流到更高质量路径或保持 typed uncertainty，不放宽全局阈值；
- `gpu-renderer.md` 能准确区分 test-only evidence、production inspection 能力与仍未覆盖的域，且 fixture/profile 版本没有被原地改义。

**明确非目标：** full-frame persistent G-buffer、通用调试 UI、性能尚未证明的 active queue，以及在此工作中顺带实现 reconstruction。

## 有真实消费者的重建与复用

只有在质量基线闭合，并且存在定义明确的 filterable surface source 或 stationary-history consumer 后，才重新打开 production reconstruction。被否决的固定二像素原型不是待调参实现；新候选必须从消费合同和错误边界重新设计。

**交付：**

- 一个具备版本化 sampling、chart/seam、source revision 与颜色/谱解释的真实 filterable surface source 或 stationary-history consumer；纯 test fixture 不算第二消费者；
- 以 full-resolution direct trace 或独立 supersampling 为 oracle 的 coarse/adaptive surface reconstruction；
- exact branch/status/source chart 先分区，连续 source-space footprint 再决定 LOD、anisotropy 与 refinement；
- 对不确定、near-critical、axis、near-extreme、unsupported topology 和跨 generation/history 的样本使用完整 Kerr–Schild trace；
- 若 stationary accumulation 或 source-space reprojection 有净收益，再接入有界 history；dynamic resolution 只有在它提供额外可测价值时才进入产品。

**退出条件：**

- false acceptance 为零；source edge、critical curve、caustic、多像、高频 source 与 higher-order branch 的最终 scene-linear radiance 满足具名预算；
- pixel filter 积分每个 sample 的完整 transport 结果，不默认用中心样本的 $g^4$、band fraction 或 optical depth 代表 footprint；
- resize、cut、observer/source revision、quality-policy change 与 generation change 拒绝不相容 history；
- Metal 与 Vulkan 的端到端 GPU time、峰值资源、queue/build overhead 和 publication latency 构成更优 Pareto 点；只减少 trace 次数不算成功；
- 跨 workgroup producer/consumer 通过独立 dispatch/pass 建立可见性，资源尺寸经过 adapter limits admission，完整画面仍原子发布。

## Kerr 专用解析加速

解析或半解析路径只服务 pure Kerr、已分类 root topology 与 terminal observable。优先顺序是 high-precision/CPU oracle、root classifier、terminal solver、GPU bake-off；Cartesian Kerr–Schild 始终负责定义域外和条件不确定的 fallback。

fixed-step reciprocal-Mino candidate 已被 travel-time 反例否决；恢复它需要新的 phase-error certificate，而不是更小的经验步长。Carlson/elliptic 路线也不能从“存在闭式解”推导出 `f32` 稳定性。

**退出条件：**

- chart、physical spin、root topology、turning branch 与 terminal selection 经符号和高精度验证；
- regular、near-degenerate、near-axis、near-extreme 与高绕转样本有明确 support/fallback 分类，classifier 无误接受；
- source anchor、travel time、frequency ratio 与 branch 等 observable 同时满足预算，而非只比较 potential residual；
- 包含 classification、fallback 和 publication 的完整延迟形成更优 Pareto 点；volume checkpoints 与 future transport 不被 terminal-only solver 伪装支持。

## 视觉、资产与发行闭环

视觉工作只在 Physical Scene、Appearance Model 与 Quality Policy 的边界内推进。filterable source 资产若成为 reconstruction 的真实消费者，必须先固定采样语义、色彩/谱解释、revision 与独立 oracle；它不自动升级为稳定吸积盘模型。

**交付：** 定义清楚的 sky、surface source、noise、bloom、false color 与 asset manifest，以及 Windows/Wayland/Metal/Vulkan 的具名实机矩阵。

**退出条件：**

1. **数学：** reference ladder 的适用域、精度与收敛有记录；
2. **GPU：** 目标 Metal/Vulkan adapter 无 validation error，ABI、shader 与 resource admission 可复现；
3. **图像：** machine-readable observation 与视觉回归同时通过；
4. **生命周期：** zero extent、resize、suspend、surface/device error 无 panic、死锁或跨代发布；
5. **性能：** 具名 adapter、driver、compositor 和 profile 达到预注册的延迟与显存门槛；
6. **发行：** 依赖、资产、字体、shader、配置和许可证闭环，默认场景可离线启动；
7. **Research 入口：** 有界任务、取消、资源政策和独立 artifact 不共享 live viewport 的可变状态；
8. **声明：** UI 与 export 不把研究状态、外观效果或未验证模型描述成物理预测。

## 暂不进入主线

- 空间变化 volume GRRT、scattering、Stokes/Faraday、Kerr–Newman polarization 与 slow-light production；
- 只验证 timelike existence、未验证 radial/vertical stability 和 ISCO 的“稳定薄盘”；
- 无真实第二消费者的 render graph、solver trait、通用 compatibility layer 或全帧 observation buffer；
- 直接把 Carlson functions 搬入 WGSL，或再次调节已否决的固定二像素 reconstruction；
- 只凭 CPU/GPU agreement、截图相似、内部 residual 或提前估计的性能宣称完成。

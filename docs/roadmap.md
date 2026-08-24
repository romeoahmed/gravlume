# 能力路线与退出条件

本文只定义尚未完成工作的依赖顺序、交付物、退出条件和非目标，不维护当前实现细节。当前能力以 [Reference 证据](reference-implementation.md)与 [GPU 证据](gpu-renderer.md)为准，连续模型和误差预算以[数学物理](physics.md)与[验证合同](validation.md)为准，实验历史只保留在[研究记录](research/README.md)。

## 已完成的前置

- `surface-transport-v1` 已闭合具名 equatorial source 的 Source Anchor、Frequency Ratio、vacuum/homogeneous-slab transport 与固定 observer-frame bands；ordinary-region branch/footprint 另有 CPU/GPU 证据。
- Renderer 已有固定单槽 production inspection 和 desktop consumer，并分开返回 actual published texel 与 fresh full-KS evidence；它仍是 process-local 单样本能力，不是持久 artifact 或 science-quality policy。
- 默认画面保持 invocation-local transport、完整帧原子发布和有界资源；当前没有 production semantic map、reconstruction、temporal reuse 或全帧 observation plane。

这些条目只确定路线的起点，适用域和限制不得从本页外推；见两份实现证据。

## 依赖顺序

```text
连续字段 corpus + 独立证据
              ↓
具名 interactive / science-quality policy
              ↓
可持久化 inspection artifact + Instrument consumer
              ↓
真实 filterable source 或 stationary-history consumer
              ↓
branch-aware reconstruction / reuse
              ↓
资产、平台与发行闭环
```

Kerr 解析/半解析 terminal solver 是旁路 accelerator：它可以先建立 CPU oracle，但不能绕过 observable ladder，也不是 surface 产品闭环的前置条件。空间变化 volume、polarization、scattering 与 slow-light 各自需要新的 transport 合同。

## 连续字段证据与质量政策

这是下一项已确定工作。现有 UI 证明 request、generation、资源和生命周期 seam 可用，不证明科学支持域已经扩大。

**交付：**

- 在同一不可变 observation 上，以结构化 CPU/GPU comparison 覆盖 ordinary surface、source edge、surface/capture boundary、不同 winding/higher-order branch、critical curve 两侧、正负 Kerr spin，以及具名 near-axis/near-extreme fallback 样本；比较不经过 tone map、display encoding 或 UI composition。
- 对 discrete terminal/branch 与 continuous source anchor、frequency ratio、coordinate-time duration、radiance、event/invariant diagnostics 分别给出 independent reference、80+ bit case 或 convergence evidence；一种 RGB max error 不能代表全部 observable。
- 定义独立的 interactive 与 science-quality method identity、支持域和资源政策。Accepted physical result 使用同一 observable budget；低成本路径只能收窄适用域、refine/fallback 或返回 typed uncertainty。
- 为 Instrument 的保存/比较 consumer 实现 versioned artifact identity/schema，记录 canonical observation、method、producer revision、backend/adapter 与解释 metadata；schema 与首个 consumer 一并落地，process-local ticket 不能充当内容身份。

**退出条件：**

- corpus 中所有 accepted sample 满足[验证合同](validation.md)；不满足的样本被保守拒绝或分流，且 classifier false acceptance 为零。
- 第二质量方法使用新 ticket 与不同 method identity，不在同一 request 下静默更换 solver、step policy 或 arithmetic domain。
- source edge、critical/higher-order branch 与正负 spin 至少各有能发现 branch、phase 或 radiance 回归的独立 witness；小 invariant drift 不代替这些证据。
- [GPU 证据](gpu-renderer.md)准确区分 current production capability、test-only evidence 与未覆盖域，fixture/profile/schema 没有原地改义。

**非目标：** full-frame persistent G-buffer、通用调试 UI、无实测收益的 active queue，以及在本项工作中顺带实现 reconstruction。

## 有真实消费者的重建与复用

只有质量基线闭合，并存在定义明确的 filterable surface source 或 stationary-history consumer 后，才重新打开 production reconstruction。被否决的固定二像素 prototype 不是待调参实现；新候选必须从消费合同和错误边界重新设计。

**交付：**

- 一个具备 versioned sampling、source chart/seam、revision 与颜色/谱解释的真实 source/history consumer；纯 test fixture 不算第二消费者。
- 以 full-resolution direct trace 或独立 supersampling 为 oracle 的 coarse/adaptive candidate；exact terminal/branch/source chart 先分区，continuous footprint 再决定 LOD、anisotropy 与 refinement。
- 对 uncertainty、near-critical、axis、near-extreme、unsupported topology 和不相容 generation/history 保留完整 Kerr–Schild fallback。
- 只有 stationary accumulation 或 source-space reprojection 已证明净收益时才引入有界 history；dynamic resolution 也必须提供独立可测价值。

**退出条件：**

- source edge、critical curve、caustic、多像、高频 source 与 higher-order branch 的最终 scene-linear radiance 满足具名预算，false acceptance 为零。
- pixel filter 积分每个 sample 的完整 transport result，不用中心样本的 $g^4$、band fraction 或 optical depth 代替整个 footprint。
- resize、cut、observer/source revision、quality-policy 与 generation change 拒绝不相容 history。
- Metal/Vulkan 的总 GPU time、peak resource、queue/build overhead 与 publication latency 形成更优 Pareto 点；只减少 trace 次数不算成功。
- 跨 workgroup producer/consumer 由独立 dispatch/pass 建立可见性，资源通过 adapter/device limits admission，完整画面仍原子发布。

## Kerr 专用解析加速

解析或半解析路径只服务 pure Kerr、已分类 root topology 与 terminal observable。优先顺序是 high-precision/CPU oracle、root classifier、terminal solver、GPU bake-off；Cartesian Kerr–Schild 始终定义 unsupported/uncertain domain。

Fixed-step reciprocal-Mino candidate 已被 terminal phase 反例否决；恢复它需要新的 phase-error certificate，而不是更小的经验步长。Carlson/elliptic 路线也不能从“存在闭式解”推导出 WGSL `f32` 稳定性。

**退出条件：**

- chart、physical spin、root topology、turning branch 与 terminal selection 经符号和高精度验证。
- regular、near-degenerate、near-axis、near-extreme 与高绕转样本有明确 support/fallback 分类，classifier 无误接受。
- source anchor、coordinate time、frequency ratio 与 branch 同时满足预算，而非只比较 potential residual。
- classification、fallback 和 publication 的完整延迟形成更优 Pareto 点；terminal-only solver 不伪装支持 volume checkpoints 或 future transport。

## 视觉、资产与发行闭环

视觉工作必须保持 Physical Scene、Appearance Model 与 Quality Policy 分离。Filterable source 资产若成为 reconstruction consumer，必须先固定 sampling、颜色/谱解释、revision 与独立 oracle；它不会自动升级为稳定吸积盘模型。

**交付：** 定义明确的 sky、surface source、noise、bloom、false color 与 asset manifest，以及 Windows/Wayland/Metal/Vulkan 的具名实机矩阵。

**退出条件：**

1. **数学：** reference ladder 的适用域、precision 与 convergence 有记录。
2. **GPU：** 目标 Metal/Vulkan adapter 无 validation error，ABI、shader 与 resource admission 可复现。
3. **图像：** machine-readable observation 与视觉回归同时通过。
4. **生命周期：** zero extent、resize、suspend、surface/device error 无 panic、死锁或跨代发布。
5. **性能：** 具名 adapter、driver、compositor 和 profile 达到预注册的 latency/resource 门槛。
6. **发行：** 依赖、资产、字体、shader、配置和许可证闭环，默认场景可离线启动。
7. **Research 入口：** 有界任务、取消、资源政策和独立 artifact 不共享 live viewport 的可变状态。
8. **声明：** UI/export 不把研究状态、appearance effect 或未验证模型描述成物理预测。

## 暂不进入主线

- 空间变化 volume GRRT、scattering、Stokes/Faraday、Kerr–Newman polarization 与 slow-light production。
- 只验证 timelike existence、未验证 radial/vertical stability 和 ISCO 的“稳定薄盘”。
- 无真实第二消费者的 render graph、solver trait、通用 compatibility layer 或全帧 observation buffer。
- 直接把 Carlson functions 搬入 WGSL，或再次调节已否决的固定二像素 reconstruction。
- 只凭 CPU/GPU agreement、截图相似、内部 residual 或预估性能宣称完成。

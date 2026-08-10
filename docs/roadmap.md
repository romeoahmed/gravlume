# 实施路线与退出条件

当前实施起点是 Phase 0；实际仓库状态只由根 [README](../README.md#当前状态) 维护。阶段编号表达依赖关系，不是日历承诺。任一阶段必须以数据满足退出条件；“画面看起来正确”不能替代 machine-readable observable。

## Phase 0 · 桌面栈闭环

**交付物**：Cargo workspace 基线、lockfile、winit window、compute 写入 scene-linear HDR intermediate、display pass、egui overlay、resize/zero extent、surface recovery 和结构化 device error。

**退出条件**：

- 最新 macOS/Metal，以及具名 Windows/Linux desktop Vulkan adapter/driver 原生 smoke；
- wgpu validation 无错误，奇数 extent 和边界 workgroup 无 OOB；
- acquire/submit/present 每帧协议、suspend/resume 与 surface reconfigure 测试通过；
- wgpu 30 的每个 surface acquire variant 都有处理测试；one-shot readback 在提交后立即 idle 仍能完成；
- 发行闭包不依赖未记录的外部 shader compiler、动态库或网络资产。

## Phase 1 · 领域与 CPU reference

**交付物**：validated Observation、Observer Event/Frame、Kerr–Schild `f64` DP5(4)、dense-output event localization、[验证合同](validation.md) fixture 和 comparison report。

**退出条件**：

- Minkowski、Schwarzschild、Kerr、Kerr–Newman 特殊极限与 inverse identity 通过；
- weak-field、$\sqrt{27}M$ shadow、published trajectories 和 step/tolerance convergence 通过；
- null、E、Lz、Carter、event residual 和 observer-frame 残差分开报告；
- 无 GPU 也能运行 reference tests，oracle 的适用域和误差有记录。
- baseline/strict reference policy 对 v1 regular fixtures 满足具名阈值；near-critical fixtures 与 80 位基准的离散分类一致。
- raw-radius/reciprocal-radius 数值条件性反例已固化为 machine-readable fixture，或文档继续只把它标记为待验证候选。

## Phase 2 · Interactive trace

**交付物**：WGSL `f32` Cartesian Kerr–Schild tracer、typed termination、diagnostic fields、headless readback、sky/horizon 画面。

**退出条件**：

- CPU/GPU sample matrix 的 termination、escape direction 和 continuous observables 达到预算；
- NaN、radicand/denominator failure、step exhaustion 不会静默变黑；
- WGSL binding/layout/discriminant contract tests 通过；
- Viewport Sample → Initial View Ray 的中心、四角和 jitter fixture 与 CPU 合同一致；
- 可以交互，但尚不因“能动”声称 60 FPS。

## Phase 3 · 可解释图像

**交付物**：薄表面盘、scalar emission/absorption、Frequency Ratio、blackbody/spectral LUT、HDR capture 和中性 display transform。

**退出条件**：

- `I_nu/nu^3`、frequency ratio、常系数 slab 和 optical-depth 解析回归通过；
- Physical/Appearance controls 不能互相绕过；
- RGB source 的非物理 hue shift 被明确标记，EXR 保持 scene-linear。

## Phase 4 · 重建与预算

**交付物**：coarse/classify/refine、source footprint、branch-aware spatial reconstruction、stationary accumulation、动态分辨率和 pass profiling。

**退出条件**：

- critical curve、caustic、多像薄盘和高频 sky 专项字段误差通过；
- resize/cut/generation change 100% 拒绝旧 history；
- 选定中档独显 1080p p95 不高于 16.7 ms，集显 720p p95 不高于 33.3 ms；
- 1440p 核心中间资源低于 256 MiB，总峰值初始目标低于 512 MiB。

## Phase 5 · 视觉广度与资产闭包

**交付物**：经重新设计并标注的 jet/noise/热扰动/网格/false color/bloom，以及新的 asset manifest。

**退出条件**：每项效果明确属于 Physical 或 Appearance；来源、许可、色彩空间、方向、转换链和发行权闭合；默认场景完全离线。

## Phase 6 · 测量后的加速器

**交付物**：Exterior Mino 与 Kerr–Schild bake-off，可选 Schwarzschild LUT、Kerr analytic/transfer map、active-ray compaction。

**退出条件**：只有在同一 observable error budget 下给出显著更好的误差—时间曲线，才进入 interactive resolved plan；artifact 带 domain fingerprint、branch schema、producer 和 error bound。

## Phase 7 · 研究质量

**交付物**：Kerr 真空偏振、Jacobi beam；再评估 Kerr–Newman 偏振、slow-light 与 Stokes/Faraday。

**退出条件**：screen basis、gauge、平行输运、Walker–Penrose/EVPA、analytic slab 和至少一个独立代码对照闭合；研究路径不拖慢或复杂化默认产品。

## 完成定义

一个可发行核心至少完成 Phase 0–4，并同时满足：

1. 数学：reference ladder 与 fixture 的适用域、精度和收敛已记录；
2. GPU：Metal/Vulkan 无 validation error，ABI 和 shader 产物可复现；
3. 图像：field comparison 与视觉回归同时通过；
4. 生命周期：zero extent、resize、suspend、surface/device error 无 panic/死锁/跨代 handle；
5. 性能：在具名 adapter/driver/profile 上达到 p50/p95 和显存门槛；
6. 发行：依赖、资产、字体、shader、配置和许可证闭环；
7. 声明：UI/export 不把研究状态、外观效果或未验证模型描述成物理预测。

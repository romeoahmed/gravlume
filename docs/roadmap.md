# 能力路线与退出条件

路线按依赖关系组织，不使用会渗入文件名、API 或错误文案的 Phase 编号。当前完成度以根 [README](../README.md#当前状态) 和两份实现证据文档为准；本页只定义下一项能力何时可以声称完成。

## 已建立的基础闭环

- validated domain 与独立 CPU `f64` reference；
- native winit/egui/wgpu 生命周期；
- Metal/Vulkan Cartesian Kerr–Schild GPU trace；
- 原子完整帧发布、选择性 shadow coverage、HDR/scRGB 与 SDR 输出；
- 保守 direction reconstruction、interval Kerr capture 与完整 KS fallback；
- typed errors、GPU timestamp、fixture 和 CPU/GPU observable gate。

退出证据见 [Reference 实现](reference-implementation.md)、[GPU renderer](gpu-renderer.md)和[验证合同](validation.md)。Windows/Linux 仍需具名系统、adapter、driver 与 compositor 的实机发布证据。

## 可解释物理图像

交付：薄表面盘、scalar emission/absorption、frequency ratio、blackbody/spectral LUT、scene-linear capture。

退出条件：

- `I_nu/nu^3`、frequency ratio、常系数 slab 与 optical-depth 解析回归通过；
- Physical/Appearance controls 不能互相绕过；
- numerical failure 不伪装成物理黑色；
- EXR/科学输出绕过 display tone mapping。

## 重建与交互预算

交付：source-space footprint、branch-aware reconstruction、stationary accumulation、resize reuse 与具名质量政策。

退出条件：

- critical curve、caustic、多像薄盘和高频 source 的字段误差通过；
- resize/cut/generation change 必须拒绝不相容 history；
- 选定 Metal/Vulkan 设备达到预注册的 p50/p95 时延与峰值显存预算；
- 动态分辨率若引入，必须原子发布完整画面且有 hysteresis，不显示低分辨率阶段。

## 解析与半解析 Kerr 路线

优先研究完整 Kerr elliptic/Carlson terminal solver，并以当前 Cartesian KS 作为定义域外与数值不确定时的基线。Mino fixed-step candidate 已因高分辨率 observable 反例退出 production，不能凭低分辨率 benchmark 复活。

退出条件：

- chart/physical-spin/turning-root 约定经符号和高精度数值验证；
- regular、near-critical、near-axis、near-extreme 与高绕转样本达到同一 observable budget；
- fallback 分类无误接受，mutation 可击穿 gate；
- 完整 invalidation→publish 的 GPU p50/p95 在具名设备上形成更优 Pareto 点。

## 视觉与资产闭包

交付：经定义的 sky/disk/jet/noise/bloom/false-color 与 asset manifest。

退出条件：每项效果明确属于 Physical 或 Appearance；来源、许可、色彩空间、方向、转换链和发行权闭合；默认场景离线启动。

## 研究质量

候选：Kerr 真空偏振、Jacobi beam、Kerr–Newman 偏振、slow-light 与 Stokes/Faraday。

退出条件：screen basis、gauge、平行输运、Walker–Penrose/EVPA、analytic slab 和至少一个独立实现对照闭合；研究路径不增加默认产品的状态或依赖，除非成为 resolved plan。

## 可发行核心

1. 数学：reference ladder 的适用域、精度与收敛有记录；
2. GPU：Metal/Vulkan 无 validation error，ABI 和 shader 可复现；
3. 图像：machine-readable fields 与视觉回归同时通过；
4. 生命周期：zero extent、resize、suspend、surface/device error 无 panic/死锁/跨代发布；
5. 性能：具名 adapter/driver/profile 达到时延和显存门槛；
6. 发行：依赖、资产、字体、shader、配置和许可证闭环；
7. 声明：UI/export 不把研究状态、外观效果或未验证模型描述成物理预测。

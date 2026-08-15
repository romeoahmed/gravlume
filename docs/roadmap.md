# 能力路线与退出条件

本文只定义尚未完成能力的依赖顺序和退出条件。当前实现状态由 [Reference 证据](reference-implementation.md)与 [GPU 证据](gpu-renderer.md)维护；历史实验由[研究记录](research/)维护。路线不使用会渗入文件名、接口或错误文案的阶段编号。

## 已建立的基础

- validated domain 与独立 CPU `f64` reference；
- CPU equatorial circular surface 的 Source Anchor、Frequency Ratio 与 vacuum bolometric fixture；
- scene-owned equatorial emitter 与 GPU full-KS `g^4 I_em` 直接 HDR transport；
- path-integrated homogeneous scalar slab、diluted-blackbody $gT$、固定 observer-frame boxcar LUT
  与四份 v3 analytic/high-precision transport fixtures；
- committed branch key、CPU 五射线 source footprint，以及 ordinary-region GPU diagnostic
  Jacobian/parity gate；
- tone-map/UI 之前的 tagged scientific scene-linear readback 与 source/transport/channel/error metadata；
- native winit/egui/wgpu 生命周期；
- Metal/Vulkan Cartesian Kerr–Schild GPU trace；
- escape-direction map、interval radial capture 与完整 KS fallback；
- shadow coverage、完整帧原子发布、HDR/scRGB 与 SDR 输出；
- typed errors、GPU timing、versioned fixture 与 CPU/GPU observable gate。

Windows 和 Wayland 仍缺具名 OS、adapter、driver 与 compositor 的实机发布矩阵。

## 重建与交互预算

**交付：** source-space footprint、branch-aware reconstruction、stationary accumulation、resize reuse 与具名质量政策。

**退出条件：**

- critical curve、caustic、多像薄盘和高频 source 的字段误差通过；
- resize、cut 或 generation change 拒绝不相容 history；
- 具名 Metal/Vulkan 设备达到预注册的 p50/p95 延迟和峰值显存预算；
- 动态分辨率若引入，仍只原子发布完整画面，并有 hysteresis 与最小驻留时间。

## 解析与半解析 Kerr

**交付：** 受限域内的 elliptic/Carlson terminal solver，Cartesian Kerr–Schild 负责定义域外和条件不确定时的 fallback。

数值 fixed-step Mino candidate 已被高分辨率 travel-time 反例否决；恢复它需要新的 phase-error certificate，而不是更小的经验步长。

**退出条件：**

- chart、physical spin、root topology 与 turning branch 经符号和高精度验证；
- regular、near-critical、near-axis、near-extreme 与高绕转样本满足同一 observable budget；
- classifier 无误接受，且关键 mutation 能击穿 gate；
- 完整 invalidation-to-publication 延迟形成更优 Pareto 点；
- unsupported topology、volume checkpoints 与 future transport 不被 terminal-only solver 伪装支持。

## 视觉与资产

**交付：** 定义清楚的 sky、disk、jet、noise、bloom、false color 与 asset manifest。

**退出条件：** 每项效果属于 Physical 或 Appearance；来源、许可、色彩空间、方向、转换链和发行权闭合；默认场景能够离线启动。

## 研究能力

**候选：** Kerr 真空偏振、Jacobi beam、Kerr–Newman 偏振、slow-light、Stokes/Faraday 与 scattering。

**退出条件：** screen basis、gauge、平行输运、Walker–Penrose/EVPA、analytic slab 与至少一个独立实现对照闭合；研究路径不增加默认产品状态或依赖，除非成为正式 resolved plan。

## 可发行核心

1. **数学：** reference ladder 的适用域、精度与收敛有记录；
2. **GPU：** Metal/Vulkan 无 validation error，ABI 与 shader 可复现；
3. **图像：** machine-readable fields 与视觉回归同时通过；
4. **生命周期：** zero extent、resize、suspend、surface/device error 无 panic、死锁或跨代发布；
5. **性能：** 具名 adapter、driver 和 profile 达到延迟与显存门槛；
6. **发行：** 依赖、资产、字体、shader、配置和许可证闭环；
7. **声明：** UI 与 export 不把研究状态、外观效果或未验证模型描述成物理预测。

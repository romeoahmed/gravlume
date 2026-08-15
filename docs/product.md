# 产品范围与科学声明

本文是 Gravlume 产品语义的权威合同：定义产品是什么、明确不是什么，以及何时允许声称一项能力已经完成。当前实现状态不在这里维护，见 [Reference 证据](reference-implementation.md)与 [GPU 证据](gpu-renderer.md)。

## 产品定义

Gravlume 是一个桌面黑洞观测仪器。用户定义时空、观察者、发射体、介质和天空环境；系统计算抵达观察者的光线与可观测量，再以不混淆物理和外观的方式形成交互图像。

“仪器”意味着：

1. 像素能够追溯到 termination、source anchor、image branch、frequency ratio 与数值诊断；
2. 交互路径能够与独立 reference 对同一不可变场景作结构化比较；
3. 超出支持域、高误差或未验证的结果保持可见，不被黑色、平滑或漂亮画面掩盖。

## 使用方式

| 入口       | 用户目标                                  | 产品责任                                                                 |
| ---------- | ----------------------------------------- | ------------------------------------------------------------------------ |
| Explore    | 理解引力透镜、阴影、多像、频移和视角效应  | 安全的默认场景；明确区分物理量与外观控件                                 |
| Instrument | 复现图像和字段                            | 保存场景、质量政策、revision、GPU profile、shader contract 与资产摘要    |
| Research   | 比较 solver、精度、时空、偏振或 transport | 有界任务、取消、资源政策和独立 artifact；不共享 live viewport 的可变状态 |

## 产品边界

### 核心闭环

- Schwarzschild 与 Kerr exterior geometry；domain 保留 Kerr–Newman 参数与极端性分类；
- 外部 Observer Event、正交定向 Observer Frame 与 viewport ray；
- 独立 CPU `f64` reference 和 GPU WGSL `f32` trace；
- typed physical terminal、数值 guard、step exhaustion、uncertainty 与 failure；
- sky、薄表面、frequency ratio 与标量 emission/absorption；
- scene-linear HDR、可审计 capture 与中性 display transform；
- branch-aware classifier、选择性 refine 与不跨物理不连续面的重建；
- resize、zero extent、suspend、surface/device loss、资产失败和无效输入的显式生命周期。

这份清单定义产品边界，不代表每项已经实现。完成度以[能力路线](roadmap.md)和实现证据为准。

### 允许的扩展

- Kerr 解析/半解析 terminal solver、Schwarzschild LUT 与 transfer map；
- source-space differential、stationary accumulation、受约束 temporal correspondence 与动态分辨率；
- 可解释的薄盘模型、标量体介质和 retarded-time 数据合同；
- Kerr 真空偏振与 Jacobi beam；reference 闭合后的 Stokes/Faraday 研究；
- 明确标为永恒解析解理想化展示的 Analytic-Extension View。

## 非目标

- 通用游戏引擎、ECS、场景编辑器或 renderer plugin 平台；
- 宇宙 octree、恒星人口、双星/多体、行星、化学、生命、文明或社会模拟；
- 把普通 RGB 天空图解释为光谱 radiance；
- 把程序化噪声、bloom、false color、haze 或“photon ring 强调”解释为 GRRT；
- 依赖硬件 ray tracing、单一 backend `f64` 或可选 GPU feature 才能正确运行；
- 默认网络资产、遥测或在线服务；
- 在没有目标 plasma、谱模型、分辨率和误差定义时声称“完整偏振”或“完整 GRRT”。

## 三类模型

| 模型             | 决定什么                                                                                       | 不得做什么                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Physical Scene   | spacetime、observer、emitters、media、sky 与可物理解读的 path observation                      | 不包含 exposure、bloom 或求解器选择                                   |
| Appearance Model | exposure、display transform、bloom、false color 与 overlay                                     | 不修改初始 null direction、frequency ratio、emissivity 或 termination |
| Quality Policy   | observable error、frame time、memory、resolution、refinement、reconstruction 与 transport tier | 不把内部 solver 名称变成场景语义，也不隐藏失败                        |

只改变 Appearance 不应重新 trace。Physical 或 transport 语义变化必须换代相应 generation。

## 默认物理语义

- 使用几何单位 `G=c=1`；UI 与 artifact 显式记录质量尺度，GPU 可归一化到 `M=1`；
- 默认场景是亚极端 Kerr exterior，$q_e=0$，不跨 Cauchy horizon；
- View Ray 携带 future-directed Photon Momentum；Backward Trace 只改变数值遍历方向；
- 频率由 $-p\cdot u>0$ 定义，负值不能取绝对值后继续解释；
- 默认 fast-light；slow-light 是另一种 source 数据语义，不是质量开关；
- physical result 在 display transform 前保持 scene-linear；普通 RGB 色相移动属于 Appearance。
- `visible-boxcar-v1` 的 RGB 是三个具名 observer-frame 波段，不是 sRGB/CIE color；只有 surface
  radiance texel 可按该 metadata 解释，analytic sky 与 failure 仍是明确的预览/诊断类别。
- homogeneous slab 只声明总 optical depth 的解析 scalar transfer；空间变化 volume、scattering、
  稳定吸积盘与完整 GRRT 必须各自满足独立合同后才能宣称。

具体连续模型和阈值分别由[数学物理合同](physics.md)与[验证合同](validation.md)定义。

## 完成定义

“有代码”“能出图”“通过物理与数值验证”“达到交互预算”“可发行”是五个独立状态。一项能力只有在数学、observable、GPU 协议、生命周期、性能、资产与声明边界同时通过时才能称为完成。最低发布条件见[可发行核心](roadmap.md#可发行核心)。

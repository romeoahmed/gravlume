# 产品范围与科学声明

## 1. 产品定义

Gravlume 是一个桌面黑洞观测仪器：用户定义时空、观察者、发射体、介质和天空环境，系统计算抵达观察者的光线及其可观测量，再以明确区分物理与外观的方式形成交互图像。

这里的“仪器”有三个含义：

1. 图像像素能够回到 termination、source anchor、image branch、frequency ratio 和误差诊断；
2. 交互路径与独立 reference 可以对同一不可变场景作结构化比较；
3. 未验证、高误差或超出模型适用域的结果保持可见，不以黑色、平滑或漂亮画面掩盖。

## 2. 目标用户与入口

### 2.1 Explore

面向希望理解引力透镜、黑洞阴影、多像、吸积盘频移和视角效应的用户。默认打开一个孤立、亚极端 Kerr 黑洞的 exterior view；参数有安全范围，UI 明确说明哪些控件是物理量、哪些只是外观。

### 2.2 Instrument

面向需要可重复图像和字段输出的开发者或研究者。场景、质量策略、软件 revision、GPU profile、shader contract 和资产摘要能够写入 artifact；EXR/字段捕获绕过非物理 display transform。

### 2.3 Research

面向 solver 比较、CPU/GPU agreement、Kerr–Newman、偏振、slow-light 或 transfer-map 实验。研究入口有有界任务、取消、精度与资源政策，不共享 live viewport 的可变状态。

## 3. 首版范围

### 3.1 必须形成闭环

- Schwarzschild 与 Kerr exterior geometry；领域层保留 Kerr–Newman 参数和极端性分类。
- 外部 Observer Event 与正交、定向 Observer Frame。
- CPU `f64` reference trace；GPU WGSL `f32` interactive trace。
- escape、horizon crossing、emitter hit、singularity guard、step exhaustion 和 numerical failure 的类型化分类。
- sky 与薄表面发射，正确的局部频率比和标量 emission/absorption。
- scene-linear HDR、中性 display transform、egui overlay 与可审计捕获。
- coarse trace、几何 classifier、选择性 refine 和不跨 image-branch 的空间重建。
- resize、zero extent、surface loss、device loss、资产失败和无效参数的明确状态机。

### 3.2 后续但仍在产品边界内

- Kerr 专用解析/半解析路径、Schwarzschild LUT 与 Kerr transfer map。
- source-space ray differential/Jacobian、stationary accumulation、受约束 temporal correspondence 和动态分辨率。
- 可解释的薄盘温度/发射模型、标量体介质、retarded-time 数据契约。
- Kerr 真空偏振和 Jacobi beam；完整 Stokes/Faraday 只在 reference 验证闭合后进入研究档。
- Analytic-Extension View，只作为永恒解析解的理想化展示，不解释为真实坍缩黑洞内部。

## 4. 明确非目标

- 通用游戏引擎、ECS、场景编辑器或 renderer plugin 平台；
- 宇宙 octree、恒星人口、双星/多体轨道、行星结构、宜居带、化学、生命、文明、经济或社会模拟；
- 把普通 RGB 天空图解释为光谱 radiance；
- 把程序化噪声、bloom、false color、haze 或“photon ring 强调”解释为 GRRT；
- 依赖硬件 ray tracing、单后端 `f64` 或任何可选 GPU feature 才能正确运行；这些能力只能加速或扩展已验证基线；
- 默认网络资产、遥测或在线服务；
- 未定义目标 plasma、谱模型、分辨率和误差时声称“完整偏振/完整 GRRT”。

## 5. 三个互不替代的模型

### Physical Scene

包含 spacetime、observer、emitters、participating media 与 sky environment。它决定可物理解读的 path observation。变更它会使相关 geometry/transport generation 换代。

### Appearance Model

包含 exposure、display transform、bloom、false color 和 overlays。它只能消费物理结果，不能修改初始 null direction、frequency ratio、emissivity 或 termination。只改 appearance 不应重新 trace。

### Quality Policy

包含 observable error、frame-time、memory、resolution、refinement、reconstruction 和 transport tier 的预算。它不公开 solver 名称；内部 resolved plan 可以切换算法，但必须保持场景语义并报告失败。

## 6. 默认物理语义

- 使用几何单位 `G=c=1`，存档/UI 边界显式记录质量尺度；GPU 打包可再无量纲化为 `M=1`。
- 默认是亚极端 Kerr exterior view；$q_e=0$，不跨 Cauchy horizon。
- View Ray 携带 future-directed Photon Momentum；Backward Trace 只是相反的数值遍历方向。
- 频率始终由 `-p·u > 0` 定义，不能对负值取绝对值后继续解释。
- 默认 fast-light：发射体与介质冻结在一个有 revision 的快照。slow-light 是另一种数据语义，不是“更高质量”按钮。
- Physical result 在 display transform 前保持 scene-linear；普通 RGB source 的色相移动必须标为 Appearance。

首个可复现 Observation、reference policy、fixture schema 与验收阈值固定在[验证合同](validation.md)；它是开发合同，不是已实现声明。

## 7. 什么叫“完成”

“有代码”“能形成画面”“物理与数值已验证”“达到交互性能”“可发行”是五个独立状态。某阶段只有同时满足其数学、GPU 协议、生命周期、视觉字段、性能和资产来源门槛才算完成。最低要求见[路线图](roadmap.md)与[验证矩阵](architecture.md#12-验证矩阵与完成定义)。

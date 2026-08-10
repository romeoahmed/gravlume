# Gravlume

Gravlume 是一个从零开始实现的 Rust 桌面科学可视化项目，用于研究并交互展示孤立黑洞附近的光传播与可观测外观。名称由 *gravitation* 与 *lumen* 组合而来：项目关心的是引力如何改变抵达观察者的光，而不是复刻某个既有引擎。

## 当前状态

仓库处于 **设计与验证基线阶段**。目前只有 Cargo 生成的最小二进制、版本化 fixture 和待实现合同；尚未实现窗口、GPU tracer、吸积盘或研究 workbench。文档中的性能目标、算法候选和阶段能力都是验收条件，不是已完成功能。

## 产品边界

Gravlume 计划提供：

- Schwarzschild、Kerr，以及研究模式下的 Kerr–Newman 时空；
- 任意外部观察者的 null-geodesic 反向追迹；
- 可验证的频率比、标量辐射传输、吸积盘/天空成像和线性 HDR 输出；
- 理解 termination、像支、源坐标与 footprint 的自适应重建；
- 独立 CPU `f64` reference、GPU `f32` 交互路径和可复现比较报告；
- 明确分开的物理模型、外观模型与质量策略。

首版不包含宇宙生成、恒星人口、轨道系统、行星、化学、生命、文明或通用游戏引擎。最大解析延拓、完整 Stokes/Faraday 传输和动态三维 plasma 属于有门槛的研究能力，不是默认产品承诺。

## Credit

Inspired by [NPGS](https://github.com/baopinshui/NPGS); Gravlume is an independent Rust implementation.

## 文档入口

1. [产品范围](docs/product.md)：用户能力、非目标和科学声明边界。
2. [数学物理合同](docs/physics.md)：约定、连续方程、observer 与可观测量。
3. [验证合同](docs/validation.md)：CPU reference、首个 Observation、fixture schema 和数值验收预算。
4. [架构与实现合同](docs/architecture.md)：领域模型、公开接口、GPU 生命周期、ABI 和资源换代。
5. [渲染算法与研究门槛](docs/rendering.md)：solver、传输、footprint、adaptive、temporal 与高风险方法。
6. [Rust 平台合同](docs/platform.md)：Cargo 依赖、Vulkan/Metal、设备基线与可选加速 variant。
7. [实施路线](docs/roadmap.md)：阶段交付物、退出条件和完成定义。
8. [文档总览](docs/README.md)：阅读路径、证据标记和维护规则。

## 本地状态检查

```bash
cargo check
cargo test
```

在 Phase 0 引入桌面依赖前，`Cargo.toml` 刻意保持最小。版本只在真实调用者出现时加入，并由 Cargo 解算后的 `Cargo.lock` 固定；平台文档中的版本是实施起点，不替代 manifest。

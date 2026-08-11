# Gravlume

**以可验证的物理量为核心，构建原生黑洞科学可视化工具。**

Gravlume 是一个从零实现的 Rust 桌面项目，目标是在 Metal 与 Vulkan 上交互研究孤立黑洞附近的光传播及其可观测外观。名称由 *gravitation* 与 *lumen* 组合而来：这里关心的是引力如何改变抵达观察者的光，而不只是生成一张看起来像黑洞的图。

## 当前状态

项目目前已进入 **Phase 1：领域与 CPU reference**。原生窗口仍展示验证渲染链路的诊断画面，**还不是 GPU 物理光线追迹器**；科学追迹当前通过无 GPU 的 Rust reference 测试与接口运行。

已经实现：

- Rust 2024 Cargo workspace 与锁定依赖；
- winit 原生生命周期和 egui overlay；
- wgpu compute 写入 scene-linear `rgba16float`，再经 SDR display pass 输出；
- resize、zero extent、suspend/resume 与 surface recovery；
- 结构化 GPU 错误、提交后纹理释放和非阻塞 timestamp readback；
- 覆盖能力选择、资源代际、显示变换和 GPU 合同的测试。
- 私有字段、原子提交的 validated `Observation`、Observer Event/Frame 与 viewport 初始光线；
- Cartesian Kerr–Schild `f64` Hamilton RHS、DP5(4) FSAL、自适应误差控制与 quartic dense output；
- typed horizon/escape/equatorial/singularity/resource terminal、事件 bracket/残差、守恒量 drift 与 baseline/strict comparison report；
- 严格 v1 TOML fixture seam、Schwarzschild 80 位 fixture 回归及保持输入顺序的专用 Rayon pool。

尚未实现：GPU `f32` geodesic tracer、吸积盘/天空成像、自适应重建和研究工作台。CPU reference 的当前证据范围与未覆盖 oracle 梯级见 [Phase 1 实现与证据](docs/phase-1.md)，不能从当前 fixture 外推。

macOS 使用 Metal；Windows 与 Linux 使用 Vulkan。Windows/Linux 发布支持仍需在具名系统、adapter 与 driver 上补齐验证证据。

## 快速开始

需要 Rust 1.97，以及满足 WebGPU baseline、`TIMESTAMP_QUERY` 和项目 `rgba16float` usage 要求的非软件 GPU adapter。

```bash
cargo run --locked
```

执行一次完整 present 与 GPU timing readback 后自动退出：

```bash
GRAVLUME_SMOKE_ONCE=1 cargo run --locked
```

运行仓库检查：

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

原生 GPU 测试需要可用的 Metal 或 Vulkan adapter。依赖事实以 `Cargo.toml` 与 `Cargo.lock` 为准。

## 代码结构

| 路径 | 职责 |
|---|---|
| `src/main.rs` | 日志初始化与桌面程序入口 |
| `crates/gravlume-desktop` | winit/egui 生命周期、事件与重绘调度 |
| `crates/gravlume-domain` | validated scene、observer/frame、viewport ray 与 Kerr–Schild `f64` 领域数学 |
| `crates/gravlume-reference` | 独立 DP5(4) CPU oracle、dense events、fixture、并行 batch 与 comparison report |
| `crates/gravlume-render` | wgpu 能力协商、frame graph、资源、计时与错误语义 |
| `crates/gravlume-render/src/shaders` | 运行时加载的已审查 WGSL |
| `tests/fixtures/v1` | reference/interactive comparison 的版本化科学输入 |
| `docs` | 产品、物理、验证、架构、平台和实施合同 |

## 设计与验证原则

- 物理正确性由 termination、守恒量、频率比、源坐标等 machine-readable observable 判断，不以“画面像”替代。
- CPU `f64` reference 与 GPU `f32` interactive path 保持独立，二者一致也不自动证明正确。
- Physical、Appearance 与 Quality 策略分离；研究能力不能伪装成默认产品承诺。
- 生命周期与 GPU 资源采用显式状态、代际和 typed error；生产路径不依赖 panic 恢复。
- 性能结论必须记录 profile、平台、adapter、driver、extent 与统计口径。

## 路线图

Phase 1 当前建立领域模型与 CPU reference；Phase 2 实现可诊断的 GPU geodesic tracing；Phase 3 加入辐射传输与可解释图像；Phase 4 完成重建、预算与性能门槛。资产广度、实验加速器和偏振等研究能力位于后续阶段。完整交付物与退出条件见[实施路线](docs/roadmap.md)。

## 文档

建议按以下顺序阅读：

1. [产品范围](docs/product.md) — 能力、非目标与科学声明边界；
2. [数学物理](docs/physics.md) — 坐标、时空、观察者与辐射约定；
3. [验证合同](docs/validation.md) — reference policy、fixture 与误差预算；
4. [架构合同](docs/architecture.md) — 模块接口、生命周期、GPU ABI 与资源换代；
5. [渲染研究](docs/rendering.md) — solver、传输、重建和实验门槛；
6. [平台合同](docs/platform.md) — Cargo 闭包、Metal/Vulkan 基线与能力协商；
7. [文档总览](docs/README.md) — 证据标记与维护规则。

贡献前请阅读 [AGENTS.md](AGENTS.md)。代码或行为改变必须同步维护受影响的规范性文档。

## 致谢

项目受 [NPGS](https://github.com/baopinshui/NPGS) 启发；Gravlume 是独立设计与实现。

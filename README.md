# Gravlume

Gravlume 是一个以可验证物理量为核心的原生黑洞可视化工具。它计算抵达观察者的光线与可观测量，并明确区分物理场景、数值质量和最终外观；目标不是只生成一张“像黑洞”的图。

本页只提供项目入口和运行摘要，不作为物理、验收阈值、平台支持或当前能力的权威来源。完整权威边界从[文档地图](docs/README.md)进入。

项目使用 Rust 2024、winit、wgpu 与 WGSL。CPU `f64` reference 和 GPU `f32` renderer 分别实现连续模型，通过结构化 observable 比较，而不共享离散求解器。

## 当前边界

当前实现包含 validated domain、独立 CPU reference、原生 GPU renderer、结构化 scientific inspection/capture 与桌面生命周期。精确覆盖和未覆盖域分别见 [Reference 证据](docs/reference-implementation.md)与 [GPU 证据](docs/gpu-renderer.md)；后续工作的依赖与退出条件只在[能力路线](docs/roadmap.md)维护。

默认画面使用简化的 equatorial circular bolometric surface 验证几何、频移、HDR 和失败可见性；
blackbody/boxcar 与 homogeneous scalar slab 也有独立证据，但都不代表稳定吸积盘或完整 GRRT。

## 快速开始

需要 [`Cargo.toml`](Cargo.toml) 声明的 Rust toolchain，以及满足项目 [GPU 基线](docs/platform.md#gpu-基线)的原生 Metal 或 Vulkan adapter。

```bash
cargo run --locked
```

运行一次完整的 trace、publication、presentation 与 GPU timing readback 后退出：

```bash
GRAVLUME_SMOKE_ONCE=1 cargo run --locked
```

完整验证：

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

原生 GPU 测试需要可用 adapter。依赖版本和 feature closure 始终以 `Cargo.toml` 与 `Cargo.lock` 为准。

## 文档入口

- [文档地图](docs/README.md)：按任务选择最小可信上下文，并定位唯一权威来源；
- [产品范围](docs/product.md)：产品边界、非目标与科学声明；
- [平台合同](docs/platform.md)：支持 target、图形后端、HDR 与发布证据。

## 致谢

项目受 [NPGS](https://github.com/baopinshui/NPGS) 启发；Gravlume 是独立设计与实现。

# Gravlume

Gravlume 是一个以可验证物理量为核心的原生黑洞可视化工具。它计算抵达观察者的光线与可观测量，并明确区分物理场景、数值质量和最终外观；目标不是只生成一张“像黑洞”的图。

项目使用 Rust 2024、winit、wgpu 与 WGSL。CPU `f64` reference 和 GPU `f32` renderer 分别实现连续模型，通过结构化 observable 比较，而不共享离散求解器。

## 当前能力

| 已进入主线                                                                  | 尚未形成产品闭环                                      |
| --------------------------------------------------------------------------- | ----------------------------------------------------- |
| validated Kerr–Newman domain、observer frame 与 viewport ray                | branch-aware source reconstruction 与 temporal reuse  |
| 独立 DP5(4) CPU reference、版本化 transport fixture 与误差报告              | 空间变化体介质、scattering、polarization 与 slow-light |
| Cartesian Kerr–Schild GPU trace、薄表面频移、标量 slab 与固定光谱波段       | near-critical、near-axis 与 near-extreme 的更广证据   |
| 完整 KS fallback、完整帧原子发布、shadow coverage 与 resize 生命周期        | 稳定磁盘/喷流模型、研究工作台与资产闭包               |
| tagged scene-linear scientific readback 与 macOS/Windows/Wayland 显示状态接入 | Windows 与 Wayland 的具名设备实机发布矩阵             |

默认画面使用简化的 equatorial circular bolometric surface 验证几何、频移、HDR 和失败可见性；
blackbody/boxcar 与 homogeneous scalar slab 也有独立证据，但都不代表稳定吸积盘或完整 GRRT。
已验证范围见 [Reference 证据](docs/reference-implementation.md)与 [GPU 证据](docs/gpu-renderer.md)。

## 快速开始

需要 Rust 1.97，以及满足项目 [GPU 基线](docs/platform.md#gpu-基线)的原生 Metal 或 Vulkan adapter。

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

## 支持平台

| 平台             | 图形后端         | 状态                                                  |
| ---------------- | ---------------- | ----------------------------------------------------- |
| macOS            | Metal            | 当前开发与运行时验证平台                              |
| Windows 11 22H2+ | Vulkan           | 已实现；仍需具名 adapter/driver 的 HDR 实机矩阵       |
| Linux            | Vulkan + Wayland | 已实现；仍需具名 compositor/adapter/driver 的实机矩阵 |

D3D12、GLES、X11、浏览器 WebGPU 与 WebGL 不在支持合同内。HDR 不可用或状态不可靠时，应用带原因降级至 SDR；FP16 中间纹理本身不构成 HDR 输出。

## 仓库结构

| 路径                             | 职责                                                           |
| -------------------------------- | -------------------------------------------------------------- |
| `src/main.rs`                    | 日志初始化与桌面入口                                           |
| `crates/gravlume-domain`         | validated scene、时空、observer、view 与 `f64` 领域数学        |
| `crates/gravlume-reference`      | 独立 CPU reference、event、fixture、batch 与 comparison        |
| `crates/gravlume-native-display` | AppKit、WinRT 与 Wayland display-state 的窄安全边界            |
| `crates/gravlume-render`         | wgpu trace、加速、frame publication、HDR/SDR 合成与 GPU timing |
| `crates/gravlume-desktop`        | winit/egui 生命周期、调度与用户界面                            |
| `docs`                           | 规范合同、当前证据、设计说明与研究决策                         |

## 设计原则

- 以 termination、方向、event residual、travel time、频率比和 invariant drift 判断正确性，不以截图代替科学 observable。
- Physical Scene、Appearance 与 Quality Policy 相互独立；外观设置不能改写物理结果。
- accelerator 只在可检查的支持域内生效；任何不确定性回退已验证基线，不猜测确定结果。
- GPU candidate 完成且 generation 匹配后才整体发布；用户不会看到 tile 扫描或低分辨率过渡帧。
- 性能数字必须绑定 revision、平台、adapter、backend、场景、extent、profile 与统计口径。

## 文档入口

- [文档总览](docs/README.md)：按规范、证据、设计和研究记录导航；
- [产品范围](docs/product.md)：产品边界、非目标与科学声明；
- [数学物理合同](docs/physics.md)：连续模型、符号和 observable；
- [验证合同](docs/validation.md)：reference policy、fixture 与误差预算；
- [架构合同](docs/architecture.md)：模块职责、生命周期、GPU ABI 与资源所有权；
- [平台合同](docs/platform.md)：原生后端、HDR 状态与发布证据；
- [能力路线](docs/roadmap.md)：下一项能力的退出条件。

## 致谢

项目受 [NPGS](https://github.com/baopinshui/NPGS) 启发；Gravlume 是独立设计与实现。

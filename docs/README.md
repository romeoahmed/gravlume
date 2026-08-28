# Gravlume 文档地图

本文是项目文档与源码上下文的导航入口，不定义产品、物理、API 或当前能力。它帮助读者只加载完成当前任务所需的最小可信上下文；具体事实仍由下列权威来源定义。

## 最小上下文包

| 任务                       | 先读                                                                             | 再核对源码/证据                                                                                                                                                                                                                      |
| -------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 理解产品与当前边界         | [产品范围](product.md) → [能力路线](roadmap.md)                                  | [Reference 证据](reference-implementation.md)、[GPU 证据](gpu-renderer.md)                                                                                                                                                           |
| 修改 domain 或连续模型     | [数学物理](physics.md) → [验证合同](validation.md)                               | [`gravlume-domain`](../crates/gravlume-domain/)、[`gravlume-reference`](../crates/gravlume-reference/) 及其 fixture/tests                                                                                                             |
| 修改 GPU trace 或 WGSL     | [数学物理](physics.md) → [验证合同](validation.md) → [架构合同](architecture.md) | [trace core](../crates/gravlume-render/src/trace.rs)、[inspection modules](../crates/gravlume-render/src/trace/)、[WGSL](../crates/gravlume-render/src/shaders/)、[GPU tests](../crates/gravlume-render/src/gpu_trace_tests.rs)与 [GPU 证据](gpu-renderer.md) |
| 修改 publication/lifecycle | [架构合同](architecture.md) → [平台合同](platform.md)                            | [renderer](../crates/gravlume-render/src/renderer.rs)、[frame ownership](../crates/gravlume-render/src/renderer/frame.rs)、desktop [app](../crates/gravlume-desktop/src/app.rs) 与 native smoke                                                                        |
| 修改 native HDR/display    | [平台合同](platform.md) → [架构合同](architecture.md)                            | [native display](../crates/gravlume-native-display/)、[capability resolver](../crates/gravlume-render/src/capabilities.rs)、[display](../crates/gravlume-render/src/display.rs)与 [HDR 决策](research/native-hdr-output.md)                                      |
| 研究新算法                 | [渲染设计](rendering.md) → [研究索引](research/README.md) → 对应记录             | 独立 oracle、[可复算脚本](research/scripts/)、受影响的 production baseline 与完整 observable gate                                                                                                                                   |
| 修改依赖、feature 或工具链 | workspace [`Cargo.toml`](../Cargo.toml)、crate manifests、[`Cargo.lock`](../Cargo.lock)、[平台合同](platform.md) | `cargo tree -e features` 与三个目标平台的 feature closure                                                                                                                                                                            |

贡献前始终先读 [AGENTS.md](../AGENTS.md)；它定义仓库工作流、边界和必跑检查。

## 权威来源

| 事实                                     | 唯一权威来源                                                               |
| ---------------------------------------- | -------------------------------------------------------------------------- |
| 依赖版本、Rust edition 与 Cargo features | [`Cargo.toml`](../Cargo.toml)、[`Cargo.lock`](../Cargo.lock)               |
| public Rust interface                    | crate source 与 rustdoc                                                    |
| WGSL entry point、binding 与布局         | shader source、host DTO、ABI/GPU tests                                     |
| 产品边界与科学声明                       | [产品范围](product.md)                                                     |
| 连续物理模型                             | [数学物理](physics.md)                                                     |
| profile、fixture 与 tolerance            | [验证合同](validation.md)、版本化 TOML fixture                             |
| 模块所有权与生命周期                     | [架构合同](architecture.md)                                                |
| target、backend 与 HDR 状态              | [平台合同](platform.md)                                                    |
| 当前实现、测试与适用域                   | [Reference 证据](reference-implementation.md)、[GPU 证据](gpu-renderer.md) |
| 未完成工作的依赖与退出条件               | [能力路线](roadmap.md)                                                     |
| 实验、性能历史与候选取舍                 | 对应 [research decision record](research/README.md)                        |

发生冲突时先核对源码、manifest 或对应规范，再修摘要；不要把另一份摘要复制成新的“真相”。规范文档约束行为但不声称能力已经实现，证据文档只描述当前仓库，研究记录不授权 production 行为。

## Workspace 地图

| 路径                             | 稳定职责                                                             |
| -------------------------------- | -------------------------------------------------------------------- |
| [`src/main.rs`](../src/main.rs)                                      | process entry：日志初始化与 desktop 启动                             |
| [`crates/gravlume-domain`](../crates/gravlume-domain/)                | validated values、scene、spacetime、observer、view 与独立 `f64` 数学 |
| [`crates/gravlume-reference`](../crates/gravlume-reference/)          | CPU oracle、event、fixture、comparison、batch 与 footprint           |
| [`crates/gravlume-native-display`](../crates/gravlume-native-display/) | 原生 display-state 的窄安全边界                                      |
| [`crates/gravlume-render`](../crates/gravlume-render/)                | wgpu/WGSL trace、publication、inspection、capture、display 与 timing |
| [`crates/gravlume-desktop`](../crates/gravlume-desktop/)              | winit/egui 组合根、输入、调度与用户可见状态                          |
| [`docs/research/scripts`](research/scripts/)                           | [锁定的 Python 研究工具](research/python-research-tooling.md)；不进入 Cargo runtime dependency closure |

模块边界与文件所有权的细节见[架构合同](architecture.md#依赖方向)和[Renderer modules](architecture.md#renderer-modules)，这里不维护第二份接口清单。

## 证据语言

研究与数学文档可用 `[P]`、`[A]`、`[N]`、`[X]` 分别标记一手来源、写明定义域的代数推导、可复算数值证据和未验证假设。公式一致不证明离散收敛，CPU/GPU 一致不证明两者都正确，截图相似不证明 branch 或物理量正确，adapter probe 也不构成发布矩阵。

## 维护规则

- 首段说明用途、权威范围和状态；规范写“必须/不得”，证据写“已覆盖/未覆盖”，研究写“假设/方法/结果/决策/恢复条件”。
- 一个事实只定义一次；其他文档只给足以导航的摘要和相对链接。
- 公式注明坐标、符号、单位、时间定向与适用域；数值注明 precision、输入、observable 和 tolerance。
- 性能绑定 revision、平台、adapter、backend、scene、extent、profile、样本设计和统计量，不能用 ray count 代替端到端延迟。
- API、行为、阈值、平台支持或科学声明变化时，同步更新其唯一权威文档和当前证据摘要。
- 研究方案落地或否决后，把索引与正文状态改为结果口吻；不要让“建议/应实现”继续描述当前路径。

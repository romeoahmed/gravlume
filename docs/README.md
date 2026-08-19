# Gravlume 文档

本目录把项目知识分成四类：**规范合同**定义必须成立的语义，**实现证据**描述仓库现在证明了什么，**设计与路线**说明尚未完成的方向，**研究记录**保存可复算的实验与决策。不同类别不能互相冒充。

## 从哪里开始

| 目标                     | 阅读路径                                                                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| 了解项目                 | [根 README](../README.md) → [产品范围](product.md) → [能力路线](roadmap.md)                                                      |
| 修改数学或 solver        | [数学物理](physics.md) → [验证合同](validation.md) → [Reference 证据](reference-implementation.md) / [GPU 证据](gpu-renderer.md) |
| 修改 renderer 或生命周期 | [架构合同](architecture.md) → [平台合同](platform.md) → [GPU 证据](gpu-renderer.md)                                              |
| 研究新算法               | [渲染设计](rendering.md) → [研究索引](research/README.md) → 对应决策记录                                                         |
| 贡献代码                 | [AGENTS.md](../AGENTS.md) → 受影响的规范合同                                                                                     |

## 规范合同

| 文档                        | 权威范围                                                |
| --------------------------- | ------------------------------------------------------- |
| [产品范围](product.md)      | 产品边界、非目标、科学声明与完成定义                    |
| [数学物理](physics.md)      | 坐标、符号、连续模型、事件与物理 observable             |
| [验证合同](validation.md)   | reference policy、fixture schema、误差预算与验收矩阵    |
| [架构合同](architecture.md) | 模块职责、接口、生命周期、GPU ABI 与资源所有权          |
| [平台合同](platform.md)     | 原生 target、backend、HDR 状态、WGSL/GPU 基线与发布证据 |

规范合同约束实现；它们不声称所有能力已经完成。修改已有 schema、profile、discriminant 或科学含义时必须引入新版本，而不是原地改义。

## 实现证据

| 文档                                                | 回答的问题                                                       |
| --------------------------------------------------- | ---------------------------------------------------------------- |
| [Reference 实现与证据](reference-implementation.md) | `gravlume-domain` 与 `gravlume-reference` 现在实现并验证了什么？ |
| [GPU Renderer 实现与证据](gpu-renderer.md)          | GPU trace、加速、发布和显示路径现在实现并验证了什么？            |

证据文档只记录当前仓库。公式、阈值、平台语义和历史实验应链接到权威合同或研究记录，不在证据文档维护副本。

## 设计与路线

| 文档                               | 作用                                                                   |
| ---------------------------------- | ---------------------------------------------------------------------- |
| [渲染算法与研究门槛](rendering.md) | 比较 geometry、transport、sampling、temporal 与 display 方案的适用边界 |
| [能力路线](roadmap.md)             | 解析未完成能力的依赖顺序、前置条件、交付物与退出条件                   |

设计文档可以描述候选，但不能把候选写成已实现能力。采用或否决一项实验后，把生产事实回写规范/证据，把推理与反例留在研究记录。

## 研究记录

[`research/`](research/) 保存符号推导、数值实验、性能方法和已采用/已拒绝决策。研究脚本是证据生成器，不进入运行时依赖闭包。状态和 production 影响以 [研究索引](research/README.md)为准。

## 权威性顺序

| 事实                             | 唯一权威来源                         |
| -------------------------------- | ------------------------------------ |
| crate 版本与 Cargo features      | `Cargo.toml`、`Cargo.lock`           |
| public Rust interface            | crate source 与 rustdoc              |
| WGSL entry point、binding 与布局 | shader source、host DTO、ABI tests   |
| 连续物理模型                     | `physics.md`                         |
| profile、fixture 与 tolerance    | `validation.md`、版本化 TOML fixture |
| 当前测试与适用域                 | 两份实现证据文档                     |
| 性能历史与候选取舍               | 对应 research decision record        |

发生冲突时先修权威来源，再更新引用它的摘要；不要通过复制另一份文字“修一致”。

## 证据标记

研究与数学文档可使用以下标记：

| 标记  | 含义                                         |
| ----- | -------------------------------------------- |
| `[P]` | 标准、官方文档、同行评审论文或作者发布的实现 |
| `[A]` | 写明假设和定义域的代数或符号推导             |
| `[N]` | 写明输入、精度、算法和 observable 的数值复算 |
| `[X]` | 尚待 Gravlume 验证的假设，不是已实现能力     |

公式一致不证明离散收敛，CPU/GPU 一致不证明两者都正确，截图相似不证明 branch 或物理量正确，adapter probe 也不构成发布矩阵。

## 写作与维护规则

- 首段说明文档用途、权威范围和状态；读者不应依靠文件历史推断它是否仍有效。
- 一个事实只定义一次。其他文档给出短摘要和相对链接。
- 规范写“必须/不得”；证据写“已覆盖/未覆盖”；研究写“假设/方法/结果/决策/恢复条件”。
- 使用描述性标题与稳定相对链接，不依赖临时绝对路径、源码行号或 roadmap 阶段名。
- 公式注明坐标顺序、度规符号、单位、时间定向与适用域；数值注明 precision、输入、observable 和 tolerance。
- 性能注明 revision、平台、adapter、backend、场景、extent、profile、样本设计与统计量；ray count 不能冒充端到端延迟。
- API、行为、阈值、平台支持或科学声明变化时，在同一变更中更新权威文档。

# Rust 平台合同

Gravlume 面向原生桌面：最新 macOS 使用 Metal，Windows 与 Linux 使用 Vulkan。平台合同只规定已选择的工具链、最低设备能力和可验证的加速路径；依赖的最终版本与 feature closure 由 `Cargo.toml`、`Cargo.lock` 和三平台测试决定。

## 1. 工具链与依赖

首个实现使用 Rust 1.97、edition 2024。下表是已核对的兼容起点；真正的版本与 feature closure 以提交后的 manifest、lockfile 和三平台测试为准。

| 职责 | 版本组 | 使用边界 |
|---|---|---|
| desktop GPU/UI | [`wgpu 30.0.0`](https://docs.rs/wgpu/30.0.0/wgpu/)、[`winit 0.30.13`](https://docs.rs/winit/0.30.13/winit/)、`egui` / `egui-winit` / [`egui-wgpu 0.36.1`](https://docs.rs/crate/egui-wgpu/0.36.1/source/Cargo.toml) | 同组升级；egui-wgpu 的 manifest 已对齐 wgpu 30 与 winit 0.30.13 |
| math/ABI | [`glam 0.33.3`](https://docs.rs/glam/0.33.3/glam/)、[`encase 0.12.0`](https://docs.rs/encase/0.12.0/encase/) | glam 只作实现数学；领域类型和 WGSL DTO 独立 |
| shader tool | [`naga 30.0.0`](https://docs.rs/naga/30.0.0/naga/)、[`wesl 0.4.2`](https://docs.rs/wesl/0.4.2/wesl/) | 只在 build/tool/test graph；runtime 加载已验证 WGSL |
| runtime support | `thiserror 2.0.20`、`tracing 0.1.44`、[`tracing-subscriber 0.3.23`](https://docs.rs/tracing-subscriber/0.3.23/tracing_subscriber/)、`pollster 1.0.1`、`serde 1.0.229`、`toml 1.1.4`、`image 0.25.10`、`rayon 1.12.0` | 按真实调用者加入；image 只启用资产格式，Rayon 使用有界专用 pool |
| test/UI helper | [`approx 0.5.1`](https://docs.rs/approx/0.5.1/approx/)、[`strum 0.28`](https://docs.rs/strum/0.28.0/strum/)、[`strum_macros 0.28`](https://docs.rs/strum_macros/0.28.0/strum_macros/) | glam 的 `approx` 仍使用 0.5，避免并存 0.6 RC；strum 通过 `derive` feature 引入同版 macros，且不生成持久化名、GPU discriminant 或本地化文本 |
| analytic research | [`ellip 1.1.1`](https://docs.rs/ellip/1.1.1/ellip/)、[`ellip-rayon 1.1.1`](https://docs.rs/ellip-rayon/1.1.1/ellip_rayon/) | ellip 先验证 root/branch/time/observable；ellip-rayon 只比较批量齐次调用，不与 sample-level Rayon 嵌套 |

desktop executable 安装唯一 `tracing-subscriber`，首版只需 `fmt` 与 `env-filter`。用户输入的 filter 使用 `EnvFilter::builder().with_regex(false)` 严格解析；科学 artifact 不依赖异步日志队列。

## 2. Cargo 闭包

`wgpu`、`egui-wgpu` 和平台 window features 按 Metal/Vulkan 目标裁剪；直接 Naga、WESL、fixture 与 benchmark 依赖不进入运行时闭包。每次升级审查 `cargo tree -e features` 与重复协议 crate，并通过 shader generation、headless GPU contract 和三平台 native smoke。辅助 crate 不得造成第二套 wgpu、egui、winit 或 glam/approx 协议版本。

## 3. 发布平台基线

| 目标 | 后端 | 发布条件 |
|---|---|---|
| 最新稳定 macOS | Metal | native surface、headless compute、shader 与生命周期测试通过 |
| Windows desktop | Vulkan | 具名系统、adapter、driver 通过同一测试集 |
| Linux desktop | Vulkan | 具名发行版、adapter、driver 通过同一测试集 |

D3D12、GLES、Web/WebGL 不在首版兼容声明内。release build 只创建目标后端的 instance，不用 `Backends::PRIMARY` 或 `all()` 扩大测试责任。

候选 adapter 必须：

1. [`DownlevelCapabilities::is_webgpu_compliant`](https://docs.rs/wgpu/30.0.0/wgpu/struct.DownlevelCapabilities.html#method.is_webgpu_compliant) 为真；
2. 满足项目 limits 以及每个实际纹理的 usage/format 要求；
3. 支持 `TIMESTAMP_QUERY` 与 `CLEAR_TEXTURE`；
4. 不是 software adapter，专用 headless 测试除外；
5. 从 `SurfaceCapabilities::format_capabilities` 选择合法的 format/color-space pair，并从 capabilities 选择 present mode。

这是一条明确的现代桌面基线。不满足时返回 `UnsupportedPlatform`，不改变物理模型、精度或资源语义继续运行。三平台要求离散 termination/branch 一致，连续 observable 在各自误差预算内收敛；不要求逐位相同。

## 4. Device 能力解析

Cargo feature 决定编译哪些 backend 和 shader frontend；[`FeaturesWebGPU`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWebGPU.html) 与 [`FeaturesWGPU`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWGPU.html) 决定 adapter 可创建什么 device。两层不能混用。

创建 device 前，私有 `CapabilityPlan`：

1. 读取 adapter info、features、limits、downlevel 与 surface capabilities；
2. 对每个候选纹理读取 adapter format features，并按最终 enabled features 复核 [`TextureFormat::guaranteed_format_features`](https://docs.rs/wgpu/30.0.0/wgpu/enum.TextureFormat.html#method.guaranteed_format_features)；
3. 从 Quality、Diagnostics 和显式 research request 解析所需 variant；
4. 验证 feature 依赖和 format usages；项目 limits 用 adapter 的 resolution/alignment 要求分别合并，不能复制 adapter 全部上限；
5. 一次性请求所需集合，并把最终 feature、limit、format 与 shader key 写入 artifact。

禁止请求 `Features::all()` 或 adapter 报告的全集。device 创建后若研究任务需要另一能力，创建新的 workbench device；普通 viewport 不为研究路径重建 device。公开接口只暴露项目级 policy 与 capability diagnostics。

## 5. GPU 能力策略

### 5.1 原生基线

- core WGSL `f32`、compute、storage buffer/texture 和 WebGPU baseline limits 承担 geometry、transport、reconstruction 与 display；
- `TIMESTAMP_QUERY` 提供 pass-boundary GPU 时间；结果乘 `Queue::get_timestamp_period`；
- `CLEAR_TEXTURE` 负责语义确实为零的 history/diagnostic 初始化；
- scene-linear HDR 使用 `rgba16float`，metadata 使用整数 buffer/texture；创建前验证实际 usages。

[`TIMESTAMP_QUERY`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWebGPU.html#associatedconstant.TIMESTAMP_QUERY) 与 [`CLEAR_TEXTURE`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWGPU.html#associatedconstant.CLEAR_TEXTURE) 均覆盖 Metal 与 Vulkan，且有直接消费者，因此作为发布要求而非兼容分支。

### 5.2 可选加速 variant

wgpu 30 的 [`FeaturesWebGPU`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWebGPU.html) 与 [`FeaturesWGPU`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWGPU.html) 中覆盖 Metal 与 Vulkan 的能力，均可进入 adapter-resolved plan。“当前没有 consumer”只表示本次不请求。只有下列证据可以淘汰某个 adapter/workload variant：

1. adapter 不报告该 feature、limit 或 format usage；
2. 官方规定的 feature 依赖、stage 或 shader 合同无法满足；
3. 正确性或性能 A/B 未达到预先定义的门槛。

下列是已有明确黑洞可视化 workload 的首批 variant：

| feature | 可接受用途 | Gate |
|---|---|---|
| [`SUBGROUP`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWGPU.html#associatedconstant.SUBGROUP) | active-ray ballot/reduction、compact queue 构建、tile/statistics reduction | 不假定 subgroup width 或 lane/local-invocation 映射；Metal/Vulkan 分别 A/B |
| [`SUBGROUP_BARRIER`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWGPU.html#associatedconstant.SUBGROUP_BARRIER) | 同一 subgroup 内确有跨 lane 内存依赖的 compute kernel | 同时请求 `SUBGROUP` |
| [`SHADER_F16`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWebGPU.html#associatedconstant.SHADER_F16) | LUT decode、reconstruction、appearance | geodesic、frequency、radiance 与 invariant 保持 `f32`；逐字段 A/B |
| [`FLOAT32_FILTERABLE`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWebGPU.html#associatedconstant.FLOAT32_FILTERABLE) | `r32/rg32/rgba32float` LUT 的硬件过滤 | 实际 LUT 与手工插值比较误差、带宽和 GPU time |
| [`IMMEDIATES`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWebGPU.html#associatedconstant.IMMEDIATES) | 高频、小型 compute/render per-dispatch 常量 | 非零 `max_immediate_size`；证明 [`ComputePass::set_immediates`](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePass.html#method.set_immediates) 优于现有 upload |
| [`SHADER_FLOAT32_ATOMIC`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWGPU.html#associatedconstant.SHADER_FLOAT32_ATOMIC) | forward packet/splat 的 many-to-one `f32` 累加 | 只使用 load/store/add/sub/exchange；与分层归约比较总时间、争用和数值误差 |

wgpu/Naga 30 的 native subgroup 路径可用，但与当前标准 WGSL 的 enable 协议尚未对齐：请求 `Features::SUBGROUP`，shader 使用已实现的 subgroup builtin，但在这一版本中**不写** `enable subgroups;`；Naga 30 会拒绝该指令。这是随 wgpu/Naga 版本锁定的 shader source contract，不是对 subgroup 的禁用。CI 必须用锁定 Naga parse/validate 该 variant，记录 `AdapterInfo::subgroup_min_size/max_size`，并在升级时重新核对指令、builtin 命名和 stage 限制。[官方 feature 文档](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWGPU.html#associatedconstant.SUBGROUP) [实现差异 #5555](https://github.com/gfx-rs/wgpu/issues/5555) [Naga enable 跟踪 #8202](https://github.com/gfx-rs/wgpu/issues/8202)

### 5.3 Metal/Vulkan 可选项注册表

下表保留 wgpu 30 文档中覆盖两类目标后端的其余现代能力；部分能力只在特定 GPU/OS 版本可用，因此始终以 adapter 查询为准。它们不进入同一 device 全集，`CapabilityPlan` 只请求实际管线的依赖闭包。

| 能力 | 项目内可选用途 | 启用边界 |
|---|---|---|
| `TEXTURE_FORMAT_16BIT_NORM` | 紧凑的 sky/source/LUT 通道 | 逐 format/usage 查询；不用于需保留 HDR 范围的字段 |
| `TEXTURE_COMPRESSION_ASTC_HDR` | HDR environment/appearance 资产 | 必须有 adapter 支持、canonical 源和图像误差验收 |
| `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` | 使用设备额外的 storage/render/sample format usage | 只为已查询的具体 format/usage 启用，不把它当成格式全开关 |
| `TEXTURE_COMPRESSION_BC`, `TEXTURE_COMPRESSION_BC_SLICED_3D`, `TEXTURE_COMPRESSION_ASTC`, `TEXTURE_COMPRESSION_ASTC_SLICED_3D` | sky/environment 与 3D source 资产 | 按 adapter 产出资产 variant；科学 fixture 保留无损 canonical 源 |
| `RG11B10UFLOAT_RENDERABLE`, `BGRA8UNORM_STORAGE`, `FLOAT32_FILTERABLE` | HDR intermediate、direct storage display、float LUT 过滤 | 分别核对 render/storage/filter usage，不从一个 feature 推断其他 usage |
| `MAPPABLE_PRIMARY_BUFFERS` | unified-memory adapter 的 upload/readback | 只在共享内存上 A/B；官方警告其在非共享内存上可严重损害性能 |
| `TEXTURE_BINDING_ARRAY`, `STORAGE_RESOURCE_BINDING_ARRAY`, `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING`, `STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING` | 多 sky/source/snapshot 资源集与非一致索引 | 按官方依赖分开请求；ABI 和 limit 必须随 variant 验证 |
| `ADDRESS_MODE_CLAMP_TO_ZERO`, `ADDRESS_MODE_CLAMP_TO_BORDER` | LUT/source 边界语义 | 边界色必须与物理 out-of-domain policy 一致 |
| `SHADER_I16`, `SHADER_INT64` | 紧凑状态、长程计数器或精确整数 key | 不代替 geodesic 的浮点误差验收；`i16/u16` shader 需 `enable wgpu_int16;` |
| `SHADER_INT64_ATOMIC_MIN_MAX`, `TEXTURE_ATOMIC`, `TEXTURE_INT64_ATOMIC` | 诊断范围、histogram/binning、精确整数 accumulation | 核对官方操作子集、MSL/Vulkan 扩展和争用成本 |
| `MEMORY_DECORATION_COHERENT` | 确有跨 invocation 可见性需求的持久队列实验 | 不能代替缺失的 dispatch 内 global barrier；Metal 要求 MSL 3.2+ |
| `VERTEX_WRITABLE_STORAGE`, `MULTIVIEW`, `MULTISAMPLE_ARRAY`, `DEPTH_CLIP_CONTROL`, `DEPTH32FLOAT_STENCIL8`, `INDIRECT_FIRST_INSTANCE`, `DUAL_SOURCE_BLENDING`, `CLIP_DISTANCES`, `PRIMITIVE_INDEX`, `POLYGON_MODE_LINE`, `SHADER_BARYCENTRICS` | stereo/diagnostic/source-mesh/display raster 路径 | 只进入使用对应 stage/resource 的管线，不污染 compute-only device plan |
| `EXPERIMENTAL_COOPERATIVE_MATRIX` | 8×8 `f32` reconstruction/matrix kernel 研究 | 官方当前仅实现 8×8 `f32`；只在独立研究构建中审查 `ExperimentalFeatures::enabled()` 的 `unsafe` 合同并做精确 A/B |

`EXPERIMENTAL_MESH_SHADER`/`EXPERIMENTAL_MESH_SHADER_POINTS` 虽有 Metal 与 Vulkan 后端，但 wgpu 30 的 Naga 路径仅支持 Vulkan；Metal 需要绕过解析与验证的 `PASSTHROUGH_SHADERS` `unsafe` API，因此不属于当前 safe-WGSL 集。该限制只针对 wgpu/Naga 30 的工具链合同。

## 6. WGSL、WESL 与 ABI

GPU 语义以 [WGSL specification](https://www.w3.org/TR/WGSL/) 和锁定的 wgpu/Naga 为准。WGSL core 没有 `f64`；浮点 rounding、subnormal、NaN/Inf 与 fusion 只按规范承诺设计。shader 不依赖某一 adapter 的偶然结果。

WESL 是构建期组合工具：`wesl.toml` 固定 `2026_pre` edition，输入只使用 imports 与 conditional translation；生成的纯 WGSL 提交到仓库。`cargo xtask shaders --check` 要求生成结果无 diff，并对每个 entry/variant 执行 Naga parse/validate、binding、override 与 ABI contract test。runtime 只加载已验证 WGSL。

Rust/WGSL ABI 遵循 [Rust type layout](https://doc.rust-lang.org/stable/reference/type-layout.html) 与 [WGSL memory layout](https://www.w3.org/TR/WGSL/#memory-layouts)：领域类型不直接作 GPU DTO，enum 使用固定 `u32` discriminant 和 checked conversion，`f64 → f32` pack 逐字段验证 finite、范围、归一化与 orientation。

## 7. 发布证据

| 层 | 必须证据 |
|---|---|
| dependency | lockfile；无协议型重复版本；只编译目标后端 |
| resolver | required feature closure、minimum limits、format usages 与结构化拒绝原因 |
| shader | 生成无 diff；全部 entry/variant 由锁定 Naga parse/validate；ABI 精确匹配 |
| headless | 基线与每个启用 variant 在 Metal/Vulkan 上通过奇数 extent、边界 workgroup、resize 与 readback |
| numerical | variant 不改变 termination/branch；连续字段满足 Validation Profile |
| performance | 固定 OS、adapter、driver、power mode、scene、extent、warm-up、样本数、p50/p95 与显存峰值 |
| release | macOS/Metal、Windows/Vulkan、Linux/Vulkan 无 validation error；记录 resolved capabilities |

单一 adapter 探针不能代替发布矩阵。版本升级作为整组变更审查，必须重新生成 shader、解析 Cargo feature closure，并复跑三平台合同。

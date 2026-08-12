# Interactive Trace 实现与证据

本文记录当前 interactive GPU tracer 的实现边界。连续模型、误差预算与阶段退出条件仍分别以[数学物理合同](physics.md)、[验证合同](validation.md)和[实施路线](roadmap.md)为准；本文只描述仓库中已经存在且可复算的能力。

## 已实现闭环

- `GpuEngine::new` 接受 validated `Observation`。host 端先按质量 $M$ 无量纲化，再通过受检 `f64 → f32` 转换写入 144-byte `TraceUniforms`；projection/sample、event surfaces 与 step policy 各自占独立字段，无法表示的输入返回 `TraceInputError`，不会饱和或静默截断为无穷。
- `TraceUniforms` 使用显式 `#[repr(C)]` 标量数组并派生 `Pod/Zeroable`；测试 readback DTO 只聚合已分别解码的三个 plane，不冒充 GPU 内存布局。生产路径不保留只服务测试捕获的 record DTO 或 compute entry point。GPU 把同一语义记录拆成 direction/time、invariant drift、metadata 三个 16-byte-per-pixel storage plane；这使 3840×2160 的每个 binding 为 126.6 MiB，落在 WebGPU 保证的 128 MiB storage-binding limit 内。termination discriminant 固定为 horizon、escape、singularity guard、step exhaustion、numerical failure 与 uncertain 六类。
- WGSL 独立实现 `f32` ingoing Cartesian Kerr–Schild radius/geometry、轴线解析分支、closed-form Hamilton RHS、negative-affine geometric-step RK4、线性 event fraction、端点导数约束的三次 Hermite dense state 与 null/E/Lz/Carter drift。终止步的 travel time、invariant drift、direction 与 event residual 全部提交同一 localized state，不包含 event surface 之后的 RK endpoint。radicand、denominator 和 finite guard 都写入 machine-readable failure flags。
- 每个像素以当前 HDR texture extent 构造 top-left viewport sample，因此 resize 后的 aspect ratio 与 record index 不依赖启动时尺寸。8×8 workgroup 越界 invocation 显式返回。
- escape 使用方向编码的解析高动态范围天空，horizon 写物理黑色；singularity、exhaustion、uncertain 和 numerical failure 使用不同的非黑诊断颜色。interactive failure 不冒充物理 terminal，也不静默变黑。
- 生产 frame graph 以 trace target 替代早期诊断 gradient，继续复用中性 display transform、egui overlay、timestamp readback 和既有 surface 生命周期。

## ABI 与依赖决定

`glam` 只在 `gravlume-domain` 内部用于 CPU `DVec3` 数学，feature closure 保持为 `f64 + std`。GPU 上传边界不暴露 `glam` 类型，因此不启用 `glam/bytemuck`：当前布局保证来自自有 `#[repr(C)]` 数组结构、编译期 `Pod` 检查、host offset/size 测试和 Naga WGSL validation。若未来公开接口直接上传 `glam` 类型，必须先为具体类型与 WGSL 对齐重新建立 ABI 测试，不能仅靠 feature 开关推断布局正确。

`num-traits::ToPrimitive` 是 renderer 的直接依赖，只用于把 validated binary64 observation 打包为有限 binary32；overflow/NaN/Inf 在资源创建前成为 typed error。

## 当前自动化证据

`cargo test -p gravlume-render --all-targets --locked` 当前覆盖：

- termination discriminant 的双向 checked mapping；
- host uniform size/field offset 与 WGSL 五 binding/生产 entry-point 合同；
- Naga 对独立生产 WGSL 的 parse 与 validation；测试 capture entry point 由真实 GPU pipeline 执行覆盖；
- 物理等价的质量尺度变换产生逐 bit 相同的无量纲 GPU trace record；
- center、四角与 jitter 的 CPU/WGSL initial direction agreement，角误差不高于 `2e-6 rad`，初始 null residual 不高于 `8e-5`；
- 默认 Kerr Observation 的 7×5 headless regular matrix 中五个具名样本：GPU termination 与 CPU reference 一致，escape direction 角误差不高于 `3.82e-4 rad`，event residual 不高于 `5e-3`；
- 9×9 extent 跨 workgroup boundary 的每像素 record-plane/HDR 写入；
- 4K UHD extent 在 WebGPU 默认 buffer limits 下通过容量验证，超过 texture/binding/buffer limit 的 resize 在分配前返回 typed error；
- 默认 regular matrix 的 localized travel time 与 CPU reference 相差不高于 `1e-3 M`，每项记录的 invariant drift 均不高于 interactive `0.05` budget；

这些测试需要可用的原生 Metal 或 Vulkan adapter。CPU/GPU 使用不同精度与不同积分器；agreement 只说明当前样本落在预算内，不构成独立物理证明。

## 适用域与未外推项

- 当前 CPU/GPU matrix 是默认 exterior Kerr scene 的小规模 regular 样本，不覆盖 near-critical 分支、Kerr–Newman 参数扫描、near-axis Carter 条件性或不同 escape radius。
- interactive RK4 采用固定的 radius-scaled step policy；守恒量超过预算时返回 `Uncertain`，当前没有 Phase 4 的 classify/refine 第二遍追迹。
- 解析天空用于验证方向、HDR 与 terminal 可见性，不是物理 source model。薄盘、frequency ratio、emission/absorption 和 spectral output 属于 Phase 3。
- 没有 60 FPS 或显存预算声明。Windows/Linux 仍需在具名 adapter、driver 和 release profile 上记录 smoke；当前测试通过不能替代该平台证据。

## 实现来源

- WGSL storage layout 与 built-in invocation 语义：[W3C WGSL address-space layout constraints](https://www.w3.org/TR/WGSL/#address-space-layout-constraints) 与 [WGSL built-in values](https://www.w3.org/TR/WGSL/#builtin-values)。
- wgpu 资源、limits 与 dispatch 接口：[`DeviceExt::create_buffer_init`](https://docs.rs/wgpu/30.0.0/wgpu/util/trait.DeviceExt.html#tymethod.create_buffer_init)、[`Limits`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Limits.html)、[`DeviceDescriptor::required_limits`](https://docs.rs/wgpu/30.0.0/wgpu/type.DeviceDescriptor.html#structfield.required_limits)、[`BindingType`](https://docs.rs/wgpu/30.0.0/wgpu/enum.BindingType.html)、[`ComputePass::dispatch_workgroups`](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePass.html#method.dispatch_workgroups) 与 [`Buffer::map_async`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Buffer.html#method.map_async)。
- WGSL parser/validator：[Naga 30.0.0](https://docs.rs/naga/30.0.0/naga/)；host byte contract：[`bytemuck::Pod`](https://docs.rs/bytemuck/1.25.2/bytemuck/trait.Pod.html) 与 [`bytes_of`](https://docs.rs/bytemuck/1.25.2/bytemuck/fn.bytes_of.html)。
- Rust host layout 与 discriminant 语义：[Rust Reference type layout](https://doc.rust-lang.org/stable/reference/type-layout.html) 与 [enumerations](https://doc.rust-lang.org/reference/items/enumerations.html)；受检数值转换：[`num_traits::ToPrimitive`](https://docs.rs/num-traits/0.2.19/num_traits/cast/trait.ToPrimitive.html)。

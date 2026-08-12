# Interactive Trace 实现与证据

本文记录当前 interactive GPU tracer 的实现边界。连续模型、误差预算与阶段退出条件仍分别以[数学物理合同](physics.md)、[验证合同](validation.md)和[实施路线](roadmap.md)为准；本文只描述仓库中已经存在且可复算的能力。

## 已实现闭环

- `GpuEngine::new` 接受 validated `Observation`。host 端先按质量 $M$ 无量纲化，再通过受检 `f64 → f32` 转换写入 144-byte `TraceUniforms`；interactive profile 要求派生的 $\omega_{obs}$ 在具名预算内归一为 1，且 canonical binary64 parameter state 不得在 f32 pack 后改变。当前 Kerr–Newman 模型平稳且 GPU observable 只保存非负 coordinate-time duration，因此初始 shader time 明确归零，避免 absolute binary32 time 的 ULP 使 travel time 依赖坐标原点；空间 event、frame 与方向仍来自 validated Observation。违反这些 seam 条件或存在无法表示的字段时返回 `TraceInputError`。projection/sample、event surfaces 与 step policy 各自占独立字段。
- `TraceUniforms` 使用显式 `#[repr(C)]` 标量数组并派生 `Pod/Zeroable`；测试 readback DTO 只聚合已分别解码的三个 plane，不冒充 GPU 内存布局。生产路径不保留只服务测试捕获的 record DTO 或 compute entry point。GPU 把同一语义记录拆成 direction/time、invariant drift、metadata 三个 16-byte-per-pixel storage plane；3840×2160 的单 binding 为 126.6 MiB，仍落在 WebGPU 保证的 128 MiB storage-binding limit 内。隐藏 candidate 56 B/pixel、published scene 8 B/pixel、UI 4 B/pixel；普通 viewport 按 2560×1440 封顶约 239 MiB，two-phase rebuild 的理论新旧峰值约 478 MiB。超预算 resize 在分配前返回 typed validation error。termination discriminant 固定为 horizon、escape、singularity guard、step exhaustion、numerical failure 与 uncertain 六类。
- WGSL 独立实现 `f32` ingoing Cartesian Kerr–Schild radius/geometry、轴线解析分支、closed-form Hamilton RHS、negative-affine geometric-step RK4、线性 event fraction、端点导数约束的三次 Hermite dense state 与 null/E/Lz/Carter drift。终止步的 travel time、invariant drift、direction 与 event residual 全部提交同一 localized state，不包含 event surface 之后的 RK endpoint；四项 drift 任一超过 `0.05` 都把确定终止降为 `Uncertain`。singularity guard 的正值乘积在计算前检测溢出并向有限最大值饱和，因此远场 guard side 不使其余可表示几何失败。radicand、denominator 和 finite guard 都写入 machine-readable failure flags。
- 每个像素以当前 HDR texture extent 构造 top-left viewport sample，因此 resize 后的 aspect ratio 与 record index 不依赖启动时尺寸。trace 以额外 16-byte dispatch uniform 提供线性 pixel offset，继续使用 8×8 workgroup；每个 batch 只覆盖自己的连续 pixel interval，尾部越界 invocation 显式返回。
- escape 使用方向编码的解析高动态范围天空，horizon 写物理黑色；singularity、exhaustion、uncertain 和 numerical failure 使用不同的非黑诊断颜色。interactive failure 不冒充物理 terminal，也不静默变黑。
- 生产 frame graph 把不可见 `candidate` 与只含完整画面的 FP16 `published scene` 分离。compute 可以按 batch 推进，但 incomplete candidate 从不进入 display bind group；每个 $1/4\to1/2\to1$ resolution tier 的最后一批在同一 queue timeline 中先完成 trace、再整图发布、最终 present，因此画面只在完整 tier 边界跳变，不再从上到下扫描。低分辨率发布使用 nearest reconstruction，避免对 termination/source-direction discontinuity 做无依据插值；native tier 最终覆盖它。完成 submission 的 timestamp/readback 返回后才释放旧 candidate 并分配下一 tier，避免两份完整 record planes 在途重叠；resize generation 变化后，旧 completion 只完成 GPU 回收，不能推进或发布新 generation。
- scene 与 egui 不再直接画到同一 gamma target：published scene 保持 extended-linear sRGB，egui 单独画到透明 `Rgba8Unorm`，final pass 对 premultiplied gamma UI 做 unpremultiply → sRGB decode → linear-premultiply，再在线性空间以 SDR reference white 合成。SDR 保持原有 Reinhard 映射；HDR 使用 FP16 `Rgba16Float + ExtendedSrgbLinear`，1.0 以下保持 identity，亮部按实时 headroom 压缩。精确 HDR pair 或可靠平台状态缺失时以 typed reason 降级到 SDR，不把 unknown 当作 HDR。
- 每个 accepted RK4 endpoint 的 exact geometry/RHS 同时供 event、invariant 与下一步 $k_1$ 使用；这不是把 classical RK4 的 $k_4$ 当 FSAL，而是复用原先为 endpoint 另算的 derivative。普通步由源码级 5 RHS/8 geometry 降为 4 RHS algebra/4 geometry，积分 tableau、step policy 与现有 observable gate 不变。
- 每个 extent generation 建立新的 trace 状态；一个 batch 在途时不追加更多 trace work，timestamp 回读完成后按具名 soft budget 有界调整下一批大小。已完成 generation 只重画 published scene/UI，不重复追迹。overlay 与 smoke 记录完成比例、batch count、累计 compute time 和最大 batch time。

## ABI 与依赖决定

`glam` 只在 `gravlume-domain` 内部用于 CPU `DVec3` 数学，feature closure 保持为 `f64 + std`。GPU 上传边界不暴露 `glam` 类型，因此不启用 `glam/bytemuck`：当前布局保证来自自有 `#[repr(C)]` 数组结构、编译期 `Pod` 检查、host offset/size 测试和 Naga WGSL validation。若未来公开接口直接上传 `glam` 类型，必须先为具体类型与 WGSL 对齐重新建立 ABI 测试，不能仅靠 feature 开关推断布局正确。

`num-traits::ToPrimitive` 是 renderer 的直接依赖，只用于把 validated binary64 observation 打包为有限 binary32；overflow/NaN/Inf 在资源创建前成为 typed error。

## 当前自动化证据

`cargo test -p gravlume-render --all-targets --locked` 当前覆盖：

- termination discriminant 的双向 checked mapping；
- host uniform size/field offset 与 WGSL 六 binding/生产 entry-point 合同；
- Naga 对独立生产 WGSL 的 parse 与 validation；测试 capture entry point 由真实 GPU pipeline 执行覆盖；
- 物理等价的质量尺度变换产生逐 bit 相同的无量纲 GPU trace record；
- observer/target coordinate time 同时平移到 $10^8 M$ 不改变 termination、escape direction 或 travel time；
- center、四角与 jitter 的 CPU/WGSL initial direction agreement，角误差不高于 `2e-6 rad`，初始 null residual 不高于 `8e-5`；
- 默认 Kerr Observation 的 7×5 headless regular matrix 中五个具名样本：GPU termination 与 CPU reference 一致，escape direction 角误差不高于 `3.82e-4 rad`，event residual 不高于 `5e-3`；
- 9×9 extent 跨 workgroup boundary 的每像素 record-plane/HDR 写入；
- 17×9 extent 的多 batch dispatch 与单 dispatch 逐 bit 相同；
- 2560×1440 native internal pixel budget 边界通过验证；4K native trace 及其他超过单个 frame-resource bundle budget 的 resize 在分配前返回 typed error；
- 默认 regular matrix 的 localized travel time 与 CPU reference 相差不高于 `1e-3 M`，每项记录的 invariant drift 均不高于 interactive `0.05` budget；

这些测试需要可用的原生 Metal 或 Vulkan adapter。CPU/GPU 使用不同精度与不同积分器；agreement 只说明当前样本落在预算内，不构成独立物理证明。

## 适用域与未外推项

- 当前 CPU/GPU matrix 是默认 exterior Kerr scene 的小规模 regular 样本，不覆盖 near-critical 分支、Kerr–Newman 参数扫描、near-axis Carter 条件性或不同 escape radius。
- interactive RK4 采用固定的 radius-scaled step policy；守恒量超过预算时返回 `Uncertain`，当前没有 Phase 4 的 classify/refine 第二遍追迹。
- 解析天空用于验证方向、HDR 与 terminal 可见性，不是物理 source model。薄盘、frequency ratio、emission/absorption 和 spectral output 属于 Phase 3。
- 没有 60 FPS 声明；batch 只控制 watchdog/主循环响应，完整 tier 发布只修正可见语义，resolution ladder 与 endpoint reuse 才降低首帧/总计算成本。现有旧基准用于说明原始根因，不能直接当作本实现的新性能结论；新实现仍需固定 build/profile/adapter 的 warm p50/p95。该证据不外推到 Windows/Linux，也不替代 Phase 4 的三平台动态分辨率验收。当前只声明分配前的 frame-resource peak上限，不声称已测得完整 driver GPU memory peak。

## 实现来源

- WGSL storage layout 与 built-in invocation 语义：[W3C WGSL address-space layout constraints](https://www.w3.org/TR/WGSL/#address-space-layout-constraints) 与 [WGSL built-in values](https://www.w3.org/TR/WGSL/#builtin-values)。
- wgpu 资源、limits 与 dispatch 接口：[`DeviceExt::create_buffer_init`](https://docs.rs/wgpu/30.0.0/wgpu/util/trait.DeviceExt.html#tymethod.create_buffer_init)、[`Limits`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Limits.html)、[`DeviceDescriptor::required_limits`](https://docs.rs/wgpu/30.0.0/wgpu/type.DeviceDescriptor.html#structfield.required_limits)、[`BindingType`](https://docs.rs/wgpu/30.0.0/wgpu/enum.BindingType.html)、[`ComputePass::dispatch_workgroups`](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePass.html#method.dispatch_workgroups)、[`Queue::write_buffer`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html#method.write_buffer)、[`Queue::get_timestamp_period`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html#method.get_timestamp_period) 与 [`Buffer::map_async`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Buffer.html#method.map_async)。
- HDR surface pair 与平台状态：[`wgpu` surface color spaces and HDR output](https://docs.rs/wgpu/30.0.0/wgpu/#surface-color-spaces-and-hdr-output)、[Apple custom tone mapping](https://developer.apple.com/documentation/metal/performing-your-own-tone-mapping)、[Windows `IDisplayInformationStaticsInterop::GetForWindow`](https://learn.microsoft.com/en-us/windows/win32/api/windows.graphics.display.interop/nf-windows-graphics-display-interop-idisplayinformationstaticsinterop-getforwindow) 与 [Windows reference-white mapping](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range#match-your-apps-reference-white-to-the-os-sdr-reference-white-level)。
- WGSL parser/validator：[Naga 30.0.0](https://docs.rs/naga/30.0.0/naga/)；host byte contract：[`bytemuck::Pod`](https://docs.rs/bytemuck/1.25.2/bytemuck/trait.Pod.html) 与 [`bytes_of`](https://docs.rs/bytemuck/1.25.2/bytemuck/fn.bytes_of.html)。
- Rust host layout 与 discriminant 语义：[Rust Reference type layout](https://doc.rust-lang.org/stable/reference/type-layout.html) 与 [enumerations](https://doc.rust-lang.org/reference/items/enumerations.html)；受检数值转换：[`num_traits::ToPrimitive`](https://docs.rs/num-traits/0.2.19/num_traits/cast/trait.ToPrimitive.html)。

# GPU Renderer 实现与证据

本文记录当前 GPU renderer 的实现边界。连续模型、误差预算与能力退出条件仍分别以[数学物理合同](physics.md)、[验证合同](validation.md)和[能力路线](roadmap.md)为准；本文只描述仓库中已经存在且可复算的能力。

## 已实现闭环

- `Renderer::new` 接受 validated `Observation`。host 端先按质量 $M$ 无量纲化，再通过受检 `f64 → f32` 转换写入 144-byte `TraceUniforms`；GPU profile 要求派生的 $\omega_{obs}$ 在具名预算内归一为 1，且 canonical binary64 extremality 不得在 f32 pack 后改变。当前 Kerr–Newman 模型平稳且 GPU observable 只保存非负 coordinate-time duration，因此初始 shader time 明确归零，避免 absolute binary32 time 的 ULP 使 travel time 依赖坐标原点；空间 event、frame 与方向仍来自 validated Observation。违反这些 seam 条件或存在无法表示的字段时返回 `GpuTraceInputError`。view/sample、event surfaces 与 step policy 各自占独立字段。
- `TraceUniforms` 使用显式 `#[repr(C)]` 标量数组并派生 `Pod/Zeroable`；测试 readback DTO 只聚合已分别解码的三个 plane，不冒充 GPU 内存布局。production shader 使用 uniform、FP16 output、dispatch 与 packed direction-reconstruction map 四个 binding；direction/time、invariant drift、metadata 三个 `16 B/pixel` plane 只由显式 diagnostic capture 创建。新 extent generation 是 candidate HDR `8 B/pixel` + UI `4 B/pixel`，另有按 tile-grid 大小计算的 transfer/coverage scratch；上一完整 scene 独立持有 `8 B/pixel`。candidate 完成后直接提升 texture view，不保留第二张同尺寸 published copy。4K 的两个 scratch 合计 `2,271,620 B/candidate`；初次 generation、cold rebuild、completed-scene rebuild 分别为 `101,804,428 B`、`203,608,848 B`、`201,337,220 B`，均通过 256 MiB gate。仍有 4K candidate 在追迹时再建同尺寸 replacement 需要 `269,964,040 B`，因此会在分配前返回 typed budget error。zero extent 与 suspend 都保留已安装 bundle，停止新 compute/acquire；suspend 期间的 resize 只由窗口系统合并，surface 成功恢复后再读取最新 physical inner size 重建一次。这保证任何 replacement 分配始终把仍安装的旧 bundle 纳入资源计划，不产生未计账的 retired generation。termination discriminant 固定为 horizon、escape、singularity guard、step exhaustion、numerical failure 与 uncertain 六类。
- WGSL 独立实现 `f32` Cartesian Kerr–Schild radius/geometry、显式 ingoing/outgoing principal direction、轴线解析分支、closed-form Hamilton RHS、negative-affine geometric-step RK4、线性 event fraction、端点导数约束的三次 Hermite dense state 与 null/E/Lz/Carter drift。交互相机使用 outgoing chart：反向相机追迹在 ingoing chart 的过去视界会遇到 coordinate barrier，而 outgoing chart 让该族光线正则穿越。uniform 中的 `spin` 始终是物理 $a=J/M$；shader 只在 oblate spatial map 内使用 chart-signed twist $s a$，radius、$\Sigma$、$\Delta$ 与 frame dragging 仍以同一个 physical spin 定义。终止步的 travel time、invariant drift、direction 与 event residual 全部提交同一 localized state，不包含 event surface 之后的 RK endpoint；四项 drift 任一超过 `0.05` 都把确定终止降为 `Uncertain`。singularity guard 的正值乘积在计算前检测溢出并向有限最大值饱和，因此远场 guard side 不使其余可表示几何失败。radicand、denominator 和 finite guard 都写入 machine-readable failure flags。
- 每个像素以当前 HDR texture extent 构造 top-left image sample，因此 resize 后的 aspect ratio 与 record index 不依赖启动时尺寸。额外 16-byte dispatch uniform 保存二维 tile origin；每个 batch 只覆盖自己的不重叠 `8×8` tile 矩形。node pass 在 4-pixel shared grid 上只追一次真实 KS ray，并把 branch 与 octahedral Escape direction 压成 `u32`；resolve pass 读取每 tile 的 `3×3` stencil，稳定时用四个 4×4 子格连续重建单位 Escape direction，不稳定、近场或 odd extent 尾 tile 执行逐像素 KS fallback。两个 pass 位于同一 command buffer 的独立 wgpu usage scope，完整候选帧仍只在最后一批结束后原子发布。
- 纯 Kerr presentation 可在受限支持域内先运行 interval Bernstein capture certificate：backward ray 初始向内且 12 段径向势的全部向外扩张 Bernstein 下界严格为正时直接写 Horizon；带电、near-extreme、near-axis、远场超界或任何区间/condition 不确定都回退完整 KS。该 shortcut 不生成 scientific travel-time/drift record；显式 diagnostic capture 仍走完整 solver。
- reciprocal second-order Mino RK4 曾作为 pure-Kerr terminal candidate 实现，但 `320×180` 独立 reference 暴露了超出 `1e-3 M` travel-time budget 的 accepted ray；低分辨率 factor envelope 与 reciprocal constraint 没有约束累计 phase error。因此该 solver、pipeline constants 与 benchmark variants 已从 production 删除。corrected physical-spin/outgoing KS–BL seam 与形式化脚本保留为后续 elliptic/Carlson terminal solver 的基础；当前 interval certificate 未接受的 ray 一律执行 Cartesian KS。
- escape 使用无 seam、低阶球面多项式与六个局部轴向色标编码的解析高动态范围天空，horizon 写物理黑色；该方向图避免薄网格在临界曲线附近产生过曝 alias，同时保留 source direction 的可读性。singularity、exhaustion、uncertain 和 numerical failure 使用不同的非黑诊断颜色。GPU failure 不冒充物理 terminal，也不静默变黑。
- 生产 frame graph 把不可见的 native-resolution `candidate` 与上一张完整 FP16 `published scene` 分离。compute batch 不再 acquire surface、运行 egui 或 present；incomplete candidate 从不进入 display bind group。最后一批 timestamp/readback 完成且 generation 仍匹配时，candidate view 原子提升为 published scene，然后只请求一次 presentation。没有整图 publication copy、低分辨率 tier、扫描式 reveal 或跨 termination/source-direction discontinuity 的临时插值。resize 期间旧完整 scene 按 aspect-fit 显示，比例外填黑；stale completion 只完成 GPU 回收，不能发布新 generation。
- scene 与 egui 不再直接画到同一 gamma target：published scene 保持 extended-linear sRGB，egui 单独画到透明 `Rgba8Unorm`，final pass 对 premultiplied gamma UI 做 unpremultiply → sRGB decode → linear-premultiply，再在线性空间以 SDR reference white 合成。SDR 保持原有 Reinhard 映射；HDR 使用 FP16 `Rgba16Float + ExtendedSrgbLinear`，1.0 以下保持 identity，亮部按实时 headroom 压缩。精确 HDR pair 或可靠平台状态缺失时以 typed reason 降级到 SDR，不把 unknown 当作 HDR。
- 每个 accepted RK4 endpoint 的 exact geometry/RHS 同时供 event、invariant 与下一步 $k_1$ 使用；这不是把 classical RK4 的 $k_4$ 当 FSAL，而是复用原先为 endpoint 另算的 derivative。普通步由源码级 5 RHS/8 geometry 降为 4 RHS algebra/4 geometry。切换到适配 backward trace 的 outgoing chart 后，旧 ingoing barrier 时代的 `r<6M` 十倍减速被删除；当前 radius-scaled policy 为 `min(0.1r, 8M)`、下限 `0.005M`，所有变更仍受 reference observable 与四守恒量 gate 约束。
- 每个 extent generation 建立新的 trace 状态；一个 batch 在途时不追加更多 trace work，timestamp 回读完成后按具名 soft budget和设备 `max_compute_workgroups_per_dimension` 共同限制下一批大小。已完成 generation 只重画 published scene/UI，不重复追迹。overlay 不以每批 surface redraw 换取跳动百分比；smoke 只有在 matching generation 的 presentation submission 确认 queue complete 后退出，并记录 batch count、累计 compute time 和最大 batch time。

## ABI 与依赖决定

`glam` 只在 `gravlume-domain` 内部用于 CPU `DVec3` 数学，feature closure 保持为 `f64 + std`。GPU 上传边界不暴露 `glam` 类型，因此不启用 `glam/bytemuck`：当前布局保证来自自有 `#[repr(C)]` 数组结构、编译期 `Pod` 检查、host offset/size 测试和 Naga WGSL validation。若未来公开接口直接上传 `glam` 类型，必须先为具体类型与 WGSL 对齐重新建立 ABI 测试，不能仅靠 feature 开关推断布局正确。

`num-traits::ToPrimitive` 是 renderer 的直接依赖，只用于把 validated binary64 observation 打包为有限 binary32；overflow/NaN/Inf 在资源创建前成为 typed error。

## 当前自动化证据

`cargo test -p gravlume-render --all-targets --locked` 当前覆盖：

- termination discriminant 的双向 checked mapping；
- host uniform size/field offset 与 production WGSL 四 binding 的 address-space/access/format 合同；
- Naga 对独立生产 WGSL 的 parse 与 validation；测试 capture entry point 由真实 GPU pipeline 执行覆盖；
- 物理等价的质量尺度变换产生逐 bit 相同的无量纲 GPU trace record；
- observer/target coordinate time 同时平移到 $10^8 M$ 不改变 termination、escape direction 或 travel time；
- 默认 80×45 outgoing Kerr 视野的每个像素都确定终止为 Horizon/Escape，且两类结果均存在；
- center、四角与 jitter 的 CPU/WGSL initial direction agreement，角误差不高于 `2e-6 rad`，初始 null residual 不高于 `8e-5`；
- 默认 Kerr Observation 的 7×5 headless regular matrix 中六个具名样本（四角、顶边、中心 capture）：GPU termination 与 CPU reference 一致，escape direction 角误差不高于 `3.82e-4 rad`，event residual 不高于 `5e-3`；
- 9×9 extent 跨 workgroup boundary 的每像素 record-plane/HDR 写入；
- 17×9 extent 的多 batch dispatch 与单 dispatch 逐 bit 相同；
- direction reconstruction 与 workgroup-local baseline 的全像素 branch 等价、重建 Escape direction 误差 gate，以及 interval Kerr capture 在默认/负自旋/近场/远场 profile 的 branch/direction 等价；near-extreme 与 near-axis profile 必须零接受并执行完整 KS；
- 4K native pixel boundary、设备 dispatch-dimension batch 上限和 cold/worst transactional core-resource plan；超 4K pixel policy 的 resize 在分配前返回 typed error；
- 默认 regular matrix 的 localized travel time 与 CPU reference 相差不高于 `1e-3 M`，每项记录的 invariant drift 均不高于 GPU `0.05` budget；

这些测试需要可用的原生 Metal 或 Vulkan adapter。CPU/GPU 使用不同精度与不同积分器；agreement 只说明当前样本落在预算内，不构成独立物理证明。

## 适用域与未外推项

- 当前 CPU/GPU matrix 是默认 exterior Kerr scene 的小规模 regular 样本，不覆盖 near-critical 分支、Kerr–Newman 参数扫描、near-axis Carter 条件性或不同 escape radius。
- GPU RK4 采用固定的 radius-scaled step policy；守恒量超过预算时返回 `Uncertain`，当前没有第二遍 classify/refine 追迹。
- 解析天空用于验证方向、HDR 与 terminal 可见性，不是物理 source model。薄盘、frequency ratio、emission/absorption 和 spectral output 尚未实现。
- 没有 60 FPS 声明；batch 只控制 watchdog/主循环响应，原生完整画面的原子发布修正可见语义，outgoing chart、重新标定的 geometric step、endpoint reuse 和保守 image/phase-space accelerator 才减少总计算。开发期 160×90 Metal A/B 中，旧实现平均/最坏为 882/2048 步，新 KS 策略为 61/132 步。在 Apple M5/Metal、1280×720 的历史位置对称 GPU A/B 中，共享 direction reconstruction map 相对 workgroup-local transfer 的改善为 `27.923–37.941%`，interval Kerr capture 在此基础上的增量为 `5.946–9.156%`。数值 Mino candidate 虽曾测得 `-35.768%`，但高分辨率 observable gate 否决了该收益，不能计入 production。shadow coverage 最新 run 为 `+2.803%`、95% CI `[-0.060%, +6.923%]`，因此只按视觉正确性保留，不宣称确定成本。这些是单机单 backend 的历史增量证据，不是 60 FPS 或跨平台声明；当前只声明分配前的 frame-resource peak 上限，不声称已测得完整 driver GPU memory peak。

## 实现来源

- WGSL storage layout 与 built-in invocation 语义：[W3C WGSL address-space layout constraints](https://www.w3.org/TR/WGSL/#address-space-layout-constraints) 与 [WGSL built-in values](https://www.w3.org/TR/WGSL/#builtin-values)。
- backward ray tracing 的 Kerr–Schild chart 选择：[Bozzola, Chan & Paschalidis 2023](https://doi.org/10.1103/PhysRevD.108.084004)。
- wgpu 资源、limits 与 dispatch 接口：[`DeviceExt::create_buffer_init`](https://docs.rs/wgpu/30.0.0/wgpu/util/trait.DeviceExt.html#tymethod.create_buffer_init)、[`Limits`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Limits.html)、[`DeviceDescriptor::required_limits`](https://docs.rs/wgpu/30.0.0/wgpu/type.DeviceDescriptor.html#structfield.required_limits)、[`BindingType`](https://docs.rs/wgpu/30.0.0/wgpu/enum.BindingType.html)、[`ComputePass::dispatch_workgroups`](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePass.html#method.dispatch_workgroups)、[`Queue::write_buffer`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html#method.write_buffer)、[`Queue::get_timestamp_period`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html#method.get_timestamp_period) 与 [`Buffer::map_async`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Buffer.html#method.map_async)。
- HDR surface pair 与平台状态：[`wgpu` surface color spaces and HDR output](https://docs.rs/wgpu/30.0.0/wgpu/#surface-color-spaces-and-hdr-output)、[Apple custom tone mapping](https://developer.apple.com/documentation/metal/performing-your-own-tone-mapping)、[Windows `IDisplayInformationStaticsInterop::GetForWindow`](https://learn.microsoft.com/en-us/windows/win32/api/windows.graphics.display.interop/nf-windows-graphics-display-interop-idisplayinformationstaticsinterop-getforwindow) 与 [Windows reference-white mapping](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range#match-your-apps-reference-white-to-the-os-sdr-reference-white-level)。
- WGSL parser/validator：[Naga 30.0.0](https://docs.rs/naga/30.0.0/naga/)；host byte contract：[`bytemuck::Pod`](https://docs.rs/bytemuck/1.25.2/bytemuck/trait.Pod.html) 与 [`bytes_of`](https://docs.rs/bytemuck/1.25.2/bytemuck/fn.bytes_of.html)。
- Rust host layout 与 discriminant 语义：[Rust Reference type layout](https://doc.rust-lang.org/stable/reference/type-layout.html) 与 [enumerations](https://doc.rust-lang.org/reference/items/enumerations.html)；受检数值转换：[`num_traits::ToPrimitive`](https://docs.rs/num-traits/0.2.19/num_traits/cast/trait.ToPrimitive.html)。

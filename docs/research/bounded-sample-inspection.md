# 有界单样本 GPU 路径审计

本文记录按需复算一个像素样本的 test-only GPU 证据、被拒绝的候选和恢复条件；它不定义 production interface。当前实现事实见 [GPU Renderer 实现与证据](../gpu-renderer.md)，未来交付以[路线图](../roadmap.md)为准，物理定义与误差预算分别以[数学物理合同](../physics.md)和[验证合同](../validation.md)为准。

**状态：固定尺寸 test-only record 已采用；production consumer interface 延后；`@workgroup_size(1)` 候选仍因 Metal 反例而拒绝。** 研究基线是 `9f39b8d798d3889ecb3032b5dcc92ad64103c6ad`，当前技术栈以 workspace [`Cargo.toml`](../../Cargo.toml) 与 `Cargo.lock` 为准。

## 问题与结论

既有 [`capture_trace_sample`](../../crates/gravlume-render/src/gpu_capture.rs)只 dispatch 包含目标像素的 tile，但仍按完整 viewport 分配四个 `16 B/pixel` record plane 与 HDR texture，并复制整个 extent。它还把两个 branch count 压成 16 bit，不适合作为 exact branch 证据。

可否证假设是：不改变 `trace_pixel_at`、RK4 step policy 或 event state machine，只增加一个固定 request、一个固定 record 与 plan-specific f32 scene-value sink，即可复算单条 ray 并取得 termination、exact branch、source、Frequency Ratio、travel time、scene-linear output 与 event/invariant diagnostics。三种现有 `TracePlan` 的真实 GPU test 均通过，因此该技术切片被采用；它没有证明 UI、generation、取消、排队或长期 artifact interface。

初版曾把 production-style generation、request ID、Busy/cancel/supersede/poll 状态机放入 test helper。仓库没有对应 production consumer，这些字段只能回显 host context，不能增加 GPU 科学证据。按“第二个真实消费者前不引入 public seam”的仓库规则，当前实现删除这套兼容性垫片，只保留一个同步 Interface：

```text
validated Observation + ImageSample
  -> fresh plan-matched full-KS trace
  -> fixed f32/u32 record
  -> typed test evidence
```

Module 内部隐藏 pipeline、bind group、request/result/readback buffer 与 decode；测试只学习一次调用和一个结果类型。若未来出现 interactive 与 validation 两个真实 consumer，再根据共同需求设计 generation、取消和错误语义，而不是从 test helper 冻结 Interface。

## Host-shareable layout

WGSL 的 `u32/f32` size/alignment 是 `4/4`，`vec4<u32>` 与 `vec4<f32>` 是 `16/16`，structure size 向最大 member alignment 取整。[WGSL host-shareable layout](https://www.w3.org/TR/WGSL/#alignment-and-size)给出这些规则。Rust DTO 使用 `#[repr(C, align(16))]` 与同序 `[u32; 4]/[f32; 4]`；[`repr(C)` layout](https://doc.rust-lang.org/reference/type-layout.html#the-c-representation)固定 declaration-order offset，[`bytemuck::Pod` derive](https://docs.rs/bytemuck/1.25.2/bytemuck/derive.Pod.html)拒绝含 padding 的 struct。

双方都不使用 host-shared `vec3`、`bool`、Rust enum 或 implicit padding。

### Request：32 bytes

| offset | lane | 语义 |
| ---: | --- | --- |
| 0 | `vec4<u32>` | pixel x/y、viewport width/height |
| 16 | `vec4<f32>` | subpixel x/y、reserved = 0 |

输入直接使用 domain 已验证的 `ImageSample`，并在 observation view seam 再确认 sample 属于该 extent。Request 通过 [`DeviceExt::create_buffer_init`](https://docs.rs/wgpu/30.0.0/wgpu/util/trait.DeviceExt.html#tymethod.create_buffer_init)创建，只声明实际使用的 `UNIFORM` usage；没有后续覆盖，因此不需要 `COPY_DST`。

### Result：96 bytes

| offset | lane | 语义 |
| ---: | --- | --- |
| 0 | `vec4<u32>` | termination、failure flags、steps、event candidate bits |
| 16 | `vec4<u32>` | radial turnings、equatorial crossings、bitcast signed winding、initial polar side |
| 32 | `vec4<f32>` | source coordinates / escape direction、travel time |
| 48 | `vec4<f32>` | 完整 plan-specific scene-linear RGBA 与 output tag |
| 64 | `vec4<f32>` | event residual、三个 reserved zero |
| 80 | `vec4<f32>` | null、energy、$L_z$、Carter maximum drift |

Result 的 alignment、size 与 storage-array stride 都是 `16, 96, 96`；两个 branch count 不压缩。`NumericalFailure` 没有 exact terminal branch，只接受全零 branch sentinel并返回 `None`；`Uncertain` 可能携带 provisional counters，但仍返回 `None`。其他已提交终止保留完整 branch。Host 还检查 termination、polar side、failure/candidate bits、finite values、reserved lane 与 scene tag 的合法组合。

短生命周期 record 与 host decoder 来自同一编译单元，不持久化也不跨版本读取，因此 version、producer/domain tag、request/generation echo 都是无消费者的 compatibility fields。Producer 与 arithmetic domain 仍由当前 pipeline 事实固定为 fresh full-KS retrace 与 WGSL binary32，但不伪装成 GPU 回传的额外证据。

## Shader composition 与 dispatch

[`sample_inspection.wgsl`](../../crates/gravlume-render/src/shaders/sample_inspection.wgsl)只保留一个 dispatch/guard/store 实现；analytic 与 surface 文件各自提供一个 plan-specific `inspected_scene_value` adapter。Surface production path 把原先直接 `textureStore` 的逻辑提取为返回 `vec4<f32>` 的纯函数，presentation 与 inspection 再分别写入 texture 或 record。这样 output seam 只有一个实现，避免 test sink 复制 transport/tag 条件。

最初候选使用 `@workgroup_size(1)`。Apple M5 Metal 上它能返回正确 terminal、source 与 radiance，却把同一 canonical sample 的 travel time、maximum drift 和部分 branch fields 返回为零；恢复 production 的 `8×8` specialization 后字段完整。WGSL 只定义 invocation/workgroup 语义，不承诺 backend compiler 对等变换或具体 subgroup width，见 [compute workgroups](https://www.w3.org/TR/WGSL/#compute-shader-workgroups)。因此当前保留：

```wgsl
@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn inspect_sample(@builtin(local_invocation_index) local_index: u32) {
    if local_index != 0u {
        return;
    }
    // one invocation executes trace_pixel_at and writes the record
}
```

Host 固定 `dispatch_workgroups(1, 1, 1)`。硬件启动 64 个 invocation，但只有 lane 0 在建立 ray state 前通过 guard；一条 geodesic 的 RK/event loop 仍串行，没有 `var<workgroup>`、barrier、atomic append 或 subgroup 假设。这里的 `8×8` 是 correctness evidence，不是吞吐或 SIMD 声明。

## Pipeline、资源与 readback

Inspection 是一次性 test pipeline，bind group 不与其他 pipeline 共享。`wgpu 30` 明确允许 `layout: None` 从所选 shader entry point推导 default layout，再用 [`ComputePipeline::get_bind_group_layout`](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePipeline.html#method.get_bind_group_layout)创建只供该 pipeline 使用的 bind group；[ComputePipelineDescriptor](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePipelineDescriptor.html#structfield.layout)也明确把这列为 simple pipeline 的便利模式。若未来需要跨 pipeline 复用 bind group，应恢复显式 shared layout，而不是沿用 default layout。

固定逻辑资源为：

| 资源 | bytes | usage |
| --- | ---: | --- |
| request | 32 | `UNIFORM` |
| result | 96 | `STORAGE | COPY_SRC` |
| readback | 96 | `COPY_DST | MAP_READ` |

合计 224 logical bytes，与 viewport extent 无关；这不是 driver allocation 或显存峰值声明。Portable WebGPU 要求 `MAP_READ` 只与 `COPY_DST`组合，[`BufferUsages`](https://docs.rs/wgpu/30.0.0/wgpu/struct.BufferUsages.html)与 [WebGPU buffer usages](https://www.w3.org/TR/webgpu/#buffer-usage)给出该约束，因此 storage result 与 staging readback 保持分离。

同一 encoder 先 dispatch、再执行 96-byte `copy_buffer_to_buffer`。提交后复用测试设备已有 readback helper：`map_async(Read)`、绑定 `SubmissionIndex` 的 `Device::poll(Wait)`、`get_mapped_range`、drop view、`unmap`。[`Buffer::map_async`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Buffer.html#method.map_async)说明 callback 由 submit/poll 驱动且 mapped buffer 与 GPU 使用互斥；[`PollType::Wait`](https://docs.rs/wgpu/30.0.0/wgpu/type.PollType.html)在 native backend 等待指定 submission 并执行 callback。测试基础设施错误保持测试失败，不再复制一套 production-style异步错误状态机。

## 可导出的 observable 与限制

| observable | test-only 可用性 | 限制 |
| --- | --- | --- |
| typed termination / failure flags | 直接 | unknown discriminant 拒绝 decode |
| event candidates / ambiguity / residual | 直接 | ambiguity 由 candidate bit count 派生 |
| exact branch key | terminal-dependent | `NumericalFailure` 与 `Uncertain` 为 unavailable |
| surface source / Frequency Ratio | accepted Surface | source lanes 是 `(r/M, phi, g)` |
| escape direction | accepted Escape | RGB 是 analytic orientation preview，不是 spectrum |
| travel time | 直接 | 当前 profile 为以 $M$ 无量纲化的 coordinate-time duration |
| scene-linear result | output seam | f32、tone map/display/UI 之前，按 channel model 解释 |
| invariant diagnostics | 直接 | drift 不是独立误差 certificate |

下列字段不能靠固定 record 得到：event bracket width、localized affine parameter、terminal state、逐步 min/max step、ordered checkpoints、Jacobi field、parity/footprint、独立 high-precision certificate。它们需要改变 solver state、邻 ray 或独立 oracle；当前不得用零值伪装。

## 接纳证据与恢复条件

当前自动化证据包括：

1. Rust size/alignment/offset 固定为 32/96 byte，逻辑资源为 224 byte；
2. 大于 `0xffff` 的 branch count 与负 winding 完整 round-trip；numerical-failure 非零 placeholder 被拒绝，`Uncertain` 不暴露 provisional branch；
3. canonical bolometric Surface 对独立 reference 比较 exact branch、anchor、$g$、travel time、f32 radiance、event residual 与 drift；
4. analytic Escape 保持 non-spectral kind；blackbody plan 返回 f32 scene-linear bands并满足既有 spectral budget；
5. production renderer resources 与 crate public interface 不含 inspection。

尚未闭合 source edge、Surface/Escape boundary、不同 winding/higher-order branch、critical curve 两侧、正负 spin 连续字段 corpus，以及 resize/suspend/device-error 的 production lifecycle。CPU/GPU agreement仍不是物理证明；所有正式接纳域继续受[验证合同](../validation.md)约束。

只有至少两个真实 consumer 证明同一 Interface，才重新设计 production generation、取消、排队、error 与 artifact identity；只有独立 quality policy 实际存在，才扩展 quality Interface。恢复 `@workgroup_size(1)` 必须在 Metal/Vulkan 的完整 record 上消除上述字段反例，不能只比较 terminal 或 radiance。若连续 observable 超预算，应收窄支持域、refine/fallback 或返回 typed uncertainty，不放宽阈值。

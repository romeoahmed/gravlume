# 有界单样本 GPU 路径审计

本文研究并记录 production 中按需审计一个像素样本的最小 GPU seam，回答 ABI、dispatch、readback、identity、取消与可观测量边界。它保存已采用设计及反证，不定义 production interface；当前实现事实见 [GPU 证据](../gpu-renderer.md)，权威产品要求仍是[路线图](../roadmap.md)，数值与物理阈值仍以[验证合同](../validation.md)和[数学物理合同](../physics.md)为准。

**状态：第一版已采用；`@workgroup_size(1)` 子候选已被 Metal 反例否决。** 研究以仓库 revision `9f39b8d798d3889ecb3032b5dcc92ad64103c6ad`为源码基线，实现在 Rust 2024、`wgpu 30.0.0` 与 Apple M5 Metal 上完成 TDD 接纳；依赖事实来自该 revision 的 [`Cargo.toml`](https://github.com/romeoahmed/gravlume/blob/9f39b8d798d3889ecb3032b5dcc92ad64103c6ad/Cargo.toml)和 `wgpu 30` 官方文档。本记录没有引入新物理恒等式，故未为此新增推导；实现后用锁定的 `uv` 环境复跑 `verify_scalar_transport.py`，$g^4$、blackbody shift、slab limits、80-digit fixture oracle 与 LUT midpoint scan 全部通过，LUT 最大绝对误差为 `2.52468526986e-6`。

## 问题与可否证假设

路线图要求 inspection 与画面绑定同一 observation/generation/profile，返回 typed termination、exact branch、source anchor、frequency ratio、travel time、scene-linear radiance 与 event/invariant diagnostics，同时不让默认画面常驻全分辨率 G-buffer。当前 test-only [`capture_trace_sample`](https://github.com/romeoahmed/gravlume/blob/9f39b8d798d3889ecb3032b5dcc92ad64103c6ad/crates/gravlume-render/src/gpu_capture.rs)虽然只选择包含目标像素的一个 tile，却仍：

- dispatch 一个 `8×8` workgroup，即运行 64 条 ray；
- 按完整 observation extent 分配四个 `16 B/pixel` record plane 和 HDR texture；
- 复制并 map 全 extent，而不是一个 record；
- 在 [`trace_capture.wgsl`](https://github.com/romeoahmed/gravlume/blob/9f39b8d798d3889ecb3032b5dcc92ad64103c6ad/crates/gravlume-render/src/shaders/trace_capture.wgsl)中把两个 branch count 饱和压成各 16 bit，不能作为 production 的 exact branch ABI。

可否证假设是：**不改动 `trace_pixel_at`、RK4 step policy 或 event state machine，只增加单活跃 ray request、plan-specific radiance sink、一个定长 record 与异步 readback，即可闭合第一版 production inspection。** 若实现必须创建 extent-scaled diagnostics、改变 solver uniform、丢失完整 `u32` branch key，或相同 sample/profile 不能通过现有 observable budget，本候选即失败。该主假设通过；“workgroup 必须缩成一个 invocation”的更窄子假设被实机反例否决。

## 方法

- 逐项读取当前 TracePlan、shader composition、test-only capture、publication generation、timestamp readback 与 scientific export 的源码，以 commit permalink 固定实现证据；
- 只用 WGSL/WebGPU 现行规范、`wgpu/naga 30` 与 Rust/bytemuck 官方文档推导 layout、dispatch、usage 和 mapping 语义；
- 把 shader 已返回的 `GeometricSample` 字段与路线图交付逐项对照，另把 plan-specific transport 视为 output seam，不从截图或 test helper 的私有 packing 反推 production contract；
- 通过先失败的 canonical surface test、ABI/状态机单测、三种 TracePlan 的真实 GPU round-trip 与严格 lint 接纳实现；不把逻辑 byte 账本或一次功能测试外推成性能数字。

## 现有实现能提供什么

[`GeometricSample`](https://github.com/romeoahmed/gravlume/blob/9f39b8d798d3889ecb3032b5dcc92ad64103c6ad/crates/gravlume-render/src/shaders/trace_protocol.wgsl)已经 invocation-local 地携带 termination、failure flags、event candidates、step count、event residual、三个 source coordinates、travel time、四项 maximum invariant drift 与四 lane branch key；[`trace_pixel_at`](https://github.com/romeoahmed/gravlume/blob/9f39b8d798d3889ecb3032b5dcc92ad64103c6ad/crates/gravlume-render/src/shaders/geodesic_integration.wgsl)已经接受显式 pixel、extent 与 subpixel。因此 inspection 不需要新 solver，也不需要保存 trajectory。

Surface terminal 的三个 source lanes 已由 [`surface_source_at`](https://github.com/romeoahmed/gravlume/blob/9f39b8d798d3889ecb3032b5dcc92ad64103c6ad/crates/gravlume-render/src/shaders/geodesic_events.wgsl)定义为 `(r/M, phi, g)`；其中频率比的物理定义是

\[
g=\frac{-p\cdot u_{\rm obs}}{-p\cdot u_{\rm em}}.
\]

这一定义及 invariant transfer 来自 curved-spacetime kinetic theory；bolometric source 的最终强度使用 $g^4$，而 spectral channels 必须保留自己的 channel interpretation。[Lindquist 1966](https://doi.org/10.1016/0003-4916%2866%2990207-7)与 [Younsi、Wu、Fuerst 2012](https://doi.org/10.1051/0004-6361/201219599)是这里的物理一手来源。现有 bolometric/blackbody transport 分别已经在 [`bolometric_surface_preview.wgsl`](https://github.com/romeoahmed/gravlume/blob/9f39b8d798d3889ecb3032b5dcc92ad64103c6ad/crates/gravlume-render/src/shaders/bolometric_surface_preview.wgsl)和 [`blackbody_surface_preview.wgsl`](https://github.com/romeoahmed/gravlume/blob/9f39b8d798d3889ecb3032b5dcc92ad64103c6ad/crates/gravlume-render/src/shaders/blackbody_surface_preview.wgsl)形成最终 f32 transport result，再写入 `RGBA16F`。

因此第二个真实 consumer 只证明在 **output seam** 提取共享的 plan-specific radiance function；它不证明需要 solver trait、render graph 或通用 record framework。

## 决策：一个 deep module、一个在途 request

采用内部 inspection deep module，Interface 只覆盖三项行为：提交一个绑定当前 published generation 的 sample、逻辑取消该 request、在 renderer 现有 non-blocking `poll` 中取得一次 typed completion。该 module 隐藏 pipeline、request/output/readback buffers、bind group、callback channel 与状态机；不为唯一的 wgpu implementation 引入 trait 或 adapter seam。

第一版资源上限固定为 `N_MAX = 1`。拒绝并发第二个 request，返回 typed `Busy`，比隐含覆盖 request buffer 或无界排队更可证。将来若出现真实的 bounded-region consumer，可在不改变 artifact 语义的前提下把同一 112-byte record 组成数组并通过性能证据选择一维 workgroup；第一版不预先公开 batch、queue 或 quality-policy 抽象。

采用的外部语义是：

1. request 必须携带 caller 看到的 published generation、pixel 与 subpixel；module 对当前 published extent 做边界/finite 验证，generation 不同立即返回 typed mismatch；
2. module 生成 opaque `RequestId`，并自动绑定 exact packed observation words、published extent/generation、`gpu-ks-rk4-v1` profile、producer/domain tag 与 channel model；没有 canonical serialization 前，不把临时 hash 宣称为稳定 observation identity；
3. completion 是 `Completed(artifact) | Cancelled | Superseded | Failed(error)` 的同一组 typed 语义；数值 failure/uncertainty仍是 artifact 内的 typed terminal，不伪装成 host API error；
4. 另一次 science-quality trace 必须获得新的 `RequestId` 和不同 profile identity，不能临时改写 presentation `TraceUniforms::step_policy`。当前只有一个 production GPU policy，因此第一版不暴露尚无 implementation 的 `Quality` 枚举。

## Host-shareable ABI

WGSL 的 uniform/storage 数据在 CPU 与 GPU 间不重排；双方若对 layout 理解不同是 dynamic error。`u32/f32` 的 size/alignment 是 `4/4`，`vec4<u32>` 与 `vec4<f32>` 是 `16/16`，structure size 向最大 member alignment 取整，array stride 是 element size 向 element alignment 取整。[WGSL host-shareable types](https://www.w3.org/TR/WGSL/#host-shareable-types)、[alignment and size](https://www.w3.org/TR/WGSL/#alignment-and-size)和 [address-space layout constraints](https://www.w3.org/TR/WGSL/#address-space-layout-constraints)给出完整规则。Uniform 对 array/nested structure 还有额外 16-byte 约束；device 的 dynamic binding offset alignment 是另一项 limit，不能拿来推导 structure layout。[WebGPU limits](https://www.w3.org/TR/webgpu/#limits)

第一版只使用 16-byte scalar-array blocks，不使用 host-shared `vec3`、`bool`、Rust enum 或 implicit padding：

### Request：48 bytes

| offset | WGSL / Rust block | 语义 |
| ---: | --- | --- |
| 0 | `vec4<u32>` / `[u32; 4]` | pixel x/y、published width/height |
| 16 | `vec4<f32>` / `[f32; 4]` | subpixel x/y、reserved = 0 |
| 32 | `vec4<u32>` / `[u32; 4]` | request id low/high、generation low/high |

### Result：112 bytes

| offset | WGSL / Rust block | 语义 |
| ---: | --- | --- |
| 0 | `vec4<u32>` / `[u32; 4]` | echo request/generation identity |
| 16 | `vec4<u32>` / `[u32; 4]` | ABI version、producer tag、domain tag、output/source kind |
| 32 | `vec4<u32>` / `[u32; 4]` | termination、flags、steps、event candidate bits |
| 48 | `vec4<u32>` / `[u32; 4]` | radial turnings、equatorial crossings、bitcast signed winding、initial polar side |
| 64 | `vec4<f32>` / `[f32; 4]` | 三个 source-coordinate / escape-direction lane、travel time |
| 80 | `vec4<f32>` / `[f32; 4]` | final scene-linear RGB、event residual |
| 96 | `vec4<f32>` / `[f32; 4]` | null、energy、$L_z$、Carter maximum drift |

该 result 的 alignment、size 与 storage-array stride 都是 `16, 112, 112`；完整 branch key 不压缩。`output/source kind` 决定 `source/direction` 和 RGB 是否有物理意义：只有 accepted surface terminal 能构造 `SourceAnchor + FrequencyRatio + SurfaceRadiance`；Escape lanes 是 direction，analytic sky RGB 只能标为 orientation preview；Horizon/failure/Uncertain 不得把零或 failure marker 暴露成 physical radiance。

Rust DTO 使用 `#[repr(C, align(16))]`、同序 `[u32;4]/[f32;4]`、`Pod + Zeroable`，并对 size、alignment、每个 offset 做 compile-time/test assertion。Rust Reference 只对 `repr(C)`规定 declaration-order offset 算法，[Rust type layout](https://doc.rust-lang.org/reference/type-layout.html#the-c-representation)是 host 依据；`bytemuck::Pod` 要求无任何 padding/uninitialized byte、任意 bit pattern 合法且字段本身为 `Pod`，[`Pod` safety contract](https://docs.rs/bytemuck/1.25.2/bytemuck/trait.Pod.html#safety)是 byte-cast 依据。WGSL 仍须经 Naga 30 parser/validator 与真实 GPU round-trip 验证，不能仅凭 Rust `size_of` 推断 shader layout。[Naga 30 validator](https://docs.rs/naga/30.0.0/naga/valid/index.html)

Identity echo 不替代 host provenance；它只检测错误 slot、stale bytes 或 request/context 错配。Observation/profile/channel metadata 由同一个 compiled input 原子附加，避免把字符串和 64-bit identity塞进 WGSL。ABI tag 未识别时返回 `UnsupportedRecordVersion`，不能尽力猜测。

## Dispatch、内存访问与向量边界

最初候选使用 `@workgroup_size(1)`；它在 Apple M5 Metal 上能返回正确 terminal、source 与 radiance，却把同一 canonical sample 的 travel time、maximum drift 和部分 branch fields 返回为零。既有 `8×8` capture 对同一 fixture 返回 `54.902447 M` travel time、positive initial side 与非零 drift；inspection 恢复相同 workgroup shape 后也恢复这些字段。规范不承诺该差异，故这里只把它记录为当前 compiler/backend 的可复现反证，不猜测未证明的驱动根因。

已采用 entry point 使用：

```wgsl
@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn inspect_one(@builtin(local_invocation_index) local_index: u32) {
    if local_index != 0u {
        return;
    }
    let sample = trace_pixel_at(request.pixel, request.extent, request.subpixel);
    result = inspect_result(sample);
}
```

host 固定 `dispatch_workgroups(1, 1, 1)`。WebGPU 明确规定 dispatch 参数是 workgroup 数，实际 invocation 数是 workgroup count 与 `workgroup_size` 的乘积；因此硬件启动 64 个 invocation，但只有 lane 0 建立 ray state 并执行完整 solver，其余 63 个在 solver 前返回。它与 test helper 的 64 条 ray 不同，仍只追一条 ray。[WebGPU `dispatchWorkgroups`](https://www.w3.org/TR/webgpu/#dom-gpucomputepassencoder-dispatchworkgroups)与 [`wgpu 30 ComputePass::dispatch_workgroups`](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePass.html#method.dispatch_workgroups)

一条 geodesic 的 RK/event loop 本身串行，且 sample 间没有共享 reduction 或邻域通信；第一版不需要 `var<workgroup>`、barrier、atomic append 或 indirect dispatch。WGSL 不保证不同 workgroup 并发、启动顺序或具体 subgroup width，[compute workgroup execution](https://www.w3.org/TR/WGSL/#compute-shader-workgroups)也不承诺 `vec4` 一定映射为某种 vendor SIMD。因此保留 `8×8` 是 correctness specialization，不是吞吐优化；16-byte blocks 的理由是可验证 ABI 和连续 store，也不是未经 profiler 证明的“向量化加速”。现有 solver 内部的 `vec4` RK/time/invariant 算术保持不变。

如果未来 `N_MAX > 1`，每 invocation 仍独占一个 request/record，连续 index 写连续 112-byte records；workgroup size 只能作为 internal pipeline constant 经 Metal/Vulkan correctness gate 与 benchmark 选择。任何选择都必须满足 requested device 的 `max_compute_invocations_per_workgroup`、各维 workgroup size 与 `max_compute_workgroups_per_dimension`；`Device::limits()`只返回 request_device 时实际请求到的 limits，不等于 adapter 的潜在上限。[`wgpu 30 Limits`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Limits.html)与 [`Device::limits`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Device.html#method.limits)

## Buffer usage、copy 与 map 生命周期

固定资源为：

| 资源 | bytes | usage |
| --- | ---: | --- |
| request | 48 | `UNIFORM | COPY_DST` |
| result | 112 | `STORAGE | COPY_SRC` |
| readback | 112 | `COPY_DST | MAP_READ` |

新增逻辑 buffer 账本合计 272 bytes，与 extent 无关；这不是 driver heap/真实显存占用声明。现有 176-byte `TraceUniforms`和 plan-specific blackbody LUT 继续由 compiled TracePlan 复用。Portable WebGPU 要求 `MAP_READ` 只能和 `COPY_DST`组合，所以不能把 storage result 本身直接 map；producer 与 staging readback 必须分离。[WebGPU buffer usages](https://www.w3.org/TR/webgpu/#buffer-usage)和 [`wgpu 30 BufferUsages`](https://docs.rs/wgpu/30.0.0/wgpu/struct.BufferUsages.html)

每个 request 的有序生命周期是：

1. `Queue::write_buffer`把 48-byte request 复制到 staging；它在下一次 `Queue::submit`、显式 commands 之前执行。[`wgpu 30 Queue::write_buffer`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html#method.write_buffer)
2. 同一个 encoder 依次记录 single-active-ray compute 与 `copy_buffer_to_buffer(result -> readback, 112)`。WebGPU command list 在 queue timeline 按记录顺序执行；dispatch 与 copy 是不同 usage scopes，无需 shader barrier或手写 backend transition。[WebGPU command buffers](https://www.w3.org/TR/webgpu/#command-buffers)与 [synchronization](https://www.w3.org/TR/webgpu/#synchronization)
3. `CommandEncoder::map_buffer_on_submit`把 map 安排在 producing submission 的 copy 之后；callback 只向 channel 发送 `Result`，不做 decode 或日志重活。[`wgpu 30 map_buffer_on_submit`](https://docs.rs/wgpu/30.0.0/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit)
4. interactive event loop 使用 `Device::poll(Poll)`驱动 callback；只有显式阻塞 export/test 才使用绑定 `SubmissionIndex` 的 `PollType::Wait`。[`Device::poll`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Device.html#method.poll)与 [`PollType`](https://docs.rs/wgpu/30.0.0/wgpu/type.PollType.html)
5. callback 成功后只读取 112-byte mapped range，先校验 ABI/identity，再 decode；mapped view drop 后调用 `unmap`，最后释放 slot。Mapped buffer 与 GPU 使用互斥，且 callback 必须由 submit/poll 驱动。[`BufferSlice::map_async` / mapping contract](https://docs.rs/wgpu/30.0.0/wgpu/struct.BufferSlice.html#method.map_async)与 [WebGPU buffer mapping](https://www.w3.org/TR/webgpu/#buffer-mapping)

112-byte copy 的 offsets 与 size 都是 4 的倍数，满足 WebGPU buffer-copy validation。[WebGPU buffer copy commands](https://www.w3.org/TR/webgpu/#buffer-copy) 分配前仍检查 request/result binding size、buffer size、每 stage storage binding 数与 bind-group binding 数；小尺寸不是跳过 admission 和 typed GPU allocation error 的理由。

## Generation、request identity 与取消

`RequestId`、`published_generation` 与 `SubmissionIndex`是三个不同概念：前两者是产品 identity，后者只是 wgpu 某次 queue submit 的 completion handle。[`wgpu 30 SubmissionIndex` source](https://docs.rs/wgpu/30.0.0/src/wgpu/api/queue.rs.html)只承诺标识 submit 并供 `Device::poll`等待，不能替代 observation/profile identity。

采用的状态机：

```text
Idle
  -> Submitted { context, cancelled: false, map_receiver }
  -> Ready
  -> Completed | Cancelled | Superseded | Failed
  -> Idle
```

- request 绑定**当时已发布** scene 的 generation 与 extent，而不是可能仍在计算的 installed extent；这样点击的像素与用户看到的完整 frame 同代。
- completion 前若 published generation、ObservationId 或 profile 已改变，必须产出 `Superseded`，不能把旧 record 附到新图；record 仍要完成 map/unmap 清理。
- 第一版不设置 host queue；用户取消已提交的 request 时只把 context 标为 cancelled。`wgpu 30`公开 queue interface 提供 submit、completion callback 和 poll，没有逐 submission abort operation；因此“提交后逻辑取消、完成后丢弃”是由官方 Interface 推导出的实现选择，不是 GPU 抢占保证。[`wgpu 30 Queue`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html)与 [`Queue::on_submitted_work_done`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html#method.on_submitted_work_done)
- Device/map/protocol error 即使发生在已取消 request 上也必须作为 `Failed`上报，不能被 cancellation 吞掉；否则设备健康状态会丢失。
- 同一 sample 的重试生成新 RequestId。相同 generation 不表示相同 attempt，相同 pixel 也不表示相同 subpixel/profile。

Fresh on-demand trace 是“绑定旧 frame identity 的同输入重算”，不是从已经结束的 shader invocation 取回历史局部变量；artifact 的 producer tag 明确为 `OnDemandFullKerrSchildRetrace`。Surface production 当前本来就是 full KS，因此可以比较相同 profile observable，但只按验证预算声称 agreement，不承诺跨 submission bitwise identity。若未来需要已发布 `RGBA16F` 的 exact texel representation，应另做明确的一 texel copy；不能把 f32 inspection radiance 误称为 texture bits。

## 不改 solver 可导出的 observable

| observable | 第一版可用性 | 依据与限制 |
| --- | --- | --- |
| typed termination / failure flags | 直接 | `GeometricSample.termination/flags`；host checked discriminant mapping保留 unknown code |
| event candidates / ambiguity / residual | 直接 | candidates bitset、`countOneBits > 1` 与 normalized residual 已存在 |
| exact branch key | 直接 | 四个完整 `u32` lanes；signed winding 用 bitcast round-trip，不采用 test-only 16-bit packing |
| surface source anchor | terminal-dependent | accepted Surface 时 `(r/M, phi)`；host 用 artifact mass 恢复 radius units |
| Frequency Ratio | terminal-dependent | accepted Surface 时 source lane `z = g`；不能对 Escape/Horizon解释 |
| travel time | 直接 | 当前 GPU 为 `M` 无量纲 coordinate-time duration；artifact 必须带 mass/profile |
| scene-linear radiance | output seam | 调用现有 plan-specific transport function，记录 display/tone-map/UI 之前 f32 RGB与 channel model |
| invariant diagnostics | 直接 | maximum null、energy、$L_z$、Carter drift；drift 不是独立误差 certificate |
| producer/domain/source kind | host + shader tag | compiled plan 给 producer/profile/channel；terminal 决定 source kind |

下列字段不能靠“多写一个 record”得到：event bracket width、localized affine parameter、terminal state、逐步 min/max step、ordered path checkpoints、Jacobi field、parity/footprint、独立 high-precision error certificate。它们需要改变 solver state、额外邻 ray 或独立 oracle；第一版必须显式返回 unavailable，而不是用零填充伪装。Surface 以外也没有 SourceAnchor，spatially varying volume/slow-light 仍没有 ordered path evidence。

## TDD 结果与剩余门槛

实现先以缺少 production types/helper 的 canonical surface test 进入 RED，再完成以下 GREEN evidence：

1. **Host protocol：** pixel/subpixel boundary、无 published scene、generation mismatch、Busy、wrong RequestId、cancel/cleanup/reuse 与 generation supersession 均有 typed test。
2. **ABI：** Rust size/alignment/offset 固定为 48/112 byte，逻辑资源为 272 byte；大于 `0xffff` 的 branch sentinel 保持完整 `u32` round-trip，ABI/producer/domain/identity 均 checked decode。
3. **真实 GPU：** canonical bolometric Surface 对独立 reference 比较 termination、exact branch、anchor、$g$、time、radiance、event residual 与 drift；analytic Escape 保持 non-spectral kind；blackbody inspection 返回 f32 scene-linear bands并满足既有 spectral budget。
4. **调度反证：** Apple M5 Metal 上 `@workgroup_size(1)` 的字段丢失 test 先失败；一个 `8×8` workgroup、单活跃 solver lane 的同形 specialization 通过同一 observable test。每次仍只 copy/map 112 byte。

尚未完成的路线图门槛是 source edge、Surface/Escape boundary、不同 winding/higher-order branch、critical curve 两侧、正负 spin 的连续字段 corpus，以及 resize/suspend/device-error 的完整桌面 lifecycle 矩阵；因此这里只接纳基础切片，不宣称整节路线图退出。

CPU/GPU agreement 仍不是物理证明；Frequency Ratio、radiance、event 与 branch 必须继续满足现有独立 reference 和收敛 ladder。Carter constant 作为可观测诊断的物理基础来自 [Carter 1968](https://doi.org/10.1103/PhysRev.174.1559)，但小 invariant drift 不能证明 branch 正确。

## 决策与恢复条件

**已接纳的最小 slice：** production single-sample request，固定 48/112/112-byte resources，一个与 presentation 同形的 `8×8` workgroup且仅 lane 0 进入 solver，复用不变的 full-KS `trace_pixel_at`和 plan-specific transport，non-blocking map-on-submit，opaque request identity，generation-gated typed completion。它在 renderer 内形成一个 deep module，同时完全不改变 default frame resources。

**明确不做：** full-frame record planes、small-region batch、inspection UI、science-quality第二 solver policy、reconstruction/history、active queue、trajectory checkpoint 与通用 solver Interface。

只有出现真实 bounded-region consumer，并以 correctness-gated Metal/Vulkan benchmark 证明 `N_MAX > 1` 改善端到端 latency/throughput时，才重开 batch/workgroup tuning；恢复 `@workgroup_size(1)` 还必须在 Metal/Vulkan 的完整 observable record 上消除上述字段反例，不能只比较 terminal/radiance。只有独立 quality policy 实际存在时，才扩展 quality Interface。若单样本 result 与相同 profile reference 超预算，应收窄 support/返回 uncertainty，而不是放宽阈值或把 inspection 接到 test-only footprint step policy。

# Production 按需单样本检查研究

本文研究如何把现有 test-only 单样本 GPU record 收敛为有真实消费者的 production 能力。它是**研究记录与候选设计，不定义 production interface**；最终行为仍以[路线图](../roadmap.md)、[架构合同](../architecture.md)、[数学物理合同](../physics.md)和[验证合同](../validation.md)为准，采用后的事实应回写这些权威文档。

**状态：已采用“单槽、绑定已发布 generation、异步 drain”的最小 seam，并由桌面点击消费。** 首个真实消费者要求解释屏幕像素，因此 adopted variant 还复制实际 published texel；连续字段 corpus 与第二质量政策仍开放。本文只研究单个 `ImageSample`，不授权小区域 batch、全帧 G-buffer、第二质量策略、reconstruction 或通用调试 UI。

## 问题、方法与结论

[路线图的下一项 resolved work](../roadmap.md#有界样本审计与质量基线)要求用户和验证工具取得与画面同代、同 profile 的结构化物理证据，同时明确资源上限、读取范围、generation、一致性、取消和错误语义。研究检查了：

- 当前 [`trace/inspection.rs`](../../crates/gravlume-render/src/trace/inspection.rs)及其三个 WGSL adapter、真实 GPU tests 与[既有 test-only 审计](bounded-sample-inspection.md)；
- [`renderer.rs`](../../crates/gravlume-render/src/renderer.rs)、[`renderer/frame.rs`](../../crates/gravlume-render/src/renderer/frame.rs)、[`display.rs`](../../crates/gravlume-render/src/display.rs)、[`timing.rs`](../../crates/gravlume-render/src/timing.rs)和[`scientific_capture.rs`](../../crates/gravlume-render/src/scientific_capture.rs)的 publication/readback 生命周期；
- workspace `Cargo.lock` 锁定的 wgpu `30.0.1`，以及 W3C WGSL/WebGPU、wgpu、Rust 的一手规范；
- Carter separability、相对论辐射传输与 Kerr 高阶像的原始文献。

结论是：现有 32-byte request、96-byte record 和 full-KS inspection shader 已经是合适的内部深模块核心；production 不需要 solver trait、render graph、active queue 或新的全帧 plane。缺少的不是另一套追迹器，而是一个把 host identity、原子发布、实际 texel copy 与异步 readback 绑定起来的单槽状态机。

最关键的语义边界是：record 中的 `f32` scene value 是**一次 fresh full-KS/WGSL-binary32 re-trace 的求值结果**，不自动等于已发布 `Rgba16Float` texture 的实际 texel。当前 accelerated sky 帧可使用 escape map 与选择性 subpixel shadow refinement，而 inspection adapter 始终执行单个 full-KS sample；所有 plan 的最终 texture 还经历 binary16 量化。[GPU 实现证据](../gpu-renderer.md#publication-与-display)和[scientific capture 合同](../architecture.md#renderer-interface)只把已发布 texture readback 视为最终 texel 证据。因此 adopted result 同时携带独立的 `published_texel` 与 `evaluated_scene_value`，后者报告 `FullKerrSchildRetrace` producer；接口不声称两者 bit-equal。

## 科学输出模型

### 身份先于数值

每个结果至少绑定以下 host-owned identity：

```text
request id
+ immutable observation id
+ target published generation and extent
+ logical ImageSample and effective WGSL-binary32 input domain
+ fixed profile id: gpu-ks-rk4-v1
+ producer: fresh full Kerr-Schild retrace
+ arithmetic domain: WGSL binary32
```

`generation` 只在同一 renderer/observation 生命周期内有意义，不能替代 observation identity。Live UI 可使用 renderer 创建时分配的 opaque `ObservationId`；若结果持久化为 scientific artifact，还必须记录 canonical observation、producer revision、shader digest、adapter/backend 和实际 resource counters，不能把进程内 ID 或临时 float hash 冒充内容身份。[验证合同的 artifact 规则](../validation.md)要求 identity collision 被显式拒绝。

Request ID 由 host 单调分配并绑定 pending submission。Observation/profile/producer/generation 是 host 资源所有权事实，不应塞进 GPU record 做“回显证明”：回显不能证明 shader 真正使用了哪组 uniform，反而扩大 ABI。持久 record 在 dispatch 前清零、zero termination 保持非法，再由严格 decoder 拒绝 stale/partial record，才是防止旧数据误解码的有效边界。

### Terminal-specific observable

Public result 应使用 discriminated union 表达 terminal-specific 数据；private GPU DTO 继续保持 flat `vec4` lanes。这样 `NumericalFailure`/`Uncertain` 不可能携带可用 branch，Escape 不可能携带 surface `g`，普通 RGB 也不会被误读成光谱辐亮度。

| 量 | 可解释语义 | 必须显式的限制 |
| --- | --- | --- |
| typed termination | horizon、escape、equatorial surface、singularity guard、step exhaustion、numerical failure、uncertain | singularity guard 与 step exhaustion 是数值边界，不是物理 source；event 定义以[物理合同](../physics.md#6-event-与-terminal-semantics)为准 |
| exact branch key | initial polar side、radial turnings、equatorial crossings、signed azimuth winding | 只对 decoder 能证明的 determinate terminal 返回；`NumericalFailure` 与 `Uncertain` 必须 unavailable。Step exhaustion 的 key 只描述已提交路径前缀，不证明一个物理 source branch |
| source anchor | Escape 的单位方向；surface 的 `(r/M, oblate azimuth)` | finite escape sphere 不是真正的 null infinity；surface anchor 只属于当前 thin equatorial source model |
| Frequency Ratio `g` | `(-p·u_obs)/(-p·u_em)`，只对 accepted surface source 可用 | 它是局部不变量；不能把普通 RGB 当作光谱。`I_nu/nu^3` 不变、bolometric intensity 按 `g^4` 变化来自 [Lindquist 1966](https://doi.org/10.1016/0003-4916%2866%2990207-7) |
| coordinate-time duration | 当前 chart/profile 下，以 `M` 无量纲化的终止时间增量 | 它不是 observer proper time；对 step exhaustion 只是已追迹前缀，字段不应笼统命名成无条件的物理 `travel_time` |
| published texel | 绑定 generation 的实际 `Rgba16Float` binary16 words 与 alpha-tag kind | 这是最终 scene texture 证据，不提供 fresh retrace 的 path provenance；exceptional representation 只继承 WebGPU texel-copy 合同 |
| evaluated scene value | tone map、display encoding 与 UI 之前的 fresh `f32` plan output | surface 才是按 channel model 解释的 physical radiance；analytic sky 是 orientation preview，failure color 是诊断可见值，horizon black 是 presentation 语义 |
| event diagnostics | candidate set、ambiguity、normalized residual | residual 只衡量已选 event 的局部定位；当前 record 不含 bracket width、localized affine parameter 或 terminal state |
| invariant diagnostics | null residual 与 `E/Lz/Carter` maximum drift | Carter 常数来自 Hamilton–Jacobi separability，[Carter 1968](https://doi.org/10.1103/PhysRev.174.1559)；按[物理合同](../physics.md)，小 drift 不能单独证明 near-critical branch、angle 或 travel time 正确 |

Host 应把 raw bitsets、tag 与 reserved lanes 严格验证后，再形成具名 Rust 类型。当前 96-byte record 没有 uncertainty-reason lane，因此只能返回 typed `Uncertain` 与现有 candidate/drift evidence；不能从零值或启发式推断“已知原因”。需要 reason taxonomy 时，必须先让 solver 显式产生并验证该证据。

### 适用域与非声明

单样本结果只继承 `gpu-ks-rk4-v1` 已接纳域，不扩大支持范围。当前 regular continuous evidence 主要是 canonical exterior Kerr surface；source edge、surface/capture boundary、不同 winding/higher-order branch、critical curve 两侧、正负 spin、near-axis 与 near-extreme 尚未闭合。[Kerr 高阶像原始计算](https://doi.org/10.1086/152223)和[现代 photon-ring/critical-curve 分析](https://doi.org/10.1103/PhysRevD.108.064043)说明这些 branch 是物理上真实的离散结构，不得靠相邻颜色平滑或低 drift 猜测。

当前 record 也不证明 event bracket、Jacobi/parity、source footprint、ordered path checkpoints、独立 high-precision certificate 或第二 science-quality GPU policy。CPU/GPU agreement 是必要的交叉实现证据，但独立 reference、high-precision case 与 convergence 仍由[验证合同](../validation.md)规定。

## 最小深模块 interface 决策

Adopted seam 让 `Renderer` 自己捕获当前 observation/publication identity，caller 只提交 validated sample；以下只表达语义形状，精确 public signature 以源码为准：

```rust
impl Renderer {
    fn request_sample_inspection(
        &mut self,
        sample: ImageSample,
    ) -> Result<SampleInspectionId, SampleInspectionRequestError>;

    fn cancel_sample_inspection(
        &mut self,
        id: SampleInspectionId,
    ) -> bool;
}

enum SampleInspectionEvent {
    Completed(SampleInspection),
    Cancelled(SampleInspectionIdentity),
    Superseded(SampleInspectionIdentity),
    Failed {
        identity: SampleInspectionIdentity,
        error: SampleInspectionFailure,
    },
}
```

Completion 应随既有 `RendererUpdate` 被 `poll` 汇总，而不是增加一个会阻塞 event loop 的同步 API。外部只学习 `Renderer` 的 request/cancel/update seam；private `SampleInspector` 独占 pipeline、bind group、fixed buffers、map callback channel、decode 和状态转移。它直接复用 sealed `TracePipeline` 的 uniform、plan 与 LUT，不暴露 `wgpu` handle、solver、pass 或 shader record。

第一版只有一个真实 profile，因此 request 不接受 quality/profile 参数；结果固定报告 `gpu-ks-rk4-v1`。未来真正存在 science-quality 执行时，它必须有自己的 profile 和新 request ID，不能在同一 request 下静默重追或改变 step policy。这符合[路线图的质量政策要求](../roadmap.md#有界样本审计与质量基线)，也避免只有一个 variant 的假扩展 seam。

Request admission 在任何 GPU mutation 前完成：

1. 必须已有 published scene，且其 generation 等于当前 `ExtentTracker` generation；窗口正在构建 replacement 时 typed 拒绝新请求，避免把当前 window pixel 默认为旧 texture coordinate；
2. renderer 捕获自己的 immutable observation identity 与 `PublishedScene::generation()/extent`，不让 caller 回显这些 ownership facts；
3. pixel 必须落在该 published extent，subpixel 必须是 domain 已验证值并可进入 WGSL binary32；
4. 固定 slot 必须为空，否则返回包含 active request ID 的 `Busy`；
5. pipeline/buffer creation 由 renderer init error scope 捕获；mapping 与 decode failure 经 completion event 保留 typed source chain，production 不使用 `expect`。

## Generation 与取消状态机

Renderer resize 时继续显示上一张完整 scene，而新 generation 在 hidden candidate 中追迹；只有完成且 generation 匹配的 candidate 才原子发布。[当前 publication 合同](../gpu-renderer.md#publication-与-display)意味着 inspection 必须针对用户实际看见的 `PublishedScene`：

```text
Idle
  └─ accepted request(target = published G) → Submitted(G)
Submitted(G)
  ├─ cancel(id) → DiscardAfterDrain(G)
  ├─ map/decode complete, published still G → Completed → Idle
  ├─ map/decode complete, published changed → Superseded → Idle
  └─ map/decode failure → Failed → Idle
DiscardAfterDrain(G)
  └─ submission/map settles, unmap/discard → Cancelled → Idle
```

WebGPU submission puts recorded commands on the queue timeline；[`GPUQueue.submit`](https://www.w3.org/TR/webgpu/#dom-gpuqueue-submit)与[buffer mapping](https://www.w3.org/TR/webgpu/#buffer-mapping)只在 GPU 使用结束后允许 CPU mapping，且 API 没有定义已提交 command buffer 的 portable preemption seam。因此 `cancel` 只能设计为逻辑取消：标记结果丢弃，但 slot 在 callback 到达、mapped view drop 且 buffer `unmap` 前仍为 Busy。这样不会复用仍被 GPU/CPU 占有的 buffer，也不会制造“已取消所以资源已空闲”的假象。

同一次 `Renderer::poll` 若既发布新 scene 又完成旧 inspection，必须先安装新 publication，再比较 inspection identity；旧结果只能发出 `Superseded`。当前 renderer 在成功 resize 安装新 extent、以及 suspend 时先逻辑取消 active request；drain 后报告 `Cancelled`。旧 scene 虽继续显示，但 resize 期间拒绝新 inspection，因为 desktop cursor 属于新 physical extent，不能未经映射就冒充旧 texture coordinate。Display-only output 更新不改变 observation/publication，因此不使结果失效。

## WGSL 与 host memory contract

### 保留 padding-free `vec4` ABI

现有 request 是两个 `vec4` lane（32 bytes），record 是六个 `vec4` lane（96 bytes）。WGSL 的 `f32/u32` size/alignment 是 `4/4`，`vec4` 是 `16/16`，structure 按最大 member alignment 取整，见 [WGSL alignment and size](https://www.w3.org/TR/WGSL/#alignment-and-size)。Rust side 继续用同序 `[u32; 4]`/`[f32; 4]`、`#[repr(C, align(16))]` 与 compile-time size/offset assertions；[`repr(C)`](https://doc.rust-lang.org/stable/reference/type-layout.html#the-c-representation)固定 declaration-order layout，[`bytemuck::Pod`](https://docs.rs/bytemuck/1.25.2/bytemuck/derive.Pod.html)要求没有 padding。

Host-shared DTO 不使用 `vec3`、`bool`、Rust enum 或 implicit padding。WGSL function-local `vec3/vec4` 仍适合频带和状态算术；“适合向量表达”不等于 storage 是 12 bytes，也不承诺硬件 SIMD。

WGSL `f32` 只服从规范声明的浮点范围与精度，implementation 可以对相关 subnormal 输入/输出 FTZ，overflow/NaN/Inf 中间结果可能成为 indeterminate；见 [WGSL floating-point evaluation](https://www.w3.org/TR/WGSL/#floating-point-evaluation)。因此 arithmetic domain、finite guards 与 typed `NumericalFailure` 是结果合同的一部分，不能用 CPU `f64` 直觉外推。

### 固定资源与 exact read range

第一版只允许一个 pending request，继续使用：

| logical resource | bytes | production usage |
| --- | ---: | --- |
| persistent request | 32 | `UNIFORM | COPY_DST` |
| persistent record | 96 | `STORAGE | COPY_SRC | COPY_DST` |
| persistent readback | 264 | `COPY_DST | MAP_READ` |

总计 392 logical buffer bytes，readback/map range 恰为 264 bytes，与 viewport extent 无关。Record 占 `[0, 96)`，实际 published texel 占 `[256, 264)`，中间空洞满足 texture row-layout alignment。Pipeline、bind group、backend allocation granularity 和 `Queue::write_buffer` 的 native staging allocation 不包含在 392-byte 数字中，不能把它称作 driver 显存峰值；wgpu 明确说明 `write_buffer` 的数据先进入 staging，随下一次 submit 执行，且 native 当前可能产生短生命周期 allocation，见 [`Queue::write_buffer`](https://docs.rs/wgpu/30.0.1/wgpu/struct.Queue.html#method.write_buffer)。一个 32-byte write 且单槽在逻辑上仍有界，但 profiler/resource counter 应实测 driver 开销。权威资源合同见[架构文档](../architecture.md#内存与资源预算)。

Persistent record 在每次 dispatch 前通过 `clear_buffer` 归零，所以它需要 `COPY_DST`；termination zero 必须保持 decoder-invalid。之后同一 encoder 严格记录：

```text
clear 96-byte record
→ one inspection compute dispatch
→ copy exactly 96 bytes record to readback[0..96]
→ copy one bound published texel to readback[256..264]
→ register read mapping on this submission
```

Portable WebGPU 的 `MAP_READ` buffer 只能与 `COPY_DST` 组合；storage record 与 staging readback 必须分离，见 [WebGPU buffer usages](https://www.w3.org/TR/webgpu/#buffer-usage)和 wgpu [`BufferUsages`](https://docs.rs/wgpu/30.0.1/wgpu/struct.BufferUsages.html)。Buffer-to-buffer copy 的 offsets/size 满足 4-byte 对齐；compute、clear、copy 是同一 ordered command list 上相继的 usage scope，不需要 shader-side barrier，见 [WebGPU synchronization](https://www.w3.org/TR/webgpu/#synchronization)与[`copyBufferToBuffer`](https://www.w3.org/TR/webgpu/#dom-gpucommandencoder-copybuffertobuffer)。

[`CommandEncoder::map_buffer_on_submit`](https://docs.rs/wgpu/30.0.1/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit)正好把 mapping 安排在该 encoder 的 GPU work 完成之后；callback 由 submit/device poll 驱动，并应只发送短消息，因为 native poll 会等待 callback 返回。Decoder 在 renderer poll 路径读取 264-byte range；读取后先 drop mapped view，再 `unmap`。Mapped buffer 与 GPU command 使用互斥，[wgpu `Buffer`](https://docs.rs/wgpu/30.0.1/wgpu/struct.Buffer.html#method.map_async)和 [WebGPU mapping](https://www.w3.org/TR/webgpu/#buffer-mapping)都不允许 mapped 状态下重新提交 GPU 使用。

### Published texel 是独立的另一类证据

第一个 UI consumer 的明确问题是“当前屏幕这个像素的实际 scene texture 值是什么”，因此结果另设 `published_texel`，并从 request 捕获的 `PublishedScene` texture 做 1×1 `copy_texture_to_buffer`；它不覆盖或改名 fresh `evaluated_scene_value`。WebGPU 的 [`GPUTexelCopyBufferLayout`](https://www.w3.org/TR/webgpu/#dictdef-gputexelcopybufferlayout)允许单 block-row 省略 `bytesPerRow`，若提供则必须是 256 的倍数；wgpu 的 [`TexelCopyBufferLayout`](https://docs.rs/wgpu/30.0.1/wgpu/struct.TexelCopyBufferLayout.html)遵守同一规则。该 copy 与 map range 已计入上述固定上限。

整帧 scientific capture 仍是显式阻塞 export；interactive click 不应为一个像素复制整帧。单槽把实际 texel 增量固定在 168 bytes 的 row-aligned readback 扩展内，同时保持 generation consistency 不等于 fresh retrace bit equality。

## 并行、workgroup 与向量化边界

当前 `@workgroup_size(8, 8, 1)`、单 workgroup dispatch、只有 `local_invocation_index == 0` 执行 trace，是 Metal 真实反例后的 correctness specialization，不是吞吐声明。[WGSL compute workgroups](https://www.w3.org/TR/WGSL/#compute-shader-workgroups)规定每个 grid point 有一个 invocation，但不保证不同 workgroup 并发、串行或启动顺序；subgroup width 由 device/compiler 选择，且与 `local_invocation_index` 没有固定映射。

因此第一版保持现有 specialization，不宣称 vectorization，也不增加 subgroup、atomic append、workgroup memory 或 active queue。单条 geodesic 的 RK/event state machine 本质上在一个 invocation 内顺序推进；ABI 中的 `vec4` 只是内存布局和局部算术分组。

只有真实 bounded-region consumer 出现后，才评估固定上限的小 batch：一个 invocation 独占一个 request/record index，无共享写、无 barrier、无 atomic；host 在 adapter limits 与项目固定上限内 admission。此时可比较 AoS 96-byte record 与 SoA，但单样本/小固定 batch 会读取完整 record，当前没有证据证明 SoA 或额外 binding 有净收益。任何 batch 都必须保持固定最大 record 数，不能把 inspection 变成 hidden extent-scaled storage。

## 错误语义与 TDD 交付顺序

Request-time typed errors 至少区分：`NoPublishedScene`、`ObservationMismatch`、`GenerationMismatch`、`SampleOutOfBounds`、`Busy { active }` 与 resource creation failure。Completion-time event 至少区分：`Completed`、`Cancelled`、`Superseded`、mapping failure 与 invalid GPU record；unknown termination、unknown flag/tag、non-finite value、nonzero reserved lane 和非法 terminal/source/scene/branch 组合必须拒绝，不返回部分成功。

实现应先增加不需要 GPU 的 failing tests，再接线资源和 shader：

1. identity/admission：无 published scene、错误 observation/generation、越界 sample、单槽 Busy；
2. lifecycle：cancel 后仍 Busy，drain 后恰好一个 `Cancelled`；resize 保留旧 published generation；新 publication 使旧 completion `Superseded`；event 只消费一次；
3. ABI/decoder：32/96 byte request/record size/offset、392 logical bytes、264-byte map range、所有 reserved/unknown/NaN/组合拒绝与 branch counter/winding round-trip；
4. policy preservation：inspection uniform、plan、LUT 与 step policy 来自同一 `TracePipeline`，增加 record sink 不修改 default frame resource plan；
5. real GPU：analytic、bolometric、blackbody 三 plan 的 ordinary samples，加 source edge、capture boundary、critical curve 两侧、different winding/higher-order 和 spin `±a` corpus；
6. independent evidence：同一 immutable observation 上分别比较 terminal/branch、source anchor、`g`、coordinate time、scene value 和每项 diagnostics；不以单一 RGB 或单一 max error 掩盖离散失败。

Interactive 与未来 science-quality policy 对 accepted physical result 使用相同 observable budgets。低成本路径只能收窄 applicability、refine/fallback 或返回 `Uncertain`；不能放宽阈值。若 science trace 是第二次 GPU execution，它必须得到新 request ID、明确 profile/producer，并独立经历 generation/cancel/error 状态机。

## 决策与恢复条件

建议 production 切片采用：

1. `Renderer` 上的 request/cancel/update 三个窄动作；
2. private、renderer-owned、固定 392 logical bytes、最多一个 pending 的 `SampleInspector`；
3. target 绑定 immutable observation 与当前 `PublishedScene` generation/extent；
4. fixed full-KS/WGSL-binary32 profile identity 和严格 typed decode；
5. non-blocking `map_buffer_on_submit` + existing renderer poll；
6. cancellation/supersession 都先 drain，再释放 slot；
7. `evaluated_scene_value` 与实际 `published_texel` 永远分开。

以下候选继续延后：bounded-region public batch、可选 profile 参数、trajectory/checkpoint、active queue、full-frame record planes、通用 inspector trait、render graph 和 reconstruction。重开条件分别是第二个真实 consumer、具名 quality implementation、明确 observable/资源预算与跨 Metal/Vulkan correctness/performance evidence，而不是“以后可能需要”。

## 一手来源

- [WGSL specification](https://www.w3.org/TR/WGSL/)：host-shareable layout、floating-point、compute/workgroup/subgroup execution semantics；
- [WebGPU specification](https://www.w3.org/TR/webgpu/)：buffer usage、copy、mapping、queue timeline 与 synchronization；
- [wgpu 30 API documentation](https://docs.rs/wgpu/30.0.1/wgpu/)及 workspace `Cargo.lock`：实际 Rust API 与锁定版本；
- [Rust Reference: type layout](https://doc.rust-lang.org/stable/reference/type-layout.html)：`repr(C)`、size、alignment 与 field offset guarantees；
- [Carter 1968](https://doi.org/10.1103/PhysRev.174.1559)：Kerr Hamilton–Jacobi separability 与第四守恒量；
- [Lindquist 1966](https://doi.org/10.1016/0003-4916%2866%2990207-7)：general-relativistic transport 与 invariant radiation distribution；
- [Cunningham & Bardeen 1973](https://doi.org/10.1086/152223)：Kerr 强引力场中的多重/高阶光学像；
- [Cárdenas-Avendaño & Lupsasca 2023](https://doi.org/10.1103/PhysRevD.108.064043)：photon ring 与 critical curve 的现代可观测结构。

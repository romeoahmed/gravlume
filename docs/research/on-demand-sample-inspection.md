# Production 按需单样本检查决策

本文记录如何把历史 test-only 单样本 GPU record 收敛为 production inspection，以及采用方案的证据、被拒绝候选和恢复条件；它不定义 public Rust interface、物理 observable 或当前支持域。当前事实分别以源码、[架构合同](../architecture.md)、[GPU 证据](../gpu-renderer.md)和[验证合同](../validation.md)为准。

**状态：最小单槽 seam 已采用，连续字段质量域仍开放。** Desktop 点击是首个真实 consumer；bounded-region batch、第二质量方法、持久 artifact、reconstruction 与通用 inspector interface 均未获授权。

## 问题与结论

历史 [`bounded-sample-inspection.md`](bounded-sample-inspection.md) 已证明：不改变 full Kerr–Schild
RK4、event policy 或 plan-specific transport，就能用固定 record 复算一条 ray。但 test helper 没有回答
publication identity、actual published texel、异步 mapping、resize/suspend cancellation 或 consumer
lifecycle。

采用方案不增加另一套 tracer，而是让 renderer 以一个固定槽位绑定四项事实：

```text
validated ImageSample
+ current published generation / extent
+ fixed full-KS WGSL-binary32 method
+ one bound published texture
```

Request 返回 process-local ticket；`Renderer::poll` 之后恰好返回一次 completed、cancelled 或 typed
failed completion。Completed result 分开携带：

- `published_texel`：从 request 所绑定 generation 实际复制的 `Rgba16Float` texel；
- `fresh_retrace`：同一 logical sample 的 fresh full-KS/WGSL-binary32 typed evidence。

两者不能合并。Presentation 可能使用 escape-map accelerator、shadow subpixel refinement，并最终写入
binary16 texture；inspection retrace 则执行单条 full-KS trace 并保留 binary32 output。数值接近或偶然
bit-equal 都不建立 producer identity。

## 科学证据边界

Ticket 只在当前 renderer process/lifetime 内标识 generation、extent 与 sample，不是 artifact identity。
持久证据还需要 canonical observation、method、producer revision、shader/backend/adapter 和解释 metadata；
路线图在出现真实持久化 consumer 前不冻结 schema。

Public terminal-specific sum type 只形成物理上合法的组合：

| 证据                        | 可解释范围                                                                                 | 明确限制                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| termination                 | horizon、escape、surface、singularity guard、step exhaustion、numerical failure、uncertain | guard/exhaustion 是数值边界，不是 physical source                                      |
| branch                      | determinate terminal 的 initial polar side、turning/crossing counts 与 winding             | failure/uncertain 无 branch；exhaustion 只有已追迹 prefix                              |
| source                      | Escape unit direction 或 surface `(r/M, oblate azimuth)`                                   | finite escape sphere 不是精确 null infinity                                            |
| frequency ratio             | accepted surface 的 $(-p\cdot u_{obs})/(-p\cdot u_{em})$                                   | 普通 RGB 不是 spectrum                                                                 |
| coordinate-time duration    | 当前 chart/profile 下的 `delta t / M`                                                      | 不是 observer proper time；exhaustion 只报告 prefix                                    |
| scene value                 | fresh plan output before display                                                           | 只有 surface channel model 给出 physical radiance；sky/failure 是 preview/diagnostic   |
| event/invariant diagnostics | candidate bits、ambiguity、residual 与 max drift                                           | 当前 record 没有 bracket width、terminal state、Jacobi/parity 或独立 error certificate |

Strict decoder 拒绝 unknown discriminant/flag/tag、non-finite value、非零 reserved lane 和非法
terminal/source/scene/branch 组合，不返回“尽可能多”的 partial success。当前 record 没有 uncertainty
reason lane，因此不能从零值或 heuristic 猜测原因。

## Interface 与生命周期决策

Caller 只学习两个动作：提交一个 validated sample，以及从既有 renderer update 取走 completion。Renderer
自己捕获 publication identity；caller 不回显 generation、选择 solver 或持有 wgpu handle。Resize/suspend
负责取消，没有无 consumer 的 public cancel seam。

```text
Idle
  └─ admit(target = published G) → Submitted(G)
Submitted(G)
  ├─ map/decode success and publication still G → Completed → Idle
  ├─ resize/suspend/new publication invalidates G → DiscardAfterDrain(G)
  └─ map/decode failure while still current → Failed → Idle
DiscardAfterDrain(G)
  └─ GPU/map ownership settles → Cancelled → Idle
```

WebGPU/wgpu 没有已提交 command buffer 的 portable preemption seam。Logical cancellation 只丢弃结果；
槽位在 callback 到达、mapped view 释放且必要的 `unmap` 完成前仍为 Busy。成功 mapping 才读取 mapped
range，并在 drop view 后 `unmap`；mapping failure 直接形成 typed failure，不对 unmapped buffer 再调用
`unmap`。[wgpu `Buffer`](https://docs.rs/wgpu/30.0.1/wgpu/struct.Buffer.html#method.map_async)明确要求 mapped
view 在 `unmap` 前释放，并禁止 CPU/GPU 同时拥有 buffer。

Renderer poll 先安装 matching new publication，再核对 inspection generation。Desktop 进入
`ViewportChanging` 后，旧 completion 不得恢复旧结果；若 resize 是 no-op 或事务式拒绝且 retained
publication 仍 current，viewport wait 必须显式结束。

## GPU protocol 与资源证据

Host/WGSL 继续使用 padding-free `vec4` lanes：request 32 bytes，record 96 bytes。WGSL `f32/u32` 的
size/alignment 为 `4/4`，`vec4` 为 `16/16`；Rust side 使用同序 scalar arrays、`repr(C, align(16))`、
`bytemuck::Pod` 与 compile-time offsets。[WGSL layout](https://www.w3.org/TR/WGSL/#alignment-and-size)与
[Rust `repr(C)`](https://doc.rust-lang.org/stable/reference/type-layout.html#the-c-representation)是布局依据。
Host-shared DTO 不使用 `vec3`、`bool`、Rust enum 或 implicit padding；function-local vector arithmetic
不等于 storage SIMD 声明。

| logical resource    | bytes | usage     |
| ------------------- | ----: | --------- |
| persistent request  |    32 | `UNIFORM  | COPY_DST` |
| persistent record   |    96 | `STORAGE  | COPY_SRC  | COPY_DST` |
| persistent readback |   104 | `COPY_DST | MAP_READ` |

总计 232 logical bytes，与 viewport extent 无关；这不包含 pipeline/bind-group/backend allocation 或
`Queue::write_buffer` staging，也不是 driver memory peak。Readback `[0, 96)` 保存 record，`[96, 104)`
保存一个 `Rgba16Float` texel。

每次提交的顺序固定为：

```text
clear record to decoder-invalid zero termination
→ dispatch one 8×8 workgroup; only local lane 0 traces
→ copy 96-byte record
→ copy one bound published texel with explicit 256-byte row pitch
→ map exactly the 104-byte readback on this submission
```

`map_buffer_on_submit` 在 producing submission 之后安排 mapping，callback 只发送短消息；native poll 会
等待 callback 返回。[wgpu 30.0.1 API](https://docs.rs/wgpu/30.0.1/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit)
明确给出这两个语义。Texture copy 的 `bytes_per_row` 若提供必须满足 256-byte alignment；wgpu 的 copy
size 计算不把最后一行之后的 padding 算入所需 buffer 尾部，因此单行 8-byte texel 不要求分配完整
256 bytes。[`TexelCopyBufferLayout`](https://docs.rs/wgpu/30.0.1/wgpu/struct.TexelCopyBufferLayout.html)

显式 row pitch 不是猜测：锁定 revision 的 Metal HAL 在省略时把零传给 native blit，真实 Apple M5
测试返回不完整 record；显式 256 后恢复。该实现证据见[锁定 wgpu source](https://github.com/gfx-rs/wgpu/blob/40f4a34ebaf56f9a046231f54125ad046239d3f3/wgpu-hal/src/metal/command.rs#L724-L735)。

同一设备上，`@workgroup_size(1)` 曾返回正确 terminal/source/radiance，却把 travel time、drift 和部分
branch fields 置零；恢复 production `8×8` specialization 后完整。因此单样本保持一个 `8×8`
workgroup 且只有 lane 0 执行。它是 backend 反例约束的 correctness specialization，不是吞吐、subgroup
width、parallelism 或 vectorization 声明。

## 被拒绝的扩张

| 候选                                           | 决策         | 重开条件                                           |
| ---------------------------------------------- | ------------ | -------------------------------------------------- |
| public solver/profile parameter                | 延后         | 第二个真实 quality implementation 与明确支持域     |
| bounded-region batch / active queue            | 延后         | 小区域 consumer、固定资源上限和 Metal/Vulkan A/B   |
| public cancel / inspector trait / render graph | 拒绝当前引入 | 第二个 consumer 证明现有深模块 interface 不足      |
| full-frame record plane                        | 拒绝         | production reconstruction consumer 与资源/误差证据 |
| 把 fresh output 当 displayed texel             | 拒绝         | 两种 producer 真正统一；不能靠数值巧合恢复         |
| `@workgroup_size(1)`                           | 拒绝         | Metal/Vulkan 完整 record 上消除已知反例            |

## 接纳证据与剩余工作

当前 tests 覆盖 exact ABI/offset/resource cap、三种 production plan 的真实 GPU record、
published-texel separation、Busy、resize/suspend cancel-drain、generation mismatch、mapping typed
failure、branch protocol 的属性测试和 completion 单次消费。Native smoke 覆盖 event-loop poll 与
publication/presentation 路径。

仍未闭合 source edge、surface/capture boundary、different winding/higher-order branch、critical curve 两侧、
正负 spin、near-axis/near-extreme continuous fields，独立 high-precision certificate、第二 quality method 和
持久 artifact。它们的交付顺序与退出条件只在[路线图](../roadmap.md#连续字段证据与质量政策)维护。

## 一手来源

- [WGSL specification](https://www.w3.org/TR/WGSL/)：host-shareable layout、floating-point 与 compute/workgroup semantics；
- [WebGPU specification](https://www.w3.org/TR/webgpu/)：buffer usage、copy、mapping、queue timeline 与 synchronization；
- [wgpu 30.0.1 documentation](https://docs.rs/wgpu/30.0.1/wgpu/)及 workspace `Cargo.lock`：实际 Rust API 与锁定版本；
- [Rust Reference: type layout](https://doc.rust-lang.org/stable/reference/type-layout.html)：`repr(C)` 的 order、alignment 与 padding；
- [Carter 1968](https://doi.org/10.1103/PhysRev.174.1559)：Kerr 第四守恒量；
- [Lindquist 1966](https://doi.org/10.1016/0003-4916%2866%2990207-7)：相对论辐射传输；
- [Cunningham & Bardeen 1973](https://doi.org/10.1086/152223)与 [Cárdenas-Avendaño & Lupsasca 2023](https://doi.org/10.1103/PhysRevD.108.064043)：高阶像、photon ring 与 critical-curve 支持域。

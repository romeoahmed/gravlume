# Production 单样本检查决策

本文记录 test-only GPU record 如何收敛为 production 单样本 inspection，以及采用方案的生命周期、
GPU protocol、真实 backend 反例和扩张边界。Public Rust interface、observable 与当前支持域仍分别以
源码、[架构合同](../architecture.md)、[验证合同](../validation.md)和 [GPU 证据](../gpu-renderer.md)
为准。

**状态：固定单槽 seam 已采用；region batch、第二质量方法和持久 artifact 仍未授权。** Desktop
点击是首个真实 consumer；test-only ordered corpus 只复用执行 kernel，不扩大 production interface。

## 决策

Renderer 以一个固定槽位绑定：

```text
validated ImageSample
+ current published generation and extent
+ fixed full-KS WGSL-binary32 method
+ one bound published texture
```

Request 返回 process-local ticket。`Renderer::poll` 最终恰好产生一次 completed、cancelled 或 typed
failed completion。Completed result 明确分开：

- `published_texel`：从 request 绑定 generation 的实际 `Rgba16Float` texture 复制；
- `fresh_retrace`：同一 logical sample 的 fresh full-KS/WGSL-binary32 typed record。

两者的 producer、precision 与 refinement path 不同。Presentation 可能追加 shadow subpixel refinement
并量化到 binary16；fresh retrace 仍是单 ray binary32。数值接近或偶然 bit-equal 不建立 producer identity。

Ticket 也不是 artifact identity。持久证据仍需 canonical observation、method、producer revision、
shader/backend/adapter 与解释 metadata；schema 只应与首个保存/比较 consumer 一并落地。

## Observable 边界

Terminal-specific sum type 只形成合法组合：

| Field | 可解释范围 | 限制 |
| --- | --- | --- |
| termination | horizon、escape、surface、guard、exhaustion、failure、uncertain | guard/exhaustion 不是 physical source |
| branch | determinate terminal 的 side、turning/crossing 与 winding | failure/uncertain 无 branch；exhaustion 只有 prefix |
| source | Escape unit direction 或 surface `(r/M, oblate azimuth)` | finite escape sphere 不是 null infinity |
| frequency ratio | accepted surface 的 $(-p\cdot u_o)/(-p\cdot u_e)$ | 普通 RGB 不是 spectrum |
| coordinate-time duration | 当前 chart/profile 下的 $\Delta t/M$ | 不是 observer proper time |
| scene value | fresh plan output before display | sky/failure 是 preview/diagnostic；只有 surface model 定义 radiance |
| diagnostics | event candidates、ambiguity、residual 与 invariant drift | 不含 bracket width、Jacobi/parity 或独立 error certificate |

Strict decoder 拒绝 unknown discriminant/flag/tag、non-finite、非零 reserved lane，以及不可能的
terminal/source/scene/branch 组合；它不返回 heuristic partial success。

## 生命周期

Caller 只提交 validated sample，并从既有 renderer update 取 completion。Renderer 捕获 publication
identity；caller 不回显 generation、选择 solver 或持有 wgpu handle。Resize/suspend 负责 logical
cancellation，没有无 consumer 的 public cancel seam。

```text
Idle
  -> admit published G -> Submitted(G)
Submitted(G)
  -> map/decode success and publication still G -> Completed -> Idle
  -> resize/suspend/new publication invalidates G -> DrainDiscard(G)
  -> current map/decode failure -> Failed -> Idle
DrainDiscard(G)
  -> GPU and mapping ownership settle -> Cancelled -> Idle
```

已提交 command buffer 没有 portable preemption seam。Cancellation 只丢弃结果；槽位必须等 callback、
mapped view 和必要的 `unmap` 完成后才能复用。Mapping failure 直接形成 typed failure，不对未 mapped
buffer 再调用 `unmap`。Renderer 先安装 matching new publication，再核对 inspection generation；旧
completion 不得覆盖 desktop 的 `ViewportChanging`。

## GPU protocol 与资源

Request 是两个 `vec4` lanes（32 bytes），record 是六个 `vec4` lanes（96 bytes）。Rust 使用同序
scalar arrays、`#[repr(C, align(16))]`、`bytemuck::Pod` 与 compile-time offset assertions。Host-shared
DTO 不使用 `vec3`、Rust enum 或 implicit padding。Runtime-sized storage arrays 让 production 绑定
$N=1$，test corpus 绑定实际 $N$；两者共用 private kernel、entry point 与 decoder。

| Persistent resource | Bytes | Usage |
| --- | ---: | --- |
| request | 32 | `STORAGE | COPY_DST` |
| record | 96 | `STORAGE | COPY_SRC | COPY_DST` |
| readback | 104 | `COPY_DST | MAP_READ` |

总计 232 logical bytes，与 viewport extent 无关；不包含 pipeline、bind group、backend allocation 或
queue staging，也不是 driver memory peak。Readback `[0,96)` 是 record，`[96,104)` 是一个
`Rgba16Float` texel。

每次提交固定为：

```text
clear decoder-invalid record
-> dispatch one 8x8 workgroup; one-element binding activates lane 0 only
-> copy 96-byte record
-> copy one published texel with explicit 256-byte row pitch
-> map exactly 104 bytes on the producing submission
```

`map_buffer_on_submit` 把 mapping 排在 producing encoder 提交之后。单行 texture copy 显式满足
256-byte row-pitch alignment，但最后一行只需为实际 8-byte texel 分配存储，不分配虚构的下一行。

## 保留的 backend 反例

Apple M5/Metal 暴露过两个必须保留的 correctness witness：

- `@workgroup_size(1)` 返回正确 terminal/source/radiance，却把 travel time、drift 与部分 branch fields
  置零；恢复 `8×8` specialization 后 record 完整。64 lanes 是已证实的 backend specialization，不是
  subgroup width、SIMD 或吞吐声明。
- 省略 one-row texture copy 的 `bytes_per_row` 时，锁定 Metal HAL 路径向 native blit 传入零并得到
  不完整 readback；显式 256-byte pitch 后恢复。

这些历史事实已经完全纳入现行 protocol；旧 224-byte test-only resource 模型和无 consumer helper
不再单独维护。

## 接纳证据与扩张边界

现有 tests 覆盖 exact ABI/offset/resource cap、三种 production plan 的 GPU record、published texel /
fresh retrace 分离、Busy、cancel-drain、generation mismatch、mapping failure、branch protocol 与 completion
单次消费。Source-edge 九点另有[独立 corpus 证据](kerr-observable-corpus.md)，但 production 仍只接纳
一个 sample。

| 候选 | 当前决定 | 重开条件 |
| --- | --- | --- |
| public solver/profile parameter | 延后 | 第二个真实 quality implementation 与明确支持域 |
| bounded-region batch / active queue | production 延后 | 真实小区域 consumer、固定资源上限与 Metal/Vulkan A/B |
| public cancel / inspector trait / render graph | 拒绝当前引入 | 第二个 consumer 证明现有 interface 不足 |
| full-frame record plane | 拒绝 | reconstruction consumer 与资源/误差证据 |
| 合并 fresh output 与 displayed texel | 拒绝 | 两种 producer 实际统一，而非数值巧合 |
| `@workgroup_size(1)` | 拒绝 | Metal/Vulkan 完整 record 上消除已知反例 |

连续字段、第二质量方法与持久 artifact 的依赖顺序只在[路线图](../roadmap.md#连续字段证据与质量政策)
维护。

## 一手来源

- [WGSL specification](https://www.w3.org/TR/WGSL/)：host-shareable layout、runtime arrays 与 compute semantics；
- [WebGPU specification](https://www.w3.org/TR/webgpu/)：buffer usage、copy、mapping 与 queue timeline；
- [wgpu 30.0.1 `CommandEncoder`](https://docs.rs/wgpu/30.0.1/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit) 与 [`Buffer`](https://docs.rs/wgpu/30.0.1/wgpu/struct.Buffer.html#method.map_async)：submission-bound mapping 与 mapped ownership；
- [Rust type layout](https://doc.rust-lang.org/stable/reference/type-layout.html)：`repr(C)` order、alignment 与 padding。

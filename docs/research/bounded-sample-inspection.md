# 有界单样本 GPU 路径审计：历史基线

本文保存 revision `9f39b8d798d3889ecb3032b5dcc92ad64103c6ad` 的 test-only 单样本 GPU 实验、Metal 反例和被后续 production 继承的最小 protocol；它不定义当前 interface、资源预算或支持域。后续采用决策见[Production 按需单样本检查](on-demand-sample-inspection.md)，当前事实见 [GPU 证据](../gpu-renderer.md)。

**状态：历史基线。** 当时使用 wgpu `30.0.0`，固定 request/record 与 `8×8 + lane 0` correctness specialization 已被 production 采用；224-byte 一次性 test resource 已由当前 232-byte 单槽模型取代。历史版本不能机械替换成 workspace 当前版本。

## 问题与结果

旧 [`capture_trace_sample`](../../crates/gravlume-render/src/gpu_capture.rs)只缩小 dispatch，仍按完整
viewport 分配四个 `16 B/pixel` planes 和 HDR texture，并复制整个 extent。它还把 branch counts 压缩
到 16 bit，不适合 exact evidence。

实验检验以下假设：不改变 `trace_pixel_at`、RK4 step policy 或 event state machine，只增加固定
request、固定 record 与 plan-specific scene-value sink，能否复算一条 ray 并返回 typed terminal、
exact branch、source、Frequency Ratio、coordinate-time duration、scene-linear output 和 diagnostics。

Analytic、bolometric surface 与 blackbody 三种 `TracePlan` 的真实 GPU tests 均通过，因此技术核心被
接纳。当时没有 production consumer，所以 generation/request echo、Busy/cancel/supersede/poll 状态机被
删除；这些字段只能回显 host context，不能增加 GPU 科学证据。后续 desktop consumer 出现后，
publication ownership 从头定义，未冻结 test helper API。

## 历史 protocol

WGSL request 使用两个 `vec4` lanes，record 使用六个 `vec4` lanes。Rust DTO 采用同序 scalar arrays、
`repr(C, align(16))`、`bytemuck::Pod` 和 compile-time offsets；host-shared 数据不使用 `vec3`、`bool`、
Rust enum 或 implicit padding。[WGSL layout](https://www.w3.org/TR/WGSL/#alignment-and-size)与
[Rust type layout](https://doc.rust-lang.org/reference/type-layout.html#the-c-representation)是依据。

| bytes/offset | lane        | 语义                                                         |
| -----------: | ----------- | ------------------------------------------------------------ |
|  request `0` | `vec4<u32>` | pixel x/y、viewport width/height                             |
| request `16` | `vec4<f32>` | subpixel x/y、reserved zero                                  |
|   record `0` | `vec4<u32>` | termination、flags、steps、event candidate bits              |
|  record `16` | `vec4<u32>` | radial/equatorial counts、signed winding、initial polar side |
|  record `32` | `vec4<f32>` | source coordinates/direction、coordinate-time duration       |
|  record `48` | `vec4<f32>` | plan-specific scene-linear RGBA/tag                          |
|  record `64` | `vec4<f32>` | event residual、reserved zero                                |
|  record `80` | `vec4<f32>` | null、energy、$L_z$、Carter max drift                        |

Record size/alignment/stride 是 `96/16/96` bytes；branch counts 保持 32 bit，negative winding 按 bitcast
round-trip。Decoder 拒绝 unknown termination/polar side、invalid flags/tag、non-finite value、非零
reserved lane 与非法 terminal/source/branch 组合。`NumericalFailure` 和 `Uncertain` 不暴露 provisional
branch。

## Workgroup 反例

最初候选使用 `@workgroup_size(1)`。Apple M5/Metal 上它能返回正确 terminal、source 与 radiance，却把
同一 canonical sample 的 coordinate time、maximum drift 和部分 branch fields 返回为零；恢复
production `8×8` specialization 后字段完整。WGSL 不承诺 backend compiler 的等价变换或固定 subgroup
width，因此基线保留：

```wgsl
@compute @workgroup_size(8, 8, 1)
fn inspect_sample(@builtin(local_invocation_index) local_index: u32) {
    if local_index != 0u {
        return;
    }
    // One invocation owns the whole sequential trace and record.
}
```

Host 只 dispatch 一个 workgroup。这里的 64 invocations 是已知 backend 反例约束的 correctness
specialization，不是吞吐、SIMD 或 subgroup 声明；恢复 `workgroup_size(1)` 必须先在 Metal/Vulkan 的
完整 record 上消除反例。

## 历史资源与 readback

| resource | bytes | usage     |
| -------- | ----: | --------- |
| request  |    32 | `UNIFORM` |
| record   |    96 | `STORAGE  | COPY_SRC` |
| readback |    96 | `COPY_DST | MAP_READ` |

合计 224 logical bytes，与 viewport extent 无关，不是 driver allocation 或 memory peak。Storage record 与
staging readback 分离；同一 encoder 先 dispatch 再复制 96 bytes，测试随后等待 submission、map、读取、
drop view 并 `unmap`。[wgpu `Buffer`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Buffer.html#method.map_async)
描述 mapped buffer 的 CPU/GPU exclusive ownership。

## 接纳边界

基线证明固定 record 能取得三种 plan 的 typed single-ray evidence，也保留了发现
`workgroup_size(1)` 错误的真实 GPU witness。它没有证明 UI、generation、cancellation、actual published
texel、持久 artifact、source edge、critical/higher-order branch、spin sweep 或独立 high-precision
certificate。

Production 已在后续决策中闭合 ticket/completion、cancel-drain、generation 与 published-texel
separation；连续字段 corpus 仍受[路线图](../roadmap.md#连续字段证据与质量政策)约束。若 accepted
observable 超预算，应收窄支持域、refine/fallback 或返回 typed uncertainty，不能放宽 tolerance。

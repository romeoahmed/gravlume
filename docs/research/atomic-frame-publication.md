# 完整帧原子发布

> **状态：已采用。** GPU 只发布完整、当前 generation 的原生分辨率 scene。研究阶段尝试的低分辨率阶梯和可见 tile 扫描均已拒绝。

## 决策

renderer 把 presentation 与 compute 生命周期分开：

```text
published scene ───────────────────────────────> present
                       hidden candidate
trace batches -> shadow coverage -> timestamp -> generation check
                                                  |
                                                  └─ promote texture view atomically
```

- `candidate` 从不进入 display bind group；最后一批完成前，窗口继续显示上一张完整 scene。
- completion 只有在 timestamp readback 成功且 generation 匹配时才晋升；resize 后的 stale work 只能回收，不能发布。
- 晋升复用 candidate `TextureView`，不做同尺寸全屏 copy，也不保留第二张新尺寸 published texture。
- compute progress 不 acquire/present surface；presentation 只由窗口/UI 事件或一次 publication 触发。
- resize 合并最新 physical extent；旧完整 scene 在新 surface 上 aspect-fit，直到 replacement 原子切换。

这解决的是视觉与生命周期正确性，不降低求解总工作。交互延迟必须由 solver 和保守 accelerator 改善。

## 为什么不显示低分辨率阶段

完整低分辨率 frame 虽不撕裂，但仍产生肉眼可见的锐度跳变；随机、棋盘、Morton 或逐 tile 顺序只改变 partial-frame 伪影形状。它们都不能满足“一次 publish 是一个完整 scene snapshot”的产品合同。因此生产不以降质阶段掩盖 GPU tracing latency。

## 资源与同步边界

- normal active frame 包含 FP16 candidate、RGBA8 UI 与按 extent 计算的 escape-map/shadow scratch；完整 scene 只保留 FP16 view。
- transactional rebuild 在分配前计算 published、installed 与 replacement 的真实 typed footprint。不得以固定 bytes-per-pixel 隐藏 scratch 或 candidate 生命周期。
- wgpu queue submission 保序；timestamp resolve/readback 位于最后的 refine 之后。CPU 不需要额外阻塞 wait 才能建立发布顺序。
- `Surface::configure` 可能等待 GPU idle，故 live resize 必须合并，不能每个 `Resized` 事件立即重配。

当前实现见 [`renderer.rs`](../../crates/gravlume-render/src/renderer.rs)、[`display.rs`](../../crates/gravlume-render/src/display.rs) 与[架构合同](../architecture.md)。

## 已采用的求解优化

研究中识别的重复 endpoint geometry/RHS 已消除：普通 classical RK4 步复用 exact endpoint derivative，避免为 event/invariant/下一步重复构造 geometry。其他 solver/accelerator 的 adopted/rejected 状态统一记录在 [GPU geodesic acceleration](gpu-geodesic-acceleration.md)，不在本文维护第二份性能 ledger。

## 仍成立的门槛

- incomplete 或 stale generation 永不 publication；
- zero extent、suspend、surface loss 与 resize 不破坏上一完整 scene 的所有权；
- smoke 必须越过 matching publication 和 presentation submission，而不是“compute 已 encode”；
- 新策略必须用完整 invalidation→publish 延迟评估，不能只报单 dispatch 或 ray count；
- 若未来引入 temporal/history，scene/view/solver/extent generation 不匹配时必须拒绝复用。

## 主要依据

- [`wgpu::Surface::configure`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Surface.html#method.configure) 的重配/idle 合同；
- [`wgpu::Queue`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Queue.html) 的有序提交与 callback 接口；
- [`winit::ApplicationHandler`](https://docs.rs/winit/0.30.13/winit/application/trait.ApplicationHandler.html) 的 redraw/lifecycle 模型；
- [WGSL execution model](https://www.w3.org/TR/WGSL/#execution)；
- 项目的[验证合同](../validation.md)。

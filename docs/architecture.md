# 架构与实现合同

本文定义 Gravlume 的 Rust Module/Interface、桌面/GPU 生命周期和数据协议。它是待实现合同，不是当前代码完成度说明；平台与依赖见 [Rust 平台合同](platform.md)，连续模型见[数学物理合同](physics.md)。

## 1. 设计主线

Gravlume 同时维护两条互相校验但不共享核心方程实现的计算路径：

1. **Reference**：CPU `f64`、自适应 DP5(4)、事件定位、丰富诊断，优先精度与可解释性；
2. **Interactive**：WGSL `f32`、有界延迟、跨 Vulkan/Metal，优先稳定状态表示与可见失败。

默认 renderer 采用分层组合，而不是一个覆盖所有物理和性能目标的万能 solver：

- Cartesian Kerr–Schild 数值追迹是任意外部观察者的跨后端基线；
- Schwarzschild/Kerr 的解析、半解析或 LUT 是带适用域的内部 accelerator；
- geometry、transport、reconstruction 和 display 是不同阶段；
- 完整 Kerr–Newman、Jacobi beam、slow-light、Stokes/Faraday 保留为研究能力。

## 2. 领域状态与提交模型

```text
VisualizationDraft
    |
    | validate + normalize + compare committed snapshot
    v
CommittedVisualization
    +-- Observation
    |     +-- PhysicalScene
    |     |     +-- Spacetime
    |     |     +-- ObserverEvent + ObserverFrame
    |     |     +-- Emitters + ParticipatingMedia + SkyEnvironment
    |     +-- ViewportProjection
    +-- AppearanceModel
    |     +-- exposure + display transform + bloom + false color + overlays
    +-- QualityPolicy
          +-- observable error + frame/memory budget
          +-- reconstruction + transport capability tier
```

UI draft 可以短暂无效，renderer 和 reference 只接收不可变、已验证的 committed snapshot。提交失败时保留上一快照，并返回稳定 issue code、field path 和解释；不上传半有效参数。

领域层必须守住：

- mass scale 正且 finite；spin/charge finite，极端性被显式分类；
- Observer Event 位于 chart 与观察域；Observer Frame 有时间定向、手性和正交归一验证；
- Photon Momentum future-directed、null 且与观察者测得正频率；Backward Trace 不改变其物理时间定向；
- termination 是显式枚举：escape、horizon crossing、emitter hit、singularity approach、step exhaustion、numerical failure 分开；
- Physical Model 产生 invariant radiance、Frequency Ratio、Optical Depth 和诊断；Appearance Model 不能改变 trace/transport；
- Quality Policy 不静默改变 Physical Scene，不把 numerical failure 伪装成物理黑色；
- CPU canonical state 使用 `f64`；GPU DTO 是私有、经范围检查与无量纲化的 `f32`。

### 2.1 Generation

快照至少记录四个 checked `u64` generation：

| Generation | 变化来源 | 必须失效的结果 |
|---|---|---|
| `geometry` | spacetime、observer、chart、projection/sample lattice、resolved trace semantics | geodesic、source map、所有 history |
| `transport` | emitter、medium、sky、source asset revision | radiance/optical depth、相关 history |
| `appearance` | exposure、display、bloom、false color | display 后段，不重新 trace |
| `extent` | surface/internal extent、format、color space | size-dependent resources 与 history |

generation 由 commit diff 和 resolved plan 自动推进，调用者没有 `reset_history()`。history 除 radiance/weight 外还携带 generation、termination/branch、source key 和必要的 observer/retarded-time key；任一不相容即拒绝。

## 3. Package 与模块方向

仓库先保持最小 root package；实现出现下列真实依赖边界后再拆分：

```text
gravlume/
├─ Cargo.toml                 # workspace + root desktop binary
├─ crates/
│  ├─ gravlume-domain/        # validated scene、observables、纯 f64 基础数学
│  ├─ gravlume-reference/     # 独立 CPU oracle、fixture/LUT、比较报告
│  ├─ gravlume-render/        # 私有 wgpu frame engine、WGSL、offscreen target
│  ├─ gravlume-desktop/       # winit/egui panels 与封闭 run lifecycle
│  └─ gravlume-research/      # bounded workbench、capture/readback、experiment
├─ shaders/                   # WESL source + checked-in generated WGSL
├─ tests/fixtures/            # versioned trajectory、field、reference image artifacts
└─ xtask/                     # shader/fixture/asset/release checks
```

依赖只向内：

```text
root -> desktop -> domain + render
render -> domain
reference -> domain
research -> domain + reference + render(offscreen)
```

`domain` 不依赖 wgpu/winit/egui；`reference` 不依赖 renderer。assets、math、passes 和 UI 先作为真实消费者内部模块，不预先创建浅 crate。CPU reference 与 GPU interactive 共享领域输入、termination 语义、fixture schema 和容差政策，**不共享一份自动生成的核心方程**，以免同一符号错误污染 oracle 与被测对象。

## 4. 对外接口

### 4.1 Domain 与 reference Interface

持久化输入、可编辑 draft、validated domain 和计算输出是四种不同类型。serde 只进入 versioned preset/fixture DTO；`PhysicalScene`、`Observation` 和 `ViewportProjection` 字段私有，构造成功后始终满足不变量。

```rust
pub struct PhysicalScene { /* private, validated f64 domain state */ }
pub struct PhysicalSceneDraft { /* editable seam input */ }
pub struct Observation { /* PhysicalScene + ViewportProjection */ }
pub struct ViewportProjection { /* private, validated */ }
pub struct ViewportSample { /* private pixel/subpixel coordinates; no derived sight state */ }
pub struct InitialViewRay { /* event + future-directed momentum */ }

impl PhysicalScene {
    pub fn commit(
        draft: PhysicalSceneDraft,
    ) -> Result<Self, ValidationReport>;
}

impl ViewportProjection {
    pub fn perspective(
        width: NonZeroU32,
        height: NonZeroU32,
        vertical_fov: Angle,
    ) -> Result<Self, ValidationReport>;

    pub fn sample(
        &self,
        pixel_x: u32,
        pixel_y: u32,
        subpixel_x: f64,
        subpixel_y: f64,
    ) -> Result<ViewportSample, ValidationReport>;
}

impl Observation {
    pub const fn new(
        scene: PhysicalScene,
        projection: ViewportProjection,
    ) -> Self;

    pub fn initial_ray(
        &self,
        sample: ViewportSample,
    ) -> Result<InitialViewRay, ValidationReport>;
}

pub struct ReferenceInstrument { /* DP5(4), events, diagnostics */ }
pub struct ReferenceRequest { /* TraceInputId + Arc<Observation> + resolved initial ray + policy */ }

impl ReferenceRequest {
    pub fn new(
        input_id: TraceInputId,
        observation: Arc<Observation>,
        sample: ViewportSample,
        policy: ReferencePolicy,
    ) -> Result<Self, ValidationReport>;
}

impl ReferenceInstrument {
    pub fn baseline_v1() -> Self;

    pub fn trace(
        &self,
        request: ReferenceRequest,
    ) -> Result<ReferenceOutcome, ReferenceRuntimeError>;
}
```

`ViewportProjection::sample` 在 seam 检查 extent、pixel index、subpixel range 与 finite FOV；成功后的 `ViewportSample` 只保存 projection-independent pixel/subpixel coordinates，不缓存 FOV/extent 派生的 sight-plane state。`Observation::new` 组合两个 validated value，不重复检查其私有内部字段。`Observation::initial_ray` 必须针对自己的 projection 重新验证并解析 sample，并在返回前建立 future-directed/null invariant；`ReferenceRequest::new` 在绑定 Observation 时完成这一步，因此跨 projection sample 不能把旧 sight state 带入新 Observation。该接口是 Viewport Sample、projection、Observer Frame、Sight Direction 与 Photon Momentum 的唯一 CPU Interface；调用者不组合 tetrad 分量或反转符号。WGSL 独立实现同一数学合同并通过中心、四角和 jitter fixture，不复用 CPU 方程生成。`ReferenceInstrument::trace` 返回 typed termination 与 diagnostics；non-convergence、step exhaustion 和 numerical failure 是 `Ok(ReferenceOutcome)`，不是 panic 或伪黑色。未满足 reference-v1 normalization 的 Observation 返回 `ReferenceRuntimeError`。

`ReferenceTracer` 是 fixture、收敛测试和研究批处理使用的低层 concrete API；它接收已经验证并按 v1 policy 精确归一化为 `M=1` 的 canonical `GeodesicState`、affine direction 和 event configuration，不公开可替换 solver trait。`ReferenceInstrument` 同时要求派生的 `omega_obs` 在数值预算内归一化为 1，未归一化输入在 seam 返回 typed configuration/runtime error。`TraceInputId` 是调用者提供并由 outcome 精确传播的稳定逻辑身份；fixture 使用其 versioned `id`，其 expected oracle 也必须绑定并核对同一 identity；Observation 路径必须为 Observation/sample/affine direction/events 的完整输入分配身份，不能用进程局部序号冒充内容 identity。`ReferenceComparison::baseline_v1` 只接受同一 input ID 的 `reference-regular-v1`/`reference-strict-v1` 角色组合；角色或 identity 错误返回 `ComparisonError`，数值预算失败才进入 `ComparisonIssue`。`ValidationReport` 的 issue code、field path 和 severity 是稳定 Interface；本地化 message 不是。未知 preset 字段、版本、枚举值或与 profile 不一致的固定值在输入 seam 拒绝，不进入 domain。主线不公开 `Metric`、`Integrator`、`EventLocator` 或 `RenderPass` trait；它们没有需要调用者替换的第二个 adapter，公开只会泄漏实现选择。

### 4.2 Desktop Instrument

普通应用路径只暴露一个生命周期入口：

```rust
pub fn run(launch: Launch) -> Result<(), RunError>;

pub struct Launch { /* private launch fields */ }

impl Default for Launch {
    // embedded assets, one Kerr exterior scene, native baseline quality
}

impl Launch {
    pub fn with_initial(self, initial: InitialVisualization) -> Self;
    pub fn with_window(self, window: WindowPreferences) -> Self;
    pub fn with_local_assets(self, policy: LocalAssetPolicy) -> Self;
}
```

常见调用：

```rust
fn main() -> Result<(), gravlume_desktop::RunError> {
    gravlume_desktop::run(Default::default())
}
```

`run` 在 desktop 主线程调用并阻塞到窗口关闭；这不是每帧等待 GPU。公开接口不出现 `WindowEvent`、`ActiveEventLoop`、`egui::Context`、`Device`、`Queue`、surface frame、pass、solver 名称或 history reset。

Implementation 负责：

1. 项目选择在首次 `resumed` 创建 window 与 surface，再请求与该 surface 兼容的 adapter/device 并配置；后续 `resumed` 必须幂等；
2. 所有窗口事件先交 egui-winit，被 UI 消费的 pointer/keyboard 不再改变 Observer Frame；
3. `RedrawRequested` 固定执行 UI → commit → generation/resource sync → trace/classify/refine → transport → reconstruct/history → display → egui → submit/present；
4. zero extent 不 acquire/configure；suspend/resume 幂等；旧 frame 结束后才 reconfigure；
5. `about_to_wait` 合并动画、egui repaint deadline 和 worker completion，静止时休眠；
6. 稳态一帧最多一次 acquire、一个主 encoder/submit 和一次 present，不调用 `Device::poll(Wait)`；
7. 默认场景完全离线，file policy 只允许显式 root 下的本地资源，不接受 URL。

### 4.3 Research Workbench

```rust
pub struct ResearchWorkbench { /* opaque, bounded */ }

impl ResearchWorkbench {
    pub fn open(options: WorkbenchOptions) -> Result<Self, WorkbenchOpenError>;
    pub fn submit(
        &self,
        request: ExperimentRequest,
    ) -> Result<ExperimentTicket, SubmitError>;
    pub fn try_take(
        &self,
        ticket: ExperimentTicket,
    ) -> Result<Option<ExperimentOutcome>, TicketError>;
    pub fn cancel(
        &self,
        ticket: ExperimentTicket,
    ) -> Result<CancelState, TicketError>;
}

pub enum ExperimentRequest {
    ReferenceTrace(ReferenceStudy),
    InteractiveAgreement(AgreementStudy),
    OffscreenCapture(CaptureStudy),
    SolverBakeOff(BakeOffStudy),
}
```

ReferenceTrace、InteractiveAgreement 和 SolverBakeOff 接收不可变 `Arc<Observation>`，使 Viewport Sample 与 Observer Frame 的语义不会泄漏到 workbench 调用者；只有 capture 需要 `Arc<CommittedVisualization>` 以包含 Appearance 与 Quality。每个 artifact 记录 observation/snapshot fingerprint、solver revision、dtype、normalization、policy/tolerance 和 producer revision。

`submit/try_take/cancel` 不等待 worker、文件、GPU 或 readback。queue、sample、step、bytes 和 wall-time 有硬上限；ticket 带 owner nonce 与不复用的 ID。取消是 cooperative，已提交的 GPU work 不撤回，只丢弃迟到结果。`TraceTermination::NumericalFailure` 是科学结果，不是 workbench runtime error。

## 5. Desktop 生命周期

`winit 0.30` 的应用模型由 [`ApplicationHandler`](https://docs.rs/winit/0.30.13/winit/application/trait.ApplicationHandler.html) 驱动。最小状态转换：

- `resumed`：创建或恢复 window/GPU；容忍重复 resumed；
- `window_event`：先处理 egui input，再处理未消费的 observer input；在 redraw 内 frame；
- resize/scale change：记录非零物理尺寸，安排 size-dependent bundle 换代；
- `about_to_wait`：根据动画、UI deadline、worker 消息决定 redraw 或 wait-until；
- `suspended`：停止 acquire/render，并按平台合同释放 surface-dependent 状态。

`egui_winit::State` 是每个 window/viewport 的输入状态。顺序是 `on_window_event` → `take_egui_input` → `Context::run` → `handle_platform_output`，不能成为无边界全局 singleton。

### 5.1 Surface 状态机

[`wgpu::Surface`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Surface.html) 的 configure/acquire 约束是实现不变量：

- window 用 `Arc<Window>` 创建 surface，window 所有权先于 GPU state；
- width/height 为零时不 configure；
- acquired `SurfaceTexture` 只存在于一次 frame 调用，present/drop 后才允许重配置；
- `Success` 与 `Suboptimal` 都携带可用 texture；后者完成本帧后安排重配；
- `Timeout`/`Occluded` 跳帧，`Outdated` 重配，`Lost` 重建 surface，`Validation` 消化已捕获的 validation error；device loss 由独立 callback 处理；
- size-dependent texture、bind group 与 history 总是同一 extent generation。

surface format/color-space 从 `format_capabilities` 的合法组合选择。wgpu 不替应用做 tone/gamut mapping：对 `*Srgb` surface 写 linear 由硬件编码；其他 HDR/wide-gamut color space 由 display pass 输出所需 transfer 与 gamut。首版发布合同是 SDR；HDR surface 只有完成色彩与显示验证后才加入，并把选择记录进 frame diagnostics。[wgpu HDR output](https://docs.rs/wgpu/30.0.0/wgpu/#surface-color-spaces-and-hdr-output)

## 6. GPU 能力与 WGSL 合同

### 6.1 原生桌面基线

WGSL core 只有 `f32`；`f16` 需要显式 enable 与设备 capability，规范没有 core `f64`。[WGSL](https://www.w3.org/TR/WGSL/) geodesic、frequency ratio、radiance accumulation 和基础偏振状态因此使用 `f32`；CPU reference、根分类、初值转换和 LUT 生成使用 `f64`。

发布设备必须满足 WebGPU-compliant compute、项目 limits/format usages 与当前有消费者的 `TIMESTAMP_QUERY`。`DeviceDescriptor` 只请求 baseline 和当前 resolved plan 的可选能力；history/diagnostic 确实需要零初始化时才加入 `CLEAR_TEXTURE`。不满足基线时返回 `UnsupportedPlatform`。feature 分类与三平台门槛只在 [Rust 平台合同](platform.md#3-发布平台基线)维护。

硬件 BVH ray tracing 处理直线/几何求交，不积分弯曲时空中的 null geodesic；不作为架构前提。

### 6.2 Compute 发散

WGSL 支持动态循环，但每 ray 的终止、步长、多像阶和 transport tier 可能造成 wave/subgroup 发散。每帧记录 step percentile、termination、refine ratio 与 pass time，再决定是否分桶或 compaction。首版 `classify → refine` 使用整屏第二次 dispatch + early return；只有 profile 证明队列构建净赚才增加 compacted/indirect 路径。

`SUBGROUP` 是 Metal/Vulkan 都支持的可选加速 variant，可用 ballot/reduction/scan 降低 active queue 构建和局部统计的 workgroup/global atomic 压力；它不会自动消除 ODE 循环发散。wgpu/Naga 30 使用 subgroup builtin 时省略标准 `enable subgroups;`，并且不假定 subgroup width 或与 local invocation 的映射；具体版本合同由 [Rust 平台合同](platform.md#52-可选加速-variant)维护。

所有 dispatch ceiling-divide，在 shader 首行检查 global ID；`8×8` 只是初始测量点，`16×8` 等按后端 benchmark，不写成永久常量。

## 7. Shader 与 CPU/GPU ABI

WGSL uniform/storage 必须满足 host-shareable alignment、size、array stride 和 address-space layout；`bool` 不是 host-shareable，`vec3<f32>` size 为 12、alignment 为 16。[WGSL memory layout](https://www.w3.org/TR/WGSL/#memory-layouts)

规则：

- 复杂 DTO 使用 `encase::ShaderType`；简单 POD 才使用 `#[repr(C)]`、显式 padding、bytemuck 和 size/offset 断言；
- ABI 不用 bool，使用 `u32` flags；不紧密排列裸 `vec3` 数组；
- fieldless enum 使用固定 `#[repr(u32)]`、显式 discriminant、checked `TryFrom<u32>` 和 CPU/WGSL contract test；
- 领域类型不直接充当 GPU DTO；pack 逐字段检查 `f64 → f32` finite、范围、归一化和 orientation；
- 低频 scene、每帧 observer、每-pass constants 分开上传；
- bind group/binding 由稳定语义手工固定，不从无序 runtime reflection 发明 ABI。

Rust 默认 layout 不能当 shader layout；任意整数 transmute 为无效 enum 是未定义行为。[Rust type layout](https://doc.rust-lang.org/stable/reference/type-layout.html)

### 7.1 Shader 生成

```text
shaders/**/*.wesl
        |
        | cargo xtask shaders
        v
generated/**/*.wgsl
        +--> Naga parse + validate
        +--> entry/binding/override contract tests
        +--> include_str! into renderer
```

WESL 只作 build-time composition/source-map adapter；runtime 消费已提交、可 diff 的纯 WGSL。`wesl.toml` 固定 `2026_pre`，generator 由 `Cargo.lock` 锁定，输入只使用 [imports](https://wesl-lang.dev/spec/Imports) 与 [conditional translation](https://wesl-lang.dev/spec/ConditionalTranslation)。CI 的 `cargo xtask shaders --check` 要求生成结果无 diff，并重新执行 Naga validation 与 ABI fixture。[WESL `wesl.toml`](https://wesl-lang.dev/spec/WeslToml)

## 8. 帧图与资源所有权

逻辑依赖固定为少量显式 pass，不引入通用 render graph：

```text
egui input + validate/commit
  -> sync generations/assets
  -> coarse trace
  -> geometry/error classify
  -> refine selected samples
  -> surface/volume transport
  -> spatial reconstruction
  -> stationary/branch-aware history
  -> scene-linear bloom
  -> display transform into gamma-space composite
  -> egui overlay on composite
  -> surface presentation encoding
  -> submit + present
```

coarse/refine 必须使用同一 resolved solver，不能在相邻像素静默混用 solver 形成 branch 裂缝。read/write history 与中间资源采用 ping-pong，不依赖 storage texture 原位读写。

renderer 内部资源分三层：

- `DeviceResources`：device/queue、shader、layout、pipeline、sampler，直到 device lost；只有无状态的最终 presentation pipeline 随 surface format 换代；
- `SceneResources`：sky/disk/LUT、scene buffer，随 geometry/transport/asset generation；
- `FrameResources`：HDR/coarse/history/bloom、gamma-space display composite 与 readback ring，随 extent/quality generation。

重建使用 two-phase install：先完整创建新 bundle、验证 binding，再原子 swap；失败时保留旧 bundle并报告事件。旧 wgpu handle 可由引用计数延寿到已提交 command 完成，但显存预算必须测新旧两套并存的峰值。

### 8.1 egui-wgpu 顺序

`Context::run` 后：

1. 安装 `textures_delta.set`；
2. 调 `Renderer::update_buffers` 并收集 paint callback command buffers；
3. 编码 scene compute/HDR，并把 display transform 写入固定 `Rgba8Unorm` gamma-space composite；
4. 在 composite 的最后一个 overlay pass 调 `Renderer::render`；
5. 把 composite 编码到当前 surface format；surface format 改变时只事务式替换这个无状态 pipeline；
6. 按 `update_buffers` 返回顺序提交 callback buffers 和 main buffer；
7. egui render 编码完成后处理 `textures_delta.free`，不在本帧使用它之前释放；
8. `window.pre_present_notify()`，再 present。

`Renderer::render` 的 render-pass lifetime 适配被隔离在一个小函数内，并在 finish encoder 前显式结束 pass。忘记 update/free 顺序不能依赖运行时偶然成功。[egui-wgpu Renderer](https://docs.rs/egui-wgpu/0.36.1/egui_wgpu/struct.Renderer.html)

## 9. Upload、readback 与并发

小型 UBO 使用 `Queue::write_buffer`；大量小块更新只有 profile 证明调用/分配成本显著时才引入 `StagingBelt`，自管 upload ring 需要更强证据。dynamic uniform/storage offset、buffer copy 和 texture row pitch 分别遵守 queried limits、`COPY_BUFFER_ALIGNMENT` 与 `COPY_BYTES_PER_ROW_ALIGNMENT`；不存在一个通用 upload alignment。ring segment 跟踪 submission completion，不能覆盖 GPU 仍在读取的 range。

readback 使用有界 `COPY_DST | MAP_READ` staging ring：本帧 encode copy，后续 `map_async`。callback 只发送 slot/ticket 状态，不 decode、写 EXR 或大比较；读取完必须 drop mapped views 再 unmap。texture→buffer row pitch 按 wgpu copy alignment 补齐，CPU artifact 去 padding。

native `map_async` callback 只有在后续 `Queue::submit`、非阻塞 `Device::poll` 或 `Instance::poll_all` 时才推进。只要存在 pending mapping/submission，event loop 就安排非阻塞 poll 与 wake 或有限 redraw；one-shot capture 后进入 idle 也必须完成。常规每帧仍禁止 `Device::poll(Wait)`。[wgpu `map_async`](https://docs.rs/wgpu/30.0.0/wgpu/struct.BufferSlice.html#method.map_async)

winit event thread 唯一拥有 surface/configure/submit 生命周期。worker 只做 asset decode、LUT 与 CPU reference，返回拥有所有权的结果；GPU create/install 在 frame-safe point。首版使用专用 Rayon pool、样本最外层并行、每条 reference trace 内确定顺序。有界同步 channel、任务资源上限和 cooperative cancellation 防止后台工作无限增长。

首次 adapter/device async 可用一次 `pollster::block_on`；普通应用不引入 Tokio、`async_trait` 或第二套常驻 runtime。

## 10. 资产、配置、错误与诊断

### 10.1 资产和配置

- 必需 shader、默认 preset 和小 LUT 可嵌入，默认场景离线启动；
- 外部资产先检查路径 root、metadata、magic、compressed bytes、pixel/channel/decoded bytes 上限，再解码；
- manifest 记录 logical ID、digest、source/license、format/channels、face orientation、primaries/transfer、物理单位或“仅外观”、转换工具；
- cubemap 以方向 fixture 校验 handedness/seam，不凭文件名猜方向；
- preset 带 `schema_version`，旧 schema 显式迁移；未知字段不由宽松 default 静默吞掉；
- research fixture 与用户 preset 分离。需要持久化的文件各自定义覆盖、临时文件、`sync_all`、目录同步、权限与失败恢复；`rename` 只是候选步骤，不被当作 Windows/macOS/Linux 共有的完整 durability 证明。

### 10.2 错误按处理者分类

- `ValidationReport`：用户能修正的 field error/warning，保留旧 snapshot；
- `AssetEvent::Failed`：logical ID、path/format/limit/decoder context，保留旧资产；
- `FrameSkip`：zero extent、timeout、outdated/lost/reconfigure，状态机恢复；
- `FatalRuntimeError`：OOM、不可恢复 device loss、shader/ABI contract、内部 invariant；
- `OracleError`：root/turning classification、non-convergence、取消、资源上限。

正常 surface flow 不 panic。device 初始化后安装 uncaptured-error 与 device-lost callback；callback 只发 typed event 回 event thread。shader/layout/pipeline/大 allocation 用 error scope，并在同一线程成对 push/pop。

### 10.3 可观测性

CPU `tracing` 字段至少包含 frame、generation、adapter/backend、surface format/color space、quality policy 和 scene fingerprint。GPU timestamp 只比较差值并乘 timestamp period；缺少该发布必需能力时设备初始化失败。

每帧至少统计 pass time、trace/refine count、steps 分位数、termination 分布、null/E/Lz/Carter drift bucket、NaN/Inf、history accept/reject 原因、asset/LUT install 和 surface/device event。默认只写本地；任何网络 telemetry 需要独立 opt-in 产品决策。

## 11. 内存与性能预算

首版布局估算每个 full-resolution internal pixel：HDR `8 B`、source/geometry key `8 B`、metadata `4 B`、history HDR `8 B`、history key `8 B`、weight `2 B`、refine index `4 B`、折算 half-res coarse `5 B`、`Rgba16Float` bloom mip chain `10.67 B`、gamma-space display composite `4 B`，合计约 **61.7 B/pixel**。它不含 heap alignment、asset、pipeline、driver 和 rebuild 双份峰值。

| Internal extent | 核心中间资源估算 | 初始政策 |
|---:|---:|---|
| 1920×1080 | 约 122 MiB | 中档独显交互基线 |
| 2560×1440 | 约 217 MiB | 默认 internal extent 上限候选 |
| 3840×2160 | 约 488 MiB | 不默认 native trace，使用上采样 |

性能验收而非当前成果：

- 选定中档独显 1080p p95 ≤ 16.7 ms，trace + reconstruct ≤ 14 ms；
- 选定集显 720p p95 ≤ 33.3 ms；
- 1440p 核心资源 < 256 MiB，总 GPU peak 初始目标 < 512 MiB；
- 动态分辨率有上下阈值、hysteresis 和最小驻留时间；
- reference/capture 只服从 error budget，不声称实时。

每份性能 artifact 固定 target、OS、adapter、driver、power mode、build profile、scene fingerprint、extent、shader/feature variant、warm-up、样本数、p50/p95 和 GPU memory peak。A/B 在运行前定义最小收益与字段误差预算；timestamp 只能报告查询点之间的 GPU 时间，register pressure 或 memory traffic 只有在记录具名 vendor profiler/offline compiler 与可复现采集方法时才作为附加证据。

## 12. 验证矩阵与完成定义

| 层 | 被测对象 | 必须通过的门槛 |
|---|---|---|
| Domain | 参数、单位、draft commit、frame | 无 NaN/Inf 漏到 snapshot；极端性正确；非法 observer 被拒绝 |
| CPU reference | Minkowski/Schwarzschild/Kerr/KN | weak-field、$\sqrt{27}M$ shadow、published path、convergence、null/E/Lz/Carter drift |
| WGSL contract | 所有 entry/feature variants | 生成无 diff；Naga parse/validate；binding/ABI size/offset/discriminant 精确匹配 |
| Headless GPU | 1×1、奇数 extent、边界 workgroup、标准 sample | 无 OOB/validation error；CPU/GPU termination 与 continuous observable 达预算 |
| Reconstruction | critical curve、caustic、多像盘、高频 sky | 不跨 branch 插值；cut/resize/generation 100% 拒绝旧 history；字段优先于 PSNR |
| Transport | vacuum、surface、analytic slab、spectral fixture | frequency ratio、`I_nu/nu^3`、optical depth、source unit 通过 |
| Display | HDR ramp、negative/non-finite、EXR | display 前 scene-linear；UI 位于 display 后；capture 绕过 appearance tone map |
| Lifecycle | resize、zero、suspend、所有 surface acquire 状态、device errors、idle readback | 无 panic/死锁/跨代 frame；one-shot capture 在 idle 中完成；可恢复错误跳帧，fatal 有上下文 |
| Performance | release matrix 的 Metal/Vulkan adapter | 记录 p50/p95、pass、steps、refine、memory peak，并达到第 11 节目标 |
| Release | binary、shader、asset、license | 离线默认场景；产物/来源/第三方通知闭环 |

视觉 golden 只是回归的一层，不能取代 termination map、source anchor、branch、Frequency Ratio、Optical Depth、steps 和 invariant drift。每个 fixture 记录 schema、producer revision、solver、precision、normalization、parameter fingerprint 与 tolerance policy。

## 13. 编译与安全政策

domain、reference、desktop 和 render Module 使用 `#![forbid(unsafe_code)]`。wgpu experimental token 或 passthrough shader 只能在独立 research feature 中接受单独 safety review，不能渗入这些 Module。

正常 validation、asset failure、surface loss 与 oracle non-convergence 不使用 panic。workspace lint 对 `unsafe_code`、broken intra-doc link、`unused_must_use` 严格；不把所有 Clippy pedantic 规则机械设为 error。release 不启用 glam `fast-math` 或依赖后端快速数学改变临界分类；profiling 构建保留足够符号关联 CPU span 与 GPU capture。

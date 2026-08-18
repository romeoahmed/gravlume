# 原生平台合同

本文定义 Gravlume 的支持 target、图形后端、HDR 状态来源、WGSL/GPU 基线与发布证据。版本和 feature closure 始终以 `Cargo.toml` 与 `Cargo.lock` 为准；本页不复制 patch 版本。

Gravlume 是原生 Rust 2024 桌面应用。支持 macOS/Metal、Windows/Vulkan 和 Linux/Wayland/Vulkan；D3D12、GLES、X11、浏览器 WebGPU 与 WebGL 不在合同内。

## Toolchain 与依赖边界

| 职责                 | 直接技术                                                 | 所有者                                           |
| -------------------- | -------------------------------------------------------- | ------------------------------------------------ |
| window/UI            | winit、egui、egui-winit、egui-wgpu                       | `gravlume-desktop`                               |
| GPU                  | wgpu、WGSL                                               | `gravlume-render`                                |
| native display state | objc2/AppKit、windows-rs/WinRT、wayland-client/protocols | `gravlume-native-display`                        |
| domain/ABI           | glam、bytemuck、num-traits                               | domain math 与显式 GPU DTO 分离                  |
| reference/fixtures   | serde、toml、rayon                                       | `gravlume-reference`                             |
| errors/diagnostics   | thiserror、tracing、pollster                             | typed crate interfaces；root 安装唯一 subscriber |
| verification         | approx、proptest、optional Criterion                     | 浮点断言、property tests 与显式 GPU bench target |

平台 Cargo features 只在对应 target 增量启用：macOS 只启用 Metal，Windows/Linux 只启用 Vulkan，Linux 只启用 Wayland。Dependency 必须属于直接消费者。修改依赖后运行 `cargo tree -e features`，确认 X11 和未授权 backend 没有进入闭包。

## 支持矩阵

| Target                  | Backend          | 最低发布证据                                                                                     |
| ----------------------- | ---------------- | ------------------------------------------------------------------------------------------------ |
| stable macOS            | Metal            | native surface、EDR state change、headless compute、lifecycle tests 与 smoke                     |
| Windows 11 build 22621+ | Vulkan           | 具名 OS/adapter/driver、HDR toggle、跨屏、lifecycle tests 与 smoke                               |
| Linux desktop           | Vulkan + Wayland | 具名 distribution/compositor/adapter/driver、color-management feedback、lifecycle tests 与 smoke |

macOS/Metal 是当前运行时验证平台。Windows 与 Wayland 已有 source/API review 和可获得的 target compile coverage；这些证据不能替代实机矩阵。

## GPU 基线

Adapter 必须：

- 是非软件、WebGPU-compliant 的 wgpu adapter；
- 支持 `TIMESTAMP_QUERY`；
- 支持项目使用的 `rgba16float` sampled/storage usages；
- 满足 texture、binding、dispatch 与项目资源政策。

Device 只请求 production 实际消费的 features，不调用 `Features::all()`，也不把 adapter 最大 buffer limits 当成项目内存预算。缺失基线意味着平台不受支持，不能通过改变物理模型、精度、资源语义或 tolerance 绕过。

当前唯一必需的可选 GPU feature 是 [`TIMESTAMP_QUERY`](https://docs.rs/wgpu/30.0.0/wgpu/struct.FeaturesWebGPU.html#associatedconstant.TIMESTAMP_QUERY)。

## HDR 解析

HDR 需要两个独立事实同时成立：

1. **传输能力：** surface 精确广告 `Rgba16Float + ExtendedSrgbLinear`；
2. **实时显示状态：** 当前承载窗口的 display 提供可靠 HDR state、finite headroom 与 reference white。

缺少 pair、明确 SDR、系统 suppression、无效亮度、原生接入不可用或查询未知都会产生带原因的 SDR contract。wgpu `DisplayHdrInfo` 全部为 `None` 表示未知，不能被重解释为 HDR active。

Scene 始终保持 extended-linear sRGB/scRGB，`1.0` 表示 SDR reference white。Final pass 在 linear light 中组合 scene 与 UI，并写入：

- `ExtendedSrgbLinear` HDR surface 的 linear values；或
- 所选 SDR surface 需要的 encoding。

FP16 intermediate 或 HDR-capable adapter 都不等于 end-to-end HDR。

## macOS

主线程 AppKit monitor 查询窗口当前 `NSScreen`：

- potential EDR component value 大于 `1.0` 表示 display capability；
- current value 提供实时 tone-map headroom，可合法回落到 `1.0`；
- window change-screen、application screen-parameter 与 HDR suppression notifications 只标记状态 dirty，主线程随后重读 screen。

依据：[potential EDR](https://developer.apple.com/documentation/appkit/nsscreen/maximumpotentialextendeddynamicrangecolorcomponentvalue)、[current EDR](https://developer.apple.com/documentation/appkit/nsscreen/maximumextendeddynamicrangecolorcomponentvalue)与 [Metal custom tone mapping](https://developer.apple.com/documentation/metal/performing-your-own-tone-mapping)。

## Windows

Windows 使用 inbox `Windows.Graphics.Display.DisplayInformation`，不引入 Windows App SDK runtime，也不在应用层用 DXGI 轮询 display state。

- Windows 11 build 22621+ 通过私有最小投影 `IDisplayInformationStaticsInterop::GetForWindow` 从 winit HWND 获取 window-bound 对象；
- UI thread 拥有 `Windows.System.DispatcherQueue`；
- monitor 缓存对象、订阅 `AdvancedColorInfoChanged`，并在 window teardown 前移除 token；
- `CurrentAdvancedColorKind == HighDynamicRange` 才满足 native HDR state；
- `MaxLuminanceInNits / SdrWhiteLevelInNits` 给出 headroom，`SdrWhiteLevelInNits / 80` 给出 scRGB reference-white scale。

依据：[GetForWindow](https://learn.microsoft.com/en-us/windows/win32/api/windows.graphics.display.interop/nf-windows-graphics-display-interop-idisplayinformationstaticsinterop-getforwindow)与 [Advanced Color/HDR](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range)。

## Linux/Wayland

Linux build 只启用 Wayland。Monitor 从 winit raw handles 创建不拥有 socket 的 guest connection，绑定 `wp_color_manager_v1` v2–3，并读取当前 surface 的 preferred parametric image description。

- 只有 description 与 information events 完整后才安装 snapshot；
- 缺协议、旧版本、非 parametric description、缺 luminance 或 dispatch failure 都保持 typed unknown 并选择 SDR；
- 应用只读取 surface feedback；Vulkan WSI/wgpu 继续拥有 presentation color-space；
- winit 是唯一 socket reader，guest queue 只 dispatch 已读取事件，并使用 distant wake guard 避免 guest-only wake 被丢弃。

协议依据：[Wayland `color-management-v1`](https://wayland.app/protocols/color-management-v1)。

## WGSL 与资源语义

Handwritten WGSL 与 Rust consumer 同放在 `crates/gravlume-render/src/shaders/`。Rust 组合 production/capture source；wgpu 在 pipeline creation 时验证，GPU contracts 创建并执行相关 entry points。仓库没有 WESL generator、checked-in generated shader 或直接 Naga dependency。

Shader 必须遵循 [WGSL specification](https://www.w3.org/TR/WGSL/)：

- core 数学是 `f32`，不假定 `f64`、固定 subgroup width 或超出规范的 NaN/subnormal/fusion 行为；
- workgroup barrier 只同步 workgroup scope，并位于 uniform control flow；
- 需要原子性的 queue/index 操作使用 storage atomics；
- binding access、texture format 与 host-shareable layout 必须精确匹配；
- 不假定 dispatch-wide/global barrier。

Final trace batch 的 base trace、shadow classification 和 refinement 是同一 command buffer 中的独立 dispatch。Classification 把 candidate 当 sampled texture，refinement 把它当 write-only storage texture；不同 dispatch 的 usage scope 与 wgpu resource transition 建立顺序。每个 edge index 只写一次，每个 listed pixel 只由一个 invocation 更新，不依赖 backend-specific race behavior。资源规则见 [WebGPU resource usages](https://gpuweb.github.io/gpuweb/#resource-usages)。

## 发布证据

| 层            | 必须记录                                                                             |
| ------------- | ------------------------------------------------------------------------------------ |
| dependency    | committed lockfile、target feature closure、无 unintended backend/protocol duplicate |
| capability    | feature/limit/format usages 与 structured rejection                                  |
| shader        | production/capture entry points、host/WGSL ABI 与执行 contracts                      |
| headless GPU  | odd extent、workgroup boundary、multi-batch、coverage、resize 与 readback            |
| numerical     | termination/branch 和 continuous observables 满足 `validation.md`                    |
| performance   | revision、OS、adapter、driver、power、scene、extent、warm-up 与 raw timestamps       |
| native output | HDR/SDR transition、cross-display、surface recovery 与完整 generation presentation   |

一个 adapter probe 不是发布矩阵。升级 wgpu、winit、平台 binding 或 Wayland protocol 时，必须一起重审 Cargo features、WGSL validation、native lifecycle、HDR transport 和三个目标平台的证据。

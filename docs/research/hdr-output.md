# 原生 HDR 输出研究

状态：研究结论已落地；scene/UI/final contract、macOS EDR、Windows WinRT、Wayland guest queue 与 typed SDR fallback 已实现。研究日期：2026-08-12。范围锁定为 Rust 1.97、`wgpu 30.0.0`、`winit 0.30.13`、`egui-wgpu 0.36.1`，以及 macOS/Metal、Windows/Vulkan、Linux/Vulkan + Wayland。Linux 明确不再支持 X11；当前开发机是 macOS，Windows/Wayland 仍需目标平台实机验证。

## 结论

研究开始时 Gravlume **只有内部 HDR 中间纹理，没有 HDR 显示输出**。trace 写 `Rgba16Float` scene-linear 值，但 display shader 立即执行 `x / (1 + x)`，结果进入 `Rgba8Unorm` gamma composite；surface resolver 又只选择 `Srgb`。因此当时屏幕最终是 SDR，不能把“使用了 FP16 texture”称为端到端 HDR。当前实现状态见文首状态行与平台合同。

推荐第一条可交付路径是扩展线性 sRGB/scRGB，而不是直接上 HDR10：

1. renderer 保持一份明确命名和定标的 `scene-linear BT.709/D65` HDR 图像；
2. egui 单独渲染到透明 `Rgba8Unorm` gamma target；
3. 最终 presentation pass 在线性空间合成 scene 与 UI，再按一个不可拆分的 `OutputContract` 写 surface；
4. HDR surface 首选精确的 `Rgba16Float + ExtendedSrgbLinear` pair，SDR 首选精确的 `*8UnormSrgb + Srgb` pair；
5. 原生状态监听只发“输出状态已脏”事件，主线程随后重新读取完整状态、重新查询 surface capabilities，并事务式安装新的 surface/pipeline/uniform generation；
6. HDR 不可用、状态未知、协议缺失或系统要求抑制 HDR 时，显式且色彩正确地切换到 SDR，并保留可诊断原因；`unknown` 不能伪装成 `SdrActive`。

Windows 官方把 FP16 scRGB 作为通用应用的推荐方案；Apple 自定义 tone mapping 的官方 Metal 路径同样是 `RGBA16Float + extendedLinearSRGB + wantsExtendedDynamicRangeContent`。wgpu 30 已能配置这两个后端。HDR10/PQ 需要 BT.709→BT.2020 gamut conversion、PQ encoding、绝对亮度与 metadata policy；wgpu 30 又不写 HDR metadata，因此不应把它当“scRGB 不可用时随手 fallback”的第二候选。[Windows Advanced Color](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range)；[Apple performing your own tone mapping](https://developer.apple.com/documentation/metal/performing-your-own-tone-mapping)；[wgpu HDR guide](https://docs.rs/wgpu/30.0.0/wgpu/#surface-color-spaces-and-hdr-output)

## 1. 研究时的实现基线与根因

研究开始时，surface selection 只在 `SurfaceCapabilities::format_capabilities` 中寻找 `Srgb`，并固定写入 `SurfaceColorSpace::Srgb`；display shader 把所有有限非负 scene 值压入 `[0, 1)`，随后转成 sRGB gamma，egui 再画入同一个 8-bit gamma composite。现实现已按本文结论替换这条路径，历史根因保留在此供决策追溯。

这导致三个根本问题：

- 高光在到达 surface 之前已经被压成 SDR，surface 即使改成 FP16 也恢复不了；
- scene、display mapping、UI reference white 和 surface encoding 没有形成一份统一合同；
- egui 与 scene 过早合并到 8-bit gamma target，之后无法在 HDR surface 的正确参考白下合成 UI。

因此修复不能只是把 `SurfaceConfiguration.format` 改成 `Rgba16Float`。

## 2. wgpu 30 已提供什么

### 2.1 Surface capability 是精确的 format/color-space pair

`Surface::get_capabilities` 返回 `format_capabilities`，每项是一个 texture format 及其支持的 `SurfaceColorSpaces`。HDR 必须从这里选择精确 pair；`formats` 只列出可与 `Auto` 一起使用的安全格式，而 `Auto` 不会主动选择需要不同 shader encoding 的 HDR。配置一个未广告的 pair 会触发 validation failure。[wgpu HDR guide](https://docs.rs/wgpu/30.0.0/wgpu/#surface-color-spaces-and-hdr-output)；[`SurfaceCapabilities`](https://docs.rs/wgpu/30.0.0/wgpu/struct.SurfaceCapabilities.html)；[`Surface::configure`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Surface.html#method.configure)

wgpu 不做 tone mapping 或 gamut mapping。只有写 `*Srgb` texture view 时硬件会执行 sRGB OETF；`ExtendedSrgbLinear`、encoded extended sRGB、PQ、HLG 都要求应用写入已经符合目标合同的值。[`SurfaceColorSpace`](https://docs.rs/wgpu/30.0.0/wgpu/enum.SurfaceColorSpace.html)

首版只实现两个 output encoder：

| 模式 | 精确 pair | final shader 写入 |
|---|---|---|
| SDR | `{Bgra,Rgba}8UnormSrgb + Srgb` | `[0,1]` linear BT.709，由 format 执行 sRGB OETF |
| HDR extended-linear | `Rgba16Float + ExtendedSrgbLinear` | linear BT.709/D65 extended-range surface units |

非-sRGB 的 8-bit SDR format 可以作为次级 SDR pair，但必须由 final shader 手动执行 sRGB OETF。它是一个明确的 `SdrEncoding::ManualSrgb`，不能通过 `format.is_srgb()` 隐式散落在 pipeline 中。

### 2.2 `DisplayHdrInfo` 是 advisory snapshot，不是状态订阅

`Surface::display_hdr_info(&Adapter)` 每次都重新查询 OS；它没有缓存，也不提供 change event。所有字段都是 `Option`，全 `None` 表示 unknown，而不是 SDR。Metal 必须在主线程调用。[`Surface::display_hdr_info`](https://docs.rs/wgpu/30.0.0/wgpu/struct.Surface.html#method.display_hdr_info)；[`DisplayHdrInfo`](https://docs.rs/wgpu/30.0.0/wgpu/struct.DisplayHdrInfo.html)

能力与当前状态必须分开：

- surface pair 回答“这个 presentation path 能否配置某种 encoding”；
- native display snapshot 回答“当前主显示器、OS policy 和亮度还剩多少可用范围”；
- `DisplayHdrInfo::tone_map_headroom` 只给 advisory tone-map target，不是 surface 配置开关。

后端现状已从 locked crate source 复核：

| 后端 | wgpu 30 surface HDR | `DisplayHdrInfo` |
|---|---|---|
| Metal | `Rgba16Float` 广告 extended-linear/encoded extended spaces；configure 设置 `CAMetalLayer.wantsExtendedDynamicRangeContent` 和 CGColorSpace | 从承载 layer 的 `NSWindow.screen` 读 current/potential/reference EDR；主线程限定 |
| DX12 | `Rgba16Float + ExtendedSrgbLinear`；`Rgb10a2Unorm + Bt2100Pq`；configure 调 `SetColorSpace1` | 通过 DXGI output + SDR-white query 返回 nits、primaries、coarse state |
| Vulkan | 保留 driver 报告的 `VkSurfaceFormatKHR(format, colorSpace)` pair | 非 Windows native Vulkan 没有实现，Linux 返回全 unknown |

直接来源：[wgpu-hal Metal surface](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/metal/surface.rs)、[wgpu-hal Metal capabilities](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/metal/adapter.rs)、[wgpu-hal DX12 capabilities](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/dx12/adapter.rs)、[wgpu-hal DXGI HDR query](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/auxil/dxgi/hdr.rs)、[wgpu-hal Vulkan swapchain](https://docs.rs/crate/wgpu-hal/30.0.0/source/src/vulkan/swapchain/native.rs)。

### 2.3 wgpu 的明确缺口

- 没有 native display change subscription；必须由平台层监听后触发重查。
- Linux/Vulkan 没有 display luminance/headroom snapshot。
- 不调用 `vkSetHdrMetadataEXT` 或 DXGI `SetHDRMetaData`。`Bt2100Pq` 只配置 colorspace，不等同于完整 HDR10 metadata pipeline。[`DisplayHdrInfo`](https://docs.rs/wgpu/30.0.0/wgpu/struct.DisplayHdrInfo.html)；[`VK_EXT_hdr_metadata`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_hdr_metadata.html)
- 不替应用定义 scene exposure、reference white、tone curve、gamut mapping 或 UI brightness。

## 3. 色彩与合成算法

### 3.1 先定义 scene contract

`Rgba16Float` 只说明存储，不说明颜色。首版建议把 published scene contract 固定为：

```text
primaries: BT.709 / sRGB
white point: D65
transfer: linear
alpha: opaque
scale: 1.0 == renderer diffuse/reference white before output adaptation
range: finite, non-negative, highlights may exceed 1.0
```

final display pass 才根据 live output contract 做曝光/tone mapping。当前无条件 Reinhard `x/(1+x)` 应只属于 SDR output mapping；HDR 路径必须把可用 headroom 作为上界，保留 `>1` 高光。具体 tone curve 可以迭代，但必须满足：单调、有限、连续；diffuse white 稳定；最大输出不超过 contract；headroom 降低时压缩高光而不是重跑 trace。

输出状态变化只使 presentation generation 失效，**不使 geodesic trace generation 失效**。相机/scene 不变时不应因为用户改亮度、窗口跨屏或 HDR toggle 而重算 trace。

### 3.2 egui 不能直接改画到 FP16 surface

`egui-wgpu 0.36.1` 根据 `output_color_format.is_srgb()` 选择 shader：sRGB format 时输出 linear 交给硬件编码；非-sRGB format 时使用它偏好的 gamma-framebuffer entry。把 renderer 直接指向 `Rgba16Float` 会让它把 gamma UI 数值写进一个线性 scRGB surface，颜色错误。[locked renderer selection](https://docs.rs/crate/egui-wgpu/0.36.1/source/src/renderer.rs)；[locked egui shader](https://docs.rs/crate/egui-wgpu/0.36.1/source/src/egui.wgsl)

干净的边界是两张独立输入：

```text
published scene HDR (Rgba16Float, linear BT.709)
                              \
                               final output mapping + linear composite -> surface
                              /
egui overlay (Rgba8Unorm, gamma, transparent)
```

egui target 每帧 clear 为 transparent，只画 UI。final pass 对 UI 的 premultiplied gamma 结果先按 alpha unpremultiply，再 sRGB-decode，再重新 linear-premultiply；然后与 tone-mapped scene 做 source-over，最后把两者一起乘平台的 `reference_white_scale`。直接对 premultiplied gamma RGB 调 `srgb_to_linear`，或只缩放 UI 而不缩放同一 scene reference white，都是错误的。

平台 surface units 不完全相同：

- Apple EDR 的 `1.0` 是当前 SDR white；UI scale 为 `1.0`。
- Windows FP16 scRGB 的 `1.0` 是 nominal 80 nits；应用自行合成 SDR UI 时，官方要求乘 `SdrWhiteLevelInNits / 80`。tone-map headroom 是 `MaxLuminance / SdrWhiteLevel`，而 surface peak unit 是 `MaxLuminance / 80`。[Windows reference-white mapping](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range#match-your-apps-reference-white-to-the-os-sdr-reference-white-level)
- Wayland `windows_scrgb` description 同样定义 `1.0 == 80 cd/m²`，参考白未知时协议建议 203 nits/`2.5375`；但任意 parametric description 不能套这个常数，必须从其完整 description 建立 scale。[locked color-management XML](https://docs.rs/crate/wayland-protocols/0.32.13/source/protocols/staging/color-management/color-management-v1.xml)

所以当前 scRGB `OutputContract` 应直接携带同时作用于 scene 与 UI 的 `reference_white_scale` 和相对于该白点的 `tone_map_headroom`，shader 不应按 OS 猜常数。

## 4. macOS：AppKit EDR

Apple 官方自定义 tone-map 路径是 extended linear sRGB、`RGBA16Float`、`wantsExtendedDynamicRangeContent = true`，并按承载窗口所在 `NSScreen.maximumExtendedDynamicRangeColorComponentValue` 映射当前上限。这个值会随亮度、环境、供电和显示器变化；Apple 明确要求监听 screen-parameter notification，并考虑窗口被移到另一个显示器。[performing your own tone mapping](https://developer.apple.com/documentation/metal/performing-your-own-tone-mapping)；[`maximumExtendedDynamicRangeColorComponentValue`](https://developer.apple.com/documentation/appkit/nsscreen/maximumextendeddynamicrangecolorcomponentvalue)；[`didChangeScreenParametersNotification`](https://developer.apple.com/documentation/appkit/nsapplication/didchangescreenparametersnotification)；[`NSWindow.didChangeScreenNotification`](https://developer.apple.com/documentation/appkit/nswindow/didchangescreennotification)

推荐监听四类通知：

- `NSWindowDidChangeScreenNotification`，且 object 限定当前窗口；
- `NSApplicationDidChangeScreenParametersNotification`；
- `NSApplicationShouldBeginSuppressingHighDynamicRangeContentNotification`；
- `NSApplicationShouldEndSuppressingHighDynamicRangeContentNotification`。

收到任意通知只通过 `EventLoopProxy` 合并发送一个 `OutputStateDirty`。主线程随后读取 `NSApplication.applicationShouldSuppressHighDynamicRangeContent` 与承载窗口的 `NSScreen` current/potential EDR；不要在 Objective-C block 中重配 surface。平台 snapshot 与 Windows 采用同一个窄 API，而 render 不再按 OS 分支。Apple 明确要求自绘 HDR app 配合 begin/end notification 使用 suppression property；它应覆盖 headroom 并产生 typed `SystemSuppressed` SDR 原因。[`applicationShouldSuppressHighDynamicRangeContent`](https://developer.apple.com/documentation/appkit/nsapplication/applicationshouldsuppresshighdynamicrangecontent)

Apple 的 current headroom 存在启动闭环：没有 layer 请求 EDR 时 current 可能是 `1.0`。因此选择 transport 时使用 exact surface pair、suppression 和已知的 `potential` capability；`potential > 1` 时可先配置 extended-linear，再取一次 current。`potential > 1` 但 `current == 1` 表示 HDR transport 已 armed、当前没有高光余量，不应误判成永久 SDR。若 snapshot 的 current/potential 都暂时 unknown，则以 `HeadroomUnknown` 原因走 SDR，并在窗口重新关联 screen 或下一次参数通知后重查；不能用 `1.0` 代替 unknown 并悄悄开启 HDR transport。

crate 选择：直接使用当前 objc2 代际的 [`objc2-app-kit 0.3.2`](https://docs.rs/objc2-app-kit/0.3.2/objc2_app_kit/) 与 [`objc2-foundation 0.3.2`](https://docs.rs/objc2-foundation/0.3.2/objc2_foundation/)，`default-features = false`，只开 `NSApplication`、`NSWindow`、`NSNotification`、`NSOperation`、`NSString`、`block2` 所需 feature。用 `MainThreadMarker` 固化主线程约束；用 block-based `NSNotificationCenter` 保存 observer token，并在 `Drop` 中注销。`objc2-app-kit 0.3.2` 已生成 EDR suppression property/notifications 与 window/screen notifications；无需手写 selector。

## 5. Windows：inbox WinRT + HWND interop

### 5.1 可以避免 DXGI，但不是 `GetForCurrentView`

传统 winit window 是顶层 Win32 `HWND`，没有 `CoreApplicationView`。`DisplayInformation::GetForCurrentView` 取得的是“当前线程 CoreApplicationView”的对象，且非 application-view UI thread 可失败；它不是 Win32 HWND API。[`GetForCurrentView`](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.display.displayinformation.getforcurrentview)

Windows 11 build 22621+ 的正式桌面桥是 `IDisplayInformationStaticsInterop::GetForWindow(HWND, ...)`。它返回 inbox `Windows.Graphics.Display.DisplayInformation`，会挂接该 HWND 的消息循环以跟踪跨屏和 DPI 变化。官方要求：调用线程拥有该顶层窗口、线程有运行中的 `Windows.System.DispatcherQueue`、缓存对象至窗口销毁、注销 event token，然后释放最后一个引用。[`GetForWindow`](https://learn.microsoft.com/en-us/windows/win32/api/windows.graphics.display.interop/nf-windows-graphics-display-interop-idisplayinformationstaticsinterop-getforwindow)

因此状态查询/事件路径可以完全避免 DXGI：

```text
HWND
  -> IDisplayInformationStaticsInterop::GetForWindow
  -> Windows.Graphics.Display.DisplayInformation
  -> AdvancedColorInfoChanged
  -> GetAdvancedColorInfo snapshot
```

`AdvancedColorInfo` 给出 active `CurrentAdvancedColorKind`、supported kinds、peak/full-frame/min/SDR-white nits 和 primaries；它是 snapshot，事件到达后必须重新获取。[`AdvancedColorInfo`](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.display.advancedcolorinfo)；[`AdvancedColorInfoChanged`](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.display.displayinformation.advancedcolorinfochanged)

注意：较老的 Advanced Color 指南仍写着 Win32 只能轮询 DXGI，这是 2022 年桌面 interop 发布前的段落；build 22621+ 应以更新后的 `GetForWindow` 官方页面为准。wgpu 内部仍用 DXGI 实现自己的 `DisplayHdrInfo`，但 Gravlume 的 change monitor 和 Windows live snapshot 不需要再复制那套轮询。

### 5.2 DispatcherQueue 是 inbox Windows API

GetForWindow event 要求当前 winit UI thread 有 `Windows.System.DispatcherQueue`。若线程还没有，使用 `CreateDispatcherQueueController` 创建 `DQTYPE_THREAD_CURRENT` queue，保留 controller；winit 本来就在该线程 pump Win32 messages。退出前调用 `ShutdownQueueAsync` 并继续 pump 至完成，不能只 drop 后让 queue/thread lifetime 泄漏。[`CreateDispatcherQueueController`](https://learn.microsoft.com/en-us/windows/win32/api/dispatcherqueue/nf-dispatcherqueue-createdispatcherqueuecontroller)；[`DispatcherQueueOptions`](https://learn.microsoft.com/en-us/windows/win32/api/dispatcherqueue/ns-dispatcherqueue-dispatcherqueueoptions)

这不是 Windows App SDK 的 `Microsoft.UI.Dispatching.DispatcherQueue`，不引入 Windows App Runtime。

### 5.3 crate 与最小 interop projection

使用 Microsoft 官方 [`windows 0.62.2`](https://github.com/microsoft/windows-rs)，target-specific features 至少包括：

```text
Foundation
Graphics_Display
System
Win32_Foundation
Win32_System_WinRT
```

locked `windows 0.62.2` source 已包含 `DisplayInformation`、`AdvancedColorInfo`、`AdvancedColorInfoChanged`/remove token、`CreateDispatcherQueueController`。但 Windows SDK header-only 的 `IDisplayInformationStaticsInterop` 不在该 crate 生成模块中；这一点已对完整 crate source 搜索确认。推荐在 native platform crate 的私有 `windows::ffi` 中，依据官方 `windows.graphics.display.interop.h` 定义唯一一个最小 `windows_core::Interface` projection，只暴露 `GetForWindow` 所需 vtable；激活 factory 和 ABI 转换都封装成安全 `WindowsDisplayMonitor::new(HWND)`。不要把 COM pointer、GUID、vtable 或 raw `HWND` 泄漏给 desktop/render crate。

仓库全局 `unsafe_code = "forbid"` 不能与这些正式 FFI 共存。新建一个极窄的 `gravlume-native-display` crate，**不继承** workspace 的 `unsafe_code = "forbid"`；crate root 用 `#![deny(unsafe_code)]`，只在三个私有 OS FFI module 上局部 `#[allow(unsafe_code)]`，并启用 `#![forbid(unsafe_op_in_unsafe_fn)]`。每个 unsafe block 写明 pointer lifetime/thread/apartment invariant；对外 API 全部 safe。这样不会为了“零 unsafe”改用过时或更重的 API。

### 5.4 不使用 Windows App SDK / `windows-app`

`Microsoft.Graphics.Display.DisplayInformation::CreateForWindowId` 可以针对 desktop `WindowId`，但必须从已有 `Microsoft.UI.Dispatching.DispatcherQueue` 的线程调用；未打包应用还要部署 Windows App SDK runtime 并用 Bootstrapper 初始化，或者承担 self-contained deployment。[`CreateForWindowId`](https://learn.microsoft.com/en-us/windows/windows-app-sdk/api/winrt/microsoft.graphics.display.displayinformation.createforwindowid)；[unpackaged deployment](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/deploy-unpackaged-apps)

这对一个 winit/Win32 HDR 状态监听器没有收益。Microsoft 的 `windows-app-rs` 已归档，并明确称为 experimental、not ready for production、Windows App SDK 对非 .NET/Visual Studio toolchain 过度耦合。[Microsoft `windows-app-rs`](https://github.com/microsoft/windows-app-rs)。拒绝该路线。

### 5.5 Windows output selection

wgpu DX12 会无条件广告 FP16 scRGB，因为 DWM 可在 SDR display 上下转换；因此 exact pair 本身不能证明 HDR 当前 active。只有 `CurrentAdvancedColorKind == HighDynamicRange` 才选择 HDR contract。`StandardDynamicRange`/`WideColorGamut` 选择 SDR；WinRT snapshot 失败或 OS 早于 22621 也选择 SDR，但原因分别是 `ReportedSdr`、`StateQueryFailed`、`UnsupportedOsVersion`，不能合并成一个 bool。

## 6. Wayland-only Linux

### 6.1 协议和版本

`wayland-protocols 0.32.13` 的 staging `color-management-v1.xml` 当前 interface version 是 **3**。协议仍处于 testing phase；兼容扩展会 bump interface version，不兼容变更会另开 major。实现最低要求 v2，使用 `preferred_changed2` 与 `ready2` 的 64-bit identity；compositor 只提供 v1 时显式 `ProtocolTooOld { offered: 1, required: 2 }` 并走 SDR，不能悄悄使用 deprecated 32-bit identity。[upstream protocol](https://gitlab.freedesktop.org/wayland/wayland-protocols/-/blob/main/staging/color-management/color-management-v1.xml)；[locked XML](https://docs.rs/crate/wayland-protocols/0.32.13/source/protocols/staging/color-management/color-management-v1.xml)；[generated client API](https://docs.rs/wayland-protocols/0.32.13/wayland_protocols/wp/color_management/v1/)

需要的对象：

- `wp_color_management_output_v1`：`image_description_changed` 后跟 `wl_output.done`；description immutable，变化后重新 `get_image_description`；
- `wp_color_management_surface_feedback_v1`：`preferred_changed2(hi, lo)` 表示 compositor 对整个 `wl_surface` 的新 preferred description；未知 identity 时重新 `get_preferred`；
- `wp_image_description_v1`：只能在 `ready2` 后使用；`failed(cause, msg)` 是 typed protocol failure；
- `wp_image_description_info_v1`：收集 primaries/TF/luminances/target volume，直到 `done` 才原子安装 snapshot。

### 6.2 跨屏与多 output

不要用“最后一次 `wl_surface.enter` 的 output”决定颜色。窗口可同时跨多个 output；protocol 专门提供 surface feedback，让 compositor 根据整个 surface 的可见输出、质量与性能给出 preferred description。`wl_surface.enter/leave` 和 output description 只用于诊断与验证；production resolver 以 `preferred_changed2` 为窗口级 source of truth。

description 的 `target_luminance` 是内容目标色容积的 advisory/theoretical 上限，不能当作显示器实际 headroom。headroom 使用同一 preferred parametric description 的 primary `luminances.max_lum / reference_lum`；只有收齐 `ready2 -> get_information -> done` 的一致 snapshot 才原子安装。identity 可去重 description；高低 32 位组合必须是 `(u64::from(hi) << 32) | u64::from(lo)`。

### 6.3 与 Vulkan WSI 的所有权

Vulkan `VkSurfaceFormatKHR` 本身就是精确的 `(format, colorSpace)` pair；wgpu 30 将 `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`、PQ、HLG 等映射到 `SurfaceColorSpace` 并在 swapchain create info 中写 `imageColorSpace`。[`VkSurfaceFormatKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSurfaceFormatKHR.html)；[`VkColorSpaceKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorSpaceKHR.html)

Vulkan 规范还明确说明：Wayland 上只有 `VK_COLOR_SPACE_PASS_THROUGH_EXT` 才允许应用自行创建 `wp_color_management_surface_v1` 而不触发 `surface_exists`。因此 Gravlume 在 wgpu 管理 Vulkan WSI 时：

- 只创建 read-only `wp_color_management_surface_feedback_v1` 和可选 output observers；
- **不**创建 `wp_color_management_surface_v1`，不直接 `set_image_description`；
- 实际 surface encoding 只通过 wgpu 的 exact pair 配置，由 Vulkan WSI 拥有颜色 surface signaling。

在 wgpu 没有 pass-through surface mode 的情况下，背着 WSI 再设置 Wayland surface description 会形成双 owner，应拒绝。[Vulkan `VkColorSpaceKHR` Wayland note](https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorSpaceKHR.html)

### 6.4 winit/calloop hook

`winit 0.30.13` 的 public Wayland API 只有 backend detection、`xdg_toplevel` 和 monitor native id；raw-window-handle 提供 `wl_display*`/`wl_surface*`。内部确实使用 `wayland-client`、SCTK 和 calloop，但不暴露 `Connection`、`QueueHandle<WinitState>` 或 `LoopHandle`，因此应用不能把 protocol source 插入 winit 的内部 calloop。[winit #4131](https://github.com/rust-windowing/winit/issues/4131) 截至 2026-08-12 仍是没有关联 PR 的 open enhancement：它正确识别了亮度/参考白查询缺口，但其 output getter 例子不处理跨多个 output 的窗口，production 应使用 surface preferred feedback。[locked public API](https://docs.rs/crate/winit/0.30.13/source/src/platform/wayland.rs)；[locked winit features](https://docs.rs/crate/winit/0.30.13/source/Cargo.toml)

可行且由 wayland-rs 文档支持的 integration 是 guest backend：

1. 从 raw display handle 调 `wayland_backend::client::Backend::from_foreign_display`；
2. 用 `Connection::from_backend` 包装，不拥有/关闭 winit connection；
3. 从 raw surface pointer 构造只读 feedback 所需 proxy；
4. 自建 guest `EventQueue`；winit 继续负责读 socket，应用在 `ApplicationHandler::about_to_wait` 调 `dispatch_pending`；
5. protocol callback 只更新 native monitor state 并用 `EventLoopProxy` 请求主循环处理；
6. window/connection drop 前按 protocol 顺序销毁所有 guest object，再销毁 backend。

wayland-client 文档明确把 foreign display/raw-window-handle 列为该 API 的用途；guest queue 的 `dispatch_pending` 也专门说明 host program 已读 socket 时会派发 pending events。locked winit 的 `poll_events_with_timeout` 还有一个关键行为：无限 `ControlFlow::Wait` 下，若 Wayland fd readable 但 winit 自己没有 dispatched event，会把这次唤醒视为 spurious 并在调用应用 callback 前继续等待。给 event loop 一个遥远但有限的 `WaitUntil` 后，fd readiness 仍立即返回，且该过滤条件不再成立；这不是周期轮询，也不与 winit 抢读 socket。[`wayland-client` FFI integration](https://docs.rs/crate/wayland-client/0.31.15/source/src/lib.rs)；[`Backend::from_foreign_display`](https://docs.rs/wayland-backend/0.3.16/wayland_backend/client/struct.Backend.html#method.from_foreign_display)；[guest `dispatch_pending`](https://docs.rs/crate/wayland-client/0.31.15/source/src/event_queue.rs)；[locked winit event loop](https://docs.rs/crate/winit/0.30.13/source/src/platform_impl/linux/wayland/event_loop/mod.rs)

这个 seam 已从 locked source 推导并完成 target 编译，但当前开发机是 macOS，仍需在目标 compositor 上验证协议事件、跨屏、hotplug 与清理。若实机表明 guest queue 无法可靠唤醒或清理，正确升级路径是给 winit 提交/维护一个暴露 protocol integration 的窄 patch；不是另起线程竞争读取同一个 Wayland socket，也不是接管 winit 私有 calloop。

### 6.5 Wayland HDR admission

Linux `DisplayHdrInfo` 为 unknown；只看 Vulkan pair 又无法获得 window-level preferred/reference white。HDR admission 同时要求：

- compositor advertises `wp_color_manager_v1 >= 2`；
- compositor advertises parametric descriptions，且 surface preferred parametric description 完整 ready；
- 精确 `Rgba16Float + ExtendedSrgbLinear` pair 当前仍在 capabilities；
- primary maximum/reference-white luminance 均为有限正值，且 maximum 高于 reference white。

不把 `windows_scrgb` feature 当成显示器 HDR 状态：它只表示 compositor 能替客户端创建该预定义 content description。实际 transport 合同来自 Vulkan/wgpu 精确 `ExtendedSrgbLinear` pair；Khronos 将它定义为 extended sRGB linear/scRGB，wgpu v30 官方 example 用 `value = nits / 80`。preferred description 只提供 compositor 针对整个 surface 选择的显示/观看环境参数，因此输出 white scale 是 `reference_lum / 80`。PQ 不作为自动 fallback；只有将 BT.2020/PQ/metadata 全套实现为独立 `OutputEncoding::Bt2100Pq` 后才能加入 preference list。[Vulkan `VkColorSpaceKHR`](https://docs.vulkan.org/refpages/latest/refpages/source/VkColorSpaceKHR.html) [wgpu v30 HDR shader](https://github.com/gfx-rs/wgpu/blob/v30/examples/standalone/03_hdr_surface/src/shader.wgsl)

## 7. 推荐内部 API 和事务边界

新建极窄的 `gravlume-native-display` crate，只依赖 raw handles 与 OS crates；`gravlume-render` 不导入 AppKit/WinRT/Wayland 类型，`gravlume-domain` 完全不知道 display。

```rust
pub enum LiveDynamicRange {
    Hdr {
        current_headroom: Option<f32>,
        potential_headroom: Option<f32>,
        sdr_white_nits: Option<f32>,
        peak_nits: Option<f32>,
    },
    Sdr,
    SuppressedBySystem,
    Unknown(UnknownDisplayReason),
}

pub struct DisplaySnapshot {
    pub generation: u64,
    pub dynamic_range: LiveDynamicRange,
    pub source: DisplayStateSource,
}

pub enum SdrReason {
    UserRequested,
    HdrSurfacePairMissing,
    DisplayReportedSdr,
    SystemSuppressed,
    DisplayStateUnknown(UnknownDisplayReason),
    WaylandProtocolMissing,
    WaylandProtocolTooOld { offered: u32, required: u32 },
    WaylandDescriptionFailed,
    WaylandEncodingUnverified,
}

pub enum OutputDecision {
    Hdr(OutputContract),
    Sdr {
        contract: OutputContract,
        reason: SdrReason,
    },
}
```

实际 resolver 是纯函数：

```text
resolve_output(
  policy,
  exact_surface_capabilities,
  display_snapshot,
  platform_semantics,
) -> Result<OutputDecision, NoPresentableSdrSurface>
```

`PreferHdr` 永远产生 HDR 或带原因的色彩正确 SDR；`ForceSdr` 产生 `UserRequested`。不需要 `RequireHdr` 让普通桌面应用因为 HDR 缺失而退出。真正的 fatal error 只剩 surface 连 SDR pair 都没有。

`OutputContract` 至少原子携带：

```text
surface format + color space
output encoder
reference-white scene-to-surface scale（同时作用于 scene 与 UI）
tone-map headroom/peak target
diagnostic source/reason
generation
```

更新顺序：

```text
native event(s)
  -> coalesced OutputStateDirty
  -> main-thread complete snapshot
  -> re-query SurfaceCapabilities
  -> pure resolve
  -> prepare pipeline/uniform/config
  -> wait until no acquired SurfaceTexture
  -> configure + atomically install generation
  -> redraw existing published scene
```

surface config 与 final shader encoder 不能跨 generation 混用；如果 prepare/configure 失败，保留旧完整 frame，重新 resolve SDR contract，并记录 fallback reason。输出切换不重算 trace。

## 8. Cargo feature closure 与 X11 删除

Linux target 修改目标：

```toml
egui-winit = { workspace = true, features = ["wayland"] }
winit = { workspace = true, features = ["wayland", "wayland-dlopen"] }
wayland-client = { version = "0.31", features = ["system"] }
wayland-protocols = { version = "0.32", features = ["client", "staging"] }
```

删除所有 `x11` feature。`wayland-dlopen` 只是 libwayland 的加载策略，不是 X11 fallback，可以保留；没有 Wayland/libwayland 时启动应返回明确平台错误。`wayland-client/system` 是 foreign-display API 的必要 feature，且能与 winit 已启用的 `wayland-backend/client_system` 合并。[`winit` features](https://docs.rs/crate/winit/0.30.13/source/Cargo.toml)；[`wayland-client` features](https://docs.rs/crate/wayland-client/0.31.15/source/Cargo.toml)；[`wayland-protocols` features](https://docs.rs/crate/wayland-protocols/0.32.13/source/Cargo.toml)

macOS direct dependency 使用 objc2 0.3 代际，避免把 winit 仍携带的旧 objc2 AppKit projection 暴露到新 module。Windows direct dependency使用当前已锁定的 `windows 0.62`；“wgpu 间接依赖了 windows”不等于 platform crate 可以省略 direct dependency。

依赖变更后必须审计：

```text
cargo tree -e features -p gravlume-desktop --target x86_64-unknown-linux-gnu
cargo tree -e features -p gravlume-desktop --target x86_64-pc-windows-msvc
cargo tree -e features -p gravlume-desktop --target aarch64-apple-darwin
cargo tree -i x11-dl --target x86_64-unknown-linux-gnu
cargo tree -i x11rb --target x86_64-unknown-linux-gnu
```

后两项必须为空；同时检查没有 `xkbcommon-dl/x11` feature。Wayland 自身仍可能合理依赖 `xkeysym`/xkbcommon，不能按名字误删。

## 9. 测试矩阵

### 9.1 纯 resolver/色彩测试

- exact pair selection：不能把全局 `formats` 或 color-space union 当 pair；
- `Unknown != Sdr`：输出是 SDR fallback，但 reason 必须保持 unknown cause；
- system suppression 覆盖 Apple potential/current headroom；
- Windows `reference_white_scale = sdr_white / 80`、`headroom = peak / sdr_white`，拒绝非有限/非正输入；
- Wayland v1 → `ProtocolTooOld`；v2/v3 64-bit identity 合成、去重、`ready2/failed/done` state machine；
- output generation 变化只 invalidates presentation，不 invalidates trace；
- SDR reference image 与现有 SDR 行为在 tolerance 内一致；HDR path 对 `>1` highlight 不提前 clamp；
- premultiplied gamma UI unpremultiply/decode/repremultiply 与 alpha source-over 的解析样例。

只保留可观察合同测试，不钉死 notification token 数量、私有 enum layout 或 shader 源文本。

### 9.2 GPU contract

- offscreen final pass：已知 scene/UI 输入分别验证 SDR encode 和 extended-linear output；
- output contract generation 与 pipeline/config 一致，不允许旧 encoder 画新 surface；
- HDR→SDR 切换使用同一 published scene，不提交 trace compute；
- native adapter 若不广告 HDR pair，测试只验证 typed SDR decision，不 skip 后假装 HDR 已覆盖。

### 9.3 实机平台矩阵

| 平台 | 必测变化 |
|---|---|
| macOS internal EDR | configure 后 current headroom、brightness change、suppression begin/end |
| macOS 双屏 | HDR↔SDR 拖动、窗口跨屏、拔插显示器、off-screen/miniaturized |
| Windows 11 22621+ | HDR toggle、SDR-white slider、HDR↔SDR 跨屏、event token 注销、DispatcherQueue shutdown |
| Windows 旧于 22621 | 明确 `UnsupportedOsVersion` SDR fallback，无 DXGI 偷偷替代 |
| Wayland protocol v2 | preferred change、description info、双 output straddle、output hotplug |
| Wayland protocol v3 | v2 路径兼容；未知 v3 feature 不影响已知字段 |
| Wayland 无协议/v1 | `ProtocolMissing`/`ProtocolTooOld` SDR fallback |
| Wayland lifecycle | window destroy 前 guest objects 清理；反复创建窗口；compositor disconnect |

视觉验收必须包含：SDR UI white 在 HDR 模式切换前后亮度语义稳定；高光只在 HDR contract 下超过 diffuse white；HDR/SDR 跨屏过程中不闪出 gamma 错误的一帧。无法用普通 screenshot 证明物理亮度；至少记录 OS state、surface pair、headroom/reference-white、adapter/backend 和 final output contract。高置信发布还需 HDR capture/测量设备或平台官方 HDR diagnostic 工具。

## 10. 明确拒绝的方案

- 只把 surface 改为 `Rgba16Float`；当前 tone map 和 egui encoding 仍会错。
- 让 egui 直接画非-sRGB FP16 surface；`egui-wgpu` 会选择 gamma framebuffer path。
- 把 `DisplayHdrInfo::default()` 当 SDR；它是 unknown。
- 每帧无条件轮询所有 native API；事件只需 coalesce 后按需取完整 snapshot。
- Windows 使用 `GetForCurrentView`；winit HWND 没有 CoreApplicationView。
- 为了 `CreateForWindowId` 引入 Windows App SDK/runtime/archived `windows-app` crate。
- Windows 重新实现 DXGI output enumeration；build 22621+ 已有正式 HWND WinRT interop。
- Wayland 选“last entered output”；surface feedback 才处理多 output。
- 在 wgpu Vulkan WSI 旁创建第二个 `wp_color_management_surface_v1`；会与 WSI 争夺 surface color owner。
- Wayland 缺协议时强行配置未经验证的 HDR pair，或把 PQ 当自动 fallback。
- 因 HDR/亮度/跨屏变化重跑 geodesic trace。

## 11. 推荐实施顺序

1. 建立 scene/output/UI 三份明确 color contract，拆开 egui overlay 与 scene；保持 SDR 默认并验证无回归。
2. 实现纯 `resolve_output`、`OutputContract`、presentation generation 和事务式 reconfigure。
3. macOS：objc2 notifications + wgpu EDR snapshot；交付 `ExtendedSrgbLinear`。
4. Windows：inbox DispatcherQueue + HWND `DisplayInformation` interop + `AdvancedColorInfoChanged`；交付 FP16 scRGB 与 reference-white scaling。
5. Linux：删除 X11，以只读 Wayland surface feedback guest queue 接入 parametric display state；精确 Vulkan/wgpu pair 与完整状态同时成立才启用 extended-linear HDR，否则保留 typed SDR fallback，并在具名 compositor 上补齐实机证据。
6. 只有 profiling 证明 FP16 bandwidth 是瓶颈，且完整 PQ/gamut/metadata policy 已定义后，再研究 `Rgb10a2Unorm + Bt2100Pq`。

这条顺序先解决“端到端色彩合同”这个根因，再逐平台接状态变化；不会用 surface 格式切换制造一个名义 HDR、实际 gamma/白点错误的实现。

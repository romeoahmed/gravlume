# 原生 HDR 输出决策

> **状态：已采用；Windows/Wayland 待目标平台实机发布验证。** Scene/UI/final contract、macOS EDR、Windows inbox WinRT、Wayland `color-management-v1` 状态读取和 typed SDR fallback 已进入生产。Linux 不支持 X11。

## 根因与选择

内部 `Rgba16Float` 纹理不等于 HDR 显示。旧路径在 display shader 中 tone-map 到 8-bit sRGB，因此端到端仍是 SDR。生产现在使用单一 scene contract：

- scene 是 extended-linear sRGB/scRGB，`1.0` 表示 SDR reference white；
- egui 先绘制到独立 gamma-encoded premultiplied RGBA8 target；final pass 解码并在线性空间合成；
- HDR 选择 `Rgba16Float + ExtendedSrgbLinear`，再按可靠的平台 reference white/headroom 压缩亮部；
- SDR 选择明确的 sRGB pair 和 SDR mapping；
- exact surface pair、当前 display state 或有效亮度参数缺一时返回带原因的 SDR contract，`unknown` 不冒充 HDR active。

PQ/HDR10 不是 scRGB 失败后的随手 fallback：它需要 BT.2020 gamut、PQ、绝对亮度和 metadata policy，而 wgpu 30 不管理完整 HDR metadata。

## 共享边界

`gravlume-native-display` 只产出平台无关的 `DynamicRange` snapshot，并在原生状态变化时通知桌面主循环重新查询。`gravlume-render` 消费这个 DTO 与 `SurfaceCapabilities::format_capabilities`，选择精确 format/color-space pair。renderer 不含 OS 分支，native-display 不配置 surface。

状态变化只使 presentation contract 失效，不使 geodesic candidate 失效。主线程事务式更新 pipeline/uniform/surface selection；不会因为窗口跨屏、亮度或 HDR toggle 重跑 tracing。

## 平台决定

### macOS

使用 AppKit `NSScreen` current/potential EDR headroom 和 suppression notifications；只有 current/potential/reference-white 信息完整有效时启用 HDR。输出仍由 wgpu Metal surface 承载。

### Windows

使用 inbox `Windows.Graphics.Display.DisplayInformation` 的 `AdvancedColorInfoChanged`。传统 winit HWND 不能用 `GetForCurrentView()`；Windows 11 22H2+ 通过官方 `IDisplayInformationStaticsInterop::GetForWindow(HWND, ...)` 取得 WinRT 对象。实现只保留这个最小 HWND interop projection，不引入 Windows App SDK runtime、DXGI 查询或 UWP CoreApplication 假设。需要时为当前 Win32 UI thread 创建 inbox `Windows.System.DispatcherQueue`，退出时先注销 event 再完成 queue shutdown。

### Wayland

只支持 `color-management-v1` v2+。应用读取 surface preferred feedback 和 luminance；Vulkan WSI/wgpu 仍是 presentation color-space owner，应用不对同一 `wl_surface` 再安装第二套 color description。winit 未暴露其内部 calloop/queue，因此 native-display 使用独立 guest queue，但不能竞争读取同一 socket。缺协议、缺反馈、输出未知或 transport pair 不匹配时保持 typed SDR。

## 明确拒绝

- 仅凭 FP16 intermediate 或 adapter `DisplayHdrInfo` 声称当前显示 HDR；
- Windows `DisplayInformation::GetForCurrentView()`、Windows App SDK runtime 或应用层 DXGI 轮询；
- Wayland 用最后一个 `wl_surface.enter` output 代替 surface preferred feedback；
- X11 HDR 支持；
- unknown 状态时猜测 HDR，或把 PQ 当无 metadata 的自动 fallback；
- egui 直接画入 extended-linear target 后假设 alpha/gamma 自动正确。

## 发布验证矩阵

- 纯 resolver：format/color-space pair、invalid headroom/reference white、unknown→typed SDR；
- GPU：scene-linear composition、premultiplied UI decode、HDR headroom、SDR mapping；
- macOS：同屏/跨屏、HDR toggle、system suppression、suspend/resume；
- Windows：SDR/HDR monitor、跨屏、Advanced Color toggle、DispatcherQueue shutdown；
- Wayland：协议缺失/v1/v2、多 output、preferred feedback 更新、compositor restart。

macOS 是当前已执行平台；Windows 与 Wayland 的代码可编译和静态审查不替代上述实机证据。

## 官方依据

- [wgpu 30 HDR surface guide](https://docs.rs/wgpu/30.0.0/wgpu/#surface-color-spaces-and-hdr-output) 与 [`SurfaceCapabilities`](https://docs.rs/wgpu/30.0.0/wgpu/struct.SurfaceCapabilities.html)；
- [Apple custom tone mapping](https://developer.apple.com/documentation/metal/performing-your-own-tone-mapping)；
- [Microsoft Advanced Color/HDR](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range) 与 [`GetForWindow`](https://learn.microsoft.com/en-us/windows/win32/api/windows.graphics.display.interop/nf-windows-graphics-display-interop-idisplayinformationstaticsinterop-getforwindow)；
- [Wayland color-management-v1](https://wayland.app/protocols/color-management-v1)；
- [Vulkan `VkColorSpaceKHR`](https://docs.vulkan.org/refpages/latest/refpages/source/VkColorSpaceKHR.html)。

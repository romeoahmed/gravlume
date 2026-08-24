# 连续字段 corpus 的首个独立证据切片

本文研究路线图“连续字段 corpus + 独立证据”的首个可实现切片，给出可否证假设、有限适用域、独立数学 witness 与 test-only GPU 批量协议建议；它是研究记录，不定义 production 行为、稳定 public API、fixture profile 或质量政策。当前合同仍以[数学物理](../physics.md)、[验证合同](../validation.md)、[Reference 证据](../reference-implementation.md)和 [GPU 证据](../gpu-renderer.md)为准。

**状态：建议采用首切片；批量执行 seam 可落地，独立高精度 artifact 尚未闭合。** 在独立 artifact、逐字段比较和分类 margin 全部完成前，不得把 CPU/GPU agreement 或本记录外推为路线图整项完成。

**研究方法：** 本次只交叉审阅上述 normative contract、当前 domain/reference/render 源码，以及下列 APS/AAS/A&A、W3C WGSL/WebGPU 一手来源；没有运行 Cargo，也没有生成或拟合数值 expectation。`x=640, y=12..20` 因而只是待 independent generator 认证的 seed，不是本记录新造的 scientific fixture。

## 1. 问题、假设与有限适用域

可否证假设是：对同一 immutable `Observation` 的一个具名、有限 `ImageSample` 序列，test-only GPU path 能以与样本数成正比的 buffer 执行一次 ordered batch，复用 production full Kerr–Schild WGSL-binary32 retrace 的 terminal-specific record；每个样本的 discrete terminal/branch 先与独立 witness 精确一致，随后 fresh binary32 source、transfer、phase 与 diagnostics 分字段满足[适用的 GPU gate](../validation.md#53-gpu-renderer-agreement)。最终 `RGBA16F` texture gate 不属于该假设；它也不需要 full-frame record plane、solver trait、render graph 或 production queue。

建议的首个 corpus 是 `canonical-kerr-source-edge-strip-v1`，但该名字在 artifact schema 被真实 consumer 采用前只是研究标签：

- 引用 `kerr-exterior-observation-v1`：$M=1$、$a=+0.8M$、$q_e=0$、ingoing Cartesian Kerr–Schild、`1280×720` canonical viewport；参数和十进制初值已有[唯一 normative 定义](../validation.md#3-kerr-exterior-observation-v1)。
- 引用现有 vacuum inverse-cube bolometric equatorial surface：$r\in[6M,20M]$、prograde circular emitter；物理模型及 $I_{\rm obs}=g^4I_{\rm em}$ 由[验证合同](../validation.md#32-surface-observable)定义，不在新 artifact 复制。
- 首批候选固定为中心 subpixel、`x=640, y=12..20` 的九个 case。当前源码中的有界 batch comparison 已把它们作为 source-edge seed（[`gpu_trace_tests/surface.rs`](../../crates/gravlume-render/src/gpu_trace_tests/surface.rs)）；正式 artifact 必须由独立 generator 逐个固定 terminal、edge signed margin 与 observable，不能在测试运行时按 CPU/GPU 结果筛样本。
- 支持域只包含 artifact 中列出的 case identity，不包含像素邻域，更不包含 $a<0$、$q_e\ne0$、higher-order/winding、critical curve、near-axis、near-extreme、spectral/absorbing transport 或任意 viewport。最靠近 edge 的 case 只有在独立 witness 给出非零且大于总误差证书的 margin 后才能 accepted；否则移入 typed boundary/uncertain strata，不能放宽阈值。

这个切片同时含 ordinary surface 与 source 外侧 escape，可先证明 ordered sparse capture、terminal transition、surface continuous fields 和 escape direction；它不声称已经覆盖路线图要求的 surface/capture boundary、critical 两侧、正负 spin 或 higher-order branch。

## 2. 现有模块与最小 seam

| Module | 已有深度 | 首切片仍缺的证据 |
| --- | --- | --- |
| Domain | [`Observation`](../../crates/gravlume-domain/src/scene.rs) 聚合 validated scene/view；[`ImageSample`](../../crates/gravlume-domain/src/view.rs) 绑定 pixel 与 subpixel。GPU batch 不应再发明未验证的坐标 DTO。 | corpus case identity、artifact provenance 与独立 expectation 不属于 domain model。 |
| Reference | [`ReferenceOutcome`](../../crates/gravlume-reference/src/outcome.rs) 已分离 terminal、`TraceBranchKey`、coordinate time、localized event 与 diagnostics；[`SurfaceObservable`](../../crates/gravlume-reference/src/surface.rs) 已给 source anchor、$g$ 与 radiance。 | [`ReferenceComparison`](../../crates/gravlume-reference/src/comparison.rs) 比较的是同一 Cartesian $f64$ integrator 的 regular/strict policy；它是 convergence evidence，不是独立方程/chart witness。现有 [v2 fixture](../../crates/gravlume-reference/fixtures/v2/kerr-surface-observable.toml) 也只有一个 surface case。 |
| Render | production [`SampleRetrace`](../../crates/gravlume-render/src/trace/inspection.rs) 已形成 terminal-specific sum type、固定 method identity 与 diagnostics；[`protocol.rs`](../../crates/gravlume-render/src/trace/inspection/protocol.rs) 已有 32-byte request、96-byte record 和 strict decoder。 | batch 只应是 `#[cfg(test)]` adapter；独立 artifact 与 structured comparison 不能由 renderer 自己生成。既有 full-frame [`gpu_capture.rs`](../../crates/gravlume-render/src/gpu_capture.rs) 不应成为 sparse corpus 的资源模型。 |

最小 interface 应保持 private/test-only，并把 GPU 资源、binding、dispatch 与 mapping 隐藏在深模块内：

```rust
#[cfg(test)]
fn capture_sample_corpus(
    observation: &Observation,
    samples: &[ImageSample],
) -> Result<Vec<SampleRetrace>, SampleInspectionError>;
```

它只承诺：所有 sample 属于同一 observation extent；空输入返回空输出；输出长度与顺序严格等于输入；一个 `TracePlan`/uniform snapshot 服务整个 batch；每项通过既有 strict decoder 返回与单样本 inspection 相同的 semantic `SampleRetrace`。它不接受 solver/profile 参数，不返回 wgpu handle、texture 或 full-frame index，也不形成 public artifact interface。当前内部 [`trace/inspection/corpus.rs`](../../crates/gravlume-render/src/trace/inspection/corpus.rs) 在该 seam 内完成 checked size/limit admission、linear allocation、dispatch、mapping 与 decode；production 单槽和 corpus 只以 binding element count/dispatch count 区分，共用 private kernel，不保留第二套 shader entry point 或 bindings。Outer test helper 可以 fail-fast，但 decoder failure 仍由深模块返回。

这符合“小 interface、深实现”的边界：只有一个真实 test consumer 时不增加 trait；第二个真实 consumer 出现前也没有理由引入 render graph 或 compatibility layer。

## 3. 独立数学 witness

### 3.1 为什么必须分开 discrete 与 continuous

Carter 证明 Kerr/Kerr–Newman Hamilton–Jacobi equation 可分离并产生第四守恒量，geodesic 可化为显式 quadrature（[Carter 1968](https://doi.org/10.1103/PhysRev.174.1559)）。Gralla–Lupsasca 对 pure Kerr exterior null geodesic 分类 radial/polar potential roots，并给出 manifestly real elliptic integrals 与 Jacobi-function 参数曲线（[Null geodesics of the Kerr exterior](https://doi.org/10.1103/PhysRevD.101.044032)）。这些结果允许一个与当前 Cartesian Kerr–Schild ODE 不同的 BL/Mino witness；但 root topology、turning/crossing 次数和 unwrapped azimuth winding 是 path 的离散身份，不能被“最终坐标很接近”替代。

Kerr 的 direct 与 highly bent rays 形成不同 image sequence；相邻高阶像具有独立的 rotation 与 time-delay 结构（[Gralla–Lupsasca 2020](https://link.aps.org/accepted/10.1103/PhysRevD.101.044031)）。因此 branch/winding、source phase 与 coordinate time 必须分别比较。Cunningham 的原始 Kerr disk transfer-function 工作也把 redshift 与 focusing 作为从 emitter 到 observer 的独立作用量（[Cunningham 1975](https://adsabs.harvard.edu/pdf/1975ApJ...202..788C)），不能用一个 RGB max error 代替 source、$g$ 与 radiance。

| 类别 | corpus 字段 | 独立 witness 与验收规则 |
| --- | --- | --- |
| Identity | observation/profile/case ID、pixel/subpixel、method、producer revision | canonical decimal input 必须 exact 绑定；CPU regular、CPU strict、GPU 与 independent artifact 是四个 producer，不能只凭同名 label 合并。 |
| Discrete physical | terminal；surface-edge inside/outside；initial polar side；radial turnings；equatorial crossings；signed azimuth winding | 由 separated root topology、turning sign changes、first valid event 与 unwrapped $\phi$ 推导；exact equality。任何 mismatch 立即拒绝，不再计算 continuous “综合分数”。 |
| Discrete numerical | competing-event set/ambiguity、failure flags、typed uncertainty | independent path 保存到各 competing event 的 signed margin，再按 versioned event/tie policy 推导 expected singleton/ambiguity；accepted ordinary case 要求 zero numerical flags。它们是 method evidence，不是假装成新的物理 observable。 |
| Continuous source | surface $(r,\phi_s)$；escape unit direction；source-edge signed radial margin | $\phi_s$ 使用 project ingoing oblate chart；surface 以 `hypot(Δr, r_mean·wrap(Δφ_s))`，escape 以 angle，edge 以 signed value及其误差证书比较。公式与 gate 已由[验证合同](../validation.md#52-reference-agreement)定义。 |
| Continuous transfer | frequency ratio $g$；emitted/observed bolometric intensity | $g=(-p\cdot u_{obs})/(-p\cdot u_{em})$；circular emitter 从 Kerr equatorial orbit 直接构造，可对照 [Bardeen–Press–Teukolsky 1972](https://adsabs.harvard.edu/pdf/1972ApJ...178..347B)。$I_\nu/\nu^3$ 是 invariant intensity，local frequency 为 $-k\cdot u$（[Younsi–Wu–Fuerst 2012](https://doi.org/10.1051/0004-6361/201219599)）；对频率积分后 vacuum bolometric factor 为 $g^4$。逐 semantic channel 比较，`BolometricRepeated` 三 lane 还须分别等于同一 scalar expectation。 |
| Continuous phase | coordinate-time duration | separated $t$ integral独立累积，再按 exact BL↔ingoing Kerr–Schild map 转换两端；absolute gate，不与 source phase 合并。项目 convention 与适用限制见 [`kerr-schild-mino-map.md`](kerr-schild-mino-map.md)。 |
| Continuous diagnostics | localized event residual；null/$E$/$L_z$/$\mathcal Q$ max drift | exact witness 的守恒量为零基线；GPU/CPU 各自按每个 diagnostic 的 profile ceiling 验收。小 drift 只是必要条件，不替代 terminal、branch、source 或 radiance witness。 |

### 3.2 可复算 generator 方法

首个 independent artifact 应由 repository research script 离线生成，使用至少 100 decimal digits（远高于 80 binary bits），但不进入 Cargo runtime dependency：

1. 从 canonical artifact 的十进制字符串独立重建 observer event、tetrad 与每个 camera covector；不得调用 Rust `ObservationTracer`、复制其 terminal state，或把 GPU 输出作为初值修正。
2. 将 initial state 转为 Boyer–Lindquist covector，独立计算 $E=-p_t$、$L_z=p_\phi$ 与 Carter constant；用 separated $R(r)$、$\Theta(\theta)$ 分类 roots 和 initial signs。方程与完整 exterior solution 以 [Carter 1968](https://doi.org/10.1103/PhysRev.174.1559) 和 [Gralla–Lupsasca 2020](https://doi.org/10.1103/PhysRevD.101.044032) 为一手依据。
3. 在 Mino parameter 中按 turning points 分段求解，依次检查 horizon、finite escape sphere 与每次 equatorial crossing；只有 $r\in[6M,20M]$ 的 crossing 是 surface terminal。累积 unwrapped $\phi_{BL}$、$t_{BL}$、turning/crossing counts 和 event margins，再用项目已验证的 [KS/Mino chart seam](kerr-schild-mino-map.md)转换为 $(r,\phi_s)$ 与 ingoing coordinate-time duration。
4. surface case 从 independent circular four-velocity 计算 $g$，再计算 $I_{em}=(r/6M)^{-3}$ 与 $I_{obs}=g^4I_{em}$；escape case 计算 localized state 的 affine-oriented unit coordinate direction。不得从 GPU RGB 反推任一 expectation。
5. 同一 case 以加倍 working precision、收紧 quadrature/root tolerance 和两种等价 elliptic/quadrature evaluation 重算。artifact 保存高精度十进制 expectation、离散字段、每字段 truncation/rounding bound、edge/competing-event margin、方程来源、generator revision、precision 与依赖 lock；两次结果若不能把误差界压到验收 budget 以下，该 case 必须 typed unsupported。

现有 fixture schema v1–v3 不得原地增加 corpus 语义；实现时需要新的 versioned schema（自然的下一个候选是 v4），并让 parser 拒绝 unknown field、duplicate case ID、越界 sample、非 canonical observation reference 和不完整 provenance。artifact 先作为 CPU regular/strict 与 GPU 的共同外部证据；CPU/GPU 彼此 agreement 仍只是额外一层。

## 4. WGSL/WebGPU 批量协议

### 4.1 Host-shared layout

WGSL host-shareable `f32/u32` 的 alignment/size 是 `4/4`，`vec4<f32|u32>` 是 `16/16`，而 `vec3` 是 alignment 16、size 12（[WGSL alignment and size](https://www.w3.org/TR/WGSL/#alignment-and-size)）。host-shared `f32` 以 IEEE-754 binary32、little-endian byte order 存储（[WGSL internal layout](https://www.w3.org/TR/WGSL/#internal-layout-of-values)）。因此首切片应复用 inspection ABI：

- request 是两个 `vec4` lane，共 32 bytes：pixel/extent 与 effective binary32 subpixel；
- record 是六个 `vec4` lane，共 96 bytes：metadata、branch、source/time、scene value、event diagnostics、四项 invariant drift；
- Rust 侧继续使用同序 scalar arrays、`#[repr(C, align(16))]`、`Pod` 与 compile-time size/alignment/offset assertions；不跨 seam 放 WGSL `bool`、Rust enum、`vec3` 或 implicit padding。

request/record 都是 runtime-sized storage array。WGSL 以 effective binding size 和 element stride决定 array length（[buffer binding determines runtime-sized array length](https://www.w3.org/TR/WGSL/#buffer-binding-determines-runtime-sized-array-element-count)）；runtime-sized array 不能作为普通 uniform payload，且 uniform/storage 有不同额外布局约束（[address-space layout constraints](https://www.w3.org/TR/WGSL/#address-space-layout-constraints)）。因此不需要另加 count uniform：shader 对 request `arrayLength` bounds-check，output binding 由 host 用相同 $N$ 分配。Production inspection 绑定 $N=1$，corpus 绑定请求的 $N$；两者由同一 entry point 和 auto-derived private layout 执行，不用 compatibility shader。

AoS 复用避免第二套 decoder/ABI，是这个小型 evidence corpus 的最小正确 seam；它不宣称是所有 adapter 上最优 coalescing。`vec4` 在这里证明 padding-free layout 与 component grouping，不证明硬件 SIMD、subgroup width 或 store fusion。WGSL 只有显式 subgroup operations 才定义 subgroup-level SIMT communication（[WGSL subgroup operations](https://www.w3.org/TR/WGSL/#subgroup-operations)）。若具名 benchmark 证明 96-byte AoS store 是瓶颈，可在 test-only adapter 内比较六个 `array<vec4>` SoA planes，decoded interface 保持不变；没有测量前不冻结该复杂度。

### 4.2 Dispatch、资源与可见性

建议保持已经有 Metal correctness witness 的 `8×8×1` workgroup，每个 active invocation 独占一个 request/record，linear index 超出 $N$ 立即返回；dispatch 为 `ceil(N/64),1,1`。64 invocations 低于 WebGPU core/compatibility baseline 的 256/128 上限，单维 dispatch baseline 上限是 65535，但实际提交仍必须读取 requested device limits（[WebGPU limits](https://www.w3.org/TR/webgpu/#limits)）。此前 `@workgroup_size(1)` 在真实 Metal adapter 上丢失部分 record，而 `8×8` 恢复，故首切片不能借“样本少”改回单 invocation；反例记录见 [`on-demand-sample-inspection.md`](on-demand-sample-inspection.md#gpu-protocol-与资源证据)。

每个 invocation 只读自己的 request、只写自己的 record，不使用 workgroup memory、atomics、barrier 或跨 invocation reduction，所以不存在 shader 内 producer/consumer visibility 问题。WGSL synchronization built-ins 的 execution/memory scope 都只是 `Workgroup`，`storageBarrier` 也不提供 cross-workgroup publication（[WGSL scoped operations and memory semantics](https://www.w3.org/TR/WGSL/#memory-semantics)）；首切片应通过“无共享写入”消除该需求，而不是添加无效 barrier。

提交顺序为 compute pass → copy output storage buffer 到 `COPY_DST | MAP_READ` readback → submit → 等待 producing submission → map exact readback range。WebGPU 规定每个 compute dispatch 是独立 usage scope，scope 内操作可能并发；copy command 有自己的 race validation（[WebGPU synchronization](https://www.w3.org/TR/webgpu/#synchronization)）。`MAP_READ` 只能与 `COPY_DST` 组合，`mapAsync` 要等 GPU 完成对该 buffer 的使用后才让 host 访问（[WebGPU buffer mapping](https://www.w3.org/TR/webgpu/#buffer-mapping)）；所以不在 shader 内加 barrier，host 也不能在 mapping 完成前读取。

对 $N$ 个样本，logical corpus buffers 是 `32N` request + `96N` output + `96N` readback = `224N` bytes，另有既存 uniform/pipeline/backend allocation；这不是 driver peak。分配前必须 checked-multiply，并分别验证 `u32` sample count、`maxStorageBufferBindingSize`、`maxBufferSize` 与 `maxComputeWorkgroupsPerDimension`；不能只引用标准默认值而跳过实际 requested device limits。空 corpus 在创建 zero-sized buffer 前直接返回。

### 4.3 Binary32 语义

buffer 中的 `f32` bit layout 是 binary32，不代表 runtime arithmetic 是跨 backend bit-exact IEEE-754。WGSL 不指定 rounding mode，允许部分 subnormal flush-to-zero，并允许规范列出的 reassociation/fusion 与 finite-math assumptions（[WGSL differences from IEEE-754](https://www.w3.org/TR/WGSL/#floating-point-differences)、[rounding and accuracy](https://www.w3.org/TR/WGSL/#floating-point-accuracy)、[reassociation and fusion](https://www.w3.org/TR/WGSL/#floating-point-reassociation)）。因此：

Fresh corpus record 与 production texture 是两条不同 producer 证据。WGSL 规定 `textureStore` 对写入值应用 inverse channel transfer function，而 `16float` 的写入转换是 `quantizeToF16(T)`（[texel formats](https://www.w3.org/TR/WGSL/#texel-formats)、[`textureStore`](https://www.w3.org/TR/WGSL/#textureStore-builtin)）；因此 binary32 record 不能证明最终 `RGBA16F` gate，后者必须由独立 texture-path evidence 验收。

- exact bit equality 只用于 `u32` discriminant/flags/counts、reserved zero 与显式 bitcast protocol；
- source、direction、$g$、time、radiance、residual 和 drift 都以 finite numeric value及各自 budget 比较，不比较 float bit pattern，也不跨 adapter 要求相同末位；
- strict decoder 拒绝 non-finite、unknown tag/flag 和非法 terminal-field combination；首个 corpus 不把 subnormal-dependent case 纳入 accepted domain。

## 5. 测试顺序与退出条件

实现应遵循 evidence-first 的顺序，而不是先用 GPU 结果固化 expected：

1. **Independent RED：** 新 schema/parser 测试先固定九个 case identity、exact discrete fields、continuous expectations、per-field bounds 与 provenance；unknown/duplicate/mismatched observation 必须拒绝。
2. **Batch protocol RED：** 覆盖 empty、input order、duplicate sample、不同 subpixel、最后 partial workgroup，并至少用 `N=65` 穿过 workgroup boundary；invalid extent、size overflow/limit admission 与 malformed readback 必须保守失败。
3. **CPU ladder：** regular 与 strict 各自对 independent artifact 比较；另外保留 regular/strict convergence comparison。两者不能合并成一个“CPU accepted”。
4. **GPU ladder：** batch result 逐项对 artifact 比较 exact terminal/branch，再分别比较 direction/source、$g$、time、semantic radiance、event residual 和每个 invariant drift；batch/single retrace 只作为相同 method 的回归对照，不是独立 oracle。
5. **ABI/平台：** 保留 Rust/WGSL size/offset assertions、shader creation/validation，并在具名 Metal 与 Vulkan adapter 各运行一次真实 batch；不得从向量语法推断性能。

首切片只有同时满足以下条件才关闭：

- 每个 accepted case 的 discrete fields 与 independent artifact exact 一致；edge/competing-event margin 明确大于独立 truncation bound、CPU budget 与 GPU budget 的保守合成，classifier false acceptance 为零。
- 每个 fresh binary32 observable 单独满足[验证合同](../validation.md#5-验收预算)中适用的 source、transfer、phase 与 diagnostics gate；没有 RGB max 聚合、tone map、display encoding 或 `RGBA16F` publication 参与该 comparison，最终 texture gate 仍需独立证据。
- artifact 保存 100-decimal-digit 重算/convergence、独立 chart/equation来源与 generator provenance；CPU/GPU agreement 不被描述为独立 witness。
- GPU allocation 随 $N$ 线性增长、保留 request order，且没有 full-frame record/texture、production queue、solver trait、render graph 或新 public seam。

## 6. 决策、扩展顺序与恢复条件

**决策：** 首先采用 test-only ordered batch adapter，并以 canonical source-edge strip 建立第一个 independent artifact。复用现有 `SampleRetrace` protocol 是当前最小 interface；不为一次 corpus 引入第二 record model、quality selector 或 production buffer。batch path 即使通过当前 CPU comparison，也只能标记“执行 seam 已证明”，直到 high-precision BL/Mino artifact 逐字段闭合。

后续按独立 strata 扩展，不把首切片 profile 原地改义：

1. surface/capture boundary 与 distance-to-boundary paired cases；
2. different winding/higher-order branch 和 critical curve 两侧，分别保存离散 branch 与连续 phase/radiance witness；
3. $a<0$ 直接重算 case，而不是只把 $a>0$ 结果镜像成“独立”证据；
4. near-axis/near-extreme root-degenerate cases，先实现无 false acceptance 的 classifier 与 Cartesian Kerr–Schild fallback；
5. 再讨论第二个 interactive/science-quality method 与 versioned persistent artifact consumer。

恢复/重开条件如下：

- BL/Mino generator 在 root degeneracy、axis chart 或 event competition 上不能给出小于 gate 的证书时，该 case 保持 unsupported；只有新的高精度 representation、published trajectory/author implementation 或严格 interval bound 才能重开，不能以小 invariant drift替代。
- AoS→SoA、不同 workgroup size、subgroup operation 或多 dispatch producer/consumer 只有在 correctness-approved workload 的 Metal/Vulkan profile 证明收益且重新验证可见性后重开。
- public corpus/inspector interface、solver trait、render graph 或 production full-frame semantic buffer只有第二个真实 consumer 证明当前 deep module 不足时重开；test fixture 本身不算第二个 consumer。

## 7. 一手来源

- [Carter, *Global Structure of the Kerr Family of Gravitational Fields* (1968)](https://doi.org/10.1103/PhysRev.174.1559)：Hamilton–Jacobi separability、第四守恒量与 explicit quadratures。
- [Gralla & Lupsasca, *Null geodesics of the Kerr exterior* (2020)](https://doi.org/10.1103/PhysRevD.101.044032)：root classification、real elliptic integrals 与完整 exterior null-geodesic curves。
- [Gralla & Lupsasca, *Lensing by Kerr black holes* (2020)](https://link.aps.org/accepted/10.1103/PhysRevD.101.044031)：direct/highly-bent image sequence、rotation 与 time-delay structure。
- [Bardeen, Press & Teukolsky (1972)](https://adsabs.harvard.edu/pdf/1972ApJ...178..347B)、[Cunningham (1975)](https://adsabs.harvard.edu/pdf/1975ApJ...202..788C)与 [Younsi, Wu & Fuerst (2012)](https://doi.org/10.1051/0004-6361/201219599)：Kerr circular emitter、disk transfer function 与 covariant intensity/frequency transfer。
- [WGSL specification](https://www.w3.org/TR/WGSL/)：host-shared layout、compute/workgroup memory model 与 floating-point contract。
- [WebGPU specification](https://www.w3.org/TR/webgpu/)：device limits、usage scopes、copy、mapping 与 queue/device timelines。

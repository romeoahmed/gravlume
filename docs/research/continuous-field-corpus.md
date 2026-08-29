# 连续字段 corpus 首切片：九点 semantic fields 与独立证据边界

本文记录路线图“连续字段 corpus + 独立证据”的首个切片：已经采用的 test-only ordered batch、固定九点 outer-source-edge semantic witness、当前证据能证明什么，以及仍开放的 texture、artifact 与后续科学 strata。它不定义 production 行为、public API、fixture profile 或质量政策；这些事实分别以[数学物理](../physics.md)、[验证合同](../validation.md)、[Reference 证据](../reference-implementation.md)和 [GPU 证据](../gpu-renderer.md)为准。

**状态：固定九点 source-edge corpus 的 semantic witness 已采用；统一 `RGBA16F` texture gate、持久 artifact 与其他 strata 未闭合。** Center-subpixel `(640,12..20)` 全部已有 separated BL/Mino 120/180 位独立证书、CPU regular/strict consumer 与 fresh WGSL-binary32 structured-field gate；正式证书的全 corpus maximum normalized delta 为 `3.85612445201e-94`。这只关闭当前九个具名 case 的 terminal、branch、source/escape、transfer、phase 与 diagnostics 语义字段，不把它们升级为持久 scientific fixture，不补齐九点统一 texture publication 证据，也不扩大 production 支持域。

## 1. 已采用决策与有限适用域

已经证明的工程命题是：同一 immutable `Observation` 的有限 `ImageSample` 序列，可以用随样本数线性增长的 buffer 一次有序执行，并复用 production full Kerr–Schild inspection 的 kernel、terminal-specific record 与 strict decoder。它不需要 full-frame record plane、solver trait、render graph 或 production queue。

当前 source-edge seed 固定为：

- `kerr-exterior-observation-v1`：$M=1$、$a=+0.8M$、$q_e=0$、ingoing Cartesian Kerr–Schild 与 `1280×720` canonical viewport；输入只引用[唯一规范定义](../validation.md#3-kerr-exterior-observation-v1)；
- vacuum inverse-cube bolometric equatorial surface：$r\in[6M,20M]$ 与 prograde circular emitter；模型和 $I_{\rm obs}=g^4I_{\rm em}$ 只由[验证合同](../validation.md#32-surface-observable)定义；
- center subpixel、`x=640, y=12..20` 的九个 case；[`gpu_trace_tests/surface.rs`](../../crates/gravlume-render/src/gpu_trace_tests/surface.rs)在测试运行时要求它们同时包含 Escape 与 Equatorial Surface，并逐项比较 converged reference 与 GPU fields；
- 九点的 root/event order、branch 和 continuous observable 均由[高精度 BL/Mino witness](high-precision-bl-mino-witness.md#61-corpus-precision-certificate)独立约束；其中相邻 `(640,13)/(640,14)` 分别位于 outer edge 外/内侧，给出 signed-margin classification bracket；
- 适用域仅是这组回归输入，不包含其像素邻域、$a<0$、$q_e\ne0$、higher-order/winding、critical curve、near-axis、near-extreme、spectral/absorbing transport 或任意 viewport。

当前证据层必须保持分开：

| 层                   | 当前结果                                                                                            | 不能外推为                                |
| -------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Batch protocol       | ordered sparse dispatch、partial workgroup、重复样本 multiplicity 与 single-retrace equality 已通过 | 独立物理证据或 production batch API       |
| CPU convergence      | 同一 Cartesian `f64` integrator 的 regular/strict outcome 收敛                                      | 独立 equation/chart witness               |
| GPU agreement        | fresh binary32 terminal/branch、source、transfer、phase 与 diagnostics 满足适用 gate                | CPU 与 GPU 共同正确                       |
| Texture path         | canonical v2 case 另有 `RGBA16F` 证据                                                               | 当前九点 seed 的 texture publication gate |
| Independent witness  | 九点均有可复算 BL/Mino 120/180 位证书；尚无持久 schema/artifact                                | 已关闭完整 source-edge science domain     |

## 2. Private execution seam

| Module    | 稳定职责                                                                                                                                                                                                               | 明确不拥有                                                               |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Domain    | [`Observation`](../../crates/gravlume-domain/src/scene.rs) 聚合 validated scene/view；[`ImageSample`](../../crates/gravlume-domain/src/view.rs) 绑定 pixel 与 subpixel                                                 | corpus identity、provenance 或 expectation                               |
| Reference | [`ReferenceOutcome`](../../crates/gravlume-reference/src/outcome.rs) 分离 terminal、branch、time、event 与 diagnostics；[`SurfaceObservable`](../../crates/gravlume-reference/src/surface.rs) 提供 source/$g$/radiance | 独立于自身 integrator 的 high-precision witness                          |
| Render    | [`SampleRetrace`](../../crates/gravlume-render/src/trace/inspection.rs) 与 [`protocol.rs`](../../crates/gravlume-render/src/trace/inspection/protocol.rs) 定义 terminal-specific evidence 和 strict decode             | artifact generator、public corpus interface 或 full-frame evidence plane |

内部 seam 只保证：

- 所有 sample 属于同一 observation extent；空输入不创建设备资源；
- 输出长度、顺序和重复项 multiplicity 与输入严格一致；
- 一个 sealed `TracePlan` 和 uniform snapshot 服务整个 batch；caller 不选择 solver/profile；
- 每项通过与 production 单槽相同的 private kernel、record ABI 和 strict decoder；
- 分配前受检计算线性 byte size，并核对 requested device 的 storage binding、buffer 和 dispatch limits。

实现分别位于 [`kernel.rs`](../../crates/gravlume-render/src/trace/inspection/kernel.rs)与 [`corpus.rs`](../../crates/gravlume-render/src/trace/inspection/corpus.rs)。前者由 production 单槽和 test corpus 共用；后者只在 `cfg(test)` 下拥有 batch allocation、mapping 与 ordered decode。这里不固化 private Rust 函数签名，也不把 test helper 当成第二个真实 consumer。

## 3. 独立数学 witness

### 3.1 Discrete identity 先于 continuous error

Carter 的 Hamilton–Jacobi separability 与显式 quadrature（[Carter 1968](https://doi.org/10.1103/PhysRev.174.1559)），以及 pure Kerr exterior 的 root classification 与 real elliptic solution（[Gralla–Lupsasca 2020](https://doi.org/10.1103/PhysRevD.101.044032)），允许建立不同于当前 Cartesian Kerr–Schild ODE 的 BL/Mino witness。Root topology、turning/crossing count 和 unwrapped azimuth winding 是 path identity；最终坐标接近不能替代它们。

Direct 与 highly bent Kerr rays 形成不同 image sequence，并有不同 rotation/time-delay structure（[Gralla–Lupsasca 2020](https://link.aps.org/accepted/10.1103/PhysRevD.101.044031)）。Disk transfer 也把 source、redshift 与 focusing 分开（[Cunningham 1975](https://adsabs.harvard.edu/pdf/1975ApJ...202..788C)）。因此不能用一个 RGB max error 合并 branch、source phase、frequency ratio 和 coordinate time。

| 类别                   | Artifact 字段                                                                          | 独立 witness 与验收                                                                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Identity               | observation/profile/case ID、pixel/subpixel、method、producer revision                 | exact 绑定 canonical decimal input；independent、CPU regular、CPU strict 与 GPU 是四个 producer                                               |
| Discrete physical      | terminal、edge inside/outside、initial polar side、turnings、crossings、signed winding | separated root topology、sign changes、first valid event 与 unwrapped $\phi$；exact equality                                                  |
| Discrete numerical     | competing events、ambiguity、failure flags、typed uncertainty                          | 保存每个 event 的 signed margin，再按 versioned tie policy 推导；accepted ordinary case 要求 zero flags                                       |
| Continuous source      | surface $(r,\phi_s)$、Escape unit direction、edge signed radial margin                 | surface 用 `hypot(Δr, r_mean·wrap(Δφ_s))`，escape 用 angle，edge 另带误差证书；gate 由[验证合同](../validation.md#52-reference-agreement)定义 |
| Continuous transfer    | frequency ratio、emitted/observed bolometric intensity                                 | 独立 circular four-velocity 与 $g=(-p\cdot u_{obs})/(-p\cdot u_{em})$；按 semantic channel 比较，不从 RGB 反推                                |
| Continuous phase       | coordinate-time duration                                                               | 独立累积 separated $t$ integral，并按 exact BL↔ingoing KS map 转换两端；使用 absolute gate                                                    |
| Continuous diagnostics | event residual、null/$E$/$L_z$/$\mathcal Q$ max drift                                  | 每项单独验收；小 drift 不能替代 terminal、branch、source 或 radiance witness                                                                  |

### 3.2 已采用的九点 source-edge witness

[`_source_edge.py`](scripts/src/gravlume_research/checks/bl_mino/_source_edge.py) 从规范十进制输入独立重建
canonical observer、frame 与 Photon Momentum，不导入 domain/reference crate，也不读取 CPU/GPU trace
输出。它在 BL chart 中形成 $E,L_z,\mathcal Q$，按 Mino-separated potentials 与已分类 turning
segments 求 event 顺序与 observable；KS coordinate-time/azimuth shift 同时使用 quadrature 与闭式
horizon primitive 交叉检查。

当前 research command 只接受固定 center-subpixel `(640,12..20)`：`y=12,13` 为 Escape，`y=14..20`
为 ordinary surface。两个 outside case 独立证明第一次 equatorial crossing 位于 $r_{out}$ 之外且
Escape 早于下一次 crossing；七个 inside case 独立恢复 Surface Source Anchor、Frequency Ratio、
coordinate time 与 bolometric radiance。九点全部经 120/180 decimal-digit precision doubling，exact
branch、逐 observable 结果、signed edge/event-order margin 与误差证书集中在
[高精度 BL/Mino witness](high-precision-bl-mino-witness.md)。全 corpus maximum normalized delta 为
`3.85612445201e-94`。Reference tests 只消费舍入后的 expectation，并同时约束 regular/strict；这些
named cases 不是新 fixture schema、持久 artifact、production solver 或像素邻域的授权。

### 3.3 Artifact generator

现有 research generator 已覆盖上述固定九点。继续扩展 corpus 时，它必须保持 repository 内可复算、但不进入 Cargo runtime closure：

1. 从 canonical 十进制输入独立重建 observer event、tetrad 与 camera covector，不调用 Rust tracer，也不以 GPU 输出修正初值；
2. 转为 Boyer–Lindquist covector，独立计算 $E,L_z,\mathcal Q$，用 separated $R(r),\Theta(\theta)$ 分类 roots 与 initial signs；
3. 在 Mino parameter 中分 turning segment 检查 horizon、finite escape sphere 与 equatorial crossings，累计 unwrapped $\phi_{BL}$、$t_{BL}$、branch counts 和 event margins，再经项目已验证的 [KS/Mino seam](kerr-schild-mino-map.md)转换 observable；
4. surface case 独立计算 circular four-velocity、$g$ 与 $g^4I_{em}$；escape case 计算 affine-oriented unit coordinate direction；
5. 以至少 100 decimal digits、加倍 working precision、收紧 root/quadrature tolerance 和两种等价 evaluation 重算，并保存逐字段 bound、来源、generator revision 与 dependency lock。误差界不能压到 gate 以下的 case 必须 typed unsupported。

现有 fixture schema v1–v3 不得原地增加 corpus 语义。只有持久 artifact consumer 落地时才引入新的 schema identity/version，并拒绝 unknown field、duplicate case ID、越界 sample、mismatched observation 与不完整 provenance；本记录不预占具体版本号。

## 4. WGSL/WebGPU contract

### 4.1 Layout 与 runtime array

WGSL host-shareable `f32/u32` 的 alignment/size 是 `4/4`，`vec4` 是 `16/16`，`vec3` 则是 alignment 16、size 12（[WGSL alignment and size](https://www.w3.org/TR/WGSL/#alignment-and-size)）。当前 inspection ABI 因而使用两个 `vec4` request lane（32 bytes）与六个 `vec4` record lane（96 bytes）；Rust 侧以同序 scalar arrays、`#[repr(C, align(16))]`、`Pod` 和 compile-time size/alignment/offset assertions 封闭布局，不跨 seam 放 WGSL `bool`、Rust enum、`vec3` 或 implicit padding。

Request/record 是 runtime-sized storage arrays。元素数由 effective buffer binding size 与 stride 决定，shader 可用 `arrayLength` 读取（[WGSL runtime-sized array element count](https://www.w3.org/TR/WGSL/#buffer-binding-determines-runtime-sized-array-element-count)）。Production 绑定 $N=1$，test corpus 绑定实际 $N$；同一 entry point 因此不需要 count uniform、第二套 decoder 或 compatibility shader。

### 4.2 Dispatch、资源与可见性

Shared kernel 固定 `8×8×1` workgroup，active invocation 独占一个 request/record，超出 $N$ 立即返回，dispatch 为 `ceil(N/64),1,1`。64 invocations 低于 WebGPU core/compatibility 的 256/128 baseline，但提交仍以 `Device::limits()` 返回的 requested limits 为准；不能把 adapter 的更高能力当作 device contract（[WebGPU limits](https://www.w3.org/TR/webgpu/#limits)、[wgpu `Device::limits`](https://docs.rs/wgpu/30.0.1/wgpu/struct.Device.html#method.limits)）。`@workgroup_size(1)` 的 Metal record 反例保留在[单样本采用决策](on-demand-sample-inspection.md#gpu-protocol-与资源证据)。

每个 invocation 无共享写入、workgroup memory、atomic、barrier 或 reduction，因此不需要 shader 内同步。WGSL synchronization built-ins 的 execution/memory scope 都是 Workgroup；`storageBarrier` 不能建立跨 workgroup publication（[WGSL memory semantics](https://www.w3.org/TR/WGSL/#memory-semantics)）。当前设计通过 ownership 消除同步，而不是加入无效 barrier。

对 $N$ 个样本，logical buffers 为 `32N` request + `96N` output + `96N` readback = `224N` bytes，不含 uniform、pipeline、backend allocation 或 staging。Host 在分配前 checked-multiply，并核对 `u32` count、`maxStorageBufferBindingSize`、`maxBufferSize` 与 `maxComputeWorkgroupsPerDimension`；空 corpus 在创建 zero-sized buffer 前返回。提交顺序是 compute → copy record buffer → submit/wait → map exact readback → ordered decode。

### 4.3 Binary32 与 texture evidence

Host-shared `f32` 的 byte representation 是 binary32，不代表 runtime arithmetic 跨 backend bit-exact。WGSL 允许规定范围内的 subnormal flush、reassociation/fusion，且不固定 rounding mode（[floating-point differences](https://www.w3.org/TR/WGSL/#floating-point-differences)、[accuracy](https://www.w3.org/TR/WGSL/#floating-point-accuracy)、[reassociation and fusion](https://www.w3.org/TR/WGSL/#floating-point-reassociation)）。因此只有 discriminant、flags、counts、reserved zero 和 explicit bitcast protocol 使用 exact bits；source、direction、$g$、time、radiance、residual 与 drift 都按 finite value 和各自 budget 比较。

Fresh record 与 production texture 是不同 producer。`rgba16float` storage write 还会执行 `quantizeToF16`（[WGSL texel formats](https://www.w3.org/TR/WGSL/#texel-formats)），所以 binary32 corpus 不能证明最终 `RGBA16F` publication gate。Strict decoder 也必须拒绝 non-finite、unknown tag/flag 和非法 terminal-field combination；首个 corpus 不接纳依赖 subnormal 保留的 case。

## 5. 已关闭的语义切片与开放边界

当前证据链已覆盖：

- regular sparse sequence 按 request order 与逐样本 retrace/reference 对应；
- 65 个逆序 case 穿过 partial workgroup，并以重复 request 验证 multiplicity/order；
- 固定九点先由 120/180 位独立 BL/Mino witness 约束 exact path identity、edge/event-order margin 与
  terminal-specific continuous fields；正式证书的全 corpus maximum normalized delta 为
  `3.85612445201e-94`；
- Reference consumer 对九点分别要求 regular/strict 满足 exact branch，以及 Escape
  position/direction/time 或 Surface source/frequency/time/intensity 的 semantic gates；
- ordered GPU consumer 先要求 `y=12,13` 为 Escape、`y=14..20` 为 Surface，再逐 terminal 比较 fresh
  binary32 branch、source/transfer、time 与 diagnostics；
- canonical `(640,16)` 另有已有 GPU/texture tests，形成独立 witness → CPU regular/strict → fresh
  binary32 → `RGBA16F` 的完整纵向链；其他八点没有最后一层统一 texture gate；
- production $N=1$ 与 test $N>1$ 使用同一 kernel、record ABI 和 strict decoder。

因此已关闭的是这九个具名 sample 的 semantic fields，不是完整连续字段路线图。仍缺：持久独立
schema/artifact；具名 Metal/Vulkan 双平台 batch 证据；九点统一的最终 `RGBA16F` texture gate；以及
surface/capture boundary、different winding、critical/higher-order branch、negative-spin、near-axis、
near-extreme 和路线图其余 strata。

采用本语义切片所需的条件已经同时满足：

- 每个 accepted case 的 discrete fields 与 independent BL/Mino certificate exact 一致，edge/event margin 大于保守合成误差，classifier false acceptance 为零；
- 每个 continuous field 单独满足适用 gate，不经过 tone map、display encoding、RGB aggregate 或 texture path 替代；
- 可复算 generator 保存独立 equation/chart、precision convergence、逐字段 bound 与 provenance；
- GPU batch 保持线性有界、输入有序和 private/test-only，不引入 production queue、solver trait、render graph 或 full-frame plane。

持久 artifact 是下一层 evidence-product seam，而不是倒推本轮 semantic witness 状态的前置条件；只有
出现真实 consumer 后才应版本化其 schema。最终 `RGBA16F` publication 也必须作为不同 producer 单独验收。

## 6. 九点 semantic fields 的采用与恢复条件

### 6.1 选择：补齐当前九点 source-edge field

本次采用保持 observation、surface、viewport 与 test-only ordered execution seam 不变，为
`(640,12)/(640,15)/(640,17..20)` 补齐与已有三点同型的独立 expectation 和 signed margin，并让 Rust
regular/strict 与 fresh GPU record 消费。GPU test 原已把 `y=12..20` 作为一个有序 seed 执行；若只认证
其中任意真子集，就会留下“同一 batch 中部分有 independent identity、部分只有 CPU/GPU agreement”
的混合证据。补齐六点后形成一条跨 outer edge 的有符号连续场，同时没有引入新的 observation、root
topology、transport 或 GPU protocol。

本轮没有选择后续 strata：

| 候选 | 额外数学/实现义务 | 本轮结论 |
| ---- | --------------- | -------- |
| 补齐六点 source-edge field | 与已采用 pair 相同的 pure-Kerr simple radial/polar turning segments；复用同一 BL/Mino equations、CPU outcome 与 private GPU batch | **已采用**；只扩 evidence data 与私有 generator aggregation |
| surface/capture boundary | 需要新的 Physical Scene、horizon/surface competing-event order 与独立 margin | 后续独立 stratum，不混入本切片 |
| different winding / higher-order / critical 两侧 | 需要多 turning segment、distance-to-critical 标签和 case-specific tolerance；direct 与 highly bent rays 不是同一路径族 | 后续独立 stratum；依据 [Kerr lensing 的 image sequence](https://arxiv.org/abs/1910.12873) |
| $a<0$、near-axis、near-extreme | 当前 witness 把 canonical $a=+0.8$、simple roots 与非轴 chart 写入适用域；改变它们必须重新认证 physical-spin、emitter branch 与 root conditioning | 后续独立 stratum；不能由正自旋局部 stencil 外推 |

Carter separability 只保证可写成 quadrature，不保证任意 root topology 可复用同一数值图；root
classification 与 manifestly-real 分段形式见 [Carter 1968](https://doi.org/10.1103/PhysRev.174.1559)和
[Gralla–Lupsasca 2020](https://arxiv.org/abs/1910.12881)。因此本切片选择“同拓扑闭合”，不是把 generator
抽象成通用 Kerr solver。

### 6.2 120/180 位采用证书

锁定环境中的统一 `gravlume-research bl-mino-surface` 调用
[`_source_edge.py`](scripts/src/gravlume_research/checks/bl_mino/_source_edge.py) 消费私有 separated equations，对固定九点做
120/180 decimal-digit 完整重算；全 corpus maximum normalized delta 为
`3.85612445201e-94`。下表列出本次补齐六点的逐 case 结果。Signed margin 定义为
$20M-r_{\rm first\ crossing}$（Escape）或 $20M-r_{\rm source}$（Surface）；负号表示 crossing 在
outer edge 外，正号表示 surface hit 在 edge 内。`maximum normalized delta` 对 generator 声明的全部
semantic scalar/vector lanes 取
$|x_{120}-x_{180}|/\max(1,|x_{180}|)$ 的最大值。

| Sample | Terminal | signed margin / $M$ | maximum normalized delta |
| ------ | -------- | -----------------: | -----------------------: |
| `(640,12)` | Escape | `-0.164713144840` | `2.34e-94` |
| `(640,15)` | Surface | `+0.221771201618` | `8.39e-117` |
| `(640,17)` | Surface | `+0.476238776365` | `5.12e-116` |
| `(640,18)` | Surface | `+0.602528680428` | `4.32e-116` |
| `(640,19)` | Surface | `+0.728194882195` | `2.04e-111` |
| `(640,20)` | Surface | `+0.853241495437` | `5.53e-107` |

六点都满足当前具名 identity：initial polar side `positive`、一个 radial turning、一个 polar turning、
signed winding `0`；Escape 在 terminal 前有一次 equatorial crossing，Surface terminal 前为零次。
`(640,12)` 的 Escape-before-next-crossing Mino margin 约为 `0.224620`，为正。连同已采用的
`(640,13)/(640,14)/(640,16)`，九点 signed margin 随 `y=12..20` 严格递增，符号只在相邻
`13/14` 之间改变。正式统一证书的 precision delta 远小于 `1e-80`；它是可复算的 semantic witness，
但不是持久 schema/artifact。mpmath precision context、quadrature 与 root verification 的能力和限制见
[官方文档](https://mpmath.org/doc/1.3.0/)。

### 6.3 采用时的 RED 与最小 GREEN

唯一新增行为属于 research generator；production tracer 和 shader 已经能执行九点。因此采用前的 RED
固定为 Python 的 deterministic internal-seam test：

```text
test_source_edge_corpus_orders_cases_across_the_outer_edge
```

它要求 private corpus builder 返回固定顺序 `(640,12)..(640,20)`，terminal 序列为
`Escape, Escape, Surface×7`，所有 case precision 一致，signed margins 严格递增且只在 `13/14`
之间变号。采用前的旧实现只有 canonical builder 与 adjacent-pair builder，因此 RED 因缺少该 interface
失败，而不是用错误 expectation 人为制造失败。最小 GREEN 把相同 topology 的具名 cases 聚合成 typed
private tuple，并复用已有 surface/escape computation 与 per-case validation；该测试现已通过。pytest
使用允许的最低高精度做快速 deterministic contract；120/180 位端到端证书仍由 scientific CLI 承担，
pytest 不重复替代它。

Rust 是 evidence consumer 而非新 production 行为：把现有 independent BL/Mino test 扩为九点 table，
对六个新 case 消费舍入后的 expectation；已有 ordered GPU test 继续通过同一 `ObservationTracer` outcome
逐项绑定 fresh record。测试只观察 terminal-specific outcome，不冻结 quadrature helper、打印格式、pass
count 或 shader 内部步骤。

### 6.4 已通过的逐层验收

| 层 | 必须满足 |
| -- | -------- |
| Independent BL/Mino | case ID/order 与 discrete identity exact；120/180 位逐 field maximum normalized delta `<1e-80`；initial null、Mino constraint 与 chart primitive residual 各 `<10^{15-p}`；surface/edge/event-order margin 有正确符号，所有字段 real finite |
| Classification | 九点 signed margin 严格递增；`y=12,13` 为 Escape、`y=14..20` 为 Surface；每个 accepted case 的 `|edge margin|` 大于 independent bound、CPU `2e-9 M` event-position gate 与 GPU `5e-3 M` surface-position gate 的保守合成值；最小现有 margin 是 `(640,13)` 的 `0.03524M` |
| CPU regular/strict | terminal 与 branch exact；event position/Source Anchor `2e-9 M`、Escape direction `2e-9 rad`、Frequency Ratio relative `2e-9`、travel time `2e-8 M`、四项 drift 各 `5e-9`；独立 emitted/observed bolometric intensity 使用现有 `1e-12` absolute witness gate |
| Fresh WGSL binary32 | terminal/branch exact且 zero false acceptance；Escape/source angle `3.82e-4 rad`、surface radius `5e-3 M`、Frequency Ratio relative `2e-3`、travel time `1e-3 M`、event residual `5e-3`、四项 drift 各 `≤0.05`、structured bolometric radiance relative `2e-3`；non-finite、flags、`Uncertain` 或非法 record 一律失败 |

这些 gate 直接引用[验证合同](../validation.md#5-验收预算)，不得为了新点原地放宽。Fresh binary32
record 与最终 `RGBA16F` texture 是不同 producer；本切片只关闭九点的 terminal-specific semantic
fields，不声称补上 texture publication gate，也不改变 canonical v2 已有 texture evidence。

### 6.5 Stable seam、WGSL layout 与并行边界

- Research module 的 interface 仍是单一 `bl-mino-surface` check；case aggregation、root segments 与
  certificate types 保持 private，不增加任意 pixel/observation 的 scientific API。一个固定 generator
  consumer 不足以证明新的 public seam。
- Reference expectations 留在具名 test table，不修改 fixture v1–v3 schema。没有持久 artifact consumer
  前不预占 artifact version、producer envelope 或 migration policy。
- Render 继续复用 `cfg(test)` 的 ordered corpus helper、同一个 `SampleInspectionKernel`、96-byte record
  与 strict decoder；不增加 production queue、count uniform、solver trait、render graph 或 full-frame
  semantic plane。
- 九点只需一个 `8×8×1` workgroup，logical request/output/readback 为
  `9×(32+96+96)=2016` bytes；每个 active invocation 独占 `index` 对应的 request/record，55 个越界 lane
  在 `arrayLength` guard 返回。这里没有共享写入、reduction 或跨 workgroup producer/consumer，加入
  barrier/subgroup 不会增加证据。WGSL 规定 runtime-array length 由 binding 决定，且同步 built-ins 的
  memory/execution scope 为 Workgroup（[runtime array](https://www.w3.org/TR/WGSL/#buffer-binding-determines-runtime-sized-array-element-count)、
  [synchronization](https://www.w3.org/TR/WGSL/#memory-semantics)）。
- `vec4` lanes 是 host-shareable layout 合同，不是跨设备 SIMD/vectorization 保证。WGSL `f32/u32` 为
  alignment/size `4/4`、`vec4` 为 `16/16`；Rust 必须同时保留 `#[repr(C, align(16))]` 与 size/alignment/
  offset assertions，因为 `align` 自身不保证 field order（[WGSL layout](https://www.w3.org/TR/WGSL/#alignment-and-size)、
  [Rust type layout](https://doc.rust-lang.org/reference/type-layout.html#the-alignment-modifiers)）。
- Pixel/index/discriminant/branch words exact；continuous fields 只按上述 semantic budgets 比较。WGSL 不固定
  rounding mode，允许 subnormal flush、reassociation 与 fusion；新 case 不能依赖 bit-exact arithmetic 或
  subnormal 保留（[floating-point differences](https://www.w3.org/TR/WGSL/#floating-point-differences)、
  [reassociation and fusion](https://www.w3.org/TR/WGSL/#reassociation-and-fusion)）。

本切片采用后，按独立 strata 扩展：surface/capture boundary；different winding/higher-order branch
与 critical curve 两侧；独立重算的 $a<0$ case；最后是 near-axis/near-extreme root degeneracy。每层都
先保存 discrete identity，再比较 continuous fields。

- Generator 在 root degeneracy、axis chart 或 event competition 上不能给出小于 gate 的证书时，该 case 保持 unsupported；只能由更高精度 representation、published independent implementation 或严格 interval bound 重开。
- AoS→SoA、不同 workgroup size、subgroup operation 或多 dispatch producer/consumer，只有 correctness-approved workload 的 Metal/Vulkan profile 证明净收益且重新验证可见性后重开。
- Public corpus/inspector interface、solver trait、render graph 或 production full-frame semantic buffer，只有第二个真实 consumer 证明当前 deep module 不足时重开；test fixture 不算第二个 consumer。

## 7. 一手来源

- [Carter, _Global Structure of the Kerr Family of Gravitational Fields_ (1968)](https://doi.org/10.1103/PhysRev.174.1559)：Hamilton–Jacobi separability、第四守恒量与 explicit quadratures；
- [Mino, _Perturbative Approach to an Orbital Evolution around a Supermassive Black Hole_ (2003)](https://doi.org/10.1103/PhysRevD.67.084027)：分离 radial/polar motion 使用的 Mino parameter；
- [Gralla & Lupsasca, _Null geodesics of the Kerr exterior_ (2020)](https://doi.org/10.1103/PhysRevD.101.044032)：root classification、real elliptic integrals 与 exterior null-geodesic curves；
- [Gralla & Lupsasca, _Lensing by Kerr black holes_ (2020)](https://link.aps.org/accepted/10.1103/PhysRevD.101.044031)：direct/highly-bent image sequence、rotation 与 time delay；
- [Bardeen, Press & Teukolsky (1972)](https://adsabs.harvard.edu/pdf/1972ApJ...178..347B)、[Cunningham (1975)](https://adsabs.harvard.edu/pdf/1975ApJ...202..788C)与 [Younsi, Wu & Fuerst (2012)](https://doi.org/10.1051/0004-6361/201219599)：Kerr circular emitter、disk transfer 与 covariant intensity/frequency transfer；
- [WGSL specification](https://www.w3.org/TR/WGSL/)：host-shareable layout、runtime array、compute memory model 与 floating-point contract；
- [WebGPU specification](https://www.w3.org/TR/webgpu/)：device limits、usage scopes、copy、mapping 与 device timeline；
- [mpmath documentation](https://mpmath.org/doc/1.3.0/)：arbitrary-precision context、quadrature、root finding 与 precision-doubling 复算基础；依赖解析见[统一 Python 研究工具链](python-research-tooling.md)。

# 连续字段 corpus 首切片：执行 seam、相邻 edge pair 与独立证据边界

本文记录路线图“连续字段 corpus + 独立证据”的首个切片：已经采用的 test-only ordered batch、相邻 outer-source-edge pair、当前 source-edge seed 能证明什么，以及仍需独立 artifact 才能关闭的科学证据。它不定义 production 行为、public API、fixture profile 或质量政策；这些事实分别以[数学物理](../physics.md)、[验证合同](../validation.md)、[Reference 证据](../reference-implementation.md)和 [GPU 证据](../gpu-renderer.md)为准。

**状态：执行 seam、一个 canonical witness 与一对相邻 outer-edge witness 已采用；完整 corpus 证据待闭合。** 当前 seed 的九点都提供 CPU regular/strict convergence 与 fresh WGSL-binary32 agreement；其中 `(640,13)` Escape、`(640,14)` Equatorial Surface 和与 v2 fixture 重合的 `(640,16)` ordinary surface 另有 separated BL/Mino high-precision witness。其余六点仍没有独立 expectation、分类 margin 和最终 texture-path gate，整个 seed 不能被称为 scientific fixture，也不能扩大 production 支持域。

## 1. 已采用决策与有限适用域

已经证明的工程命题是：同一 immutable `Observation` 的有限 `ImageSample` 序列，可以用随样本数线性增长的 buffer 一次有序执行，并复用 production full Kerr–Schild inspection 的 kernel、terminal-specific record 与 strict decoder。它不需要 full-frame record plane、solver trait、render graph 或 production queue。

当前 source-edge seed 固定为：

- `kerr-exterior-observation-v1`：$M=1$、$a=+0.8M$、$q_e=0$、ingoing Cartesian Kerr–Schild 与 `1280×720` canonical viewport；输入只引用[唯一规范定义](../validation.md#3-kerr-exterior-observation-v1)；
- vacuum inverse-cube bolometric equatorial surface：$r\in[6M,20M]$ 与 prograde circular emitter；模型和 $I_{\rm obs}=g^4I_{\rm em}$ 只由[验证合同](../validation.md#32-surface-observable)定义；
- center subpixel、`x=640, y=12..20` 的九个 case；[`gpu_trace_tests/surface.rs`](../../crates/gravlume-render/src/gpu_trace_tests/surface.rs)在测试运行时要求它们同时包含 Escape 与 Equatorial Surface，并逐项比较 converged reference 与 GPU fields；
- 其中相邻 `(640,13)/(640,14)` 分别位于 outer edge 外/内侧；前者的第一次 equatorial crossing 位于 source domain 外，随后 Escape，后者在 outer edge 内终止为 surface。两者的 root/event order、branch 和 continuous observable 由[高精度 BL/Mino witness](high-precision-bl-mino-witness.md#62-outer-edge-pair)独立约束；
- 适用域仅是这组回归输入，不包含其像素邻域、$a<0$、$q_e\ne0$、higher-order/winding、critical curve、near-axis、near-extreme、spectral/absorbing transport 或任意 viewport。

当前证据层必须保持分开：

| 层                   | 当前结果                                                                                            | 不能外推为                                |
| -------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Batch protocol       | ordered sparse dispatch、partial workgroup、重复样本 multiplicity 与 single-retrace equality 已通过 | 独立物理证据或 production batch API       |
| CPU convergence      | 同一 Cartesian `f64` integrator 的 regular/strict outcome 收敛                                      | 独立 equation/chart witness               |
| GPU agreement        | fresh binary32 terminal/branch、source、transfer、phase 与 diagnostics 满足适用 gate                | CPU 与 GPU 共同正确                       |
| Texture path         | canonical v2 case 另有 `RGBA16F` 证据                                                               | 当前九点 seed 的 texture publication gate |
| Independent artifact | 无持久 schema/artifact；`(640,13)/(640,14)/(640,16)` 有可复算 BL/Mino witness                     | 已关闭完整 source-edge science domain     |

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

### 3.2 已采用的 canonical 与 outer-edge pair witness

[`verify_bl_mino_surface_witness.py`](scripts/verify_bl_mino_surface_witness.py) 从规范十进制输入独立重建
canonical observer、frame 与 Photon Momentum，不导入 domain/reference crate，也不读取 CPU/GPU trace
输出。它在 BL chart 中形成 $E,L_z,\mathcal Q$，按 Mino-separated potentials 与已分类 turning
segments 求 event 顺序与 observable；KS coordinate-time/azimuth shift 同时使用 quadrature 与闭式
horizon primitive 交叉检查。

当前 external seam 只接受 `(640,16,0.5,0.5)` ordinary surface 和固定
`(640,13,0.5,0.5)/(640,14,0.5,0.5)` outer-edge pair。Pair 的 outside case 独立证明第一次
equatorial crossing 位于 $r_{out}$ 之外且 Escape 早于下一次 crossing；inside case 独立恢复 surface
Source Anchor、Frequency Ratio、coordinate time 与 bolometric radiance。两者和 canonical case 都经
120/180 decimal-digit precision doubling，exact branch、逐 observable 结果、signed edge/event-order
margin 与误差证书集中在[高精度 BL/Mino witness](high-precision-bl-mino-witness.md)。Reference tests 只
消费舍入后的 expectation，并同时约束 regular/strict；这些 named cases 不是新 fixture schema、持久
artifact、production solver 或整个九点 stencil 的授权。

### 3.3 Artifact generator

现有 research generator 已覆盖上述三个 named cases。继续扩展 corpus 时，它必须保持 repository 内可复算、但不进入 Cargo runtime closure：

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

## 5. 当前证据与关闭条件

当前 GPU tests 已覆盖：

- regular sparse sequence 按 request order 与逐样本 retrace/reference 对应；
- 65 个逆序 case 穿过 partial workgroup，并以重复 request 验证 multiplicity/order；
- 九点 outer source-edge seed 同时包含 Escape 与 Surface；每点先通过 CPU regular/strict comparison，再逐 terminal 比较 fresh GPU branch、source/transfer、time 与 diagnostics；
- 相邻 `(640,13)/(640,14)` 先由 120/180 位独立 BL/Mino witness 约束 outside/inside classification、
  path identity、edge/event-order margin 和 terminal-specific continuous fields，再由 CPU regular/strict
  消费 Escape position/direction/time 与 Surface source/frequency/time/radiance，并由同一 ordered GPU
  corpus 的 fresh binary32 record 绑定 exact terminal/branch、逐字段对 reference 验收；
- canonical `(640,16)` 由独立 BL/Mino witness 约束 path identity 与 continuous surface observable，
  再经 CPU regular/strict 与已有 GPU/texture tests 形成一条纵向链；
- production $N=1$ 与 test $N>1$ 使用同一 kernel、record ABI 和 strict decoder。

仍缺：独立 schema/artifact；`(640,12)/(640,15)/(640,17..20)` 六点的 BL/Mino expectation、
edge/competing-event margin 与逐字段 bound；具名 Metal/Vulkan 双平台 batch 证据；相邻 pair 与九点
统一的最终 `RGBA16F` texture gate；以及 negative-spin、critical/higher-order branch 和路线图其余
strata。

这个切片只有在以下条件同时满足后才能关闭：

- 每个 accepted case 的 discrete fields 与 independent artifact exact 一致，edge/event margin 大于保守合成误差，classifier false acceptance 为零；
- 每个 continuous field 单独满足适用 gate，不经过 tone map、display encoding、RGB aggregate 或 texture path 替代；
- artifact 保存独立 equation/chart、precision convergence、逐字段 bound 与 generator provenance；
- GPU batch 保持线性有界、输入有序和 private/test-only，不引入 production queue、solver trait、render graph 或 full-frame plane。

## 6. 扩展与恢复条件

下一最小切片是保持 observation、surface、viewport 与 test-only ordered execution seam 不变，为
`(640,12)/(640,15)/(640,17..20)` 生成同型独立 expectation 和 signed margin，并让 Rust
regular/strict 与 fresh GPU record 消费；这只关闭当前九点 seed 的 semantic fields，不顺带冻结持久
artifact schema 或改变 WGSL ABI。随后才按独立 strata 扩展：surface/capture boundary；different
winding/higher-order branch 与 critical curve 两侧；独立重算的 $a<0$ case；最后是
near-axis/near-extreme root degeneracy。每层都先保存 discrete identity，再比较 continuous fields。

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
- [mpmath 1.3 documentation](https://mpmath.org/doc/1.3.0/)：arbitrary-precision context、quadrature、root finding 与 precision-doubling 复算基础。

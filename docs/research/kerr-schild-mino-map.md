# Kerr–Schild ↔ Boyer–Lindquist/Mino 零步 seam

本文记录 KS/BL/Mino 局部变换的符号与数值证据，不定义 production integrator；normative chart convention 以[数学物理合同](../physics.md)为准，当前采用状态以源码和实现证据为准。

**状态：数学 seam 已采用；numerical Mino 已拒绝。** Physical-spin convention 已修复，Gate A/B 通过；数值 reciprocal-Mino 的 trajectory/f32 Gate C/D 随后被高分辨率反例否决。本文只封闭 Kerr–Newman chart convention 与 pure-Kerr Mino state 的局部零步 seam；trajectory 与 step-factor 证据单独记录在 [Mino step selection](mino-step-selection.md)，避免把坐标恒等式误写成整个积分器的证明。

可复现的 SymPy 证明实现为
[`kerr_schild_map.py`](scripts/src/gravlume_research/checks/kerr_schild_map.py)。它是 docs 下的研究工具，
不构成 runtime/build 依赖；同一检查同时保留 legacy RED witness 与 corrected GREEN
proof，避免只留下临时目录中不可复现的证据。

## 结论

1. 对外参数始终是固定右手 Cartesian orientation 中的物理自旋
   \(a=J/M\)。令 \(s=+1\) 为 ingoing、\(s=-1\) 为 outgoing；正确的 chart spatial
   twist 是 \(a_s=s a\)，而 radius、\(\Sigma\)、\(\Delta\) 仍只依赖物理 \(a^2\)。
2. legacy outgoing 把 spatial twist 留在 \(+a\)，同时整体翻转 principal spatial
   covector。它精确等价于标准 Boyer–Lindquist 自旋 \(-a\)，所以正 \(a\) target 的
   \(g_{t\phi}\) 反号。这是连续模型错误，不是积分误差。
3. corrected convention 在两 chart 中都精确 pull back 到同一个 physical-spin
   Kerr–Newman BL metric；position、tangent、canonical covector、Hamilton norm、
   \(E,L_z,\mathcal Q\) 与 affine/Mino scale 的局部 round trip 全部闭合。
4. BL Jacobian 仍含 \(1/\Delta\)，axis 上 \(\phi\) 仍退化。因此这次修复只解除
   physical-spin blocker，不把 BL/Mino 变成 horizon-crossing 或 axis-regular
   replacement。

## 1. 规范 chart convention

仓库 canonical state 是

\[
Y_{\rm CKS}=(t_s,x,y,z,p_{t_s},p_x,p_y,p_z),\qquad
H=\tfrac12 g^{\mu\nu}p_\mu p_\nu,
\]

signature 为 \((-+++)\)。momentum 是 future-directed physical covector；交互相机沿
negative affine increment 回溯，不把 momentum 本身翻成 past-directed。

令 \(a_s=s a\)。chart-handed oblate map 是

\[
x=(r\cos\phi_s-s a\sin\phi_s)\sin\theta,
\quad
y=(r\sin\phi_s+s a\cos\phi_s)\sin\theta,
\quad z=r\cos\theta.
\tag{1}
\]

Cartesian Kerr–Schild null covector 为

\[
l_\mu=\left(
1,
\frac{s r x+a y}{r^2+a^2},
\frac{s r y-a x}{r^2+a^2},
\frac{s z}{r}
\right).
\tag{2}
\]

将 (2) 拉回 \((t_s,r,\theta,\phi_s)\) 得

\[
l=dt_s+s\,dr-a\sin^2\theta\,d\phi_s.
\tag{3}
\]

注意 (1) 的 twist 随 chart 变号，而 (2) 中 azimuthal spin term 不随 chart 变号。
这不是把 outgoing 的 public spin 存成 \(-a\)；`spin_m()`、uniform 与 UI 仍传物理
\(a\)。observer placement 使用 (1)，metric/tetrad 则从同一个 corrected geometry
派生，因而无需另一份 observer-side sign shim。

Kerr–Newman metric 为

\[
g_{\mu\nu}=\eta_{\mu\nu}
+\frac{2Mr-q_e^2}{\Sigma}l_\mu l_\nu,
\quad
\Sigma=r^2+a^2\cos^2\theta,
\quad
\Delta=r^2-2Mr+a^2+q_e^2.
\tag{4}
\]

由 (1) 得平坦背景

\[
d\bar s^2=-dt_s^2+dr^2+\Sigma d\theta^2
-2s a\sin^2\theta\,dr\,d\phi_s
+(r^2+a^2)\sin^2\theta\,d\phi_s^2.
\tag{5}
\]

因此两 chart 都有同一个

\[
g_{t_s\phi_s}
=-\frac{(2Mr-q_e^2)a\sin^2\theta}{\Sigma}.
\tag{6}
\]

## 2. 旧实现的 RED witness

修复前的 outgoing map 仍使用 ingoing twist：

\[
x=(r\cos\phi-a\sin\phi)\sin\theta,\qquad
y=(r\sin\phi+a\cos\phi)\sin\theta,
\]

而 principal spatial covector 被整体乘以 \(s=-1\)。在 spheroidal coordinates 中
这给出

\[
l_{\rm legacy}=dt_s+s\left(dr-a\sin^2\theta\,d\phi_s\right),
\]

从而 \(g_{t\phi}=-s(2Mr-q_e^2)a\sin^2\theta/\Sigma\)。任何只依赖 \(r\) 的
\(t,\phi\) shift 都不能改变固定 \(r\) 的这个分量，所以 legacy outgoing 只能对应
BL spin \(-a\)。

持久化脚本在 pure Kerr 上独立构造该 legacy metric/Jacobian，并对 physical \(+a\)
BL target 精确得到

\[
\left(J_{\rm legacy}^{T}g_{\rm legacy}J_{\rm legacy}
-g_{\rm BL}(+a)\right)_{t\phi}
=\frac{4Mra\sin^2\theta}{\Sigma},
\tag{7}
\]

其 support 只有对称的 \((t,\phi)\)、\((\phi,t)\) 两项。代入
\((M,r,a,\sin^2\theta)=(1,5,2/3,3/7)\) 精确为 \(360/1591\)。这项必须保持
`RED_AS_EXPECTED`，防止未来退回旧 handedness。

## 3. Boyer–Lindquist 零步 seam

corrected chart 与同一个 physical-spin BL chart 的 differential 是

\[
\boxed{
dt_s=dt_B+s\frac{2Mr-q_e^2}{\Delta}dr,
\qquad
d\phi_s=d\phi_B+s\frac a\Delta dr
}.
\tag{8}
\]

Campanelli et al. 的 ingoing/outgoing Kerr transformations 分别带相反的 azimuth
shift；Adamo–Newman 给出 Kerr–Newman BL/null/oblate/Cartesian KS 形式。两篇来源的
signature 或 Cartesian azimuth orientation 与本项目不完全相同，因此 (8) 最终由
(1)–(6) 的项目 convention 作 exact metric pullback 固定，而不是逐字移植符号。

位置的局部逆映射为

\[
r=r(x,y,z),\qquad \mu=\cos\theta=z/r,
\tag{9}
\]

\[
e^{i\phi_s}=\frac{x+i y}{(r+i s a)\sin\theta},\qquad
\phi_s=\operatorname{atan2}(r y-s a x,\,r x+s a y).
\tag{10}
\]

若全局 state 需要绝对 \(t_B,\phi_B\)，必须选择积分常数。以 observer radius
\(r_0\) 锚定 \(T(r_0)=\Phi(r_0)=0\)，令

\[
T'=\frac{2Mr-q_e^2}{\Delta},\qquad \Phi'=\frac a\Delta,
\]

则

\[
t_B=t_s-sT(r;r_0),\qquad \phi_B=\phi_s-s\Phi(r;r_0).
\tag{11}
\]

这些 primitives 只能在不跨 \(\Delta=0\) 的同一连通区使用；near-extremal 时应使用
专门极限或累计 local differential，不能直接相减两个病态 log。

### 3.1 切向量

令点号表示 canonical affine parameter \(\sigma\) 的导数。由 (8)：

\[
\dot t_B=\dot t_s-s\frac{2Mr-q_e^2}{\Delta}\dot r,
\quad
\dot\phi_B=\dot\phi_s-s\frac a\Delta\dot r,
\quad
\dot\theta_B=\dot\theta.
\tag{12}
\]

Cartesian → spheroidal 局部导数为

\[
\dot r=\nabla r\cdot\dot{\mathbf x},\qquad
\dot\mu=\frac{r\dot z-z\dot r}{r^2},
\tag{13}
\]

\[
\dot\phi_s=
\frac{x\dot y-y\dot x}{x^2+y^2}
+s\frac a{r^2+a^2}\dot r.
\tag{14}
\]

反向构造 Cartesian tangent 使用

\[
\partial_r\mathbf x=(\sin\theta\cos\phi_s,
\sin\theta\sin\phi_s,\cos\theta),
\]

\[
\partial_\theta\mathbf x=(\cot\theta\,x,\cot\theta\,y,-r\sin\theta),
\qquad
\partial_{\phi_s}\mathbf x=(-y,x,0).
\tag{15}
\]

### 3.2 Canonical 协向量

one-form invariance \(p^{(s)}_\alpha dq_s^\alpha=p^{(B)}_\alpha dq_B^\alpha\) 给出

\[
\boxed{
p^B_t=p^s_t,\quad p^B_\theta=p^s_\theta,\quad p^B_\phi=p^s_\phi,
\quad
p^B_r=p^s_r+s\frac{2Mr-q_e^2}{\Delta}p^s_t
+s\frac a\Delta p^s_\phi
}.
\tag{16}
\]

Cartesian covector 到 spheroidal chart 的分量为

\[
p^s_r=\sin\theta(\cos\phi_s p_x+\sin\phi_s p_y)+\cos\theta p_z,
\tag{17}
\]

\[
p_\theta=\cot\theta(xp_x+yp_y)-r\sin\theta p_z,
\qquad p_\phi=xp_y-yp_x.
\tag{18}
\]

逆式使用

\[
p_i=p^s_r\partial_i r+p_\theta\partial_i\theta+p_\phi\partial_i\phi_s,
\]

\[
\nabla\phi_s=\left(-\frac y{\rho^2},\frac x{\rho^2},0\right)
+s\frac a{r^2+a^2}\nabla r.
\tag{19}
\]

\(\rho=0\) 是 azimuth seam；不能先形成奇异的 \(\phi\) 分量再期待 Carter axis limit
补救。第一版 Mino candidate 应在 exact/near-axis 回退到 Cartesian KS。

## 4. Pure Kerr Mino 零步状态

Mino fast-path 的首版产品域仍限定 pure Kerr。使用

- \(\sigma\)：仓库 affine parameter，\(dx^\mu/d\sigma=p^\mu\)；
- \(\gamma\)：\(d\sigma/d\gamma=\Sigma\)；
- \(\tau=E\gamma\)：energy-rescaled Mino parameter；
- \(b=L_z/E,\ \eta=\mathcal Q/E^2\)。

future-directed photon 有 \(E=-p_t>0\)，所以 Mino parameter 与 affine parameter
同向；negative tracing increment 不翻转 \(E\)。定义

\[
P=r^2+a^2-a b,\qquad A=(b-a)^2+\eta,
\]

\[
R=P^2-\Delta A,
\qquad
U=\eta+(a^2-\eta-b^2)\mu^2-a^2\mu^4.
\tag{20}
\]

两个 chart 都使用同一个 physical \(a\)，而不是 \(s a\)。零步必须满足

\[
v_r=\frac{dr}{d\tau}=\frac{\Sigma}{E}\dot r,
\quad v_r^2=R,
\qquad
v_\mu=\frac{d\mu}{d\tau}=\frac{\Sigma}{E}\dot\mu,
\quad v_\mu^2=U.
\tag{21}
\]

sign 来自实际 Hamilton tangent，不从 position、\(p_r^{KS}\) 或 backward traversal
猜测。BL canonical momenta 为

\[
p_r^B=\frac{E v_r}{\Delta},\qquad
p_\theta=-\frac{E v_\mu}{\sqrt{1-\mu^2}}.
\tag{22}
\]

turning point 的 sign 是显式 branch state，不能用 `signum(0)` 永久锁死。

## 5. 可复算验证

检查使用 lock 解析的 SymPy、180 位十进制精度与 \(10^{-80}\) boundary normalized-residual
gate。它不导入项目代码，而是独立构造 corrected metric、
legacy metric、标准 BL metric 与 Jacobians。

符号恒等式保持 exact `Rational`，按表达式类别显式使用 `Poly`、`cancel` 或
`trigsimp`，不调用启发式 `simplify()`；near-boundary 数值检查把 exact substitutions
直接传给 `evalf`，避免先替换再数值化造成精度损失。Near-axis、near-horizon 与
near-extremality 使用具名 exact `Rational`/radical case，不依赖随机序列或 binary64
中间值。这遵循 SymPy 的
[programmatic best practices](https://docs.sympy.org/latest/explanation/best-practices.html)
与 [`evalf(subs=...)` 建议](https://docs.sympy.org/latest/modules/evalf.html)。依赖版本与完整
执行 gate 见[统一 Python 研究工具链](python-research-tooling.md)。

运行命令：

```text
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research kerr-schild-map
```

2026-08-28 的完整摘要输出：

```text
python=3.14.7
sympy=1.14.0
symbolic.metric_pullback=PASS branches=ingoing,outgoing
symbolic.cartesian_oblate_map=PASS
symbolic.tangent_covector_duality=PASS
symbolic.affine_mino=PASS
symbolic.corrected_physical_spin=PASS branches=ingoing,outgoing
symbolic.legacy_outgoing=RED_AS_EXPECTED mismatch=4*M*a*r*u/(a**2*(1 - u) + r**2)
symbolic.legacy_outgoing_sample=RED_AS_EXPECTED g_tphi=g_phit=360/1591
boundary.precision_digits=180
boundary.near_axis=PASS u=1.5000000e-70 abs_delta=15.562500 M2_minus_a2=0.43750000 metric=6.5174535e-552 duality=0 mino=9.9448448e-557
boundary.near_horizon=PASS u=0.50000000 abs_delta=1.9843135e-60 M2_minus_a2=0.43750000 metric=2.6182860e-492 duality=0 mino=2.4969921e-498
boundary.near_extremality=PASS u=0.75000000 abs_delta=5.1961524e-80 M2_minus_a2=3.0000000e-60 metric=1.5455632e-471 duality=0 mino=2.3030687e-479
RESULT=PASS
```

这些 near-boundary substitutions 只验证 denominator 仍非零时的局部代数，不授权在
exact axis 或 \(\Delta=0\) 求值，也不给出 f32 conditioning 或 full-trajectory 保证。

### 5.1 Runtime mutation contracts

生产修复另由以下最小合同保护：

- domain integration test 证明 outgoing 的 oblate twist 与 ingoing 相反；
- domain unit test 用独立完整 BL Kerr–Newman metric，在 \(q_e=0.3\)、正负 spin、
  ingoing/outgoing 上逐分量验证 pullback；
- axis test 固定 \(\partial_x l_y=-a/(r^2+a^2)\)、
  \(\partial_y l_x=+a/(r^2+a^2)\)，能杀死把整个 axis gradient 乘 \(s\) 的 mutation；
- 现有 CPU/GPU center/corners/jitter initial-ray 与 regular termination matrix 继续验证
  WGSL/CPU seam；
- versioned ingoing fixture 保持 bitwise 不变。

Kerr–Newman charge 只进入 \(2Mr-q_e^2\) 与 \(\Delta\)；chart twist 修复不改变 radius
的 \(a^2\) 或 charge classification。上述 \(q_e=0.3\) full-metric contract 防止把 pure-Kerr
符号修复误写成不兼容 Kerr–Newman 的特例。

## 6. 迁移与可见变化

这是有意的 breaking physical correction：对于同一 outgoing `(r, theta, phi_s, a)`，
Cartesian \(x,z\) 在 \(\phi_s=0\) 保持，\(y\) 的 twist 反号；一般 azimuth 下位置按
(1) 变化。metric、observer tetrad 与 initial rays 随 corrected geometry 一致更新。

因此 outgoing GPU renderer 截图中的 frame-dragging handedness、lensed procedural-sky
结构与 shadow-edge 像素位置可能变化；不能保留旧 hard-coded edge 坐标。edge contract
应确定性搜索相邻 Horizon/Escape pair，再与 CPU reference 分类比较。ingoing chart、
radius、horizon radii 与 parameter-state fixtures 不应变化。

## 7. 剩余门槛

### Gate A — 连续模型：PASS

public `spin_m`、\(g_{t\phi}\)、frame dragging、BL \(a=J/M\) 与 oblate azimuth 在两
chart 中同义；Rust、WGSL 与 normative physics contract 使用同一公式。

### Gate B — 零步恒等式：convention seam PASS

legacy RED、corrected metric/position/covector/tangent GREEN、axis continuity、正负 spin、
Kerr–Newman charge 与 CPU/GPU initial-ray seam 已覆盖。这个 Gate 只证明初值变换；完整
trajectory 由独立 observable matrix 约束。

### Gate C — 受限轨迹 oracle：扩大分辨率后 FAIL

80×45 matrix 曾在 pure Kerr、finite \(E>0\)、off-axis、受限 subextremal exterior 内通过，
但扩展到 `320×180` 后，pixel `(175, 51)` 的 accepted result 相对独立 Cartesian KS
reference 产生约 `2.661354e-3 M` travel-time error，超过 `1e-3 M` contract。reciprocal
constraint 与 winding gate 没有给出 terminal phase 的充分误差界，因此 Gate C 为 FAIL。

### Gate D — `f32`/WGSL 与性能：虽有加速仍 REJECTED

Apple M5/Metal、1280×720 的历史 256-pair ABBA 中，restricted Mino 相对 interval
capture + KS 改善 `35.768%`，95% CI `[-36.390%, -35.189%]`。性能结果仍说明 separable
polynomial dynamics 有价值，但 correctness gate 优先；高分辨率反例使 production
candidate 失效，相关 WGSL、pipeline 和 benchmark variant 已删除。

## 8. 一手来源与适用域

- Campanelli et al. Eq. (22)–(24)、(34)–(36) 给出 ingoing/outgoing Kerr coordinate
  transformations 的相反 azimuth shift。
  [Campanelli et al. 2001](https://arxiv.org/pdf/gr-qc/0010034)
- Adamo–Newman Sec. 3.1 给出 Kerr–Newman BL/null/oblate/Cartesian KS 形式与物理
  parameter interpretation。
  [Adamo–Newman 2016](https://arxiv.org/pdf/1410.6626)
- Bozzola–Chan–Paschalidis Eq. (4)–(7)、Sec. III–IV 支持两条 Cartesian KS
  principal directions 与 backward ray 的 outgoing-chart 选择；它不单独固定本项目
  cross-chart azimuth convention。
  [Bozzola et al. 2023](https://arxiv.org/pdf/2310.02321)
- Mino Eq. (2.1)–(2.6) 与 Fujita–Hikida Eq. (2) 定义
  \(d\sigma/d\gamma=\Sigma\)；其主要产品域不是本项目的 GPU null tracer。
  [Mino 2003](https://arxiv.org/pdf/gr-qc/0302075)
  [Fujita–Hikida 2009](https://arxiv.org/pdf/0906.1420)
- Gralla–Lupsasca v3 Eq. (1)–(16) 是 pure-null Kerr exterior 的
  energy-rescaled Mino/affine seam 主要来源；其范围不授权 axis、extremal 或 horizon
  crossing。
  [Gralla–Lupsasca 2020 v3](https://arxiv.org/pdf/1910.12881)

最终判定：**physical-spin blocker 已根因修复；零步 seam 只授权后续候选的坐标初值，
不授权 numerical Mino production。当前 fixed-step candidate 已否决；KS 仍是支持域外与
不确定轨迹的必要基线。**

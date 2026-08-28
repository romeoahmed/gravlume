# Cartesian Kerr–Schild RK4 代数约化与事件定位

本文保存 outgoing Cartesian KS/RK4 的代数约化、实验顺序和准入证据，不定义当前 solver 或 tolerance；连续方程、验收预算与当前实现分别以[数学物理](../physics.md)、[验证合同](../validation.md)和 [GPU 证据](../gpu-renderer.md)为准。

**状态：主要代数约化与事件定位已采用；step policy 仍实验。** 研究基线为 `a5048a0490b0fd05d19fd40857886824179ecb07`。production 随后采用 contracted null-gradient、compact geometry、discriminant-root `Sigma`、共享 reciprocal、六维 dynamic phase、全域 Cartesian Carter invariant、单调 Hermite event localization 与 Kerr–Newman interval capture；新的 step controller 仍未获授权。

## 结论

1. **继续以 outgoing Cartesian KS + canonical Hamiltonian + affine RK4 为通用基线。**
   backward ray tracing 选 outgoing chart 有因果/数值依据；当前路径还要覆盖
   Kerr–Newman、近视界、axis、surface/未来 path sampling，不能因 pure-Kerr 解析解存在就删除
   KS fallback。[Bozzola–Chan–Paschalidis 2023](https://arxiv.org/pdf/2310.02321)
2. 第一优先级不是更换积分器，而是把每个 RK stage 的 KS RHS 化到真正需要的标量/向量：
   使用 factored \(\nabla f\)、直接计算 \(\nabla(\ell\cdot p)\)，不构造完整
   \(3\times3\) null-vector Jacobian。discriminant root 在实数代数上就是 \(\Sigma\)，并已直接
   成为 production `Sigma`；CPU reference 保留独立 residual reconstruction，避免 renderer 与
   oracle 共享同一 binary32 计算图。所有 production 恒等式已用 SymPy exact algebra 验证。
3. phase state 可从 \((t,x,y,z,p_t,p_x,p_y,p_z)\) 约化为动态
   \((x,y,z,p_x,p_y,p_z)\)，但每条 ray 必须保存自己的常量
   \(E=-p_t\)，并用同一 RK stages 累计 coordinate-time increment。observer-frame
   `omega_obs = 1` **不等于**所有像素的 coordinate energy \(E=1\)。
4. event guard 现使用 endpoint value/derivative 的 cubic Hermite polynomial；只有 derivative
   Bézier control values 证明全区间单调且导数不过小才 refinement，否则保留原 chord。固定六次
   safeguarded Newton 满足有限步 residual 合同；两次迭代虽恢复渐近阶数，却在真实角点留下
   `1.68e-2` residual，不能用渐近论证覆盖有限步合同。
5. 真正的 Sundman 变换会把 RHS 改成 \(dy/ds=\chi(y)F(y)\)，每个 RK stage 都要计算
   \(\chi\)。它没有“必然更快或更准”的定理；当前 `h = c r` 已经是合法 variable-step
   RK4。Sundman 不应先于精确代数约化、event Hermite 与已有 RHS-count telemetry。
6. interval Bernstein capture 已进入 production。neutral Kerr–Newman radial potential 仍是
   同一个 quartic family，只多出 \(-q_e^2[(b-a)^2+\eta]\) 常数项；production 只在严格亚极端、
   远离 axis/near-extreme 且 outward-rounded Bernstein coefficients 全部严格为正时接受，否则
   执行完整 KS。
7. WGSL 允许浮点重关联，`fma` 也不保证真正 fused。代数恒等不意味着 binary32 bit identity；
   正确性以 observable contract 为准。数学约化全部合入后只运行一次 production aggregate
   benchmark 作为结果确认；benchmark 不作为代数恒等式、结构缩减或保守证书的准入门槛。

## 1. 与现有研究的边界

本文只补现有 ledger 中尚未形式化的 **KS RK4 单步代数与 event root**，不重做以下工作：

| 已有结论                                                                                                                  | 本文处理方式                                            |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| outgoing KS、8×8、endpoint reuse、committed-only singularity、global escape-direction map、interval radial capture 已采用 | 作为 baseline，不重新证明性能数字                       |
| numerical reciprocal-Mino fixed-step 因 terminal travel-time 反例被拒绝                                                   | 不恢复，不再扫描 fixed Mino factor                      |
| pure-Kerr elliptic/Carlson terminal solver 是独立高收益路线                                                               | 不实现 special functions；只定义它与 KS fallback 的边界 |
| wavefront/subgroup 需要 active-ratio 证据                                                                                 | 不引入 backend-specific queue 或 subgroup 假设          |
| atomic full-frame publication 与 HDR presentation                                                                         | 不变；所有 candidate 仍一次发布完整 generation          |

Cartesian KS 对 GPU 友好的主要原因不是“公式短”，而是 horizon/axis regularity、rank-one
metric 与可合并的 RHS 数据流。GRay2 给出 Cartesian KS metric、隐式 oblate radius，并证明
经过代数重排后 GPU 上可以胜过表面更简单的 Boyer–Lindquist 形式；其结论也强调 coordinate
time 与 affine parameter 的 RHS 复杂度相当，应按产品 observable 选择，而非只按变量数选择。
[GRay2, Sec. II](https://arxiv.org/abs/1706.07062)

当前默认场景历史统计约为平均/最坏 `61/132` accepted RK4 steps；普通 step 有四次
geometry/RHS evaluation。因此，在不改变 ray 数的 KS 路径里，最直接的成本对象是每张图约

\[
N_{\rm pixel}\,N_{\rm step}\,4\,C_{\rm RHS},
\]

而不是再次微调 dispatch batch。下面的约化正针对 \(C_{\rm RHS}\)。

## 2. Hamiltonian 约化：不能把 observer frequency 当作 coordinate energy

使用 signature \((-+++ )\) 与 chart branch \(s\in\{-1,+1\}\)。物理量先写成未缩放形式：

\[
g_{\mu\nu}=\eta_{\mu\nu}+f\,\ell_\mu\ell_\nu,
\qquad
g^{\mu\nu}=\eta^{\mu\nu}-f\,\ell^\mu\ell^\nu,
\]

\[
f=\frac{2Mr-q_e^2}{\Sigma},
\qquad
\ell^\mu=(-1,\boldsymbol\ell),
\]

\[
\boldsymbol\ell=
\left(
\frac{s r x+a y}{r^2+a^2},
\frac{s r y-a x}{r^2+a^2},
\frac{s z}{r}
\right).
\tag{1}
\]

令 \(\mathbf p=(p_x,p_y,p_z)\)、\(E=-p_t\) 和

\[
q=p_\mu\ell^\mu=E+\boldsymbol\ell\cdot\mathbf p.
\tag{2}
\]

stationarity 给出 \(p_t'=0\)，但 \(E\) 由每个 pixel 的 tetrad ray 决定。项目把
observer-frame frequency 归一为

\[
\omega_{\rm obs}=-u_{\rm obs}^\mu p_\mu=1,
\]

这只是一个 tetrad contraction，除非 observer 恰好与 coordinate Killing frame 对齐，否则不推出
\(E=1\)。因此附件中“固定 \(E=1\) 后只积分六变量”的建议不能逐字进入当前 seam；正确约化是
**六个动态变量 + 一个 per-ray 常量 \(E\)**。

canonical Hamiltonian 为

\[
H=\frac12\left(-E^2+\mathbf p^2-fq^2\right).
\tag{3}
\]

直接求导得到完整的 reduced affine system：

\[
\boxed{
\begin{aligned}
\mathbf x' &= \mathbf p-fq\boldsymbol\ell,\\
\mathbf p' &= fq\,\nabla(\boldsymbol\ell\cdot\mathbf p)
+\frac12q^2\nabla f,\\
t' &= E+fq,\\
E' &=0.
\end{aligned}}
\tag{4}
\]

这不是 coordinate-time parameterization。RK4 仍对 affine parameter 积分；每步用同一四个
stage 的 \(t'=E+fq\) 形成

\[
\Delta t=\frac h6\left(T_1+2T_2+2T_3+T_4\right).
\tag{5}
\]

对 terminal fraction \(\theta\)，令 \(T_a,T_b\) 是 step 两端的 time RHS，标准 Hermite
basis 为 \(H_{10},H_{01},H_{11}\)，则相对时间仍可精确按当前 dense policy 构造：

\[
\delta t_H(\theta)
=H_{10}(\theta)hT_a+H_{01}(\theta)\Delta t+H_{11}(\theta)hT_b.
\]

state 不需要保存 absolute \(t\)，但 travel time 逐步累计 \(|\Delta t|\)，terminal step 累计
\(|\delta t_H|\)，未来 retarded-time observable 也继续消费同一 increment。这样同时保留共同
coordinate-time translation invariance 与现有 travel-time contract。若只删除 `position.x`
而用 endpoint \(T_1h\) 代替 (5)，会把时间积分降阶，不获准。

## 3. 每个 RK stage 的精确代数约化

### 3.1 radius discriminant 就是 \(\Sigma\)

定义

\[
B=x^2+y^2+z^2-a^2,
\qquad u=r^2,
\qquad
u^2-Bu-a^2z^2=0.
\tag{6}
\]

当前稳定 physical branch 为

\[
u=\frac{B+\sqrt{B^2+4a^2z^2}}2,
\]

并在 \(B<0\) 时用 conjugate form 避免 cancellation。令

\[
D=\sqrt{B^2+4a^2z^2}.
\]

由 (6) 有

\[
\boxed{\Sigma=r^2+\frac{a^2z^2}{r^2}=2u-B=D.}
\tag{7}
\]

因此 radius solver 已经计算出的 root 可直接作为 \(\Sigma\)，无需再形成
`a²*z²/r²`。相应梯度为

\[
\nabla r=\frac1\Sigma
\left(xr,yr,\frac{z(r^2+a^2)}r\right),
\tag{8}
\]

\[
\boxed{
\nabla\Sigma=\frac2\Sigma
\left(Bx,By,z(B+2a^2)\right).
}
\tag{9}
\]

写 \(N=2Mr-q_e^2\)、\(f=N/\Sigma\)，quotient gradient 可约为

\[
\boxed{
\nabla f=\frac{2M\nabla r-f\nabla\Sigma}{\Sigma}.
}
\tag{10}
\]

(7)–(10) 对 Kerr 与 Kerr–Newman 同时成立。production 直接复用 discriminant root、gradient 和
reciprocal，缩短 live expression；这项准入来自定义域更大、依赖图更短和完整 observable gate，
不再用局部 A/B 计时决定 exact algebra。

### 3.2 只计算 contracted null Jacobian

Hamilton force 只消费

\[
\mathbf w=\nabla(\boldsymbol\ell\cdot\mathbf p)
=(\partial_i\ell_j)p_j,
\]

不消费完整 \(\partial_i\ell_j\)。令 \(K=r^2+a^2\)，定义

\[
\mathbf e=
\left(
\frac{s r p_x-a p_y}{K},
\frac{a p_x+s r p_y}{K},
\frac{s p_z}{r}
\right),
\tag{11}
\]

\[
C=
\frac{(s x-2r\ell_x)p_x+(s y-2r\ell_y)p_y}{K}
-\frac{s z p_z}{r^2}.
\tag{12}
\]

chain rule 精确给出

\[
\boxed{\mathbf w=\mathbf e+C\nabla r.}
\tag{13}
\]

这允许 geometry/RHS 边界从“三个 `vec4` null derivatives”变为已有
\(r,\nabla r,\boldsymbol\ell\) 与 momentum 的一次 contraction。SymPy 在 ideal scalar CSE
模型中只把相关 null algebra 从 63 降到 59 operations；因此主要假设是减少 live values、临时
向量与依赖链，而非声称巨大的 FLOP 节省。只有 shader compiler output、GPU timestamp 与
observable gate 能决定是否进入 production。

### 3.3 scaled geometry 的维度

production 为远场稳定性用

\[
S=\max(|x|,|y|,|z|,|a|),
\quad \hat{\mathbf x}=\mathbf x/S,
\quad \hat a=a/S
\]

计算 geometry。上式必须在 hatted variables 中执行；导数回 physical coordinate 时：

\[
\nabla_{\mathbf x} f
=\frac{2\hat M\nabla_{\hat{\mathbf x}}\hat r
-f\nabla_{\hat{\mathbf x}}\hat\Sigma}
{\hat\Sigma S},
\qquad
\nabla_{\mathbf x}(\boldsymbol\ell\cdot\mathbf p)
=\frac{\hat{\mathbf e}+\hat C\nabla\hat r}{S}.
\tag{14}
\]

少掉或重复 `1/S` 都会让近远场表现相反。axis 上 \(x=y=0\) 时 radius 的现有 analytic branch
仍应保留；(13) 的 \(z\) 分量在 exact algebra 中相消为有限值，但 binary32 不能依赖两个大项
“恰好相消”来替代 axis branch。

### 3.4 Carter invariant 的全域 Cartesian 形式

令

\[
\varpi^2=x^2+y^2,
\quad K=r^2+a^2,
\quad S=xp_x+yp_y,
\quad P_\perp^2=p_x^2+p_y^2,
\]

以及 \(L_z=xp_y-yp_x\)。二维 Lagrange identity 给出

\[
\boxed{S^2+L_z^2=\varpi^2P_\perp^2}.
\]

将它代入 Boyer–Lindquist 形式的 Carter constant 后，所有 \(1/\varpi\)、
\(1/\sin\theta\) 和 square root 都精确消去：

\[
\boxed{
\mathcal Q=
\frac{z^2}{r^2}\left(KP_\perp^2-a^2E^2\right)
-2zS p_z
+\frac{r^2\varpi^2}{K}p_z^2.
}
\tag{15}
\]

式 (15) 不借助 null constraint，因此 Carter drift 与 Hamiltonian residual 仍是独立诊断。
它在 axis 上可直接代入，而非只靠极限定义：\(x=y=0,z^2=r^2\) 时

\[
\mathcal Q=(r^2+a^2)(p_x^2+p_y^2)-a^2E^2.
\]

当 \(a=0\) 时，二维/三维 Lagrange identity 进一步给出

\[
\mathcal Q=L_x^2+L_y^2=L^2-L_z^2.
\]

production 从 scaled geometry 已缓存的 \(1/r\) 与 \(1/(r^2+a^2)\) 构造
\((\cos^2\theta,\sin^2\theta)\) `vec2`，不新增除法，也不先形成可能溢出的 physical
\(z^2\) 或 \(\varpi^2\)。因此 axis 不再是 Carter evaluator 的控制流概念；CPU reference 则
保留独立 trigonometric/axis 形式作为交叉 oracle。

## 4. Event localization：对 guard 做 dense interpolation

当前算法先用 endpoint guard value 的 chord 得到

\[
\theta_{\rm chord}=\frac{G_* -G_0}{G_1-G_0},
\]

再把 \(\theta\) 交给 cubic dense state。对 nonlinear \(G(y)\)，chord 的 interior function
defect 是 \(O(h^2)\)；高阶 state interpolant 无法追回错误的 event fraction。

对 \(G(\lambda)=g(y(\lambda))-G_*\) 使用 endpoint value 和 derivative：

\[
\begin{aligned}
G_H(\theta)={}&(2\theta^3-3\theta^2+1)G_0
+(\theta^3-2\theta^2+\theta)h\dot G_0\\
&+(-2\theta^3+3\theta^2)G_1
+(\theta^3-\theta^2)h\dot G_1.
\end{aligned}
\tag{16}
\]

对任何 cubic \(G\)，(16) exact；quartic term \(c_4\lambda^4\) 的 defect 精确为

\[
c_4h^4\theta^2(\theta-1)^2.
\tag{17}
\]

因此若 event 是 transversal，\(|\dot G|\ge m>0\)，root perturbation 由 inverse-function
bound 控制为 \(O(h^4/m)\)。这与 Runge–Kutta continuous extension/event-location 的标准
做法一致。[Shampine 1985](https://doi.org/10.1137/0722060)
[Shampine 1988](https://doi.org/10.1016/0893-9659%2888%2990062-6)

现有 event surfaces 的 derivative 不需要新 geometry：

\[
\dot G_r=\nabla r\cdot\mathbf x',
\tag{18}
\]

\[
G_D=r^4+a^2z^2-D_*,
\qquad
\dot G_D=4r^3\dot r+2a^2z\dot z.
\tag{19}
\]

式 (19) 只用于未饱和的 near-event guard；若 endpoint 的 overflow-safe
`singularity_measure` 已经正向饱和，不能对饱和值伪造 derivative，应该保留原 side
classification 或回退。

production 以 chord fraction 初始化，先把 Hermite derivative 写成 quadratic Bézier。三个
derivative control values

\[
m_0,\qquad 3(G_1-G_0)-m_0-m_1,\qquad m_1
\]

必须全部与 endpoint 总变化同向，才能证明 \([0,1]\) 上单调；判据严格而不加 epsilon，失败即
回到 chord。通过后还要求当前 derivative 大于相对于 control-polygon scale 的
\(\sqrt{\epsilon_{f32}}\) floor，并执行六次 bracketed Newton/midpoint。event priority、endpoint
bracket 和 singularity/horizon/escape 选择顺序完全不变。

一次 Newton 足以恢复渐近四阶，并不代表生产有限步两次就够。真实 GPU/reference matrix 的角点
在两次版本上得到 `1.6827391e-2` event residual；六次版本恢复到 `5e-3` 合同内。迭代只在终止
步对标量 cubic 求值，不增加 geometry/RHS。更大 step policy 仍是独立实验，不能借 E1 的定位
改善放宽积分误差。

这里必须区分 root solver 与 guard model 两层误差。若 \(\tau_H\) 是 Hermite cubic 的精确根、
\(\tau_*\) 是真实 guard 的根，则

\[
|\widehat\tau-\tau__|
\le |\widehat\tau-\tau_H|+|\tau_H-\tau__|.
\]

固定迭代只缩小第一项；第二项由 finite step 上的 Hermite model error 决定。因此六次是由当前
有限步 observable contract 选出的工程参数，不是从渐近阶数推导出来的常数。若以后再次出现
event residual plateau，diagnostic capture 应同时导出 \(|G_H(\widehat\tau)|\) 与 localized state 上
重新计算的真实 guard residual：前者已到 binary32 floor 而后者不再下降，才说明继续增加 Newton
次数没有意义。正常 production record 只保留后者，避免为调试量扩张逐像素 ABI。

## 5. Step policy 的数学边界

对固定平滑 trajectory、固定 branch 与 uniform step \(h\)，classical RK4 有

\[
y_{n+1}-y(\lambda_n+h)=h^5C_5(y_n)+O(h^6),
\]

故在总 affine length \(L\) 上

\[
W(h)\sim 4L/h,
\qquad
e_{\rm global}(h)=O(h^4).
\tag{20}
\]

在这个 asymptotic envelope 中，work 对 \(h\) 单调下降、error 对 \(h\) 单调上升；理论最优
不是某个神秘常数，而是**满足 observable budget 的最大步长**。若把所有步放大 \(\alpha\)，
理想 work 乘 \(\alpha^{-1}\)，主阶误差乘 \(\alpha^4\)。这能先排除明显不合理的 sweep，却不能
证明 terminal pixel error 逐点单调：event truncation、branch、roundoff 和 final partial step
会改变误差常数。

当前 `h = clamp(c_r r, h_min, h_max)` 隐含假设 \(|\mathbf x'|\approx1\)。一个仍保持 affine
RK4、且只复用 first-stage RHS 的候选是

\[
h_x=c_x\frac{r}{\max(\|\mathbf x'\|,\epsilon_x)},
\qquad
h_p=c_p\frac{\|\mathbf p\|}{\max(\|\mathbf p'\|,\epsilon_p)},
\]

\[
h=\operatorname{clamp}(\min(h_x,h_p),h_{\min},h_{\max}).
\tag{21}
\]

这套未 clamp 的 controller 还具有 current \(h\propto r\) 没有的 affine-normalization
协变性：若同一 null path 只重标定
\((E,\mathbf p)\mapsto\kappa(E,\mathbf p)\)，则
\(\mathbf x'\mapsto\kappa\mathbf x'\)、
\(\mathbf p'\mapsto\kappa^2\mathbf p'\)，从而
\((h_x,h_p)\mapsto(h_x,h_p)/\kappa\)。因此
\(h\mathbf x'/r\) 与 \(h\mathbf p'/\|\mathbf p\|\) 不变。项目当前在 seam 固定
observer frequency，所以这不是修复现有输入错误；它是 (21) 比纯 radius step 更自然的数学
性质。\(h_{\min},h_{\max}\) 若要保留该性质，也必须按 per-ray normalization 缩放，或继续明确
只支持当前归一化 profile。

在 derivative 近似不变的一步内，(21) 分别限制 relative spatial displacement 与 momentum
bend；它可能在 photon region 自动比单独 \(r\) 更保守、在远场更积极。它**不是 LTE
estimator**，所以只能作为 S1 heuristic。正确实验顺序是：

1. high-precision oracle 上按 (20) 先求每条 representative ray 的最大可接受 envelope；
2. 用这些结果拟合极少数 \((c_x,c_p)\) candidate，不在 GPU 上盲扫几十个 factor；
3. GPU 只比较 baseline、一个 conservative candidate 和一个 envelope-edge candidate；
4. 任一 branch/observable 失败即缩小 supported envelope 或删除 S1，不能由 invariant drift
   代替 terminal phase error。

### 为什么现在不优先 Sundman

选择 \(d\lambda/ds=\chi(y)\) 后，真正的 transformed system 是

\[
\frac{dy}{ds}=\chi(y)F(y),
\qquad
\frac{dt}{ds}=\chi(y)T(y).
\tag{22}
\]

RK stages 必须在各自 stage state 重算 \(\chi\)。把 \(\chi(y_n)\) 只算一次其实仍是
variable affine step，不是对 (22) 的 RK4。harmonic/min-combined controller 会增加 reciprocal、
min/max 与 SIMT divergence，而且改变五阶 error tensor；没有一般单调改进定理。

所以 Sundman 只在以下证据同时出现后升级：RHS 已约化；step histogram 显示同一路径在
far/near region 的 stiffness ratio 明显；(21) 仍需大量 conservative steps；一个具名
\(\chi\) 在高精度 oracle 上减少总 RHS 且收紧 terminal observable envelope。否则它是额外
复杂度，不是根因突破。

## 6. Kerr–Newman 的严格 capture 扩展

对 neutral photon，定义 \(b=L_z/E\)、\(\eta=\mathcal Q/E^2\) 与

\[
\Delta=r^2-2Mr+a^2+q_e^2.
\]

Carter-separated radial potential 为

\[
\frac{R(r)}{E^2}
=\left(r^2+a^2-ab\right)^2
-\Delta\left[(b-a)^2+\eta\right].
\tag{23}
\]

精确展开：

\[
\boxed{
\frac{R(r)}{E^2}
=r^4+(a^2-b^2-\eta)r^2
+2M\left[(b-a)^2+\eta\right]r
-a^2\eta
-q_e^2\left[(b-a)^2+\eta\right].
}
\tag{24}
\]

Wang–Lee–Lin 的 Eq. (33)–(36) 给出相同 quartic coefficients。
[Kerr–Newman geodesics](https://arxiv.org/pdf/2208.11906)

因此 current Kerr interval Bernstein certificate 的结构不需 quartic root solve；只需让 constant
coefficient、outer horizon

\[
r_+=M+\sqrt{M^2-a^2-q_e^2}
\]

与 packed-parameter interval 同源。准入仍必须满足：\(E\) 正、有限且 normalization
\((b,\eta)\) 可表示；subextremal horizon 存在；current
backward radial orientation 是向 horizon 的那一支；对覆盖 \([r_+,r_{obs}]\) 的所有 outward-
rounded Bernstein segments 均严格证明 \(R>0\)；任何 coefficient、horizon、energy 或 sign
不确定即 KS fallback。exact extremal、superextremal、charged photon 不在首版支持域。

这项是可证 conservative fast path，但只扩大 Kerr–Newman coverage；默认 pure-Kerr 的
polynomial 完全不变，因此不能把它记为默认性能优化。

## 7. WGSL、Naga 与 binary32 语义

WGSL 对本研究有三个直接约束：

- runtime overflow/NaN 在 finite-math assumption 下可成为 indeterminate value，代数重排不能
  先形成 overflow 再期待后续 clamp 修复；
- implementation 可以重关联和融合浮点运算；
- `fma(a,b,c)` 的标准语义允许 ordinary multiply 后 ordinary add，不保证 IEEE 单次 rounding。

规范依据分别见 [floating-point evaluation](https://www.w3.org/TR/WGSL/#floating-point-evaluation)、
[reassociation and fusion](https://www.w3.org/TR/WGSL/#floating-point-evaluation) 与
[`fma`](https://www.w3.org/TR/WGSL/#fma-builtin)。

研究基线锁定的 Naga `30.0.0` 源码审计显示 WGSL `fma` 当时分别下发为 MSL `fma`、SPIR-V
`GLSL.std.450 Fma` 和 HLSL `mad`：

- [MSL writer source](https://docs.rs/crate/naga/30.0.0/source/src/back/msl/writer.rs)
- [SPIR-V writer source](https://docs.rs/crate/naga/30.0.0/source/src/back/spv/block.rs)
- [HLSL writer source](https://docs.rs/crate/naga/30.0.0/source/src/back/hlsl/writer.rs)

这只说明 translator output，不保证 driver 最终指令或精度。因而 pairwise RK4 combine

\[
(k_1+k_4)+2(k_2+k_3)
\]

与显式 `fma` 只能作为最后的 M1 micro-experiment：它们不能成为正确性依赖，也不值得为不同
native backend 维护不同 shader。项目是 native-only 不改变这个事实；wgpu/Naga 仍是统一 API
和 shader contract。

## 8. 形式化验证

使用仓库锁定的 [Python 研究工具链](python-research-tooling.md)建立 exact symbols，并用
radius constraint 消去一个平方变量。持久化的
[`kerr_schild_rhs.py`](scripts/src/gravlume_research/checks/kerr_schild_rhs.py)
证明 \(\Sigma\)/gradient identities、ingoing/outgoing 两 branch 的 contracted null derivative、
六维 Hamilton system、全域 Carter/Schwarzschild limit、Kerr–Newman radial quartic 和 Hermite
order/monotonicity control polygon。结论不依赖临时绝对路径。

执行模型：

```text
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research kerr-schild-rhs
```

持久化脚本预期输出：

```text
Sigma discriminant identity: PASS
Sigma and Kerr-Schild scalar gradients: PASS
Both Kerr-Schild branch contractions: PASS
Schwarzschild projector limit: PASS
Six-dimensional Hamilton system: PASS
Axis-regular Carter invariant: PASS
Kerr-Newman radial quartic: PASS
Cubic Hermite event order: PASS
RESULT=PASS
```

持久化 verifier 的 exact checks：

1. \(\Sigma^2=B^2+4a^2z^2\) on (6)；
2. (8)、(9) 与 implicit-radius differentiation 等价；
3. (10) 与 direct quotient differentiation 等价；
4. (13) 与逐项 \(\partial_i\ell_jp_j\) 等价；
5. (4) 与 \(\mathbf p'=-\partial H/\partial\mathbf x\)、
   \(t'=-\partial H/\partial E\) 等价；
6. (15) 与 trigonometric Carter expression 等价，并直接给出 axis/Schwarzschild limit；
7. (24) 与 (23) 等价；
8. cubic Hermite reproduces degree ≤3，quartic defect 为 (17)。

这些结果证明 real-arithmetic identity 与 Hermite interpolation order，不证明 binary32
conditioning、compiler register count、GPU speed 或 full-trajectory correctness。

## 9. 实验矩阵与顺序

表中已采用项都已收敛进唯一 production shader；未采用项不留下永久 shader selector、环境变量
协议或 benchmark-only API。

| ID  | 单一变化                              | 数学状态                   | 主要测量                                           | 进入下一层的条件                              |
| --- | ------------------------------------- | -------------------------- | -------------------------------------------------- | --------------------------------------------- |
| R0  | research baseline                     | baseline                   | historical GPU duration、RHS/step distribution     | 已冻结 revision/scene fingerprint             |
| R1  | (7)、(9)、(10)                        | exact；已采用              | SymPy + all-pixel observable                       | direct `Sigma`/gradients 进入 production      |
| R2  | (13)，移除 full null Jacobian payload | exact；已采用              | SymPy + generated shader + GPU contract            | no dynamic Jacobian loop/array                |
| R3  | 6D dynamic phase + per-ray `E` + (5)  | exact ODE；已采用          | time translation/travel/direction gate             | `RhsResult` 以 `vec4<t,x,y,z>` 打包           |
| R4  | 全域式 (15)                           | exact；已采用              | Carter/axis/Schwarzschild proof + drift gate       | 无 axis seam，不借 null constraint            |
| RΣ  | R1–R4 cumulative                      | exact continuum；已采用    | final aggregate GPU + invalidation→publish         | 只在全部完成后确认一次结果                    |
| E1  | (16)–(19)，原 step 不变               | fourth-order guard；已采用 | finite-step event residual、direction、travel time | 单调/derivative 不确定时 chord fallback       |
| S1  | E1 + (21)                             | heuristic                  | total RHS、step quantiles、observable max          | oracle-supported envelope；端到端稳定改善     |
| C1  | (24) 的 KN interval capture           | strict certificate；已采用 | KN full-KS branch/direction equivalence            | unsupported/near-extreme 保守 fallback        |
| M1  | pairwise RK combine/显式 `fma`        | WGSL permits variation     | only paired GPU A/B                                | 不成为 correctness dependency；跨复测稳定才留 |

R1–R4 的准入依据是 exact identity、定义域、状态/依赖图缩减和 observable contract，而不是单项
计时阈值。只有全部组合稳定后才测 RΣ cumulative Pareto point；最终数据用于量化成果和发现明显
整体退化，不替代数学与接口判断。

### 9.1 正确性场景

| 维度        | 最小覆盖                                                                                                               |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| spacetime   | Minkowski/Schwarzschild；Kerr `a/M = ±0.8, ±0.99`；KN 多个满足 `a²+q_e²<1` 的组合；near-extreme；superextreme fallback |
| observer    | `r/M = 5, 30, 100`；near-axis、equatorial、generic inclination；共同 coordinate-time translation                       |
| camera      | FOV `15°,45°,90°`；center/corners；critical curve 两侧；odd extents                                                    |
| events      | horizon、escape、singularity guard、near-tangent bracket、同一步多 surface priority                                    |
| resolutions | 小型全像素 oracle lattice；720p/1080p/1440p performance；resize invalidation→publish                                   |

每个 accelerated/modified pixel 都沿用 `docs/validation.md`：termination 一致；escape direction
≤ `3.82e-4 rad`；travel time ≤ `1e-3 M`；null/`E`/`Lz`/Carter drift 各 ≤ `0.05`；event
position/residual ≤ `5e-3 M`；regular domain 不新增未回退 failure/uncertain。临界或 conditioning
不确定可以保守回退，不能确定地给错 branch。

### 9.2 性能方法

1. 数学恒等式、binary32 domain 和 observable contract 先决定是否接受；不为每个局部改写反复
   ABBA。全部 production 约化完成后，复用 `GpuTimings` 与 Criterion `iter_custom` 只运行一次
   aggregate confirmation。[wgpu `ComputePass::write_timestamp`](https://docs.rs/wgpu/30.0.0/wgpu/struct.ComputePass.html#method.write_timestamp)
2. 最终记录 adapter/backend、power mode、revision/dirty state、extent、scene 与 output gate；结果与
   历史基线比较时明确 DVFS/热状态限制，不把一次计时反过来否定 exact dependency reduction。
3. 同时记录 invalidation→atomic-publish wall latency；Criterion GPU duration 不包含 submit、wait、
   map 与 publication，不能替代 resize 体感。
4. aggregate 至少记录 RHS/geometry evaluations、step p50/p90/p95/p99/max、termination 与 fallback
   distribution。只有具名 vendor profiler/offline compiler 才能报告 register/occupancy；源码字段数
   只能是假设。
5. 当前 macOS/Metal 可跑完整数值/性能 gate；Windows/Linux 在真实 target 前只允许说
   WGSL/Naga/compile 通过，不得把未运行 Vulkan 当作跨平台性能结论。实现仍保持一份 portable WGSL。

最终累计确认在 2026-08-14 只运行一次：Apple M5/Metal、production `1280×720`、30 个
Criterion samples/990 次 trace 的 GPU timestamp estimate 为 `14.861 ms`，95% interval
`[14.824, 14.893] ms`，吞吐约 `62.015 Melem/s`。相对同名保存基线的 time change 为
`-20.661%`，interval `[-21.693%, -19.560%]`；30 个样本中有 5 个 outlier。这个结果确认累计
production 点没有整体退化，不把改善比例分摊给某一条恒等式，也不代表包含 resize、surface、
UI 与 publication 的整帧 wall latency。

## 10. 接口与 fallback 边界

这些约化应该停留在 shader-private deep module，不扩大 Rust public API：

- `TerminalMap` 仍只承诺 branch/direction/time 等 terminal observable；
- `PathSampler` 未来需要 ordered checkpoints，不能由 6D terminal fast path 伪造；
- `DiagnosticCapture` 保留现有记录语义；若内部不再动态积分 \(p_t\)，energy drift 应明确写为构造上
  exact zero，而不是删除字段或改变 record ABI；
- interval capture、未来 elliptic terminal solver 与任意 heuristic 都只返回
  `accepted terminal` 或 machine-readable `fallback`，fallback 从 validated camera initial state
  重跑完整 outgoing KS，不保存半条未证明 continuation state。

R1–R4、E1 与 C1 已完成。下一步只有在独立误差模型明确时才做 S1；M1 与 Sundman 仍低优先。
完整 elliptic solver 仍可能给出更大的 pure-Kerr terminal 加速，但它与这条通用 KS 基线互补，
不是删除 fallback 的理由。最终 benchmark 只确认累计 production 结果，不再驱动逐行代数选择。

## 主要一手来源

- Chan, Medeiros, Özel, Psaltis, _GRay2: A General Purpose Geodesic Integrator for Kerr
  Spacetimes_：Cartesian KS、代数重排、GPU 与 affine/coordinate-time 成本讨论。
  [arXiv:1706.07062](https://arxiv.org/abs/1706.07062)
- Bozzola, Chan, Paschalidis, _Not all spacetime coordinates for general-relativistic ray tracing
  are created equal_：backward ray 与 outgoing KS chart 的适定性。
  [arXiv:2310.02321](https://arxiv.org/abs/2310.02321)
- Wang, Lee, Lin, _Null and time-like geodesics in Kerr–Newman black hole exterior_：neutral
  photon radial/angular potential 与 quartic coefficients。
  [arXiv:2208.11906](https://arxiv.org/abs/2208.11906)
- Shampine, _Interpolation for Runge–Kutta Methods_；Shampine, _Locating special events when
  solving ODEs_：continuous extension 与 event location。
  [DOI 10.1137/0722060](https://doi.org/10.1137/0722060)
  [DOI 10.1016/0893-9659(88)90062-6](https://doi.org/10.1016/0893-9659%2888%2990062-6)
- W3C, _WebGPU Shading Language_：binary32、finite-math、reassociation/fusion 与 `fma`。
  [WGSL specification](https://www.w3.org/TR/WGSL/)
- gfx-rs, Naga `30.0.0` source 与 wgpu `30.0.0` timestamp API。
  [Naga on docs.rs](https://docs.rs/naga/30.0.0/naga/)
  [wgpu on docs.rs](https://docs.rs/wgpu/30.0.0/wgpu/)

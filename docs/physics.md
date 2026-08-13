# 数学物理合同

本文固定 Gravlume 的符号、连续模型和可观测量。任何实现若采用另一 chart 或状态表示，必须显式转换到这里定义的 observable 再比较。离散积分、fixture 和容差属于[验证合同](validation.md)。

## 1. 约定

### 1.1 单位与坐标

- 几何单位 $G=c=1$，质量、长度和时间同量纲；
- 四维坐标顺序 $X^\mu=(t,x,y,z)$；
- 度规符号 $(-+++)$，$\eta_{\mu\nu}=\operatorname{diag}(-1,1,1,1)$；
- 自旋轴沿 $+z$，正 $a=J/M$ 按右手定则；
- geometric electric charge 写作 $q_e$，Carter 常数写作 $\mathcal Q$，禁止都写成 `Q`；
- canonical CPU 状态使用 `f64`；UI/存档记录质量尺度，GPU pack 可无量纲化为 $M=1$。

Rust/glam 的 `Vec4(x,y,z,w)` 不直接代表四向量；领域 wrapper 必须注明分量顺序。

### 1.2 参数状态

Kerr–Newman family 的几何参数满足：

\[
M>0,\qquad a=J/M.
\]

\[
a^2+q_e^2
\begin{cases}
<M^2 & \text{subextremal},\\
=M^2 & \text{extremal},\\
>M^2 & \text{superextremal}.
\end{cases}
\]

superextremal 是理想化、无事件视界的解，不是一般意义上的“无效参数”，但默认产品不把它称为黑洞。$q_e$ 是几何化电荷参数；没有明确 SI 转换和天体模型时，只能解释为对解族的研究。

参数状态按 canonical binary64 输入的实际 bit pattern 判定，不按十进制显示值或已舍入的平方和判定。实现必须用指数对齐的整数 significand 等等价方法精确比较 $M^2$ 与 $a^2+q_e^2$，并从同一精确差值计算 horizon discriminant；只按 $\max(|M|,|a|,|q_e|)$ 缩放后再做普通浮点平方仍不足以保护近极端离散分类。Rust `f64` 的 binary64 布局与 [`to_bits`](https://doc.rust-lang.org/stable/std/primitive.f64.html#method.to_bits) 是这一 seam 的语言合同。

## 2. Cartesian Kerr–Schild geometry

### 2.1 椭球半径

以 $z$ 为自旋轴，Kerr–Schild radius $r\ge0$ 由

\[
\frac{x^2+y^2}{r^2+a^2}+\frac{z^2}{r^2}=1
\]

隐式定义。令 $R^2=x^2+y^2+z^2$、$u=r^2$、$b=R^2-a^2$，则

\[
u^2-bu-a^2z^2=0.
\]

稳定求根：

\[
u=
\begin{cases}
\dfrac{b+\sqrt{b^2+4a^2z^2}}2,& b\ge0,\\[6pt]
\dfrac{2a^2z^2}{\sqrt{b^2+4a^2z^2}-b},& b<0.
\end{cases}
\]

第二式避免 $b<0$ 时大数相消。轴线上必须先用解析恒等式 $r=|z|$ 及其导数极限，不能要求可表示的 $r$ 还必须有可表示的 $r^2$。测试必须覆盖 $a=0$、轴线、赤道面、远场和接近 ring 的条件数；不能以 `max(epsilon, ...)` 后的值替代未截断恒等式。

隐式微分给出

\[
\partial_xr=\frac{x r^3}{r^4+a^2z^2},\qquad
\partial_yr=\frac{y r^3}{r^4+a^2z^2},
\]

\[
\partial_zr=\frac{z r(r^2+a^2)}{r^4+a^2z^2}.
\]

定义域排除 ring singularity $r=0,z=0,x^2+y^2=a^2$。默认 non-negative-$r$ 单叶实现还把 $r=0,z=0,x^2+y^2<a^2$ 作为显式 `ChartBoundary`，不把 branch disk 误报成物理 ring singularity。Analytic-Extension View 若需要 signed $r$，必须使用另一显式 chart/state；默认 radius 函数不暗含符号分支。

### 2.2 Chart-handed oblate map 与 Kerr–Schild null covector

令 $s=+1$ 表示默认 ingoing branch，$s=-1$ 表示 outgoing branch。$a=J/M$
始终是相对固定右手 $(x,y,z)$ orientation 的物理自旋；chart 的 oblate spatial twist
为 $a_s=s a$：

\[
x=(r\cos\phi_s-s a\sin\phi_s)\sin\theta,\quad
y=(r\sin\phi_s+s a\cos\phi_s)\sin\theta,\quad
z=r\cos\theta.
\]

半径、$\Sigma$、$\Delta$ 与参数分类只依赖物理 $a^2$，不把 $a_s$ 误存成另一份
spacetime parameter。定义

\[
l_\mu=\left(
1,
\frac{s r x+a y}{r^2+a^2},
\frac{s r y-a x}{r^2+a^2},
\frac{s z}{r}
\right),
\]

\[
l^\mu=\eta^{\mu\nu}l_\nu=\left(
-1,
\frac{s r x+a y}{r^2+a^2},
\frac{s r y-a x}{r^2+a^2},
\frac{s z}{r}
\right).
\]

在 spheroidal chart 中等价地有

\[
l=dt_s+s\,dr-a\sin^2\theta\,d\phi_s.
\]

所以两个 branch 在固定 $r,\theta$ 上都给出同一个物理
$g_{t\phi}=-(2Mr-q_e^2)a\sin^2\theta/\Sigma$。椭球约束精确推出
$\eta^{\mu\nu}l_\mu l_\nu=0$。`s` 是 chart/principal-null convention，不是光子
传播方向、正负频率或物理自旋的重定义。ingoing/outgoing 的相反 azimuth shift 可对照
[Campanelli et al. Eq. (22)–(24)、(34)–(36)](https://arxiv.org/pdf/gr-qc/0010034)；
Kerr–Newman 的 physical parameter 与 BL/KS 形式见
[Adamo–Newman Sec. 3.1](https://arxiv.org/pdf/1410.6626)。

### 2.3 Metric 与 inverse

定义

\[
f=\frac{2Mr^3-q_e^2r^2}{r^4+a^2z^2}
=\frac{2Mr-q_e^2}{r^2+a^2\cos^2\theta}.
\]

Kerr–Newman 的 Kerr–Schild 形式是

\[
g_{\mu\nu}=\eta_{\mu\nu}+f l_\mu l_\nu,
\qquad
g^{\mu\nu}=\eta^{\mu\nu}-f l^\mu l^\nu.
\]

因为 $l$ 对 Minkowski metric 为 null，秩一 inverse 是精确式：

\[
g_{\mu\alpha}g^{\alpha\nu}-\delta_\mu{}^\nu
=-f^2(l_\rho l^\rho)l_\mu l^\nu=0.
\]

原始解与笛卡尔表达可对照 [Newman et al. 1965](https://doi.org/10.1063/1.1704351) 和 [Adamo–Newman 2014](https://arxiv.org/abs/1410.6626)。[P] 实现至少测试 metric symmetry、inverse identity、$q_e\to0$ Kerr、$a,q_e\to0$ Schwarzschild 和 $M,a,q_e\to0$ Minkowski 极限。

## 3. Horizon、ergoregion 与观察域

\[
\Delta=r^2-2Mr+a^2+q_e^2,
\qquad
r_\pm=M\pm\sqrt{M^2-a^2-q_e^2}.
\]

亚极端时 $r_+$ 是外事件视界，$r_-$ 是 Cauchy horizon；极端时重合；超极端无实 horizon。

stationary-limit surfaces：

\[
r_{\mathrm{sl},\pm}(\theta)
=M\pm\sqrt{M^2-q_e^2-a^2\cos^2\theta}.
\]

horizon 与 stationary limit 不同。在 ergoregion 中 $\partial_t$ 不是 timelike，不能构造“坐标静止”observer；领域构造器必须拒绝或选择另一物理 observer。

Analytic-Extension View 只表示永恒解析解。Cauchy horizon 对扰动不稳定，质量膨胀见 [Poisson–Israel 1989](https://doi.org/10.1103/PhysRevLett.63.1663)；UI 不把延拓区解释为现实塌缩内部。[P]

## 4. Null geodesic 的 Hamilton 系统

对中性 photon，canonical state 为 $x^\mu,p_\mu$：

\[
H(x,p)=\frac12g^{\mu\nu}p_\mu p_\nu=0,
\]

\[
\dot x^\mu=g^{\mu\nu}p_\nu,
\qquad
\dot p_\mu=-\frac12\partial_\mu g^{\alpha\beta}p_\alpha p_\beta.
\]

令 $S=l^\mu p_\mu$，Kerr–Schild 结构化简为

\[
\dot x^\mu=\eta^{\mu\nu}p_\nu-fS l^\mu,
\]

\[
\dot p_i=\frac12S^2\partial_i f
+fS p_\mu\partial_i l^\mu,
\qquad
\dot p_t=0.
\]

GPU 与 CPU 分别实现同一闭式右端，不使用 finite difference。令

\[
A=r^2+a^2,\qquad D=r^4+a^2z^2,\qquad
N=2Mr^3-q_e^2r^2,\qquad f=N/D.
\]

对 $i\in\{x,y,z\}$，记 $r_i=\partial_i r$、$\delta_{iz}$ 为 Kronecker delta：

\[
D_i=4r^3r_i+2a^2z\delta_{iz},\qquad
N_i=(6Mr^2-2q_e^2r)r_i,
\]

\[
\partial_i f=\frac{N_iD-ND_i}{D^2}.
\]

空间分量 $\ell_x=l^x,\ell_y=l^y,\ell_z=l^z$ 的导数是

\[
\partial_i\ell_x=
\frac{s(r_i x+r\delta_{ix})+a\delta_{iy}}A
-\ell_x\frac{2rr_i}A,
\]

\[
\partial_i\ell_y=
\frac{s(r_i y+r\delta_{iy})-a\delta_{ix}}A
-\ell_y\frac{2rr_i}A,
\]

\[
\partial_i\ell_z=s\left(\frac{\delta_{iz}}r-\frac{z r_i}{r^2}\right),
\qquad \partial_i l^t=0.
\]

这些表达与第 2.1 节的 $r_i$ 一起构成 current 的 Kerr–Schild RHS 合同；任一 denominator guard 在求值前触发 typed numerical failure，不能以 clamp 改写方程。上述导数、$l^2=0$ 和 rank-one inverse 已用精确符号代数独立复算。[A]

stationarity 给出 $E=-p_t$，axisymmetry 给出 $L_z=xp_y-yp_x$。第 2.2 节的
chart-handed oblate coordinates 与同一个 physical-spin Kerr–Newman BL chart 局部满足

\[
dt_s=dt_{\rm BL}
+s\frac{2Mr-q_e^2}{\Delta}dr,\qquad
d\phi_s=d\phi_{\rm BL}+s\frac a\Delta dr.
\]

因此固定 $r$ 时 $p_t,p_\phi,p_\theta$ 相同。Cartesian covector 给出

\[
p_\theta=\cot\theta(xp_x+yp_y)-r\sin\theta\,p_z,
\qquad p_\phi=L_z.
\]

对中性 null geodesic，本项目的 Carter 常数定义为

\[
\mathcal Q=p_\theta^2+\cos^2\theta
\left(\frac{L_z^2}{\sin^2\theta}-a^2E^2\right).
\]

数值实现不先求 $\phi$。当 $\rho=\sqrt{x^2+y^2}>0$ 时直接计算

\[
\frac{L_z}{\sin\theta}
=\sqrt{r^2+a^2}\frac{xp_y-yp_x}{\rho};
\]

在精确轴线上使用连续极限

\[
\mathcal Q_{\rm axis}=(r^2+a^2)(p_x^2+p_y^2)-a^2E^2.
\]

$a=0$ 时该定义精确化为 $\mathcal Q=L^2-L_z^2$。常数来自 Hamilton–Jacobi separability，见 [Carter 1968](https://doi.org/10.1103/PhysRev.174.1559)。坐标变换、轴极限和 Schwarzschild 极限已用符号代数验证；near-axis 与 Killing-tensor contraction 的 overlap test 仍是实现 gate。`[P][A]`

每条 trace 至少报告：

- normalized null residual；
- $E,L_z,\mathcal Q$ 相对/绝对 drift；
- accepted/rejected steps 和最小/最大 step；
- event residual、bracket 和 ambiguity；
- finite/radicand/denominator guard；
- terminal classification。

约束 drift 是诊断，不是唯一误差。临界轨道可在守恒漂移很小时仍产生错误 branch 或数弧度角度误差。

## 5. Observer Frame 与 View Ray

observer four-velocity $u$ 与 spatial tetrad $e_{(i)}$ 满足

\[
u\cdot u=-1,\quad
u\cdot e_{(i)}=0,\quad
e_{(i)}\cdot e_{(j)}=\delta_{ij}.
\]

给定 observer rest space 中单位 arrival direction $n=n^ie_{(i)}$，future-directed photon momentum：

\[
p^\mu=\omega_{\rm obs}(u^\mu+n^ie_{(i)}^\mu),
\qquad \omega_{\rm obs}>0.
\]

于是 $p^2=0$ 且 $-p\cdot u=\omega_{\rm obs}$。

### 5.1 Viewport Sample 与初始光线

Observer Frame 的空间轴写作 $(e_R,e_U,e_A)$：$e_R$ 指 image-right，$e_U$ 指 image-up，$e_A$ 是中心像素收到的物理 arrival direction，并要求 $\epsilon(u,e_R,e_U,e_A)>0$。观察方向是 $-e_A$，不要把它命名为 photon direction。

对物理 extent $W\times H$，pixel index 从左上角开始，subpixel offset $\delta_x,\delta_y\in[0,1]$：

\[
\xi=2\frac{i+\delta_x}{W}-1,
\qquad
\eta=1-2\frac{j+\delta_y}{H}.
\]

令 vertical field of view 为 $\alpha\in(0,\pi)$、aspect $\gamma=W/H$：

\[
s_x=\gamma\tan(\alpha/2)\xi,
\qquad
s_y=\tan(\alpha/2)\eta.
\]

Sight Direction、arrival direction 与 Photon Momentum 分别为

\[
d=\frac{s_xe_R+s_ye_U-e_A}{\sqrt{1+s_x^2+s_y^2}},
\qquad n=-d,
\qquad p=\omega_{\rm obs}(u+n).
\]

因此 image-right/up 增大时 Backward Trace 朝 $+e_R/+e_U$ 走，而物理 photon 的空间动量相反。`Observation::initial_ray` 独占这一映射；CPU/WGSL 分别实现，并以中心、四角、奇数 extent 和固定 jitter fixture 验证 $d^2=1,p^2=0,-p\cdot u=\omega_{\rm obs}$。[A]

默认 stationary observer 只在 $g_{tt}<0$ 构造：

\[
u=\frac{\partial_t}{\sqrt{-g_{tt}}}.
\]

对任意 seed vector，rest-space view 是 $\Pi_u(v)=v+(u\cdot v)u$。默认 frame 将 observer 指向 target 的 seed 投影、归一化为中心 Sight Direction $d_0$，取 $e_A=-d_0$；再投影 spin-$z$ up hint 并从 $d_0$ 中 Gram–Schmidt 得到 $e_U$，最后选择唯一满足 orientation 的 $e_R$。up hint 退化时必须显式选择备用轴并记录，而非产生不连续翻转。

Backward Trace 使用负 affine step 或等价反向状态演化，从 observer 走向 source；保存的 Photon Momentum 仍 future-directed。所有 emitter/observer frequency 使用该物理 momentum，不能把 traversal tangent 代入后取绝对值。

### 5.2 Fermi–Walker observer motion

加速 observer 满足

\[
\nabla_u u^\mu=a^\mu,
\qquad a^\mu=a^{(i)}e_{(i)}^\mu.
\]

非旋转 frame 的 Fermi–Walker transport：

\[
\nabla_u e_{(i)}^\mu
=(u^\mu a_\nu-a^\mu u_\nu)e_{(i)}^\nu
=u^\mu a^{(i)}.
\]

连续式保持 $u\cdot e_i=0$ 和 $e_i\cdot e_j=\delta_{ij}$。离散实现若步后 Gram–Schmidt，必须同时记录投影前约束、投影后约束和 transport residual；“最后正交”不能证明正确输运。[Walker 1935](https://doi.org/10.1017/S0013091500008166) [P]

## 6. Event 与 terminal semantics

numerical baseline 的 event function 固定如下；$R_{\rm esc}$、$D_{\rm guard}$ 和步数上限来自 Validation Profile，不属于 Physical Scene：

| 候选 | 连续函数 | 额外条件 | crossing |
|---|---|---|---|
| outer horizon | $F_h=r-r_+$ | sub/extremal exterior；superextremal 不安装 | outside → inside |
| escape | $F_e=r-R_{\rm esc}$ | event 已 armed | inside → outside |
| equatorial surface | $F_d=z$ | localized $r\in[r_{\rm in},r_{\rm out}]$；surface enabled | either |
| singularity guard | $F_s=D-D_{\rm guard}$，$D=r^4+a^2z^2$ | numerical guard，不是物理表面 | safe → guard |
| chart boundary | chart-specific $F_c$ | 仅所选 chart 有限域时安装 | inside → outside |

初始点恰在某 event surface 时，该 event 必须先离开一个 profile 规定的 arming band，再允许反向 crossing；这使“从 escape sphere 入射、转向后返回”的 fixture 不会在 $\lambda=0$ 立即终止。step exhaustion、non-finite、非法 radicand/denominator 与 reject exhaustion 是 terminal conditions，不伪装成连续根。

reference 的 accepted step 若对 armed event 跨符号，使用[验证合同第 1 节](validation.md#1-cpu-reference-方法) dense output 在 $\theta\in[0,1]$ 内 bracket，并以 safeguarded Brent/bisection 定位；不得在 bracket 外 extrapolate。报告 event affine parameter、state、bracket width、函数尺度与 normalized residual。保证收敛的 bracketed 方法见 [Brent 1971/1973](https://maths-people.anu.edu.au/~brent/pub/pub006.html)。[P]

若同一步跨多个 event，选择 affine traversal 上最早者。候选的 localized affine parameter 在 `event_tie_tolerance` 内时，结果携带全部 candidate 和 ambiguity flag；稳定序列化顺序为 singularity/chart → horizon → emitter → escape，但该顺序不能删除 ambiguity。turning point 是动力学状态，不与 terminal event 混用。GPU 近似事件定位必须与 reference event observable 比较，而不是只比最终颜色。

外层 escape surface 是数值边界。不要将 metric 淡出到 Minkowski 而遗漏 fade gradient；更清楚的基线是在有限半径终止并把 escape direction/impact parameters 映射到规定的 asymptotic environment，随后量化有限边界误差。

## 7. Frequency 与 radiative transfer

任意 observer/emitter 测得

\[
\omega=-p_\mu u^\mu>0,
\qquad
g\equiv\frac{\nu_{\rm obs}}{\nu_{\rm em}}
=\frac{-p\cdot u_{\rm obs}}{-p\cdot u_{\rm em}}.
\]

这是局部不变量，不依赖坐标。若 observer 处归一为 $\omega_{\rm obs}=1$，source 是无穷远静止 observer，才可简化为 $g=1/E_\infty$；有限边界、移动 source 或 ergoregion 不能套用。

collisionless geometric optics 中

\[
\frac{I_\nu}{\nu^3}=\text{constant},
\]

所以

\[
I_{\nu,\rm obs}(\nu_{\rm obs})
=g^3 I_{\nu,\rm em}(\nu_{\rm obs}/g).
\]

只有频率积分的 bolometric intensity 才有

\[
I_{\rm obs}=g^4 I_{\rm em}.
\]

基础见 [Lindquist 1966](https://doi.org/10.1016/0003-4916%2866%2990207-7)。[P] 普通 RGB 不是光谱，不能同时作波长移动和 $g^3/g^4$ 定量解释。

标量 emission/absorption 使用 optical-depth slab 的解析更新：

\[
I_{\rm out}=I_{\rm in}e^{-\Delta\tau}
+S\left(1-e^{-\Delta\tau}\right),
\]

并用 `expm1` 等稳定形式处理小 $\Delta\tau$。source function、frequency convention、affine orientation 和 invariant coefficient 的转换在一处推导，并以 vacuum、pure absorption、constant slab、optically thin/thick limits 验证。[Younsi et al. 2012](https://doi.org/10.1051/0004-6361/201219599) [P]

## 8. Equatorial circular emitter 与 disk 边界

中性测试粒子的 Kerr–Newman equatorial circular angular velocity：

\[
\Omega_\pm=
\frac{\pm\sqrt{Mr-q_e^2}}
{r^2\pm a\sqrt{Mr-q_e^2}}.
\]

必要条件 $Mr\ge q_e^2$，但还必须验证轨道存在、timelike 且位于所选 branch。对

\[
u^\mu=u^t(\partial_t+\Omega\partial_\phi),
\]

\[
u^t=\left[-(g_{tt}+2\Omega g_{t\phi}
+\Omega^2g_{\phi\phi})\right]^{-1/2},
\qquad
\omega_{\rm em}=u^t(E-\Omega L_z).
\]

根式非正时必须返回“不存在该 timelike orbit”，不能 clamp 后继续当物理解。

dimensionless $x=r/M,\alpha=a/M,q=q_e/M$ 的一个 circular specific-energy branch：

\[
\mathcal E(x)=
\frac{x^2-2x+q^2+\alpha\sqrt{x-q^2}}
{x\sqrt{x^2-3x+2q^2+2\alpha\sqrt{x-q^2}}}.
\]

它用于 CPU orbit/ISCO 研究时必须检查完整 allowed domain 与 branch，不能只对固定区间做无条件单峰最小化。特殊基准：Schwarzschild $r_{\rm ISCO}=6M$，extremal RN 为 $4M$，extremal retrograde Kerr 为 $9M$，prograde extremal Kerr 极限为 $M$。Kerr–Newman 轨道域见 [Pugliese et al. 2013](https://doi.org/10.1103/PhysRevD.88.024042)。[P]

薄盘温度若只使用

\[
F_N(r)=\frac{3M\dot M}{8\pi r^3}
\left(1-\sqrt{\frac{r_{\rm in}}r}\right),
\qquad T_{\rm eff}\propto F_N^{1/4},
\]

必须标为 Newtonian radial profile。它不是 Page–Thorne/Novikov–Thorne GR disk；完整相对论通量及其平稳、轴对称、薄盘、近圆测地等假设见 [Page–Thorne 1974](https://articles.adsabs.harvard.edu/pdf/1974ApJ...191..499P)。[P]

## 9. Polarization 最低合同

linear polarization vector $f^\mu$ 满足

\[
p\cdot f=0,\qquad f\cdot f=1,
\qquad p^\nu\nabla_\nu f^\mu=0,
\]

并有 gauge freedom

\[
f^\mu\sim f^\mu+\beta p^\mu.
\]

Kerr 中可以使用 Penrose–Walker constant 加速真空输运；但采用前必须从本项目的 signature、index placement、tetrad、spin axis 和 momentum orientation 推导，并与直接平行输运比较。原始基础见 [Walker–Penrose 1970](https://doi.org/10.1007/BF01649445) 和 [Connors–Stark 1977](https://doi.org/10.1038/269128a0)。[P]

screen basis 不能只把 camera right/up 分别投影并归一化。若

\[
R'=R-(R\cdot n)n,\qquad
Y'=Y-(Y\cdot n)n,
\]

即使 $R\cdot Y=0$，也有

\[
R'\cdot Y'=-(R\cdot n)(Y\cdot n),
\]

通常不为零。必须先构造一根稳定 screen vector，再在 screen plane 内叉乘或 Gram–Schmidt 得到第二根，并检查 handedness、条件数和 gauge invariance。

Kerr polarization 的最低 gates：沿轨迹的 $p\cdot f$、$f\cdot f$、gauge transform、Walker–Penrose drift、screen EVPA 与直接 ODE agreement。Kerr–Newman 或 plasma Stokes 在这些门槛闭合前不进入默认产品。

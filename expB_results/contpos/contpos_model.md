# 连续 UAV 位置子问题：数学模型与可求解性分析（任务 1）

日期：2026-10-14。阶段 contpos（连续位置优化研究）。本文件只做推导与数值事实核验，
不修改任何生产代码；任务 2 诊断脚本与任务 3 求解器建议均以此为准。

**声明约束（本阶段硬规则）**：不宣称位置问题为凸；不把投影梯度法称为 SCA；
Binding K5 保留为安全回退；不宣称全局最优或严格 oracle。源码依据：
`blocka_optimizer.py`（P_R 与 Γ 标定）、`channel_model.py`（pure LoS）、
`utils_sys.py::cal_channel_fading_gain` / `cal_uav_propulsion_energy`、
`_grid_residual.py::solve_full`（对偶输出，exec 前缀复用）。

---

## 0. 范围与约定

单时隙、内层输入全部冻结的连续位置子问题：

- **固定输入**（给定快照）：用户位置 $p_i=(x_i,y_i)$、关联 $c_i\in\{0..3\}$、
  需求 $d_i=Q_i+A_i\ge 0$、权重 $\beta_i$（OFF/负权重用户不参与）、能量状态
  $E_u=E_u^{dec}$、飞艇位置 $a$、UAV 水平初位置 $q_u^0$、**UAV 高度固定**
  $z_u=z_u^0\in[100,200]$（本阶段先固定飞艇与高度，只优化水平位置）。
- **决策变量**：4 架 UAV 的水平终位置 $q=(q_1,\dots,q_4)\in\mathbb{R}^{4\times2}$，
  等价水平位移 $\Delta_u=q_u-q_u^0$。
- **净位移速度**：$v_u=\|\Delta_u\|_2/\tau$，$\tau=$ `slot_len`=1 s。口径为
  **实际净位移**（起点到终点，非路径长；盒裁剪只会缩短 $\|\Delta\|$，与
  `eval_cand` 完全一致）。
- **外层目标**（最小化）：

  $$J(q)=\underbrace{\psi_E\sum_{u}E_u\,P(v_u)}_{\text{能量项}}\;-\;\underbrace{U^*(q)}_{\text{Block A 值函数}},\qquad
    \psi_E=\frac{1}{PW\_MAX}=\frac{1}{683},\quad U^*(q)=\max_{(W,D)\in\mathrm{P_R}(q)}\sum_i\beta_i D_i .$$

  这与 `_grid_residual.py::evalJ` / `_t2_validate.py::eval_cand` 的计算式逐项相同
  （`J = PSI_E * E0@P - beta@D_Mb`），因此本文推导的梯度就是实际评估函数的梯度。
- **内层 Block A** = 现有 P_R（cvxpy，`rel_entr` 负指数锥形式），约束、活跃集、
  求解器回退顺序全部不变；**用户关联在本子问题内固定**（不含关联整数层）。
- **安全回退**：方案 A（Binding K5）始终可用；任何连续方案失败/非法时返回 K5，
  最终候选 = 过滤非法后的 $\min(\text{K5},\text{连续方案})$，保证不更差、不非法。

### 关键常数（源码核验）

| 量 | 值 | 来源 |
|---|---|---|
| $\tau$ / `boundary` / $v_{xy}^{max}$ / $z$ 域 | 1 s / 250 m / 30 m/s / [100,200] | params_sys |
| $W_U$（接入池） / $W_A$（回传池） | 40 MHz / {20, 30} MHz | bw_total_Hz / 显式传参 |
| $PW\_MAX$ / $\psi_E$ | 683 W / 1/683 | params_sys L197（仅标度，无钳制） |
| `pTx_downlink` / `pTx_airship` | 0.5 W (27 dBm) / 0.3162 W (25 dBm) | params_sys |
| $N_0^{acc}$ / $N_0^{bh}$（PSD） | $2\times10^{-18}$ / $2\times10^{-19}$ W/Hz | blocka_optimizer L51-52 |
| $P_{ref}$（`ref_receive_pw`） | $10^{-4}$ W | params_sys |
| $c_{acc}=P_{dl}/(N_0^{acc}\cdot10^6)$ | $2.5\times10^{11}$ | γ = c·g（MHz） |
| $c_{bh}=P_{air}/(N_0^{bh}\cdot10^6)$ | $1.58114\times10^{12}$ | 同上 |
| $P_0$ / $P_i$ / $U_{tip}$ / $v_0$ | 56.7190 W / 31.3534 W / 80 / 4.03 | Zeng Eq.13 实算（非论文 79.856/88.628 注释值） |

---

## 1. 内层 Block A（P_R）及其对偶

对给定位置 $q$，Γ 由位置唯一确定（见 §4），内层为：

$$
\begin{aligned}
U^*(\gamma)=\quad &\max_{W\ge0,\,D\ge0,\,W_b\ge0}\ \sum_{i\in\mathcal{A}}\beta_i D_i\\
\text{s.t.}\quad
& D_i \le \tau\,W_i\log_2\!\Big(1+\frac{\gamma_i}{W_i}\Big)
&&(\mu_i\ge0,\ \text{接入速率})\\
& D_i \le d_i
&&(\xi_i\ge0,\ \text{需求封顶})\\
& \textstyle\sum_i W_i \le W_U
&&(\nu_{acc}\ge0,\ \text{接入池})\\
& \textstyle\sum_{i:c_i=u} D_i \le \tau\,W_{b,u}\log_2\!\Big(1+\frac{\gamma^b_u}{W_{b,u}}\Big)
&&(\nu_u\ge0,\ \text{回传，}u\in\mathcal{U}_{srv})\\
& \textstyle\sum_u W_{b,u} \le W_A
&&(\nu_{bh}\ge0,\ \text{回传池})
\end{aligned}
$$

其中 $\mathcal{A}$ 为活跃集（`demand>0 & assoc>=0 & beta>0`），$\mathcal{U}_{srv}$
为实际服务的 UAV 集合；带宽单位 MHz、速率 Mb/s、$\gamma$ 单位 MHz（口径与
`solve_blocka` 一致：$f(W)=W\log_2(1+\gamma/W)=-\mathrm{rel\_entr}(W,W+\gamma)/\ln 2$）。

内层是凸问题（$f$ 在 $W$ 上严格凹：$f''(W)=-\gamma^2/(\ln2\cdot W(W+\gamma)^2)<0$，
hypograph 约束凸，目标线性），且 **Slater 成立**：取充分小的 $W_i,W_{b,u}>0$ 与
$0<D_i<\min(\tau f(W_i), d_i)$，两条池约束留有严格余量 ⇒ 强对偶 + 对偶可达，
KKT 充要。`_grid_residual.py::solve_full`（L83-91）已返回所需对偶量：
`lam_acc`（$\mu$）、`nu_bh`（$\nu_u$）、`nu_acc_pool`、`nu_bh_pool`
（实现注意：对 `lam_acc`、池对偶取 `abs`，对 `nu_bh` 取 `max(0,·)`，沿用此口径）。

**位置只通过 $\gamma(q)$ 进入内层**：活跃集、需求、$\beta$、池大小、$\tau$、关联
均与 $q$ 无关。这是外层梯度成立的前提（§7 条目 1）。

---

## 2. 可行域

逐 UAV 的可行集（水平二维）：

$$\mathcal{F}_u=\big\{\,\|q_u-q_u^0\|_2/\tau \le 30\,\big\}\ \cap\ \big\{\,|x_u|,|y_u|\le 250\,\big\}\ \cap\ \big\{\,P(\|q_u-q_u^0\|_2/\tau)\le 683\,\big\},\qquad \mathcal{F}=\prod_u \mathcal{F}_u .$$

（速度约束在净位移口径下就是欧氏半径 $30\tau$ 的盘；高度不动 ⇒ 无垂直项。）

### 引理 1（功率子水平集是单区间）

实测（`cal_uav_propulsion_energy` 逐点，无任何内部钳制——L447-499 已核验）：

- $P(0)=88.072$；$P$ 在 $[0,5.74]$ 严格递减至 $P_{\min}=81.862$，其后严格递增；
- $P''<0$ 恰在连续区间 $[0,2.96]$（步长 0.005 扫描，594 个负区间全部相邻），
  在 $(2.96,30]$ 上 $P''>0$ ⇒ $[5.74,30]$ 上 $P'\ge0$ 单调不减；
- $P(30)=683.0661>683$，$P(30^{-})<683$ ⇒ $P(v)=683$ 在上升支有唯一根
  $v_{pw}=29.998922$（brentq）。

于是 $\{v\in[0,30]:P(v)\le683\}=[0,\,29.998922]$：$P$ 连续、先降后升、单次穿越。
（初值 $88.072\ll683$，远高于下极值 $81.862$，所以不会出现双区间。）

**推论**：
1. 功率约束给出半径 $v_{pw}\tau=29.998922$ 的盘，**严格强于**速度盘（30）；
   即当前系统里实际绑定的移动约束是功率约束（差 0.001 m/s，等效即 $v\le v_{pw}$）。
2. $\mathcal{F}_u$ = 盘 ∩ 盒 ⇒ **可行集是凸集**；本问题的非凸性**完全来自目标**
   （§8）。回传奇点 $q_u=a$（UAV 与飞艇重合）用 $\|q_u-a\|\ge\varepsilon$ 的
   可行性检查排除（实践中不会发生，见 §4）。
3. **阈值依赖性（警示）**：若功率阈值落在 $(81.862,\,88.072)$，子水平集变成
   $[0,a]\cup[b,c]$ 两个区间 ⇒ 可行域为环带（非凸）。当前 $PW\_MAX=683$ 远在
   $88.072$ 之上，故为单区间；此结论依赖"阈值 > P(0)"这一事实，不能外推。

---

## 3. 信道模型与位置梯度

`utils_sys.py::cal_channel_fading_gain`（L15-22）：

$$g(p_i,q_u)=A(\theta)\,P_{ref}\,d^{-n(\theta)},\quad d=\sqrt{r^2+h^2},\ r=\|p_i-q_u\|_2,\ h=z_u^0,\ \theta=\arctan\!\big(h/(10^{-3}+r)\big)\ \text{(deg)}.$$

`channel_model.py` 当前配置为 **pure LoS**：`prob_LoS` 中 $a=0\Rightarrow P_{LoS}\equiv1$
（与 $\theta$ 无关），于是

- 路径损耗指数 $n=-1.5\cdot P_{LoS}+3.5\equiv 2.0$（自由空间，常数）；
- 阴影增益均值 $A=P_{LoS}+(1-P_{LoS})\cdot0.2\equiv 1$（常数）；

**故 $g=P_{ref}/(r^2+h^2)$ 精确成立**，$\theta$、$10^{-3}$ 护栏全部退化消失。

### 3.1 接入侧梯度

$$\gamma_i(q)=c_{acc}\,g_i,\qquad
\nabla_{q_u} g_i=\frac{2P_{ref}\,(p_i-q_u)}{(r^2+h^2)^2}=\frac{2\,g_i\,(p_i-q_u)}{r^2+h^2}\ \ (c_i=u),$$

$\nabla_{q_v}g_i=0\ (v\ne c_i)$。分母 $\ge h^2=150^2>0$ ⇒ **处处光滑、无奇点**
（即使 $r=0$ 也由 $h$ 兜底；$10^{-3}$ 护栏只在 $h=0$ 时才需要，而 $h\ge100$）。

### 3.2 回传侧梯度

`blocka_optimizer.py::cal_gamma_backhaul_MHz`（L60-64）：$g^b_u=P_{ref}/\|q_u-a\|_2^2$
（三维平方距离，airship 固定），$\gamma^b_u=c_{bh}g^b_u$，故

$$\nabla_{q_u} g^b_u=-\frac{2P_{ref}\,(q_u-a)}{\|q_u-a\|_2^{4}} .$$

光滑域要求 $\|q_u-a\|>0$；可行性检查中加 $\varepsilon$ 保护即可
（快照中飞艇与 UAV 不会重合）。

### 3.3 一般（仰角相关）形式与退化说明

若恢复原概率 LoS 配置，$A'(\theta)\ne0$、$n'(\theta)\ne0$，则
$\nabla g=P_{ref}\big[A' d^{-n}-A n d^{-n-1}\nabla d-A\ln d\cdot d^{-n}\,n'(\theta)\,\nabla\theta\big]$，
即除距离项外还含 $\nabla\theta$ 项（$\theta$ 经 $r$ 依赖位置）。**当前配置
$A'\equiv0,\ n'\equiv0$，仰角导数项恒为零**——因此 §3.1/§3.2 的简洁式在整个
可行域内精确，不存在"近似忽略仰角"的问题。

---

## 4. 推进功率：解析式、可微性、非凸性

`utils_sys.py::cal_uav_propulsion_energy`（L447-499，Zeng et al. 2019 Eq.13，
$V_z=0$ 时垂直项为 0）：

$$P(V)=\underbrace{P_0\Big(1+\tfrac{3V^2}{U_{tip}^2}\Big)}_{\text{桨叶}}+\underbrace{P_i\sqrt{\sqrt{1+\tfrac{V^4}{4v_0^4}}-\tfrac{V^2}{2v_0^2}}}_{\text{诱导}}+\underbrace{\tfrac12 d_0\rho s A\,V^3}_{\text{寄生}},\qquad
\begin{aligned}P_0&=\tfrac{\delta}{8}\rho sA\omega^3R^3=56.7190\\ P_i&=\tfrac{(1+k)W^{1.5}}{\sqrt{2\rho A}}=31.3534\end{aligned}$$

（$\omega=200,R=0.4,v_0=4.03,d_0=0.6,s=0.12,\rho=1.225,A=\pi R^2,U_{tip}=80,\delta=0.012,k=0.1,W=10$；注释里的 $P_0=79.856,P_i=88.628$ 是论文参考值，**未被使用**。）

### 4.1 导数

$$P'(V)=\frac{6P_0V}{U_{tip}^2}+P_i\frac{u'(V)}{2\sqrt{u(V)}}+\tfrac32 d_0\rho s A\,V^2,\qquad
u(V)=\sqrt{1+\tfrac{V^4}{4v_0^4}}-\tfrac{V^2}{2v_0^2},$$
$$u'(V)=\frac{V^3}{2v_0^4}\Big(1+\tfrac{V^4}{4v_0^4}\Big)^{-1/2}-\frac{V}{v_0^2}.$$

- **零速**：$u(0)=1,\ u'(0)=0\Rightarrow P'(0)=0$（精确，非数值巧合；
  实测 $P(10^{-6})-P(0)=-4.5\times10^{-13}$）。
- **零速二阶**：$P''(0)=\dfrac{6P_0}{U_{tip}^2}-\dfrac{P_i}{2v_0^2}=-0.9121$
  （解析；数值差分 $-0.9114$ 吻合）。
- **分段凸性**（步长 0.005）：$P''<0$ 于 $[0,2.96]$（连续），$P''>0$ 于
  $(2.96,30]$；$P'$ 在 $(0,5.74)$ 为负、$(5.74,30]$ 为正。
- **非单调**：$P$ 从 $88.072$ 降到 $81.862$（$V=5.74$）再升到 $683.0661$（$V=30$）。

### 4.2 复合 $E(q)=P(\|\Delta\|/\tau)$ 的可微性

- $\Delta\ne0$：$\nabla_{q}E=\dfrac{P'(v)}{\tau}\dfrac{\Delta}{\|\Delta\|}$，$C^\infty$。
- $\Delta=0$：因 $P'(0)=0$，Taylor 给出 $E=P(0)+\tfrac12 P''(0)\|\Delta\|^2/\tau^2+o(\|\Delta\|^2)$
  ⇒ 梯度连续延拓为 $\mathbf 0$；**$C^2$ 但非 $C^3$**（$V^3$ 项 ⇒ $\|\Delta\|^3$，
  三阶导在 0 处跳变）。一阶/二阶梯度法只需 $C^1/C^2$，足够。
- **非凸性（关键）**：$P''(0)<0$ ⇒ $E_u>0$ 时能量项在 $\Delta=0$ 处**局部严格凹**
  （悬停是能量项的局部极大，任意小位移都省电——因为诱导功率主导悬停）；
  叠加 $P$ 在 $[0,5.74)$ 非单调 ⇒ 能量项全局既不凸也不凹。
  只有在 $v\ge5.74$ 子域上（$P$ 凸且不减）$P\circ\|\cdot\|$ 才凸，而定义域包含
  $[0,5.74)$。

---

## 5. 值函数与对偶梯度（包络定理）

### 5.1 内层值对 $\gamma$ 的敏感度

对最大值问题、约束写作 $g(x,\gamma)\le0$ 且乘子 $\lambda\ge0$：$dU^*/dp=-\lambda^\top\partial g/\partial p$
（验证例：$\max D\ \text{s.t.}\ D\le\gamma\Rightarrow U^*=\gamma$，$g=D-\gamma$，
$\lambda=1$，$-\lambda\,\partial g/\partial\gamma=+1$）。本问题中

$$\frac{\partial U^*}{\partial\gamma_i}=-\mu_i\frac{\partial}{\partial\gamma_i}\Big[D_i-\tau W_i\log_2\big(1+\tfrac{\gamma_i}{W_i}\big)\Big]_{(W^*,D^*)}=\mu_i\cdot\frac{\tau\,W_i^*}{\ln2\,(W_i^*+\gamma_i)}\ \ge 0,$$

同理 $\dfrac{\partial U^*}{\partial\gamma^b_u}=\nu_u\cdot\dfrac{\tau\,W_{b,u}^*}{\ln2\,(W_{b,u}^*+\gamma^b_u)}\ge0$。
（恒等式 $\frac{d}{d\gamma}W\log_2(1+\gamma/W)=\frac{W}{\ln2\,(W+\gamma)}$；$W_i^*=0$
时右端趋于 0，连续补值。）**池对偶 $\nu_{acc},\nu_{bh}$ 与需求对偶 $\xi$ 不直接出现**
——它们通过 $W^*,\mu,\nu$ 的取值间接影响梯度，但 $\gamma$ 无直接依赖。

### 5.2 链式法则与完整梯度

$\gamma_i=c_{acc}g_i(q_{c_i})$、$\gamma^b_u=c_{bh}g^b_u(q_u)$，一次 `solve_full(q)`
同时给出 $J(q)$ 与全部对偶量（**无需对 $W^*,D^*$ 求导**，包络定理）：

$$\boxed{\ \nabla_{q_u}J(q)=\underbrace{\psi_E E_u\frac{P'(v_u)}{\tau}\frac{\Delta_u}{\|\Delta_u\|}}_{=0\ (\Delta_u=0)}\;-\;\underbrace{\sum_{i:c_i=u}\mu_i\frac{\tau W_i^*}{\ln2(W_i^*+\gamma_i)}\,c_{acc}\,\nabla_{q_u}g_i}_{\text{接入项}}\;-\;\underbrace{\nu_u\frac{\tau W_{b,u}^*}{\ln2(W_{b,u}^*+\gamma^b_u)}\,c_{bh}\,\nabla_{q_u}g^b_u}_{\text{回传项}}\ }$$

符号直觉：位移增大 ⇒ $\nabla g$ 指向用户/飞艇反方向（增益下降 ⇒ $U^*$ 下降 ⇒
$J$ 上升，负号来源）；位移增大 ⇒ 功率项方向由 $P'$ 决定（$v<5.74$ 时 $P'<0$，
**小步移动反而是下降方向**）。单位：$W\in$ MHz、$\gamma\in$ MHz、
$c_{acc},c_{bh}$ 见常数表、$\mu,\nu$ 为 `solve_full` 返回口径。

计算成本：闭式梯度 = 与目标评估**同一次** Block A 求解（对偶随解免费返回）；
对比数值梯度（8 维中心差分）需 $2\times8=16$ 次求解/点。

---

## 6. 正则性与可微条件清单

| # | 条件 | 状态 | 说明 |
|---|---|---|---|
| 1 | 关联与活跃集固定（不随 $q$ 变） | **成立** | 子问题定义即固定；`active` 掩码只依赖 demand/type/$\beta$。$U^*$ 只经 $\gamma(q)$ 依赖 $q$。 |
| 2 | 内层 Slater / 强对偶 / 对偶可达 | **成立** | §1：严格可行点显式可取，P_R 凸 ⇒ KKT 充要，$\mu,\nu,\xi$ 存在。 |
| 3 | 对偶解局部唯一（严格互补/非退化） | **无保证** | 失效处 $U^*$ 出现折点，公式给出的是求解器对偶点对应的单侧/次梯度元素而非真梯度。发生频率与误差由任务 2 有限差分对拍量化。 |
| 4 | $\gamma(q)$ 光滑、无奇点 | **成立** | 接入分母 $\ge h^2>0$ 处处成立；回传需 $\|q_u-a\|>0$（可行性检查加 $\varepsilon$）。 |
| 5 | $P$ 可微、$P'$ 有界 | **成立** | $(0,30]$ 上 $C^\infty$；$0$ 处 $P'(0)=0$ ⇒ 复合可微且梯度延拓为 0；$C^2$ 非 $C^3$；$P''$ 在 $[0,30]$ 有界 ⇒ $\nabla J$ Lipschitz（除内层折点外）。 |
| 6 | 求解器精度 | **逐解记录** | `optimal_inaccurate` ⇒ 对偶值带噪，任务 2 记录状态计数；此前 48 快照 ×2 带宽 1152 解中仅 2 次 inaccurate。 |
| 7 | 对偶符号口径 | **沿用生产代码** | `solve_full` 对 $\mu$、池对偶用 `abs`，对 $\nu_u$ 用 `max(0,·)`；若某解返回负对偶，`abs` 会翻转该分量符号 —— 有限差分对拍可系统性捕获此类错误。 |
| 8 | 盒边界最优点 | **投影/KKT 处理** | 梯度在边界仍有效；投影梯度/SLSQP 以盒约束处理，不需修改公式。 |
| 9 | $E_u=0$ 的快照 | **已知情形** | 能量项及其梯度为零 ⇒ $J=-U^*$，$\Delta=0$ 处一般非驻点（存在改善方向）；此前 48 快照中 25 个 $E_0$ 全零，此类快照的收益全部来自速率项。 |

---

## 7. 为什么整体非凸（不作凸性声明）

四条**独立**理由，任何一条都足以否定凸性：

1. **能量项非凸**（§4.2）：$P''(0)=-0.9121<0$ ⇒ 悬停点处能量项局部严格凹
   （$E_u>0$ 时）；且 $P$ 在 $[0,5.74)$ 非单调，复合既不凸也不凹。
2. **速率项无凸性保障**（§3）：$\gamma(q)=c/(r^2+h^2)$ **非凹**——Hessian 符号不定
   （$q=p$ 处沿水平方向负定；远处径向 $\gamma_{ss}\nabla s\nabla s^\top$ 主导为正）；
   且 $U^*$ 对 $\gamma$ 的凹性未证（共享接入/回传池上的 max 结构不保持凹性，
   单用户单约束的 $\min(d,\tau W_U\log_2(1+\gamma/W_U))$ 之外没有可援引的定理）。
   非凹函数的非凹函数的复合 ⇒ $-U^*(\gamma(q))$ 既不凸也不凹无从谈起。
3. **两项相加**：可行集是凸集（盒 ∩ 盘，引理 1），但目标两项均非凸 ⇒
   **非凸性完全来自目标函数**。
4. **额外困难（非凸性之外）**：内层活跃集随 $q$ 变化（需求封顶绑定切换、
   池绑定切换）⇒ $U^*$ 仅分段 $C^1$、存在折点（这是非光滑性，独立于凸性）；
   若放开关联，则叠加混合整数层（本子问题已固定关联，不含此层）。

数值层面的非凸证据（多初值敏感性、局部解散布）由任务 2 实测给出，本文不预设结论。

---

## 8. 方法定位（给任务 2/3 的框架）

- **问题类别**：盒 ∩ 盘约束（凸集）、分段光滑、非凸、8 维（方案 C）或 2 维
  （方案 B）；目标梯度可用且每次评估只需 1 次 Block A 求解（§5.2）。
- **合法方法**：非凸投影梯度（回溯线搜）、SLSQP/信赖域 SQP、CMA-ES/Powell/
  Nelder-Mead 无导数法、多初值局部搜索。**全部只保证收敛到驻点/局部解**；
  报告中不得出现"全局最优""严格 oracle"字样。
- **投影梯度法 ≠ SCA**：SCA 要求构造凸代理问题序列并证明代理不等式等结构性质；
  本文的梯度是**原目标的梯度**，不做代理替换，故不称 SCA、不做任何 SCA 声明。
- **Binding K5 = 安全回退**：最终候选 = 合法性过滤后的
  $\min\{\text{K5},\text{方案 B},\text{方案 C}\}$，任何异常（求解失败、
  非法约束、NaN、超时）回落 K5 ⇒ 不可能比 K5 更差或非法。
- **计量口径**：所有方案按 Block A 求解次数计预算，记录每次耗时与状态
  （任务 2 的 CSV 列），供任务 3 判断"有限预算下哪条路线最有希望"。

### 任务 2 将依据本文验证的六项

1. 方案 B（2 维方向混合 + 步长连续）相对 Binding K5 的增益；
2. 方案 C（8 维热启动于 K5）相对方案 B 的额外增益；
3. 增益是否集中于"接入—回传共同敏感"状态（mixed 层 / $n_\mu,n_\nu>0$）；
4. 高预算连续优化能否发现候选集之外的明显更优位置；
5. §5 对偶梯度 vs 中心有限差分的一致性（含折点处的单侧行为）；
6. 哪条连续路线在有限 Block A 预算下最有希望（收益/调用/耗时/异常综合）。

# 独立算法与实现审计 — 第一轮报告

日期：2026-10-14
审计员：Independent Algorithm & Implementation Auditor（本轮只审计，不修复、不重构、不开发新算法）
范围：静态代码审计 + 少量独立测试（未运行 ≥501 槽完整实验，未重训模型）

---

## 0. 本轮实际读取的核心文件

**生产代码（uav-bs-placement-drl-modified-v13b-K20-s11/）**

| 文件 | 角色 |
|---|---|
| `main.ipynb`（55 cells，cell 22/24/26 = Stationary/Centroid/Proposed） | 仿真主入口 + 三方法循环 + KPI 导出 |
| `main_executed.ipynb` | 上一次执行的输出（含约束自检打印） |
| `params_sys.py` | 系统参数、队列/能量/速度/功率常量 |
| `utils_sys.py` | `cal_channel_fading_gain`、`cal_uav_propulsion_energy`、`mos_func`、`kmeans_clustering`、`gen_heatmap`、`update_queue_Mb`、`find_active_users` |
| `mobility_model.py` | `adjust_location` / `adjust_altitude` |
| `utils.py` | 单位换算、`MHz/dB/dBm` |
| `blocka_optimizer.py` | Block A 凸优化（`solve_blocka` / `blocka_allocate`） |
| `relay_hybrid_controller.py` | **生产 Proposed 控制器**（候选打分 + REINFORCE，J=drift+MOS） |
| `relay_lyapunov_controller.py` | 基类（能量/回传虚拟队列、交付子问题二分） |
| `relay_hybrid_slow_controller.py` | 慢速变体 |
| `joint_ac_controller.py` | **离线审计所用控制器**（J=ψ_E·E·P − β·D，β 公式与用户描述一致） |

**实验/审计代码**

- `expA_dual_diagnostics.py`（生成 `expA_results/snapshots.npz`，152 冻结快照 + `base_diagnostics.npz`）
- `expB_wa30_audit.py`、`expB_ablation_guidance.py`、`expB_scenario_audit.py`、`expB_selective_mobility.py`
- `expB_stage1.py` / `expB_stage2.py` / `expB_task*.py`（闭环重放）
- `run_blocka_comparison.py`（README 头号结论的来源）
- 报告：`expA_results/expA_summary.md`；`expB_results/{scenario_audit,selective_mobility,wa30_audit}/*.md`
- 数据：`expA_results/{snapshots,base_diagnostics}.npz`、`expB_results/**/*.csv|npz`

**未找到（必须指出）**

- **正式模型文档缺失**：代码引用的 `BLOCKA求解.md`、`actor-critic部分.md` 在仓库中不存在（全盘搜索无命中）。因此"官方模型"无法据文档裁定，只能对比代码之间是否自洽。这本身是首要缺口。
- 生产运行的**逐槽功率日志不存在**：KPI 导出列（`qlen-Mb-avg(active)`、`mos-weighted-avg` 等）不含功率/速度，故无法从产物统计超限槽数，只能依赖 cell 26 的自检打印。

---

## 1. 项目完整调用链

```
params_sys.py (参数; rng=np.random.default_rng() 未播种; np.random.seed(0))
        │
main.ipynb cell 6–8  ← 读取 sim-input/*.xlsx（SFM 位置、Weibull 到达、业务类型）
        │              （三方法共用同一份外生输入 → 公平）
        ▼
cell 22 Stationary  ┐
cell 24 Centroid    ├─ 各自 501 槽循环：find_active_users → kmeans_clustering(无 seed)
cell 26 Proposed    ┘   → (Proposed) HybridCandidateController
        │
        ├─ (Proposed) ctrl.begin_slot(heatmap, uav_locs, airship_loc)     [relay_hybrid_controller L49]
        ├─ ctrl.decide_and_score(...)                                     [L62]
        │     ├─ _forward_single()  → actor 出 mu                          [L124]
        │     ├─ K=20 候选 = mu + N(0,1.1²)                                [L69-70]
        │     └─ 每个候选: ctrl.motion(c) → (pos,vel)                       [L106]
        │           → cal_channel_fading_gain → R_acc（**固定 30/60/10 池**）
        │           → cal_backhaul_capacity_Mb → _solve_cluster(二分)       [L135]
        │           → J = −(drift_Q+drift_Z) + V·MOS                        [L88-90]
        ├─ 执行: ctrl.motion(best) → adjust_location/adjust_altitude
        │        → uav.pw_prop_W[t] = cal_uav_propulsion_energy(V_xy, V_z)  [cell26 L97]
        ├─ ctrl.solve_delivery(...)（生产交付；**不调用 blocka_allocate**）
        ├─ Q/E 更新 + mos_func
        └─ ctrl.commit_slot / update_virtual_queues

离线链（另一条）:
expA_dual_diagnostics.py
   → JointACCritic.begin_slot → β_i = ψ_Q·Q_i + V·w_k/(λ_k·Σ_{j∈I+} w_j)   [joint_ac L167-180]
   → decide_and_score(mode='directed') K=3=[stay, dir0.25, dir0.5]         [L259-291]
   → _eval_candidate: blocka_allocate + J = ψ_E·(E·P) − β·D                [L242-257]
   → snapshots.npz (152 快照)
expB_*.py → 读 snapshots.npz → 再解 Block A → J = ψ_E·(E0·P) − β·D
```

**映射要点**：用户描述的 `J = ψ_E Σ E_u P_u − Σ β_i D_i` 只存在于 **`joint_ac_controller.py`（离线）**；生产 cell 26 用的是**另一套** J（drift + MOS），且带宽用固定类型池而非 Block A。详见 P0-2。

---

## 2. 已确认的 P0 问题

### P0-1　生产 "Proposed" 违反 UAV 瞬时推进功率硬上限（683 W）

**证据（生产自身输出，非我构造）**
`main_executed.ipynb` cell 26 输出：
```
Constraint check backhaul: max_t [sum(S_u) - sum(R_bh)] = 0.0000 Mb (<=0 = satisfied)
Constraint check motion: max |v| = 30.41 m/s (cap 30), max pw = 745.6 W (cap 683)
```

**代码位置 / 行为**
- `relay_hybrid_controller.py` L106-122 `motion()`：
  `v[:2] = speed_xy * c[:2]/max(1,‖c[:2]‖)`（水平上限 30 m/s）
  `v[2]  = speed_z  * c[2]/max(1,|c[2]|)`（垂直上限 5 m/s）
  两者**独立归一化**，可同时取满。
- `utils_sys.py` L447-499 `cal_uav_propulsion_energy`。
- 我的独立测试 `_audit_tmp/t3_motion.py`：

| 输入 | v_xy | v_z | P |
|---|---|---|---|
| ‖c_xy‖≥1 | 30.0000 | 0.0 | **683.0661 W** |
| ‖c_xy‖≥1 且 c_z=1 | 30.0000 | 5.000 | **745.5724 W** |

  30 m/s 纯水平即已 683.0661 W **> 683**；30+5 同时取满 = 745.57 W，超限 **+62.6 W（+9.2%）**。

**对应的数学模型/设计要求**
`params_sys.py` L197 `PW_MIN, PW_MAX = 82, 683` 被定义为推进功率硬上限；能量虚拟队列
`E_u(t+1)=max(E_u+P_u−P_th,0)` 的前提是 `P_u ≤ PW_MAX` 恒成立。

**为什么是问题**
1. 可行域未被约束：动作参数化允许 `P > 683`，控制器只按 J 排序，从不检查功率。
2. 这是用户明确点名的历史问题（"水平速度 30 m/s 时 P≈683.07 W 略高于 683 W"）——**当前代码未修复**，且实际闭环中还叠加了垂直分量，超限幅度更大。
3. notebook 打印了超限却没有任何断言/裁剪 → 静默违规。

**对现有结果的影响**
生产 Proposed 的 KPI（README: Stationary 2.670 / Centroid 3.313 / Proposed 3.950，BLOCKA 列 5.000）是在该轨迹下产生的；被执行的飞行动作不满足物理硬约束，故这些 KPI 对应的"可行解"不成立。

**推荐修复**：在候选生成阶段把 `P ≤ PW_MAX` 作为硬约束（联合裁剪 ‖v_xy‖、|v_z|，或用 `cal_uav_propulsion_energy` 反解最大允许速度）；并把 `uav_speedxy_max` 降到使纯水平功率 < 683（≈29.99 m/s 以下）。**本轮未修改任何生产代码。**

---

### P0-2　三套互不一致的"目标函数/架构"，离线审计验证的不是线上算法

| # | J 定义 | 实现位置 | 谁在用 |
|---|---|---|---|
| (a) | `J = ψ_E·Σ E_u P_u − Σ β_i D_i` | `joint_ac_controller.py` L18, L253-255；β 见 L167-180 | **仅离线**：expA / expB 全部审计 |
| (b) | `J = −(drift_Q + drift_Z) + V·MOS` | `relay_hybrid_controller.py` L88-90 | **生产** `main.ipynb` cell 26 |
| (c) | `J = ψ_E·(E0·P) − β·D` | `expB_wa30_audit.py` L391；`expB_ablation_guidance.py` L183 | 离线审计 |

**证据**
- `main.ipynb` 中 `blocka` 出现 **0 次**（`grep -c blocka main.ipynb → 0`）；cell 26 第 (2) 步用固定 30/60/10 类型池算 `bw_alloc_Hz_all`，第 (4) 步 `R_access = bw_alloc_Hz_all·log2(1+pTx·g/noise)`，第 (5) 步用 `solve_delivery`（二分），**从不调用 `blocka_allocate`**。
- (b) 的 `relay_lyapunov_controller` 基类里**没有能量虚拟队列 E**，只有回传虚拟队列 Z；而 (a)/(c) 的能量队列 E 由 `JointACCritic.E` 维护。二者不是同一算法。
- 用户描述的 β 公式与 (a) 完全一致：`beta = psi_Q*max(Q,0) + V*w[type]/(lam*w_sum)`，`w_sum=Σ_{active} w[type_j]`，`lam=λ_type(i)`（`joint_ac_controller.py` L172-180）。

**为什么是问题**
离线审计（Binding/Dual K5、ADP2b、oracle 对照、留出验证）全部建立在 (a)/(c) 之上；而线上跑的是 (b) 且带宽规则不同。因此"离线已通过验证"**不能**推出"线上 Proposed 的性能成立"。用户明确要求"如果文档、代码和摘要不一致，必须明确指出差异及其影响"——此处即为最大差异。

**影响**：所有离线增益结论（ADP2b gain 2.0279、Binding≈Dual 等价等）只能说明"在 (a) 的目标与运动预算下"的性质，不能作为线上 KPI 的依据。README 的 "BLOCKA 全面最优" 又来自**第三种**变体（见 P1-2）。

**推荐**：先确定唯一权威模型（补上缺失的模型文档），再让离线审计与线上控制器使用同一 J、同一带宽规则。

---

## 3. 已确认的 P1 问题

### P1-1　用户关联（KMeans）未播种 → 结果不可复现、跨方法不配对

- 代码：`utils_sys.py` L364-393 `kmeans_clustering(loc_users, loc_uavs=None, ...)`，`loc_uavs=None` 分支**不传 `init`、不传 `random_state`**，`n_init=10` 随机初始化。
- 调用点：cell 22 L29、cell 24 L27、cell 26 L60，三处均为 `kmeans_clustering(loc_users)`，**共享全局 numpy RNG 顺序消费**；只有 `HybridCandidateController.__init__`（`relay_lyapunov_controller.py` L70-71）会 `np.random.seed(0)`，且发生在 Stationary/Centroid 之后。
- 我的测试 `_audit_tmp/t4_rng.py`、`t4b_kmeans.py`：
  - 同一位置输入、不同 RNG 状态 → 标签 `n_diff=100/100`；核对质心后确认是**标签置换**（同一划分，索引被换）。
  - `t=4` 时 `cluster k → 最近 UAV` 由 seed0 的 `[1,3,0,2]` 变为 seed7 的 `[2,0,3,1]`。即 **cluster id 与 UAV 初始角点无固定对应**。
- 影响量化 `_audit_tmp/t4c_impact.py`（Stationary 角点几何，t=4/20/100/300/440）：
  `mean log2(1+SNR)` 在 seed 0/7/12345 间摆幅 **0.29–0.45 bit/s/Hz**（≈4–6%）。
- 另：`params_sys.py` L5 `rng = np.random.default_rng()` **未播种**，`gen_heatmap`（`utils_sys.py` L219）的噪声每次不同（`t4_rng.py` 实测两次调用不相等），与 README "全脚本 `np.random.seed(0)` 可复现" 的说法冲突。

**影响**：跨方法比较带入了不受控的 4–6% 关联噪声，量级可与 README 宣称的部分效应（如 Weibull vs Pareto ±0.02）相比；且实验不可逐位复现。

**修复**：`kmeans_clustering` 传入 `random_state=0`（或 `init=loc_uavs` 使 cluster 与 UAV 位置绑定）；`rng = np.random.default_rng(0)`。

---

### P1-2　BLOCKA 变体的 Proposed：用"类型池代理"打分、却执行 Block A（评分/执行不一致）

- 代码：`run_blocka_comparison.py` 文档串 L12-13 —— "cell 26 (Proposed): **candidate scoring keeps the cheap type-pool proxy**, executed allocation + delivery replaced by BLOCKA"；实现见 L118 起（cell 26 的 (4)+(5) 被替换为 `blocka_allocate`，而 (2) 类型池块保留作为打分代理）。
- 即：BLOCKA 变体里，**移动候选的 J 仍按固定类型池带宽计算**，但真正执行的交付用 Block A。
- **影响**：README 头号结论（BLOCKA Proposed 加权 MOS = 5.000）来自"移动为代理目标优化、执行为另一目标"的控制器；该结论混合了两个目标的不一致。

**推荐**：BLOCKA 变体的候选打分也必须用 `blocka_allocate`（成本上升，但这是公平比较的前提）。

---

### P1-3　固定带宽基线 与 BLOCKA 使用**不同的噪声模型** → 对比被混淆

- 基线（cell 26 L111 / `utils_sys.cal_channel_capacity_Mb`）：`R = W·log2(1 + g·pTx/noise_pw_total)`，**噪声功率与 W 无关**。
- Block A（`blocka_optimizer.py` L51-57, L121）：`R = W·log2(1 + G/W)`，`G = g·pTx/(noise_pw_total/W_ref)`，`W_ref=bw_per_user=0.5 MHz` → **噪声功率 = N0·W 随 W 缩放**。
- 两者仅在 `W = W_ref = 0.5 MHz` 处相等（`_audit_tmp/t5_models.py`，SNR=15 dB）：

| W (MHz) | 基线 Mb/s | Block A Mb/s | 比值 |
|---|---|---|---|
| 0.2 | 1.0056 | 1.2646 | 1.258 |
| 0.4 | 2.0111 | 2.1363 | 1.062 |
| **0.5** | 2.5139 | 2.5139 | **1.000** |
| 0.8 | 4.0222 | 3.5008 | 0.870 |
| 1.5 | 7.5417 | 5.2930 | 0.702 |
| 5.0 | 25.1390 | 10.2869 | 0.409 |

- 类型池 Video 份额 = 0.6×40 = 24 MHz，若 ~30 个活跃 Video 用户 → ≈0.8 MHz/用户，落在"Block A 模型更低"的一侧。
- **影响**："BLOCKA vs 类型池/α/到达率比例" 的 MOS 差同时包含（i）联合优化 与（ii）噪声模型改变两个因素，不能归因于前者。`blocka_optimizer` 文档串已说明"above/below the reference the physical model bends the rate curve"，但未讨论该对比的可比性。

---

### P1-4　"Oracle" 不是 Dual-K5 候选集的超集 → "ADP2b ≈ oracle" 的边界被高估

- 数据：`expB_results/wa30_audit/task2_validation_sweep.csv` 中 `dir ∈ {beta, bind}` **不含 dual**（我核对：`dirs: ['beta' 'bind']`）。
- 项目自身已发现该矛盾：`task1_consistency_audit_report.md` L21-23 —— "oracle 网格不是 k5best 超集 … oracle ≤ k5best 在 si=14/23 为假（dual 方向在网格外）"。
- **影响**：`adaptive_step_report.md` 中 "ADP2b 与 11 次调用的 oracle 完全一致" 仅在 beta/bind 网格内成立；对 Dual 基线无上界意义。属**已披露**问题，但作为头号结论引用时须带限定。

---

## 4. 已确认的 P2 问题

### P2-1　求解器回退接受不可行解，且无可行性校验
- `blocka_optimizer.py` L141-169：依次尝试 `CLARABEL → SCS`，凡返回 `optimal/optimal_inaccurate` 即接受，**从不校验返回点是否满足约束**，仅 `np.maximum(W.value,0)`。
- 我的测试 `_audit_tmp/t1c_violation.py`、`t1d_detail.py`（152 快照 × 20/30 MHz）：**si=10（t=23）** CLARABEL 抛 `SolverError` → 回退 SCS → 返回 `optimal`，但
  `sum(W) = 40.004787 MHz > 40`（超 4.79 kHz），且 **user 14 的 `D = 0.057301 Mb` 而 `W = 0`** → 接入约束被违反 0.0573 Mb。
- 项目已披露：`expA_summary.md` `max_access_resid_Mb = 0.057301`，`anomalies.csv` `solver-fallback,23,base -> SCS`。**披露了但未修复**。
- 物理上该用户信道极强（`gamma_acc=603.4 MHz`，W→0 的真实速率上限 `G/ln2≈870 Mb`），故 0.0573 Mb 不造成实际损害；但**代码自身的速率模型/约束**在此点不自洽（`R_access_Mb` 用 `W·log2(1+G/max(W,1e-12))` 在 W=0 处得 0，而 `relu_entr` 形式在 W=0 处也为 0 —— 真实极限 `G/ln2>0`，存在表示层面的跳变）。
- 闭环中 Block A 每槽解 K 次（生产 20 候选 × 501 槽 ≈ 10⁴ 次），按 expA 观测到的 ~1/501 回退率，回退会反复出现。
- **推荐**：接受解前做一次原始约束残差校验（`D ≤ τR_acc`、`ΣD_u ≤ τR_bh_u`、`ΣW ≤ W_U`、`ΣWb ≤ W_A`），失败则收紧容差重解或降级为上一可行解。

### P2-2　快照 `Q` 存的是**更新后**队列 Q(t+1)，不是决策时刻 Q(t)
- `expA_dual_diagnostics.py` L301 更新 `Q`，L448 才 `snap[...] = dict(Q=Q.copy(), ...)` → 存的是 Q(t+1)。
- 我的测试 `_audit_tmp/t1c_violation.py`：`max|Q_stored − (demand − A)| = 40.13`（若为 Q_prev 应为 ~0）→ 确认非决策时刻值。
- expB 通过 `demand = D + dem_slack` 重建需求，规避了该陷阱；但键名 `Q` 具误导性，后续脚本误用会静默出错。

### P2-3　`joint_ac_controller._to_motion` 的边界/垂直处理与执行端不一致
- `_to_motion` L348-350 位置裁剪到 `±BOUNDARY(250)`，而执行端用 `adjust_location`（`mobility_model.py` L30）裁剪到 `±(boundary−1)=249`。
- 测试 `_audit_tmp/t3_motion.py`：起始 x=249 时，**评价位置 x=250.0，执行位置 x=249.0**（差 1 m）。`relay_hybrid_controller.motion` 两端都是 249（一致）。
- 另：`_to_motion` 的垂直分量未做 `max(1,|b_z|)` 归一化（水平做了），仅因调用方把动作裁到 [−1,1] 才安全（b_z=5 时会得 v_z=25 > 5 m/s 上限）。属防御性缺陷。

### P2-4　KPI 统计口径细节
- `main.ipynb` cell 53：`qlen_Mb_avg = Σqlen / active_cnt`，`active_cnt = Σ(active_state)`，而 `active_state=(qlen+traffic>0)` **依赖各方法自身队列** → 分母跨方法不同，`qlen-Mb-avg (active)` 不是严格同口径。（`mos-weighted-avg` 因三方法共享同一 ON/OFF 而可比。）
- 陈旧注释：cell 53 注释写 "(0.35 Web / 0.25 Video / 0.4 Messaging)"，`params_sys.py` L156 实为 `[0.35, 0.30, 0.35]`；`params_sys.py` L74 注释 "50 MHz w/ 100 users" 与 `bw_total_Hz=40 MHz` 不符。
- 功率口径：expB 明确区分 `P_max`（单 UAV 最大）与 `P_sum`（4 架之和）；notebook 打印 `max over UAVs & slots` —— 口径清楚，无问题。

---

## 5. 独立测试中**通过**的项（应予确认，避免误伤）

| 检查 | 方法 | 结果 |
|---|---|---|
| Block A 约束真实可行（D≤Q+A、ΣD_u≤τR_bh_u、ΣW≤W_U、ΣWb≤W_A） | `t1b` 152 快照 × 2 带宽 | **0 违规**（唯一例外见 P2-1 的 SCS 回退） |
| 离线 stay-J 可复现 | `t1b` T2，我独立算 vs `task2_wa30_methods.csv::J0` | `max|diff| = 5.7e-14` |
| 快照 D 可复现 | `t1b`（20 MHz） | `max|diff| = 2.7e-2`（仅 si=10 因求解器不同） |
| 交付子问题不超自身回传容量 | `t1b` | 0 违规 |
| β 公式与用户描述一致 | `joint_ac_controller.py` L167-180 | 一致（ψ_Q=1/queue_norm，V=lyapunov_param，λ=类型速率，Σ 取活跃集） |
| **ν 是否被共享池 ν_pool 污染** | `expB_ablation_guidance.py` L92-95、L123-125 | **未污染**：ν 取自 per-UAV 回传约束 `bh_cons`，池对偶单独记为 `nu_bh_pool` |
| expB 闭环配对性 | `expB_results/stage1/stage1_runs.npz` | 三方案 `cids_hash` **完全相同**（关联配对）；`exec_dev = 0`（评价=执行）；`decomp_err≈1e-13` |
| expB 闭环运动合规 | 同上 | `P ≤ 328.16 < 683`，`vxy ≤ 22.5`，`vz = 0`，`vair ≤ 3.75`，**0 违规** |
| 项目自纠的飞艇映射 | `expB_wa30_audit.py::move_air` L405-410 用 `SPD_XY·ρ` | **确认违规**（7.5–15 m/s > 5 m/s）；项目已作废 "+0.093"，我确认其作废正确 |
| 项目自纠的 hybrid 超功率 | `task3_hybrid/task3b_slow` npz `P_hist` max 745.57 | **确认**（与 P0-1 同源：vxy=30 且 vz=5） |

> 注：离线 expB 审计把 ρ 限制在 {0.25, 0.5, 0.75}（v_xy ≤ 22.5），**因此其"无违规"结论天然覆盖不到生产端**（生产候选为 N(0,1.1²) 无界高斯 + tanh，可达 30+5 m/s）。这正是 P0-1 未被离线审计发现的原因。

---

## 6. 有代码依据但尚未复现的风险

1. **生产 Proposed 的超限槽比例未知**：无逐槽功率日志（KPI xlsx 无功率列）。expB 的 `task3_hybrid` 复现中 2004 个 UAV-槽有 **1214 个** P>683；生产端是否同量级需一次带功率记录的运行确认。
2. **`_eval_candidate` 用指令速度而非实际净位移算 P**（`joint_ac_controller.py` L245-247）：位置被边界裁剪时，实际净位移 < τ·v，故 P 被高估，J 对该候选偏保守。我已在边界个案中确认存在差异（P2-3），但未量化 152 快照上的平均偏差。
3. **KMeans 置换对 MOS 的净影响**：我只量化了接入谱效（0.29–0.45 bit/s/Hz）；对最终加权 MOS 的影响未测（需重跑，本轮未做）。
4. **`run_blocka_comparison.py` 的 cell 26 补丁是否逐条匹配当前 notebook 文本**：补丁基于字符串替换（`patch_once`），notebook 已改动（git status 显示 `main.ipynb` 为 modified），补丁可能失配或静默不生效。未逐条验证。
5. **`relay_hybrid_slow_controller.py` 的用途**（1932 字节，疑为 `n_candidates`/`eval_sigma` 缩减版）：未确认它是否被任何生产/实验路径引用。

## 7. 仅凭现有材料无法判断的问题

1. **哪一套 J 是"官方"模型**：缺 `BLOCKA求解.md` / `actor-critic部分.md`，无法裁定 (a)/(b) 谁是权威。
2. **README 头号数字与产物的对应关系**：git 仅 2 次提交，全部 expB 工作未跟踪（`?? expB_*`），无法把 README 的 2.670/3.313/3.950/5.000 映射到具体运行产物。
3. **`joint_ac_controller`（expA 用）是否计划取代 `relay_hybrid_controller`（生产用）**：无迁移记录。
4. **MOS/队列 KPI 是否在"同一随机种子、同一仿真长度"下比较**：expB 自述"同轨迹非独立种子"（`summary_audit_and_validation.md` L41），跨方法种子一致性未从产物核实。

---

## 8. 下一轮最值得检查的 3–5 个位置（按优先级）

1. **`main.ipynb` cell 26 + `run_blocka_comparison.py` 的 cell 26 补丁**：逐行确认"打分用类型池代理、执行用 Block A"的确切位置与 J 偏差量级（P1-2）；同时确认补丁文本是否仍匹配当前 notebook（风险 4）。
2. **生产端功率超限的规模**：加一次带 `P_hist`（per-slot × per-UAV）记录的 Proposed 运行，统计 P>683 的槽占比与 |v| 分布（P0-1 的影响面）。本轮未跑（遵守"不自动跑 ≥501 槽"）。
3. **`joint_ac_controller._eval_candidate`（L242-257）**：在 152 快照上量化"指令速度 P" vs "实际净位移 P" 的差，确认边界裁剪对 J 排序的影响（风险 2）。
4. **`utils_sys.kmeans_clustering` 与 cell 22/24/26 的调用序列**：用固定 `random_state` 重算三方法 KPI，量化 P1-1 对加权 MOS 的净影响（风险 3）。
5. **`blocka_optimizer.solve_blocka` 的接受判据**：在 501 槽全量上统计 CLARABEL 失败率与 SCS 回退后的真实约束残差（P2-1 的影响面）。

---

## 9. 本轮产物

- 报告：`_audit_tmp/AUDIT_ROUND1.md`（本文件）
- 独立测试脚本（**独立保存，未覆盖任何原实验结果**）：
  - `_audit_tmp/t1_blocka_feasibility.py`、`t1b_feasibility.py`、`t1c_violation.py`、`t1d_detail.py`
  - `_audit_tmp/t3_motion.py`、`t4_rng.py`、`t4b_kmeans.py`、`t4c_impact.py`、`t5_models.py`、`dbg1.py`
- **未修改任何生产代码**（`blocka_optimizer.py` / `relay_hybrid_controller.py` / `main.ipynb` / `params_sys.py` 等均未改动）。

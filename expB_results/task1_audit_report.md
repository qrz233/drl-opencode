# 任务 1 审计：Binding/Util 测试集结果一致性 + 旧 Hybrid MOS=4.9967 配置核实

日期：2026-10-12  
数据源：`expB_results\ablation_guidance\ablation_slots.csv`（121 测试快照）、`git show HEAD:...` notebook/controller、blocka 产物目录。

---

## 1. Binding vs Util：为什么测试集 J 几乎完全一致

### 1.1 权重公式（已验证生效）

| 变体 | 公式 | 校准最优 | 测试集 mean w |
|------|------|----------|---------------|
| binding | `w = a_b · 1[nu > 1e-6]` | a_b=1.0 | 0.517 |
| util | `w = clip((eta − eta0)/1.0, 0, 1)` | a_eta=1.0, eta0=0.5 | 0.909 |
| dual | `w = nu/(0.1 + nu)` | c_ν=0.1 固定 | 0.364 |

- `util w formula match: True`（逐元素精确匹配 clip 公式）
- `binding w formula match: True`（w ∈ {0,1}）
- 非全等：`max|w_b − w_u| ≈ 0.997`，59/121 快照 w 不同。

### 1.2 关键发现：η 与紧绑定指示几乎等价

交叉表（每个 UAV-slot，binding a=1.0 / util a=1.0, η₀=0.5）：

|  | eta < 1 | eta ≥ 1 |
|--|---------|---------|
| bind=0 | 234 | 0 |
| bind=1 | 2 | 248 |

- `eta ≥ 1 ⇒ bind=1`（100%）；`nu > 1e-6 ⇒ bind=1`（100%）；corr(bind, w_util)=1.0
- 当 bind=1：eta mean=3.26 → util 的 w 饱和到 1.0（与 binding 相同）
- 当 bind=0：eta mean=0.91 → util 的 w ≈ 0.81；binding 的 w=0

### 1.3 为什么 J 差仅 ~1e-8

当 w 不同（59 个快照），`J_g25`/`J_g50`/`J_best` 最大差仍仅 **2.6e-8 / 3.2e-8 / 1.9e-8**。

机制：
1. **bind=1 时**：binding 与 util 均 w=1 → 引导方向同为 e_bh，候选相同。
2. **bind=0 时**：binding w=0（纯 e_β），util w≈0.81（偏 e_bh）。但此时 nu≈0（backhaul 不紧），Block A 重解中交付增益对 x/y 方向不敏感 → 引导候选 J 与 β 候选 J 在数值噪声级相同。
3. **argmin 平局**：J 几乎平坦时，best_idx 可不同（binding 偏好 idx4=β0.5、util 偏好 idx3=β0.25），但胜者 J 值仍差 <1e-8；P_sum 可差数百 W（不同 rho → 不同速度），但 dJ = dU − dP_E 抵消。

### 1.4 结论

- **变体代码已正确生效**（w 公式精确匹配、w 分布不同）。
- **测试集结果一致是问题几何的性质，不是缓存/实现 bug**：η≥1 与紧绑定在该数据上几乎重合；非紧时方向引导对 J 不敏感。
- 这支持消融报告的判定：Binding、Util 对 Dual 均“充分”；Dual 的连续 nu 相对二值紧绑定无测试集增益。

### 1.5 best_idx 分布（测试集）

| 变体 | idx0 stay | idx1 β0.25 | idx2 dual0.25 | idx3 β0.5 | idx4 dual0.5 |
|------|-----------|------------|---------------|-----------|--------------|
| binding | 11 | 34 | 8 | 33 | 35 |
| util | 7 | 22 | 7 | 47 | 38 |
| dual | 16 | 32 | 8 | 37 | 28 |

binding vs dual：best_idx 仅 10/121 不同（8.3%）。

---

## 2. 旧 Hybrid MOS=4.9967（README Proposed+BLOCKA）配置核实

### 2.1 产物目录

`simulation\sim-output\relay4uav-...-sfm-wb-blocka, n_slots=501, n_users=100, n_uavs=4, ld=2.4 Mbps, v_user_avg=1, vxy_uav_max=30, uOFF=1, k=10\`

生成脚本：`run_blocka_comparison.py`（patch notebook cell 0/4/6/8/22/24/26 + folder tag）。

### 2.2 控制器参数（HEAD main.ipynb cell 26 源码，已核实）

```python
ctrl = HybridCandidateController(
    ...,
    sigma_decay_slots=n_slots, seed=0,
    n_candidates=20, eval_sigma=1.1   # ← 实际 K=20, σ=1.1
)
```

**与 simulation_parameters.txt 的矛盾已解释**：该 txt 由 cell 50 写出的是 A2C agent 的 `n_decisions=10` 与 `std_var_exploit=0.1`，**不是** Hybrid 控制器的 K/σ。Hybrid 真实参数为 **K=20、eval_sigma=1.1、seed=0**（folder tag `K20-s1.1` 亦印证）。

### 2.3 执行端与评分（run_blocka_comparison cell 26 补丁）

| 环节 | 旧 MOS=5.000 运行 | 当前 HEAD cell 26（已修） |
|------|-------------------|---------------------------|
| 候选评分 | type-pool 30/60/10 代理带宽 + 评价位信道 | 同左（代理） |
| 评价位移 | `uav_loc + c · speed_xy · tau`（快，30c） | 同左 |
| 执行位移 | `uav.x[t-1] + c · slot_len`（慢，c 为速度） | `ctrl.motion` 统一快映射 |
| 交付/分配 | **BLOCKA** 在执行位（NEW_P26） | BLOCKA 在执行位 |
| K/σ/seed | 20 / 1.1 / 0 | 20 / 1.1 / 0 |

### 2.4 随机输入

- SFM 用户移动：`generate_mobility_SFM(..., seed=11)`（cell 4）
- Weibull 到达：`seed=0`（cell 6，复用 ON/OFF xlsx）
- ON/OFF 状态与类型：现有 xlsx（uON=501, uOFF=1）
- 控制器：`seed=0`（`np.random.seed(0)` 于 `RelayLyapunovController.__init__`）
- kmeans：共享同一 np.random 流

### 2.5 KPI 复核（产物 xlsx）

- 加权 MOS = 4.9967（99.5% 用户槽 MOS=5.0 触顶）
- 队列 active mean ≈ 0.117 Mb；信道容量 mean ≈ 4.13
- **功率未记录**于旧 xlsx → 任务 2 需补日志后才能与 Dual K5 逐架功率对比

### 2.6 可比性结论

旧 MOS=5.000 与当前 501 槽实验**不可直接同约束对比**：
1. 评分几何（type-pool 代理 at 快评价位）≠ 执行分配（BLOCKA at 慢执行位）；
2. 评/执速度不一致（30c vs c）；
3. MOS 大量触顶（5.0）掩盖差异；
4. 单 seed、无功率日志。

任务 2 目标：在**严格复现旧策略与参数**的前提下补全轨迹/功率/速度/回传/交付日志，验证能否复现 wMOS≈4.9967 与队列≈0.117，成功后标记为 `Hybrid-lookahead` 历史基线，不修改策略本身。

# 任务 1：数值一致性审计（四项修正）报告

日期：2026-10-14。范围：纯重放既有 CSV/npz，不跑新仿真、不改算法、不改系统参数。
输出：`task1_caliber.csv`（1.1）、`task1_decomp.csv`（1.2）、`task1_airship_audit.csv` +
`task1_motion_compliance.csv`（1.3）、`task1_earlystop.csv`（1.4）。
脚本：`_t1_caliber.py` / `_t1_decomp.py` / `_t1_airship_audit.py` / `_t1_earlystop.py`。
数据：`task2_step_sweep.csv`（40 快照 × 12 行）、`task2_wa30_methods.csv`、
`task2_extended_vs_k5.csv`、`task3_adaptive_replay.csv`、expA `snapshots.npz`、
全部 expB 闭环 npz。

## 1.1 口径统一（K5 / ADP2b / oracle 三口径对账）

**结论：报告里 "1.654 + 0.369 = 2.0229" 的缺口是两套 K5 基线混用，不是算术错误。**

- 恒等式（同一口径内逐快照严格成立，max|resid| = 0）：
  - fix5 口径：gain_fix5 1.6538 + gap_fix5_oracle 0.3741 = gain_oracle 2.0279 ✓
  - k5best 口径：gain_k5best 1.6588 + gap_k5best_oracle 0.3691 = 2.0279 ✓
  - 混用（fix5 增益 + k5best gap）= 2.0229，偏差 0.0050 = mean(k5best − fix5)
- k5best ≠ fix5 的原因：k5best 取 min(k5_bind, k5_dual, k5_k3)（methods 路径，dual 方向
  不在 sweep 网格），k5best − fix5 mean=−0.0050，n<0 的 si=[13, 14, 23]。
- **oracle 网格不是 k5best 的超集**：oracle ≤ fix5 全真、oracle ≤ adp2b 全真、
  oracle ≤ k5best 在 si=14/23 为假（gap −0.0164/−0.0271，dual 方向在网格外）。
  "oracle（扩展网格最优）≤ 一切基线" 的说法必须限定在 sweep 网格内候选（fix5/adp2b）。
- 交叉验证：J0 sweep vs methods 完全相同（diff=0）；k5_bind 两路径差 1.14e-13；
  oracle == task2_extended_vs_k5 的 J_ext（3.6e-15）。
- ADP2b 按冻结规则从 sweep 重算：calls {3:16, 4:17, 5:7} mean 3.775、gain mean 2.0279，
  与 `adaptive_step_report.md` 一致；vs fix5 = 6/33/1（@1e-9）与 5/35/0（@1e-6）、
  vs k5dual = 9/28/3（@1e-9）与 8/30/2 sum_neg=−0.0435（@1e-6）——报告用的两组阈值
  分别复现（fix5 用 1e-9，k5dual 用 1e-6），引用时需注明各自阈值。
- 每快照 oracle 候选 11 个（stay×2 去重）、valid 10（rho=1.0 双向功率无效）；
  solve 合计 9.24s，0.019s/候选。

## 1.2 J 增益分解 dJ = dU − dC_E

**结论：分解恒等式成立（max|resid|=2.1e-13，560 行），报告引用的数字全部复现。**

- ρ=0.75 vs Stay：dC_E mean **+17.83**（报告 +17.8）；dU beta +1.08 / bind +2.33
  （报告 "+1.1~+2.3"）。dJ = −16.75 / −15.50。
- 边际 0.75 vs 0.50：dC_E mean **+13.54**（报告 +13.5）、dU +0.30~+0.73
  （报告 "+0.3~0.7"）→ 大 ρ 增益被能量成本吞没的机制成立。
- 能量曲线非单调：ρ=0.1/0.25 的 dC_E 为负（−0.24/−0.32，悬停 88.07W >
  83.79W@7.5m/s），小步移动反而省电。
- 条件化：扩展网格 ρ*=0.75 的快照 n=11（bind 方向 n=11/11）在该点
  dU=+2.95、dC_E=+0.20、dJ=+2.75 —— 收益成立只在这些快照；
  非 ρ*75 快照 ρ=0.75 的 dJ mean −23.3。全样本 ρ=0.75 负 dJ 占 48/80。

## 1.3 飞艇移动映射与历史结果可引用性

**结论：发现 2 处违规映射（均已标作废）、2 处超功率上限的未引用结果；expA 快照与
全部 K5/Dual/Binding 闭环合规。**

静态清单 18 行（`task1_airship_audit.csv`）：VALID 13、VALID(UAV 侧) 1、
NON-CITABLE 2、CAVEAT 2。

NON-CITABLE：
1. `expB_wa30_audit.py::move_air` L405-410：飞艇步长 `SPD_XY*rho`（7.5–15 m/s >
   SPD_AIR=5.0）→ `task2_wa30_methods.csv::dJ_dual_air,air_tag,air_gain_over_uav`
   及 **"+0.093" 全部作废**。已在 `wa30_diagnostic_report.md` 两处加作废标注
   （`adaptive_step_report.md` L121-124 此前已声明）。
2. `expB_task3_frozen.py::eval_action_dual` L388（`va=SPD_XY*rho*e_a`）→
   `expA_native_slots.csv` 的 J_du_* / dJ_du* / dual_selected / du_* / J_old_betaK3
   列不可引用（`task3_frozen_run.log` L59 "0.270" 同废）。无 md 报告引用这些列。
   同文件 J_hy_* / dJ_hy*（airship fixed）与 expB_unified_speed_slots.csv 合法。

CAVEAT（超 PW_MAX=683，速度本身合法 vxy=30/vz=5 恰在限）：
- `task3_hybrid/task3_seed{0,2}.npz::hybrid_P_hist`（max 745.57，超限槽 1214/1188）
- `task3b_slow/task3b_seed{0,2}.npz::fast_P_hist`（同上）
- 均为 Actor 闭环产物，无任何 md 报告引用其 KPI（ablation 设计文档用的是
  Hybrid-slow，P≤138 合规）。

动态核验（`task1_motion_compliance.csv`，134 个 npz 速度/功率序列）：
- expA 快照轨迹合规（按槽间隔归一）：UAV 槽内位移 ≤15≤30、飞艇 ≤2.5≤5、
  chain/gap 同界；z 恒 150；|xy| UAV 220.5 / air 17.7 ≤250。
- expB 闭环：K5/Dual/Binding 全部序列 ok（vxy≤22.5、vair≤3.75、P≤328.16）；
  唯一违规即上面 8 个 hybrid/fast P 序列。
- 合法 SPD_AIR=5.0 的脚本：stage0/1/2、task2_long、task3_binding_closed、
  scenario_audit；`expB_selective_mobility.py` 的违规分支是死代码（调用点
  `move_air=False` / 默认 `air_fixed=True`，实际运行合法）。
- "+0.056"（20 MHz 飞艇增益）来自 scenario_audit（SPD_AIR=5.0 合法），仍有效，
  但与作废的 30 MHz +0.093 不可比。

## 1.4 早停风险（冻结 ADP2b 规则）

**结论：在 40 快照设计集上早停实际无损（唯一漏损 1.45e-9）；不声称 ADP2b 必不劣于
fix5/k5dual；非单峰真实存在但未造成漏检。**

- 分支分布 {3:16, 4:17, 5:7} 与报告一致。
- **实际漏检**（J_adp2b − J_oracle > 1e-9）：仅 si=85，1.45e-9（@1e-6 为 0）。
  stop@3 分支 16 快照中仅 si=85 有未评估 rho 打破基线（beta@0.5/bind@0.5，
  幅度即 1.45e-9）。
- 方向反转：0.25 败者方向在 ρ=0.5 打破 stop@3 基线 5 次、ρ=0.75 4 次，但
  **从未打败 0.25 胜者方向同 ρ 值**（other_dir_50 mean=+1.197）；真正改变答案的
  仅 si=85。ADP2b 不评估败者方向的大 ρ 在本集上无代价。
- 非单峰（实测、不预设）：dstar 方向内部 argmin 28/40、dother 27/40；
  J(rho) 有升有降 20/40；ρ=0.1（规则从不评估）是全局 argmin 的 4 个快照
  （si=97/104/121/125，均 stop3，漏损 ≤3.4e-10）。
- stop@4 分支 17 快照：规则因 J50≥J25 停，但 J75<J50 的 0 个、J75 打破已评估最优的
  0 个 → 第二次早停条件在本集无漏检。
- 趋势扩展（calls 4–5）贡献 mean 0.790、合计 31.60，落在 7 个快照
  （si=2,13,18,25,32,49,53）——早停的代价由"该停就停"换来的 40 次平均 3.775 调用。
- **不声称必不劣**：vs fix5 胜/平/负 = 6/33/1（1e-9，负=si85）或 5/35/0（1e-6）；
  vs k5dual = 9/28/3（1e-9，负=si14/23/85）或 8/30/2（1e-6，负=si14/23，
  sum_neg=−0.0435）。si=14/23 的负值来自 dual 方向在 sweep 网格外。

## 总口径备忘（引用数字时）

| 数字 | 口径 | 状态 |
|---|---|---|
| 2.0279（ADP2b/oracle 增益 mean） | sweep 网格 + 冻结规则 | 有效 |
| 1.654（fix5）/ 1.659（k5best） | 两套 K5，勿混用 | 有效但不可互换 |
| 0.374 / 0.369（gap） | 分别配 fix5 / k5best | 有效但不可互换 |
| +17.8 / +13.5 / +1.1~2.3（dC_E、dU） | dJ=dU−dC_E 分解 | 有效 |
| 飞艇 30 MHz +0.093 | 违规 SPD_XY*rho 映射 | **作废** |
| 飞艇 20 MHz +0.056 | 合法 SPD_AIR=5.0 | 有效 |
| expA_native_slots dJ_du 等列 | 违规 airship eval | **不可引用** |
| task3_hybrid / task3b fast KPI | P=745.6>683，无 md 引用 | 慎用（未被引用） |
| W/T/L 6/33/1 vs 5/35/0 | eps 1e-9 vs 1e-6 | 引用需注明阈值 |

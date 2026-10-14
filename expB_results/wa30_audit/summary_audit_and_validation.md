# 汇总：数值一致性审计 + ADP2b 独立验证（两任务完成，暂停）

日期：2026-10-14。本轮范围：任务1（四项修正，纯重放不跑新仿真）+ 任务2（冻结规则
独立快照验证，20/30 MHz）。任务3（闭环）**按约定不做**。不调系统参数、不训练
Actor、不实现 SCA/Deep Unfolding、不开发新候选。

详细报告：
- `task1_consistency_audit_report.md`（任务1 四项审计 + 口径备忘表）
- `task2_validation_report.md`（任务2 48 快照 × 2 带宽验证）
- 数据：`task1_caliber.csv` / `task1_decomp.csv` / `task1_airship_audit.csv` /
  `task1_motion_compliance.csv` / `task1_earlystop.csv` /
  `task2_validation_selection.csv` / `task2_validation_sweep.csv` /
  `task2_validation_calibers.csv`
- 脚本：`_t1_caliber.py` `_t1_decomp.py` `_t1_airship_audit.py`
  `_t1_earlystop.py` `_t2_validate.py`（全部在 expB_results\wa30_audit\）

## 任务 1（四项修正）

1. **口径统一**：报告缺口 "1.654+0.369=2.0229" 是 fix5 与 k5best 两套 K5 基线混用
   （差 0.0050）；同口径恒等式逐快照严格成立。**oracle 网格不是 k5best 超集**
   （si=14/23 dual 方向在网格外），"oracle≤一切基线" 须限定 sweep 网格内候选。
   ADP2b 冻结规则重算与 `adaptive_step_report.md` 全对齐（calls {3:16,4:17,5:7}
   mean 3.775，gain 2.0279）。
2. **J 分解**：dJ=dU−dC_E 恒等式 resid 2.1e-13；ρ=0.75 dC_E=+17.83 / 边际 +13.54
   复现报告 +17.8/+13.5；dU +1.1~2.3（vs Stay）、+0.3~0.7（边际）复现；小 ρ 省电
   （悬停 88.07>7.5m/s 83.79）导致能量项非单调。
3. **飞艇审计**：2 处违规映射作废——`expB_wa30_audit.py::move_air`（SPD_XY·ρ →
   **"+0.093" 全部作废**，已在 wa30_diagnostic_report.md 两处标注）、
   `expB_task3_frozen.py::eval_action_dual`（expA_native_slots dJ_du 等列不可引用，
   无 md 引用）。expA 快照与 K5/Dual/Binding 闭环全部合规（134 npz 序列仅 8 个
   hybrid/fast 超 P=683，无 md 引用）。selective_mobility 违规分支是死代码。
   "+0.056"（20 MHz）合法保留。
4. **早停风险**：设计集实际漏检仅 si=85 的 1.45e-9（@1e-6 为 0）；stop@4 的第二次
   早停 0 漏检；非单峰真实存在（内部 argmin 28/40）但未造成代价；rho=0.1 盲区
   4 例全部 ≤3.4e-10。**不声称必不劣**：vs fix5 @1e-9 6/33/1、@1e-6 5/35/0；
   vs k5dual @1e-9 9/28/3、@1e-6 8/30/2（si=14/23 来自 dual 方向在网格外）。

## 任务 2（独立验证，规则零改动）

- 验证集：设计 40 之外 112 快照分三层（mixed/early_cong/stable）各抽 16 = **48**，
  评估前冻结。**同轨迹（slots 4..500）非独立种子，统计仅描述性。**
- 协议：每带宽重解 base（binding 方向各自计算），sweep ρ{0,.1,.25,.5,.75,1.0}×
  {beta,bind}，飞艇固定合法映射，逐候选 5 项合法性判定；ρ=1.0 两带宽 0/48 合法
  （纯功率 683.07>683）。1152 solve / 21s，无失败。
- **主结果（20 / 30 MHz）**：
  - gain K5→ADP2b→oracle：1.2508→1.4140→1.4140 ／ 1.4172→1.6167→1.6204
  - ADP2b vs K5 @1e-6：**10/38/0 ／ 4/44/0（零负例）**；@1e-9 唯一负例为
    si=117@20 的平-J 数值噪声（2.7e-8）
  - ADP2b==oracle：两带宽各 47/48
  - calls mean：4.23 ／ 3.96；stop@3 漏检 0 真实（20 MHz 1 例为噪声）
  - K5→oracle gap：0.1633 ／ 0.2032（集中在 mixed 层）
  - 配对带宽差（同 48 快照）：ADP2b mean +0.2026（30 高于 20）
- **唯一真实盲区**：si=35@30（mixed）漏 0.180——d\*=beta 在 0.25 胜出后，败者
  方向 bind 在 ρ=0.75 反超（方向反转）。设计集 0 次、留出集 1 次；按冻结约定
  **不改规则**，记录为已知上限 ~0.18/快照的盲区。
- 分层：增益集中在 mixed（双侧瓶颈），stable 层三口径全平。

## 结论

1. 任务1 四项修正完成：口径缺口解释并消除、分解复现、违规数据全部标废、早停规则
   在设计集无实质漏检。
2. ADP2b 冻结规则通过独立留出验证：两带宽均不劣于固定 K5（@1e-6 零负例），
   接近扩展 oracle，调用均 3.96-4.23 次。
3. 已知且接受的限制：同轨迹非独立、oracle 非 k5best 超集、方向反转盲区（1 例 0.18）、
   +0.093 与 expA_native dJ_du 列永久不可引用。

**状态：任务 1+2 完成，任务 3 未启动（按约定），此处暂停待指示。**

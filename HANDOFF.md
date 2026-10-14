# HANDOFF — drl-opencode 项目交接文档

> **给下一个 agent**：读完本文件即可接手。按 §2 的顺序读原始文档；
> §7 是"未完成/待批准"清单，**动手前先向用户确认**。
> 更新日期：2026-10-14（contpos 阶段三任务全部完成）。

---

## 1. 环境与运行约定（必须遵守）

- **解释器**：`& "d:\Users\23175\anaconda3\envs\uav-drl\python.exe" -X utf8 script.py`
  （conda env `uav-drl`；`-X utf8` 必带，控制台是 GBK，中文会炸）。
- **导入前置**：任何脚本 `import params_sys / utils_sys` 前必须
  `os.chdir(r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11')`
  （params_sys 在 import 时创建相对目录，不 chdir 会 mkdir 失败）。
- **PowerShell**：无 `head`；管道用 `Select-Object -First`；并行命令用多工具调用，
  不用 `&&`（PS5.1 不支持，用 `; if ($?) { }`）。
- **长输出**：脚本内 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')`。
- **复用机制**：诊断脚本通过 exec 复用 `_grid_residual.py` 的前缀（约 185 行，
  含 `solve_full/gammas/beta_dirs/unit/SNAP/load_snap/evalJ`）：
  ```python
  SRC = open(r'...\expB_results\wa30_audit\_grid_residual.py', encoding='utf-8').read()
  cut = SRC.index('rows = []')
  ns = {}; exec(compile(SRC[:cut], '_grid_prefix', 'exec'), ns)
  ```
  该前缀会 print 一堆仿真横幅，属正常。
- **写文件**：write 工具写超长 JSON 会失败 → 分块；大 CSV 由脚本 `to_csv` 生成。

## 2. 必读文档（按顺序）

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | `expB_results\contpos\contpos_model.md` | **任务1**：连续位置子问题数学模型（可行域/梯度/对偶梯度/正则性/非凸性）——后续一切连续优化工作的理论依据 |
| 2 | `expB_results\contpos\contpos_report.md` | **任务2**：32 实例诊断结果与 6 项重点输出 |
| 3 | `expB_results\contpos\contpos_solver_advice.md` | **任务3**：求解器建议 + §5 待批准的后续验证 |
| 4 | `expB_results\wa30_audit\summary_audit_and_validation.md` | 前一阶段（审计+ADP2b 验证）总结 |
| 5 | `expB_results\wa30_audit\adaptive_step_report.md` | **ADP2b 冻结规则唯一出处**（勿改） |

## 3. 项目状态：两个阶段均已完成

### 阶段 A（前）：一致性审计 + ADP2b 独立验证 — 完成
- 任务1 四项审计（口径/分解/飞艇/早停）→ `task1_consistency_audit_report.md` + 4 CSV。
- 任务2：48 快照（冻结 selection）验证 **ADP2b vs Binding K5 vs ρ网格 oracle**，
  20/30 MHz → `task2_validation_report.md` + 3 CSV。结论：ADP2b vs K5 =
  10/38/0（20M）与 4/44/0（30M）零负例；oracle 漏检仅 si=35@30 的 +0.180
  （方向反转真盲区）与 si=117@20 的 2.7e-8（数值噪声）。
- **ADP2b 规则已冻结**（stay→β@0.25→bind@0.25→d\*@0.5→d\*@0.75，EPS=1e-9、
  tie=1e-6），用户明确：不再改早停、暂缓闭环。

### 阶段 B（本）：连续位置优化研究 — 三任务完成
1. **模型**（`contpos_model.md`）：J(q)=ψ_E·E@P(‖Δ‖/τ) − U*(γ(q))，min，
   可行域=盒∩盘（**凸集**，R_MAX=v_pw·τ=29.998922，功率严格强于速度约束30）；
   pure-LoS ⇒ g=P_ref/(r²+h²) 与回传 g=P_ref/‖q−a‖² 的解析梯度；
   包络定理对偶梯度闭式（1 解同时得 J 与 ∇J）；**不宣称凸、不称 SCA**。
2. **诊断**（`contpos_report.md`）：16 冻结快照 ×{20,30} MHz，
   A=K5(5解)/B=2维(λ,ρ)连续(~126解)/C=8维SLSQP+对偶梯度+3初值(~125解)，
   9268 解 0 失败。结果：**B−A = +0.557/+0.915（13/16、12/16 胜，零负）；
   C−B = +0.032/+0.214；89%/94% 增益集中在 mixed 层（双池同绑）；
   B、C 打破前阶段 ρ 网格 oracle 13/13、12/12（最大 +5.02）；
   对偶梯度 vs FD：44/44 非平凡点 cos≥0.99、352/352 分量同号。**
3. **建议**（`contpos_solver_advice.md`）：对偶梯度采用；SLSQP+投影+多初值为主力
   （4 项工程护栏）；**SCA 不建议开发**；分层预算 T0=K5/T1=B/T2=C 门控在池绑定态。

## 4. 文件地图

```
drl-opencode\
├─ HANDOFF.md                       ← 本文件
├─ expB_results\contpos\            ← 阶段 B 产出（当前活跃）
│  ├─ contpos_model.md / contpos_report.md / contpos_solver_advice.md
│  ├─ contpos_selection.csv         16 快照（冻结于评估前）
│  ├─ contpos_results.csv           32 行/68 列（每 bw×si）
│  ├─ contpos_cstarts.csv           C 的 96 个初值行
│  ├─ contpos_gradcheck.csv         FD vs 对偶 512 行
│  └─ _contpos_diag.py              主脚本（--smoke 可跑 1 实例）
├─ expB_results\wa30_audit\         ← 阶段 A 产出 + 可复用机械
│  ├─ _grid_residual.py             ** exec 前缀源（solve_full/gammas/SNAP）**
│  ├─ _t1_*.py / _t2_validate.py    阶段 A 脚本（exec 复用模式范例）
│  ├─ task1_*.csv / task2_validation_*.csv / task2_step_sweep.csv
│  ├─ task1_consistency_audit_report.md / task2_validation_report.md
│  ├─ summary_audit_and_validation.md / adaptive_step_report.md / wa30_diagnostic_report.md
├─ expB_results\scenario_audit\task1_audit.csv      20MHz 全量 152 行（分层用）
├─ expB_results\wa30_audit\task1_wa30_stats.csv     30MHz 全量 152 行（分层用）
├─ expA_results\snapshots.npz                       152 快照单轨迹（唯一快照源）
└─ uav-bs-placement-drl-modified-v13b-K20-s11\      仿真工程（勿随意改）
   ├─ blocka_optimizer.py   P_R + Γ标定（solve_blocka 无对偶；对偶在 _grid_residual.solve_full）
   ├─ utils_sys.py          cal_channel_fading_gain(L15) / cal_uav_propulsion_energy(L447)
   ├─ channel_model.py      pure LoS（a=0 ⇒ P_LoS≡1, 指数2.0, fading 1.0）
   └─ params_sys.py         PW_MAX=683(仅标度) PW_MIN=82 boundary=250 τ=1 z∈[100,200]
```

## 5. 冻结项与硬规则（违反=返工）

1. **ADP2b 规则冻结**（`adaptive_step_report.md`）：不改早停、不改 EPS/tie。
2. **本阶段指令**（用户原话约束）：不跑闭环、不引入 Actor/Deep Unfolding、
   **不修改生产算法**、不做新长时域仿真。
3. **表述红线**：不宣称位置问题为凸；**投影梯度≠SCA**，不得称 SCA；
   局部搜索不得称"全局最优/严格 oracle"（oracle 仅指前阶段**合法 ρ 网格**候选集）。
4. **Binding K5 = 安全回退**：任何方案最终候选 = min 合法{K5, 连续}，
   保证不更差、不非法。
5. **统计口径**：152 快照全部来自**单条轨迹**（expA snapshots.npz）——
   所有统计只作**描述性**解读，非独立种子推断；同轨迹快照不得当独立样本做检验。
6. 飞艇与 UAV 高度**固定**（先水平 8 维）；功率用**实际净位移**；
   合法性逐候选判定（status/v≤30/P≤683/|xy|≤250/z∈[100,200]）。

## 6. 已核验的数值事实（可直接引用）

- **P(v)**（Zeng Eq.13，无内部钳制）：P(0)=88.072，argmin=81.862@v=5.74，
  P(30)=683.0661；P''(0)=6P₀/U_tip²−P_i/(2v₀²)=**−0.9121**（P₀=56.7190,
  P_i=31.3534，**不是**注释里的 79.856/88.628）；P''<0 恰在 [0,2.96] 连续段；
  P'(0)=0（精确）；{P≤683}=[0, 29.998922]（brentq）。
- 复合 P(‖Δ‖/τ)：Δ=0 处 C² 非 C³，梯度延拓为 0；**能量项在悬停点局部严格凹**
  （非凸性的直接见证）。
- Γ：γ_acc=c_acc·g，c_acc=2.5e11；γ_bh=c_bh·g_bh，c_bh=1.58114e12（MHz 口径）。
- 对偶梯度实现：`_contpos_diag.py::dual_grad`（含 `Pprime` 自检）；
  FD 抽查 h=0.5，16 解/点。
- 求解：CLARABEL→SCS，~0.02 s/解；`optimal_inaccurate` 按有效解对待
  （阶段 A 1152 解中 2 次；阶段 C 深位移区 177 次且慢 ~45×，最慢单例 444.6 s）。
- SLSQP 行为：96 初值中 11 个 success=0（maxiter60 未达 ftol=1e-9，因活跃集折点）、
  12 个终点越盘 ~ε → **必须投影回 R_MAX**（投影后 0 失败）。
- 多初值：最优初值 k5 12 / rand 11 / stay 9 → **单热启动不普适**。

## 7. 当前状态与下一步（接手点）

- **todowrite 全部完成**；无 in-progress 工作；无被 block 的事项。
- 下一步候选（出自 `contpos_solver_advice.md` §5，**均需用户批准后再做**）：
  1. T1 瘦身版 B（5×5 网格+NM ≈ 60 解）在冻结 48 快照上对照现版 B（~126 解）。
  3. T2 门控（base 解 `pool_acc & pool_bh` 或 ν>0 触发 C）的收益损失评估。
  3. 若集成：先以**离线上界诊断**方式接入（不进闭环），保留 FD 抽查与逐候选合法性。
- 若用户要"继续修改优化生产算法/跑闭环"——这**超出当前授权**，
  需要用户显式重新授权（此前用户明确说 ADP2b 冻结、闭环暂缓）。

## 8. 常见坑速查

- `solve_blocka()`（blocka_optimizer.py）**不返回对偶**；对偶只在
  `_grid_residual.py::solve_full`（lam_acc/nu_bh/nu_acc_pool/nu_bh_pool，
  abs/max(0,·) 口径）。勿混用。
- 前缀 exec 里 `DIRS` 是 8 方向罗盘数组，会与你自己的 `DIRS=('beta','bind')`
  冲突——你的定义必须写在 exec **之后**（阶段 A/B 脚本皆如此）。
- `J0` 量级跨实例差异大（−503 ~ +0.06），跨实例均值未加权；比较请看逐实例表。
- `rho=1.0` 必然功率违规（P=683.07>683），故 oracle 实际在 ρ≤0.75 网格取值。
- 分层定义：mixed=任一带宽双池同绑（前阶段冻结）；early_cong=非mixed且
  qlen≥中位数 0.0301；stable=非mixed且<中位数。与"本带宽池绑定"是两回事，
  引用时区分 `stratum` 列与当带宽 `pool_*` 列。
- 前阶段 `wa30_diagnostic_report.md` 中 "+0.093" 已作废标注
  （expA `move_air` 违反 SPD_AIR=5.0），勿引用。

## 9. 克隆与改名指引（将本仓库克隆为 `drl-workbuddy` 时）

前置事实：仓库 remote = `https://github.com/qrz233/drl-opencode.git`（main 分支，
~58.5 MiB/266 文件）；约 **94 处硬编码** `D:\Learn\work\drl-opencode`
（~40 个 .py + 本文件与报告）；sim-output 下有 >260 字符长路径。

```powershell
# 1) 克隆（core.longpaths 必带，否则 sim-output 长路径报 Filename too long）
git clone -c core.longpaths=true https://github.com/qrz233/drl-opencode.git D:\Learn\work\drl-workbuddy
cd D:\Learn\work\drl-workbuddy
git config core.longpaths true

# 2) 批量替换硬编码路径（.py/.md，UTF-8 无 BOM；只改文本，不碰 .npz/.csv）
Get-ChildItem -Recurse -Include *.py,*.md -File |
  ForEach-Object {
    $c = [IO.File]::ReadAllText($_.FullName)
    $n = $c -replace 'D:\\Learn\\work\\drl-opencode', 'D:\Learn\work\drl-workbuddy'
    if ($n -ne $c) { [IO.File]::WriteAllText($_.FullName, $n, (New-Object Text.UTF8Encoding($false))) }
  }

# 3) 冒烟验证（~10 s；路径没替换对会立即报错）
& "d:\Users\23175\anaconda3\envs\uav-drl\python.exe" -X utf8 expB_results\contpos\_contpos_diag.py --smoke
```

- **若 GitHub 仓库名也改**：Settings→Rename（旧 URL 自动跳转），
  然后 `git remote set-url origin https://github.com/qrz233/drl-workbuddy.git`。
- 只改本地文件夹名则跳过上一条；替换完成后 `git status` 应显示 ~40 个 .py
  与本文件被修改，属预期，由新仓库自行提交。
- 若克隆前源仓库未提交新文件（如本 HANDOFF），先在源仓库提交推送，否则克隆拿不到。

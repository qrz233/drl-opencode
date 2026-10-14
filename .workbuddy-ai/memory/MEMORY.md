# 项目长期备忘（drl-opencode / 飞艇辅助多 UAV 通信 + Lyapunov 优化）

## 环境与运行约定

- Python 环境：conda env `uav-drl` → `D:\Users\23175\anaconda3\envs\uav-drl\python.exe`
  （含 cvxpy 1.7.5 / sklearn / torch-cpu / TF）。托管 python 3.13 **无 numpy**，不要用。
- 运行任何脚本前先 `cd uav-bs-placement-drl-modified-v13b-K20-s11`：`params_sys.py` 在
  导入时会用相对路径创建 `simulation/sim-output/<folder tag>`，cwd 不对会 `FileNotFoundError`。
- 主仿真：`main.ipynb`（55 cells；cell 22=Stationary, 24=Centroid, 26=Proposed）。
- 对比实验靠 `run_*_comparison.py` 做"备份→patch cell→执行整本 notebook→还原"。

## 关键架构事实（审计后确认）

- **存在三套 J**：
  - 生产 `relay_hybrid_controller.py` + `main.ipynb` cell26：`J=−(drift_Q+drift_Z)+V·MOS`，
    无能量虚拟队列；带宽用固定 30/60/10 类型池，**不调用 blocka**。
  - 离线 `joint_ac_controller.py`：`J=ψ_E·Σ E_u P_u − Σ β_i D_i`，
    `β_i = ψ_Q·Q_i + V·w_k/(λ_k·Σ_{j∈I+} w_j)`（expA/expB 全部审计基于此）。
  - expB 离线脚本：`J=ψ_E·(E0·P) − β·D`（与上一行同式）。
- 152 冻结快照在 `expA_results/snapshots.npz`；注意其 `Q` 键存的是 **Q(t+1)**，
  需求须用 `demand = D + dem_slack` 重建。
- `expA_results/base_diagnostics.npz` 含 per-slot 对偶/残差（`acc_resid` 等）。

## 已知未修复缺陷（详见 `_audit_tmp/AUDIT_ROUND1.md`）

1. 生产 Proposed 功率超硬上限（打印 745.6 W > PW_MAX=683）——`relay_hybrid_controller.motion`
   水平/垂直独立取满。
2. `utils_sys.kmeans_clustering` 未播种 → 用户关联不可复现、跨方法不配对。
3. 固定带宽基线 vs Block A 使用不同噪声模型（仅 W=0.5 MHz 相等）。
4. `solve_blocka` 接受 SCS 回退的不可行 `optimal`，无可行性校验。

## 审计约定

- 审计/临时脚本统一放仓库根 `_audit_tmp/`；不改生产代码，不覆盖 `expA_results` / `expB_results`。

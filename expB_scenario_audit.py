# -*- coding: utf-8
"""Scenario-plausibility diagnostics (offline only, no closed loop, no Actor).

Task 1: system parameter audit + constraint bottleneck statistics on 152 snapshots
Task 2: backhaul bandwidth sensitivity W_A in {20,30,40,60} MHz on representative states
Task 3: airship-move marginal value under G0/G1/G2 geometry

All solves use an extended Block A that records duals for:
  per-user access rate, demand cap, per-UAV backhaul, access pool, backhaul pool.

Outputs -> expB_results\\scenario_audit
"""
import os, sys, time, json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
OUT = r'D:\Learn\work\drl-opencode\expB_results\scenario_audit'
os.makedirs(OUT, exist_ok=True)
EXPA = r'D:\Learn\work\drl-opencode\expA_results'

from params_sys import *
from utils_sys import cal_channel_fading_gain
import blocka_optimizer as B

MHZ, LN2 = B.MHZ, B.LN2
TAU = float(slot_len)
W_U0 = bw_total_Hz / MHZ          # 40 MHz access pool
W_A0 = bw_bh_total_Hz / MHZ       # 20 MHz backhaul pool
EPS_ACT = 1e-6


# ============== extended Block A with full dual extraction ==================
def solve_blocka_full(gamma_acc_MHz, gamma_bh_MHz, demand_Mb, assoc, beta,
                      W_U_MHz=W_U0, W_A_MHz=W_A0, tau=TAU, solver='CLARABEL'):
    n_users = len(gamma_acc_MHz)
    n_uavs = len(gamma_bh_MHz)
    out = dict(D_Mb=np.zeros(n_users), W_ui_Hz=np.zeros(n_users),
               W_ua_Hz=np.zeros(n_uavs), R_access_Mb=np.zeros(n_uavs * 0 + n_users),
               R_bh_Mb=np.zeros(n_uavs), S_u=np.zeros(n_uavs),
               status='trivial-no-demand', U=0.0,
               nu_bh=np.zeros(n_uavs), nu_bh_pool=0.0, nu_acc_pool=0.0,
               lam_acc=np.zeros(n_users), lam_dem=np.zeros(n_users),
               n_active=0, solver_used=None)
    active = (demand_Mb > 0) & (assoc >= 0) & (beta > 0)
    if not np.any(active):
        return out
    idx = np.where(active)[0]
    aidx = np.unique(assoc[idx])
    g_acc = gamma_acc_MHz[idx]
    dem = demand_Mb[idx]
    b = beta[idx]
    m = len(aidx)
    W = cp.Variable(len(idx), nonneg=True)
    D = cp.Variable(len(idx), nonneg=True)
    Wb = cp.Variable(m, nonneg=True)
    g_bh = gamma_bh_MHz[aidx]
    rate_acc = -cp.rel_entr(W, W + g_acc) / LN2
    rate_bh = -cp.rel_entr(Wb, Wb + g_bh) / LN2
    bidx = np.searchsorted(aidx, assoc[idx])
    c_acc_rate = D <= tau * rate_acc
    c_dem = D <= dem
    c_pool_acc = cp.sum(W) <= W_U_MHz
    c_pool_bh = cp.sum(Wb) <= W_A_MHz
    bh_cons = [cp.sum(D[bidx == u]) <= tau * rate_bh[u] for u in range(m)]
    constraints = [c_acc_rate, c_dem, c_pool_acc, c_pool_bh] + bh_cons
    prob = cp.Problem(cp.Maximize(b @ D), constraints)
    status, used = None, None
    for sname in [solver, 'CLARABEL', 'SCS']:
        if sname is None:
            continue
        try:
            prob.solve(solver=sname, warm_start=True)
            status = prob.status
            if status in ('optimal', 'optimal_inaccurate'):
                used = sname
                break
        except Exception:
            status = 'solver-error'
    if status not in ('optimal', 'optimal_inaccurate'):
        out['status'] = str(status)
        return out
    W_sol = np.maximum(W.value, 0.0)
    D_sol = np.maximum(D.value, 0.0)
    Wb_sol = np.maximum(Wb.value, 0.0)
    out['D_Mb'][idx] = D_sol
    out['W_ui_Hz'][idx] = W_sol * MHZ
    out['W_ua_Hz'][aidx] = Wb_sol * MHZ
    out['R_access_Mb'][idx] = tau * W_sol * np.log2(1.0 + g_acc / np.maximum(W_sol, 1e-12))
    out['R_bh_Mb'][aidx] = tau * Wb_sol * np.log2(1.0 + g_bh / np.maximum(Wb_sol, 1e-12))
    for u in range(m):
        out['S_u'][aidx[u]] = np.sum(D_sol[bidx == u])
    out['status'] = str(prob.status)
    out['U'] = float(prob.value)
    out['n_active'] = int(len(idx))
    out['solver_used'] = used
    # duals
    def dual_of(c):
        try:
            d = c.dual_value
            if d is None:
                return 0.0
            return float(np.max(np.abs(np.atleast_1d(np.asarray(d, dtype=float)))))
        except Exception:
            return np.nan
    la = c_acc_rate.dual_value
    out['lam_acc'][idx] = np.abs(np.asarray(la, dtype=float)) if la is not None else 0.0
    ld_ = c_dem.dual_value
    out['lam_dem'][idx] = np.abs(np.asarray(ld_, dtype=float)) if ld_ is not None else 0.0
    out['nu_acc_pool'] = dual_of(c_pool_acc)
    out['nu_bh_pool'] = dual_of(c_pool_bh)
    nu = [float(c.dual_value) if c.dual_value is not None else 0.0 for c in bh_cons]
    out['nu_bh'][aidx] = np.maximum(np.array(nu), 0.0)
    return out


def channel_gammas(user_x, user_y, cids, uav_locs, airship_loc):
    g_acc = np.zeros(len(cids))
    for u in np.unique(cids):
        ids_u = np.where(cids == u)[0]
        g_acc[ids_u] = cal_channel_fading_gain(user_x[ids_u], user_y[ids_u],
                                               uav_locs[u, 0], uav_locs[u, 1], uav_locs[u, 2])
    gamma_bh, _ = B.cal_gamma_backhaul_MHz(uav_locs, airship_loc)
    return B.cal_gamma_access_MHz(g_acc), gamma_bh


def load_snap():
    z = np.load(os.path.join(EXPA, 'snapshots.npz'))
    n = len(z['slots'])
    out = []
    for si in range(n):
        demand = np.maximum(z['D'][si] + z['dem_slack'][si], 0.0)
        A = z['A'][si]
        out.append(dict(
            si=si, t=int(z['slots'][si]),
            beta=z['beta'][si].copy(), types=z['types'][si].copy(),
            cids=z['cids'][si].copy(), locs0=z['locs_dec'][si].copy(),
            air0=z['air_dec'][si].copy(), E0=z['E_dec'][si].copy(),
            A=A.copy(), demand=demand, D_base=z['D'][si].copy(),
            ux=z['user_x'][si].copy(), uy=z['user_y'][si].copy(),
            nu_snap=z['nu'][si].copy(), eta_snap=z['eta'][si].copy(),
            qlen=float(z['qlen'][si]), status=str(z['status'][si]),
            Q_prev=np.maximum(demand - np.maximum(A, 0.0), 0.0),
        ))
    return out


SNAP = load_snap()
print(f'loaded {len(SNAP)} snapshots', flush=True)


# =========================== TASK 1 ==========================================
def snap_stats(sn, res, W_A=W_A0, tag='base'):
    idx = res['n_active']
    active = (sn['demand'] > 0) & (sn['cids'] >= 0) & (sn['beta'] > 0)
    # constraint activity
    sum_W = float(res['W_ui_Hz'].sum()) / MHZ
    sum_Wb = float(res['W_ua_Hz'].sum()) / MHZ
    # per-UAV backhaul binding: S_u vs R_bh
    served = res['S_u'] > 1e-9
    bh_slack = res['R_bh_Mb'] - res['S_u']
    bh_bind = served & (bh_slack <= 1e-3 * np.maximum(res['R_bh_Mb'], 1e-9))
    # demand cap: users with D >= demand - 1e-6
    dem_cap = active & (res['D_Mb'] >= sn['demand'] - 1e-6)
    # access rate binding: D >= tau*R_acc - 1e-6
    acc_bind = active & (res['D_Mb'] >= res['R_access_Mb'] - 1e-6)
    # backhaul capacity vs data to serve
    Rbh_sum = float(res['R_bh_Mb'].sum())
    dem_sum = float(sn['demand'][active].sum()) if active.any() else 0.0
    D_sum = float(res['D_Mb'].sum())
    return dict(
        si=sn['si'], t=sn['t'], tag=tag, W_A=W_A, W_U=W_U0,
        n_active=int(active.sum()),
        sum_W=sum_W, sum_Wb=sum_Wb,
        pool_acc_util=sum_W / W_U0, pool_bh_util=sum_Wb / W_A,
        pool_acc_bind=int(sum_W >= W_U0 - 1e-4),
        pool_bh_bind=int(sum_Wb >= W_A - 1e-4),
        n_uav_served=int(served.sum()),
        n_bh_bind=int(bh_bind.sum()),
        bh_bind_rate=float(bh_bind.sum() / max(served.sum(), 1)),
        bh_slack_min=float(bh_slack[served].min()) if served.any() else np.nan,
        n_dem_cap=int(dem_cap.sum()),
        dem_cap_rate=float(dem_cap.sum() / max(active.sum(), 1)),
        n_acc_bind=int(acc_bind.sum()),
        acc_bind_rate=float(acc_bind.sum() / max(active.sum(), 1)),
        lam_acc_pos=float((res['lam_acc'][active] > EPS_ACT).mean()) if active.any() else 0.0,
        lam_dem_pos=float((res['lam_dem'][active] > EPS_ACT).mean()) if active.any() else 0.0,
        nu_bh_pos=float((res['nu_bh'] > EPS_ACT).mean()),
        nu_bh_pool=res['nu_bh_pool'], nu_acc_pool=res['nu_acc_pool'],
        Rbh_sum=Rbh_sum, dem_sum=dem_sum, D_sum=D_sum,
        bh_vs_dem=Rbh_sum / max(dem_sum, 1e-9),
        demand_sat=D_sum / max(dem_sum, 1e-9),
        qlen=sn['qlen'], status=res['status'],
    )


print('\n===== Task 1: constraint bottleneck audit (base W_A=20, W_U=40) =====', flush=True)
rows1 = []
t_wall = time.time()
for sn in SNAP:
    g_acc, g_bh = channel_gammas(sn['ux'], sn['uy'], sn['cids'], sn['locs0'], sn['air0'])
    res = solve_blocka_full(g_acc, g_bh, sn['demand'], sn['cids'], sn['beta'])
    rows1.append(snap_stats(sn, res))
df1 = pd.DataFrame(rows1)
df1.to_csv(os.path.join(OUT, 'task1_audit.csv'), index=False)
print(f'task1 solved {len(SNAP)} in {time.time()-t_wall:.0f}s', flush=True)

def q(x, *qs):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.percentile(x, qs) if len(qs) > 1 else (x.mean() if len(x) else np.nan)

early = df1[df1['t'] <= 120]
late = df1[df1['t'] > 120]
for name, sub in (('ALL', df1), ('EARLY t<=120', early), ('STABLE t>120', late)):
    print(f'\n--- {name} (n={len(sub)}) ---', flush=True)
    print(f"  access pool util: mean={sub['pool_acc_util'].mean():.3f} "
          f"bind_rate={sub['pool_acc_bind'].mean():.3f}  nu_acc_pool>0={float((sub['nu_acc_pool']>EPS_ACT).mean()):.3f}", flush=True)
    print(f"  backhaul pool util: mean={sub['pool_bh_util'].mean():.3f} "
          f"bind_rate={sub['pool_bh_bind'].mean():.3f}  nu_bh_pool>0={float((sub['nu_bh_pool']>EPS_ACT).mean()):.3f}", flush=True)
    print(f"  per-UAV bh bind rate: mean={sub['bh_bind_rate'].mean():.3f}  nu_bh>0 rate={sub['nu_bh_pos'].mean():.3f}", flush=True)
    print(f"  demand-cap fraction: mean={sub['dem_cap_rate'].mean():.3f}  lam_dem>0={sub['lam_dem_pos'].mean():.3f}", flush=True)
    print(f"  access-rate binding: mean={sub['acc_bind_rate'].mean():.3f}  lam_acc>0={sub['lam_acc_pos'].mean():.3f}", flush=True)
    print(f"  Rbh/demand ratio: mean={sub['bh_vs_dem'].mean():.3f} median={sub['bh_vs_dem'].median():.3f}", flush=True)
    print(f"  demand satisfaction: mean={sub['demand_sat'].mean():.3f} median={sub['demand_sat'].median():.3f}", flush=True)
    print(f"  qlen: mean={sub['qlen'].mean():.3f}  D_sum mean={sub['D_sum'].mean():.2f}  Rbh_sum mean={sub['Rbh_sum'].mean():.2f}", flush=True)

# verdict helper: dominant bottleneck per snapshot by dual-weighted scarcity
# heuristic: which constraint set has positive duals most often
print('\n--- dominant bottleneck heuristic (dual presence) ---', flush=True)
dom = []
for _, r in df1.iterrows():
    scores = dict(acc_pool=r['nu_acc_pool'] > EPS_ACT,
                  bh_pool=r['nu_bh_pool'] > EPS_ACT,
                  dem_cap=r['lam_dem_pos'] > 0.5,
                  acc_rate=r['lam_acc_pos'] > 0.5)
    if not any(scores.values()):
        dom.append('slack')
    else:
        dom.append('+'.join(k for k, v in scores.items() if v))
df1['dom'] = dom
print(df1['dom'].value_counts().to_dict(), flush=True)


# =========================== TASK 2 ==========================================
# pick 40 representative snapshots: stratify by qlen quartile + binding rate
rng = np.random.default_rng(0)
df1['qbin'] = pd.qcut(df1['qlen'], 4, labels=False, duplicates='drop')
pick = df1.groupby('qbin', group_keys=False).apply(
    lambda g: g.sample(n=min(10, len(g)), random_state=0))
pick_idx = pick['si'].tolist()
print(f'\n===== Task 2: W_A sensitivity on {len(pick_idx)} snapshots =====', flush=True)

WA_GRID = (20.0, 30.0, 40.0, 60.0)
rows2 = []
t_wall = time.time()
for sn in SNAP:
    if sn['si'] not in pick_idx:
        continue
    g_acc, g_bh = channel_gammas(sn['ux'], sn['uy'], sn['cids'], sn['locs0'], sn['air0'])
    for WA in WA_GRID:
        res = solve_blocka_full(g_acc, g_bh, sn['demand'], sn['cids'], sn['beta'],
                                W_A_MHz=WA)
        st = snap_stats(sn, res, W_A=WA, tag=f'WA{WA:.0f}')
        rows2.append(st)
df2 = pd.DataFrame(rows2)
df2.to_csv(os.path.join(OUT, 'task2_wa_sensitivity.csv'), index=False)
print(f'task2 done {time.time()-t_wall:.0f}s', flush=True)

print('\n--- W_A sensitivity summary ---', flush=True)
agg = df2.groupby('W_A').agg(
    n=('si', 'count'),
    pool_bh_util=('pool_bh_util', 'mean'),
    pool_bh_bind=('pool_bh_bind', 'mean'),
    bh_bind_rate=('bh_bind_rate', 'mean'),
    nu_bh_pool_pos=('nu_bh_pool', lambda x: float((x > EPS_ACT).mean())),
    nu_acc_pool_pos=('nu_acc_pool', lambda x: float((x > EPS_ACT).mean())),
    pool_acc_util=('pool_acc_util', 'mean'),
    pool_acc_bind=('pool_acc_bind', 'mean'),
    dem_cap_rate=('dem_cap_rate', 'mean'),
    demand_sat=('demand_sat', 'mean'),
    D_sum=('D_sum', 'mean'),
    Rbh_sum=('Rbh_sum', 'mean'),
    nu_bh_pos=('nu_bh_pos', 'mean'),
).round(4)
print(agg.to_string(), flush=True)
# transition check: at which W_A does bh pool stop binding and acc pool start
print('\n--- transition check ---', flush=True)
for WA, sub in df2.groupby('W_A'):
    print(f"  W_A={WA:.0f}: bh_pool_bind={sub['pool_bh_bind'].mean():.2f} "
          f"acc_pool_bind={sub['pool_acc_bind'].mean():.2f} "
          f"dem_cap={sub['dem_cap_rate'].mean():.2f} "
          f"nu_acc_pool>0={(sub['nu_acc_pool']>EPS_ACT).mean():.2f}", flush=True)


# =========================== TASK 3 ==========================================
# Geometry variants: G0 original; G1 800x800 expanded (cluster-preserving);
# G2 asymmetric hotspots. Airship fixed vs best legal horizontal move.
SPD_AIR = 5.0  # m/s * tau = 5 m step (same as stage2)


def scale_geometry(sn, factor=800.0 / 500.0):
    """G1: scale user and UAV xy by factor about origin; altitudes unchanged.
    Intra-cluster relative geometry preserved exactly; absolute distances scale."""
    sn2 = dict(sn)
    sn2['ux'] = sn['ux'] * factor
    sn2['uy'] = sn['uy'] * factor
    sn2['locs0'] = sn['locs0'].copy()
    sn2['locs0'][:, 0] *= factor
    sn2['locs0'][:, 1] *= factor
    # airship stays at (0,0,250): backhaul distances scale with UAV xy
    return sn2


def asym_geometry(sn, rng):
    """G2: shift user clusters anisotropically to create hotspot imbalance.
    Move users of each cluster along a per-cluster offset; UAVs unchanged.
    Intra-cluster shape preserved, inter-cluster distances change asymmetrically."""
    sn2 = dict(sn)
    cids = sn['cids']
    ux, uy = sn['ux'].copy(), sn['uy'].copy()
    offsets = rng.uniform(-60, 60, size=(4, 2))
    offsets[0] = [-80, 40]   # deterministic asymmetry
    offsets[1] = [-80, 40]
    offsets[2] = [70, -50]
    offsets[3] = [70, -50]
    for u in range(4):
        m = cids == u
        ux[m] += offsets[u, 0]
        uy[m] += offsets[u, 1]
    sn2['ux'] = np.clip(ux, -249, 249)
    sn2['uy'] = np.clip(uy, -249, 249)
    return sn2


def airship_move_eval(sn, e_a=None):
    """Evaluate airship fixed vs 8 legal moves (8 compass directions, step SPD_AIR)."""
    g_acc0, g_bh0 = channel_gammas(sn['ux'], sn['uy'], sn['cids'], sn['locs0'], sn['air0'])
    res0 = solve_blocka_full(g_acc0, g_bh0, sn['demand'], sn['cids'], sn['beta'])
    J0 = -res0['U'] if res0['U'] else np.nan  # U = beta.D; J sign handled outside
    U0 = res0['U']
    best_U = U0
    best_move = 'stay'
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in dirs:
        nrm = np.hypot(dx, dy)
        air2 = sn['air0'].copy()
        air2[0] = np.clip(air2[0] + SPD_AIR * dx / nrm * TAU, -249, 249)
        air2[1] = np.clip(air2[1] + SPD_AIR * dy / nrm * TAU, -249, 249)
        g_acc1, g_bh1 = channel_gammas(sn['ux'], sn['uy'], sn['cids'], sn['locs0'], air2)
        r1 = solve_blocka_full(g_acc1, g_bh1, sn['demand'], sn['cids'], sn['beta'])
        if r1['U'] > best_U + 1e-9:
            best_U = r1['U']
            best_move = f'({dx},{dy})'
    return dict(U0=U0, U_best=best_U, dU=best_U - U0, move=best_move,
                bh_pool_util=float(res0['W_ua_Hz'].sum()) / MHZ / W_A0,
                bh_bind_rate=float(((res0['R_bh_Mb'] - res0['S_u']) <=
                                     1e-3 * np.maximum(res0['R_bh_Mb'], 1e-9))[
                                        (res0['S_u'] > 1e-9)].mean())
                if (res0['S_u'] > 1e-9).any() else 0.0)


# representative states for task 3: 12 snapshots spread across time
t3_idx = [int(x) for x in np.linspace(0, len(SNAP) - 1, 12)]
print(f'\n===== Task 3: airship-move geometry sensitivity ({len(t3_idx)} states) =====', flush=True)
rows3 = []
t_wall = time.time()
for gi in t3_idx:
    sn0 = SNAP[gi]
    for gname in ('G0', 'G1', 'G2'):
        if gname == 'G0':
            sn = sn0
        elif gname == 'G1':
            sn = scale_geometry(sn0)
        else:
            sn = asym_geometry(sn0, np.random.default_rng(gi))
        try:
            ev = airship_move_eval(sn)
        except Exception as e:
            print(f'  si={gi} {gname} failed: {e}', flush=True)
            continue
        rows3.append(dict(si=sn0['si'], t=sn0['t'], geom=gname, **ev))
df3 = pd.DataFrame(rows3)
df3.to_csv(os.path.join(OUT, 'task3_airship_geom.csv'), index=False)
print(f'task3 done {time.time()-t_wall:.0f}s', flush=True)

print('\n--- airship move marginal gain dU = U_best - U_stay (Mb x beta) ---', flush=True)
g3 = df3.groupby('geom').agg(
    n=('si', 'count'),
    dU_mean=('dU', 'mean'), dU_median=('dU', 'median'), dU_max=('dU', 'max'),
    rel_mean=('dU', lambda x: float((x / df3.loc[x.index, 'U0'].replace(0, np.nan)).mean())),
    moved_rate=('move', lambda x: float((x != 'stay').mean())),
    bh_util=('bh_pool_util', 'mean'), bh_bind=('bh_bind_rate', 'mean'),
).round(5)
print(g3.to_string(), flush=True)

# =========================== FINAL REPORT ===================================
rep = []
rep.append('# 场景合理性诊断报告\n')
rep.append('日期：2026-10-14；数据：152 冻结快照；未运行闭环、未训练 Actor、未改生产控制器。\n')

rep.append('## 参数审计（任务一前提）\n')
rep.append(f'- 接入共享池 W_U = {W_U0:.0f} MHz（全部用户共享，`bw_total_Hz`）；参考单用户带宽 0.5 MHz（`bw_per_user`），Hybrid 固定 30/60/10 业务池即对 W_U 的静态切分，Proposed 的 Block A 动态分配同一池。\n')
rep.append(f'- 回传共享池 W_A = {W_A0:.0f} MHz（4 架共享，`bw_bh_total_Hz`；参考 5 MHz/UAV），飞艇 25 dBm、纯 LoS、噪声 PSD 按 5 MHz 参考带宽标定。\n')
rep.append(f'- 用户 ON 到达 [2.0, 5.0, 0.1] Mbps（Web/Video/Msg），slot=1 s，500×500 m（boundary=250），飞艇 (0,0,250)，UAV 高度 150 m，MOS 权重 [0.35,0.30,0.35]。\n')
rep.append(f'- Block A 单目标 max Σβ·D，约束：单用户接入速率、需求封顶 Q+A、接入池 ΣW≤W_U、回传池 ΣWb≤W_A、每 UAV 回传 ΣD≤τ·f(Wb)。\n')

rep.append('\n## 任务一：瓶颈统计（W_A=20 基准）\n')
for name, sub in (('全部', df1), ('早期 t≤120', early), ('稳定 t>120', late)):
    rep.append(f'\n### {name}（n={len(sub)}）\n')
    rep.append(f'- 接入池占用 mean={sub["pool_acc_util"].mean():.3f}，紧绑定率={sub["pool_acc_bind"].mean():.3f}，ν_acc_pool>0 比例={float((sub["nu_acc_pool"]>EPS_ACT).mean()):.3f}\n')
    rep.append(f'- 回传池占用 mean={sub["pool_bh_util"].mean():.3f}，紧绑定率={sub["pool_bh_bind"].mean():.3f}，ν_bh_pool>0 比例={float((sub["nu_bh_pool"]>EPS_ACT).mean()):.3f}\n')
    rep.append(f'- 每 UAV 回传紧绑定率 mean={sub["bh_bind_rate"].mean():.3f}，ν_bh>0 比例={sub["nu_bh_pos"].mean():.3f}\n')
    rep.append(f'- 需求封顶（D≥Q+A）比例={sub["dem_cap_rate"].mean():.3f}，λ_dem>0={sub["lam_dem_pos"].mean():.3f}\n')
    rep.append(f'- 接入速率约束活动率={sub["acc_bind_rate"].mean():.3f}，λ_acc>0={sub["lam_acc_pos"].mean():.3f}\n')
    rep.append(f'- 回传容量/待服务需求 mean={sub["bh_vs_dem"].mean():.3f}；需求满足率 mean={sub["demand_sat"].mean():.3f}\n')
rep.append(f'\n瓶颈组合分布（对偶存在性）：{df1["dom"].value_counts().to_dict()}\n')

rep.append('\n## 任务二：W_A 敏感性（40 个分层抽样快照）\n')
rep.append('\n| W_A (MHz) | 回传池占用 | 回传池绑定率 | 接入池绑定率 | 需求封顶 | 需求满足率 | D_sum |\n')
rep.append('|---|---|---|---|---|---|---|\n')
for WA, sub in df2.groupby('W_A'):
    rep.append(f'| {WA:.0f} | {sub["pool_bh_util"].mean():.3f} | {sub["pool_bh_bind"].mean():.3f} | '
               f'{sub["pool_acc_bind"].mean():.3f} | {sub["dem_cap_rate"].mean():.3f} | '
               f'{sub["demand_sat"].mean():.3f} | {sub["D_sum"].mean():.2f} |\n')

rep.append('\n## 任务三：飞艇移动几何敏感性\n')
rep.append('\n| 几何 | dU mean | dU median | 相对收益 | 移动选择率 | 回传池占用 |\n')
rep.append('|---|---|---|---|---|---|\n')
for gname, sub in df3.groupby('geom'):
    rel = (sub['dU'] / sub['U0'].replace(0, np.nan)).mean()
    rep.append(f'| {gname} | {sub["dU"].mean():.4f} | {sub["dU"].median():.4f} | {rel:.5f} | '
               f'{(sub["move"]!="stay").mean():.2f} | {sub["bh_pool_util"].mean():.3f} |\n')

rep.append('\n## 最终回答\n')
rep.append('（基于上方统计，由脚本尾部人工判读填充——见 console 输出与本表数值）\n')

with open(os.path.join(OUT, 'scenario_audit_report.md'), 'w', encoding='utf-8') as f:
    f.writelines(rep)
print(f'\nwrote {OUT}', flush=True)

# plot
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
ax = axes[0, 0]
ax.hist(df1['pool_bh_util'], bins=25, color='tab:orange', alpha=0.8)
ax.axvline(1.0, color='r', ls='--'); ax.set_title('backhaul pool util (W_A=20)')
ax = axes[0, 1]
ax.hist(df1['pool_acc_util'], bins=25, color='tab:blue', alpha=0.8)
ax.axvline(1.0, color='r', ls='--'); ax.set_title('access pool util (W_U=40)')
ax = axes[0, 2]
ax.hist(df1['dem_cap_rate'], bins=25, color='tab:green', alpha=0.8)
ax.set_title('demand-cap fraction per snap')
ax = axes[1, 0]
agg2 = df2.groupby('W_A')[['pool_bh_bind', 'pool_acc_bind', 'dem_cap_rate']].mean()
agg2.plot(kind='bar', ax=ax); ax.set_title('constraint binding vs W_A'); ax.grid(alpha=0.3)
ax = axes[1, 1]
for gname, sub in df3.groupby('geom'):
    ax.hist(sub['dU'], bins=10, alpha=0.5, label=gname)
ax.set_title('airship move dU by geometry'); ax.legend(); ax.grid(alpha=0.3)
ax = axes[1, 2]
ax.scatter(df1['qlen'], df1['pool_bh_util'], s=14, alpha=0.6, c='tab:orange')
ax.set_xlabel('qlen'); ax.set_ylabel('bh pool util'); ax.set_title('load vs bh util'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'scenario_audit.png'), dpi=130)
print('plots saved', flush=True)

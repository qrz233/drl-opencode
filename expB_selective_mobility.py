# -*- coding: utf-8
"""Selective Mobility feasibility diagnostic (offline only, no production changes).

Uses the 152 frozen Experiment-A snapshots.

Baselines (joint motion, all UAVs + optional airship move):
  stay, beta-0.25, beta-0.5, binding-0.25, binding-0.5   (= Binding K5)

Selective candidates: for each UAV u, only u moves (others + airship fixed):
  directions {beta, bh, binding} x rho {0.25, 0.5}

All candidates: Block A re-solve, unified J = psi_E*(E0.P) - beta.D

Recorded per snapshot:
  best joint J, best selective J, heuristic-selective J (UAV with max nu),
  per-UAV nu / binding / queue pressure / E0 / P delta,
  Block A solve counts, energy (sum P) for joint vs selective.

Case A vs B:
  A: a single heuristic choice (max-nu UAV) captures most of selective gain
  B: only full enumeration over UAVs finds the gain (heuristic << best)

Outputs -> expB_results\\selective_mobility
"""
import os, sys, time
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
OUT = r'D:\Learn\work\drl-opencode\expB_results\selective_mobility'
os.makedirs(OUT, exist_ok=True)
EXPA = r'D:\Learn\work\drl-opencode\expA_results'

from params_sys import *
from utils_sys import cal_channel_fading_gain, cal_uav_propulsion_energy
import blocka_optimizer as B

MHZ, LN2 = B.MHZ, B.LN2
PSI_E = 1.0 / PW_MAX
TAU = float(slot_len)
SPD_XY = float(uav_speedxy_max)
BOUNDARY = float(boundary)
DIR_EPS, NU_EPS = 1e-6, 1e-9
C_NU = 0.1
BIND_EPS = 1e-6
RHOS = (0.25, 0.5)
DIR_NAMES = ('beta', 'bh', 'binding')


def solve_blocka_diag(gamma_acc_MHz, gamma_bh_MHz, demand_Mb, assoc, beta,
                      W_U_MHz=None, W_A_MHz=None, tau=TAU, solver='CLARABEL'):
    n_users = len(gamma_acc_MHz)
    n_uavs = len(gamma_bh_MHz)
    W_U_MHz = bw_total_Hz / MHZ if W_U_MHz is None else W_U_MHz
    W_A_MHz = bw_bh_total_Hz / MHZ if W_A_MHz is None else W_A_MHz
    out = dict(D_Mb=np.zeros(n_users), R_bh_Mb=np.zeros(n_uavs),
               W_ua_Hz=np.zeros(n_uavs), status='trivial-no-demand',
               U=0.0, nu_full=np.zeros(n_uavs), nu_raw=None, aidx=None)
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
    constraints = [D <= tau * rate_acc, D <= dem,
                   cp.sum(W) <= W_U_MHz, cp.sum(Wb) <= W_A_MHz]
    bh_cons = [cp.sum(D[bidx == u]) <= tau * rate_bh[u] for u in range(m)]
    constraints += bh_cons
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
    out['W_ua_Hz'][aidx] = Wb_sol * MHZ
    out['R_bh_Mb'][aidx] = tau * Wb_sol * np.log2(1.0 + g_bh / np.maximum(Wb_sol, 1e-12))
    out['status'] = str(prob.status)
    out['U'] = float(prob.value)
    nu = [float(c.dual_value) if c.dual_value is not None else np.nan for c in bh_cons]
    out['nu_raw'] = np.array(nu)
    out['nu_full'][aidx] = np.array(nu)
    return out


def channel_gammas(user_x, user_y, cids, uav_locs, airship_loc):
    g_acc = np.zeros(len(cids))
    for u in np.unique(cids):
        ids_u = np.where(cids == u)[0]
        g_acc[ids_u] = cal_channel_fading_gain(user_x[ids_u], user_y[ids_u],
                                               uav_locs[u, 0], uav_locs[u, 1], uav_locs[u, 2])
    gamma_bh, _ = B.cal_gamma_backhaul_MHz(uav_locs, airship_loc)
    return B.cal_gamma_access_MHz(g_acc), gamma_bh


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > DIR_EPS else np.zeros_like(v)


def eval_positions(locs, air, ctx, E0, v_xy, v_z):
    P = np.array([float(cal_uav_propulsion_energy(V=np.linalg.norm(v_xy[u]), Vz=v_z[u])[0])
                  for u in range(n_uavs)])
    g_acc, g_bh = channel_gammas(ctx['user_x'], ctx['user_y'], ctx['cids'], locs, air)
    res = solve_blocka_diag(g_acc, g_bh, ctx['demand'], ctx['cids'], ctx['beta'])
    J = PSI_E * float(E0 @ P) - float(ctx['beta'] @ res['D_Mb'])
    return dict(J=J, P=P, res=res, D_sum=float(ctx['beta'] @ res['D_Mb']),
                nu=res['nu_full'].copy())


def beta_directions(beta, types, cids, locs0, air0, user_x, user_y):
    active = (types >= 0) & (beta > 0)
    user_xy = np.stack([user_x, user_y], axis=1)
    q_xy = locs0[:, :2]
    cent = q_xy.copy()
    for u in range(n_uavs):
        m = (cids == u) & active
        if not np.any(m):
            continue
        bw = float(beta[m].sum())
        if bw > 1e-12:
            cent[u] = (beta[m, None] * user_xy[m]).sum(axis=0) / bw
    B_u = np.array([float(beta[(cids == u) & active].sum()) for u in range(n_uavs)])
    if B_u.sum() > 1e-12:
        air_target = (B_u[:, None] * q_xy).sum(axis=0) / B_u.sum()
    else:
        air_target = air0[:2].copy()
    d_xy = cent - q_xy
    nrm = np.linalg.norm(d_xy, axis=1, keepdims=True)
    e_u = np.where(nrm > DIR_EPS, d_xy / np.maximum(nrm, DIR_EPS), 0.0)
    d_a = air_target - air0[:2]
    na = float(np.linalg.norm(d_a))
    e_a = d_a / na if na > DIR_EPS else np.zeros(2)
    return e_u, e_a


def apply_move(locs0, u, e2, rho, air_fixed=True, air=None, e_a=None):
    """Move only UAV u horizontally by rho*SPD_XY*e2; others unchanged.
    Airship fixed unless air/e_a given."""
    locs = locs0.copy()
    v_xy = np.zeros((n_uavs, 2))
    v_z = np.zeros(n_uavs)
    v_xy[u] = SPD_XY * rho * e2
    locs[u, 0] = np.clip(locs0[u, 0] + v_xy[u, 0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    locs[u, 1] = np.clip(locs0[u, 1] + v_xy[u, 1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    if air_fixed:
        air_out = air0_global.copy()
    else:
        air_out = air.copy()
        va = SPD_XY * rho * e_a
        air_out[0] = np.clip(air_out[0] + va[0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
        air_out[1] = np.clip(air_out[1] + va[1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    return locs, air_out, v_xy, v_z


def apply_joint(locs0, e_u, rho, air0_, e_a, move_air=True):
    locs = locs0.copy()
    v_xy = SPD_XY * rho * e_u
    v_z = np.zeros(n_uavs)
    for u in range(n_uavs):
        locs[u, 0] = np.clip(locs0[u, 0] + v_xy[u, 0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
        locs[u, 1] = np.clip(locs0[u, 1] + v_xy[u, 1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    air_out = air0_.copy()
    if move_air:
        va = SPD_XY * rho * e_a
        air_out[0] = np.clip(air_out[0] + va[0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
        air_out[1] = np.clip(air_out[1] + va[1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    return locs, air_out, v_xy, v_z


# ============================ load snapshots =================================
snap = np.load(os.path.join(EXPA, 'snapshots.npz'))
S_N = len(snap['slots'])
print(f'loaded {S_N} frozen snapshots', flush=True)

rows = []
case_rows = []
t_wall = time.time()

for si in range(S_N):
    t = int(snap['slots'][si])
    beta = snap['beta'][si].copy()
    types = snap['types'][si].copy()
    cids = snap['cids'][si].copy()
    locs0 = snap['locs_dec'][si].copy()
    air0 = snap['air_dec'][si].copy()
    E0 = snap['E_dec'][si].copy()
    A = snap['A'][si].copy()
    D_base = snap['D'][si].copy()
    dem_slack = snap['dem_slack'][si].copy()
    ux = snap['user_x'][si]
    uy = snap['user_y'][si]
    nu_snap = snap['nu'][si].copy()          # recorded base duals
    eta_snap = snap['eta'][si].copy()
    qlen_snap = snap['qlen'][si]
    demand = np.maximum(D_base + dem_slack, 0.0)
    Q_prev = np.maximum(demand - np.maximum(A, 0.0), 0.0)
    ctx = dict(cids=cids, user_x=ux, user_y=uy, Q_prev=Q_prev, A=A,
               types=types, beta=beta, demand=demand, locs0=locs0, air0=air0)
    air0_global = air0.copy()

    # directions
    e_u_beta, e_a_beta = beta_directions(beta, types, cids, locs0, air0, ux, uy)
    e_bh = np.array([unit(air0[:2] - locs0[u, :2]) for u in range(n_uavs)])
    nu = np.maximum(np.nan_to_num(nu_snap, nan=0.0), 0.0)
    binding = nu > BIND_EPS
    w_bind = binding.astype(float)
    e_u_bind = np.zeros((n_uavs, 2))
    for u in range(n_uavs):
        e_u_bind[u] = unit((1.0 - w_bind[u]) * e_u_beta[u] + w_bind[u] * e_bh[u])
    dir_bank = {'beta': e_u_beta, 'bh': e_bh, 'binding': e_u_bind}

    # per-UAV queue pressure (sum beta of own cluster)
    qp = np.array([float(beta[cids == u].sum()) for u in range(n_uavs)])

    # ---- stay baseline
    n_solves = 0
    ev0 = eval_positions(locs0.copy(), air0.copy(), ctx, E0,
                         np.zeros((n_uavs, 2)), np.zeros(n_uavs))
    n_solves += 1
    J0 = ev0['J']

    # ---- joint Binding K5 (all UAVs move; airship fixed to match Binding spec)
    joint = {}
    for name, e_u, rho in (
            ('stay', np.zeros((n_uavs, 2)), 0.0),
            ('b25', e_u_beta, 0.25), ('b50', e_u_beta, 0.50),
            ('bind25', e_u_bind, 0.25), ('bind50', e_u_bind, 0.50)):
        if name == 'stay':
            joint[name] = ev0
            continue
        locs, air_o, vxy, vz = apply_joint(locs0, e_u, rho, air0, e_a_beta, move_air=False)
        joint[name] = eval_positions(locs, air_o, ctx, E0, vxy, vz)
        n_solves += 1
    J_joint = {k: v['J'] for k, v in joint.items()}
    best_joint_name = min(J_joint, key=J_joint.get)
    J_joint_best = J_joint[best_joint_name]
    dJ_joint = J0 - J_joint_best

    # ---- selective: only UAV u moves; airship fixed
    sel_records = []
    for u in range(n_uavs):
        for dname in DIR_NAMES:
            e2 = dir_bank[dname][u]
            if np.linalg.norm(e2) < DIR_EPS:
                continue
            for rho in RHOS:
                locs, air_o, vxy, vz = apply_move(locs0, u, e2, rho)
                ev = eval_positions(locs, air_o, ctx, E0, vxy, vz)
                n_solves += 1
                sel_records.append(dict(
                    u=u, dir=dname, rho=rho, J=ev['J'], dJ=J0 - ev['J'],
                    D_sum=ev['D_sum'], P_sum=float(ev['P'].sum()),
                    P_u=float(ev['P'][u]), nu_u=float(nu[u]),
                    binding_u=int(binding[u]), qp_u=float(qp[u]),
                    E0_u=float(E0[u])))
    df_sel = pd.DataFrame(sel_records)
    if len(df_sel) == 0:
        continue
    best_sel_row = df_sel.loc[df_sel['J'].idxmin()]
    J_sel_best = float(best_sel_row['J'])
    dJ_sel = J0 - J_sel_best

    # heuristic: pick UAV with max nu (ties -> max qp), then best dir/rho for that UAV
    u_heur = int(np.argmax(nu))
    sub_h = df_sel[df_sel['u'] == u_heur]
    best_h_row = sub_h.loc[sub_h['J'].idxmin()]
    J_sel_heur = float(best_h_row['J'])
    dJ_heur = J0 - J_sel_heur

    # oracle single fixed UAV: best over u of (best dir/rho for that u)
    per_u_best = df_sel.groupby('u')['J'].min()
    u_oracle = int(per_u_best.idxmin())
    J_sel_oracle_u = float(per_u_best.min())
    dJ_oracle_u = J0 - J_sel_oracle_u

    # how many UAVs needed (enumerate in order of nu desc) to reach joint gain
    order = np.argsort(-nu)
    J_seq = []
    for k in range(n_uavs):
        u_k = int(order[k])
        J_seq.append(float(df_sel[df_sel['u'] == u_k]['J'].min()))
    n_uav_to_joint = next((k + 1 for k, Js in enumerate(J_seq) if Js <= J_joint_best), n_uavs)

    # energy comparison (best joint vs best selective)
    P_joint_best = float(joint[best_joint_name]['P'].sum())
    P_sel_best = float(best_sel_row['P_sum'])
    P0 = float(ev0['P'].sum())

    rows.append(dict(
        slot=t,
        J0=J0, J_joint_best=J_joint_best, dJ_joint=dJ_joint,
        best_joint=best_joint_name,
        J_sel_best=J_sel_best, dJ_sel=dJ_sel,
        sel_u=int(best_sel_row['u']), sel_dir=best_sel_row['dir'], sel_rho=float(best_sel_row['rho']),
        J_sel_heur=J_sel_heur, dJ_heur=dJ_heur,
        heur_u=u_heur, heur_dir=best_h_row['dir'], heur_rho=float(best_h_row['rho']),
        J_sel_oracle_u=J_sel_oracle_u, dJ_oracle_u=dJ_oracle_u, oracle_u=u_oracle,
        n_uav_to_joint=n_uav_to_joint,
        nu_max=float(nu.max()), nu_sum=float(nu.sum()), n_binding=int(binding.sum()),
        qp_sum=float(qp.sum()), qlen=qlen_snap, E0_sum=float(E0.sum()),
        P0=P0, P_joint_best=P_joint_best, P_sel_best=P_sel_best,
        P_sel_u=float(best_sel_row['P_u']),
        n_solves=n_solves,
        n_sel_cand=len(df_sel),
        sel_gap_vs_joint=dJ_sel - dJ_joint,
        heur_gap_vs_sel=dJ_heur - dJ_sel,
        heur_gap_frac=(dJ_heur / dJ_sel) if dJ_sel > 1e-12 else np.nan,
        sel_better_joint=int(dJ_sel >= dJ_joint - 1e-9),
        sel_any_positive=int(dJ_sel > 1e-9),
    ))

    # detailed per-candidate table (long form)
    det = df_sel.copy()
    det['slot'] = t
    det['J0'] = J0
    det['J_joint_best'] = J_joint_best
    case_rows.append(det)

    if (si + 1) % 30 == 0:
        print(f'  {si+1}/{S_N}  {time.time()-t_wall:.0f}s  '
              f'dJ_joint={dJ_joint:.3f} dJ_sel={dJ_sel:.3f} dJ_heur={dJ_heur:.3f}', flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'selective_slots.csv'), index=False)
df_det = pd.concat(case_rows, ignore_index=True)
df_det.to_csv(os.path.join(OUT, 'selective_candidates.csv'), index=False)
print(f'done {S_N} snapshots in {time.time()-t_wall:.0f}s', flush=True)


# ============================ Summary ========================================
def describe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, p10=np.nan, p90=np.nan)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)))


print('\n===== Selective Mobility offline diagnostic =====', flush=True)
print('dJ joint Binding-K5 best vs Stay:', describe(df['dJ_joint']), flush=True)
print('dJ selective best (enum all UAV) vs Stay:', describe(df['dJ_sel']), flush=True)
print('dJ heuristic (max-nu UAV) vs Stay:', describe(df['dJ_heur']), flush=True)
print('dJ oracle fixed UAV vs Stay:', describe(df['dJ_oracle_u']), flush=True)
print(f"\nselective >= joint rate: {df['sel_better_joint'].mean():.3f}", flush=True)
print(f"selective any positive dJ rate: {df['sel_any_positive'].mean():.3f}", flush=True)
print('sel_gap (dJ_sel - dJ_joint):', describe(df['sel_gap_vs_joint']), flush=True)
print('heur_gap (dJ_heur - dJ_sel):', describe(df['heur_gap_vs_sel']), flush=True)
hf = df['heur_gap_frac'].dropna()
print(f'heur_gap_frac (heur/sel when sel>0): mean={hf.mean():.3f}  '
      f'frac_heur_covers_80pct={(hf >= 0.8).mean():.3f}  '
      f'frac_heur_covers_50pct={(hf >= 0.5).mean():.3f}', flush=True)
print('n_uav_to_joint:', describe(df['n_uav_to_joint']), flush=True)
print(f"n_binding mean={df['n_binding'].mean():.2f}  nu_max mean={df['nu_max'].mean():.3f}", flush=True)

print('\n--- energy ---', flush=True)
print('P0:', describe(df['P0']), flush=True)
print('P joint best:', describe(df['P_joint_best']), flush=True)
print('P selective best:', describe(df['P_sel_best']), flush=True)
print(f"selective energy saving vs joint: "
      f"mean={(df['P_joint_best'] - df['P_sel_best']).mean():.2f} W", flush=True)

print('\n--- compute ---', flush=True)
print(f"Block A solves per snapshot: stay+joint+selective = "
      f"1+4+{int(df['n_sel_cand'].mean())} = {int(df['n_solves'].mean())}", flush=True)
print(f"selective-only solves = {int(df['n_sel_cand'].mean())} "
      f"(4 UAV x 3 dir x 2 rho, zero-dir skipped)", flush=True)

# which UAV/dir/rho wins selective
print('\n--- selective winner distribution ---', flush=True)
print('by UAV:', df['sel_u'].value_counts().sort_index().to_dict(), flush=True)
print('by dir:', df['sel_dir'].value_counts().to_dict(), flush=True)
print('by rho:', df['sel_rho'].value_counts().to_dict(), flush=True)
print('binding among winners:', df_det.loc[
    df_det.groupby('slot')['J'].idxmin(), 'binding_u'].mean()
      if len(df_det) else 'n/a', flush=True)

# correlation: does binding/nu predict winning UAV?
det_best = df_det.loc[df_det.groupby('slot')['J'].idxmin()].copy()
print(f"\nwinner binding rate: {det_best['binding_u'].mean():.3f}  "
      f"(base binding rate {(df['n_binding'] / n_uavs).mean():.3f})", flush=True)
print(f"winner nu_u mean: {det_best['nu_u'].mean():.3f}  "
      f"(overall nu_max mean {df['nu_max'].mean():.3f})", flush=True)

# Case A vs B verdict
print('\n===== Case A vs B =====', flush=True)
cover80 = (hf >= 0.8).mean() if len(hf) else 0.0
sel_ge_joint = df['sel_better_joint'].mean()
heur_covers = df['heur_gap_frac'].fillna(0)
print(f'Case-A indicators: heur covers 80% of selective gain in {cover80:.1%} snapshots;', flush=True)
print(f'  selective >= joint in {sel_ge_joint:.1%} snapshots;', flush=True)
print(f'  mean n_uav needed to match joint = {df["n_uav_to_joint"].mean():.2f} / 4', flush=True)

# plots
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
ax = axes[0, 0]
ax.boxplot([df['dJ_joint'].dropna(), df['dJ_sel'].dropna(),
            df['dJ_heur'].dropna(), df['dJ_oracle_u'].dropna()],
           labels=['joint K5', 'sel enum', 'sel heur', 'sel oracle-u'], showfliers=False)
ax.set_ylabel(r'$\Delta J$ vs Stay'); ax.set_title('gain comparison'); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.scatter(df['dJ_joint'], df['dJ_sel'], s=12, alpha=0.6)
lim = [min(df['dJ_joint'].min(), df['dJ_sel'].min()),
       max(df['dJ_joint'].max(), df['dJ_sel'].max()) + 0.1]
ax.plot(lim, lim, 'k--', lw=0.8)
ax.set_xlabel(r'$\Delta J$ joint'); ax.set_ylabel(r'$\Delta J$ selective')
ax.set_title('selective vs joint'); ax.grid(alpha=0.3)

ax = axes[0, 2]
ax.scatter(df['dJ_sel'], df['dJ_heur'], s=12, alpha=0.6, c='tab:green')
lim = [min(df['dJ_sel'].min(), df['dJ_heur'].min()),
       max(df['dJ_sel'].max(), df['dJ_heur'].max()) + 0.1]
ax.plot(lim, lim, 'k--', lw=0.8)
ax.set_xlabel(r'$\Delta J$ sel enum'); ax.set_ylabel(r'$\Delta J$ heur')
ax.set_title('Case A: heur vs enum'); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.hist(df['heur_gap_frac'].dropna(), bins=20, color='tab:green', alpha=0.7)
ax.axvline(0.8, color='r', ls='--', lw=1, label='80%')
ax.set_xlabel('heur / sel gain fraction'); ax.set_title('Case A coverage')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.scatter(df['nu_max'], df['dJ_sel'], s=12, alpha=0.6, c='tab:orange')
ax.set_xlabel(r'$\max_u \nu_u$'); ax.set_ylabel(r'$\Delta J$ selective')
ax.set_title('backhaul pressure vs gain'); ax.grid(alpha=0.3)

ax = axes[1, 2]
ax.scatter(df['P_sel_best'], df['dJ_sel'], s=12, alpha=0.6, c='tab:red',
           label='sel')
ax.scatter(df['P_joint_best'], df['dJ_joint'], s=12, alpha=0.6, c='tab:blue',
           label='joint')
ax.set_xlabel('sum P (W)'); ax.set_ylabel(r'$\Delta J$'); ax.set_title('energy vs gain')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'selective_mobility.png'), dpi=140)
print(f'\nwrote {OUT}', flush=True)

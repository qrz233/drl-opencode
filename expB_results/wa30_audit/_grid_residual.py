# -*- coding: utf-8 -*-
"""Task 3-C coverage test at W_A=30:
greedy single-UAV grid descent (16 dirs x 2 rho) starting from the Dual-K5
best joint position. Residual gain beyond K5 => fixed-direction deficiency.
"""
import os, sys, time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
import numpy as np
import pandas as pd
import cvxpy as cp

PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
OUT = r'D:\Learn\work\drl-opencode\expB_results\wa30_audit'
EXPA = r'D:\Learn\work\drl-opencode\expA_results'

from params_sys import *
from utils_sys import cal_channel_fading_gain, cal_uav_propulsion_energy
import blocka_optimizer as B

MHZ, LN2 = B.MHZ, B.LN2
TAU = float(slot_len)
W_U0 = bw_total_Hz / MHZ
W_A30 = 30.0
PSI_E = 1.0 / PW_MAX
SPD_XY = float(uav_speedxy_max)
BOUNDARY = float(boundary)
DIR_EPS, NU_EPS, EPS_ACT = 1e-6, 1e-9, 1e-6
C_NU, BIND_EPS = 0.1, 1e-6
RHOS = (0.25, 0.5)


def solve_full(gamma_acc, gamma_bh, demand, assoc, beta, W_U=W_U0, W_A=W_A30):
    out = dict(D_Mb=np.zeros(len(gamma_acc)), U=0.0, W_ui_Hz=np.zeros(len(gamma_acc)),
               W_ua_Hz=np.zeros(len(gamma_bh)), R_bh_Mb=np.zeros(len(gamma_bh)),
               S_u=np.zeros(len(gamma_bh)), lam_acc=np.zeros(len(gamma_acc)),
               nu_bh=np.zeros(len(gamma_bh)), nu_bh_pool=0.0, nu_acc_pool=0.0,
               status='trivial-no-demand')
    active = (demand > 0) & (assoc >= 0) & (beta > 0)
    if not np.any(active):
        return out
    idx = np.where(active)[0]
    aidx = np.unique(assoc[idx])
    g_acc, dem, b = gamma_acc[idx], demand[idx], beta[idx]
    m = len(aidx)
    W = cp.Variable(len(idx), nonneg=True)
    D = cp.Variable(len(idx), nonneg=True)
    Wb = cp.Variable(m, nonneg=True)
    g_bh = gamma_bh[aidx]
    rate_acc = -cp.rel_entr(W, W + g_acc) / LN2
    rate_bh = -cp.rel_entr(Wb, Wb + g_bh) / LN2
    bidx = np.searchsorted(aidx, assoc[idx])
    c_acc = D <= TAU * rate_acc
    c_dem = D <= dem
    c_pa = cp.sum(W) <= W_U
    c_pb = cp.sum(Wb) <= W_A
    bhc = [cp.sum(D[bidx == u]) <= TAU * rate_bh[u] for u in range(m)]
    prob = cp.Problem(cp.Maximize(b @ D), [c_acc, c_dem, c_pa, c_pb] + bhc)
    status = None
    for sname in ('CLARABEL', 'SCS'):
        try:
            prob.solve(solver=sname, warm_start=True)
            status = prob.status
            if status in ('optimal', 'optimal_inaccurate'):
                break
        except Exception:
            status = 'solver-error'
    if status not in ('optimal', 'optimal_inaccurate'):
        out['status'] = str(status)
        return out
    Wv = np.maximum(W.value, 0.0)
    Dv = np.maximum(D.value, 0.0)
    Wbv = np.maximum(Wb.value, 0.0)
    out['D_Mb'][idx] = Dv
    out['W_ui_Hz'][idx] = Wv * MHZ
    out['W_ua_Hz'][aidx] = Wbv * MHZ
    out['R_bh_Mb'][aidx] = TAU * Wbv * np.log2(1.0 + g_bh / np.maximum(Wbv, 1e-12))
    for u in range(m):
        out['S_u'][aidx[u]] = np.sum(Dv[bidx == u])
    out['status'] = str(prob.status)
    out['U'] = float(prob.value)
    la = c_acc.dual_value
    if la is not None:
        out['lam_acc'][idx] = np.abs(np.asarray(la, dtype=float))
    if c_pa.dual_value is not None:
        out['nu_acc_pool'] = float(np.abs(c_pa.dual_value))
    if c_pb.dual_value is not None:
        out['nu_bh_pool'] = float(np.abs(c_pb.dual_value))
    nuv = [float(c.dual_value) if c.dual_value is not None else 0.0 for c in bhc]
    out['nu_bh'][aidx] = np.maximum(np.array(nuv), 0.0)
    return out


def gammas(ux, uy, cids, locs, air):
    g = np.zeros(len(cids))
    for u in np.unique(cids):
        ids = np.where(cids == u)[0]
        g[ids] = cal_channel_fading_gain(ux[ids], uy[ids], locs[u, 0], locs[u, 1], locs[u, 2])
    gb, _ = B.cal_gamma_backhaul_MHz(locs, air)
    return B.cal_gamma_access_MHz(g), gb, g


def load_snap():
    z = np.load(os.path.join(EXPA, 'snapshots.npz'))
    out = []
    for si in range(len(z['slots'])):
        out.append(dict(
            si=si, t=int(z['slots'][si]), beta=z['beta'][si].copy(),
            types=z['types'][si].copy(), cids=z['cids'][si].copy(),
            locs0=z['locs_dec'][si].copy(), air0=z['air_dec'][si].copy(),
            E0=z['E_dec'][si].copy(), demand=np.maximum(z['D'][si] + z['dem_slack'][si], 0.0),
            ux=z['user_x'][si].copy(), uy=z['user_y'][si].copy(),
            qlen=float(z['qlen'][si])))
    return out


SNAP = load_snap()


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > DIR_EPS else np.zeros_like(v)


def beta_dirs(beta, types, cids, locs0, air0, ux, uy):
    active = (types >= 0) & (beta > 0)
    uxy = np.stack([ux, uy], 1)
    q = locs0[:, :2]
    cent = q.copy()
    for u in range(4):
        m = (cids == u) & active
        if np.any(m):
            bw = float(beta[m].sum())
            if bw > 1e-12:
                cent[u] = (beta[m, None] * uxy[m]).sum(0) / bw
    B_u = np.array([float(beta[(cids == u) & active].sum()) for u in range(4)])
    at = (B_u[:, None] * q).sum(0) / B_u.sum() if B_u.sum() > 1e-12 else air0[:2].copy()
    d = cent - q
    n = np.linalg.norm(d, axis=1, keepdims=True)
    e_u = np.where(n > DIR_EPS, d / np.maximum(n, DIR_EPS), 0.0)
    da = at - air0[:2]
    na = float(np.linalg.norm(da))
    return e_u, (da / na if na > DIR_EPS else np.zeros(2))


def evalJ(locs, air, sn, E0, vxy):
    ga, gb, g = gammas(sn['ux'], sn['uy'], sn['cids'], locs, air)
    res = solve_full(ga, gb, sn['demand'], sn['cids'], sn['beta'])
    P = np.array([float(cal_uav_propulsion_energy(V=np.linalg.norm(vxy[u]), Vz=0.0)[0])
                  for u in range(4)])
    J = PSI_E * float(E0 @ P) - float(sn['beta'] @ res['D_Mb'])
    return J, res


def move_single(locs, u, e_dir, rho):
    locs2 = locs.copy()
    v = SPD_XY * rho * e_dir
    locs2[u, 0] = np.clip(locs[u, 0] + v[0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    locs2[u, 1] = np.clip(locs[u, 1] + v[1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    vxy = np.zeros((4, 2))
    vxy[u] = v
    return locs2, vxy


# 16 compass directions
DIRS = []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        if (dx, dy) != (0, 0):
            n = np.hypot(dx, dy)
            DIRS.append(np.array([dx / n, dy / n], float))
print(f'{len(DIRS)} dirs x {len(RHOS)} rho per UAV per pass', flush=True)


def joint_eval(locs, air, sn, E0, e_u, rho):
    locs2 = locs.copy()
    vxy = SPD_XY * rho * e_u
    for u in range(4):
        locs2[u, 0] = np.clip(locs[u, 0] + vxy[u, 0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
        locs2[u, 1] = np.clip(locs[u, 1] + vxy[u, 1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    J, res = evalJ(locs2, air, sn, E0, vxy)
    return J, locs2


rows = []
t0 = time.time()
d2 = pd.read_csv(os.path.join(OUT, 'task2_wa30_methods.csv'))
dual_best_si = set(d2['si'].tolist())

for sn in SNAP:
    if sn['si'] not in dual_best_si:
        continue
    beta, types, cids = sn['beta'], sn['types'], sn['cids']
    locs0, air0, E0 = sn['locs0'], sn['air0'], sn['E0']

    ga0, gb0, _ = gammas(sn['ux'], sn['uy'], cids, locs0, air0)
    base = solve_full(ga0, gb0, sn['demand'], cids, beta)
    nu = np.maximum(base['nu_bh'], 0.0)
    e_u_beta, e_a_beta = beta_dirs(beta, types, cids, locs0, air0, sn['ux'], sn['uy'])
    e_bh = np.array([unit(air0[:2] - locs0[u, :2]) for u in range(4)])
    w_d = np.where(nu > NU_EPS, nu / (C_NU + nu), 0.0)
    e_u_dual = np.array([unit((1 - w_d[u]) * e_u_beta[u] + w_d[u] * e_bh[u]) for u in range(4)])

    # reproduce dual-K5 best joint position
    J0, _ = evalJ(locs0.copy(), air0, sn, E0, np.zeros((4, 2)))
    best_J, best_loc, best_tag = J0, locs0.copy(), 'stay'
    for tag, rho, eu in (('b25', .25, e_u_beta), ('b50', .5, e_u_beta),
                         ('du25', .25, e_u_dual), ('du50', .5, e_u_dual)):
        J, loc2 = joint_eval(locs0, air0, sn, E0, eu, rho)
        if J < best_J - 1e-12:
            best_J, best_loc, best_tag = J, loc2, tag

    # greedy coordinate grid descent from K5 best (2 passes)
    cur_J, cur_loc = best_J, best_loc.copy()
    residual = 0.0
    applied = []
    for p in range(2):
        improved = False
        for u in range(4):
            local_best = (cur_J, None, None)
            for d in DIRS:
                for rho in RHOS:
                    loc2, v = move_single(cur_loc, u, d, rho)
                    J, _ = evalJ(loc2, air0, sn, E0, v)
                    if J < local_best[0] - 1e-9:
                        local_best = (J, loc2, (u, d, rho))
            if local_best[1] is not None and local_best[0] < cur_J - 1e-9:
                cur_J, cur_loc = local_best[0], local_best[1]
                applied.append((p, local_best[2]))
                improved = True
        if not improved:
            break
    residual = best_J - cur_J

    rows.append(dict(si=sn['si'], t=sn['t'], qlen=sn['qlen'],
                     J0=J0, k5_best=best_J, k5_tag=best_tag,
                     dJ_k5=J0 - best_J, dJ_grid=J0 - cur_J,
                     residual=residual, n_moves=len(applied),
                     pool_acc_bind=int(base['nu_acc_pool'] > EPS_ACT),
                     n_mu=int((base['lam_acc'] > EPS_ACT)[
                         (sn['demand'] > 0) & (cids >= 0) & (beta > 0)].sum())))
    print(f"  si={sn['si']:>3} k5={J0-best_J:7.3f} grid={J0-cur_J:7.3f} "
          f"resid={residual:.4f} moves={len(applied)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'taskC_grid_residual.csv'), index=False)
r = df['residual']
print(f"\nresidual beyond K5: mean={r.mean():.4f} med={r.median():.4f} "
      f"p90={r.quantile(0.9):.4f} max={r.max():.4f}")
print(f"snapshots with residual>0.05: {(r > 0.05).mean():.3f}  "
      f">0.5: {(r > 0.5).mean():.3f}  >2.0: {(r > 2.0).mean():.3f}")
print(f"mean dJ_k5={df['dJ_k5'].mean():.3f}  mean dJ_grid={df['dJ_grid'].mean():.3f}")
print(f"elapsed {time.time()-t0:.0f}s -> {OUT}", flush=True)

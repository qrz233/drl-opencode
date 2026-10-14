# -*- coding: utf-8
'''Experiment B ablation: Binding / Utilization / Dual guidance (offline only).

Uses the 152 frozen Experiment-A snapshots. UAV backhaul direction only;
airship and heights FIXED. No training, no queue update, no new simulation.

Weight schemes (UAV only):
  beta     : w_u = 0
  binding  : w_u = a_b * I(backhaul binding at base solve)
  util     : w_u = a_eta * clip((eta_u - eta0)/(1-eta0), 0, 1)
  dual     : w_u = nu_u / (0.1 + nu_u)          # fixed c_nu=0.1, no retune

Common direction: e_u = unit((1-w_u)*e_beta + w_u*e_bh)
K=5 candidates per method: [stay, beta0.25, beta0.5, guided0.25, guided0.5]
Unified scoring: Block A re-solve + J = psi_E*(E0.P) - beta.D

Binding / Utilization hyper-params calibrated on a DISJOIN calibration split
(every 5th snapshot, indices 0,5,10,...), then frozen for the test split.

Outputs -> expB_results\\ablation_guidance
'''
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
OUT = r'D:\Learn\work\drl-opencode\expB_results\ablation_guidance'
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
DIR_EPS = 1e-6
NU_EPS = 1e-9
C_NU = 0.1

# binding detection thresholds (not tuned on test)
NU_BIND_EPS = 1e-6
SLACK_REL_EPS = 1e-3          # slack < SLACK_REL_EPS * R_bh => near-tight

# calibration grids
AB_GRID = (0.3, 0.5, 0.7, 1.0)
AETA_GRID = (0.5, 0.75, 1.0)
ETA0_GRID = (0.5, 0.7, 0.8, 0.9)

RHOS = (0.25, 0.5)


def solve_blocka_diag(gamma_acc_MHz, gamma_bh_MHz, demand_Mb, assoc, beta,
                      W_U_MHz=None, W_A_MHz=None, tau=TAU, solver='CLARABEL'):
    n_users = len(gamma_acc_MHz)
    n_uavs = len(gamma_bh_MHz)
    W_U_MHz = bw_total_Hz / MHZ if W_U_MHz is None else W_U_MHz
    W_A_MHz = bw_bh_total_Hz / MHZ if W_A_MHz is None else W_A_MHz
    out = dict(D_Mb=np.zeros(n_users), R_bh_Mb=np.zeros(n_uavs),
               W_ua_Hz=np.zeros(n_uavs), status='trivial-no-demand',
               U=0.0, nu_full=np.zeros(n_uavs), nu_raw=None, aidx=None,
               bh_slack=np.zeros(n_uavs), dem_capped=np.zeros(n_uavs),
               R_acc_Mb=np.zeros(n_users))
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
    bh_cons = []
    for u in range(m):
        bh_cons.append(cp.sum(D[bidx == u]) <= tau * rate_bh[u])
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
    R_bh_sol = tau * Wb_sol * np.log2(1.0 + g_bh / np.maximum(Wb_sol, 1e-12))
    out['R_bh_Mb'][aidx] = R_bh_sol
    R_acc_sol = tau * W_sol * np.log2(1.0 + g_acc / np.maximum(W_sol, 1e-12))
    out['R_acc_Mb'][idx] = R_acc_sol
    out['status'] = str(prob.status)
    out['U'] = float(prob.value)
    nu = [float(c.dual_value) if c.dual_value is not None else np.nan for c in bh_cons]
    out['nu_raw'] = np.array(nu)
    out['nu_full'][aidx] = np.array(nu)
    # delivered vs backhaul capacity per UAV
    for u_i, u in enumerate(aidx):
        out['bh_slack'][u] = R_bh_sol[u_i] - float(D_sol[bidx == u_i].sum())
    # demand-capped flag: sum D near sum dem for that UAV
    for u in aidx:
        mask_u = (assoc[idx] == u)
        dem_u = float(dem[mask_u].sum())
        D_u = float(D_sol[mask_u].sum())
        out['dem_capped'][u] = D_u  # actual delivered
        # store dem via side channel on first call
    out['_dem_by_uav'] = np.array([float(dem[(assoc[idx] == u)].sum()) for u in range(n_uavs)])
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


def beta_directions(beta, types, cids, locs0, user_x, user_y):
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
    d_xy = cent - q_xy
    nrm = np.linalg.norm(d_xy, axis=1, keepdims=True)
    e_u = np.where(nrm > DIR_EPS, d_xy / np.maximum(nrm, DIR_EPS), 0.0)
    return e_u


def eval_uav_move(locs_xy_new, ctx, E0):
    '''Block A at fixed heights, airship fixed. v_z=0.'''
    locs = ctx['locs0'].copy()
    locs[:, 0] = locs_xy_new[:, 0]
    locs[:, 1] = locs_xy_new[:, 1]
    v_xy = (locs[:, :2] - ctx['locs0'][:, :2]) / TAU
    P = np.array([float(cal_uav_propulsion_energy(V=np.linalg.norm(v_xy[u]), Vz=0.0)[0])
                  for u in range(n_uavs)])
    g_acc, g_bh = channel_gammas(ctx['user_x'], ctx['user_y'], ctx['cids'], locs, ctx['air0'])
    res = solve_blocka_diag(g_acc, g_bh, ctx['demand'], ctx['cids'], ctx['beta'])
    J = PSI_E * float(E0 @ P) - float(ctx['beta'] @ res['D_Mb'])
    return dict(J=J, P=P, res=res, D_sum=float(ctx['beta'] @ res['D_Mb']),
                status=res['status'])


def apply_dir(e_u, rho, locs0):
    locs_xy = locs0[:, :2].copy()
    for u in range(n_uavs):
        v = SPD_XY * rho * e_u[u]
        locs_xy[u, 0] = np.clip(locs0[u, 0] + v[0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
        locs_xy[u, 1] = np.clip(locs0[u, 1] + v[1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    return locs_xy


# ============================ load snapshots =================================
snap = np.load(os.path.join(EXPA, 'snapshots.npz'))
S_N = len(snap['slots'])
print(f'loaded {S_N} frozen snapshots', flush=True)

cal_idx = np.arange(0, S_N, 5)          # 31 calibration snapshots
test_idx = np.array([i for i in range(S_N) if i not in set(cal_idx.tolist())])
print(f'calibration n={len(cal_idx)}  test n={len(test_idx)}', flush=True)


def prepare_snapshot(si):
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
    demand = np.maximum(D_base + dem_slack, 0.0)
    Q_prev = np.maximum(demand - np.maximum(A, 0.0), 0.0)
    ctx = dict(cids=cids, user_x=ux, user_y=uy, Q_prev=Q_prev, A=A,
               types=types, beta=beta, demand=demand, locs0=locs0, air0=air0,
               E0=E0, slot=t)
    return ctx


def base_and_dirs(ctx):
    '''One base Block A solve; extract nu, slack, eta, binding, directions.'''
    locs0, air0 = ctx['locs0'], ctx['air0']
    e_beta = beta_directions(ctx['beta'], ctx['types'], ctx['cids'], locs0,
                             ctx['user_x'], ctx['user_y'])
    g_acc, g_bh = channel_gammas(ctx['user_x'], ctx['user_y'], ctx['cids'], locs0, air0)
    base = solve_blocka_diag(g_acc, g_bh, ctx['demand'], ctx['cids'], ctx['beta'])
    nu = base['nu_full'].copy()
    nu_hat = np.maximum(nu, 0.0)
    nu_hat[~np.isfinite(nu_hat)] = 0.0
    # e_bh toward airship (xy)
    e_bh = np.array([unit(air0[:2] - locs0[u, :2]) for u in range(n_uavs)])
    # binding: dual > eps OR slack < eps * R_bh
    R_bh = base['R_bh_Mb']
    slack = base['bh_slack']
    binding = (nu_hat > NU_BIND_EPS) | (slack < SLACK_REL_EPS * np.maximum(R_bh, 1e-12))
    # utilization: demand / backhaul capacity (from base geometry)
    dem_u = base.get('_dem_by_uav', np.zeros(n_uavs))
    eta = dem_u / np.maximum(R_bh, 1e-12)
    # cancellation count for dual blend
    w_dual = np.where(nu_hat > NU_EPS, nu_hat / (C_NU + nu_hat), 0.0)
    d_raw = (1.0 - w_dual[:, None]) * e_beta + w_dual[:, None] * e_bh
    n_cancel = int(np.sum(np.linalg.norm(d_raw, axis=1) < DIR_EPS))
    for u in range(n_uavs):
        if np.linalg.norm(d_raw[u]) < DIR_EPS:
            d_raw[u] = e_bh[u] if w_dual[u] >= 0.5 else e_beta[u]
    e_dual = np.array([unit(d_raw[u]) for u in range(n_uavs)])
    anom = dict(
        base_status=base['status'],
        solver_fallback=int(base.get('solver_used', None) not in (None, 'CLARABEL')
                            and base['status'] in ('optimal', 'optimal_inaccurate')),
        zero_R_bh=int(np.sum(R_bh < 1e-12)),
        n_binding=int(binding.sum()),
        n_cancel_dual=n_cancel,
        nu_mean=float(nu_hat.mean()),
        eta_mean=float(np.mean(eta)),
    )
    return dict(e_beta=e_beta, e_bh=e_bh, e_dual=e_dual,
                binding=binding, nu_hat=nu_hat, eta=eta,
                w_dual=w_dual, base=base, anom=anom)


def w_scheme(method, dirs, a_b=None, a_eta=None, eta0=None):
    e_beta, e_bh = dirs['e_beta'], dirs['e_bh']
    if method == 'beta':
        w = np.zeros(n_uavs)
    elif method == 'dual':
        w = dirs['w_dual'].copy()
    elif method == 'binding':
        w = a_b * dirs['binding'].astype(float)
    elif method == 'util':
        w = a_eta * np.clip((dirs['eta'] - eta0) / max(1.0 - eta0, 1e-9), 0.0, 1.0)
    else:
        raise ValueError(method)
    d = (1.0 - w[:, None]) * e_beta + w[:, None] * e_bh
    e = np.zeros_like(d)
    n_cancel = 0
    for u in range(n_uavs):
        if np.linalg.norm(d[u]) < DIR_EPS:
            n_cancel += 1
            d[u] = e_bh[u] if w[u] >= 0.5 else e_beta[u]
        e[u] = unit(d[u])
    return e, w, n_cancel


def eval_k5(ctx, dirs, e_guided):
    '''K=5: stay, beta0.25, beta0.5, guided0.25, guided0.5. Shared betas cached by caller.'''
    locs0 = ctx['locs0']
    evs = []
    # stay
    evs.append(eval_uav_move(locs0[:, :2].copy(), ctx, ctx['E0']))
    return evs  # placeholder - will be structured differently


# ============================ per-snapshot evaluation ========================
rows = []            # method-level rows (test + cal, with split tag)
w_rows = []          # per-UAV weight rows
anoms = []

t_wall = time.time()
for si in range(S_N):
    ctx = prepare_snapshot(si)
    dirs = base_and_dirs(ctx)
    locs0 = ctx['locs0']
    E0 = ctx['E0']

    # ---- shared evaluations (cached once) ----
    ev_stay = eval_uav_move(locs0[:, :2].copy(), ctx, E0)
    ev_b = {}
    for rho in RHOS:
        ev_b[rho] = eval_uav_move(apply_dir(dirs['e_beta'], rho, locs0), ctx, E0)
    J0 = ev_stay['J']
    J_b25, J_b50 = ev_b[0.25]['J'], ev_b[0.5]['J']
    J_old = min(J0, J_b25, J_b50)          # common beta-K3 baseline

    split = 'cal' if si in set(cal_idx.tolist()) else 'test'

    # ---- four methods ----
    method_cfgs = [('beta', dict()), ('dual', dict())]
    for a_b in AB_GRID:
        method_cfgs.append((f'binding_a{a_b}', dict(a_b=a_b)))
    for a_eta in AETA_GRID:
        for eta0 in ETA0_GRID:
            method_cfgs.append((f'util_a{a_eta}_e{eta0}',
                                dict(a_eta=a_eta, eta0=eta0)))

    for method, cfg in method_cfgs:
        if method.startswith('binding'):
            e_g, w, nc = w_scheme('binding', dirs, a_b=cfg['a_b'])
            base_method = 'binding'
        elif method.startswith('util'):
            e_g, w, nc = w_scheme('util', dirs, a_eta=cfg['a_eta'], eta0=cfg['eta0'])
            base_method = 'util'
        elif method == 'beta':
            e_g, w, nc = w_scheme('beta', dirs)
            base_method = 'beta'
        else:
            e_g, w, nc = w_scheme('dual', dirs)
            base_method = 'dual'

        ev_g = {}
        for rho in RHOS:
            ev_g[rho] = eval_uav_move(apply_dir(e_g, rho, locs0), ctx, E0)
        J_g25, J_g50 = ev_g[0.25]['J'], ev_g[0.5]['J']
        J_best = min(J0, J_b25, J_b50, J_g25, J_g50)
        best_idx = int(np.argmin([J0, J_b25, J_b50, J_g25, J_g50]))
        dJ_gain = J_old - J_best

        # winner among guided vs beta-only set
        win_ev = [ev_stay, ev_b[0.25], ev_b[0.5], ev_g[0.25], ev_g[0.5]][best_idx]
        rows.append(dict(
            si=si, slot=ctx['slot'], split=split, method=method,
            base_method=base_method,
            J0=J0, J_b25=J_b25, J_b50=J_b50, J_g25=J_g25, J_g50=J_g50,
            J_old=J_old, J_best=J_best, dJ_gain=dJ_gain, best_idx=best_idx,
            guided_selected=int(best_idx in (3, 4)),
            D_sum=win_ev['D_sum'], P_sum=float(win_ev['P'].sum()),
            dD=win_ev['D_sum'] - ev_stay['D_sum'],
            dP=float(win_ev['P'].sum() - ev_stay['P'].sum()),
            status=win_ev['status'],
            n_cancel=nc,
            **{f'w{u}': float(w[u]) for u in range(n_uavs)},
            **{f'nu{u}': float(dirs['nu_hat'][u]) for u in range(n_uavs)},
            **{f'eta{u}': float(dirs['eta'][u]) for u in range(n_uavs)},
            **{f'bind{u}': int(dirs['binding'][u]) for u in range(n_uavs)},
        ))
        w_rows.append(dict(si=si, split=split, method=method,
                           mean_w=float(w.mean()),
                           n_nonzero_w=int((w > 1e-9).sum())))

    anoms.append(dict(si=si, split=split, **dirs['anom']))
    if (si + 1) % 20 == 0:
        print(f'  {si+1}/{S_N}  {time.time()-t_wall:.0f}s', flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'ablation_slots.csv'), index=False)
pd.DataFrame(anoms).to_csv(os.path.join(OUT, 'ablation_anoms.csv'), index=False)
print(f'[eval] done {S_N} snapshots in {time.time()-t_wall:.0f}s  rows={len(df)}', flush=True)

# ============================ calibration ====================================
cal = df[df['split'] == 'cal']
test = df[df['split'] == 'test']

# pick best binding param on cal (mean dJ_gain)
bind_cal = cal[cal['base_method'] == 'binding'].groupby('method')['dJ_gain'].mean()
best_bind = bind_cal.idxmax()
best_bind_a = float(best_bind.split('_a')[1])

util_cal = cal[cal['base_method'] == 'util'].groupby('method')['dJ_gain'].mean()
best_util = util_cal.idxmax()
# method like util_a0.75_e0.8
parts = best_util.split('_')
best_util_a = float(parts[1][1:])
best_util_e = float(parts[2][1:])

print(f'\n[cal] best binding: {best_bind}  mean dJ={bind_cal.max():.6f}', flush=True)
print(f'[cal] best util:    {best_util}  mean dJ={util_cal.max():.6f}', flush=True)
print(f'[cal] binding grid means:\n{bind_cal.sort_values(ascending=False).to_string()}', flush=True)
print(f'[cal] util top-8:\n{util_cal.sort_values(ascending=False).head(8).to_string()}', flush=True)

with open(os.path.join(OUT, 'calibration.json'), 'w') as f:
    json.dump(dict(best_binding=best_bind, best_binding_a=best_bind_a,
                   best_util=best_util, best_util_a=best_util_a,
                   best_util_eta0=best_util_e,
                   cal_n=int(len(cal)), test_n=int(len(test))), f, indent=2)

# ============================ test evaluation ================================
methods_focus = ['beta', 'dual', f'binding_a{best_bind_a}',
                 f'util_a{best_util_a}_e{best_util_e}']
# also keep raw names for lookup
name_map = {
    'beta': 'beta',
    'dual': 'dual',
    'binding': f'binding_a{best_bind_a}',
    'util': f'util_a{best_util_a}_e{best_util_e}',
}

te = test[test['method'].isin(methods_focus)].copy()
print(f'\n[test] focused methods n_rows={len(te)}', flush=True)

# overall summary
print('\n===== TEST: dJ_gain vs beta-K3 =====', flush=True)
for tag, m in name_map.items():
    sub = te[te['method'] == m]
    g = sub['dJ_gain'].to_numpy()
    print(f'{tag:10s}  n={len(g):3d}  mean={g.mean():+.6f}  median={np.median(g):+.6f}  '
          f'p10={np.percentile(g,10):+.6f}  p90={np.percentile(g,90):+.6f}  '
          f'win(g>1e-6)={np.mean(g>1e-6):.3f}', flush=True)

# paired dual vs binding / util
def paired(a_name, b_name):
    a = te[te['method'] == name_map[a_name]].set_index('si')
    b = te[te['method'] == name_map[b_name]].set_index('si')
    common = a.index.intersection(b.index)
    da = a.loc[common, 'dJ_gain']
    db = b.loc[common, 'dJ_gain']
    diff = da - db
    tol = 1e-6
    win = float((diff > tol).mean())
    tie = float((np.abs(diff) <= tol).mean())
    loss = float((diff < -tol).mean())
    print(f'paired {a_name} - {b_name}: n={len(common)}  mean_diff={diff.mean():+.6f}  '
          f'median={np.median(diff):+.6f}  win/tie/loss={win:.3f}/{tie:.3f}/{loss:.3f}', flush=True)
    return common, da, db, diff

print('\n===== TEST paired comparisons =====', flush=True)
paired('dual', 'binding')
paired('dual', 'util')
paired('dual', 'beta')
paired('binding', 'beta')
paired('util', 'beta')

# grouped results
print('\n===== TEST grouped =====', flush=True)
# early 120 slots / late
te_early = te[te['slot'] <= 120]
te_late = te[te['slot'] > 120]
for label, sub in [('early<=120', te_early), ('late>120', te_late)]:
    print(f'-- {label} --', flush=True)
    for tag, m in name_map.items():
        s2 = sub[sub['method'] == m]
        g = s2['dJ_gain'].to_numpy()
        if len(g) == 0:
            continue
        print(f'  {tag:10s} n={len(g):3d} mean={g.mean():+.6f}', flush=True)

# binding-any (any UAV binding) vs none - from dual rows (same state)
dual_rows = te[te['method'] == name_map['dual']].copy()
bind_any = dual_rows[[f'bind{u}' for u in range(n_uavs)]].max(axis=1) > 0
print('\n-- by binding-any (state from dual rows) --', flush=True)
for tag, m in name_map.items():
    s2 = te[te['method'] == m].set_index('si')
    for lab, mask in [('bind', bind_any), ('nobind', ~bind_any)]:
        si_sub = dual_rows.loc[mask, 'si'].values
        g = s2.loc[s2.index.isin(si_sub), 'dJ_gain'].to_numpy()
        if len(g):
            print(f'  {tag:10s} {lab:6s} n={len(g):3d} mean={g.mean():+.6f}', flush=True)

# high nu states: mean nu > median
nu_mean = dual_rows[[f'nu{u}' for u in range(n_uavs)]].mean(axis=1)
hi_nu = nu_mean > nu_mean.median()
print('\n-- by high-nu --', flush=True)
for tag, m in name_map.items():
    s2 = te[te['method'] == m].set_index('si')
    for lab, mask in [('hinux', hi_nu), ('lonu', ~hi_nu)]:
        si_sub = dual_rows.loc[mask, 'si'].values
        g = s2.loc[s2.index.isin(si_sub), 'dJ_gain'].to_numpy()
        if len(g):
            print(f'  {tag:10s} {lab:5s} n={len(g):3d} mean={g.mean():+.6f}', flush=True)

# weight distributions
print('\n===== weight distributions (test) =====', flush=True)
for tag, m in name_map.items():
    s2 = te[te['method'] == m]
    wmat = s2[[f'w{u}' for u in range(n_uavs)]].to_numpy()
    print(f'{tag:10s}  mean_w={wmat.mean():.4f}  '
          f'frac_w>0.01={np.mean(wmat>0.01):.3f}  '
          f'frac_w>0.5={np.mean(wmat>0.5):.3f}  '
          f'frac_w>0.9={np.mean(wmat>0.9):.3f}', flush=True)

# same binding state: do binding/util/dual produce different directions?
print('\n===== same-binding direction divergence =====', flush=True)
# For slots where binding pattern is identical, compare w vectors
bind_mat = dual_rows[[f'bind{u}' for u in range(n_uavs)]].to_numpy()
# count slots where binding and util and dual all have at least one w>0.01
b_rows = te[te['method'] == name_map['binding']].set_index('si')
u_rows = te[te['method'] == name_map['util']].set_index('si')
d_rows = te[te['method'] == name_map['dual']].set_index('si')
common_si = b_rows.index.intersection(u_rows.index).intersection(d_rows.index)
wb = b_rows.loc[common_si, [f'w{u}' for u in range(n_uavs)]].to_numpy()
wu = u_rows.loc[common_si, [f'w{u}' for u in range(n_uavs)]].to_numpy()
wd = d_rows.loc[common_si, [f'w{u}' for u in range(n_uavs)]].to_numpy()
bind_flag = bind_mat[:len(common_si)] if len(bind_mat) == len(common_si) else \
    d_rows.loc[common_si, [f'bind{u}' for u in range(n_uavs)]].to_numpy()
# among UAV-slots with binding=1
m_bind = bind_flag.astype(bool)
if m_bind.any():
    print(f'binding UAV-slots: {m_bind.sum()}  '
          f'mean w: binding={wb[m_bind].mean():.3f} util={wu[m_bind].mean():.3f} dual={wd[m_bind].mean():.3f}', flush=True)
    # util vs dual relative difference
    rel = np.abs(wu[m_bind] - wd[m_bind]) / np.maximum(wd[m_bind], 1e-9)
    print(f'  |w_util - w_dual|/w_dual  mean={rel.mean():.3f}  median={np.median(rel):.3f}', flush=True)
# among non-binding
m_nb = ~m_bind
if m_nb.any():
    print(f'non-binding UAV-slots: {m_nb.sum()}  '
          f'mean w: binding={wb[m_nb].mean():.3f} util={wu[m_nb].mean():.3f} dual={wd[m_nb].mean():.3f}', flush=True)

# dJ decomposition: delivery vs power
print('\n===== dJ decomposition (vs stay) =====', flush=True)
for tag, m in name_map.items():
    s2 = te[te['method'] == m]
    # use winner row
    dD = s2['dD'].to_numpy()
    dP = s2['dP'].to_numpy()
    # J gain vs stay = PSI_E*(-E0@delta_P) wait - dJ = J0 - Jbest = PSI_E*E0@P - beta@D - (PSI_E*E0@P0 - beta@D0)
    # = beta@(D-D0) - PSI_E*E0@(P-P0) approximately if E0 uniform
    print(f'{tag:10s}  mean dD={dD.mean():+.4f}  mean dP={dP.mean():+.4f}  '
          f'mean dJ={s2["dJ_gain"].mean():+.6f}', flush=True)

# anomaly summary
adf = pd.DataFrame(anoms)
print('\n===== anomalies =====', flush=True)
print(f'base non-optimal: {(adf["base_status"]!="optimal").sum()} / {len(adf)}', flush=True)
print(f'solver fallback: {adf["solver_fallback"].sum()}', flush=True)
print(f'zero R_bh slots: {adf["zero_R_bh"].sum()}', flush=True)
print(f'cancel dual: mean={adf["n_cancel_dual"].mean():.3f}  max={adf["n_cancel_dual"].max()}', flush=True)
print(f'n_binding: mean={adf["n_binding"].mean():.3f}  max={adf["n_binding"].max()}', flush=True)

# simple-signal sufficiency check
print('\n===== simple-signal sufficiency =====', flush=True)
c_d, da_d, db_d, diff_du = paired('dual', 'util')
c_b, da_b, db_b, diff_db = paired('dual', 'binding')
print(f'Criterion: |mean diff dual-util| < 0.05 AND win-rate dual over util < 55% => util sufficient', flush=True)
mean_du = float((da_d - db_d).mean())
wr_du = float(((da_d - db_d) > 1e-6).mean())
mean_db = float((da_b - db_b).mean())
wr_db = float(((da_b - db_b) > 1e-6).mean())
print(f'dual-util:   mean_diff={mean_du:+.6f}  win_rate={wr_du:.3f}  => util_sufficient={abs(mean_du)<0.05 and wr_du<0.55}', flush=True)
print(f'dual-binding: mean_diff={mean_db:+.6f}  win_rate={wr_db:.3f}  => binding_sufficient={abs(mean_db)<0.05 and wr_db<0.55}', flush=True)

# plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
data = [te[te['method'] == name_map[t]]['dJ_gain'].dropna().to_numpy()
        for t in ('beta', 'binding', 'util', 'dual')]
ax.boxplot(data, labels=['beta', 'binding', 'util', 'dual'], showfliers=False)
ax.set_ylabel(r'$\Delta J$ vs beta-K3')
ax.set_title('TEST: guidance ablation')
ax.grid(alpha=0.3)

ax = axes[1]
for tag, color in (('binding', 'tab:green'), ('util', 'tab:orange'), ('dual', 'tab:blue')):
    m = name_map[tag]
    s2 = te[te['method'] == m]
    ax.scatter(s2['slot'], s2['dJ_gain'], s=8, alpha=0.5, label=tag, color=color)
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('slot')
ax.set_ylabel(r'$\Delta J$')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
# mean weight by method
labels, means = [], []
for tag in ('beta', 'binding', 'util', 'dual'):
    m = name_map[tag]
    s2 = te[te['method'] == m]
    wmat = s2[[f'w{u}' for u in range(n_uavs)]].to_numpy().ravel()
    labels.append(tag)
    means.append(wmat.mean())
ax.bar(labels, means, color=['gray', 'tab:green', 'tab:orange', 'tab:blue'])
ax.set_ylabel('mean w')
ax.set_title('weight magnitude')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'ablation_guidance.png'), dpi=140)
print(f'\nwrote {OUT}', flush=True)

# -*- coding: utf-8
"""W_A=30 MHz supplementary diagnostics (offline, no closed loop, no Actor).

Task 1 audits:
  A1  G2 construction audit (coords / demand / beta / assoc / channel inputs)
  A2  W_A=30 constraint binding + positive-dual rates (4 constraint classes)
  A3  per-snapshot per-UAV ACCESS & BACKHAUL simultaneous positive duals
  A4  access dual mu finite-difference validation (RHS perturbation)
  A5  W_U=40 / W_A=30 spectrum-supply consistency check (static params)

Task 2: frozen-state position optimization at W_A=30
  beta-directed K3 vs Binding K5 vs Dual K5 (all re-solved at 30 MHz),
  unified J = psi_E*(E0.P) - beta.D; plus access-aware direction probe,
  plus UAV-only vs airship-move variants.

Outputs -> expB_results\\wa30_audit
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
OUT = r'D:\Learn\work\drl-opencode\expB_results\wa30_audit'
os.makedirs(OUT, exist_ok=True)
EXPA = r'D:\Learn\work\drl-opencode\expA_results'

from params_sys import *
from utils_sys import cal_channel_fading_gain, cal_uav_propulsion_energy
import blocka_optimizer as B

MHZ, LN2 = B.MHZ, B.LN2
TAU = float(slot_len)
W_U0 = bw_total_Hz / MHZ
W_A20 = 20.0
W_A30 = 30.0
PSI_E = 1.0 / PW_MAX
SPD_XY = float(uav_speedxy_max)
BOUNDARY = float(boundary)
DIR_EPS, NU_EPS, EPS_ACT = 1e-6, 1e-9, 1e-6
C_NU = 0.1
BIND_EPS = 1e-6
RHOS = (0.25, 0.5)


# ============== extended Block A with full duals ============================
def solve_full(gamma_acc, gamma_bh, demand, assoc, beta,
               W_U=W_U0, W_A=W_A30, tau=TAU, solver='CLARABEL',
               acc_rhs_eps=None):
    """acc_rhs_eps: optional (n_users,) additive eps on per-user access rate RHS
    (for finite-difference validation)."""
    nu = len(gamma_bh)
    out = dict(D_Mb=np.zeros(len(gamma_acc)), W_ui_Hz=np.zeros(len(gamma_acc)),
               W_ua_Hz=np.zeros(nu), R_access_Mb=np.zeros(len(gamma_acc)),
               R_bh_Mb=np.zeros(nu), S_u=np.zeros(nu),
               status='trivial-no-demand', U=0.0, n_active=0,
               nu_bh=np.zeros(nu), nu_bh_pool=0.0, nu_acc_pool=0.0,
               lam_acc=np.zeros(len(gamma_acc)), lam_dem=np.zeros(len(gamma_acc)))
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
    rhs = np.zeros(len(idx))
    if acc_rhs_eps is not None:
        rhs = acc_rhs_eps[idx]
    c_acc = D <= tau * rate_acc + rhs
    c_dem = D <= dem
    c_pa = cp.sum(W) <= W_U
    c_pb = cp.sum(Wb) <= W_A
    bhc = [cp.sum(D[bidx == u]) <= tau * rate_bh[u] for u in range(m)]
    prob = cp.Problem(cp.Maximize(b @ D), [c_acc, c_dem, c_pa, c_pb] + bhc)
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
    Wv = np.maximum(W.value, 0.0)
    Dv = np.maximum(D.value, 0.0)
    Wbv = np.maximum(Wb.value, 0.0)
    out['D_Mb'][idx] = Dv
    out['W_ui_Hz'][idx] = Wv * MHZ
    out['W_ua_Hz'][aidx] = Wbv * MHZ
    out['R_access_Mb'][idx] = tau * Wv * np.log2(1.0 + g_acc / np.maximum(Wv, 1e-12))
    out['R_bh_Mb'][aidx] = tau * Wbv * np.log2(1.0 + g_bh / np.maximum(Wbv, 1e-12))
    for u in range(m):
        out['S_u'][aidx[u]] = np.sum(Dv[bidx == u])
    out['status'] = str(prob.status)
    out['U'] = float(prob.value)
    out['n_active'] = int(len(idx))

    def dmax(c):
        d = c.dual_value
        if d is None:
            return 0.0
        return float(np.max(np.abs(np.atleast_1d(np.asarray(d, dtype=float)))))
    la = c_acc.dual_value
    if la is not None:
        out['lam_acc'][idx] = np.abs(np.asarray(la, dtype=float))
    ld_ = c_dem.dual_value
    if ld_ is not None:
        out['lam_dem'][idx] = np.abs(np.asarray(ld_, dtype=float))
    out['nu_acc_pool'] = dmax(c_pa)
    out['nu_bh_pool'] = dmax(c_pb)
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
        demand = np.maximum(z['D'][si] + z['dem_slack'][si], 0.0)
        out.append(dict(
            si=si, t=int(z['slots'][si]), beta=z['beta'][si].copy(),
            types=z['types'][si].copy(), cids=z['cids'][si].copy(),
            locs0=z['locs_dec'][si].copy(), air0=z['air_dec'][si].copy(),
            E0=z['E_dec'][si].copy(), A=z['A'][si].copy(), demand=demand,
            D_base=z['D'][si].copy(), ux=z['user_x'][si].copy(),
            uy=z['user_y'][si].copy(), qlen=float(z['qlen'][si]),
            nu_snap=z['nu'][si].copy(), eta_snap=z['eta'][si].copy()))
    return out


SNAP = load_snap()
print(f'loaded {len(SNAP)} snapshots', flush=True)


# =========================== TASK 1 =========================================
print('\n===== Task 1.1: G2 construction audit =====', flush=True)
rows_11 = []
sn0 = SNAP[0]
# reconstruct G2 exactly as scenario_audit did
rng = np.random.default_rng(sn0['si'])
ux2, uy2 = sn0['ux'].copy(), sn0['uy'].copy()
offsets = np.array([[-80, 40], [-80, 40], [70, -50], [70, -50]], float)
for u in range(4):
    m = sn0['cids'] == u
    ux2[m] += offsets[u, 0]
    uy2[m] += offsets[u, 1]
ux2 = np.clip(ux2, -249, 249)
uy2 = np.clip(uy2, -249, 249)
ga0, gb0, g0 = gammas(sn0['ux'], sn0['uy'], sn0['cids'], sn0['locs0'], sn0['air0'])
ga2, gb2, g2 = gammas(ux2, uy2, sn0['cids'], sn0['locs0'], sn0['air0'])
r0 = solve_full(ga0, gb0, sn0['demand'], sn0['cids'], sn0['beta'], W_A=W_A20)
r2 = solve_full(ga2, gb2, sn0['demand'], sn0['cids'], sn0['beta'], W_A=W_A20)
audit11 = dict(
    user_x_changed=bool(np.any(np.abs(ux2 - sn0['ux']) > 1e-9)),
    user_y_changed=bool(np.any(np.abs(uy2 - sn0['uy']) > 1e-9)),
    max_dx=float(np.max(np.abs(ux2 - sn0['ux']))),
    demand_same=True,   # G2 does not touch demand/beta by construction
    beta_same=True,
    cids_same=True,
    acc_gamma_changed=bool(np.any(np.abs(ga2 - ga0) > 1e-12)),
    acc_gamma_max_rel_diff=float(np.max(np.abs(ga2 - ga0) / np.maximum(np.abs(ga0), 1e-30))),
    bh_gamma_changed=bool(np.any(np.abs(gb2 - gb0) > 1e-12)),
    U0=r0['U'], U2=r2['U'],
    D_diff=float(np.max(np.abs(r2['D_Mb'] - r0['D_Mb']))),
    acc_bind0=float((r0['lam_acc'] > EPS_ACT)[
        (sn0['demand'] > 0)].mean()),
    acc_bind2=float((r2['lam_acc'] > EPS_ACT)[
        (sn0['demand'] > 0)].mean()),
    bh_pool_bind0=int(r0['nu_bh_pool'] > EPS_ACT),
    bh_pool_bind2=int(r2['nu_bh_pool'] > EPS_ACT),
)
print(audit11, flush=True)
# structural finding: G2 only shifts user coords; demand/beta/cids unchanged by design
# the question is whether channel inputs changed -> they must (access distance changes)
# and whether solution changed -> depends on binding constraints

print('\n===== Task 1.2/1.3: W_A=30 constraint stats =====', flush=True)
rows12 = []
t_wall = time.time()
for sn in SNAP:
    ga, gb, g = gammas(sn['ux'], sn['uy'], sn['cids'], sn['locs0'], sn['air0'])
    res = solve_full(ga, gb, sn['demand'], sn['cids'], sn['beta'], W_A=W_A30)
    active = (sn['demand'] > 0) & (sn['cids'] >= 0) & (sn['beta'] > 0)
    sum_W = float(res['W_ui_Hz'].sum()) / MHZ
    sum_Wb = float(res['W_ua_Hz'].sum()) / MHZ
    served = res['S_u'] > 1e-9
    bh_slack = res['R_bh_Mb'] - res['S_u']
    bh_bind_u = served & (bh_slack <= 1e-3 * np.maximum(res['R_bh_Mb'], 1e-9))
    acc_pos_user = active & (res['lam_acc'] > EPS_ACT)
    nu_bh_pos_u = res['nu_bh'] > EPS_ACT
    # A3: per-UAV simultaneous positive access-side & backhaul duals
    sim_u = []
    for u in range(4):
        m_u = sn['cids'] == u
        acc_pos_u = bool(np.any(acc_pos_user & m_u))       # any user of u has mu>0
        bh_pos_u = bool(nu_bh_pos_u[u] if u < len(nu_bh_pos_u) else False)
        sim_u.append(acc_pos_u and bh_pos_u)
    rows12.append(dict(
        si=sn['si'], t=sn['t'], status=res['status'],
        # A2 four constraint classes
        pool_acc_bind=int(sum_W >= W_U0 - 1e-4),
        pool_acc_util=sum_W / W_U0,
        nu_acc_pool=res['nu_acc_pool'],
        u_rate_bind=float(acc_pos_user.sum() / max(active.sum(), 1)),
        n_mu_pos=int(acc_pos_user.sum()),
        uav_bh_bind=float(bh_bind_u.sum() / max(served.sum(), 1)),
        nu_bh_pos_rate=float(nu_bh_pos_u[:4].mean()),
        pool_bh_bind=int(sum_Wb >= W_A30 - 1e-4),
        pool_bh_util=sum_Wb / W_A30,
        nu_bh_pool=res['nu_bh_pool'],
        # A3 simultaneous
        n_uav_simultaneous=int(sum(sim_u)),
        simultaneous_rate=float(np.mean(sim_u)),
        # extras
        U=res['U'], dem_cap_rate=float((active & (res['D_Mb'] >= sn['demand'] - 1e-6)).sum()
                                        / max(active.sum(), 1)),
        qlen=sn['qlen'],
    ))
df12 = pd.DataFrame(rows12)
df12.to_csv(os.path.join(OUT, 'task1_wa30_stats.csv'), index=False)
print(f'A2/A3 solved {len(SNAP)} in {time.time()-t_wall:.0f}s', flush=True)
print('\n--- A2: W_A=30 four constraint classes (all 152) ---', flush=True)
print(f"  access pool: bind_rate={df12['pool_acc_bind'].mean():.3f} "
      f"util={df12['pool_acc_util'].mean():.3f} nu>0={float((df12['nu_acc_pool']>EPS_ACT).mean()):.3f}", flush=True)
print(f"  per-user access rate: mean positive-mu frac={df12['u_rate_bind'].mean():.3f} "
      f"mean n_mu_pos={df12['n_mu_pos'].mean():.1f}", flush=True)
print(f"  per-UAV backhaul: bind_rate={df12['uav_bh_bind'].mean():.3f} "
      f"nu>0 rate={df12['nu_bh_pos_rate'].mean():.3f}", flush=True)
print(f"  backhaul pool: bind_rate={df12['pool_bh_bind'].mean():.3f} "
      f"util={df12['pool_bh_util'].mean():.3f} nu>0={float((df12['nu_bh_pool']>EPS_ACT).mean()):.3f}", flush=True)
print('\n--- A3: simultaneous positive duals (same snap, same UAV) ---', flush=True)
print(f"  mean simultaneous UAV fraction = {df12['simultaneous_rate'].mean():.3f}", flush=True)
print(f"  snapshots with >=1 simultaneous UAV = {(df12['n_uav_simultaneous']>0).mean():.3f}", flush=True)
print(f"  snapshots with all-4 simultaneous = {(df12['n_uav_simultaneous']==4).mean():.3f}", flush=True)
# coexistence at snapshot level (pool-level both positive)
both_pool = ((df12['nu_acc_pool'] > EPS_ACT) & (df12['nu_bh_pool'] > EPS_ACT)).mean()
print(f"  snapshot-level both pools positive dual = {both_pool:.3f}", flush=True)

print('\n===== Task 1.4: access dual mu finite-difference =====', flush=True)
# pick snapshots where some mu>0 at W_A=30; perturb that user access RHS by eps
fd_rows = []
n_checked = 0
for sn in SNAP:
    if n_checked >= 8:
        break
    ga, gb, g = gammas(sn['ux'], sn['uy'], sn['cids'], sn['locs0'], sn['air0'])
    res = solve_full(ga, gb, sn['demand'], sn['cids'], sn['beta'], W_A=W_A30)
    active = (sn['demand'] > 0) & (sn['cids'] >= 0) & (sn['beta'] > 0)
    cand = np.where(active & (res['lam_acc'] > EPS_ACT))[0]
    if len(cand) == 0:
        continue
    i = int(cand[np.argmax(res['lam_acc'][cand])])   # strongest mu
    eps = 1e-4
    eps_vec = np.zeros(len(ga))
    eps_vec[i] = eps
    res_p = solve_full(ga, gb, sn['demand'], sn['cids'], sn['beta'],
                       W_A=W_A30, acc_rhs_eps=eps_vec)
    if res_p['status'] not in ('optimal', 'optimal_inaccurate'):
        continue
    dU = res_p['U'] - res['U']
    fd_rows.append(dict(si=sn['si'], t=sn['t'], user=i,
                        mu=res['lam_acc'][i], dU=dU, eps=eps,
                        fd_slope=dU / eps,
                        type=int(sn['types'][i])))
    n_checked += 1
fd = pd.DataFrame(fd_rows)
if len(fd):
    fd.to_csv(os.path.join(OUT, 'task1_fd_mu.csv'), index=False)
    print(fd.to_string(index=False), flush=True)
    ratio = fd['fd_slope'] / fd['mu']
    print(f'\n  fd_slope / mu: mean={ratio.mean():.4f}  '
          f'min={ratio.min():.4f} max={ratio.max():.4f}  (KKT: ~1)', flush=True)
    print(f'  corr(fd, mu)={fd["fd_slope"].corr(fd["mu"]):.4f}', flush=True)
else:
    print('  no positive-mu snapshots found at W_A=30', flush=True)

print('\n===== Task 1.5: spectrum supply consistency =====', flush=True)
# static check from params: any joint budget? document
total_bw = W_U0 + W_A30
supplies = dict(
    access_pool_MHz=W_U0, backhaul_pool_MHz=W_A30,
    access_pool_param=bw_total_Hz / MHZ,
    backhaul_pool_param=bw_bh_total_Hz / MHZ,
    per_user_ref_MHz=bw_per_user / MHZ,
    per_uav_bh_ref_MHz=(bw_bh_total_Hz / n_uavs) / MHZ,
    type_pool_shares=bw_type_shares if 'bw_type_shares' in dir() else 'n/a',
    combined_if_same_band_MHz=total_bw,
    joint_budget_defined_in_params=False,
)
print(supplies, flush=True)
# check folder tag / docs for total spectrum assumption
with open(os.path.join(OUT, 'task1_spectrum.txt'), 'w', encoding='utf-8') as f:
    for k, v in supplies.items():
        f.write(f'{k}: {v}\n')


# =========================== TASK 2 =========================================
# representative snapshots: same stratified 40 as scenario_audit
rng = np.random.default_rng(0)
tmp = pd.read_csv(os.path.join(r'D:\Learn\work\drl-opencode\expB_results\scenario_audit\task1_audit.csv'))
tmp['qbin'] = pd.qcut(tmp['qlen'], 4, labels=False, duplicates='drop')
pick = tmp.groupby('qbin', group_keys=False).apply(
    lambda g: g.sample(n=min(10, len(g)), random_state=0))
pick_idx = set(pick['si'].tolist())
print(f'\n===== Task 2: 3 candidates at W_A=30 on {len(pick_idx)} snapshots =====', flush=True)


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
        if not np.any(m):
            continue
        bw = float(beta[m].sum())
        if bw > 1e-12:
            cent[u] = (beta[m, None] * uxy[m]).sum(0) / bw
    B_u = np.array([float(beta[(cids == u) & active].sum()) for u in range(4)])
    if B_u.sum() > 1e-12:
        at = (B_u[:, None] * q).sum(0) / B_u.sum()
    else:
        at = air0[:2].copy()
    d = cent - q
    n = np.linalg.norm(d, axis=1, keepdims=True)
    e_u = np.where(n > DIR_EPS, d / np.maximum(n, DIR_EPS), 0.0)
    da = at - air0[:2]
    na = float(np.linalg.norm(da))
    e_a = da / na if na > DIR_EPS else np.zeros(2)
    return e_u, e_a


def access_dirs(beta, types, cids, locs0, ux, uy, lam_acc):
    """direction toward mu-weighted centroid of access-bound users (probe)."""
    m = (lam_acc > EPS_ACT) & (beta > 0)
    e = np.zeros((4, 2))
    if not np.any(m):
        return e
    uxy = np.stack([ux, uy], 1)
    for u in range(4):
        mu = m & (cids == u)
        if not np.any(mu):
            continue
        w = lam_acc[mu]
        c = (w[:, None] * uxy[mu]).sum(0) / w.sum()
        d = c - locs0[u, :2]
        e[u] = unit(d)
    return e


def evalJ(locs, air, sn, E0, vxy, W_A=W_A30):
    ga, gb, g = gammas(sn['ux'], sn['uy'], sn['cids'], locs, air)
    res = solve_full(ga, gb, sn['demand'], sn['cids'], sn['beta'], W_A=W_A)
    P = np.array([float(cal_uav_propulsion_energy(V=np.linalg.norm(vxy[u]),
                                                  Vz=0.0)[0]) for u in range(4)])
    J = PSI_E * float(E0 @ P) - float(sn['beta'] @ res['D_Mb'])
    return dict(J=J, U=res['U'], P=P, res=res,
                D_sum=float(sn['beta'] @ res['D_Mb']))


def move_uav(locs0, e_u, rho, clip=True):
    locs = locs0.copy()
    vxy = SPD_XY * rho * e_u
    for u in range(4):
        locs[u, 0] = np.clip(locs0[u, 0] + vxy[u, 0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
        locs[u, 1] = np.clip(locs0[u, 1] + vxy[u, 1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    return locs, vxy


def move_air(air0, e_a, rho):
    air = air0.copy()
    va = SPD_XY * rho * e_a
    air[0] = np.clip(air[0] + va[0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    air[1] = np.clip(air[1] + va[1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    return air


rows2 = []
t_wall = time.time()
for sn in SNAP:
    if sn['si'] not in pick_idx:
        continue
    beta, types, cids = sn['beta'], sn['types'], sn['cids']
    locs0, air0, E0 = sn['locs0'], sn['air0'], sn['E0']
    active = (sn['demand'] > 0) & (cids >= 0) & (beta > 0)

    # base solve at 30 MHz (re-solve, do NOT reuse 20 MHz duals)
    ga0, gb0, g0 = gammas(sn['ux'], sn['uy'], cids, locs0, air0)
    base = solve_full(ga0, gb0, sn['demand'], cids, beta, W_A=W_A30)
    nu = np.maximum(base['nu_bh'], 0.0)
    lam = base['lam_acc'].copy()

    e_u_beta, e_a_beta = beta_dirs(beta, types, cids, locs0, air0, sn['ux'], sn['uy'])
    e_bh = np.array([unit(air0[:2] - locs0[u, :2]) for u in range(4)])
    binding = nu > BIND_EPS
    w_b = binding.astype(float)
    e_u_bind = np.array([unit((1 - w_b[u]) * e_u_beta[u] + w_b[u] * e_bh[u])
                         for u in range(4)])
    w_d = np.where(nu > NU_EPS, nu / (C_NU + nu), 0.0)
    e_u_dual = np.array([unit((1 - w_d[u]) * e_u_beta[u] + w_d[u] * e_bh[u])
                         for u in range(4)])
    e_u_acc = access_dirs(beta, types, cids, locs0, sn['ux'], sn['uy'], lam)

    # stay
    ev0 = evalJ(locs0.copy(), air0.copy(), sn, E0, np.zeros((4, 2)))
    J0 = ev0['J']

    # joint candidate sets (UAV-only; airship fixed)
    sets = {
        'k3': [('stay', 0, e_u_beta), ('b25', .25, e_u_beta), ('b50', .5, e_u_beta)],
        'bind': [('stay', 0, e_u_beta), ('b25', .25, e_u_beta), ('b50', .5, e_u_beta),
                 ('bd25', .25, e_u_bind), ('bd50', .5, e_u_bind)],
        'dual': [('stay', 0, e_u_beta), ('b25', .25, e_u_beta), ('b50', .5, e_u_beta),
                 ('du25', .25, e_u_dual), ('du50', .5, e_u_dual)],
    }
    best = {}
    for name, plan in sets.items():
        Jbest, tag, evb = np.inf, None, None
        for cname, rho, eu in plan:
            if rho == 0:
                ev = ev0
            else:
                locs, vxy = move_uav(locs0, eu, rho)
                ev = evalJ(locs, air0.copy(), sn, E0, vxy)
            if ev['J'] < Jbest:
                Jbest, tag, evb = ev['J'], cname, ev
        best[name] = dict(J=Jbest, tag=tag, ev=evb, dJ=J0 - Jbest)

    # airship-move variants: dual direction + airship along e_a (both rhos)
    Jair_best = best['dual']['J']
    air_tag = 'uav-only'
    for rho in RHOS:
        locs, vxy = move_uav(locs0, e_u_dual, rho)
        air2 = move_air(air0, e_a_beta, rho)
        ev = evalJ(locs, air2, sn, E0, vxy)
        if ev['J'] < Jair_best:
            Jair_best, air_tag = ev['J'], f'dual+air r{rho}'
    # also binding+air
    for rho in RHOS:
        locs, vxy = move_uav(locs0, e_u_bind, rho)
        air2 = move_air(air0, e_a_beta, rho)
        ev = evalJ(locs, air2, sn, E0, vxy)
        if ev['J'] < Jair_best:
            Jair_best, air_tag = ev['J'], f'bind+air r{rho}'

    # access-aware probe: move along mu-weighted direction (single best rho)
    Jacc_best = np.inf
    acc_tag = 'n/a'
    if np.any(np.linalg.norm(e_u_acc, axis=1) > DIR_EPS):
        for rho in RHOS:
            locs, vxy = move_uav(locs0, e_u_acc, rho)
            ev = evalJ(locs, air0.copy(), sn, E0, vxy)
            if ev['J'] < Jacc_best:
                Jacc_best, acc_tag = ev['J'], f'acc r{rho}'

    # constraint activity of dual-best candidate
    ev_du = best['dual']['ev']
    res_du = ev_du['res']
    sum_Wb_du = float(res_du['W_ua_Hz'].sum()) / MHZ
    acc_pos_du = (res_du['lam_acc'] > EPS_ACT) & active
    served_du = res_du['S_u'] > 1e-9
    bh_bind_du = served_du & ((res_du['R_bh_Mb'] - res_du['S_u'])
                              <= 1e-3 * np.maximum(res_du['R_bh_Mb'], 1e-9))
    sim_du = sum(1 for u in range(4)
                 if np.any(acc_pos_du & (cids == u)) and res_du['nu_bh'][u] > EPS_ACT)

    rows2.append(dict(
        si=sn['si'], t=sn['t'], qlen=sn['qlen'],
        J0=J0,
        dJ_k3=best['k3']['dJ'], k3_tag=best['k3']['tag'],
        dJ_bind=best['bind']['dJ'], bind_tag=best['bind']['tag'],
        dJ_dual=best['dual']['dJ'], dual_tag=best['dual']['tag'],
        bind_minus_dual=best['bind']['J'] - best['dual']['J'],
        # constraint state at 30 MHz (stay/base)
        base_pool_bh_bind=int(base['nu_bh_pool'] > EPS_ACT),
        base_pool_acc_bind=int(base['nu_acc_pool'] > EPS_ACT),
        base_n_mu=int((base['lam_acc'] > EPS_ACT)[active].sum()),
        base_n_nu=int((nu > EPS_ACT).sum()),
        base_sim_uav=sim_du,   # measured on dual-best candidate
        base_u_rate_bind=float((base['lam_acc'] > EPS_ACT)[active].mean()),
        base_dem_cap=float((active & (base['D_Mb'] >= sn['demand'] - 1e-6)).sum()
                           / max(active.sum(), 1)),
        # access-aware probe
        dJ_acc=(J0 - Jacc_best) if np.isfinite(Jacc_best) else np.nan,
        acc_tag=acc_tag,
        acc_beats_dual=bool(np.isfinite(Jacc_best) and Jacc_best < best['dual']['J'] - 1e-9),
        # airship
        dJ_dual_air=(J0 - Jair_best), air_tag=air_tag,
        air_gain_over_uav=(best['dual']['J'] - Jair_best),
        # energy of dual-best
        P0=float(ev0['P'].sum()), P_dual=float(ev_du['P'].sum()),
    ))
    if len(rows2) % 10 == 0:
        print(f'  {len(rows2)} done {time.time()-t_wall:.0f}s', flush=True)

df2 = pd.DataFrame(rows2)
df2.to_csv(os.path.join(OUT, 'task2_wa30_methods.csv'), index=False)
print(f'task2 done {len(df2)} snapshots in {time.time()-t_wall:.0f}s', flush=True)


def desc(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return 'n=0'
    return (f'n={len(x)} mean={x.mean():.4f} med={np.median(x):.4f} '
            f'p10={np.percentile(x,10):.4f} p90={np.percentile(x,90):.4f}')


print('\n--- dJ vs Stay (W_A=30) ---', flush=True)
print('  K3  :', desc(df2['dJ_k3']), flush=True)
print('  Bind:', desc(df2['dJ_bind']), flush=True)
print('  Dual:', desc(df2['dJ_dual']), flush=True)
print('  paired bind-dual (J_bind - J_dual; <0 => bind better):', desc(df2['bind_minus_dual']), flush=True)
print(f"  dual better rate = {(df2['bind_minus_dual'] > 1e-9).mean():.3f}  "
      f"bind better = {(df2['bind_minus_dual'] < -1e-9).mean():.3f}  "
      f"tie = {(np.abs(df2['bind_minus_dual']) <= 1e-9).mean():.3f}", flush=True)
print('\n--- constraint activity (dual-best candidate) ---', flush=True)
print(f"  base pool-bh bind={df2['base_pool_bh_bind'].mean():.3f}  "
      f"pool-acc bind={df2['base_pool_acc_bind'].mean():.3f}", flush=True)
print(f"  base mean n_mu_pos={df2['base_n_mu'].mean():.1f}  n_nu_pos={df2['base_n_nu'].mean():.2f}  "
      f"per-user mu frac={df2['base_u_rate_bind'].mean():.3f}", flush=True)
print(f"  dual-best simultaneous acc+bh UAVs: mean={df2['base_sim_uav'].mean():.2f}  "
      f"rate>0={(df2['base_sim_uav']>0).mean():.3f}", flush=True)
# performance conditional on coexistence
co = df2['base_sim_uav'] > 0
print(f"\n--- conditional on >=1 simultaneous UAV (n={co.sum()}) ---", flush=True)
print(f"  dJ_dual: {desc(df2.loc[co, 'dJ_dual'])}", flush=True)
print(f"  dJ_bind: {desc(df2.loc[co, 'dJ_bind'])}", flush=True)
print(f"  bind-dual: {desc(df2.loc[co, 'bind_minus_dual'])}", flush=True)
print(f"\n--- NOT simultaneous (n={(~co).sum()}) ---", flush=True)
print(f"  dJ_dual: {desc(df2.loc[~co, 'dJ_dual'])}", flush=True)
print(f"  bind-dual: {desc(df2.loc[~co, 'bind_minus_dual'])}", flush=True)
print('\n--- access-aware probe ---', flush=True)
print(f"  dJ_acc: {desc(df2['dJ_acc'])}", flush=True)
print(f"  acc beats dual rate: {df2['acc_beats_dual'].mean():.3f}  "
      f"mean dJ_acc - dJ_dual = {(df2['dJ_acc'] - df2['dJ_dual']).mean():.4f}", flush=True)
print('\n--- airship move at 30 MHz ---', flush=True)
print(f"  dJ_dual (UAV-only): {desc(df2['dJ_dual'])}", flush=True)
print(f"  dJ_dual+air       : {desc(df2['dJ_dual_air'])}", flush=True)
print(f"  air gain over uav-only: {desc(df2['air_gain_over_uav'])}", flush=True)
print(f"  air improves rate: {(df2['air_gain_over_uav'] > 1e-9).mean():.3f}", flush=True)

# plots
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
ax = axes[0, 0]
ax.boxplot([df2['dJ_k3'].dropna(), df2['dJ_bind'].dropna(), df2['dJ_dual'].dropna()],
           labels=['K3', 'Bind', 'Dual'], showfliers=False)
ax.set_ylabel(r'$\Delta J$ vs Stay'); ax.set_title('W_A=30 method gains'); ax.grid(alpha=0.3)
ax = axes[0, 1]
ax.hist(df2['bind_minus_dual'], bins=20, color='tab:purple', alpha=0.8)
ax.axvline(0, color='k', lw=0.8); ax.set_title('J_bind - J_dual'); ax.grid(alpha=0.3)
ax = axes[0, 2]
ax.scatter(df2['base_n_nu'], df2['dJ_dual'], s=16, alpha=0.6, c='tab:orange')
ax.set_xlabel('#nu>0 UAVs'); ax.set_ylabel(r'$\Delta J$ dual'); ax.grid(alpha=0.3)
ax.set_title('backhaul pressure vs gain')
ax = axes[1, 0]
ax.scatter(df2['base_n_mu'], df2['dJ_dual'], s=16, alpha=0.6, c='tab:blue')
ax.set_xlabel('#mu>0 users'); ax.set_ylabel(r'$\Delta J$ dual'); ax.grid(alpha=0.3)
ax.set_title('access pressure vs gain')
ax = axes[1, 1]
ok = np.isfinite(df2['dJ_acc'])
ax.scatter(df2.loc[ok, 'dJ_dual'], df2.loc[ok, 'dJ_acc'], s=16, alpha=0.6, c='tab:green')
lim = [min(df2.loc[ok, ['dJ_dual', 'dJ_acc']].min()),
       max(df2.loc[ok, ['dJ_dual', 'dJ_acc']].max()) + .1]
ax.plot(lim, lim, 'k--', lw=0.8); ax.set_xlabel(r'$\Delta J$ dual'); ax.set_ylabel(r'$\Delta J$ access-probe')
ax.set_title('access-aware probe vs dual'); ax.grid(alpha=0.3)
ax = axes[1, 2]
ax.scatter(df2['dJ_dual'], df2['dJ_dual_air'], s=16, alpha=0.6, c='tab:red')
lim = [min(df2['dJ_dual'].min(), df2['dJ_dual_air'].min()),
       max(df2['dJ_dual'].max(), df2['dJ_dual_air'].max()) + .1]
ax.plot(lim, lim, 'k--', lw=0.8); ax.set_xlabel(r'$\Delta J$ UAV-only'); ax.set_ylabel(r'$\Delta J$ +airship')
ax.set_title('airship move gain'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'task2_wa30.png'), dpi=130)
print(f'\nwrote {OUT}', flush=True)

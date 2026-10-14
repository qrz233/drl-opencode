# -*- coding: utf-8 -*-
'''Experiment B Stage 0: offline dual-guided candidate validation on the
152 Experiment-A snapshots (read-only; no production code touched).

Per snapshot (decision-time state, time-aligned as in expA):
  base solve at (locs_dec, air_dec) -> nu_u (per-UAV backhaul duals), Wb*,
    pool_bh_dual (logged only, NOT used in weights)
  nu_hat_u = max(nu_u, 0);  w_u = nu_hat/(c_nu + nu_hat), 0 if nu_hat < 1e-9
  UAV blend d_u = (1-w_u) e_u^beta + w_u e_u^bh; near-zero norm -> deterministic
    fallback (w>=0.5 -> pure BH, else pure beta); cancellations counted
  Airship primary: true backhaul-rate gradient
    g_a = sum_u nu_hat_u * grad_{q_a} R_bh,u(Wb*)   (xy only; zero -> stay)
  Airship heuristic control: nu-weighted centroid direction
  Candidates (all re-solve full Block A, original Lyapunov
    J = psi_E*(E.P) - beta.D):
    K5 dual   : [stay, beta-0.25, beta-0.5, dual-0.25, dual-0.5]
    K5 control: [stay, beta-0.1, beta-0.25, beta-0.5, beta-0.75]
    aux: UAV-only / airship-only / centroid-joint dual candidates (c_nu=0.1)
    c_nu sensitivity {0.05, 0.1, 0.2} on joint dual candidates
  Core metrics: dJ_gain = min(J0,J1,J2) - min(J0..J4); dual-selected rate.

Outputs -> D:\\Learn\\work\\drl-opencode\\expB_results
'''
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
OUT = r'D:\Learn\work\drl-opencode\expB_results'
os.makedirs(OUT, exist_ok=True)
EXPA = r'D:\Learn\work\drl-opencode\expA_results'

from params_sys import *
from utils_sys import cal_channel_fading_gain, cal_uav_propulsion_energy
import blocka_optimizer as B
from mobility_model import adjust_location, adjust_altitude

MHZ, LN2 = B.MHZ, B.LN2
BOUNDARY = float(boundary)
PSI_E = 1.0 / PW_MAX
TAU = float(slot_len)
SPD_XY, SPD_Z = float(uav_speedxy_max), float(uav_speedz_max)
SPD_AIR = 5.0                      # AIRSHIP_SPEED_MAX (joint_ac_controller.py)
DIR_EPS = 1e-6                     # direction / blend norm threshold
NU_EPS = 1e-9                      # dual treated as zero below this
C_NUS = (0.05, 0.1, 0.2)
C_MAIN = 0.1
RHOS = (0.25, 0.5)


# ----------------------- verbatim dual solver + gamma helpers (expA) ------
def solve_blocka_diag(gamma_acc_MHz, gamma_bh_MHz, demand_Mb, assoc, beta,
                      W_U_MHz=None, W_A_MHz=None, tau=TAU, solver='CLARABEL'):
    '''Verbatim copy of blocka_optimizer.solve_blocka + dual readout.'''
    n_users = len(gamma_acc_MHz)
    n_uavs = len(gamma_bh_MHz)
    W_U_MHz = bw_total_Hz / MHZ if W_U_MHz is None else W_U_MHz
    W_A_MHz = bw_bh_total_Hz / MHZ if W_A_MHz is None else W_A_MHz
    out = dict(D_Mb=np.zeros(n_users), R_bh_Mb=np.zeros(n_uavs),
               W_ua_Hz=np.zeros(n_uavs), status='trivial-no-demand',
               U=0.0, solver_used=None, nu_full=np.zeros(n_uavs),
               nu_raw=None, aidx=None, bh_slack=None)
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
    c0 = D <= tau * rate_acc
    constraints = [c0, D <= dem, cp.sum(W) <= W_U_MHz, cp.sum(Wb) <= W_A_MHz]
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
        except Exception as e:
            print(f'[blocka-diag] solver {sname} raised: {e!r}', flush=True)
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
    out['solver_used'] = used
    out['pool_bh_dual'] = float(constraints[3].dual_value)
    nu = [float(c.dual_value) if c.dual_value is not None else np.nan for c in bh_cons]
    out['nu_raw'] = np.array(nu)
    out['nu_full'][aidx] = np.array(nu)
    bh_rate_sol = Wb_sol * np.log2(1.0 + g_bh / np.maximum(Wb_sol, 1e-12))
    out['bh_slack'] = np.array([tau * bh_rate_sol[u] - np.sum(D_sol[bidx == u])
                                for u in range(m)])
    out['aidx'] = aidx
    return out


def channel_gammas(user_x, user_y, cids, uav_locs, airship_loc, beta, demand, types):
    demand = np.maximum(demand, 0)
    active = (demand > 0) & (types >= 0) & (beta > 0)
    g_acc = np.zeros(len(cids))
    for u in np.unique(cids[active]):
        ids_u = np.where(cids == u)[0]
        g_acc[ids_u] = cal_channel_fading_gain(user_x[ids_u], user_y[ids_u],
                                               uav_locs[u, 0], uav_locs[u, 1], uav_locs[u, 2])
    gamma_bh, _ = B.cal_gamma_backhaul_MHz(uav_locs, airship_loc)
    return B.cal_gamma_access_MHz(g_acc), gamma_bh


# ------------------------------------------------------------ 1. load data
snap = np.load(os.path.join(EXPA, 'snapshots.npz'))
S = len(snap['slots'])
print(f'loaded {S} snapshots', flush=True)

n_uavs = snap['locs_dec'].shape[1]
n_users = snap['beta'].shape[1]


# ------------------------------------------------- 2. per-snapshot engine
def beta_directions(beta, types, cids, locs0, air0):
    '''Mirror of JointACCritic.begin_slot centroid logic (xy only).'''
    active = (types >= 0) & (beta > 0)
    user_xy = np.stack([snap_user_x, snap_user_y], axis=1)
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


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > DIR_EPS else np.zeros_like(v)


def to_motion(a, locs0, air0):
    '''Mirror of JointACCritic._to_motion.'''
    bU = a[:3 * n_uavs].reshape(n_uavs, 3)
    bA = a[3 * n_uavs:]
    v = np.zeros((n_uavs, 3))
    for u in range(n_uavs):
        bxy = bU[u, :2]
        v[u, :2] = SPD_XY * bxy / max(1.0, np.linalg.norm(bxy))
        v[u, 2] = SPD_Z * bU[u, 2]
    locs = locs0 + TAU * v
    locs[:, 0] = np.clip(locs[:, 0], -BOUNDARY, BOUNDARY)
    locs[:, 1] = np.clip(locs[:, 1], -BOUNDARY, BOUNDARY)
    locs[:, 2] = np.clip(locs[:, 2], z_min, z_max)
    va = SPD_AIR * bA / max(1.0, float(np.linalg.norm(bA)))
    air = air0 + TAU * np.array([va[0], va[1], 0.0])
    air[0] = np.clip(air[0], -BOUNDARY, BOUNDARY)
    air[1] = np.clip(air[1], -BOUNDARY, BOUNDARY)
    air[2] = air0[2]
    return locs, v, air


def eval_J(a, ctx, locs0, air0, E0):
    '''Full Block A re-solve + original Lyapunov J at candidate positions.'''
    locs, v, air = to_motion(a, locs0, air0)
    P = np.array([float(cal_uav_propulsion_energy(V=np.linalg.norm(v[u, :2]),
                                                  Vz=v[u, 2])[0])
                  for u in range(n_uavs)])
    res = B.blocka_allocate(cids=ctx['cids'], user_x=ctx['user_x'],
                            user_y=ctx['user_y'], uav_locs=locs,
                            airship_loc=air, queue_Mb=ctx['Q_prev'],
                            traffic_Mb=ctx['A'], types=ctx['types'],
                            beta=ctx['beta'], tau=TAU)
    J = PSI_E * float(E0 @ P) - float(ctx['beta'] @ res['D_Mb'])
    return dict(J=J, P=P, res=res, locs=locs, air=air, v=v)


def airship_grad_direction(locs0, air0, nu_hat, Wb_MHz):
    '''g_a = sum_u nu_hat_u * grad_{q_a} R_bh,u (xy only); zero -> stay.'''
    d = locs0 - air0[None, :]                       # (M,3)
    d2 = np.sum(d ** 2, axis=1)
    kappa = ref_receive_pw * pTx_airship / (B.NOISE_PSD_BH * MHZ)
    g = kappa / d2                                   # gamma_bh MHz
    # dR/dgamma = tau/ln2 * Wb/(Wb+g);  dgamma/dq_a = 2*g/d2 * (q_u - q_a)
    coef = (TAU / LN2) * (Wb_MHz / np.maximum(Wb_MHz + g, 1e-12)) * (2.0 * g / d2)
    grad = (coef * nu_hat)[:, None] * d              # (M,3)
    g_a = grad.sum(axis=0)
    g_a[2] = 0.0
    return g_a


rows, anomalies = [], []
cancel_uav_total = 0
cancel_air_total = 0
n_uav_eval = 0

t_wall = time.time()
for si in range(S):
    t = int(snap['slots'][si])
    beta = snap['beta'][si].copy()
    types = snap['types'][si].copy()
    cids = snap['cids'][si].copy()
    locs0 = snap['locs_dec'][si].copy()
    air0 = snap['air_dec'][si].copy()
    E0 = snap['E_dec'][si].copy()
    A = snap['A'][si].copy()
    D_base_stored = snap['D'][si].copy()
    dem_slack = snap['dem_slack'][si].copy()
    snap_user_x = snap['user_x'][si]
    snap_user_y = snap['user_y'][si]

    # ---- time alignment: decision demand = D + dem_slack (base solve);
    # snapshot Q is POST-execution Q(t+1) and must not be used as Q_prev.
    demand = np.maximum(D_base_stored + dem_slack, 0.0)
    Q_prev = np.maximum(demand - np.maximum(A, 0.0), 0.0)
    ctx = dict(cids=cids, user_x=snap_user_x, user_y=snap_user_y,
               Q_prev=Q_prev, A=A, types=types, beta=beta)

    # ---- base solve at decision geometry (duals)
    g_acc, g_bh = channel_gammas(snap_user_x, snap_user_y, cids, locs0, air0,
                                 beta, demand, types)
    base = solve_blocka_diag(g_acc, g_bh, demand, cids, beta)
    # reproduction check 1: base vs stored
    u_diff = abs(base['U'] - float(snap['U0'][si]))
    d_diff = float(np.max(np.abs(base['D_Mb'] - D_base_stored)))
    nu_diff = float(np.max(np.abs(base['nu_full'] - snap['nu'][si]))) \
        if base['nu_raw'] is not None else np.nan
    base_ok = base['status'] in ('optimal', 'optimal_inaccurate', 'trivial-no-demand')
    if not base_ok or d_diff > 1e-5 or u_diff > 1e-4:
        anomalies.append(dict(kind='base-repro', slot=t, status=base['status'],
                              d_diff=d_diff, u_diff=u_diff))
    if base['solver_used'] not in (None, 'CLARABEL'):
        anomalies.append(dict(kind='solver-fallback', slot=t,
                              detail=f"base->{base['solver_used']}"))

    nu = base['nu_full'].copy()
    nu_hat = np.maximum(nu, 0.0)
    nu_hat[~np.isfinite(nu_hat)] = 0.0
    Wb_MHz = base['W_ua_Hz'] / MHZ
    pool_bh = base.get('pool_bh_dual', np.nan)

    # ---- directions
    e_u_beta, e_a_beta = beta_directions(beta, types, cids, locs0, air0)
    e_bh = np.zeros((n_uavs, 2))
    for u in range(n_uavs):
        e_bh[u] = unit(air0[:2] - locs0[u, :2])

    g_a = airship_grad_direction(locs0, air0, nu_hat, Wb_MHz)
    if np.linalg.norm(g_a[:2]) > DIR_EPS:
        e_a_grad = unit(g_a[:2])
    else:
        e_a_grad = np.zeros(2)
        cancel_air_total += 1

    if nu_hat.sum() > NU_EPS:
        cent_a = (nu_hat[:, None] * locs0[:, :2]).sum(axis=0) / nu_hat.sum()
        e_a_cent = unit(cent_a - air0[:2])
    else:
        e_a_cent = np.zeros(2)

    ang_gc = np.nan
    if np.linalg.norm(e_a_grad) > DIR_EPS and np.linalg.norm(e_a_cent) > DIR_EPS:
        c_ = float(np.clip(np.dot(e_a_grad, e_a_cent), -1.0, 1.0))
        ang_gc = float(np.degrees(np.arccos(c_)))

    # ---- per-c_nu UAV blend with deterministic cancellation fallback
    blend = {}
    for c in C_NUS:
        w = np.where(nu_hat > NU_EPS, nu_hat / (c + nu_hat), 0.0)
        d_u = (1.0 - w[:, None]) * e_u_beta + w[:, None] * e_bh
        nrm = np.linalg.norm(d_u, axis=1)
        for u in range(n_uavs):
            n_uav_eval += 1
            if nrm[u] < DIR_EPS:
                cancel_uav_total += 1
                d_u[u] = e_bh[u] if w[u] >= 0.5 else e_u_beta[u]
        e_u = np.zeros_like(d_u)
        for u in range(n_uavs):
            e_u[u] = unit(d_u[u])
        blend[c] = (w, e_u)

    w_main, e_u_dual = blend[C_MAIN]

    # ---- candidate actions (z comp 0 for all directed moves)
    def a_dir(e_u_, e_a_, rho):
        a = np.zeros(3 * n_uavs + 2)
        a[:3 * n_uavs] = np.hstack([rho * e_u_, np.zeros((n_uavs, 1))]).ravel()
        a[3 * n_uavs:] = rho * e_a_
        return a

    a0 = np.zeros(3 * n_uavs + 2)
    # K5 control amplitudes on pure-beta direction
    ctrl_rhos = (0.10, 0.25, 0.50, 0.75)
    # main K5 dual candidates (gradient airship)
    evals = {}
    evals['J0'] = eval_J(a0, ctx, locs0, air0, E0)          # stay (fresh solve)
    for rho, tag in ((0.25, 'b25'), (0.5, 'b50'), (0.1, 'b10'), (0.75, 'b75')):
        evals[f'J_{tag}'] = eval_J(a_dir(e_u_beta, e_a_beta, rho),
                                   ctx, locs0, air0, E0)
    dual_main = {}
    for c in C_NUS:
        for rho, tag in ((0.25, '25'), (0.5, '50')):
            key = f'J_d{tag}' if c == C_MAIN else f'J_d{tag}_c{str(c).replace(".", "")}'
            dual_main[(c, rho)] = eval_J(a_dir(blend[c][1], e_a_grad, rho),
                                         ctx, locs0, air0, E0)
            evals[key] = dual_main[(c, rho)]
    # aux decomposition (c_nu = 0.1)
    for rho, tag in ((0.25, '25'), (0.5, '50')):
        evals[f'J_u{tag}'] = eval_J(a_dir(e_u_dual, np.zeros(2), rho),
                                    ctx, locs0, air0, E0)
        evals[f'J_a{tag}'] = eval_J(a_dir(np.zeros((n_uavs, 2)), e_a_grad, rho),
                                    ctx, locs0, air0, E0)
        evals[f'J_g{tag}c'] = eval_J(a_dir(e_u_dual, e_a_cent, rho),
                                     ctx, locs0, air0, E0)

    # ---- solver / invalid accounting
    fail = sum(1 for k, ev in evals.items()
               if ev['res']['status'] not in ('optimal', 'optimal_inaccurate',
                                              'trivial-no-demand'))
    if fail:
        anomalies.append(dict(kind='cand-fail', slot=t, detail=f'{fail} solves'))

    J0 = evals['J0']['J']
    J_b = [evals['J_b25']['J'], evals['J_b50']['J']]
    J_ctrl = [J0] + [evals[f'J_b{int(r*100):02d}']['J'] for r in ctrl_rhos]
    k5_evs = [evals['J0'], evals['J_b25'], evals['J_b50'],
              dual_main[(C_MAIN, 0.25)], dual_main[(C_MAIN, 0.5)]]
    J_dual = [ev['J'] for ev in k5_evs]           # full K5: stay/beta/dual
    k3_evs = [evals['J0'], evals['J_b25'], evals['J_b50']]
    J_old = min(J0, *J_b)
    J_new = min(J_dual)
    J_ctrl_new = min(J_ctrl)
    dJ_gain = J_old - J_new
    dJ_gain_ctrl = J_old - J_ctrl_new
    win5 = int(np.argmin(J_dual))
    dual_selected = win5 in (3, 4)
    k3 = int(np.argmin([J0, *J_b]))
    aux_dual_vs_beta = min(J_b) - min(dual_main[(C_MAIN, 0.25)]['J'],
                                      dual_main[(C_MAIN, 0.5)]['J'])

    # decomposition gains (c_nu=0.1)
    g_uav = J_old - min(J0, evals['J_u25']['J'], evals['J_u50']['J'])
    g_air = J_old - min(J0, evals['J_a25']['J'], evals['J_a50']['J'])
    g_cent = J_old - min(J0, evals['J_g25c']['J'], evals['J_g50c']['J'])
    dJ_c005 = J_old - min(J0, dual_main[(0.05, 0.25)]['J'], dual_main[(0.05, 0.5)]['J'])
    dJ_c020 = J_old - min(J0, dual_main[(0.2, 0.25)]['J'], dual_main[(0.2, 0.5)]['J'])

    # ---- reproduction check 2: executed candidate among stay/beta candidates
    exec_locs = snap['locs_exec'][si]
    exec_air = snap['air_exec'][si]
    cand_pos = [(evals['J0']['locs'], evals['J0']['air']),
                (evals['J_b25']['locs'], evals['J_b25']['air']),
                (evals['J_b50']['locs'], evals['J_b50']['air'])]
    best_match, best_md = -1, np.inf
    for ki, (lc, ac) in enumerate(cand_pos):
        lc2 = np.array([[adjust_location(lc[u, 0], bound=boundary),
                         adjust_location(lc[u, 1], bound=boundary),
                         adjust_altitude(lc[u, 2], zmin=z_min, zmax=z_max)]
                        for u in range(n_uavs)])
        md = max(float(np.max(np.abs(lc2 - exec_locs))),
                 float(np.max(np.abs(ac - exec_air))))
        if md < best_md:
            best_md, best_match = md, ki
    exec_match = best_md < 1e-6
    if not exec_match:
        anomalies.append(dict(kind='exec-match', slot=t, detail=f'md={best_md:.3e}'))

    # ---- capacity / delivery / power deltas when dual candidate wins
    row = dict(slot=t, J0=J0, J_b10=evals['J_b10']['J'], J_b25=J_b[0],
               J_b50=J_b[1], J_b75=evals['J_b75']['J'],
               J_d25=dual_main[(C_MAIN, 0.25)]['J'],
               J_d50=dual_main[(C_MAIN, 0.5)]['J'],
               J_d25_c005=dual_main[(0.05, 0.25)]['J'],
               J_d50_c005=dual_main[(0.05, 0.5)]['J'],
               J_d25_c020=dual_main[(0.2, 0.25)]['J'],
               J_d50_c020=dual_main[(0.2, 0.5)]['J'],
               J_u25=evals['J_u25']['J'], J_u50=evals['J_u50']['J'],
               J_a25=evals['J_a25']['J'], J_a50=evals['J_a50']['J'],
               J_g25c=evals['J_g25c']['J'], J_g50c=evals['J_g50c']['J'],
               J_old=J_old, J_new=J_new, J_ctrl_new=J_ctrl_new,
               dJ_gain=dJ_gain, dJ_gain_ctrl=dJ_gain_ctrl,
               dual_selected=dual_selected, win5=win5, k3=k3,
               aux_dual_vs_beta=aux_dual_vs_beta,
               g_uav=g_uav, g_air=g_air, g_cent=g_cent,
               dJ_c005=dJ_c005, dJ_c020=dJ_c020,
               ang_gc=ang_gc, pool_bh=pool_bh,
               nu0=nu[0], nu1=nu[1], nu2=nu[2], nu3=nu[3],
               w0=w_main[0], w1=w_main[1], w2=w_main[2], w3=w_main[3],
               base_ok=base_ok, u_diff=u_diff, d_diff=d_diff, nu_diff=nu_diff,
               exec_match=exec_match, k_exec=best_match, cand_fail=fail)

    old_ev = k3_evs[k3]
    new_ev = k5_evs[win5]
    winner_changed = win5 != k3
    dR_bh = float(np.sum(new_ev['res']['R_bh_Mb'] - old_ev['res']['R_bh_Mb']))
    dR_acc = float(np.sum(new_ev['res']['R_access_Mb'] - old_ev['res']['R_access_Mb']))
    dU = float(beta @ (new_ev['res']['D_Mb'] - old_ev['res']['D_Mb']))
    dP = float(np.sum(new_ev['P'] - old_ev['P']))
    row.update(dR_bh=dR_bh, dR_acc=dR_acc, dU=dU, dP=dP,
               winner_changed=winner_changed,
               P_old=float(old_ev['P'].sum()), P_new=float(new_ev['P'].sum()),
               J_decomp_err=abs((dU - PSI_E * float(E0 @ (new_ev['P'] - old_ev['P'])))
                                - dJ_gain))
    rows.append(row)
    if (si + 1) % 30 == 0:
        print(f'  {si + 1}/{S}  {time.time() - t_wall:.0f}s', flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'stage0_slots.csv'), index=False)
anom_df = pd.DataFrame(anomalies)
anom_df.to_csv(os.path.join(OUT, 'stage0_anomalies.csv'), index=False)
print(f'done {S} snapshots in {time.time() - t_wall:.1f}s', flush=True)


# ------------------------------------------------------------- 3. summary
def describe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                max=float(x.max()))


g = df['dJ_gain'].to_numpy()
sel = df['dual_selected'].to_numpy(dtype=bool)
pos = g > 1e-6

print('\n================ Experiment B Stage 0 (offline, 152 snapshots) ================')
print(f'base repro: u_diff max={df.u_diff.max():.3e}  d_diff max={df.d_diff.max():.3e}  '
      f'nu_diff max={df.nu_diff.max():.3e}  all_ok={bool(df.base_ok.all())}')
print(f'exec match (stay/beta candidates reproduce controller move): '
      f'{df.exec_match.sum()}/{S} = {df.exec_match.mean()*100:.1f}%')
print(f'\n--- core: dJ_gain = J_old - J_new (K3 -> K5 dual) ---')
print(f'  mean={g.mean():.6f}  median={np.median(g):.6f}  p90={np.percentile(g,90):.6f}  '
      f'max={g.max():.6f}')
print(f'  frac(dJ>1e-6)={pos.mean()*100:.1f}%   frac(dJ>0.01)={(g>0.01).mean()*100:.1f}%   '
      f'frac(dJ>0.1)={(g>0.1).mean()*100:.1f}%')
print(f'  dual-selected (win in k3/k4): {sel.sum()}/{S} = {sel.mean()*100:.1f}%   '
      f'mean dJ | selected={g[sel].mean() if sel.any() else float("nan"):.6f}')
print(f'  negative dJ (dual hurts): {(g < -1e-6).sum()} slots, min={g.min():.2e}')
print(f'\n--- non-dual K5 control (extra beta amplitudes) ---')
gc = df['dJ_gain_ctrl'].to_numpy()
print(f'  ctrl dJ_gain mean={gc.mean():.6f}  frac>1e-6={(gc>1e-6).mean()*100:.1f}%')
print(f'  dual advantage over ctrl: mean={(g - gc).mean():.6f}  '
      f'frac(dual>ctrl)={((g - gc) > 1e-6).mean()*100:.1f}%')
print(f'\n--- auxiliary: dual vs beta head-to-head ---')
a = df['aux_dual_vs_beta'].to_numpy()
print(f'  min(J_dual)-min(J_beta) mean={a.mean():.6f}  frac(dual better)={(a>1e-6).mean()*100:.1f}%')
print(f'\n--- decomposition (c_nu=0.1) ---')
for col, name in (('g_uav', 'UAV-only'), ('g_air', 'airship-only(grad)'),
                  ('g_cent', 'joint w/ centroid airship')):
    v = df[col].to_numpy()
    print(f'  {name:24s}: mean dJ={v.mean():.6f}  frac>1e-6={(v>1e-6).mean()*100:.1f}%')
print(f'\n--- c_nu sensitivity ---')
for col, c in (('dJ_c005', 0.05), ('dJ_c020', 0.2)):
    v = df[col].to_numpy()
    print(f'  c_nu={c}: mean dJ={v.mean():.6f}  frac>1e-6={(v>1e-6).mean()*100:.1f}%')
print(f'  c_nu=0.1: mean dJ={g.mean():.6f}  frac>1e-6={(g>1e-6).mean()*100:.1f}%')
print(f'\n--- gradient vs centroid airship direction ---')
ang = df['ang_gc'].to_numpy()
angf = ang[np.isfinite(ang)]
print(f'  angle deg: median={np.median(angf):.1f}  p90={np.percentile(angf,90):.1f}  '
      f'frac(<30deg)={(angf<30).mean()*100:.1f}%  n_defined={len(angf)}/{S}')
J_cent_new = np.minimum(df.J0, np.minimum(df.J_g25c, df.J_g50c))
dJ_cent = df.J_old - J_cent_new
dJ_grad = df.J_old - df.J_new
print(f'  joint dJ: grad-airship mean={dJ_grad.mean():.6f}  '
      f'centroid-airship mean={dJ_cent.mean():.6f}  '
      f'frac(grad>cent)={((dJ_grad - dJ_cent) > 1e-6).mean()*100:.1f}%')
print(f'\n--- capacity / delivery / power ---')
print(f'  winner changed vs K3: {int(df.winner_changed.sum())}/{S} '
      f'(of which dual-selected: {int((df.winner_changed & sel).sum())})')
for mask, label in ((df.winner_changed.to_numpy(), 'all changed'),
                    (sel, 'dual-selected')):
    if mask.any():
        sub = df[mask]
        print(f'  [{label}] n={mask.sum()}: dR_bh mean={sub.dR_bh.mean():+.4f} Mb  '
              f'dR_acc mean={sub.dR_acc.mean():+.4f}  dU=beta*dD mean={sub.dU.mean():+.4f}  '
              f'dP mean={sub.dP.mean():+.3f} W')
        print(f'           J decomp err max={sub.J_decomp_err.max():.2e}')
print(f'\n--- robustness ---')
print(f'  direction cancellations: uav-blend {cancel_uav_total}/{n_uav_eval} = '
      f'{100*cancel_uav_total/max(n_uav_eval,1):.2f}%   '
      f'airship-grad-zero slots={cancel_air_total}')
print(f'  anomalies: {len(anom_df)}  (exec-match fails={(~df.exec_match).sum()}, '
      f'base fails={(~df.base_ok).sum()}, cand fails>0={(df.cand_fail>0).sum()})')

# decision helper
rec = dict(
    mean_dJ=float(g.mean()), frac_gain=float(pos.mean()),
    dual_sel=float(sel.mean()), ctrl_mean=float(gc.mean()),
    stage0_pass=bool(g.mean() > 0 and pos.mean() > 0.10 and (g - gc).mean() > 0),
)
print(f'\n>>> STAGE0 PASS (proceed to closed-loop): {rec["stage0_pass"]}')

# ------------------------------------------------------------- 4. figures
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.hist(g, bins=40, color='steelblue', edgecolor='k', alpha=0.85)
ax.axvline(0, color='r', ls='--', lw=1)
ax.set_xlabel(r'$\Delta J_{gain}=J_{old}-J_{new}$')
ax.set_ylabel('snapshots')
ax.set_title(f'dJ_gain  mean={g.mean():.4f}  frac>0={pos.mean()*100:.1f}%')

ax = axes[1]
ax.plot(df.slot, g, '.', ms=4, color='steelblue', label=r'$\Delta J_{gain}$')
m = sel
ax.plot(df.slot[m], g[m], 'o', ms=5, mfc='none', mec='darkorange',
        label=f'dual selected ({sel.mean()*100:.0f}%)')
ax.axhline(0, color='r', ls='--', lw=1)
ax.set_xlabel('slot')
ax.legend(fontsize=8)
ax.set_title('gain over trajectory')

ax = axes[2]
bins = np.linspace(0, 90, 25)
ax.hist(angf, bins=bins, color='seagreen', edgecolor='k', alpha=0.85)
ax.set_xlabel('angle grad vs centroid (deg)')
ax.set_ylabel('snapshots')
ax.set_title(f'airship dir agreement  median={np.median(angf):.0f}deg')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'stage0_results.png'), dpi=140)
print(f'\nwrote {OUT}\\stage0_slots.csv, stage0_anomalies.csv, stage0_results.png')

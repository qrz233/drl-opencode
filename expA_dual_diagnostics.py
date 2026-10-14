# -*- coding: utf-8 -*-
'''Experiment A: Block A dual-variable diagnostics (read-only, no control changes).

Replays the validated directed-only run (-sfm-wb-jac-nd, mode='directed') and at
every slot solves Block A at the NO-MOVEMENT (decision-time) geometry:
  1. extracts duals mu_{u,i} (access) and nu_u (backhaul),
  2. checks non-negativity / primal residuals / complementary slackness,
  3. validates against capacity-perturbation finite differences
     (backhaul delta = 0.1% / 1% of C^BH; access delta = 0.1% / 1% of cap).

No controller, constraint, beta, candidate or executed action is modified.
The dual-extracting solver is a verbatim copy of blocka_optimizer.solve_blocka
(+ optional RHS relaxation + dual readout); every slot it is checked against
the original solve_blocka for exact equivalence.

Outputs -> D:\\Learn\\work\\drl-opencode\\expA_results
'''
import os, sys, time, glob
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
OUT = r'D:\Learn\work\drl-opencode\expA_results'
os.makedirs(OUT, exist_ok=True)

from params_sys import *
from utils_sys import (kmeans_clustering, update_queue_Mb, find_active_users,
                       cal_channel_fading_gain)
import blocka_optimizer as B
from joint_ac_controller import JointACCritic

MHZ, LN2 = B.MHZ, B.LN2
TIGHT_TOL = 1e-6        # relative slack tolerance for "binding"
NU_TOL = 1e-8           # dual treated as zero below this
REL_SCALES = (0.001, 0.01)   # perturbation scales: 0.1% / 1% of capacity


# --------------------------------------------------------------- 1. inputs
def from_exel(fname):
    return np.transpose(pd.read_excel(fname, index_col=0).to_numpy())


sim_in = os.path.join(PROJ, 'simulation', 'sim-input')
tag_std = f"nusers={n_users}, nslots={n_slots}, ntype=3, lambda_Mb={lambd_Mb:.1f}, uON={ON_duration_mean_tslot:.0f}, uOFF={OFF_duration_mean_tslot:.0f}"
xaxis_all = from_exel(os.path.join(sim_in, f"xlocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx"))
yaxis_all = from_exel(os.path.join(sim_in, f"ylocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx"))
arrival = from_exel(os.path.join(sim_in, f"arrival traffic, dist=weibull, {tag_std}.xlsx"))
tr_type = from_exel(os.path.join(sim_in, f"traffic type, {tag_std}.xlsx"))
print('inputs loaded', xaxis_all.shape, arrival.shape, flush=True)


# ------------------------------- 2. dual-extracting copy of solve_blocka
def solve_blocka_diag(gamma_acc_MHz, gamma_bh_MHz, demand_Mb, assoc, beta,
                      W_U_MHz=None, W_A_MHz=None, tau=slot_len, solver='CLARABEL',
                      relax_bh=None, relax_acc=None):
    '''Verbatim copy of blocka_optimizer.solve_blocka + dual readout.

    relax_bh : dict {local_uav_index: delta_Mb} -> sum D <= tau*R_bh + delta
    relax_acc: dict {active_position: delta_Mb} -> D[j]  <= tau*R_acc + delta
    With relax_* None the constraint construction is identical to the original.
    '''
    n_users = len(gamma_acc_MHz)
    n_uavs = len(gamma_bh_MHz)
    W_U_MHz = bw_total_Hz / MHZ if W_U_MHz is None else W_U_MHz
    W_A_MHz = bw_bh_total_Hz / MHZ if W_A_MHz is None else W_A_MHz

    out = dict(
        W_ui_Hz=np.zeros(n_users), W_ua_Hz=np.zeros(n_uavs),
        D_Mb=np.zeros(n_users), R_access_Mb=np.zeros(n_users),
        R_bh_Mb=np.zeros(n_uavs), S_u=np.zeros(n_uavs), status='trivial-no-demand',
        U=0.0, solver_used=None,
        mu_full=np.zeros(n_users), nu_full=np.zeros(n_uavs),
        dem_full=np.zeros(n_users),
        mu_raw=None, nu_raw=None, idx=None, aidx=None,
        acc_slack=None, bh_slack=None, dem_slack=None,
        pool_acc_dual=None, pool_bh_dual=None,
        acc_resid=np.nan, bh_resid=np.nan, dem_resid=np.nan,
        cs_acc=np.nan, cs_bh=np.nan, dual_min=np.nan,
    )
    active = (demand_Mb > 0) & (assoc >= 0) & (beta > 0)
    if not np.any(active):
        return out

    idx = np.where(active)[0]
    n = len(idx)
    aidx = np.unique(assoc[idx])
    g_acc = gamma_acc_MHz[idx]
    dem = demand_Mb[idx]
    b = beta[idx]
    m = len(aidx)

    W = cp.Variable(n, nonneg=True)
    D = cp.Variable(n, nonneg=True)
    Wb = cp.Variable(m, nonneg=True)
    g_bh = gamma_bh_MHz[aidx]

    rate_acc = -cp.rel_entr(W, W + g_acc) / LN2
    rate_bh = -cp.rel_entr(Wb, Wb + g_bh) / LN2

    bidx = np.searchsorted(aidx, assoc[idx])
    assert np.array_equal(aidx[bidx], assoc[idx]), 'assoc/auidx mismatch'

    if relax_acc is not None and np.any(np.asarray(relax_acc) > 0):
        c0 = D <= tau * rate_acc + np.asarray(relax_acc, dtype=float)
    else:
        c0 = D <= tau * rate_acc
    constraints = [
        c0,                                     # access link (elementwise)
        D <= dem,                                # queue + arrival cap
        cp.sum(W) <= W_U_MHz,                    # shared access pool
        cp.sum(Wb) <= W_A_MHz,                   # shared backhaul pool
    ]
    bh_cons = []
    for u in range(m):
        d = float((relax_bh or {}).get(u, 0.0))
        bh_cons.append(cp.sum(D[bidx == u]) <= tau * rate_bh[u] + d)
    constraints += bh_cons

    prob = cp.Problem(cp.Maximize(b @ D), constraints)

    last_err, status, used = None, None, None
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
            last_err = e
            status = 'solver-error'
            print(f'[blocka-diag] solver {sname} raised: {e!r}', flush=True)
    if status not in ('optimal', 'optimal_inaccurate'):
        out['status'] = str(status)
        return out

    W_sol = np.maximum(W.value, 0.0)
    D_sol = np.maximum(D.value, 0.0)
    Wb_sol = np.maximum(Wb.value, 0.0)

    out['W_ui_Hz'][idx] = W_sol * MHZ
    out['D_Mb'][idx] = D_sol
    out['R_access_Mb'][idx] = tau * W_sol * np.log2(1.0 + g_acc / np.maximum(W_sol, 1e-12))
    out['W_ua_Hz'][aidx] = Wb_sol * MHZ
    out['R_bh_Mb'][aidx] = tau * Wb_sol * np.log2(1.0 + g_bh / np.maximum(Wb_sol, 1e-12))
    for u in range(m):
        out['S_u'][aidx[u]] = np.sum(D_sol[bidx == u])
    out['status'] = str(prob.status)
    out['U'] = float(prob.value)
    out['solver_used'] = used

    # ---------------- duals ----------------
    mu = np.asarray(c0.dual_value, dtype=float).ravel() if c0.dual_value is not None else None
    dem_d = np.asarray(constraints[1].dual_value, dtype=float).ravel()
    nu = [float(c.dual_value) if c.dual_value is not None else np.nan for c in bh_cons]
    out['pool_acc_dual'] = float(constraints[2].dual_value)
    out['pool_bh_dual'] = float(constraints[3].dual_value)
    if mu is not None:
        out['mu_raw'] = mu
        out['mu_full'][idx] = mu
        out['nu_raw'] = np.array(nu)
        out['nu_full'][aidx] = np.array(nu)
        out['dem_full'][idx] = dem_d
        dual_min = min(mu.min(), np.nanmin(nu))
        out['dual_min'] = float(dual_min)

        # primal residuals + complementary slackness at the solution
        acc_rate_sol = W_sol * np.log2(1.0 + g_acc / np.maximum(W_sol, 1e-12))
        bh_rate_sol = Wb_sol * np.log2(1.0 + g_bh / np.maximum(Wb_sol, 1e-12))
        out['acc_slack'] = tau * acc_rate_sol - D_sol          # (n,) >=0
        out['dem_slack'] = dem - D_sol
        out['bh_slack'] = np.array([tau * bh_rate_sol[u] - np.sum(D_sol[bidx == u])
                                    for u in range(m)])
        out['acc_resid'] = float(np.max(np.maximum(D_sol - tau * acc_rate_sol, 0)))
        out['dem_resid'] = float(np.max(np.maximum(D_sol - dem, 0)))
        out['bh_resid'] = float(np.max(np.abs(np.minimum(out['bh_slack'], 0))))
        out['cs_acc'] = float(np.max(np.abs(mu * out['acc_slack'])))
        out['cs_bh'] = float(np.max(np.abs(np.array(nu) * out['bh_slack'])))
    out['idx'], out['aidx'] = idx, aidx
    return out


def channel_gammas(user_x, user_y, cids, uav_locs, airship_loc, beta, demand, types):
    '''Same gamma construction as blocka_optimizer.blocka_allocate.'''
    demand = np.maximum(demand, 0)
    active = (demand > 0) & (types >= 0) & (beta > 0)
    g_acc = np.zeros(len(cids))
    for u in np.unique(cids[active]):
        ids_u = np.where(cids == u)[0]
        g_acc[ids_u] = cal_channel_fading_gain(user_x[ids_u], user_y[ids_u],
                                               uav_locs[u, 0], uav_locs[u, 1], uav_locs[u, 2])
    gamma_bh, _ = B.cal_gamma_backhaul_MHz(uav_locs, airship_loc)
    return B.cal_gamma_access_MHz(g_acc), gamma_bh


# ------------------------------------------------- 3. nd trajectory replay
from mobility_model import adjust_location, adjust_altitude

nd_dirs = glob.glob(os.path.join(PROJ, 'simulation', 'sim-output',
                                 '*K20-s1.1-sfm-wb-jac-nd, n_slots=*'))
assert nd_dirs, 'nd run output folder not found'
KPI_XLSX = os.path.join(sorted(nd_dirs)[-1], 'Proposed-kpis-over-time.xlsx')
assert os.path.exists(KPI_XLSX), f'missing {KPI_XLSX}'
kpi_ref = pd.read_excel(KPI_XLSX, index_col=0)
print('KPI reference:', KPI_XLSX, flush=True)

ctrl = JointACCritic(
    n_users=n_users, n_uavs=n_uavs, n_grids=int(2 * boundary / grid_size),
    ss_min_types=ss_min_types, ss_max_types=ss_max_types,
    mos_weights=mos_type_weights, type_rates=ON_data_arrival_mean_Mb, d0=d0,
    V_lyap=lyapunov_param, queue_norm=queue_norm_Mb,
    n_candidates=10, eval_sigma=0.5,
    airship_movable=True, mode='directed', seed=0)

S = int(n_slots)
prev_locs = np.column_stack([init_locations_uav[:, 0], init_locations_uav[:, 1],
                             np.full(n_uavs, float(z_init_proposed))])
airship_pos = np.array([airship_x, airship_y, airship_z], dtype=float)
Q = np.zeros(n_users)

# per-slot diagnostic arrays
mu_all = np.zeros((S, n_users))
acc_slack_all = np.zeros((S, n_users))
dem_slack_all = np.zeros((S, n_users))
D_diag_all = np.zeros((S, n_users))
beta_all = np.zeros((S, n_users))
dem_cap_all = np.zeros((S, n_users), dtype=bool)
nu_all = np.zeros((S, n_uavs))
C_all = np.zeros((S, n_uavs))
Su_all = np.zeros((S, n_uavs))
bh_slack_all = np.zeros((S, n_uavs))
eta_all = np.full((S, n_uavs), np.nan)
U0_all = np.full(S, np.nan)
base_status = np.empty(S, dtype='<U32')
base_solver = np.empty(S, dtype='<U16')
acc_resid_all = np.full(S, np.nan)
bh_resid_all = np.full(S, np.nan)
cs_acc_all = np.full(S, np.nan)
cs_bh_all = np.full(S, np.nan)
dual_min_all = np.full(S, np.nan)
eq_diff_all = np.full(S, np.nan)
eq_status = np.empty(S, dtype='<U32')
pool_acc_all = np.full(S, np.nan)
pool_bh_all = np.full(S, np.nan)
E_after_all = np.zeros((S, n_uavs))
cids_hist = np.zeros((S, n_users), dtype=np.int8)
qlen_replay = np.full(S, np.nan)
chcapa_replay = np.full(S, np.nan)

# layered snapshot slots: early / mid / late + forced t=20, t=440
snap_primary = sorted(set(list(range(4, 122, 2)) + list(range(150, 352, 5)) +
                          list(range(400, 501, 2)) + [20, 440]))
snap_slots = set(snap_primary)
snap_extra = set()          # anomaly-driven additions (capped)
SNAP_CAP_EXTRA = 40
snap = {}

bh_rows, ac_rows, anomalies = [], [], []


print('replaying nd trajectory + Block A diagnostics ...', flush=True)
t_wall = time.time()
for t in range(S):
    Q_prev = Q.copy()
    A_t = arrival[:, t]
    types_t = tr_type[:, t]
    isActive = find_active_users(queue_len_Mb=Q_prev, incoming_traffic_Mb=A_t)
    loc_users = np.hstack((xaxis_all[:, [t]], yaxis_all[:, [t]], np.zeros((n_users, 1))))
    cmat, cids, c_centroids = kmeans_clustering(loc_users)

    # ---------------- controller decision (mirror of NEW_CELL26) ------------
    airship_dec = airship_pos.copy()
    beta = ctrl.begin_slot(user_x=xaxis_all[:, t], user_y=yaxis_all[:, t],
                           Q=Q_prev, A=A_t, types=types_t, cids=cids,
                           uav_locs=prev_locs, airship_pos=airship_dec)
    E_dec = ctrl.E.copy()
    best = ctrl.decide_and_score(speed_xy=uav_speedxy_max, speed_z=uav_speedz_max)
    locs_exec = np.array([[adjust_location(best['uav_locs'][k, 0], bound=boundary),
                           adjust_location(best['uav_locs'][k, 1], bound=boundary),
                           adjust_altitude(best['uav_locs'][k, 2], zmin=z_min, zmax=z_max)]
                          for k in range(n_uavs)])
    airship_pos = best['airship_pos'].copy()
    ctrl.commit_slot(best)
    if (t >= t_training_start) and (t % training_interval == 0):
        ctrl.train_step()
    ctrl.update_energy(P_exec=best['P'])

    # executed outcome -> queue + KPI replay series (cell26 semantics)
    D_exec = best['res']['D_Mb']
    R_exec = best['res']['R_access_Mb']
    Q = np.maximum(Q_prev + A_t - D_exec, 0.0)
    n_act = int(isActive.sum())
    qlen_replay[t] = float(np.sum(Q[isActive]) / n_act) if n_act > 0 else np.nan
    chcapa_replay[t] = float(np.mean(R_exec))
    E_after_all[t] = ctrl.E
    cids_hist[t] = cids

    # ------------- Block A at decision-time (no-move) geometry --------------
    demand = np.maximum(Q_prev, 0) + np.maximum(A_t, 0)
    g_acc, g_bh = channel_gammas(xaxis_all[:, t], yaxis_all[:, t], cids,
                                 prev_locs, airship_dec, beta, demand, types_t)
    out0 = solve_blocka_diag(g_acc, g_bh, demand, cids, beta, tau=slot_len)
    ref = B.blocka_allocate(cids=cids, user_x=xaxis_all[:, t], user_y=yaxis_all[:, t],
                            uav_locs=prev_locs, airship_loc=airship_dec,
                            queue_Mb=Q_prev, traffic_Mb=A_t, types=types_t,
                            beta=beta, tau=slot_len)
    dmax = float(np.max(np.abs(out0['D_Mb'] - ref['D_Mb'])))
    eq_diff_all[t] = dmax
    eq_status[t] = ref['status']
    solvable_diag = out0['status'] in ('optimal', 'optimal_inaccurate', 'trivial-no-demand')
    solvable_ref = ref['status'] in ('optimal', 'optimal_inaccurate', 'trivial-no-demand')
    if dmax > 1e-6 or solvable_diag != solvable_ref:
        raise RuntimeError(f'equivalence FAILED at t={t}: dmax={dmax:.3e}, '
                           f'diag={out0["status"]}, ref={ref["status"]}')
    if out0['status'] != ref['status']:
        anomalies.append(('equiv-status', t, f'diag={out0["status"]} ref={ref["status"]}'))

    U0_all[t] = (out0['U'] if out0['status'] in ('optimal', 'optimal_inaccurate')
                 else (0.0 if out0['status'] == 'trivial-no-demand' else np.nan))
    base_status[t] = out0['status']
    base_solver[t] = out0['solver_used'] or ''
    if out0['solver_used'] not in (None, 'CLARABEL'):
        anomalies.append(('solver-fallback', t, f'base -> {out0["solver_used"]}'))
    if out0['status'] not in ('optimal', 'optimal_inaccurate', 'trivial-no-demand'):
        anomalies.append(('base-fail', t, out0['status']))
    if out0.get('dual_min', np.nan) is not None and np.isfinite(out0['dual_min']):
        if out0['dual_min'] < -1e-6:
            anomalies.append(('dual-negative', t, f'min={out0["dual_min"]:.2e}'))

    base_ok = out0['mu_raw'] is not None and np.isfinite(out0.get('U', np.nan))
    if base_ok:
        idx, aidx = out0['idx'], out0['aidx']
        m = len(aidx)
        mu_all[t] = out0['mu_full']
        acc_slack_all[t, idx] = out0['acc_slack']
        dem_slack_all[t, idx] = out0['dem_slack']
        D_diag_all[t] = out0['D_Mb']
        beta_all[t] = beta
        dem_cap_all[t, idx] = out0['dem_slack'] <= 1e-9
        nu_all[t, aidx] = out0['nu_raw']
        C_all[t, aidx] = out0['R_bh_Mb'][aidx]
        Su_all[t, aidx] = out0['S_u'][aidx]
        bh_slack_all[t, aidx] = out0['bh_slack']
        with np.errstate(divide='ignore', invalid='ignore'):
            eta_all[t, aidx] = np.where(out0['R_bh_Mb'][aidx] > 1e-12,
                                        out0['bh_slack'] / np.maximum(out0['R_bh_Mb'][aidx], 1e-12),
                                        np.nan)
        acc_resid_all[t] = out0['acc_resid']
        bh_resid_all[t] = out0['bh_resid']
        cs_acc_all[t] = out0['cs_acc']
        cs_bh_all[t] = out0['cs_bh']
        dual_min_all[t] = out0['dual_min']
        pool_acc_all[t] = out0['pool_acc_dual']
        pool_bh_all[t] = out0['pool_bh_dual']

        # ---------------- backhaul capacity perturbation tests --------------
        bidx = np.searchsorted(aidx, cids[idx])
        cnt_u = np.bincount(bidx, minlength=m)
        dem_u = np.bincount(bidx, weights=demand[idx], minlength=m)
        C_u = out0['R_bh_Mb'][aidx]
        S_u = out0['S_u'][aidx]
        sl_u = out0['bh_slack']
        eta_u = np.where(C_u > 1e-12, sl_u / np.maximum(C_u, 1e-12), np.nan)
        nu_u = np.asarray(out0['nu_raw'], dtype=float)
        for u in range(m):
            row0 = dict(slot=t, uav=int(aidx[u]), n_users=int(cnt_u[u]),
                        demand=float(dem_u[u]), C=float(C_u[u]), S=float(S_u[u]),
                        slack=float(sl_u[u]), eta=float(eta_u[u]), nu=float(nu_u[u]))
            if not np.isfinite(C_u[u]) or C_u[u] <= 1e-9:
                bh_rows.append({**row0, 'rel': np.nan, 'delta': np.nan, 'U0': U0_all[t],
                                'U2': np.nan, 'm_fd': np.nan, 'status': 'skipped-C0',
                                'solver': ''})
                continue
            for rel in REL_SCALES:
                delta = rel * float(C_u[u])
                r2 = solve_blocka_diag(g_acc, g_bh, demand, cids, beta, tau=slot_len,
                                       relax_bh={u: delta})
                ok2 = r2['mu_raw'] is not None and np.isfinite(r2.get('U', np.nan))
                m_fd = (r2['U'] - U0_all[t]) / delta if ok2 else np.nan
                bh_rows.append({**row0, 'rel': rel, 'delta': delta, 'U0': U0_all[t],
                                'U2': r2['U'] if ok2 else np.nan, 'm_fd': m_fd,
                                'status': r2['status'], 'solver': r2['solver_used'] or ''})
                if r2['status'] not in ('optimal', 'optimal_inaccurate'):
                    anomalies.append(('fd-bh-fail', t,
                                      f'u={aidx[u]} rel={rel} {r2["status"]}'))
                if r2['solver_used'] not in (None, 'CLARABEL'):
                    anomalies.append(('fd-bh-solver-fallback', t,
                                      f'u={aidx[u]} rel={rel} {r2["solver_used"]}'))

        # ---------------- access capacity perturbation tests ----------------
        cap = out0['R_access_Mb'][idx]
        sl = out0['acc_slack']
        dsl = out0['dem_slack']
        mu = out0['mu_raw']
        n_pos = len(idx)
        tight = (sl <= np.maximum(TIGHT_TOL * cap, 1e-12)) & (dsl > 1e-9) & (cap > 1e-9)
        loose = (sl > np.maximum(0.1 * cap, 1e-12)) & (dsl > 1e-9) & (cap > 1e-9)
        picks = []
        ti = np.where(tight)[0]
        if ti.size:
            order = ti[np.argsort(-mu[ti])]
            picks += [(int(p), 'tight') for p in order[:2]]
        picked = {p for p, _ in picks}
        si = np.array([p for p in np.where(loose)[0] if p not in picked])
        if si.size:
            picks.append((int(si[np.argsort(-sl[si])][0]), 'loose'))
        for pos, role in picks:
            i_glob = int(idx[pos])
            cap_i = float(cap[pos])
            row0 = dict(slot=t, user=i_glob, uav=int(cids[i_glob]), role=role,
                        mu=float(mu[pos]), cap=cap_i, D=float(out0['D_Mb'][idx[pos]]),
                        slack=float(sl[pos]), eta=float(sl[pos] / max(cap_i, 1e-12)),
                        dem=float(demand[i_glob]), demand_capped=bool(dsl[pos] <= 1e-9))
            for rel in REL_SCALES:
                delta = rel * cap_i
                dvec = np.zeros(n_pos)
                dvec[pos] = delta
                r2 = solve_blocka_diag(g_acc, g_bh, demand, cids, beta, tau=slot_len,
                                       relax_acc=dvec)
                ok2 = r2['mu_raw'] is not None and np.isfinite(r2.get('U', np.nan))
                m_fd = (r2['U'] - U0_all[t]) / delta if ok2 else np.nan
                ac_rows.append({**row0, 'rel': rel, 'delta': delta, 'U0': U0_all[t],
                                'U2': r2['U'] if ok2 else np.nan, 'm_fd': m_fd,
                                'status': r2['status'], 'solver': r2['solver_used'] or ''})
                if r2['status'] not in ('optimal', 'optimal_inaccurate'):
                    anomalies.append(('fd-ac-fail', t,
                                      f'i={i_glob} rel={rel} {r2["status"]}'))
                if r2['solver_used'] not in (None, 'CLARABEL'):
                    anomalies.append(('fd-ac-solver-fallback', t,
                                      f'i={i_glob} rel={rel} {r2["solver_used"]}'))

    # ---------------------------- snapshots ---------------------------------
    slot_anomaly = bool(anomalies) and anomalies[-1][1] == t
    if t in snap_slots or (slot_anomaly and len(snap_extra) < SNAP_CAP_EXTRA):
        if t not in snap_slots:
            snap_extra.add(t)
        snap[t] = dict(
            Q=Q.copy(), A=A_t.copy(), types=np.asarray(types_t).copy(),
            cids=np.asarray(cids).copy(), active=isActive.copy(),
            beta=beta.copy(), user_x=xaxis_all[:, t].copy(), user_y=yaxis_all[:, t].copy(),
            mu=mu_all[t].copy(), acc_slack=acc_slack_all[t].copy(),
            dem_slack=dem_slack_all[t].copy(), D=D_diag_all[t].copy(),
            nu=nu_all[t].copy(), C=C_all[t].copy(), S=Su_all[t].copy(),
            bh_slack=bh_slack_all[t].copy(), eta=eta_all[t].copy(),
            locs_dec=prev_locs.copy(), locs_exec=locs_exec.copy(),
            air_dec=airship_dec.copy(), air_exec=airship_pos.copy(),
            E_dec=E_dec.copy(), E_after=ctrl.E.copy(),
            U0=U0_all[t], solver=base_solver[t], status=base_status[t])

    prev_locs = locs_exec.copy()
    if (t + 1) % 100 == 0:
        print(f'  t={t + 1}/{S}  elapsed {time.time() - t_wall:.0f}s  '
              f'snapshots={len(snap)} anomalies={len(anomalies)}', flush=True)

print(f'replay done in {time.time() - t_wall:.1f}s, snapshots={len(snap)}', flush=True)


# ------------------------------------------- 4. replay fidelity vs nd KPIs
assert len(kpi_ref) == S, f'KPI rows {len(kpi_ref)} != {S}'
kpi_qlen = kpi_ref['qlen-Mb-avg (active)'].to_numpy(dtype=float)
kpi_chcapa = kpi_ref['chcapa-Mb-avg (active)'].to_numpy(dtype=float)
nan_match = bool(np.array_equal(np.isnan(qlen_replay), np.isnan(kpi_qlen)))
qlen_max = float(np.nanmax(np.abs(qlen_replay - kpi_qlen)))
chcapa_max = float(np.nanmax(np.abs(chcapa_replay - kpi_chcapa)))
FIDELITY_OK = nan_match and qlen_max <= 1e-6 and chcapa_max <= 1e-6
pd.DataFrame(dict(slot=np.arange(S), qlen_replay=qlen_replay, qlen_kpi=kpi_qlen,
                  qlen_absdiff=np.abs(qlen_replay - kpi_qlen),
                  chcapa_replay=chcapa_replay, chcapa_kpi=kpi_chcapa,
                  chcapa_absdiff=np.abs(chcapa_replay - kpi_chcapa))
             ).to_csv(os.path.join(OUT, 'replay_validation.csv'), index=False)
print(f'replay fidelity: qlen max|d|={qlen_max:.3e}  chcapa max|d|={chcapa_max:.3e}  '
      f'nan_match={nan_match}  -> {"PASS" if FIDELITY_OK else "FAIL"}', flush=True)
if not FIDELITY_OK:
    bad = np.where(np.nan_to_num(np.abs(qlen_replay - kpi_qlen), nan=0.0) > 1e-6)[0]
    print(f'first mismatched slots: {bad[:20]}')
    print('REPLAY DIVERGED from the recorded nd run -- aborting (results invalid).')
    sys.exit(2)


# ----------------------------------------------- 5. export tables + archives
bh_df = pd.DataFrame(bh_rows)
ac_df = pd.DataFrame(ac_rows)
bh_df.to_csv(os.path.join(OUT, 'bh_fd_tests.csv'), index=False)
ac_df.to_csv(os.path.join(OUT, 'ac_fd_tests.csv'), index=False)

anom_df = pd.DataFrame([dict(kind=k, slot=s, detail=d) for k, s, d in anomalies],
                       columns=['kind', 'slot', 'detail'])
anom_df.to_csv(os.path.join(OUT, 'anomalies.csv'), index=False)

np.savez_compressed(
    os.path.join(OUT, 'base_diagnostics.npz'),
    mu=mu_all, acc_slack=acc_slack_all, dem_slack=dem_slack_all, dem_cap=dem_cap_all,
    D=D_diag_all, beta=beta_all, nu=nu_all, C_bh=C_all, S_u=Su_all,
    bh_slack=bh_slack_all, eta=eta_all, U0=U0_all, base_status=base_status,
    base_solver=base_solver, acc_resid=acc_resid_all, bh_resid=bh_resid_all,
    cs_acc=cs_acc_all, cs_bh=cs_bh_all, dual_min=dual_min_all,
    eq_diff=eq_diff_all, eq_status=eq_status,
    pool_acc_dual=pool_acc_all, pool_bh_dual=pool_bh_all,
    E_after=E_after_all, cids=cids_hist, qlen_kpi=qlen_replay, chcapa_kpi=chcapa_replay)

if snap:
    ts = np.array(sorted(snap))
    keys = list(snap[int(ts[0])].keys())
    snapnpz = {k: np.stack([snap[int(t)][k] for t in ts]) for k in keys}
    snapnpz['slots'] = ts
    snapnpz['qlen'] = qlen_replay[ts]
    snapnpz['chcapa'] = chcapa_replay[ts]
    np.savez_compressed(os.path.join(OUT, 'snapshots.npz'), **snapnpz)
print(f'wrote tables + base_diagnostics.npz + snapshots.npz '
      f'({len(ts)} slots)', flush=True)


# ------------------------------------------------------------- 6. statistics
ok_st = np.isin(base_status, ['optimal', 'optimal_inaccurate'])
C_pos = C_all > 1e-9
ok = ok_st[:, None] & C_pos                      # evaluable backhaul constraints
tight = ok & (eta_all <= 1e-6) & ~np.isnan(eta_all)
nu_pos = nu_all > NU_TOL
U_ok = np.isfinite(U0_all)

n_eval = int(ok.sum())
n_tight = int(tight.sum())
n_tight_nupos = int((tight & nu_pos).sum())
n_tight_nuzero = int((tight & ~nu_pos).sum())
n_slack = int((ok & ~tight).sum())
n_slack_nuzero = int((ok & ~tight & ~nu_pos).sum())
n_slack_nupos = int((ok & ~tight & nu_pos).sum())
deg = tight & ~nu_pos                            # binding but zero dual (degenerate)

# access-side aggregates over ALL active users (base solve)
cap_all = acc_slack_all + D_diag_all
active_all = beta_all > 0
acc_eval = active_all & (cap_all > 1e-9)
acc_tight = acc_eval & (acc_slack_all <= np.maximum(TIGHT_TOL * cap_all, 1e-12))
mu_pos = mu_all > NU_TOL
n_acc_eval = int(acc_eval.sum())
n_acc_tight = int(acc_tight.sum())
n_acc_tight_mupos = int((acc_tight & mu_pos).sum())
n_acc_tight_muzero = int((acc_tight & ~mu_pos).sum())
n_acc_loose = int((acc_eval & (acc_slack_all > np.maximum(0.1 * cap_all, 1e-12))).sum())
n_acc_loose_mupos = int((acc_eval & (acc_slack_all > np.maximum(0.1 * cap_all, 1e-12))
                         & mu_pos).sum())

# degenerate-group composition: demand-capped share of that UAV's active users
frac_capped = []
for tt, uu in zip(*np.where(deg)):
    mask = (cids_hist[tt] == uu) & (beta_all[tt] > 0)
    if mask.any():
        frac_capped.append(float(dem_cap_all[tt, mask].mean()))
deg_frac_capped = float(np.mean(frac_capped)) if frac_capped else np.nan
deg_slots = np.any(deg, axis=1)
deg_pool_bh = float(np.nanmean(pool_bh_all[deg_slots])) if deg_slots.any() else np.nan

# solver / status health
base_status_counts = pd.Series(base_status).value_counts().to_dict()
base_solver_counts = pd.Series(base_solver[base_solver != '']).value_counts().to_dict()
anom_kinds = anom_df['kind'].value_counts().to_dict() if len(anom_df) else {}

# finite-difference agreement (backhaul)
fd = bh_df[np.isfinite(bh_df['m_fd']) & bh_df['status'].isin(['optimal', 'optimal_inaccurate'])]
fd1 = fd[np.isclose(fd['rel'], 0.01)]
fd01 = fd[np.isclose(fd['rel'], 0.001)]
bh_stats = {}
if len(fd1):
    rho_all, p_all = spearmanr(fd1['nu'], fd1['m_fd'])
    bt = fd1[(fd1['eta'] <= 1e-6) & np.isfinite(fd1['eta'])]
    if len(bt) > 2:
        rho_t, p_t = spearmanr(bt['nu'], bt['m_fd'])
    else:
        rho_t, p_t = np.nan, np.nan
    strong = fd1['nu'] > 1e-3
    rel_err = np.abs(fd1.loc[strong, 'm_fd'] - fd1.loc[strong, 'nu']) / fd1.loc[strong, 'nu']
    med_nu = fd1['nu'].median()
    hi = fd1[fd1['nu'] > med_nu]['m_fd'].mean()
    lo = fd1[fd1['nu'] <= med_nu]['m_fd'].mean()
    bh_stats = dict(n_fd=len(fd1), rho_all=float(rho_all), p_all=float(p_all),
                    rho_tight=float(rho_t), p_tight=float(p_t),
                    med_abs_err_tight=float(np.median(np.abs(bt['m_fd'] - bt['nu']))) if len(bt) else np.nan,
                    med_rel_err_strong=float(np.median(rel_err)) if strong.sum() else np.nan,
                    pct_within10pct=(float(np.mean(np.abs(bt['m_fd'] - bt['nu'])
                                                   <= 0.1 * np.maximum(bt['nu'], 1e-9))) * 100
                                     if len(bt) else np.nan),
                    m_high_nu=float(hi), m_low_nu=float(lo))
# perturbation-scale stability
piv = fd.pivot_table(index=['slot', 'uav'], columns='rel', values='m_fd')
if 0.01 in piv.columns and 0.001 in piv.columns:
    pv = piv.dropna()
    relch = np.abs(pv[0.01] - pv[0.001]) / np.maximum.reduce(
        [np.abs(pv[0.01]), np.abs(pv[0.001]), np.full(len(pv), 1e-12)])
    bh_stats['n_stable_pairs'] = int(len(pv))
    bh_stats['pct_unstable_rel>0.5'] = float(np.mean(relch > 0.5)) * 100
else:
    bh_stats['n_stable_pairs'] = 0
    bh_stats['pct_unstable_rel>0.5'] = np.nan

# finite-difference agreement (access)
fa = ac_df[np.isfinite(ac_df['m_fd']) & ac_df['status'].isin(['optimal', 'optimal_inaccurate'])]
fa_t = fa[fa['role'] == 'tight']
fa_t1 = fa_t[np.isclose(fa_t['rel'], 0.01)]
fa_lo = fa[fa['role'] == 'loose']
fa_l1 = fa_lo[np.isclose(fa_lo['rel'], 0.01)]
ac_stats = dict(n_fd_tight=len(fa_t1), n_fd_loose=len(fa_l1))
if len(fa_t1):
    if len(fa_t1) > 2:
        rho_m, p_m = spearmanr(fa_t1['mu'], fa_t1['m_fd'])
    else:
        rho_m, p_m = np.nan, np.nan
    strong_a = fa_t1['mu'] > 1e-3
    rel_err_a = (np.abs(fa_t1.loc[strong_a, 'm_fd'] - fa_t1.loc[strong_a, 'mu'])
                 / fa_t1.loc[strong_a, 'mu'])
    ac_stats.update(rho_tight=float(rho_m), p_tight=float(p_m),
                    med_rel_err_strong=float(np.median(rel_err_a)) if strong_a.sum() else np.nan,
                    slots_with_tight_pick=int(fa_t1['slot'].nunique()))
if len(fa_l1):
    ac_stats.update(mean_mu_loose=float(fa_l1['mu'].mean()),
                    max_mu_loose=float(fa_l1['mu'].max()),
                    mean_m_loose=float(fa_l1['m_fd'].mean()),
                    max_abs_m_loose=float(np.abs(fa_l1['m_fd']).max()))
    if len(fa_l1) > 2:
        rho_lo, p_lo = spearmanr(fa_l1['mu'], fa_l1['m_fd'])
        ac_stats.update(rho_loose=float(rho_lo), p_loose=float(p_lo))


# --------------------------------------------------- 7. summary table (csv/md)
def _r(v):
    return round(float(v), 6) if isinstance(v, (int, float, np.floating, np.integer)) else v


summary = []
def add(sec, k, v):
    summary.append(dict(section=sec, metric=k, value=_r(v)))

add('replay', 'fidelity_pass', int(FIDELITY_OK))
add('replay', 'qlen_max_absdiff', qlen_max)
add('replay', 'chcapa_max_absdiff', chcapa_max)
add('equivalence', 'max_abs_diff_D', float(np.nanmax(eq_diff_all)))
add('equivalence', 'status_mismatch_slots', int(np.sum(np.asarray(base_status) != np.asarray(eq_status))))
add('base_solve', 'n_slots', S)
for k, v in base_status_counts.items():
    add('base_solve', f'status={k}', int(v))
for k, v in base_solver_counts.items():
    add('base_solve', f'solver={k}', int(v))
add('base_solve', 'max_access_resid_Mb', float(np.nanmax(acc_resid_all)))
add('base_solve', 'max_backhaul_resid_Mb', float(np.nanmax(bh_resid_all)))
add('base_solve', 'max_demand_resid_Mb', float(np.nanmax(np.maximum(-dem_slack_all, 0))))
add('base_solve', 'max_complSlack_access', float(np.nanmax(cs_acc_all)))
add('base_solve', 'max_complSlack_backhaul', float(np.nanmax(cs_bh_all)))
add('dual_sanity', 'min_dual_overall', float(np.nanmin(dual_min_all)))
add('dual_sanity', 'n_slots_dual_negative', int(np.nansum(dual_min_all < -1e-6)))
add('backhaul', 'n_eval_pairs', n_eval)
add('backhaul', 'pct_tight', 100.0 * n_tight / max(n_eval, 1))
add('backhaul', 'pct_tight_nu_positive', 100.0 * n_tight_nupos / max(n_tight, 1))
add('backhaul', 'pct_tight_nu_zero_degenerate', 100.0 * n_tight_nuzero / max(n_tight, 1))
add('backhaul', 'pct_slack_nu_zero_expected', 100.0 * n_slack_nuzero / max(n_slack, 1))
add('backhaul', 'pct_slack_nu_positive_violation', 100.0 * n_slack_nupos / max(n_slack, 1))
add('backhaul', 'mean_utilisation_S_over_C', float(np.nanmean(np.where(ok, Su_all / np.maximum(C_all, 1e-12), np.nan))))
add('backhaul', 'degenerate_frac_demand_capped', deg_frac_capped)
add('backhaul', 'degenerate_mean_pool_bh_dual', deg_pool_bh)
for k, v in bh_stats.items():
    add('fd_backhaul', k, v)
sn = ok & ~tight & nu_pos            # slack constraints that carry a positive dual
add('backhaul', 'n_slack_nu_positive', int(sn.sum()))
if sn.any():
    add('backhaul', 'slack_nu_pos_eta_median', float(np.median(eta_all[sn])))
    add('backhaul', 'slack_nu_pos_eta_max', float(np.max(eta_all[sn])))
    add('backhaul', 'slack_nu_pos_nu_max', float(np.max(nu_all[sn])))
loose_arr = (acc_eval & ~acc_tight &
             (acc_slack_all > np.maximum(0.1 * cap_all, 1e-12)) & (dem_slack_all > 1e-9))
add('pools', 'max_pool_access_dual', float(np.nanmax(pool_acc_all)))
add('pools', 'pct_slots_pool_access_binding', float(100.0 * np.mean(pool_acc_all > 1e-8)))
add('pools', 'max_pool_backhaul_dual', float(np.nanmax(pool_bh_all)))
add('pools', 'pct_slots_pool_backhaul_binding', float(100.0 * np.mean(pool_bh_all > 1e-8)))
add('access', 'n_eval_users', n_acc_eval)
add('access', 'pct_tight', 100.0 * n_acc_tight / max(n_acc_eval, 1))
add('access', 'pct_tight_mu_positive', 100.0 * n_acc_tight_mupos / max(n_acc_tight, 1))
add('access', 'pct_tight_mu_zero_degenerate', 100.0 * n_acc_tight_muzero / max(n_acc_tight, 1))
add('access', 'pct_loose', 100.0 * n_acc_loose / max(n_acc_eval, 1))
add('access', 'pct_loose_mu_positive_violation', 100.0 * n_acc_loose_mupos / max(n_acc_loose, 1))
add('access', 'max_mu_on_loose_users', float(np.max(mu_all[loose_arr])) if loose_arr.any() else np.nan)
add('access', 'pct_active_demand_capped',
    100.0 * int((active_all & dem_cap_all).sum()) / max(int(active_all.sum()), 1))
for k, v in ac_stats.items():
    add('fd_access', k, v)
for k, v in anom_kinds.items():
    add('anomalies', k, int(v))

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUT, 'expA_summary.csv'), index=False)
with open(os.path.join(OUT, 'expA_summary.md'), 'w', encoding='utf-8') as f:
    f.write('# Experiment A - Block A dual diagnostics summary\n\n')
    sec = None
    for row in summary:
        if row['section'] != sec:
            sec = row['section']
            f.write(f'\n## {sec}\n\n| metric | value |\n|---|---|\n')
        f.write(f"| {row['metric']} | {row['value']} |\n")

print('\n================ Experiment A summary ================')
print(summary_df.to_string(index=False))
if len(anom_df):
    print('\n--- anomalies (all recorded) ---')
    print(anom_df.to_string(index=False))


# ------------------------------------------------------------- 8. figures
def _save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {name}', flush=True)


# fig 1: backhaul relative slack eta vs dual nu
fig, ax = plt.subplots(figsize=(6.4, 5.0))
Es, Ns = eta_all[ok & ~tight], nu_all[ok & ~tight]
Et, Nt = eta_all[tight], nu_all[tight]
ax.scatter(Es, Ns, s=6, c='0.75', alpha=0.5, linewidths=0, label=f'slack (n={len(Es)})')
ax.scatter(Et, Nt, s=10, c='tab:blue', alpha=0.45, linewidths=0, label=f'tight (n={len(Et)})')
ax.set_xscale('symlog', linthresh=1e-9)
ax.set_yscale('symlog', linthresh=1e-8)
ax.axhline(NU_TOL, color='tab:red', lw=0.8, ls='--', label=f'nu tol={NU_TOL:g}')
ax.axvline(1e-6, color='k', lw=0.8, ls=':', label='tight tol=1e-6')
ax.set_xlabel(r'backhaul relative slack  $(C^{BH}-S)/C^{BH}$')
ax.set_ylabel(r'backhaul dual  $\nu_u$')
ax.set_title('Backhaul binding vs dual (all slots x UAVs, decision geometry)')
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.3)
ax.tick_params(axis='x', labelrotation=45)
_save(fig, 'fig1_backhaul_eta_vs_nu.png')

# fig 2: nu vs finite-difference marginal value (backhaul)
fig, ax = plt.subplots(figsize=(6.2, 5.4))
if len(fd01):
    ax.scatter(fd01['nu'], fd01['m_fd'], s=12, c='tab:orange', alpha=0.5,
               marker='x', linewidths=0.8, label=r'$\delta$=0.1% of C')
if len(fd1):
    ax.scatter(fd1['nu'], fd1['m_fd'], s=16, c='tab:blue', alpha=0.45,
               label=r'$\delta$=1% of C')
if len(fd):
    lo_ = float(min(fd['nu'].min(), fd['m_fd'].min()))
    hi_ = float(max(fd['nu'].max(), fd['m_fd'].max()))
    pad = 0.05 * (hi_ - lo_ + 1e-9)
    lims = (lo_ - pad, hi_ + pad)
    ax.plot(lims, lims, 'k--', lw=1.0, label='y = x')
ax.set_xscale('symlog', linthresh=1e-7)
ax.set_yscale('symlog', linthresh=1e-7)
ax.set_xlabel(r'dual  $\nu_u$')
ax.set_ylabel(r'marginal value  $m=(U(\delta)-U_0)/\delta$')
ax.set_title('Backhaul: dual vs capacity-perturbation finite difference')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
txt = (f"all: rho={bh_stats.get('rho_all', np.nan):.3f}\n"
       f"tight: rho={bh_stats.get('rho_tight', np.nan):.3f}")
ax.text(0.97, 0.05, txt, transform=ax.transAxes, va='bottom', ha='right', fontsize=9,
        bbox=dict(fc='white', ec='0.7', alpha=0.9))
_save(fig, 'fig2_nu_vs_finite_difference.png')

# fig 3: mu vs finite-difference marginal value (access)
fig, ax = plt.subplots(figsize=(6.2, 5.4))
for role, col, mk in (('tight', 'tab:red', 'o'), ('loose', '0.6', 's')):
    for rel, ss in ((0.01, 18), (0.001, 9)):
        sub = fa[(fa['role'] == role) & np.isclose(fa['rel'], rel)]
        if len(sub):
            ax.scatter(sub['mu'], sub['m_fd'], s=ss, c=col, marker=mk,
                       alpha=0.55 if rel == 0.01 else 0.35,
                       linewidths=0, label=f'{role}, {rel * 100:g}%')
if len(fa):
    lo_ = float(min(fa['mu'].min(), fa['m_fd'].min()))
    hi_ = float(max(fa['mu'].max(), fa['m_fd'].max()))
    pad = 0.05 * (hi_ - lo_ + 1e-9)
    lims = (lo_ - pad, hi_ + pad)
    ax.plot(lims, lims, 'k--', lw=1.0, label='y = x')
ax.set_xscale('symlog', linthresh=1e-7)
ax.set_yscale('symlog', linthresh=1e-7)
ax.set_xlabel(r'access dual  $\mu_{u,i}$')
ax.set_ylabel(r'marginal value  $m=(U(\delta)-U_0)/\delta$')
ax.set_title('Access: dual vs capacity-perturbation finite difference')
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.3)
if ac_stats.get('n_fd_tight', 0) > 0:
    txt = f"tight: rho={ac_stats.get('rho_tight', np.nan):.3f}"
else:
    txt = 'no tight access pairs\n(access binds ~ never)'
ax.text(0.97, 0.05, txt, transform=ax.transAxes, va='bottom', ha='right', fontsize=9,
        bbox=dict(fc='white', ec='0.7', alpha=0.9))
_save(fig, 'fig3_mu_vs_finite_difference.png')

# fig 4: typical UAV time series
n_tight_per_uav = tight.sum(axis=0)
u_star = int(np.argmax(n_tight_per_uav))
dem_u_series = np.zeros((S, n_uavs))
for u in range(n_uavs):
    mask_u = active_all & (cids_hist == u)
    dem_u_series[:, u] = np.where(mask_u, D_diag_all + dem_slack_all, 0.0).sum(axis=1)
t_ax = np.arange(S)
fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
axs[0, 0].plot(t_ax, eta_all[:, u_star], lw=0.9, c='tab:blue')
axs[0, 0].axhline(1e-6, color='k', ls=':', lw=0.8)
axs[0, 0].scatter(t_ax[tight[:, u_star]], eta_all[tight[:, u_star], u_star],
                  s=4, c='tab:red', zorder=3)
axs[0, 0].set_ylabel(r'rel. slack $\eta_u$')
axs[0, 1].plot(t_ax, Su_all[:, u_star], lw=0.9, label=r'$S_u$ (delivered)')
axs[0, 1].plot(t_ax, C_all[:, u_star], lw=0.9, ls='--', c='tab:orange',
               label=r'$C^{BH}_u$ (capacity)')
axs[0, 1].set_ylabel('Mb / slot')
axs[0, 1].legend(fontsize=8)
axs[1, 0].plot(t_ax, nu_all[:, u_star], lw=0.9, c='tab:green')
axs[1, 0].axhline(NU_TOL, color='k', ls=':', lw=0.8)
axs[1, 0].set_yscale('symlog', linthresh=0.02)
axs[1, 0].set_ylabel(r'dual $\nu_u$ (symlog)')
axs[1, 0].set_xlabel('slot')
axs[1, 1].plot(t_ax, dem_u_series[:, u_star], lw=0.9, c='tab:purple',
               label='demand (Q+A)')
axs[1, 1].plot(t_ax, Su_all[:, u_star], lw=0.9, c='tab:blue', label='delivered')
axs[1, 1].set_ylabel('Mb')
axs[1, 1].set_xlabel('slot')
axs[1, 1].legend(fontsize=8)
fig.suptitle(f'UAV {u_star}: backhaul bottleneck diagnostics '
             f'(tight slots: {int(n_tight_per_uav[u_star])})')
fig.tight_layout()
_save(fig, 'fig4_uav_timeseries.png')

print('\nall outputs in', OUT, flush=True)



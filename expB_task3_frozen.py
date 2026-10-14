# -*- coding: utf-8
'''Experiment B Task 3 (Step 3): frozen-state candidate comparison on the
152 Experiment-A snapshots. No training, no queue update, no new simulation.

Phase 0: quick Hybrid-slow actor training on seed-0 closed loop (501 slots)
  -> save actor weights. Needed only so the Hybrid proposal network is
  non-trivial when queried offline on frozen states.

Experiment A - native candidates (same start state, same cids/beta):
  Hybrid K5: [stay, mu, mu+s*e1, mu+s*e2, mu+s*e3] with SLOW mapping (v=c)
  Dual   K5: [stay, beta-0.25, beta-0.5, dual-0.25, dual-0.5] native dual motion
  Unified scoring for BOTH: Block A re-solve + J = psi_E*(E0.P) - beta.D

Experiment B - unified speed, direction-only contrast:
  Hybrid horizontal direction e_h = unit(mu_xy); Dual e_d = unit(blend_xy)
  Common speeds s in {2.5, 7.5} m/s, heights FIXED, re-solve Block A.
  Compare dJ vs stay, weighted delivery, propulsion power.
  Zero-direction samples marked separately.

Outputs -> expB_results\\task3_frozen
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
import torch

PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
OUT = r'D:\Learn\work\drl-opencode\expB_results\task3_frozen'
os.makedirs(OUT, exist_ok=True)
EXPA = r'D:\Learn\work\drl-opencode\expA_results'

from params_sys import *
from utils_sys import (kmeans_clustering, find_active_users, update_queue_Mb,
                       cal_channel_fading_gain, cal_uav_propulsion_energy,
                       gen_heatmap, update_vqlen_prop)
from mobility_model import adjust_location, adjust_altitude
from relay_hybrid_slow_controller import HybridSlowConsistentController
import blocka_optimizer as B

MHZ, LN2 = B.MHZ, B.LN2
PSI_E = 1.0 / PW_MAX
TAU = float(slot_len)
SPD_XY = float(uav_speedxy_max)
SPD_Z = float(uav_speedz_max)
BOUNDARY = float(boundary)
DIR_EPS = 1e-6
NU_EPS = 1e-9
C_NU = 0.1
HY_N_CAND, HY_SIGMA = 20, 1.1
HYB_SEED = 0
HY_K5 = 5
COMMON_SPEEDS = (2.5, 7.5)


def from_exel(fname):
    return np.transpose(pd.read_excel(fname, index_col=0).to_numpy())


sim_in = os.path.join(PROJ, 'simulation', 'sim-input')
tag_base = (f"nusers={n_users}, nslots={n_slots}, ntype=3, "
            f"lambda_Mb={lambd_Mb:.1f}, uON={ON_duration_mean_tslot:.0f}, "
            f"uOFF={OFF_duration_mean_tslot:.0f}")
_frozen = dict(
    xaxis=from_exel(os.path.join(sim_in, f"xlocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx")),
    yaxis=from_exel(os.path.join(sim_in, f"ylocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx")),
    arrival=from_exel(os.path.join(sim_in, f"arrival traffic, dist=weibull, {tag_base}.xlsx")),
    tr_type=from_exel(os.path.join(sim_in, f"traffic type, {tag_base}.xlsx")).astype(np.int64),
    tr_state=from_exel(os.path.join(sim_in, f"traffic state, {tag_base}.xlsx")).astype(bool),
)


def precompute_cids(exo, seed):
    np.random.seed(seed)
    xaxis_all, yaxis_all = exo['xaxis'], exo['yaxis']
    cids_all = np.zeros((n_slots, n_users), dtype=int)
    for t in range(n_slots):
        loc_users = np.hstack((xaxis_all[:, [t]], yaxis_all[:, [t]],
                               np.zeros((n_users, 1))))
        _, cids, _ = kmeans_clustering(loc_users)
        cids_all[t] = cids
    return cids_all


# ======================== Phase 0: Hybrid actor pre-training =================
ACTOR_PATH = os.path.join(OUT, 'hybrid_slow_actor_seed0.pt')


def train_and_save_actor():
    if os.path.exists(ACTOR_PATH):
        print(f'[phase0] actor already at {ACTOR_PATH}, skip training', flush=True)
        return
    print('[phase0] training Hybrid-slow actor (seed 0, 501 slots)...', flush=True)
    np.random.seed(1000)
    torch.manual_seed(HYB_SEED)
    ctrl = HybridSlowConsistentController(
        n_users=n_users, n_uavs=n_uavs, boundary=boundary, grid_size=grid_size,
        queue_norm=queue_norm_Mb, V_lyap=lyapunov_param, tau=slot_len,
        ss_min_types=ss_min_types, ss_max_types=ss_max_types,
        mos_weights=mos_type_weights, type_rates=ON_data_arrival_mean_Mb, d0=d0,
        sigma_decay_slots=n_slots, seed=HYB_SEED,
        n_candidates=HY_N_CAND, eval_sigma=HY_SIGMA)
    ctrl.bind_models(cal_channel_fading_gain, cal_backhaul_capacity_Mb,
                     pTx_downlink, noise_pw_total)
    exo = _frozen
    cids_all = precompute_cids(exo, 0)
    Q = np.zeros(n_users)
    vqlen_prop = np.zeros(n_uavs)
    uav_locs = np.column_stack([init_locations_uav[:, 0], init_locations_uav[:, 1],
                                np.full(n_uavs, float(z_init_proposed))])
    airship_pos = np.array([airship_x, airship_y, airship_z], float)
    xaxis_all, yaxis_all = exo['xaxis'], exo['yaxis']
    arrival, tr_type = exo['arrival'], exo['tr_type']
    t_wall = time.time()
    for t in range(n_slots):
        Q_prev = Q.copy()
        A_t = arrival[:, t]
        types_t = tr_type[:, t]
        isActive = find_active_users(queue_len_Mb=Q_prev, incoming_traffic_Mb=A_t)
        cids = cids_all[t]
        bw_alloc_Hz_all = np.zeros(n_users)
        for k_type in range(len(bw_type_pool_Hz)):
            ids_k = np.where(types_t == k_type)[0]
            active_k = ids_k[isActive[ids_k]]
            if len(active_k) > 0:
                bw_alloc_Hz_all[active_k] = bw_type_pool_Hz[k_type] / len(active_k)
        ld_Mbps = np.array([0 if types_t[i] == -1 else ON_data_arrival_mean_Mb[types_t[i]]
                            for i in range(n_users)])
        heatmap_t = gen_heatmap(xaxis_all[:, t], yaxis_all[:, t],
                                Q_prev + A_t, queue_norm_Mb)
        ctrl.begin_slot(heatmap_t, uav_locs, airship_pos)
        new_locs = np.zeros_like(uav_locs)
        for k in range(n_uavs):
            best_a, _ = ctrl.decide_and_score(
                k=k, uav_loc=uav_locs[k], airship_loc=airship_pos,
                heatmap=heatmap_t, user_x=xaxis_all[:, t], user_y=yaxis_all[:, t],
                bw_alloc_Hz=bw_alloc_Hz_all, Q=Q_prev, A=A_t, ld=ld_Mbps,
                types=types_t, cids=cids,
                speed_xy=uav_speedxy_max, speed_z=uav_speedz_max,
                zmin=z_min, zmax=z_max, boundary=boundary)
            pos_t, vel_t = ctrl.motion(best_a, uav_locs[k])
            new_locs[k] = [adjust_location(pos_t[0], bound=boundary),
                           adjust_location(pos_t[1], bound=boundary),
                           adjust_altitude(pos_t[2], zmin=z_min, zmax=z_max)]
            vqlen_prop[k] = update_vqlen_prop(
                vqlen_prop[k], cal_uav_propulsion_energy(V=np.hypot(vel_t[0], vel_t[1]), Vz=vel_t[2])[0]) if t > 0 else 0.0
        if (t >= t_training_start) and (t % training_interval == 0):
            ctrl.train_step()
        R_access = np.zeros(n_users)
        for k in range(n_uavs):
            ids_k = np.where(cids == k)[0]
            if len(ids_k) == 0:
                continue
            g_k = cal_channel_fading_gain(xaxis_all[ids_k, t], yaxis_all[ids_k, t],
                                          new_locs[k, 0], new_locs[k, 1], new_locs[k, 2])
            R_access[ids_k] = bw_alloc_Hz_all[ids_k] * np.log2(
                1 + pTx_downlink * g_k / noise_pw_total) * slot_len / 1e6
        R_bh = np.array([cal_backhaul_capacity_Mb(new_locs[k, 0], new_locs[k, 1], new_locs[k, 2])
                         for k in range(n_uavs)])
        D_opt, S_u = ctrl.solve_delivery(
            R_access=R_access, R_backhaul=R_bh, Q=Q_prev, A=A_t,
            cids=cids, active=isActive, ld=ld_Mbps, types=types_t)
        for uid in range(n_users):
            Q[uid] = update_queue_Mb(Q_prev[uid], A_t[uid], D_opt[uid])
        reward_t = ctrl.compute_reward(Q=Q_prev, A=A_t, D=D_opt, S_u=S_u,
                                       R_bh=R_bh, mos_mean=np.nan)
        ctrl.commit_slot(reward_t)
        ctrl.update_virtual_queues(S_u=S_u, R_bh=R_bh)
        uav_locs = new_locs
        if (t + 1) % 100 == 0:
            print(f'  t={t+1}/{n_slots}  {time.time()-t_wall:.0f}s', flush=True)
    torch.save(ctrl.actor.state_dict(), ACTOR_PATH)
    print(f'[phase0] saved actor -> {ACTOR_PATH}  ({time.time()-t_wall:.0f}s)', flush=True)


# ======================== shared offline Block A engine ======================
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


def eval_at_positions(locs, air, ctx, E0, v_xy=None, v_z=None):
    '''Block A re-solve + unified J at given UAV/airship positions.
    Propulsion power computed from actual displacement (or supplied v).'''
    if v_xy is None:
        v_xy = (locs[:, :2] - ctx['locs0'][:, :2]) / TAU
    if v_z is None:
        v_z = (locs[:, 2] - ctx['locs0'][:, 2]) / TAU
    P = np.array([float(cal_uav_propulsion_energy(V=np.linalg.norm(v_xy[u]), Vz=v_z[u])[0])
                  for u in range(n_uavs)])
    g_acc, g_bh = channel_gammas(ctx['user_x'], ctx['user_y'], ctx['cids'], locs, air)
    res = solve_blocka_diag(g_acc, g_bh, ctx['demand'], ctx['cids'], ctx['beta'])
    J = PSI_E * float(E0 @ P) - float(ctx['beta'] @ res['D_Mb'])
    return dict(J=J, P=P, res=res, locs=locs, air=air,
                D_sum=float(ctx['beta'] @ res['D_Mb']))


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > DIR_EPS else np.zeros_like(v)


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


# ============================ load snapshots =================================
snap = np.load(os.path.join(EXPA, 'snapshots.npz'))
S_N = len(snap['slots'])
print(f'loaded {S_N} frozen snapshots', flush=True)

# ============================ Phase 0: actor ================================
train_and_save_actor()
actor_sd = torch.load(ACTOR_PATH, map_location='cpu', weights_only=True)
probe = HybridSlowConsistentController(
    n_users=n_users, n_uavs=n_uavs, boundary=boundary, grid_size=grid_size,
    queue_norm=queue_norm_Mb, V_lyap=lyapunov_param, tau=slot_len,
    ss_min_types=ss_min_types, ss_max_types=ss_max_types,
    mos_weights=mos_type_weights, type_rates=ON_data_arrival_mean_Mb, d0=d0,
    sigma_decay_slots=n_slots, seed=HYB_SEED,
    n_candidates=HY_N_CAND, eval_sigma=HY_SIGMA)
probe.actor.load_state_dict(actor_sd)
probe.actor.eval()


# ============================ Experiment A ==================================
rows_A = []
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
    demand = np.maximum(D_base + dem_slack, 0.0)
    Q_prev = np.maximum(demand - np.maximum(A, 0.0), 0.0)
    ctx = dict(cids=cids, user_x=ux, user_y=uy, Q_prev=Q_prev, A=A,
               types=types, beta=beta, demand=demand, locs0=locs0, air0=air0)

    # Dual directions
    e_u_beta, e_a_beta = beta_directions(beta, types, cids, locs0, air0, ux, uy)
    g_acc0, g_bh0 = channel_gammas(ux, uy, cids, locs0, air0)
    base = solve_blocka_diag(g_acc0, g_bh0, demand, cids, beta)
    nu = base['nu_full'].copy()
    nu_hat = np.maximum(nu, 0.0)
    nu_hat[~np.isfinite(nu_hat)] = 0.0
    w = np.where(nu_hat > NU_EPS, nu_hat / (C_NU + nu_hat), 0.0)
    e_bh = np.array([unit(air0[:2] - locs0[u, :2]) for u in range(n_uavs)])
    d_u = (1.0 - w[:, None]) * e_u_beta + w[:, None] * e_bh
    e_u_dual = np.zeros_like(d_u)
    for u in range(n_uavs):
        if np.linalg.norm(d_u[u]) < DIR_EPS:
            d_u[u] = e_bh[u] if w[u] >= 0.5 else e_u_beta[u]
        e_u_dual[u] = unit(d_u[u])

    # Hybrid mu from trained actor (Z=0); actor is per-UAV
    heatmap_t = gen_heatmap(ux, uy, Q_prev + A, queue_norm_Mb)
    probe.Z = np.zeros(n_uavs)
    mu = np.stack([probe._forward_single(locs0[u], air0, heatmap_t)
                   for u in range(n_uavs)])

    def eval_action_hybrid(a_raw):
        locs = locs0.copy()
        v_xy = np.zeros((n_uavs, 2))
        v_z = np.zeros(n_uavs)
        for u in range(n_uavs):
            c = a_raw[u]
            nxy = float(np.linalg.norm(c[:2]))
            if nxy > SPD_XY:
                v_xy[u] = SPD_XY * c[:2] / nxy
            else:
                v_xy[u] = c[:2]
            v_z[u] = SPD_Z * c[2] if abs(c[2]) > SPD_Z else c[2]
            locs[u, 0] = np.clip(locs0[u, 0] + v_xy[u, 0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
            locs[u, 1] = np.clip(locs0[u, 1] + v_xy[u, 1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
            locs[u, 2] = np.clip(locs0[u, 2] + v_z[u] * TAU, z_min, z_max)
        return eval_at_positions(locs, air0.copy(), ctx, E0, v_xy=v_xy, v_z=v_z)

    def eval_action_dual(a_u, rho, e_a):
        locs = locs0.copy()
        v_xy = SPD_XY * rho * a_u
        v_z = np.zeros(n_uavs)
        for u in range(n_uavs):
            locs[u, 0] = np.clip(locs0[u, 0] + v_xy[u, 0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
            locs[u, 1] = np.clip(locs0[u, 1] + v_xy[u, 1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
        air = air0.copy()
        va = SPD_XY * rho * e_a
        air[0] = np.clip(air0[0] + va[0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
        air[1] = np.clip(air0[1] + va[1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
        return eval_at_positions(locs, air, ctx, E0, v_xy=v_xy, v_z=v_z)

    rng = np.random.default_rng(10000 + si)
    hy_cands = [np.zeros((n_uavs, 3)), mu.copy()]
    for j in range(HY_K5 - 2):
        hy_cands.append(mu + HY_SIGMA * rng.standard_normal((n_uavs, 3)))
    hy_evals = [eval_action_hybrid(a) for a in hy_cands]
    J_hy = [e['J'] for e in hy_evals]
    hy_best = int(np.argmin(J_hy))
    J_hy_stay, J_hy_best = J_hy[0], J_hy[hy_best]

    du_specs = [('stay', 0.0, e_u_beta, e_a_beta),
                ('b25', 0.25, e_u_beta, e_a_beta),
                ('b50', 0.50, e_u_beta, e_a_beta),
                ('d25', 0.25, e_u_dual, e_a_beta),
                ('d50', 0.50, e_u_dual, e_a_beta)]
    du_evals = [eval_action_dual(eu, rho, ea) for _, rho, eu, ea in du_specs]
    J_du = [e['J'] for e in du_evals]
    du_best = int(np.argmin(J_du))
    J_du_stay, J_du_best = J_du[0], J_du[du_best]

    J_old = min(J_du[0], J_du[1], J_du[2])
    dJ_hy = J_hy_stay - J_hy_best
    dJ_du = J_du_stay - J_du_best
    dJ_hy_vs_beta = J_old - J_hy_best
    dJ_du_vs_beta = J_old - J_du_best

    best_hy_c = hy_cands[hy_best]
    hy_vxy = np.zeros((n_uavs, 2))
    for u in range(n_uavs):
        nxy = float(np.linalg.norm(best_hy_c[u, :2]))
        hy_vxy[u] = best_hy_c[u, :2] if nxy <= SPD_XY else SPD_XY * best_hy_c[u, :2] / nxy

    rows_A.append(dict(
        slot=t,
        J_hy_stay=J_hy_stay, J_hy_best=J_hy_best, hy_best_idx=hy_best,
        J_du_stay=J_du_stay, J_du_best=J_du_best, du_best_idx=du_best,
        J_old_betaK3=J_old,
        dJ_hy=dJ_hy, dJ_du=dJ_du,
        dJ_hy_vs_beta=dJ_hy_vs_beta, dJ_du_vs_beta=dJ_du_vs_beta,
        dual_selected=int(du_best in (3, 4)),
        hy_D_sum=hy_evals[hy_best]['D_sum'], du_D_sum=du_evals[du_best]['D_sum'],
        hy_P_sum=float(hy_evals[hy_best]['P'].sum()),
        du_P_sum=float(du_evals[du_best]['P'].sum()),
        hy_vxy_mean=float(np.mean(np.linalg.norm(hy_vxy, axis=1))),
        du_vxy_mean=float(SPD_XY * du_specs[du_best][1]),
        zero_dir_hy=int(np.all(np.linalg.norm(mu[:, :2], axis=1) < DIR_EPS)),
        zero_dir_du=int(np.all(np.linalg.norm(e_u_dual, axis=1) < DIR_EPS)),
    ))
    if (si + 1) % 30 == 0:
        print(f'  ExpA {si+1}/{S_N}  {time.time()-t_wall:.0f}s', flush=True)

dfA = pd.DataFrame(rows_A)
dfA.to_csv(os.path.join(OUT, 'expA_native_slots.csv'), index=False)
print(f'[ExpA] done {S_N} snapshots in {time.time()-t_wall:.0f}s', flush=True)


# ============================ Experiment B ==================================
rows_B = []
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
    demand = np.maximum(D_base + dem_slack, 0.0)
    Q_prev = np.maximum(demand - np.maximum(A, 0.0), 0.0)
    ctx = dict(cids=cids, user_x=ux, user_y=uy, Q_prev=Q_prev, A=A,
               types=types, beta=beta, demand=demand, locs0=locs0, air0=air0)

    e_u_beta, e_a_beta = beta_directions(beta, types, cids, locs0, air0, ux, uy)
    g_acc0, g_bh0 = channel_gammas(ux, uy, cids, locs0, air0)
    base = solve_blocka_diag(g_acc0, g_bh0, demand, cids, beta)
    nu = base['nu_full'].copy()
    nu_hat = np.maximum(nu, 0.0)
    nu_hat[~np.isfinite(nu_hat)] = 0.0
    w = np.where(nu_hat > NU_EPS, nu_hat / (C_NU + nu_hat), 0.0)
    e_bh = np.array([unit(air0[:2] - locs0[u, :2]) for u in range(n_uavs)])
    d_u = (1.0 - w[:, None]) * e_u_beta + w[:, None] * e_bh
    e_u_dual = np.zeros_like(d_u)
    for u in range(n_uavs):
        if np.linalg.norm(d_u[u]) < DIR_EPS:
            d_u[u] = e_bh[u] if w[u] >= 0.5 else e_u_beta[u]
        e_u_dual[u] = unit(d_u[u])

    heatmap_t = gen_heatmap(ux, uy, Q_prev + A, queue_norm_Mb)
    probe.Z = np.zeros(n_uavs)
    mu = np.stack([probe._forward_single(locs0[u], air0, heatmap_t)
                   for u in range(n_uavs)])
    e_u_hyb = np.zeros((n_uavs, 2))
    zero_hyb = np.ones(n_uavs, dtype=bool)
    for u in range(n_uavs):
        nxy = float(np.linalg.norm(mu[u, :2]))
        if nxy > DIR_EPS:
            e_u_hyb[u] = mu[u, :2] / nxy
            zero_hyb[u] = False

    ev_stay = eval_at_positions(locs0.copy(), air0.copy(), ctx, E0,
                                v_xy=np.zeros((n_uavs, 2)), v_z=np.zeros(n_uavs))
    J0 = ev_stay['J']

    for speed in COMMON_SPEEDS:
        for tag, e_dir, zmask in (
                ('hyb', e_u_hyb, zero_hyb),
                ('dual', e_u_dual, np.linalg.norm(e_u_dual, axis=1) < DIR_EPS)):
            if np.all(zmask):
                rows_B.append(dict(slot=t, speed=speed, method=tag,
                                   zero_dir=1, dJ=np.nan, D_sum=ev_stay['D_sum'],
                                   P_sum=float(ev_stay['P'].sum()),
                                   n_moved=0))
                continue
            locs = locs0.copy()
            v_xy = speed * e_dir          # (M,2), height fixed
            for u in range(n_uavs):
                locs[u, 0] = np.clip(locs0[u, 0] + v_xy[u, 0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
                locs[u, 1] = np.clip(locs0[u, 1] + v_xy[u, 1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
            ev = eval_at_positions(locs, air0.copy(), ctx, E0,
                                   v_xy=v_xy, v_z=np.zeros(n_uavs))
            rows_B.append(dict(slot=t, speed=speed, method=tag,
                               zero_dir=int(np.all(zmask)),
                               dJ=J0 - ev['J'], D_sum=ev['D_sum'],
                               P_sum=float(ev['P'].sum()),
                               n_moved=int((~zmask).sum())))
    if (si + 1) % 30 == 0:
        print(f'  ExpB {si+1}/{S_N}  {time.time()-t_wall:.0f}s', flush=True)

dfB = pd.DataFrame(rows_B)
dfB.to_csv(os.path.join(OUT, 'expB_unified_speed_slots.csv'), index=False)
print(f'[ExpB] done {S_N} snapshots in {time.time()-t_wall:.0f}s', flush=True)

# ============================ Summary =======================================
def describe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, median=np.nan)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)))

print('\n===== Experiment A summary =====', flush=True)
print('Hybrid K5  best-J gain vs Stay:', describe(dfA['dJ_hy']), flush=True)
print('Dual   K5  best-J gain vs Stay:', describe(dfA['dJ_du']), flush=True)
print('Hybrid K5  best-J gain vs betaK3:', describe(dfA['dJ_hy_vs_beta']), flush=True)
print('Dual   K5  best-J gain vs betaK3:', describe(dfA['dJ_du_vs_beta']), flush=True)
print(f"dual_selected rate: {dfA['dual_selected'].mean():.3f}", flush=True)
print(f"Hybrid zero-dir rate: {dfA['zero_dir_hy'].mean():.3f}  "
      f"Dual zero-dir rate: {dfA['zero_dir_du'].mean():.3f}", flush=True)
print(f"Hybrid win-vs-dual (dJ_hy_vs_beta > dJ_du_vs_beta): "
      f"{(dfA['dJ_hy_vs_beta'] > dfA['dJ_du_vs_beta']).mean():.3f}", flush=True)

print('\n===== Experiment B summary =====', flush=True)
for speed in COMMON_SPEEDS:
    for meth in ('hyb', 'dual'):
        sub = dfB[(dfB['speed'] == speed) & (dfB['method'] == meth)]
        nz = sub[sub['zero_dir'] == 0]
        print(f'speed={speed} {meth}: nonzero n={len(nz)}  dJ={describe(nz["dJ"])}  '
              f'D_sum={describe(nz["D_sum"])}  P_sum={describe(nz["P_sum"])}', flush=True)

# paired comparison at each speed (non-zero both sides)
for speed in COMMON_SPEEDS:
    h = dfB[(dfB['speed'] == speed) & (dfB['method'] == 'hyb')].set_index('slot')
    d = dfB[(dfB['speed'] == speed) & (dfB['method'] == 'dual')].set_index('slot')
    common = h.index[(h['zero_dir'] == 0) & (d['zero_dir'] == 0)]
    if len(common) == 0:
        continue
    dh = h.loc[common, 'dJ']
    dd = d.loc[common, 'dJ']
    print(f'speed={speed} paired n={len(common)}: '
          f'hyb mean dJ={dh.mean():.6f}  dual mean dJ={dd.mean():.6f}  '
          f'dual better rate={(dd > dh).mean():.3f}', flush=True)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
bp_data = [dfA['dJ_hy'].dropna(), dfA['dJ_du'].dropna()]
ax.boxplot(bp_data, labels=['Hybrid K5', 'Dual K5'], showfliers=False)
ax.set_ylabel(r'$\Delta J$ vs Stay')
ax.set_title('Exp A: native K5 (unified J)')
ax.grid(alpha=0.3)

ax = axes[1]
for meth, color in (('hyb', 'tab:blue'), ('dual', 'tab:orange')):
    means, stds, speeds = [], [], []
    for speed in COMMON_SPEEDS:
        sub = dfB[(dfB['speed'] == speed) & (dfB['method'] == meth) & (dfB['zero_dir'] == 0)]
        means.append(sub['dJ'].mean())
        stds.append(sub['dJ'].std())
        speeds.append(speed)
    ax.errorbar(speeds, means, yerr=stds, marker='o', capsize=4, label=meth, color=color)
ax.set_xlabel('common speed (m/s)')
ax.set_ylabel(r'$\Delta J$ vs Stay')
ax.set_title('Exp B: direction-only contrast')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
for meth, color in (('hyb', 'tab:blue'), ('dual', 'tab:orange')):
    means, stds, speeds = [], [], []
    for speed in COMMON_SPEEDS:
        sub = dfB[(dfB['speed'] == speed) & (dfB['method'] == meth) & (dfB['zero_dir'] == 0)]
        means.append(sub['D_sum'].mean())
        stds.append(sub['D_sum'].std())
        speeds.append(speed)
    ax.errorbar(speeds, means, yerr=stds, marker='s', capsize=4, label=meth, color=color)
ax.set_xlabel('common speed (m/s)')
ax.set_ylabel(r'$\beta \cdot D$')
ax.set_title('Exp B: weighted delivery')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'task3_frozen.png'), dpi=140)
print(f'\nwrote {OUT}', flush=True)

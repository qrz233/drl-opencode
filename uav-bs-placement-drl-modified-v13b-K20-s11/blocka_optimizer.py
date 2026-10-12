"""
Variable-bandwidth resource allocation (BLOCKA).

Implements the convex program P_R from the user's document (BLOCKA求解.md):

    max  sum_u sum_{i in I_u} beta_i * D_{u,i}
    s.t. D_{u,i} <= tau * W_{u,i} * log2(1 + Gamma_{u,i} / W_{u,i})
         sum_{i in I_u} D_{u,i} <= tau * W_{u,a} * log2(1 + Gamma_{u,a} / W_{u,a})
         0 <= D_{u,i} <= Q_i(t) + A_i(t)
         sum_u sum_i W_{u,i} <= W_U        (shared access bandwidth pool)
         sum_u W_{u,a} <= W_A              (shared backhaul bandwidth pool)
         W, D >= 0

with the UAV-user association x_{u,i} and the UAV/airship positions fixed for
the slot (solved once per time slot).  R(W) = W*log2(1+Gamma/W) is strictly
concave (r''(W) = -Gamma^2/(ln2 * W * (W+Gamma)^2) < 0), so the hypograph
constraints are convex and P_R is a convex problem, solved with cvxpy
(negative exponential-cone form:  W*log2(1+G/W) = -rel_entr(W, W+G)/ln2).

Gamma calibration (link-budget consistent with the baseline fixed-SNR model):
the baseline computes SNR = P*g/noise_pw_total with bandwidth factored out of
the noise.  We interpret noise_pw_total as the noise power over the reference
bandwidth of each link and define the noise PSD N0 = noise_pw_total / W_ref:
    access:    W_ref = bw_per_user       -> Gamma_{u,i} = g_{u,i}*pTx_downlink/N0_acc
    backhaul:  W_ref = bw_bh_per_uav_Hz  -> Gamma_{u,a} = g_bh*pTx_airship/N0_bh
so at W = W_ref the concave rate equals the baseline rate W_ref*log2(1+SNR)
exactly; above/below the reference the physical model bends the rate curve.

Active set (baseline-consistent): users with a current service type
(type >= 0) AND pending demand (Q + A > 0) participate in P_R; everyone else
gets W = D = 0 (same discipline as the fixed 30/60/10 type-pool model).
beta_i = mos_type_weights[type_i] by default, so P_R maximises the same
type-weighted objective used in the system KPI (beta = 1 special case
recovers max-sum throughput).

All internal units: bandwidth in MHz, rate in Mb/s, slot length tau (s),
delivered volume in Mb per slot.
"""

import numpy as np
import cvxpy as cp

from params_sys import *
from utils_sys import cal_channel_fading_gain

LN2 = np.log(2)
MHZ = 1e6

# Noise PSD calibrated per link so that W_ref log2(1 + gP/N0) matches the
# baseline fixed-SNR capacity at the baseline per-link bandwidth.
NOISE_PSD_ACC = noise_pw_total / bw_per_user         # W / Hz
NOISE_PSD_BH = noise_pw_total / bw_bh_per_uav_Hz     # W / Hz


def cal_gamma_access_MHz(channel_gain):
    '''Gamma_{u,i} = g_{u,i} * P_downlink / N0_acc  (in MHz units).'''
    return channel_gain * pTx_downlink / NOISE_PSD_ACC / MHZ


def cal_gamma_backhaul_MHz(uav_locs, airship_loc):
    '''Gamma_{u,a} for the airship->drone LoS links (in MHz units).'''
    d2 = np.sum((uav_locs - np.asarray(airship_loc, dtype=float))**2, axis=1)
    g_bh = ref_receive_pw / d2
    return g_bh * pTx_airship / NOISE_PSD_BH / MHZ, g_bh


def solve_blocka(gamma_acc_MHz, gamma_bh_MHz, demand_Mb, assoc, beta,
                 W_U_MHz=None, W_A_MHz=None, tau=slot_len, solver='CLARABEL'):
    '''
    Solve P_R for one time slot.

    Params
    ------
    gamma_acc_MHz : (n_users,) Gamma of each user w.r.t. its associated UAV (MHz)
    gamma_bh_MHz  : (n_uavs,)  Gamma of each UAV's backhaul link (MHz)
    demand_Mb     : (n_users,) Q_i + A_i (Mb), non-negative
    assoc         : (n_users,) UAV id (0..n_uavs-1) serving each user
    beta          : (n_users,) objective weight per Mb delivered
    W_U_MHz, W_A_MHz : access / backhaul pool sizes (MHz), default from params
    tau           : slot length (s)
    solver        : cvxpy solver (CLARABEL preferred; SCS fallback)

    Returns
    ------
    dict with (all n_users / n_uavs sized, zeros outside the active set):
        W_ui_Hz, W_ua_Hz  : optimised bandwidths (Hz)
        D_Mb              : delivered volume per slot (Mb)
        R_access_Mb       : tau * f_i(W_i) per user (Mb/slot)
        R_bh_Mb           : tau * f_bh(W_u) per UAV (Mb/slot)
        S_u               : backhaul used per UAV = sum of D (Mb/slot)
        status            : cvxpy status string
    '''
    n_users = len(gamma_acc_MHz)
    n_uavs = len(gamma_bh_MHz)
    W_U_MHz = bw_total_Hz / MHZ if W_U_MHz is None else W_U_MHz
    W_A_MHz = bw_bh_total_Hz / MHZ if W_A_MHz is None else W_A_MHz

    out = dict(
        W_ui_Hz=np.zeros(n_users), W_ua_Hz=np.zeros(n_uavs),
        D_Mb=np.zeros(n_users), R_access_Mb=np.zeros(n_users),
        R_bh_Mb=np.zeros(n_uavs), S_u=np.zeros(n_uavs), status='trivial-no-demand',
    )
    active = (demand_Mb > 0) & (assoc >= 0) & (beta > 0)
    if not np.any(active):
        return out

    idx = np.where(active)[0]                     # users inside P_R
    n = len(idx)
    aidx = np.unique(assoc[idx])                  # UAVs that actually serve users
    g_acc = gamma_acc_MHz[idx]
    dem = demand_Mb[idx]
    b = beta[idx]
    m = len(aidx)

    W = cp.Variable(n, nonneg=True)               # access bw per user (MHz)
    D = cp.Variable(n, nonneg=True)               # delivered (Mb / slot)
    Wb = cp.Variable(m, nonneg=True)              # backhaul bw per served UAV (MHz)
    g_bh = gamma_bh_MHz[aidx]

    # f(W) = W log2(1 + G/W) = -rel_entr(W, W + G) / ln2   [Mb/s]
    rate_acc = -cp.rel_entr(W, W + g_acc) / LN2
    rate_bh = -cp.rel_entr(Wb, Wb + g_bh) / LN2

    # map each active user to its backhaul index (for the per-UAV coupling)
    bidx = np.searchsorted(aidx, assoc[idx])
    assert np.array_equal(aidx[bidx], assoc[idx]), 'assoc/auidx mismatch'

    constraints = [
        D <= tau * rate_acc,                       # access link
        D <= dem,                                  # queue + arrival cap
        cp.sum(W) <= W_U_MHz,                      # shared access pool
        cp.sum(Wb) <= W_A_MHz,                     # shared backhaul pool
    ]
    for u in range(m):
        constraints.append(cp.sum(D[bidx == u]) <= tau * rate_bh[u])

    prob = cp.Problem(cp.Maximize(b @ D), constraints)

    last_err = None
    for sname in [solver, 'CLARABEL', 'SCS']:
        if sname is None:
            continue
        try:
            prob.solve(solver=sname, warm_start=True)
            if prob.status in ('optimal', 'optimal_inaccurate'):
                break
        except Exception as e:                    # solver unavailable / crash
            last_err = e
            prob.status = 'solver-error'
    if prob.status not in ('optimal', 'optimal_inaccurate'):
        if last_err is not None:
            print(f'[blocka] solve failed ({prob.status}): {last_err}')
        out['status'] = str(prob.status)
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
    return out


def blocka_allocate(cids, user_x, user_y, uav_locs, airship_loc,
                    queue_Mb, traffic_Mb, types,
                    beta=None, W_U_Hz=None, W_A_Hz=None, tau=slot_len):
    '''
    One-call BLOCKA for a simulation slot: compute the channel gains from the
    given positions, then solve P_R.

    Params
    ------
    cids       : (n_users,) association (cluster id) per user
    user_x/y   : (n_users,) user positions at this slot
    uav_locs   : (n_uavs, 3) drone positions at this slot
    airship_loc: (3,) airship position
    queue_Mb   : (n_users,) backlog Q_i(t-1)
    traffic_Mb : (n_users,) arrival A_i(t)
    types      : (n_users,) service type per user (-1 if OFF)
    beta       : (n_users,) weight per Mb; default mos_type_weights[type]

    Returns: dict of solve_blocka() plus 'active' mask.
    '''
    cids = np.asarray(cids)
    demand = np.maximum(queue_Mb, 0) + np.maximum(traffic_Mb, 0)
    if beta is None:
        types = np.asarray(types)
        beta = np.where(types >= 0, np.asarray(mos_type_weights)[np.maximum(types, 0).astype(int)], 0.0)
    beta = np.asarray(beta, dtype=float)

    active = (demand > 0) & (types >= 0) & (beta > 0)   # baseline-consistent set

    g_acc = np.zeros(len(cids))
    for u in np.unique(cids[active]):
        ids_u = np.where(cids == u)[0]
        g_acc[ids_u] = cal_channel_fading_gain(user_x[ids_u], user_y[ids_u],
                                               uav_locs[u, 0], uav_locs[u, 1], uav_locs[u, 2])
    gamma_bh, _ = cal_gamma_backhaul_MHz(uav_locs, airship_loc)

    res = solve_blocka(
        gamma_acc_MHz=cal_gamma_access_MHz(g_acc),
        gamma_bh_MHz=gamma_bh,
        demand_Mb=demand, assoc=cids, beta=beta,
        W_U_MHz=(bw_total_Hz / MHZ if W_U_Hz is None else W_U_Hz / MHZ),
        W_A_MHz=(bw_bh_total_Hz / MHZ if W_A_Hz is None else W_A_Hz / MHZ),
        tau=tau,
    )
    res['active'] = active
    return res

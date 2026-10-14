# -*- coding: utf-8
'''Experiment B Stage 1: closed-loop 501-slot validation of dual-guided
candidates (read-only w.r.t. production code; no Actor / beta / psi_E changes).

Three paired schemes (identical exogenous SFM+Weibull inputs, seed=0):
  k3      : [stay, beta-0.25, beta-0.5]                  (production directed)
  betak5  : [stay, beta-{0.1,0.25,0.5,0.75}]             (non-dual K5 control)
  dualk5  : [stay, beta-{0.25,0.5}, dual-{0.25,0.5}]     (c_nu=0.1)

Check 1 (per-slot, dualk5/betak5): identity
  dJ_gain = J_K3ref - J_winner = dU - dC_E,
  dU = beta.(D_new - D_ref),  dC_E = psi_E * E.(P_new - P_ref).
Check 2: executed-vs-evaluated positions, speed/position limits, per-UAV
  propulsion power, PW_THRES violations, full E_u(t) trajectories.

Outputs -> D:\\Learn\\work\\drl-opencode\\expB_results\\stage1
'''
import os, sys, time, zlib
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
OUT = r'D:\Learn\work\drl-opencode\expB_results\stage1'
os.makedirs(OUT, exist_ok=True)
EXPA = r'D:\Learn\work\drl-opencode\expA_results'

from params_sys import *
from utils_sys import (kmeans_clustering, find_active_users, update_queue_Mb,
                       cal_channel_fading_gain, cal_uav_propulsion_energy, mos_func)
import blocka_optimizer as B
from mobility_model import adjust_location, adjust_altitude

MHZ, LN2 = B.MHZ, B.LN2
BOUNDARY = float(boundary)
PSI_E = 1.0 / PW_MAX
TAU = float(slot_len)
SPD_XY, SPD_Z = float(uav_speedxy_max), float(uav_speedz_max)
SPD_AIR = 5.0
DIR_EPS, NU_EPS = 1e-6, 1e-9
C_NU = 0.1
S = int(n_slots)
W_MOS = np.asarray(mos_type_weights, dtype=float)


# ------------------------------- shared exogenous inputs (paired across schemes)
def from_exel(fname):
    return np.transpose(pd.read_excel(fname, index_col=0).to_numpy())


sim_in = os.path.join(PROJ, 'simulation', 'sim-input')
tag_base = (f"nusers={n_users}, nslots={n_slots}, ntype=3, "
            f"lambda_Mb={lambd_Mb:.1f}, uON={ON_duration_mean_tslot:.0f}, "
            f"uOFF={OFF_duration_mean_tslot:.0f}")
xaxis_all = from_exel(os.path.join(sim_in, f"xlocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx"))
yaxis_all = from_exel(os.path.join(sim_in, f"ylocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx"))
arrival = from_exel(os.path.join(sim_in, f"arrival traffic, dist=weibull, {tag_base}.xlsx"))
tr_type = from_exel(os.path.join(sim_in, f"traffic type, {tag_base}.xlsx")).astype(int)
tr_state = from_exel(os.path.join(sim_in, f"traffic state, {tag_base}.xlsx")).astype(bool)
print('paired inputs loaded', xaxis_all.shape, arrival.shape, flush=True)

# reference -nd KPIs (scheme k3 must reproduce these)
import glob
nd_dirs = glob.glob(os.path.join(PROJ, 'simulation', 'sim-output',
                                 '*K20-s1.1-sfm-wb-jac-nd, n_slots=*'))
kpi_ref = pd.read_excel(os.path.join(sorted(nd_dirs)[-1], 'Proposed-kpis-over-time.xlsx'),
                        index_col=0)


# ------------------------------------- dual-extracting base solve (expA copy)
def solve_blocka_diag(gamma_acc_MHz, gamma_bh_MHz, demand_Mb, assoc, beta,
                      tau=TAU, solver='CLARABEL'):
    n_users_ = len(gamma_acc_MHz)
    W_U = bw_total_Hz / MHZ
    W_A = bw_bh_total_Hz / MHZ
    out = dict(D_Mb=np.zeros(n_users_), nu_full=np.zeros(len(gamma_bh_MHz)),
               W_ua_Hz=np.zeros(len(gamma_bh_MHz)),
               status='trivial-no-demand', solver_used=None)
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
                   cp.sum(W) <= W_U, cp.sum(Wb) <= W_A]
    bh_cons = [cp.sum(D[bidx == u]) <= tau * rate_bh[u] for u in range(m)]
    constraints += bh_cons
    prob = cp.Problem(cp.Maximize(b @ D), constraints)
    status = None
    for sname in [solver, 'CLARABEL', 'SCS']:
        try:
            prob.solve(solver=sname, warm_start=True)
            status = prob.status
            if status in ('optimal', 'optimal_inaccurate'):
                out['solver_used'] = sname
                break
        except Exception:
            status = 'solver-error'
    out['status'] = str(status)
    if status in ('optimal', 'optimal_inaccurate'):
        nu = [float(c.dual_value) if c.dual_value is not None else np.nan for c in bh_cons]
        out['nu_full'][aidx] = np.array(nu)
        D_sol = np.maximum(D.value, 0.0)
        out['D_Mb'][idx] = D_sol
        out['W_ua_Hz'][aidx] = np.maximum(Wb.value, 0.0) * MHZ
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


# --------------------------------------------------------------- controller
class Stage1Controller:
    '''Mirrors JointACCritic directed-mode geometry/eval; candidate sets per
    variant. No learning, no Actor.'''

    def __init__(self, n_uavs_, variant):
        self.M = n_uavs_
        self.variant = variant
        self.E = np.zeros(n_uavs_)
        self.psi_Q = 1.0 / float(queue_norm_Mb)
        self.tau = TAU
        n_users_init = n_users
        self.ld = np.asarray(ON_data_arrival_mean_Mb, dtype=float)
        self.w = np.asarray(mos_type_weights, dtype=float)
        self.V = float(lyapunov_param)

    def begin_slot(self, user_x, user_y, Q, A, types, cids, uav_locs, airship_pos):
        types = np.asarray(types)
        demand = np.maximum(Q, 0) + np.maximum(A, 0)
        lam = np.where(types >= 0, self.ld[np.maximum(types, 0).astype(int)], 1.0)
        active = (types >= 0) & (demand > 0)
        if np.any(active):
            w_sum = max(float(np.sum(self.w[types[active].astype(int)])), 1e-12)
            beta = np.where(active,
                            self.psi_Q * np.maximum(Q, 0)
                            + self.V * self.w[np.maximum(types, 0).astype(int)]
                            / (lam * w_sum), 0.0)
        else:
            beta = np.zeros(len(types))
        user_xy = np.stack([np.asarray(user_x, float), np.asarray(user_y, float)], 1)
        q_xy = np.asarray(uav_locs, float)[:, :2]
        cent = q_xy.copy()
        for u in range(self.M):
            m = (np.asarray(cids) == u) & active
            if not np.any(m):
                continue
            bw = float(beta[m].sum())
            if bw > 1e-12:
                cent[u] = (beta[m, None] * user_xy[m]).sum(axis=0) / bw
        B_u = np.array([float(beta[(np.asarray(cids) == u) & active].sum())
                        for u in range(self.M)])
        if B_u.sum() > 1e-12:
            air_target = (B_u[:, None] * q_xy).sum(axis=0) / B_u.sum()
        else:
            air_target = np.asarray(airship_pos, float)[:2].copy()
        self.ctx = dict(user_x=np.asarray(user_x, float), user_y=np.asarray(user_y, float),
                        Q=Q, A=A, types=types, cids=np.asarray(cids), beta=beta,
                        demand=demand, cent=cent, air_target=air_target)
        self.st = dict(uav_locs=np.asarray(uav_locs, float).copy(),
                       airship_pos=np.asarray(airship_pos, float).copy())
        return beta

    def _dirs(self):
        ctx, st = self.ctx, self.st
        d_xy = ctx['cent'] - st['uav_locs'][:, :2]
        nrm = np.linalg.norm(d_xy, axis=1, keepdims=True)
        e_u = np.where(nrm > DIR_EPS, d_xy / np.maximum(nrm, DIR_EPS), 0.0)
        d_a = ctx['air_target'] - st['airship_pos'][:2]
        na = float(np.linalg.norm(d_a))
        e_a = d_a / na if na > DIR_EPS else np.zeros(2)
        return e_u, e_a

    def _to_motion(self, a):
        M = self.M
        st = self.st
        bU = a[:3 * M].reshape(M, 3)
        bA = a[3 * M:]
        v = np.zeros((M, 3))
        for u in range(M):
            bxy = bU[u, :2]
            v[u, :2] = SPD_XY * bxy / max(1.0, np.linalg.norm(bxy))
            v[u, 2] = SPD_Z * bU[u, 2]
        locs = st['uav_locs'] + TAU * v
        locs[:, 0] = np.clip(locs[:, 0], -BOUNDARY, BOUNDARY)
        locs[:, 1] = np.clip(locs[:, 1], -BOUNDARY, BOUNDARY)
        locs[:, 2] = np.clip(locs[:, 2], z_min, z_max)
        va = SPD_AIR * bA / max(1.0, float(np.linalg.norm(bA)))
        air = st['airship_pos'] + TAU * np.array([va[0], va[1], 0.0])
        air[0] = np.clip(air[0], -BOUNDARY, BOUNDARY)
        air[1] = np.clip(air[1], -BOUNDARY, BOUNDARY)
        air[2] = st['airship_pos'][2]
        return locs, v, air

    def _eval(self, a):
        ctx = self.ctx
        locs, v, air = self._to_motion(a)
        P = np.array([float(cal_uav_propulsion_energy(V=np.linalg.norm(v[u, :2]),
                                                      Vz=v[u, 2])[0])
                      for u in range(self.M)])
        res = B.blocka_allocate(cids=ctx['cids'], user_x=ctx['user_x'],
                                user_y=ctx['user_y'], uav_locs=locs, airship_loc=air,
                                queue_Mb=ctx['Q'], traffic_Mb=ctx['A'],
                                types=ctx['types'], beta=ctx['beta'], tau=TAU)
        J = PSI_E * float(self.E @ P) - float(ctx['beta'] @ res['D_Mb'])
        return dict(J=J, P=P, res=res, locs=locs, v=v, air=air, a=a)

    def _a_dir(self, e_u, e_a, rho):
        a = np.zeros(3 * self.M + 2)
        a[:3 * self.M] = np.hstack([rho * e_u, np.zeros((self.M, 1))]).ravel()
        a[3 * self.M:] = rho * e_a
        return a

    def _dual_dirs(self):
        '''Base dual solve at current geometry -> UAV blend + airship gradient.'''
        ctx, st = self.ctx, self.st
        g_acc, g_bh = channel_gammas(ctx['user_x'], ctx['user_y'], ctx['cids'],
                                     st['uav_locs'], st['airship_pos'],
                                     ctx['beta'], ctx['demand'], ctx['types'])
        base = solve_blocka_diag(g_acc, g_bh, ctx['demand'], ctx['cids'], ctx['beta'])
        nu = np.maximum(np.nan_to_num(base['nu_full'], nan=0.0), 0.0)
        w = np.where(nu > NU_EPS, nu / (C_NU + nu), 0.0)
        e_u_beta, e_a_beta = self._dirs()
        e_bh = np.zeros((self.M, 2))
        for u in range(self.M):
            d = st['airship_pos'][:2] - st['uav_locs'][u, :2]
            n = float(np.linalg.norm(d))
            e_bh[u] = d / n if n > DIR_EPS else 0.0
        d_u = (1.0 - w[:, None]) * e_u_beta + w[:, None] * e_bh
        nrm = np.linalg.norm(d_u, axis=1)
        n_cancel = 0
        for u in range(self.M):
            if nrm[u] < DIR_EPS:
                n_cancel += 1
                d_u[u] = e_bh[u] if w[u] >= 0.5 else e_u_beta[u]
        e_u = np.stack([d_u[u] / max(float(np.linalg.norm(d_u[u])), DIR_EPS)
                        for u in range(self.M)])
        # airship: true backhaul-rate gradient at Wb* from the base solve
        d3 = st['uav_locs'] - st['airship_pos'][None, :]
        d2 = np.sum(d3 ** 2, axis=1)
        kappa = ref_receive_pw * pTx_airship / (B.NOISE_PSD_BH * MHZ)
        gam = kappa / np.maximum(d2, 1e-12)
        Wb = base['W_ua_Hz'] / MHZ
        coef = (TAU / LN2) * (Wb / np.maximum(Wb + gam, 1e-12)) * (2.0 * gam / d2)
        g_a = ((coef * nu)[:, None] * d3).sum(axis=0)
        g_a[2] = 0.0
        nrm_a = float(np.linalg.norm(g_a[:2]))
        e_a = g_a[:2] / nrm_a if nrm_a > DIR_EPS else np.zeros(2)
        return e_u, e_a, w, n_cancel, (nrm_a <= DIR_EPS)

    def decide_and_score(self):
        M = self.M
        e_u_beta, e_a_beta = self._dirs()
        a0 = np.zeros(3 * M + 2)
        evs, cats, plan_kinds = [], [], []
        evs.append(self._eval(a0))
        cats.append('stay')
        plan_kinds.append(('stay', 0.0))
        if self.variant == 'k3':
            plan = [('beta', 0.25), ('beta', 0.5)]
        elif self.variant == 'betak5':
            plan = [('beta', 0.1), ('beta', 0.25), ('beta', 0.5), ('beta', 0.75)]
        else:  # dualk5
            plan = [('beta', 0.25), ('beta', 0.5)]
        extra = {}
        if self.variant == 'dualk5':
            e_u_d, e_a_d, w, n_cancel, air_zero = self._dual_dirs()
            plan = plan + [('dual', 0.25), ('dual', 0.5)]
            extra = dict(w=w, n_cancel=n_cancel, air_zero=air_zero,
                         e_u_d=e_u_d, e_a_d=e_a_d)
        for kind, rho in plan:
            if kind == 'beta':
                evs.append(self._eval(self._a_dir(e_u_beta, e_a_beta, rho)))
            else:
                evs.append(self._eval(self._a_dir(extra['e_u_d'], extra['e_a_d'], rho)))
            cats.append(kind)
            plan_kinds.append((kind, rho))
        Js = np.array([e['J'] for e in evs])
        k = int(np.argmin(Js))
        best = evs[k]
        best['k'] = k
        best['cats'] = cats
        best['plan_kinds'] = plan_kinds
        best['all_J'] = Js
        best['extra'] = extra
        # within-slot K3 reference = min(stay, beta-0.25, beta-0.5); paired
        # decomposition Check 1 compares the K5 winner against this reference
        cand_k3 = [i for i, (kind, rho) in enumerate(plan_kinds)
                   if kind == 'stay' or (kind == 'beta' and rho in (0.25, 0.5))]
        J_k3 = min(evs[i]['J'] for i in cand_k3)
        k3_i = cand_k3[int(np.argmin([evs[i]['J'] for i in cand_k3]))]
        best['J_k3ref'] = J_k3
        best['k3_i'] = k3_i
        best['evs'] = evs
        return best

    def update_energy(self, P_exec):
        self.E = np.maximum(self.E + np.asarray(P_exec, float) - PW_THRES, 0.0)


# ------------------------------------------------------------------ runner
def run_scheme(variant):
    print(f'\n===== scheme: {variant} =====', flush=True)
    np.random.seed(0)
    ctrl = Stage1Controller(n_uavs, variant)
    Q = np.zeros(n_users)
    ema = np.zeros(n_users)
    prev_locs = np.column_stack([init_locations_uav[:, 0], init_locations_uav[:, 1],
                                 np.full(n_uavs, float(z_init_proposed))])
    airship_pos = np.array([airship_x, airship_y, airship_z], float)

    n_t = S
    mos_web = np.full(n_t, np.nan)
    mos_vid = np.full(n_t, np.nan)
    mos_msg = np.full(n_t, np.nan)
    wmos = np.full(n_t, np.nan)
    qlen_act = np.full(n_t, np.nan)
    web_qlen = np.full(n_t, np.nan)       # mean over active Web users
    web_qlen_all = np.full(n_t, np.nan)   # mean over all Web users
    web_qlen_max = np.full(n_t, np.nan)
    sum_D = np.zeros(n_t)
    sum_Rbh = np.zeros(n_t)
    sum_Racc = np.zeros(n_t)
    chcapa = np.full(n_t, np.nan)        # mean R_access over all users (KPI)
    bh_util = np.zeros(n_t)
    J_star = np.zeros(n_t)
    win = np.zeros(n_t, dtype=int)        # 0=stay 1=beta 2=dual
    dJ_g = np.full(n_t, np.nan)           # Check-1 decomposition (non-k3)
    dU_g = np.full(n_t, np.nan)
    dCE_g = np.full(n_t, np.nan)
    decomp_err = np.full(n_t, np.nan)
    dual_win = np.zeros(n_t, dtype=bool)
    P_hist = np.zeros((n_t, n_uavs))
    E_hist = np.zeros((n_t, n_uavs))
    vxy_max_hist = np.zeros(n_t)
    vz_max_hist = np.zeros(n_t)
    vair_max_hist = np.zeros(n_t)
    exec_dev = np.zeros(n_t)              # |exec - eval| max (Check 2)
    solver_bad = 0
    cids_hash = np.zeros(n_t, dtype=np.int64)
    air_zero_slots = 0
    cancel_total = 0

    t_wall = time.time()
    for t in range(n_t):
        Q_prev = Q.copy()
        A_t = arrival[:, t]
        types_t = tr_type[:, t]
        isActive = find_active_users(queue_len_Mb=Q_prev, incoming_traffic_Mb=A_t)
        loc_users = np.hstack((xaxis_all[:, [t]], yaxis_all[:, [t]],
                               np.zeros((n_users, 1))))
        _, cids, _ = kmeans_clustering(loc_users)
        cids_hash[t] = zlib.crc32(np.ascontiguousarray(cids).tobytes())

        beta = ctrl.begin_slot(user_x=xaxis_all[:, t], user_y=yaxis_all[:, t],
                               Q=Q_prev, A=A_t, types=types_t, cids=cids,
                               uav_locs=prev_locs, airship_pos=airship_pos)
        best = ctrl.decide_and_score()
        evs = best['evs']
        solver_bad += sum(1 for e in evs
                          if e['res']['status'] not in ('optimal', 'optimal_inaccurate',
                                                        'trivial-no-demand'))
        # Check 2: executed vs evaluated + limits
        exec_locs = np.array([[adjust_location(best['locs'][u, 0], bound=boundary),
                               adjust_location(best['locs'][u, 1], bound=boundary),
                               adjust_altitude(best['locs'][u, 2], zmin=z_min, zmax=z_max)]
                              for u in range(n_uavs)])
        exec_dev[t] = float(np.max(np.abs(exec_locs - best['locs'])))
        vxy_max_hist[t] = float(np.max(np.linalg.norm(best['v'][:, :2], axis=1)))
        vz_max_hist[t] = float(np.max(np.abs(best['v'][:, 2])))
        va = (best['air'] - ctrl.st['airship_pos'])[:2] / TAU
        vair_max_hist[t] = float(np.linalg.norm(va))

        # win category
        cat = best['cats'][best['k']]
        win[t] = 0 if cat == 'stay' else (1 if cat == 'beta' else 2)
        if variant == 'dualk5':
            dual_win[t] = (cat == 'dual')
            air_zero_slots += int(best['extra']['air_zero'])
            cancel_total += int(best['extra']['n_cancel'])
            # Check 1 paired decomposition vs within-slot K3 reference
            ref = evs[best['k3_i']]
            dU = float(beta @ (best['res']['D_Mb'] - ref['res']['D_Mb']))
            dCE = PSI_E * float(ctrl.E @ (best['P'] - ref['P']))
            dJ_g[t] = best['J_k3ref'] - best['J']
            dU_g[t] = dU
            dCE_g[t] = dCE
            decomp_err[t] = abs((dU - dCE) - dJ_g[t])
        elif variant == 'betak5':
            ref = evs[best['k3_i']]
            dU = float(beta @ (best['res']['D_Mb'] - ref['res']['D_Mb']))
            dCE = PSI_E * float(ctrl.E @ (best['P'] - ref['P']))
            dJ_g[t] = best['J_k3ref'] - best['J']
            dU_g[t] = dU
            dCE_g[t] = dCE
            decomp_err[t] = abs((dU - dCE) - dJ_g[t])

        # commit geometry + queues (cell-26 semantics)
        prev_locs = exec_locs.copy()
        airship_pos = best['air'].copy()
        ctrl.update_energy(P_exec=best['P'])
        D_opt = best['res']['D_Mb']
        R_access = best['res']['R_access_Mb']
        S_u = best['res']['S_u']
        R_bh = best['res']['R_bh_Mb']
        Q = np.maximum(Q_prev + A_t - D_opt, 0.0)
        ema = 0.999 * Q + 0.001 * ema

        # KPIs
        tt = types_t
        for k_type, name in ((0, 'mos_web'), (1, 'mos_vid'), (2, 'mos_msg')):
            m = (tt == k_type) & tr_state[:, t]
            if np.any(m):
                mos = mos_func(qlen=ema[m], lambda_=ON_data_arrival_mean_Mb[k_type],
                               ss_min_arr=ss_min_types[k_type],
                               ss_max_arr=ss_max_types[k_type])
                val = float(np.mean(mos))
                if name == 'mos_web':
                    mos_web[t] = val
                elif name == 'mos_vid':
                    mos_vid[t] = val
                else:
                    mos_msg[t] = val
        parts = [W_MOS[0] * mos_web[t], W_MOS[1] * mos_vid[t], W_MOS[2] * mos_msg[t]]
        wmos[t] = float(np.nansum(parts)) if np.any(np.isfinite(parts)) else np.nan
        n_act = int(isActive.sum())
        qlen_act[t] = float(np.sum(Q[isActive]) / n_act) if n_act else np.nan
        mw = isActive & (tt == 0)
        web_qlen[t] = float(np.mean(Q[mw])) if np.any(mw) else np.nan
        allw = tt == 0
        web_qlen_all[t] = float(np.mean(Q[allw])) if np.any(allw) else np.nan
        web_qlen_max[t] = float(np.max(Q[allw])) if np.any(allw) else np.nan
        sum_D[t] = float(np.sum(D_opt))
        sum_Rbh[t] = float(np.sum(R_bh))
        sum_Racc[t] = float(np.sum(R_access))
        chcapa[t] = float(np.mean(R_access))
        bh_util[t] = float(np.sum(S_u) / max(np.sum(R_bh), 1e-9))
        J_star[t] = best['J']
        P_hist[t] = best['P']
        E_hist[t] = ctrl.E

        if (t + 1) % 100 == 0:
            print(f'  t={t + 1}/{n_t}  {time.time() - t_wall:.0f}s  '
                  f'wmos={np.nanmean(wmos):.3f}  Qweb={np.nanmean(web_qlen):.2f}', flush=True)

    wall = time.time() - t_wall
    res = dict(variant=variant, wall=wall, mos_web=mos_web, mos_vid=mos_vid,
               mos_msg=mos_msg, wmos=wmos, qlen_act=qlen_act, web_qlen=web_qlen,
               web_qlen_all=web_qlen_all, web_qlen_max=web_qlen_max, sum_D=sum_D,
               sum_Rbh=sum_Rbh, sum_Racc=sum_Racc, chcapa=chcapa, bh_util=bh_util,
               J_star=J_star, win=win, dJ_g=dJ_g, dU_g=dU_g, dCE_g=dCE_g,
               decomp_err=decomp_err, dual_win=dual_win, P_hist=P_hist,
               E_hist=E_hist, vxy=vxy_max_hist, vz=vz_max_hist, vair=vair_max_hist,
               exec_dev=exec_dev, solver_bad=solver_bad, cids_hash=cids_hash,
               air_zero_slots=air_zero_slots, cancel_total=cancel_total,
               E_final=ctrl.E.copy())
    print(f'  done in {wall:.0f}s  solver_bad={solver_bad}', flush=True)
    return res


results = {}
for variant in ('k3', 'betak5', 'dualk5'):
    results[variant] = run_scheme(variant)

# paired-input check: clustering sequence identical across schemes
h0, h1, h2 = (results[v]['cids_hash'] for v in ('k3', 'betak5', 'dualk5'))
print(f'\n[cids hash] k3==betak5: {bool((h0 == h1).all())}   k3==dualk5: {bool((h0 == h2).all())}')

# scheme k3 must reproduce the recorded -nd run (harness validation)
r3 = results['k3']
kpi_qlen = kpi_ref['qlen-Mb-avg (active)'].to_numpy(float)
kpi_chcapa = kpi_ref['chcapa-Mb-avg (active)'].to_numpy(float)
qlen_maxdiff = float(np.nanmax(np.abs(r3['qlen_act'] - kpi_qlen)))
chcapa_maxdiff = float(np.nanmax(np.abs(r3['chcapa'] - kpi_chcapa)))
print(f'[k3 harness repro vs -nd] qlen max|d|={qlen_maxdiff:.3e}  '
      f'chcapa max|d|={chcapa_maxdiff:.3e}')
HARNESS_OK = qlen_maxdiff <= 1e-6 and chcapa_maxdiff <= 1e-6
print(f'[k3 harness repro] -> {"PASS" if HARNESS_OK else "FAIL"}')

np.savez_compressed(os.path.join(OUT, 'stage1_runs.npz'),
                    **{f'{v}_{k}': results[v][k] for v in results
                       for k in results[v] if k != 'variant'})

# ------------------------------------------------------------- summary
def wm(res, slice_=None):
    sl = slice(None) if slice_ is None else slice(*slice_)
    parts = [W_MOS[0] * np.nanmean(res['mos_web'][sl]),
             W_MOS[1] * np.nanmean(res['mos_vid'][sl]),
             W_MOS[2] * np.nanmean(res['mos_msg'][sl])]
    return float(np.sum(parts))


def decay_time(series, thr=1.0):
    idx = np.where(np.nan_to_num(series, nan=0.0) > thr)[0]
    return int(idx.max() + 1) if len(idx) else 0


print('\n==================== Stage 1 closed-loop (501 slots, seed=0) ====================')
hdr = f'{"metric":34s} {"k3":>12s} {"betak5":>12s} {"dualk5":>12s}'
print(hdr)
rows_out = {}


def prow(name, f, fmt='{:+.4f}'):
    vals = [f(results[v]) for v in ('k3', 'betak5', 'dualk5')]
    print(f'{name:34s} ' + ' '.join(fmt.format(x) if np.isscalar(x) and np.isfinite(x)
                                    else f'{x!s:>12s}' for x in vals))
    rows_out[name] = vals


prow('weighted MOS (full)', wm)
prow('weighted MOS (first 120)', lambda r: wm(r, (0, 120)))
prow('Web MOS (full)', lambda r: float(np.nanmean(r['mos_web'])), '{:.4f}')
prow('Web MOS (first 120)', lambda r: float(np.nanmean(r['mos_web'][:120])), '{:.4f}')
prow('active queue mean (Mb)', lambda r: float(np.nanmean(r['qlen_act'])), '{:.3f}')
prow('Web queue mean-active (Mb)', lambda r: float(np.nanmean(r['web_qlen'])), '{:.3f}')
prow('Web queue mean-all (Mb)', lambda r: float(np.nanmean(r['web_qlen_all'])), '{:.3f}')
prow('Web queue peak (Mb)', lambda r: float(np.nanmax(r['web_qlen_max'])), '{:.2f}')
prow('Web queue decay slot(>1Mb)', lambda r: decay_time(r['web_qlen_all']), '{:d}')
prow('sum D delivered (Mb/slot)', lambda r: float(np.mean(r['sum_D'])), '{:.2f}')
prow('backhaul capacity (Mb/slot)', lambda r: float(np.mean(r['sum_Rbh'])), '{:.2f}')
prow('access capacity (Mb/slot)', lambda r: float(np.mean(r['sum_Racc'])), '{:.2f}')
prow('backhaul utilization', lambda r: float(np.mean(r['bh_util'])), '{:.3f}')
prow('win stay', lambda r: float((r['win'] == 0).mean()), '{:.1%}')
prow('win beta', lambda r: float((r['win'] == 1).mean()), '{:.1%}')
prow('win dual', lambda r: float((r['win'] == 2).mean()), '{:.1%}')

d = results['dualk5']
print('\n--- Check 1: paired decomposition (dualk5, per-slot identity) ---')
ok = np.isfinite(d['decomp_err'])
print(f'  identity |dJ - (dU - dC_E)| max = {np.nanmax(d["decomp_err"]):.3e}   '
      f'mean = {np.nanmean(d["decomp_err"]):.3e}')
selm = d['dual_win'] & ok
print(f'  dual-win slots: {selm.sum()}   mean dJ_gain = {np.nanmean(d["dJ_g"][selm]):.4f}  '
      f'mean dU = {np.nanmean(d["dU_g"][selm]):.4f}  '
      f'mean dC_E = {np.nanmean(d["dCE_g"][selm]):.4f}')
print(f'  all slots: mean dJ_gain = {np.nanmean(d["dJ_g"]):.4f}  '
      f'frac(dJ>1e-6) = {np.nanmean(d["dJ_g"] > 1e-6):.1%}')
db = results['betak5']
selb = np.isfinite(db['decomp_err']) & (db['dJ_g'] > 1e-6)
print(f'  betak5 vs K3-ref: mean dJ = {np.nanmean(db["dJ_g"]):.4f}  '
      f'frac>1e-6 = {np.nanmean(db["dJ_g"] > 1e-6):.1%}')

print('\n--- Check 2: motion / power feasibility ---')
for v in ('k3', 'betak5', 'dualk5'):
    r = results[v]
    Pm = r['P_hist']
    print(f'  [{v}] max|exec-eval|={r["exec_dev"].max():.3f} m  '
          f'vxy_max={r["vxy"].max():.2f} (lim {SPD_XY})  vz_max={r["vz"].max():.2f} '
          f'(lim {SPD_Z})  vair_max={r["vair"].max():.2f} (lim {SPD_AIR})  '
          f'solver_bad={r["solver_bad"]}')
    print(f'        P mean per UAV = {np.round(Pm.mean(axis=0), 1).tolist()} W '
          f'(budget {PW_THRES:.0f});  slots P_u>100: '
          f'{(Pm > PW_THRES).sum(axis=0).tolist()}; '
          f'violation integral = {np.round(np.maximum(Pm - PW_THRES, 0).sum(axis=0), 0).tolist()} W*slot')
    print(f'        E(t): final={np.round(r["E_final"], 1).tolist()}  '
          f'mean={np.round(r["E_hist"].mean(axis=0), 1).tolist()}  '
          f'max={np.round(r["E_hist"].max(axis=0), 1).tolist()}  '
          f'E/t@500={np.round(r["E_final"] / 500.0, 3).tolist()}')
print(f'  dualk5: airship-zero-grad slots={d["air_zero_slots"]}  '
      f'uav direction cancellations={d["cancel_total"]}')
print(f'  wall time: ' + ', '.join(f'{v}={results[v]["wall"]:.0f}s'
                                   for v in ('k3', 'betak5', 'dualk5')))

# stage-0 caliber note (Check 1 explanation)
print('\n--- stage-0 caliber explanation ---')
print('  stage-0 dP=+45.7W was the UNWEIGHTED sum_u dP_u on winner-changed slots;')
print('  the identity uses dC_E = psi_E * sum_u E_u * dP_u (E-queue weighted),')
print('  and P(V) is non-monotone (hover 88.1W > P(7.5m/s)=83.8W < P(15m/s)=145.9W),')
print('  so unweighted dP and E-weighted dC_E can differ in magnitude AND sign.')

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
ax = axes[0, 0]
for v, c in (('k3', 'gray'), ('betak5', 'tab:blue'), ('dualk5', 'tab:orange')):
    ax.plot(results[v]['wmos'], color=c, lw=1.2, label=v)
ax.set_title('weighted MOS'); ax.legend(fontsize=8); ax.set_xlabel('slot')
ax = axes[0, 1]
for v, c in (('k3', 'gray'), ('betak5', 'tab:blue'), ('dualk5', 'tab:orange')):
    ax.plot(results[v]['mos_web'], color=c, lw=1.2, label=v)
ax.set_title('Web MOS'); ax.set_xlabel('slot')
ax = axes[0, 2]
for v, c in (('k3', 'gray'), ('betak5', 'tab:blue'), ('dualk5', 'tab:orange')):
    ax.plot(results[v]['web_qlen_all'], color=c, lw=1.2, label=v)
ax.set_title('Web queue (all Web users, Mb)'); ax.set_xlabel('slot')
ax = axes[1, 0]
for u in range(n_uavs):
    ax.plot(d['E_hist'][:, u], lw=1.0, label=f'UAV{u}')
ax.axhline(PW_THRES, color='k', ls='--', lw=0.8, label='P_thres')
ax.set_title('dualk5 energy queues E_u(t)'); ax.legend(fontsize=7); ax.set_xlabel('slot')
ax = axes[1, 1]
for u in range(n_uavs):
    ax.plot(d['P_hist'][:, u], lw=0.9, label=f'UAV{u}')
ax.axhline(PW_THRES, color='k', ls='--', lw=0.8)
ax.set_title('dualk5 propulsion power (W)'); ax.legend(fontsize=7); ax.set_xlabel('slot')
ax = axes[1, 2]
ax.plot(d['dJ_g'], lw=0.8, color='tab:orange', label='dJ_gain')
ax.plot(np.where(d['dual_win'], d['dJ_g'], np.nan), 'o', ms=3, color='k', label='dual won')
ax.axhline(0, color='r', ls='--', lw=0.8)
ax.set_title('dualk5 per-slot dJ vs K3-ref'); ax.legend(fontsize=7); ax.set_xlabel('slot')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'stage1_results.png'), dpi=130)
print(f'\nwrote {OUT}\\stage1_runs.npz, stage1_results.png')

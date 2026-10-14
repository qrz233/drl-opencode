# -*- coding: utf-8
'''Experiment B Stage 2: multi-seed paired closed-loop validation
(501 slots = full paper horizon; no production-code changes).

Seeds s = 0..4. Within each seed the three schemes share identical
exogenous inputs AND identical kmeans stream (np.random.seed(s)):
  k3      : [stay, beta-0.25, beta-0.5]                  (production directed)
  betak5  : [stay, beta-{0.1,0.25,0.5,0.75}]             (non-dual K5 control)
  dualk5  : [stay, beta-{0.25,0.5}, dual-{0.25,0.5}]     (c_nu=0.1)

Seed protocol:
  s=0 : frozen paper inputs (SFM/Weibull/ON-OFF xlsx) + kmeans seed 0
        -> k3 bit-reproduces stage-1 (harness check).
  s>0 : regenerated paired world
        SFM mobility     seed = 11 + s
        ON/OFF schedule  seed = 10000 + s  (rebinds traffic_model.rng)
        Weibull arrival  seed = s
        kmeans           seed = s

Check 1 (per-slot): dJ = dU - dC_E vs within-slot K3 reference.
Check 2: exec-vs-eval positions, speed limits, power, E_u(t) bounded.

Outputs -> D:\\Learn\\work\\drl-opencode\\expB_results\\stage2_multiseed
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
OUT = r'D:\Learn\work\drl-opencode\expB_results\stage2_multiseed'
os.makedirs(OUT, exist_ok=True)
STAGE1_NPZ = r'D:\Learn\work\drl-opencode\expB_results\stage1\stage1_runs.npz'

from params_sys import *
from utils_sys import (kmeans_clustering, find_active_users, update_queue_Mb,
                       cal_channel_fading_gain, cal_uav_propulsion_energy, mos_func)
import blocka_optimizer as B
import traffic_model as TM
from mobility_model import (adjust_location, adjust_altitude,
                            generate_mobility_SFM)

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
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ('k3', 'betak5', 'dualk5')


# ------------------------------- shared exogenous inputs (paired across schemes)
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
    tr_type=from_exel(os.path.join(sim_in, f"traffic type, {tag_base}.xlsx")).astype(int),
    tr_state=from_exel(os.path.join(sim_in, f"traffic state, {tag_base}.xlsx")).astype(bool),
)
print('frozen inputs loaded', _frozen['xaxis'].shape, _frozen['arrival'].shape, flush=True)


def make_inputs(s):
    '''Exogenous inputs for seed s (identical across the 3 schemes).'''
    if s == 0:
        return _frozen
    xaxis, yaxis = generate_mobility_SFM(
        n_users=n_users, n_slots=n_slots, slot_len=slot_len, boundary=boundary,
        speed_avg=speed_user_avg, seed=11 + s)
    saved = TM.rng
    TM.rng = np.random.default_rng(10000 + s)
    st = np.zeros((n_users, n_slots), dtype=bool)
    ty = np.full((n_users, n_slots), -1, dtype=int)
    for uid in range(n_users):
        st[uid], _, ty[uid] = TM.generate_traffic_ON_OFF(
            n_slots=n_slots, ON_duration_mean_tslot=ON_duration_mean_tslot,
            OFF_duration_mean_tslot=OFF_duration_mean_tslot,
            data_arrival_mean_Mb=ON_data_arrival_mean_Mb)
    TM.rng = saved
    arrival = TM.generate_arrival_from_schedule(
        traffic_state=st, traffic_type=ty,
        data_arrival_mean_Mb=ON_data_arrival_mean_Mb, dist='weibull', seed=s)
    return dict(xaxis=xaxis, yaxis=yaxis, arrival=arrival,
                tr_type=ty, tr_state=st)


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
def run_scheme(variant, exo, seed):
    print(f'\n===== scheme: {variant} (seed={seed}) =====', flush=True)
    np.random.seed(seed)
    ctrl = Stage1Controller(n_uavs, variant)
    Q = np.zeros(n_users)
    ema = np.zeros(n_users)
    prev_locs = np.column_stack([init_locations_uav[:, 0], init_locations_uav[:, 1],
                                 np.full(n_uavs, float(z_init_proposed))])
    airship_pos = np.array([airship_x, airship_y, airship_z], float)
    xaxis_all = exo['xaxis']
    yaxis_all = exo['yaxis']
    arrival = exo['arrival']
    tr_type = exo['tr_type']
    tr_state = exo['tr_state']

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


# --------------------------------------------------------------- KPI helpers
def wm(res, slice_=None):
    sl = slice(None) if slice_ is None else slice(*slice_)
    parts = [W_MOS[0] * np.nanmean(res['mos_web'][sl]),
             W_MOS[1] * np.nanmean(res['mos_vid'][sl]),
             W_MOS[2] * np.nanmean(res['mos_msg'][sl])]
    return float(np.sum(parts))


def decay_time(series, thr=1.0):
    idx = np.where(np.nan_to_num(series, nan=0.0) > thr)[0]
    return int(idx.max() + 1) if len(idx) else 0


KPI_FUNCS = [
    ('wmos_full', lambda r: wm(r)),
    ('wmos_120', lambda r: wm(r, (0, 120))),
    ('web_mos_full', lambda r: float(np.nanmean(r['mos_web']))),
    ('web_mos_120', lambda r: float(np.nanmean(r['mos_web'][:120]))),
    ('qlen_act_mean', lambda r: float(np.nanmean(r['qlen_act']))),
    ('web_q_mean', lambda r: float(np.nanmean(r['web_qlen_all']))),
    ('web_q_peak', lambda r: float(np.nanmax(r['web_qlen_max']))),
    ('web_q_decay', lambda r: float(decay_time(r['web_qlen_all']))),
    ('sum_D', lambda r: float(np.mean(r['sum_D']))),
    ('bh_cap', lambda r: float(np.mean(r['sum_Rbh']))),
    ('acc_cap', lambda r: float(np.mean(r['sum_Racc']))),
    ('bh_util', lambda r: float(np.mean(r['bh_util']))),
    ('win_stay', lambda r: float((r['win'] == 0).mean())),
    ('win_beta', lambda r: float((r['win'] == 1).mean())),
    ('win_dual', lambda r: float((r['win'] == 2).mean())),
]


# ---------------------------------------------------------------- main loop
all_res = {}          # all_res[seed][variant] = res dict
for s in SEEDS:
    print(f'\n================ SEED {s} ================', flush=True)
    exo = make_inputs(s)
    print(f'  inputs: x[{exo["xaxis"].shape}] arr_mean={exo["arrival"].mean():.3f} '
          f'active_rate={(exo["arrival"] > 0).mean():.3f}', flush=True)
    sr = {}
    for variant in VARIANTS:
        sr[variant] = run_scheme(variant, exo, s)
    h0, h1, h2 = (sr[v]['cids_hash'] for v in VARIANTS)
    ok_pair = bool((h0 == h1).all()) and bool((h0 == h2).all())
    print(f'\n[cids hash seed={s}] paired across schemes: {ok_pair}', flush=True)
    all_res[s] = sr

    # incremental save
    np.savez_compressed(os.path.join(OUT, f'stage2_seed{s}.npz'),
                        **{f'{v}_{k}': sr[v][k] for v in VARIANTS
                           for k in sr[v] if k != 'variant'})

# ------------------------------------------------- seed-0 harness repro (stage1)
s0 = all_res[0]
st1 = np.load(STAGE1_NPZ)
q_d = float(np.nanmax(np.abs(s0['k3']['qlen_act'] - st1['k3_qlen_act'])))
c_d = float(np.nanmax(np.abs(s0['k3']['chcapa'] - st1['k3_chcapa'])))
w_d = float(np.nanmax(np.abs(s0['k3']['wmos'] - st1['k3_wmos'])))
print(f'\n[seed0 k3 vs stage1] qlen max|d|={q_d:.3e}  chcapa max|d|={c_d:.3e}  '
      f'wmos max|d|={w_d:.3e}')
print(f'[seed0 k3 harness repro] -> {"PASS" if max(q_d, c_d, w_d) <= 1e-9 else "FAIL"}')

# ------------------------------------------------------- cross-seed KPI table
rows = []
for s in SEEDS:
    for v in VARIANTS:
        r = all_res[s][v]
        row = dict(seed=s, variant=v)
        for name, f in KPI_FUNCS:
            row[name] = f(r)
        row['solver_bad'] = int(r['solver_bad'])
        row['wall_s'] = float(r['wall'])
        if v == 'dualk5':
            row['dual_win_frac'] = float((r['win'] == 2).mean())
            row['id_err_max'] = float(np.nanmax(r['decomp_err']))
            ok = np.isfinite(r['decomp_err']) & r['dual_win']
            row['dJ_dualwin'] = float(np.nanmean(r['dJ_g'][ok])) if ok.any() else np.nan
            row['air_zero'] = int(r['air_zero_slots'])
            row['cancel'] = int(r['cancel_total'])
            row['E_final_max'] = float(np.max(r['E_final']))
        row['exec_dev_max'] = float(r['exec_dev'].max())
        row['vxy_max'] = float(r['vxy'].max())
        row['vz_max'] = float(r['vz'].max())
        row['vair_max'] = float(r['vair'].max())
        rows.append(row)
kpi_df = pd.DataFrame(rows)
kpi_df.to_csv(os.path.join(OUT, 'stage2_kpi_per_seed.csv'), index=False)

metric_names = [n for n, _ in KPI_FUNCS]
print('\n==================== Stage 2: multi-seed (501 slots, seeds 0-4) ====================')
print(f'{"metric":22s} ' + ' '.join(f'{v:>22s}' for v in VARIANTS))
for m in metric_names:
    cells = []
    for v in VARIANTS:
        vals = kpi_df.loc[kpi_df.variant == v, m].to_numpy(float)
        cells.append(f'{vals.mean():+.4f} ±{vals.std(ddof=1):.4f}')
    print(f'{m:22s} ' + ' '.join(f'{c:>22s}' for c in cells))

# paired deltas: dualk5 - k3, dualk5 - betak5 (same seed)
print('\n--- paired deltas (same seed) ---')
piv = kpi_df.pivot(index='seed', columns='variant')
for m in ('wmos_full', 'wmos_120', 'web_mos_full', 'web_mos_120',
          'web_q_mean', 'web_q_peak', 'web_q_decay', 'bh_cap', 'acc_cap'):
    dk3 = (piv[m]['dualk5'] - piv[m]['k3']).to_numpy()
    dbk = (piv[m]['dualk5'] - piv[m]['betak5']).to_numpy()
    print(f'  {m:16s} dual-k3: mean={dk3.mean():+.4f}  '
          f'[{", ".join(f"{x:+.3f}" for x in dk3)}]  '
          f'dual-betaK5: mean={dbk.mean():+.4f}')

# sign consistency
w_dk3 = (piv['wmos_full']['dualk5'] > piv['wmos_full']['k3']).sum()
w_dbk = (piv['wmos_full']['dualk5'] > piv['wmos_full']['betak5']).sum()
q_dk3 = (piv['web_q_mean']['dualk5'] < piv['web_q_mean']['k3']).sum()
print(f'  dualk5 wMOS > k3 in {w_dk3}/5 seeds; > betak5 in {w_dbk}/5; '
      f'web_q lower than k3 in {q_dk3}/5')

# Check 1 / 2 across seeds
print('\n--- Check 1 (dualk5 identity, max over seeds) ---')
print('  id_err_max per seed: '
      + ', '.join(f'{x:.2e}' for x in kpi_df.loc[kpi_df.variant == 'dualk5', 'id_err_max']))
print(f'  dual_win_frac per seed: '
      f'{kpi_df.loc[kpi_df.variant == "dualk5", "dual_win_frac"].round(3).tolist()}')
print(f'  dJ on dual-win per seed: '
      f'{kpi_df.loc[kpi_df.variant == "dualk5", "dJ_dualwin"].round(4).tolist()}')
print('\n--- Check 2 (feasibility, max over seeds) ---')
for v in VARIANTS:
    sub = kpi_df[kpi_df.variant == v]
    print(f'  [{v}] exec_dev_max={sub.exec_dev_max.max():.3f}  '
          f'vxy_max={sub.vxy_max.max():.2f}  vz_max={sub.vz_max.max():.2f}  '
          f'vair_max={sub.vair_max.max():.2f}  '
          f'solver_bad_sum={int(sub.solver_bad.sum())}')
dsub = kpi_df[kpi_df.variant == 'dualk5']
print(f'  [dualk5] E_final_max per seed={dsub.E_final_max.round(1).tolist()}  '
      f'air_zero_sum={int(dsub.air_zero.sum())}  cancel_sum={int(dsub.cancel.sum())}')
print(f'  wall per seed (k3/beta/dual): '
      + '; '.join(f'{s}: {all_res[s]["k3"]["wall"]:.0f}/'
                  f'{all_res[s]["betak5"]["wall"]:.0f}/'
                  f'{all_res[s]["dualk5"]["wall"]:.0f}s' for s in SEEDS))

# ------------------------------------------------------------------ figures
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
ax = axes[0, 0]
for v, c in (('k3', 'gray'), ('betak5', 'tab:blue'), ('dualk5', 'tab:orange')):
    arr = np.stack([all_res[s][v]['wmos'] for s in SEEDS])
    mu, sd = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
    ax.plot(mu, color=c, lw=1.4, label=v)
    ax.fill_between(range(arr.shape[1]), mu - sd, mu + sd, color=c, alpha=0.15)
ax.set_title('weighted MOS (mean±sd over 5 seeds)'); ax.legend(fontsize=8)
ax.set_xlabel('slot')
ax = axes[0, 1]
for v, c in (('k3', 'gray'), ('betak5', 'tab:blue'), ('dualk5', 'tab:orange')):
    arr = np.stack([all_res[s][v]['mos_web'] for s in SEEDS])
    mu, sd = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
    ax.plot(mu, color=c, lw=1.4, label=v)
    ax.fill_between(range(arr.shape[1]), mu - sd, mu + sd, color=c, alpha=0.15)
ax.set_title('Web MOS (mean±sd)'); ax.legend(fontsize=8); ax.set_xlabel('slot')
ax = axes[0, 2]
for v, c in (('k3', 'gray'), ('betak5', 'tab:blue'), ('dualk5', 'tab:orange')):
    arr = np.stack([all_res[s][v]['web_qlen_all'] for s in SEEDS])
    mu, sd = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
    ax.plot(mu, color=c, lw=1.4, label=v)
    ax.fill_between(range(arr.shape[1]), mu - sd, mu + sd, color=c, alpha=0.15)
ax.set_title('Web queue all (mean±sd, Mb)'); ax.legend(fontsize=8); ax.set_xlabel('slot')
ax = axes[1, 0]
w = 0.25
xs = np.arange(len(SEEDS))
for i, (v, c) in enumerate((('k3', 'gray'), ('betak5', 'tab:blue'), ('dualk5', 'tab:orange'))):
    ax.bar(xs + (i - 1) * w, piv['wmos_full'][v], width=w, color=c, label=v)
ax.set_xticks(xs); ax.set_xticklabels(SEEDS); ax.set_xlabel('seed')
ax.set_title('wMOS full per seed'); ax.legend(fontsize=8)
ax = axes[1, 1]
for i, (v, c) in enumerate((('k3', 'gray'), ('betak5', 'tab:blue'), ('dualk5', 'tab:orange'))):
    ax.bar(xs + (i - 1) * w, piv['web_q_mean'][v], width=w, color=c, label=v)
ax.set_xticks(xs); ax.set_xticklabels(SEEDS); ax.set_xlabel('seed')
ax.set_title('Web queue mean per seed (Mb)'); ax.legend(fontsize=8)
ax = axes[1, 2]
dk3 = (piv['wmos_full']['dualk5'] - piv['wmos_full']['k3']).to_numpy()
dbk = (piv['wmos_full']['dualk5'] - piv['wmos_full']['betak5']).to_numpy()
ax.bar(xs - w / 2, dk3, width=w, color='tab:orange', label='dual-k3')
ax.bar(xs + w / 2, dbk, width=w, color='tab:green', label='dual-betak5')
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels(SEEDS); ax.set_xlabel('seed')
ax.set_title('paired ΔwMOS (dual − control)'); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'stage2_results.png'), dpi=130)
print(f'\nwrote {OUT}\\stage2_kpi_per_seed.csv, stage2_seed*.npz, stage2_results.png')

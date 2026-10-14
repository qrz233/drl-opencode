# -*- coding: utf-8 -*-
"""Task 3: Binding-guided K5 closed-loop (seed0 / seed2, 501 slots).

Identical to dualk5 from expB_stage2.py EXCEPT the UAV guidance weight:
  dualk5 : w_u = nu_u / (c_nu + nu_u)          (continuous, c_nu=0.1)
  binding: w_u = 1 if binding else 0            (binary tight-binding flag)

Binding definition (from ablation calibration, frozen):
  binding_u = (nu_u > 1e-6)                     (dual active on the backhaul)
  (slack alternative not used: stage2 solve only exposes duals; on the test
   set nu>1e-6 matched bind indicators exactly: corr=1.0, crosstab 248/2)

Everything else identical to dualk5:
  candidates = [stay, beta-0.25, beta-0.5, dual-0.25, dual-0.5]
  airship gradient = same dual gradient
  Block A eval, J = psi_E*(E·P) - beta·D, E update, seeds/protocol same

Reuses stage2 dualk5 npz for paired comparison (no re-run of dualk5).
"""
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
OUT = r'D:\Learn\work\drl-opencode\expB_results\task3_binding'
os.makedirs(OUT, exist_ok=True)

from params_sys import *
from utils_sys import (kmeans_clustering, find_active_users, update_queue_Mb,
                       cal_channel_fading_gain, cal_uav_propulsion_energy, mos_func)
import blocka_optimizer as B
import traffic_model as TM
from mobility_model import adjust_location, adjust_altitude, generate_mobility_SFM

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
SEEDS = [0, 2]
BIND_EPS = 1e-6


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


def make_inputs(s):
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


class BindingK5Controller:
    '''dualk5 clone with binary binding weight instead of continuous nu weight.'''

    def __init__(self, n_uavs_):
        self.M = n_uavs_
        self.E = np.zeros(n_uavs_)
        self.psi_Q = 1.0 / float(queue_norm_Mb)
        self.tau = TAU
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
        ctx, st = self.ctx, self.st
        g_acc, g_bh = channel_gammas(ctx['user_x'], ctx['user_y'], ctx['cids'],
                                     st['uav_locs'], st['airship_pos'],
                                     ctx['beta'], ctx['demand'], ctx['types'])
        base = solve_blocka_diag(g_acc, g_bh, ctx['demand'], ctx['cids'], ctx['beta'])
        nu = np.maximum(np.nan_to_num(base['nu_full'], nan=0.0), 0.0)
        # BINARY binding weight (the only change vs dualk5)
        w = np.where(nu > BIND_EPS, 1.0, 0.0)
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
        return e_u, e_a, w, n_cancel, (nrm_a <= DIR_EPS), nu

    def decide_and_score(self):
        M = self.M
        e_u_beta, e_a_beta = self._dirs()
        a0 = np.zeros(3 * M + 2)
        evs, cats, plan_kinds = [], [], []
        evs.append(self._eval(a0))
        cats.append('stay')
        plan_kinds.append(('stay', 0.0))
        plan = [('beta', 0.25), ('beta', 0.5)]
        extra = {}
        e_u_d, e_a_d, w, n_cancel, air_zero, nu = self._dual_dirs()
        plan = plan + [('dual', 0.25), ('dual', 0.5)]
        extra = dict(w=w, n_cancel=n_cancel, air_zero=air_zero,
                     e_u_d=e_u_d, e_a_d=e_a_d, nu=nu)
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


def run_binding(seed, exo):
    print(f'\n===== Binding-K5 seed={seed} =====', flush=True)
    np.random.seed(seed)
    ctrl = BindingK5Controller(n_uavs)
    Q = np.zeros(n_users)
    ema = np.zeros(n_users)
    prev_locs = init_locations_uav[:, [0, 1]].copy()
    prev_locs = np.column_stack([prev_locs, np.full(n_uavs, float(z_init_proposed))])
    airship_pos = np.array([airship_x, airship_y, airship_z], float)

    mos_web = np.full(S, np.nan); mos_vid = np.full(S, np.nan); mos_msg = np.full(S, np.nan)
    wmos = np.full(S, np.nan); qlen_act = np.full(S, np.nan)
    web_qlen = np.full(S, np.nan); web_qlen_all = np.full(S, np.nan); web_qlen_max = np.full(S, np.nan)
    sum_D = np.zeros(S); sum_Rbh = np.zeros(S); sum_Racc = np.zeros(S)
    chcapa = np.full(S, np.nan); bh_util = np.zeros(S)
    J_star = np.zeros(S); win = np.zeros(S, dtype=int)
    dual_win = np.zeros(S, dtype=bool)
    bind_frac = np.zeros(S)           # fraction of UAVs with binding=1
    P_hist = np.zeros((S, n_uavs)); E_hist = np.zeros((S, n_uavs))
    vxy_max = np.zeros(S); vz_max = np.zeros(S); vair_max = np.zeros(S)
    cids_hash = np.zeros(S, dtype=np.int64)
    air_zero_slots = 0; cancel_total = 0; solver_bad = 0

    t_wall = time.time()
    for t in range(S):
        Q_prev = Q.copy()
        A_t = exo['arrival'][:, t]
        types_t = exo['tr_type'][:, t]
        isActive = find_active_users(queue_len_Mb=Q_prev, incoming_traffic_Mb=A_t)
        loc_users = np.hstack((exo['xaxis'][:, [t]], exo['yaxis'][:, [t]],
                               np.zeros((n_users, 1))))
        _, cids, _ = kmeans_clustering(loc_users)
        cids_hash[t] = zlib.crc32(np.ascontiguousarray(cids).tobytes())

        beta = ctrl.begin_slot(user_x=exo['xaxis'][:, t], user_y=exo['yaxis'][:, t],
                               Q=Q_prev, A=A_t, types=types_t, cids=cids,
                               uav_locs=prev_locs, airship_pos=airship_pos)
        best = ctrl.decide_and_score()
        evs = best['evs']
        solver_bad += sum(1 for e in evs
                          if e['res']['status'] not in ('optimal', 'optimal_inaccurate',
                                                        'trivial-no-demand'))
        exec_locs = np.array([[adjust_location(best['locs'][u, 0], bound=boundary),
                               adjust_location(best['locs'][u, 1], bound=boundary),
                               adjust_altitude(best['locs'][u, 2], zmin=z_min, zmax=z_max)]
                              for u in range(n_uavs)])
        vxy_max[t] = float(np.max(np.linalg.norm(best['v'][:, :2], axis=1)))
        vz_max[t] = float(np.max(np.abs(best['v'][:, 2])))
        va = (best['air'] - ctrl.st['airship_pos'])[:2] / TAU
        vair_max[t] = float(np.linalg.norm(va))

        cat = best['cats'][best['k']]
        win[t] = 0 if cat == 'stay' else (1 if cat == 'beta' else 2)
        dual_win[t] = (cat == 'dual')
        air_zero_slots += int(best['extra']['air_zero'])
        cancel_total += int(best['extra']['n_cancel'])
        bind_frac[t] = float(np.mean(best['extra']['w']))

        prev_locs = exec_locs.copy()
        airship_pos = best['air'].copy()
        ctrl.update_energy(P_exec=best['P'])
        D_opt = best['res']['D_Mb']
        R_access = best['res']['R_access_Mb']
        S_u = best['res']['S_u']
        R_bh = best['res']['R_bh_Mb']
        Q = np.maximum(Q_prev + A_t - D_opt, 0.0)
        ema = 0.999 * Q + 0.001 * ema

        tt = types_t
        for k_type, name in ((0, 'mos_web'), (1, 'mos_vid'), (2, 'mos_msg')):
            m = (tt == k_type) & exo['tr_state'][:, t]
            if np.any(m):
                mos = mos_func(qlen=ema[m], lambda_=ON_data_arrival_mean_Mb[k_type],
                               ss_min_arr=ss_min_types[k_type],
                               ss_max_arr=ss_max_types[k_type])
                val = float(np.mean(mos))
                if name == 'mos_web': mos_web[t] = val
                elif name == 'mos_vid': mos_vid[t] = val
                else: mos_msg[t] = val
        parts = [W_MOS[0]*mos_web[t], W_MOS[1]*mos_vid[t], W_MOS[2]*mos_msg[t]]
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
            print(f'  t={t+1}/{S}  {time.time()-t_wall:.0f}s  '
                  f'wmos={np.nanmean(wmos):.3f}  Qweb={np.nanmean(web_qlen):.2f}  '
                  f'bind={bind_frac[t]:.2f}', flush=True)

    wall = time.time() - t_wall
    print(f'  done in {wall:.0f}s  solver_bad={solver_bad}', flush=True)
    return dict(seed=seed, wall=wall, mos_web=mos_web, mos_vid=mos_vid, mos_msg=mos_msg,
                wmos=wmos, qlen_act=qlen_act, web_qlen=web_qlen, web_qlen_all=web_qlen_all,
                web_qlen_max=web_qlen_max, sum_D=sum_D, sum_Rbh=sum_Rbh, sum_Racc=sum_Racc,
                chcapa=chcapa, bh_util=bh_util, J_star=J_star, win=win, dual_win=dual_win,
                bind_frac=bind_frac, P_hist=P_hist, E_hist=E_hist,
                vxy=vxy_max, vz=vz_max, vair=vair_max, cids_hash=cids_hash,
                air_zero_slots=air_zero_slots, cancel_total=cancel_total,
                solver_bad=solver_bad, E_final=ctrl.E.copy())


def wm(res, sl=None):
    sl = slice(None) if sl is None else sl
    return float(np.nansum([W_MOS[0]*np.nanmean(res['mos_web'][sl]),
                            W_MOS[1]*np.nanmean(res['mos_vid'][sl]),
                            W_MOS[2]*np.nanmean(res['mos_msg'][sl])]))


all_res = {}
for s in SEEDS:
    exo = make_inputs(s)
    print(f'\n==== SEED {s} ====', flush=True)
    all_res[s] = run_binding(s, exo)

# load dualk5 for comparison
print('\nloading stage2 dualk5 for paired comparison...', flush=True)
rows = []
for s in SEEDS:
    st2 = np.load(rf'D:\Learn\work\drl-opencode\expB_results\stage2_multiseed\stage2_seed{s}.npz')
    dual = {k.replace('dualk5_', ''): st2[k] for k in st2.files if k.startswith('dualk5_')}
    b = all_res[s]
    cids_eq = bool((b['cids_hash'] == dual['cids_hash']).all())
    print(f'\n--- seed {s}: Binding vs Dual ---')
    print(f'  cids paired: {cids_eq}')
    for label, arr_b, arr_d in (
        ('wMOS full', wm(b), wm(dual)),
        ('wMOS [0:120]', wm(b, slice(0, 120)), wm(dual, slice(0, 120))),
        ('web_mos', np.nanmean(b['mos_web']), np.nanmean(dual['mos_web'])),
        ('web_q_mean', np.nanmean(b['web_qlen_all']), np.nanmean(dual['web_qlen_all'])),
        ('web_q_peak', np.nanmax(b['web_qlen_max']), np.nanmax(dual['web_qlen_max'])),
        ('sum_D', np.mean(b['sum_D']), np.mean(dual['sum_D'])),
        ('bh_cap', np.mean(b['sum_Rbh']), np.mean(dual['sum_Rbh'])),
        ('acc_cap', np.mean(b['sum_Racc']), np.nanmean(dual['sum_Racc'])),
        ('bh_util', np.mean(b['bh_util']), np.mean(dual['bh_util'])),
        ('P_mean', np.mean(b['P_hist']), np.mean(dual['P_hist'])),
        ('P_max', np.max(b['P_hist']), np.max(dual['P_hist'])),
        ('E_final_max', np.max(b['E_final']), np.max(dual['E_final'])),
    ):
        print(f'  {label:14s}  bind={arr_b:+.4f}  dual={arr_d:+.4f}  d={arr_b-arr_d:+.4f}')
    print(f'  win stay/beta/dual: bind {(b["win"]==0).mean():.2f}/{(b["win"]==1).mean():.2f}/{(b["win"]==2).mean():.2f}'
          f'  dual {(dual["win"]==0).mean():.2f}/{(dual["win"]==1).mean():.2f}/{(dual["win"]==2).mean():.2f}')
    print(f'  bind_frac mean={b["bind_frac"].mean():.3f}  '
          f'vxy_max={b["vxy"].max():.2f}  vz_max={b["vz"].max():.2f}  '
          f'vair_max={b["vair"].max():.2f}  solver_bad={b["solver_bad"]}')
    # paired win rate on wMOS (slot-wise)
    both = np.isfinite(b['wmos']) & np.isfinite(dual['wmos'])
    pw = float((b['wmos'][both] > dual['wmos'][both]).mean())
    print(f'  slot-wise wMOS win rate (bind > dual): {pw:.3f}')
    rows.append(dict(seed=s, wmos_bind=wm(b), wmos_dual=wm(dual),
                     wmos_120_bind=wm(b, slice(0, 120)), wmos_120_dual=wm(dual, slice(0, 120)),
                     web_q_bind=np.nanmean(b['web_qlen_all']),
                     web_q_dual=np.nanmean(dual['web_qlen_all']),
                     sum_D_bind=np.mean(b['sum_D']), sum_D_dual=np.mean(dual['sum_D']),
                     P_mean_bind=np.mean(b['P_hist']), P_mean_dual=np.mean(dual['P_hist']),
                     bind_frac=b['bind_frac'].mean(),
                     dual_win_bind=float((b['win']==2).mean()),
                     dual_win_dual=float((dual['win']==2).mean()),
                     slot_win_rate=pw, cids_paired=cids_eq))

pd.DataFrame(rows).to_csv(os.path.join(OUT, 'binding_vs_dual.csv'), index=False)
np.savez_compressed(os.path.join(OUT, 'task3_binding.npz'),
                    **{f'seed{s}_{k}': all_res[s][k] for s in SEEDS for k in all_res[s] if k != 'seed'})

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for s, c in ((0, 'tab:orange'), (2, 'tab:red')):
    st2 = np.load(rf'D:\Learn\work\drl-opencode\expB_results\stage2_multiseed\stage2_seed{s}.npz')
    dual = {k.replace('dualk5_', ''): st2[k] for k in st2.files if k.startswith('dualk5_')}
    b = all_res[s]
    axes[0, 0].plot(b['wmos'], color=c, lw=1.2, label=f'bind s{s}')
    axes[0, 0].plot(dual['wmos'], color=c, lw=1.0, ls='--', alpha=0.7, label=f'dual s{s}')
    axes[0, 1].plot(b['web_qlen_all'], color=c, lw=1.2)
    axes[0, 1].plot(dual['web_qlen_all'], color=c, lw=1.0, ls='--', alpha=0.7)
    axes[0, 2].plot(b['sum_D'], color=c, lw=1.2)
    axes[0, 2].plot(dual['sum_D'], color=c, lw=1.0, ls='--', alpha=0.7)
    axes[1, 0].plot(b['P_hist'].mean(axis=1), color=c, lw=1.2)
    axes[1, 0].plot(dual['P_hist'].mean(axis=1), color=c, lw=1.0, ls='--', alpha=0.7)
    axes[1, 1].plot(b['E_hist'].max(axis=1), color=c, lw=1.2)
    axes[1, 1].plot(dual['E_hist'].max(axis=1), color=c, lw=1.0, ls='--', alpha=0.7)
    axes[1, 2].plot(b['bind_frac'], color=c, lw=1.2)
axes[0, 0].set_title('wMOS solid=bind dashed=dual'); axes[0, 0].legend(fontsize=7)
axes[0, 1].set_title('Web queue all (Mb)')
axes[0, 2].set_title('sum D')
axes[1, 0].set_title('mean P (W)'); axes[1, 0].axhline(PW_THRES, color='k', ls=':', lw=0.8)
axes[1, 1].set_title('max E_u')
axes[1, 2].set_title('bind_frac (fraction UAVs binding)')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'binding_vs_dual.png'), dpi=130)
print(f'\nwrote {OUT}')

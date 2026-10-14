# -*- coding: utf-8
'''Experiment B Task 3: Hybrid baseline (fixed eval/exec) vs dualk5.

Hybrid = original paper Proposed method (HybridCandidateController):
  actor proposes K=20 candidates, each scored with true model
  (channel -> _solve_cluster bisection delivery -> Lyapunov J),
  best candidate executed via motion() (eval == exec).
  REINFORCE online training (t_training_start=0, interval=5).
  Fixed 30/60/10 type-bandwidth pools (NOT Block A).

Fix verified in working tree:
  relay_hybrid_controller.motion() unifies candidate evaluation and
  notebook execution (HEAD had eval: c*speed_xy displacement vs exec:
  raw best_a treated as m/s).

Pairing with dualk5 (stage2): identical exogenous inputs AND identical
kmeans cids stream (precomputed with np.random.seed(seed) before any
Hybrid RNG use). Hybrid's own candidate noise uses a separate seed.

Seeds: 0 and 2. Horizon: 501 slots. Compare vs stage2 dualk5 npz.

Outputs -> D:\\Learn\\work\\drl-opencode\\expB_results\\task3_hybrid
'''
import os, sys, time, zlib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
OUT = r'D:\Learn\work\drl-opencode\expB_results\task3_hybrid'
os.makedirs(OUT, exist_ok=True)
STAGE2_DIR = r'D:\Learn\work\drl-opencode\expB_results\stage2_multiseed'

from params_sys import *
from utils_sys import (kmeans_clustering, find_active_users, update_queue_Mb,
                       cal_channel_fading_gain, cal_uav_propulsion_energy,
                       mos_func, gen_heatmap, update_vqlen_prop)
import traffic_model as TM
from mobility_model import (adjust_location, adjust_altitude,
                            generate_mobility_SFM)
from relay_hybrid_controller import HybridCandidateController

W_MOS = np.asarray(mos_type_weights, dtype=float)
SEEDS = [0, 2]
S = int(n_slots)
HY_N_CAND, HY_SIGMA = 20, 1.1
HYB_SEED = 0          # torch/actor seed (matches notebook)


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
print('frozen inputs loaded', _frozen['xaxis'].shape, _frozen['arrival'].shape, flush=True)


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


def precompute_cids(exo, seed):
    '''Identical kmeans RNG stream as expB_stage2.run_scheme.'''
    np.random.seed(seed)
    xaxis_all, yaxis_all = exo['xaxis'], exo['yaxis']
    cids_all = np.zeros((n_slots, n_users), dtype=int)
    for t in range(n_slots):
        loc_users = np.hstack((xaxis_all[:, [t]], yaxis_all[:, [t]],
                               np.zeros((n_users, 1))))
        _, cids, _ = kmeans_clustering(loc_users)
        cids_all[t] = cids
    return cids_all


def run_hybrid(exo, cids_all, seed):
    print(f'\n===== hybrid (seed={seed}, K={HY_N_CAND}, sigma={HY_SIGMA}) =====', flush=True)
    np.random.seed(1000 + seed)   # Hybrid candidate-noise stream (separate from kmeans)
    torch.manual_seed(HYB_SEED)
    ctrl = HybridCandidateController(
        n_users=n_users, n_uavs=n_uavs, boundary=boundary, grid_size=grid_size,
        queue_norm=queue_norm_Mb, V_lyap=lyapunov_param, tau=slot_len,
        ss_min_types=ss_min_types, ss_max_types=ss_max_types,
        mos_weights=mos_type_weights, type_rates=ON_data_arrival_mean_Mb, d0=d0,
        sigma_decay_slots=n_slots, seed=HYB_SEED,
        n_candidates=HY_N_CAND, eval_sigma=HY_SIGMA)
    ctrl.bind_models(cal_channel_fading_gain, cal_backhaul_capacity_Mb,
                     pTx_downlink, noise_pw_total)

    Q = np.zeros(n_users)
    ema = np.zeros(n_users)
    vqlen_prop = np.zeros(n_uavs)
    uav_locs = np.column_stack([init_locations_uav[:, 0], init_locations_uav[:, 1],
                                np.full(n_uavs, float(z_init_proposed))])
    airship_pos = np.array([airship_x, airship_y, airship_z], float)
    xaxis_all, yaxis_all = exo['xaxis'], exo['yaxis']
    arrival, tr_type, tr_state = exo['arrival'], exo['tr_type'], exo['tr_state']

    n_t = S
    mos_web = np.full(n_t, np.nan)
    mos_vid = np.full(n_t, np.nan)
    mos_msg = np.full(n_t, np.nan)
    wmos = np.full(n_t, np.nan)
    qlen_act = np.full(n_t, np.nan)
    web_qlen = np.full(n_t, np.nan)
    web_qlen_all = np.full(n_t, np.nan)
    web_qlen_max = np.full(n_t, np.nan)
    sum_D = np.zeros(n_t)
    sum_Rbh = np.zeros(n_t)
    sum_Racc = np.zeros(n_t)
    bh_util = np.zeros(n_t)
    J_best = np.full(n_t, np.nan)
    P_hist = np.zeros((n_t, n_uavs))
    pw_hist = np.zeros((n_t, n_uavs))
    vxy_max_hist = np.zeros(n_t)
    vz_max_hist = np.zeros(n_t)
    exec_dev = np.zeros(n_t)
    bh_excess = np.zeros(n_t)
    cids_hash = np.zeros(n_t, dtype=np.int64)
    train_losses = []

    t_wall = time.time()
    for t in range(n_t):
        Q_prev = Q.copy()
        A_t = arrival[:, t]
        types_t = tr_type[:, t]
        isActive = find_active_users(queue_len_Mb=Q_prev, incoming_traffic_Mb=A_t)
        cids = cids_all[t]
        cids_hash[t] = zlib.crc32(np.ascontiguousarray(cids).tobytes())

        # bandwidth pools 30/60/10 (cell-26 semantics)
        bw_alloc_Hz_all = np.zeros(n_users)
        ttype_all_t = types_t
        for k_type in range(len(bw_type_pool_Hz)):
            ids_k = np.where(ttype_all_t == k_type)[0]
            active_k = ids_k[isActive[ids_k]]
            if len(active_k) > 0:
                bw_alloc_Hz_all[active_k] = bw_type_pool_Hz[k_type] / len(active_k)
        ld_Mbps = np.array([0 if (types_t[i] == -1) else ON_data_arrival_mean_Mb[types_t[i]]
                            for i in range(n_users)])

        heatmap_t = gen_heatmap(xaxis_all[:, t], yaxis_all[:, t],
                                Q_prev + A_t, queue_norm_Mb)
        ctrl.begin_slot(heatmap_t, uav_locs, airship_pos)

        new_locs = np.zeros_like(uav_locs)
        P_exec = np.zeros(n_uavs)
        for k in range(n_uavs):
            best_a, best_J = ctrl.decide_and_score(
                k=k, uav_loc=uav_locs[k], airship_loc=airship_pos,
                heatmap=heatmap_t, user_x=xaxis_all[:, t], user_y=yaxis_all[:, t],
                bw_alloc_Hz=bw_alloc_Hz_all, Q=Q_prev, A=A_t, ld=ld_Mbps,
                types=types_t, cids=cids,
                speed_xy=uav_speedxy_max, speed_z=uav_speedz_max,
                zmin=z_min, zmax=z_max, boundary=boundary)
            pos_t, vel_t = ctrl.motion(best_a, uav_locs[k])
            xt, yt, zt = pos_t
            vx, vy, vz = vel_t
            new_locs[k] = [adjust_location(xt, bound=boundary),
                           adjust_location(yt, bound=boundary),
                           adjust_altitude(zt, zmin=z_min, zmax=z_max)]
            exec_dev[t] = max(exec_dev[t], float(np.max(np.abs(new_locs[k] - pos_t))))
            vxy_max_hist[t] = max(vxy_max_hist[t], float(np.hypot(vx, vy)))
            vz_max_hist[t] = max(vz_max_hist[t], float(abs(vz)))
            pw_hist[t, k] = float(cal_uav_propulsion_energy(V=np.hypot(vx, vy), Vz=vz)[0])
            P_exec[k] = pw_hist[t, k]
            vqlen_prop[k] = update_vqlen_prop(vqlen_prop[k], pw_hist[t, k]) if t > 0 else 0.0
            if k == 0:
                J_best[t] = best_J

        if (t >= t_training_start) and (t % training_interval == 0):
            loss = ctrl.train_step()
            if loss is not None:
                train_losses.append(loss)

        # access rates at new positions
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
        bh_excess[t] = float(np.sum(S_u) - np.sum(R_bh))
        bh_util[t] = float(np.sum(S_u) / max(np.sum(R_bh), 1e-9))

        # user updates
        mos_post = np.full(n_users, np.nan)
        for uid in range(n_users):
            cid = cids[uid]
            qlen_prev_Mb = Q_prev[uid]
            Q[uid] = update_queue_Mb(qlen_prev_Mb, A_t[uid], D_opt[uid])
            ema[uid] = 0.999 * Q[uid] + 0.001 * ema[uid]
            ttype_t = int(types_t[uid])
            if ttype_t < 0:
                continue
            ld_t = ON_data_arrival_mean_Mb[ttype_t]
            if tr_state[uid, t]:
                mos_post[uid] = mos_func(qlen=max(qlen_prev_Mb + A_t[uid] - D_opt[uid], 0.0),
                                         lambda_=ld_t,
                                         ss_min_arr=ss_min_types[ttype_t],
                                         ss_max_arr=ss_max_types[ttype_t])

        reward_t = ctrl.compute_reward(Q=Q_prev, A=A_t, D=D_opt, S_u=S_u,
                                       R_bh=R_bh, mos_mean=np.nanmean(mos_post))
        ctrl.commit_slot(reward_t)
        ctrl.update_virtual_queues(S_u=S_u, R_bh=R_bh)

        uav_locs = new_locs

        # KPIs (same as stage2)
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
        P_hist[t] = P_exec

        if (t + 1) % 100 == 0:
            print(f'  t={t + 1}/{n_t}  {time.time() - t_wall:.0f}s  '
                  f'wmos={np.nanmean(wmos):.3f}  Qweb={np.nanmean(web_qlen):.2f}  '
                  f'loss={train_losses[-1] if train_losses else float("nan"):.4f}', flush=True)

    wall = time.time() - t_wall
    res = dict(variant='hybrid', wall=wall, mos_web=mos_web, mos_vid=mos_vid,
               mos_msg=mos_msg, wmos=wmos, qlen_act=qlen_act, web_qlen=web_qlen,
               web_qlen_all=web_qlen_all, web_qlen_max=web_qlen_max,
               sum_D=sum_D, sum_Rbh=sum_Rbh, sum_Racc=sum_Racc, bh_util=bh_util,
               J_best=J_best, P_hist=P_hist, pw_hist=pw_hist,
               vxy=vxy_max_hist, vz=vz_max_hist, exec_dev=exec_dev,
               bh_excess=bh_excess, cids_hash=cids_hash,
               train_losses=np.array(train_losses) if train_losses else np.zeros(0),
               E_final=ctrl.Z.copy())
    print(f'  done in {wall:.0f}s  '
          f'max|v_xy|={vxy_max_hist.max():.2f}  max|vz|={vz_max_hist.max():.2f}  '
          f'max_pw={pw_hist.max():.1f}  mean_pw={pw_hist.mean():.1f}  '
          f'max_bh_excess={bh_excess.max():.4f}  exec_dev={exec_dev.max():.3e}', flush=True)
    return res


def wm(res, slice_=None):
    sl = slice(None) if slice_ is None else slice(*slice_)
    parts = [W_MOS[0] * np.nanmean(res['mos_web'][sl]),
             W_MOS[1] * np.nanmean(res['mos_vid'][sl]),
             W_MOS[2] * np.nanmean(res['mos_msg'][sl])]
    return float(np.sum(parts))


def decay_time(series, thr=1.0):
    idx = np.where(np.nan_to_num(series, nan=0.0) > thr)[0]
    return int(idx.max() + 1) if len(idx) else 0


# ---------------------------------------------------------------- main
all_res = {}
for s in SEEDS:
    print(f'\n================ SEED {s} ================', flush=True)
    exo = make_inputs(s)
    cids_all = precompute_cids(exo, s)
    hy = run_hybrid(exo, cids_all, s)

    # load dualk5 + k3 from stage2
    st2 = np.load(os.path.join(STAGE2_DIR, f'stage2_seed{s}.npz'))
    dual = {k[len('dualk5_'):]: st2[k] for k in st2.files if k.startswith('dualk5_')}
    k3 = {k[len('k3_'):]: st2[k] for k in st2.files if k.startswith('k3_')}

    ok_pair = bool((hy['cids_hash'] == dual['cids_hash']).all())
    print(f'[cids hash seed={s}] hybrid vs dualk5 paired: {ok_pair}', flush=True)

    all_res[s] = dict(hybrid=hy, dualk5=dual, k3=k3, paired=ok_pair)
    np.savez_compressed(os.path.join(OUT, f'task3_seed{s}.npz'),
                        **{f'hybrid_{k}': hy[k] for k in hy if k != 'variant'},
                        **{f'dualk5_{k}': dual[k] for k in dual},
                        **{f'k3_{k}': k3[k] for k in k3},
                        paired=ok_pair)

# --------------------------------------------------------------- KPI table
rows = []
for s in SEEDS:
    for v in ('k3', 'dualk5', 'hybrid'):
        r = all_res[s][v]
        row = dict(seed=s, variant=v)
        row['wmos_full'] = wm(r)
        row['wmos_120'] = wm(r, (0, 120))
        row['web_mos_full'] = float(np.nanmean(r['mos_web']))
        row['web_mos_120'] = float(np.nanmean(r['mos_web'][:120]))
        row['qlen_act_mean'] = float(np.nanmean(r['qlen_act']))
        row['web_q_mean'] = float(np.nanmean(r['web_qlen_all']))
        row['web_q_peak'] = float(np.nanmax(r['web_qlen_max']))
        row['web_q_decay'] = float(decay_time(r['web_qlen_all']))
        row['sum_D'] = float(np.mean(r['sum_D']))
        row['bh_cap'] = float(np.mean(r['sum_Rbh']))
        row['acc_cap'] = float(np.mean(r['sum_Racc']))
        row['bh_util'] = float(np.mean(r['bh_util']))
        if v == 'hybrid':
            row['exec_dev_max'] = float(r['exec_dev'].max())
            row['vxy_max'] = float(r['vxy'].max())
            row['vz_max'] = float(r['vz'].max())
            row['pw_max'] = float(r['pw_hist'].max())
            row['pw_mean'] = float(r['pw_hist'].mean())
            row['bh_excess_max'] = float(r['bh_excess'].max())
            row['n_train'] = int(len(r['train_losses']))
            row['wall_s'] = float(r['wall'])
        else:
            row['exec_dev_max'] = float(r['exec_dev'].max()) if 'exec_dev' in r else 0.0
            row['vxy_max'] = float(r['vxy'].max()) if 'vxy' in r else 0.0
            row['vz_max'] = float(r['vz'].max()) if 'vz' in r else 0.0
            row['pw_max'] = np.nan
            row['pw_mean'] = np.nan
            row['bh_excess_max'] = 0.0
            row['n_train'] = 0
            row['wall_s'] = float(r['wall']) if 'wall' in r else np.nan
        rows.append(row)
kpi = pd.DataFrame(rows)
kpi.to_csv(os.path.join(OUT, 'task3_kpi.csv'), index=False)

print('\n==================== Task 3: Hybrid vs dualk5 vs k3 (501 slots) ====================')
print(f'{"metric":18s} {"seed":>4s} {"k3":>12s} {"dualk5":>12s} {"hybrid":>12s} {"hy-dual":>10s}')
for m in ('wmos_full', 'wmos_120', 'web_mos_full', 'web_mos_120',
          'web_q_mean', 'web_q_peak', 'web_q_decay', 'sum_D', 'bh_cap', 'bh_util'):
    for s in SEEDS:
        sub = kpi[kpi.seed == s].set_index('variant')
        d = sub.loc['hybrid', m] - sub.loc['dualk5', m]
        print(f'{m:18s} {s:4d} {sub.loc["k3", m]:12.4f} {sub.loc["dualk5", m]:12.4f} '
              f'{sub.loc["hybrid", m]:12.4f} {d:+10.4f}')

print('\n--- Hybrid feasibility / training ---')
for s in SEEDS:
    sub = kpi[(kpi.seed == s) & (kpi.variant == 'hybrid')].iloc[0]
    print(f'  seed={s}: exec_dev_max={sub.exec_dev_max:.3e}  vxy_max={sub.vxy_max:.2f} (lim {uav_speedxy_max})  '
          f'vz_max={sub.vz_max:.2f} (lim {uav_speedz_max})  '
          f'pw_max={sub.pw_max:.1f} (PW_MAX={PW_MAX})  pw_mean={sub.pw_mean:.1f} (PW_THRES={PW_THRES})  '
          f'bh_excess_max={sub.bh_excess_max:.4f} (<=0 ok)  n_train={int(sub.n_train)}  wall={sub.wall_s:.0f}s')

print('\n--- paired check ---')
for s in SEEDS:
    print(f'  seed={s}: hybrid vs dualk5 cids paired = {all_res[s]["paired"]}')

# ------------------------------------------------------------------ figures
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
ax = axes[0, 0]
for v, c, ls in (('k3', 'gray', '-'), ('dualk5', 'tab:orange', '-'), ('hybrid', 'tab:green', '-')):
    ax.plot(all_res[SEEDS[0]][v]['wmos'], color=c, lw=1.3, label=f'{v} s0', ls=ls)
    ax.plot(all_res[SEEDS[1]][v]['wmos'], color=c, lw=1.0, label=f'{v} s2', ls='--')
ax.set_title('weighted MOS'); ax.legend(fontsize=8); ax.set_xlabel('slot')
ax = axes[0, 1]
for v, c in (('k3', 'gray'), ('dualk5', 'tab:orange'), ('hybrid', 'tab:green')):
    ax.plot(all_res[SEEDS[0]][v]['web_qlen_all'], color=c, lw=1.3, label=f'{v} s0')
    ax.plot(all_res[SEEDS[1]][v]['web_qlen_all'], color=c, lw=1.0, label=f'{v} s2', ls='--')
ax.set_title('Web queue all (Mb)'); ax.legend(fontsize=8); ax.set_xlabel('slot')
ax = axes[1, 0]
for s, ls in ((SEEDS[0], '-'), (SEEDS[1], '--')):
    ax.plot(all_res[s]['hybrid']['pw_hist'].mean(axis=1), color='tab:green', lw=1.2,
            label=f'hybrid mean P s{s}', ls=ls)
    ax.axhline(PW_THRES, color='k', lw=0.7, ls=':', label='PW_THRES' if s == SEEDS[0] else None)
ax.set_title('Hybrid mean propulsion power (W)'); ax.legend(fontsize=8); ax.set_xlabel('slot')
ax = axes[1, 1]
xs = np.arange(len(SEEDS))
w = 0.25
for i, (v, c) in enumerate((('k3', 'gray'), ('dualk5', 'tab:orange'), ('hybrid', 'tab:green'))):
    vals = [kpi[(kpi.seed == s) & (kpi.variant == v)]['wmos_full'].iloc[0] for s in SEEDS]
    ax.bar(xs + (i - 1) * w, vals, width=w, color=c, label=v)
ax.set_xticks(xs); ax.set_xticklabels(SEEDS); ax.set_xlabel('seed')
ax.set_title('wMOS full per seed'); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'task3_hybrid_vs_dual.png'), dpi=130)
print(f'\nwrote {OUT}\\task3_kpi.csv, task3_seed*.npz, task3_hybrid_vs_dual.png')

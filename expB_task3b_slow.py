# -*- coding: utf-8
'''Experiment B Task 3b: Hybrid-slow-consistent vs Hybrid-fast vs dualk5.

Audit step 2: if the working-tree fix used option B (fast exec), add
Hybrid-slow-consistent:
  - KEEP original low-speed mapping Δp = c*Δt;
  - FIX evaluation to match execution (unified motion projection);
  - candidate generation / 30-60-10 bw pools / _solve_cluster / J /
    online REINFORCE training UNCHANGED;
  - MUST retrain online (scoring positions changed).

Seeds 0 and 2 only, 501 slots. Dual K5 and Hybrid-fast results reused
from stage2 / task3 npz (no re-simulation).

Outputs -> expB_results\\task3b_slow
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
OUT = r'D:\Learn\work\drl-opencode\expB_results\task3b_slow'
os.makedirs(OUT, exist_ok=True)
STAGE2_DIR = r'D:\Learn\work\drl-opencode\expB_results\stage2_multiseed'
TASK3_DIR = r'D:\Learn\work\drl-opencode\expB_results\task3_hybrid'

from params_sys import *
from utils_sys import (kmeans_clustering, find_active_users, update_queue_Mb,
                       cal_channel_fading_gain, cal_uav_propulsion_energy,
                       mos_func, gen_heatmap, update_vqlen_prop)
import traffic_model as TM
from mobility_model import (adjust_location, adjust_altitude,
                            generate_mobility_SFM)
from relay_hybrid_slow_controller import HybridSlowConsistentController

W_MOS = np.asarray(mos_type_weights, dtype=float)
SEEDS = [0, 2]
S = int(n_slots)
HY_N_CAND, HY_SIGMA = 20, 1.1
HYB_SEED = 0


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
    np.random.seed(seed)
    xaxis_all, yaxis_all = exo['xaxis'], exo['yaxis']
    cids_all = np.zeros((n_slots, n_users), dtype=int)
    for t in range(n_slots):
        loc_users = np.hstack((xaxis_all[:, [t]], yaxis_all[:, [t]],
                               np.zeros((n_users, 1))))
        _, cids, _ = kmeans_clustering(loc_users)
        cids_all[t] = cids
    return cids_all


def run_hybrid_slow(exo, cids_all, seed):
    print(f'\n===== hybrid-slow-consistent (seed={seed}, K={HY_N_CAND}) =====', flush=True)
    np.random.seed(1000 + seed)
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

        bw_alloc_Hz_all = np.zeros(n_users)
        for k_type in range(len(bw_type_pool_Hz)):
            ids_k = np.where(types_t == k_type)[0]
            active_k = ids_k[isActive[ids_k]]
            if len(active_k) > 0:
                bw_alloc_Hz_all[active_k] = bw_type_pool_Hz[k_type] / len(active_k)
        ld_Mbps = np.array([0 if (types_t[i] == -1) else ON_data_arrival_mean_Mb[types_t[i]]
                            for i in range(n_users)])

        heatmap_t = gen_heatmap(xaxis_all[:, t], yaxis_all[:, t],
                                Q_prev + A_t, queue_norm_Mb)
        ctrl.begin_slot(heatmap_t, uav_locs, airship_pos)

        new_locs = np.zeros_like(uav_locs)
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
            vqlen_prop[k] = update_vqlen_prop(vqlen_prop[k], pw_hist[t, k]) if t > 0 else 0.0
            if k == 0:
                J_best[t] = best_J

        if (t >= t_training_start) and (t % training_interval == 0):
            loss = ctrl.train_step()
            if loss is not None:
                train_losses.append(loss)

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

        mos_post = np.full(n_users, np.nan)
        for uid in range(n_users):
            qlen_prev_Mb = Q_prev[uid]
            Q[uid] = update_queue_Mb(qlen_prev_Mb, A_t[uid], D_opt[uid])
            ema[uid] = 0.999 * Q[uid] + 0.001 * ema[uid]
            ttype_t = int(types_t[uid])
            if ttype_t < 0:
                continue
            if tr_state[uid, t]:
                mos_post[uid] = mos_func(qlen=max(qlen_prev_Mb + A_t[uid] - D_opt[uid], 0.0),
                                         lambda_=ON_data_arrival_mean_Mb[ttype_t],
                                         ss_min_arr=ss_min_types[ttype_t],
                                         ss_max_arr=ss_max_types[ttype_t])

        reward_t = ctrl.compute_reward(Q=Q_prev, A=A_t, D=D_opt, S_u=S_u,
                                       R_bh=R_bh, mos_mean=np.nanmean(mos_post))
        ctrl.commit_slot(reward_t)
        ctrl.update_virtual_queues(S_u=S_u, R_bh=R_bh)
        uav_locs = new_locs

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

        if (t + 1) % 100 == 0:
            print(f'  t={t + 1}/{n_t}  {time.time() - t_wall:.0f}s  '
                  f'wmos={np.nanmean(wmos):.3f}  Qweb={np.nanmean(web_qlen):.2f}  '
                  f'loss={train_losses[-1] if train_losses else float("nan"):.4f}', flush=True)

    wall = time.time() - t_wall
    res = dict(wall=wall, mos_web=mos_web, mos_vid=mos_vid, mos_msg=mos_msg,
               wmos=wmos, qlen_act=qlen_act, web_qlen=web_qlen,
               web_qlen_all=web_qlen_all, web_qlen_max=web_qlen_max,
               sum_D=sum_D, sum_Rbh=sum_Rbh, sum_Racc=sum_Racc, bh_util=bh_util,
               J_best=J_best, pw_hist=pw_hist, vxy=vxy_max_hist, vz=vz_max_hist,
               exec_dev=exec_dev, bh_excess=bh_excess, cids_hash=cids_hash,
               train_losses=np.array(train_losses) if train_losses else np.zeros(0))
    print(f'  done in {wall:.0f}s  max|v_xy|={vxy_max_hist.max():.3f}  '
          f'max|vz|={vz_max_hist.max():.3f}  '
          f'max_pw={pw_hist.max():.1f}  mean_pw={pw_hist.mean():.1f}  '
          f'max_bh_excess={bh_excess.max():.4f}  exec_dev={exec_dev.max():.3e}', flush=True)
    return res


def wm(res, slice_=None):
    sl = slice(None) if slice_ is None else slice(*slice_)
    parts = [W_MOS[0] * np.nanmean(res['mos_web'][sl]),
             W_MOS[1] * np.nanmean(res['mos_vid'][sl]),
             W_MOS[2] * np.nanmean(res['mos_msg'][sl])]
    return float(np.sum(parts))


# ---------------------------------------------------------------- main
rows = []
for s in SEEDS:
    print(f'\n================ SEED {s} ================', flush=True)
    exo = make_inputs(s)
    cids_all = precompute_cids(exo, s)
    slow = run_hybrid_slow(exo, cids_all, s)

    st2 = np.load(os.path.join(STAGE2_DIR, f'stage2_seed{s}.npz'))
    dual = {k[len('dualk5_'):]: st2[k] for k in st2.files if k.startswith('dualk5_')}
    t3 = np.load(os.path.join(TASK3_DIR, f'task3_seed{s}.npz'))
    fast = {k[len('hybrid_'):]: t3[k] for k in t3.files if k.startswith('hybrid_')}

    ok_pair = bool((slow['cids_hash'] == dual['cids_hash']).all())
    print(f'[cids hash seed={s}] slow vs dualk5 paired: {ok_pair}', flush=True)

    np.savez_compressed(os.path.join(OUT, f'task3b_seed{s}.npz'),
                        **{f'slow_{k}': slow[k] for k in slow},
                        **{f'fast_{k}': fast[k] for k in fast},
                        **{f'dualk5_{k}': dual[k] for k in dual},
                        paired=ok_pair)

    for name, r, pw_key in (('hybrid_fast', fast, 'pw_hist'),
                            ('hybrid_slow', slow, 'pw_hist'),
                            ('dualk5', dual, 'P_hist')):
        pw = r[pw_key]
        row = dict(seed=s, variant=name,
                   wmos_full=wm(r), wmos_120=wm(r, (0, 120)),
                   web_mos_full=float(np.nanmean(r['mos_web'])),
                   web_mos_120=float(np.nanmean(r['mos_web'][:120])),
                   web_q_mean=float(np.nanmean(r['web_qlen_all'])),
                   web_q_peak=float(np.nanmax(r['web_qlen_max'])),
                   sum_D=float(np.mean(r['sum_D'])),
                   bh_util=float(np.mean(r['bh_util'])),
                   pw_mean=float(pw.mean()),
                   pw_peak=float(pw.max()),
                   pw_uav_mean=pw.mean(axis=0).round(1).tolist(),
                   pw_uav_peak=pw.max(axis=0).round(1).tolist(),
                   slots_over_100=int((pw > 100).sum()),
                   slots_over_683=int((pw > 683).sum()),
                   vxy_mean=float(np.mean(r['vxy'])),
                   vxy_max=float(np.max(r['vxy'])),
                   vz_max=float(np.max(r['vz'])),
                   bh_excess_max=float(np.max(r['bh_excess'])) if 'bh_excess' in r else 0.0,
                   exec_dev_max=float(np.max(r['exec_dev'])) if 'exec_dev' in r else 0.0,
                   wall_s=float(r['wall']) if 'wall' in r else np.nan,
                   n_train=int(len(r['train_losses'])) if 'train_losses' in r else 0)
        rows.append(row)

kpi = pd.DataFrame(rows)
kpi.to_csv(os.path.join(OUT, 'task3b_kpi.csv'), index=False)

print('\n==================== Task 3b: three-way comparison (501 slots) ====================')
cols = ['wmos_full', 'wmos_120', 'web_mos_full', 'web_q_mean', 'web_q_peak',
        'pw_mean', 'pw_peak', 'vxy_mean', 'vxy_max', 'bh_util', 'slots_over_683']
for s in SEEDS:
    sub = kpi[kpi.seed == s].set_index('variant')
    print(f'\n--- seed {s} ---')
    print(f'{"metric":16s} {"fast":>12s} {"slow":>12s} {"dual":>12s}')
    for m in cols:
        print(f'{m:16s} {sub.loc["hybrid_fast", m]:12.4f} '
              f'{sub.loc["hybrid_slow", m]:12.4f} {sub.loc["dualk5", m]:12.4f}')

print('\n--- power detail (per-UAV mean / peak W) ---')
for s in SEEDS:
    for v in ('hybrid_fast', 'hybrid_slow', 'dualk5'):
        row = kpi[(kpi.seed == s) & (kpi.variant == v)].iloc[0]
        print(f'  seed={s} {v:14s} mean={row.pw_mean:6.1f} peak={row.pw_peak:6.1f} '
              f'per-UAV mean={row.pw_uav_mean} peak={row.pw_uav_peak} '
              f'>100W slots={row.slots_over_100} >683W slots={row.slots_over_683}')

print('\n--- energy feasibility ---')
for s in SEEDS:
    for v in ('hybrid_fast', 'hybrid_slow', 'dualk5'):
        row = kpi[(kpi.seed == s) & (kpi.variant == v)].iloc[0]
        ok_peak = row.slots_over_683 == 0
        ok_avg = row.pw_mean <= 1.5 * PW_THRES   # loose: long-term E queue is the real budget
        print(f'  seed={s} {v:14s} peak_ok={ok_peak}  mean_pw={row.pw_mean:.1f} '
              f'(PW_THRES={PW_THRES})  verdict={"FEASIBLE" if ok_peak and row.pw_mean < 200 else "INFEASIBLE (energy)"}')

# ------------------------------------------------------------------ figures
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
ax = axes[0, 0]
for v, c in (('dualk5', 'tab:orange'), ('hybrid_fast', 'tab:red'), ('hybrid_slow', 'tab:green')):
    for s, ls in ((0, '-'), (2, '--')):
        src = {'dualk5': dual, 'hybrid_fast': fast, 'hybrid_slow': slow}
        # reload from saved npz for both seeds
        pass
for s in SEEDS:
    d = np.load(os.path.join(OUT, f'task3b_seed{s}.npz'))
    ls = '-' if s == 0 else '--'
    ax.plot(d['dualk5_wmos'], color='tab:orange', lw=1.2, ls=ls, label=f'dual s{s}')
    ax.plot(d['fast_wmos'], color='tab:red', lw=1.2, ls=ls, label=f'fast s{s}')
    ax.plot(d['slow_wmos'], color='tab:green', lw=1.2, ls=ls, label=f'slow s{s}')
ax.set_title('weighted MOS'); ax.legend(fontsize=7); ax.set_xlabel('slot')
ax = axes[0, 1]
for s in SEEDS:
    d = np.load(os.path.join(OUT, f'task3b_seed{s}.npz'))
    ls = '-' if s == 0 else '--'
    ax.plot(d['dualk5_web_qlen_all'], color='tab:orange', lw=1.2, ls=ls, label=f'dual s{s}')
    ax.plot(d['fast_web_qlen_all'], color='tab:red', lw=1.2, ls=ls, label=f'fast s{s}')
    ax.plot(d['slow_web_qlen_all'], color='tab:green', lw=1.2, ls=ls, label=f'slow s{s}')
ax.set_title('Web queue all (Mb)'); ax.legend(fontsize=7); ax.set_xlabel('slot')
ax = axes[1, 0]
for s in SEEDS:
    d = np.load(os.path.join(OUT, f'task3b_seed{s}.npz'))
    ls = '-' if s == 0 else '--'
    ax.plot(d['fast_pw_hist'].mean(axis=1), color='tab:red', lw=1.2, ls=ls, label=f'fast s{s}')
    ax.plot(d['slow_pw_hist'].mean(axis=1), color='tab:green', lw=1.2, ls=ls, label=f'slow s{s}')
    ax.plot(d['dualk5_P_hist'].mean(axis=1), color='tab:orange', lw=1.2, ls=ls, label=f'dual s{s}')
ax.axhline(PW_THRES, color='k', lw=0.7, ls=':', label='PW_THRES')
ax.axhline(PW_MAX, color='k', lw=0.7, ls='-.', label='PW_MAX')
ax.set_title('mean propulsion power per UAV (W)'); ax.legend(fontsize=7); ax.set_xlabel('slot')
ax = axes[1, 1]
xs = np.arange(len(SEEDS))
w = 0.25
for i, (v, c) in enumerate((('dualk5', 'tab:orange'), ('hybrid_fast', 'tab:red'), ('hybrid_slow', 'tab:green'))):
    vals = [kpi[(kpi.seed == s) & (kpi.variant == v)]['wmos_full'].iloc[0] for s in SEEDS]
    ax.bar(xs + (i - 1) * w, vals, width=w, color=c, label=v)
ax.set_xticks(xs); ax.set_xticklabels(SEEDS); ax.set_xlabel('seed')
ax.set_title('wMOS full per seed'); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'task3b_three_way.png'), dpi=130)
print(f'\nwrote {OUT}\\task3b_kpi.csv, task3b_seed*.npz, task3b_three_way.png')

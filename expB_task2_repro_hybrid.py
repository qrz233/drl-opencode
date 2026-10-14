# -*- coding: utf-8 -*-
"""Task 2: exact reproduction of the old Hybrid run that produced MOS≈5.000.

Matches run_blocka_comparison.py + HEAD main.ipynb cell 26 (pre motion-fix):
  - HybridCandidateController n_candidates=20, eval_sigma=1.1, seed=0
  - Eval motion: uav_loc + c * speed_xy * tau   (fast, inside decide_and_score)
  - Exec motion: uav.x[t-1] + c * slot_len      (slow, raw c as velocity)
  - Candidate scoring: type-pool bandwidth proxy (cell 2 block) at eval locs
  - Delivery: BLOCKA variable-bandwidth at *executed* positions (NEW_P26)
  - Inputs: frozen SFM xlsx (seed 11 already baked), Weibull arrival seed 0,
            traffic state/type xlsx, kmeans on loc_users each slot
  - np.random.seed(0) via controller init; kmeans + candidate noise share stream

Adds logging (not present in the original notebook KPI xlsx):
  trajectory, per-UAV propulsion power, velocity, backhaul feasibility, delivery.
"""
import os, sys, time, zlib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
OUT = r'D:\Learn\work\drl-opencode\expB_results\task2_repro'
os.makedirs(OUT, exist_ok=True)

from params_sys import *
from utils_sys import (kmeans_clustering, find_active_users, update_queue_Mb,
                       cal_channel_fading_gain, cal_uav_propulsion_energy,
                       mos_func, gen_heatmap)
import blocka_optimizer as B
from mobility_model import adjust_location, adjust_altitude

# HEAD hybrid controller (K=20, sigma=1.1, eval-consistent motion)
# Parent class from PROJ is fine (solve_delivery / queues unchanged).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'relay_hybrid_controller_HEAD',
    r'D:\Learn\work\drl-opencode\relay_hybrid_controller_HEAD.py')
_rhc = importlib.util.module_from_spec(_spec)
# ensure parent import resolves to PROJ relay_lyapunov_controller
sys.modules['relay_hybrid_controller_HEAD'] = _rhc
_spec.loader.exec_module(_rhc)
HybridCandidateController = _rhc.HybridCandidateController

S = int(n_slots)
W_MOS = np.asarray(mos_type_weights, dtype=float)
SEED = 0

# ------------------------------- frozen inputs (identical to blocka run)
def from_exel(fname):
    return np.transpose(pd.read_excel(fname, index_col=0).to_numpy())

sim_in = os.path.join(PROJ, 'simulation', 'sim-input')
n_users_, n_uavs_ = int(n_users), int(n_uavs)
tag = (f"nusers={n_users}, nslots={n_slots}, ntype=3, "
       f"lambda_Mb={lambd_Mb:.1f}, uON={ON_duration_mean_tslot:.0f}, "
       f"uOFF={OFF_duration_mean_tslot:.0f}")
xaxis = from_exel(os.path.join(sim_in, f"xlocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx"))
yaxis = from_exel(os.path.join(sim_in, f"ylocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx"))
arrival = from_exel(os.path.join(sim_in, f"arrival traffic, dist=weibull, {tag}.xlsx"))
tr_type = from_exel(os.path.join(sim_in, f"traffic type, {tag}.xlsx")).astype(int)
tr_state = from_exel(os.path.join(sim_in, f"traffic state, {tag}.xlsx")).astype(bool)
print('inputs', xaxis.shape, arrival.shape, 'arr_mean', arrival.mean(), flush=True)

airship_pos0 = np.array([airship_x, airship_y, airship_z], float)
init_uav = np.column_stack([init_locations_uav[:, 0], init_locations_uav[:, 1],
                            np.full(n_uavs_, float(z_init_proposed))])

ctrl = HybridCandidateController(
    n_users=n_users_, n_uavs=n_uavs_, boundary=boundary, grid_size=grid_size,
    queue_norm=queue_norm_Mb, V_lyap=lyapunov_param, tau=slot_len,
    ss_min_types=ss_min_types, ss_max_types=ss_max_types,
    mos_weights=mos_type_weights, type_rates=ON_data_arrival_mean_Mb, d0=d0,
    sigma_decay_slots=n_slots, seed=SEED,
    n_candidates=20, eval_sigma=1.1
)
ctrl.bind_models(cal_channel_fading_gain, cal_backhaul_capacity_Mb,
                 pTx_downlink, noise_pw_total)

Q = np.zeros(n_users_)
ema = np.zeros(n_users_)
# Match notebook UAV class: x = zeros(n_slots); x[0] = x_init.
# At t=0, decision uses x[0]=init but EXEC uses x[t-1]=x[-1]=0 (original quirk).
uav_x = np.zeros((n_uavs_, S)); uav_y = np.zeros((n_uavs_, S)); uav_z = np.zeros((n_uavs_, S))
uav_x[:, 0] = init_uav[:, 0]; uav_y[:, 0] = init_uav[:, 1]; uav_z[:, 0] = init_uav[:, 2]
airship_pos = airship_pos0.copy()

# logs
mos_web = np.full(S, np.nan); mos_vid = np.full(S, np.nan); mos_msg = np.full(S, np.nan)
wmos = np.full(S, np.nan); qlen_act = np.full(S, np.nan)
web_qlen = np.full(S, np.nan); web_qlen_all = np.full(S, np.nan); web_qlen_max = np.full(S, np.nan)
sum_D = np.zeros(S); sum_Rbh = np.zeros(S); sum_Racc = np.zeros(S)
chcapa = np.full(S, np.nan); bh_util = np.zeros(S)
bh_excess = np.zeros(S)
P_hist = np.zeros((S, n_uavs_)); E_hist = np.zeros((S, n_uavs_))
vxy_max = np.zeros(S); vz_max = np.zeros(S)
traj = np.zeros((S, n_uavs_, 3))
bestJ = np.full(S, np.nan)
cids_hash = np.zeros(S, dtype=np.int64)
Z_hist = np.zeros((S, n_uavs_))

# energy virtual queue per UAV (notebook uav.vqlen_prop)
vqlen = np.zeros(n_uavs_)
E = np.zeros(n_uavs_)

t_wall = time.time()
for t in range(S):
    Q_prev = Q.copy()
    A_t = arrival[:, t]
    types_t = tr_type[:, t]
    isActive = find_active_users(queue_len_Mb=Q_prev, incoming_traffic_Mb=A_t)

    # (1) clustering
    loc_users = np.hstack((xaxis[:, [t]], yaxis[:, [t]], np.zeros((n_users_, 1))))
    _, cids, _ = kmeans_clustering(loc_users)
    cids_hash[t] = np.int64(zlib.crc32(np.ascontiguousarray(cids).tobytes()))

    # (2) type-pool bandwidth (scoring proxy only; delivery uses BLOCKA)
    bw_alloc_Hz_all = np.zeros(n_users_)
    for k in range(len(bw_type_pool_Hz)):
        ids_k = np.where(types_t == k)[0]
        active_k = ids_k[isActive[ids_k]]
        if len(active_k) > 0:
            bw_alloc_Hz_all[active_k] = bw_type_pool_Hz[k] / len(active_k)

    # (3) movement: evaluate at fast positions, execute slow (raw c as velocity)
    # Notebook semantics: decision uses x[0] at t=0 else x[t-1]; exec ALWAYS uses x[t-1]
    # (so t=0 exec base is x[-1]=0, the original quirk).
    if t > 0:
        uav_locs_dec = np.column_stack([uav_x[:, t-1], uav_y[:, t-1], uav_z[:, t-1]])
    else:
        uav_locs_dec = np.column_stack([uav_x[:, 0], uav_y[:, 0], uav_z[:, 0]])
    exec_base = np.column_stack([uav_x[:, t-1], uav_y[:, t-1], uav_z[:, t-1]])
    heatmap_t = gen_heatmap(xaxis[:, t], yaxis[:, t], Q_prev + A_t, queue_norm_Mb)
    ld_Mbps = np.array([0 if types_t[i] == -1 else ON_data_arrival_mean_Mb[types_t[i]]
                        for i in range(n_users_)])
    ctrl.begin_slot(heatmap_t, uav_locs_dec, airship_pos)
    v_list = np.zeros((n_uavs_, 3))
    for k in range(n_uavs_):
        best_a, bJ = ctrl.decide_and_score(
            k=k, uav_loc=uav_locs_dec[k], airship_loc=airship_pos,
            heatmap=heatmap_t, user_x=xaxis[:, t], user_y=yaxis[:, t],
            bw_alloc_Hz=bw_alloc_Hz_all, Q=Q_prev, A=A_t, ld=ld_Mbps,
            types=types_t, cids=cids,
            speed_xy=uav_speedxy_max, speed_z=uav_speedz_max, zmin=z_min, zmax=z_max,
            boundary=boundary)
        bestJ[t] = bJ
        # SLOW exec: raw c as velocity (m/s), base = uav.x[t-1]
        vx, vy, vz = best_a
        xt = exec_base[k, 0] + vx * slot_len
        yt = exec_base[k, 1] + vy * slot_len
        zt = exec_base[k, 2] + vz * slot_len
        uav_x[k, t] = adjust_location(xt, bound=boundary)
        uav_y[k, t] = adjust_location(yt, bound=boundary)
        uav_z[k, t] = adjust_altitude(zt, zmin=z_min, zmax=z_max)
        v_list[k] = [vx, vy, vz]
    uav_locs = np.column_stack([uav_x[:, t], uav_y[:, t], uav_z[:, t]])
    traj[t] = uav_locs

    vxy_max[t] = float(np.max(np.linalg.norm(v_list[:, :2], axis=1)))
    vz_max[t] = float(np.max(np.abs(v_list[:, 2])))
    P = np.array([float(cal_uav_propulsion_energy(V=float(np.linalg.norm(v_list[u, :2])),
                                                  Vz=float(v_list[u, 2]))[0])
                  for u in range(n_uavs_)])
    P_hist[t] = P
    E = np.maximum(E + P - PW_THRES, 0.0)
    E_hist[t] = E
    vqlen = vqlen + P  # notebook vqlen_prop accumulates power (not capped here)

    if (t >= t_training_start) and (t % training_interval == 0):
        ctrl.train_step()

    # (4-5) BLOCKA at executed positions
    res = B.blocka_allocate(cids=cids, user_x=xaxis[:, t], user_y=yaxis[:, t],
                            uav_locs=uav_locs, airship_loc=airship_pos,
                            queue_Mb=Q_prev, traffic_Mb=A_t, types=types_t)
    D_opt = res['D_Mb']
    R_access = res['R_access_Mb']
    S_u = res['S_u']
    R_bh = res['R_bh_Mb']
    sum_Rbh[t] = float(np.sum(R_bh))
    sum_Racc[t] = float(np.sum(R_access))
    bh_util[t] = float(np.sum(S_u) / max(np.sum(R_bh), 1e-9))
    bh_excess[t] = float(np.sum(S_u) - np.sum(R_bh))
    chcapa[t] = float(np.mean(R_access))
    sum_D[t] = float(np.sum(D_opt))

    # (6) queues + MOS
    mos_post = np.full(n_users_, np.nan)
    for i in range(n_users_):
        ttype_t = int(types_t[i])
        ld_type = ON_data_arrival_mean_Mb[ttype_t] if ttype_t >= 0 else lambd_Mb
        qlen_prev = Q_prev[i]
        Q[i] = update_queue_Mb(qlen_prev, A_t[i], D_opt[i])
        ema[i] = alpha_ema * Q[i] + (1 - alpha_ema) * ema[i]
        if tr_state[i, t]:
            mos_post[i] = mos_func(qlen=max(qlen_prev + A_t[i] - D_opt[i], 0),
                                   lambda_=ld_type,
                                   ss_min_arr=ss_min_types[ttype_t],
                                   ss_max_arr=ss_max_types[ttype_t])
    for k_type, name in ((0, 'web'), (1, 'vid'), (2, 'msg')):
        m = (types_t == k_type) & tr_state[:, t]
        if np.any(m):
            mos = mos_func(qlen=ema[m], lambda_=ON_data_arrival_mean_Mb[k_type],
                           ss_min_arr=ss_min_types[k_type],
                           ss_max_arr=ss_max_types[k_type])
            val = float(np.mean(mos))
            if name == 'web': mos_web[t] = val
            elif name == 'vid': mos_vid[t] = val
            else: mos_msg[t] = val
    parts = [W_MOS[0]*mos_web[t], W_MOS[1]*mos_vid[t], W_MOS[2]*mos_msg[t]]
    wmos[t] = float(np.nansum(parts)) if np.any(np.isfinite(parts)) else np.nan
    n_act = int(isActive.sum())
    qlen_act[t] = float(np.sum(Q[isActive]) / n_act) if n_act else np.nan
    mw = isActive & (types_t == 0)
    web_qlen[t] = float(np.mean(Q[mw])) if np.any(mw) else np.nan
    allw = types_t == 0
    web_qlen_all[t] = float(np.mean(Q[allw])) if np.any(allw) else np.nan
    web_qlen_max[t] = float(np.max(Q[allw])) if np.any(allw) else np.nan

    # (7) reward / virtual queues / commit
    reward = ctrl.compute_reward(Q=Q_prev, A=A_t, D=D_opt, S_u=S_u, R_bh=R_bh,
                                 mos_mean=float(np.nanmean(mos_post)))
    ctrl.commit_slot(reward)
    ctrl.update_virtual_queues(S_u=S_u, R_bh=R_bh)
    Z_hist[t] = ctrl.Z

    if (t + 1) % 100 == 0:
        print(f'  t={t+1}/{S}  {time.time()-t_wall:.0f}s  '
              f'wmos={np.nanmean(wmos[:t+1]):.3f}  Qweb={np.nanmean(web_qlen[:t+1]):.2f}', flush=True)

wall = time.time() - t_wall
print(f'done in {wall:.0f}s', flush=True)

# ------------------------------- KPI report vs target
def wm(sl=None):
    sl = slice(None) if sl is None else sl
    return float(np.nansum([W_MOS[0]*np.nanmean(mos_web[sl]),
                            W_MOS[1]*np.nanmean(mos_vid[sl]),
                            W_MOS[2]*np.nanmean(mos_msg[sl])]))

print('\n===== Task 2 Hybrid-lookahead reproduction =====')
print(f'wMOS full = {wm():.4f}   (target 4.9967)')
print(f'web_mos={np.nanmean(mos_web):.3f}  vid={np.nanmean(mos_vid):.3f}  msg={np.nanmean(mos_msg):.3f}')
print(f'qlen_act mean={np.nanmean(qlen_act):.3f}  web_qlen mean={np.nanmean(web_qlen):.3f}  (target ~0.117 active)')
print(f'web_qlen_all mean={np.nanmean(web_qlen_all):.3f}  peak={np.nanmax(web_qlen_max):.3f}')
print(f'sum_D mean={np.mean(sum_D):.2f}  sum_Rbh mean={np.mean(sum_Rbh):.2f}  bh_util={np.mean(bh_util):.3f}')
print(f'bh_excess max={np.max(bh_excess):.4f} (<=0 feasible)')
print(f'per-UAV P mean W = {P_hist.mean(axis=0).round(2).tolist()}  max={P_hist.max():.1f}')
print(f'per-UAV E final = {E_hist[-1].round(1).tolist()}')
print(f'vxy_max={vxy_max.max():.2f}  vz_max={vz_max.max():.2f}')
print(f'ctrl.Z final={np.round(ctrl.Z,2).tolist()}  Z_max={np.round(ctrl.Z_max,2).tolist()}')
print(f'bestJ mean={np.nanmean(bestJ):.3f}')

# paired config check vs Dual K5 stage2
st2 = np.load(r'D:\Learn\work\drl-opencode\expB_results\stage2_multiseed\stage2_seed0.npz')
print('\n--- vs Dual K5 stage2 seed0 (same frozen inputs, same kmeans stream expected) ---')
print(f'Dual wMOS={np.nansum([W_MOS[0]*np.nanmean(st2["dualk5_mos_web"]), W_MOS[1]*np.nanmean(st2["dualk5_mos_vid"]), W_MOS[2]*np.nanmean(st2["dualk5_mos_msg"])]):.4f}')
print(f'cids_hash equal: {bool((cids_hash == st2["dualk5_cids_hash"]).all())}')
print(f'arrival mean match: {np.isclose(arrival.mean(), arrival.mean())}')
print(f'NOTE: Hybrid-lookahead uses type-pool scoring proxy + BLOCKA exec delivery + slow exec;')
print(f'      Dual K5 uses BLOCKA for both scoring and delivery + fast eval-consistent motion.')
print(f'      Not strictly paired; Hybrid-lookahead is a historical baseline, not a same-physics control.')

# save
np.savez_compressed(os.path.join(OUT, 'task2_hybrid_lookahead.npz'),
                    mos_web=mos_web, mos_vid=mos_vid, mos_msg=mos_msg, wmos=wmos,
                    qlen_act=qlen_act, web_qlen=web_qlen, web_qlen_all=web_qlen_all,
                    web_qlen_max=web_qlen_max, sum_D=sum_D, sum_Rbh=sum_Rbh,
                    sum_Racc=sum_Racc, chcapa=chcapa, bh_util=bh_util, bh_excess=bh_excess,
                    P_hist=P_hist, E_hist=E_hist, traj=traj, vxy=vxy_max, vz=vz_max,
                    bestJ=bestJ, cids_hash=cids_hash, Z_hist=Z_hist,
                    wall=np.array(wall))

df = pd.DataFrame(dict(slot=np.arange(S), wmos=wmos, mos_web=mos_web, mos_vid=mos_vid,
                       mos_msg=mos_msg, qlen_act=qlen_act, web_qlen=web_qlen,
                       web_qlen_all=web_qlen_all, sum_D=sum_D, sum_Rbh=sum_Rbh,
                       bh_util=bh_util, bh_excess=bh_excess, bestJ=bestJ))
for u in range(n_uavs_):
    df[f'P_uav{u}'] = P_hist[:, u]
    df[f'E_uav{u}'] = E_hist[:, u]
df.to_csv(os.path.join(OUT, 'task2_slots.csv'), index=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
axes[0, 0].plot(wmos); axes[0, 0].set_title('weighted MOS'); axes[0, 0].axhline(4.9967, color='r', ls='--', lw=0.8)
axes[0, 1].plot(web_qlen_all); axes[0, 1].set_title('Web queue all (Mb)')
axes[0, 2].plot(sum_D, label='sum D'); axes[0, 2].plot(sum_Rbh, label='sum R_bh', alpha=0.7); axes[0, 2].legend(); axes[0, 2].set_title('delivery vs backhaul')
axes[1, 0].plot(P_hist); axes[1, 0].axhline(PW_THRES, color='r', ls='--', lw=0.8); axes[1, 0].set_title('per-UAV power (W)')
axes[1, 1].plot(E_hist); axes[1, 1].set_title('energy virtual queue E')
axes[1, 2].plot(bh_excess); axes[1, 2].axhline(0, color='k', lw=0.8); axes[1, 2].set_title('sum(S_u)-sum(R_bh)')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'task2_repro.png'), dpi=130)
print(f'\nwrote {OUT}')

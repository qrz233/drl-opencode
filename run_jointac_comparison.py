# -*- coding: utf-8 -*-
'''Joint actor-critic + movable airship variant (actor-critic部分.md).

Patches main.ipynb:
    cell 0 : + cvxpy / blocka_optimizer import
    cell 4 : SFM mobility (reuses bound=250-sfm.xlsx)
    cell 6 : Weibull arrival
    cell 8 : load SFM locations + Weibull arrival
    cell 22/24 (Stationary/Centroid): BLOCKA allocation (same as run_blocka_comparison)
    cell 26 (Proposed): FULLY REPLACED by the joint multi-head actor-critic:
        movable airship, K joint candidates, full Block A per candidate,
        J = psi_E*sum E*P - sum beta*D critic, supervised actor training
    params_sys.py folder tag: -sfm-wb-jac  (movable) / -sfm-wb-jac-fa (airship fixed ablation)

Usage:  python run_jointac_comparison.py mov     # airship moves   -> -sfm-wb-jac
        python run_jointac_comparison.py fixed   # airship fixed   -> -sfm-wb-jac-fa
Ablations (no learning, directed beta-centroid candidates):
        python run_jointac_comparison.py mov --nd    -> -sfm-wb-jac-nd    (directed only, no actor)
        python run_jointac_comparison.py mov --ndls  -> -sfm-wb-jac-ndls  (directed + local coordinate search)
Collects whatever output folders exist and writes jointac_vs_baseline_results.xlsx.
'''
import json, os, glob, subprocess, sys
import numpy as np
import pandas as pd

PROJ = r"D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11"
NB = os.path.join(PROJ, "main.ipynb")
PARAMS = os.path.join(PROJ, "params_sys.py")
PY = sys.executable
W = {0: 0.35, 1: 0.30, 2: 0.35}          # MOS weights Web/Video/Messaging

VARIANT = (sys.argv[1] if len(sys.argv) > 1 else "mov").lower()
assert VARIANT in ("mov", "fixed"), "usage: run_jointac_comparison.py [mov|fixed] [--nd|--ndls]"
DRYRUN = "--dryrun" in sys.argv
MODE = ("directed" if "--nd" in sys.argv else
        "directed_ls" if "--ndls" in sys.argv else "full")
SUF = {"full": "", "directed": "-nd", "directed_ls": "-ndls"}[MODE]
TAG = {"mov": "-sfm-wb-jac", "fixed": "-sfm-wb-jac-fa"}[VARIANT] + SUF
MOVABLE = {"mov": "True", "fixed": "False"}[VARIANT]

nb_template = open(NB, "rb").read()
params_template = open(PARAMS, "rb").read()

# ---------------- shared cell sources (same as run_blocka_comparison) -------
NEW_CELL6 = '''fpath = os.path.join(sim_in_path, f"arrival traffic, dist=weibull, nusers={n_users}, nslots={n_slots}, ntype=3, lambda_Mb={lambd_Mb:.1f}, uON={ON_duration_mean_tslot:.0f}, uOFF={OFF_duration_mean_tslot:.0f}.xlsx")

if os.path.exists(fpath)==False:
    # Reuse the baseline ON/OFF schedule & service-type assignment (existing
    # xlsx), so only the ON-slot file-size marginal changes (Pareto -> Weibull).
    def _load(fname):
        return np.transpose(pd.read_excel(os.path.join(sim_in_path, fname), index_col=0).to_numpy())

    base_tag = f"nusers={n_users}, nslots={n_slots}, ntype=3, lambda_Mb={lambd_Mb:.1f}, uON={ON_duration_mean_tslot:.0f}, uOFF={OFF_duration_mean_tslot:.0f}"
    traffic_state = _load(f"traffic state, {base_tag}.xlsx").astype(bool)
    traffic_type = _load(f"traffic type, {base_tag}.xlsx").astype(int)

    arrival_traffic_Mb = generate_arrival_from_schedule(
        traffic_state=traffic_state, traffic_type=traffic_type,
        data_arrival_mean_Mb=ON_data_arrival_mean_Mb, dist='weibull', seed=0)

    to_excel(arrival_traffic_Mb, fpath)
    print(f'Generated Weibull arrival traffic in {sim_in_path}')
else:
    print("Input data for simulation exists!")
'''

OLD_BW_BLOCK = '''    # (2) Bandwidth Allocation (modified): the 50 MHz pool is split across the 3 service
    # types with shares 30% (Web) / 60% (Video) / 10% (Messaging); in each slot a type's
    # pool is shared equally by its active users. alpha = user bw / bw_per_user (reference),
    # so channel capacity = alpha * bw_per_user * log2(1+SNR) = user_bw * log2(1+SNR).
    bw_alloc_Hz_all = np.zeros(shape=(n_users,))
    ttype_all_t = traffic_type[:, t]
    for k in range(len(bw_type_pool_Hz)): 
        ids_k = np.where(ttype_all_t == k)[0]
        active_k = ids_k[isActive_all[ids_k]]
        if len(active_k) > 0: 
            bw_alloc_Hz_all[active_k] = bw_type_pool_Hz[k] / len(active_k)
    bw_alloc_all = bw_alloc_Hz_all / bw_per_user      # fraction of the reference per-user bw
'''
NEW_BW_BLOCK = '''    # (2) Bandwidth Allocation: decided jointly with delivery by BLOCKA (P_R)
    # after the UAV positions are updated in (3.5).
    ttype_all_t = traffic_type[:, t]
'''

OLD_BH_BLOCK = '''    # (3.5) Modified: backhaul constraint - airship -> drone k hard-caps the drone's
    # total served downlink rate (excess stays queued; bandwidth allocation untouched).
    bh_scale = np.ones(shape=(n_uavs,))
    bh_capa_k = np.zeros(shape=(n_uavs,))
    for k in range(n_uavs): 
        ids_k = np.where(cids == k)[0]
        if len(ids_k) == 0: 
            continue
        g_k = cal_channel_fading_gain(xaxis_all[ids_k, t], yaxis_all[ids_k, t], uavs[k].x[t], uavs[k].y[t], uavs[k].z[t])
        capa_k = bw_alloc_Hz_all[ids_k] * np.log2(1 + pTx_downlink * g_k / noise_pw_total) * slot_len / 1e6   # Mb/slot
        offered_k = np.sum(np.minimum(capa_k, queue_length_Mb[ids_k] + traffic_Mb[ids_k]))
        bh_capa_k[k] = cal_backhaul_capacity_Mb(uavs[k].x[t], uavs[k].y[t], uavs[k].z[t])
        if offered_k > bh_capa_k[k]: 
            bh_scale[k] = bh_capa_k[k] / offered_k
    bh_capa_log.append(np.mean(bh_capa_k))
    bh_scale_log.append(np.min(bh_scale))
'''
NEW_BH_BLOCK = '''    # (3.5) BLOCKA: joint variable-bandwidth + delivery optimisation (convex P_R)
    _res = blocka_allocate(
        cids=cids, user_x=xaxis_all[:, t], user_y=yaxis_all[:, t],
        uav_locs=np.array([[uav.x[t], uav.y[t], uav.z[t]] for uav in uavs]),
        airship_loc=np.array([airship_x, airship_y, airship_z]),
        queue_Mb=queue_length_Mb, traffic_Mb=traffic_Mb, types=ttype_all_t)
    bw_alloc_Hz_all = _res['W_ui_Hz']
    bw_alloc_all = bw_alloc_Hz_all / bw_per_user
    R_access = _res['R_access_Mb']
    D_opt = _res['D_Mb']
    bh_capa_k = _res['R_bh_Mb']
    bh_capa_log.append(np.mean(bh_capa_k))
    bh_scale_log.append(1.0)
'''

OLD_DELIV = '''        user.channel_capacity_Mb[t] = cal_channel_capacity_Mb(user.chgains_downlink[t], user.alpha[t])[0]
        
        qlen_prev_Mb = user.queue_length_Mb[t-1] if t > 0 else 0
        depart_Mb = cal_downlink_throughput_Mb(qlen_prev_Mb, user.arrival_traffic_Mb[t], user.channel_capacity_Mb[t]) * bh_scale[cid]
'''
NEW_DELIV = '''        user.channel_capacity_Mb[t] = R_access[id]
        
        qlen_prev_Mb = user.queue_length_Mb[t-1] if t > 0 else 0
        depart_Mb = D_opt[id]
'''

# --------------------------- new cell 26 (joint actor-critic) ---------------
NEW_CELL26 = '''# New (v14): Proposed method = joint centralized multi-head actor-critic with a
# movable airship + Lyapunov Analytic Critic (see joint_ac_controller.py and
# actor-critic部分.md):
#   - actor: shared CNN over per-UAV [backlog, pressure] maps -> local/global
#     features -> mu_U (shared, 3D) + mu_A (airship, 2D) heads; K=10 joint
#     candidates = [no-movement guard, actor output, 2 directed beta-centroid
#     moves (rho=0.25, 0.5), 6 Gaussian perturbations (sigma=0.5)]
#   - critic: solve the FULL Block A (convex P_R) at every candidate's positions
#       J = psi_E * sum_u E_u * P_u  -  sum_i beta_i * D_i*,
#       beta_i = psi_Q*Q + V*w_k/(lambda_k * sum_{I+} w) (Lyapunov priority);
#       argmin J executes geometry + bandwidth + delivery together
#   - energy queues E_u (E <- max(E + P_exec - PW_THRES, 0)) replace the old
#     backhaul virtual queue Z (Block A hard-caps backhaul every slot)
#   - actor training: periodic supervised MSE on the critic-selected normalized
#     joint action (replay + Adam, every training_interval slots)
from joint_ac_controller import JointACCritic, AIRSHIP_SPEED_MAX

# Initiate the UAV Swarm (relay drones with the original 3D movement) 
uav_swarm_proposed = [UAV(name= f'UAV {i} (Proposed)', x_init=init_locations_uav[i][0], y_init=init_locations_uav[i][1], z_init=z_init_proposed, 
                        xlim = xlim_uav[i], ylim = ylim_uav[i], speedxy_max=uav_speedxy_max, speedz_max=uav_speedz_max) for i in range(0,n_uavs)]

# Joint actor-critic controller (see joint_ac_controller.py)
ctrl = JointACCritic(
    n_users=n_users, n_uavs=n_uavs, n_grids=int(2 * boundary / grid_size),
    ss_min_types=ss_min_types, ss_max_types=ss_max_types,
    mos_weights=mos_type_weights, type_rates=ON_data_arrival_mean_Mb, d0=d0,
    V_lyap=lyapunov_param, queue_norm=queue_norm_Mb,
    n_candidates=10, eval_sigma=0.5,
    airship_movable=__AIRSHIP_MOVABLE__,
    mode=__MODE__,
    seed=0
)

# Initiate users of the Proposed swarm with movements and traffic 
users_proposed = [User(f'User {i}') for i in range(n_users)]
load_user_properties(
    users_proposed,
    xaxis_all, yaxis_all,                               # location
    arrival_traffic_Mb, traffic_state, traffic_type,    # traffic
    ema_traffic_Mb, lambd_Mb_alluser
)

# Alias for ease of code maintenance 
uavs = uav_swarm_proposed
users = users_proposed          
    
# Simulation starts!  
bh_capa_log, bh_util_log, reward_log = [], [], []
bestJ_log = []
airship_pos = np.array([airship_x, airship_y, airship_z], dtype=float)   # movable state
for t in range(0, n_slots):         # t = timeslot
    if t%100==0: 
        print(f"t={t}")
    # (0) Collect system statistics 
    queue_length_Mb = np.array([user.queue_length_Mb[t-1] for user in users]) if t > 0 else np.zeros(n_users)  # (n_users,)
    traffic_Mb = np.array([user.arrival_traffic_Mb[t] for user in users])       # shape=(n_users,)
    isActive_all = find_active_users(queue_len_Mb=queue_length_Mb, incoming_traffic_Mb=traffic_Mb)  # (n_users,)
    traffic_state_all = np.array([user.traffic_state[t] for user in users])     # shape=(n_users,)
    mos_all = np.array([user.mos[t-1] for user in users]) if t > 0 else 5 * np.ones(n_users)
    ld_Mbps = np.array([0 if (user.traffic_type[t] == -1) else ON_data_arrival_mean_Mb[user.traffic_type[t]] for user in users])
    
    # (1) User Clustering: based on locations of users 
    loc_users = np.hstack((
        np.expand_dims(xaxis_all[:,t], axis=1),
        np.expand_dims(yaxis_all[:,t], axis=1),
        np.zeros(shape=(n_users,1))
    ))    # shape = (n_users, 3)
    cmat, cids, c_centroids = kmeans_clustering(loc_users)        # cluster matrix, ids, and centroid
    
    # (3) Joint actor-critic decision: dual-channel maps + K joint candidates,
    #     full Block A per candidate, argmin J executed (airship may move)
    uav_locs = np.array([[uav.x[t-1], uav.y[t-1], uav.z[t-1]] for uav in uavs]) if t>0 \\
               else np.array([[uav.x[0], uav.y[0], uav.z[0]] for uav in uavs])
    ctrl.begin_slot(user_x=xaxis_all[:,t], user_y=yaxis_all[:,t],
                    Q=queue_length_Mb, A=traffic_Mb, types=traffic_type[:,t],
                    cids=cids, uav_locs=uav_locs, airship_pos=airship_pos)
    best = ctrl.decide_and_score(speed_xy=uav_speedxy_max, speed_z=uav_speedz_max)
    bestJ_log.append(best['J'])
    for k, uav in enumerate(uavs):
        uav.x[t] = adjust_location(best['uav_locs'][k,0], bound=boundary)
        uav.y[t] = adjust_location(best['uav_locs'][k,1], bound=boundary)
        uav.z[t] = adjust_altitude(best['uav_locs'][k,2], zmin=z_min, zmax=z_max)
        vel_k = best['velocities'][k]
        uav.velocity[t] = np.linalg.norm(vel_k)
        uav.pw_prop_W[t] = cal_uav_propulsion_energy(V=np.linalg.norm(vel_k[:2]), Vz=vel_k[2])[0]
        uav.vqlen_prop[t] = update_vqlen_prop(uav.vqlen_prop[t-1], uav.pw_prop_W[t]) if t > 0 else 0.0
    airship_pos = best['airship_pos'].copy()
    
    # (3.1) Store the transition, periodic supervised actor update, energy queue
    ctrl.commit_slot(best)
    if (t >= t_training_start) and (t % training_interval == 0): 
        ctrl.train_step()
    ctrl.update_energy(P_exec=best['P'])
    
    # (4-5) Executed Block A outcome: bandwidth + access/backhaul rates + delivery
    _res = best['res']
    R_access = _res['R_access_Mb']
    D_opt = _res['D_Mb']
    S_u = _res['S_u']
    R_bh = _res['R_bh_Mb']
    bw_alloc_Hz_all = _res['W_ui_Hz']
    bw_alloc_all = bw_alloc_Hz_all / bw_per_user
    bh_capa_log.append(np.mean(R_bh))
    bh_util_log.append(np.sum(S_u) / max(np.sum(R_bh), 1e-9))
    reward_log.append(best['J'])
    if t % 20 == 0:
        pw, bd = best['all_comp'][best['k']]
        pos_s = np.round(best['uav_locs'][:, :2], 0).astype(int).tolist()
        a_s = np.round(best['airship_pos'][:2], 0).astype(int).tolist()
        Js = np.round(np.asarray(best['all_J']), 0).astype(int).tolist()
        print(f"[diag t={t:3d}] k*={best['k']} J={best['J']:8.1f} pow={pw:7.1f} "
              f"betaD={bd:8.1f} sumRacc={np.sum(R_access):6.1f} sumRbh={np.sum(R_bh):6.1f} "
              f"sumD={np.sum(D_opt):6.1f} E={np.round(ctrl.E,0).tolist()} "
              f"Qavg={np.mean(queue_length_Mb):6.1f} pos={pos_s} air={a_s} Js={Js}")
    
    # (6) User updates with the delivery rates D
    for id, user in enumerate(users):
        cid = cids[id]
        user.cluster[t] = cids[id]
        user.active_state[t] = isActive_all[id]
        user.alpha[t] = bw_alloc_all[id]
        user.chgains_downlink[t] = cal_channel_fading_gain(user.x[t], user.y[t], uavs[cid].x[t], uavs[cid].y[t], uavs[cid].z[t])
        user.channel_capacity_Mb[t] = R_access[id]
        qlen_prev_Mb = user.queue_length_Mb[t-1] if t > 0 else 0
        user.downlink_throughput_Mb[t] = D_opt[id]
        user.queue_length_Mb[t] = update_queue_Mb(qlen_prev_Mb, user.arrival_traffic_Mb[t], D_opt[id])
        user.ema_qlen_Mb[t] = alpha_ema*user.queue_length_Mb[t] + (1-alpha_ema)*user.ema_qlen_Mb[t-1]
        ttype_t = int(user.traffic_type[t])
        ld_type_Mbps = ON_data_arrival_mean_Mb[ttype_t] if ttype_t >= 0 else user.lambd_Mb
        user.mos[t] = mos_func(qlen=user.ema_qlen_Mb[t], lambda_=ld_type_Mbps, ss_min_arr=ss_min_types[ttype_t], ss_max_arr=ss_max_types[ttype_t]) if (user.traffic_state[t]==True) else np.nan
        user.delay[t] = d0 + user.ema_qlen_Mb[t] / ld_type_Mbps

print(f"Backhaul [Proposed-JAC]: avg capacity/drone = {np.mean(bh_capa_log):.1f} Mb/slot, avg utilization = {np.mean(bh_util_log):.3f}")
print(f"Critic J*: mean = {np.mean(bestJ_log):.2f}, last-100 avg = {np.mean(bestJ_log[-100:]):.2f}")
print(f"Energy queue E (W*slot): final = {np.round(ctrl.E, 2)}")
print(f"Airship final position: {np.round(airship_pos, 1)}, movable = {ctrl.airship_movable}")
print(f"Actor replay buffer size: {len(ctrl.buffer)}")
_tot = max(sum(ctrl.win_counts.values()), 1)
print("Slot win rates: " + ", ".join(f"{k}={v/_tot:.1%}"
      for k, v in ctrl.win_counts.items() if v > 0))
print(f"J gap best-vs-2nd avg = {np.mean(ctrl.gap_best2nd):.2f} Mb, "
      f"best-vs-stay avg = {np.mean(ctrl.gap_stay):.2f} Mb")
print(f"action norm avg = {np.mean(ctrl.a_norm):.3f} (max {np.max(ctrl.a_norm):.3f}), "
      f"executed speed avg = {np.mean(ctrl.v_mean):.2f} m/s")
'''


def patch_once(src, old, new, name, count=1):
    n = src.count(old)
    assert n == count, f'{name}: expected {count} occurrence(s), found {n}'
    return src.replace(old, new)


def kpi_of(base):
    rows = {}
    for m in ["Stationary", "Centroid", "Proposed"]:
        df = pd.read_excel(os.path.join(base, f"{m}-kpis-over-time.xlsx"))
        rows[m] = {
            "weighted_mos": sum(W[k] * df[f"mos-avg-{n}"].mean()
                                for k, n in enumerate(["Web", "Video", "Messaging"])),
            "web": df["mos-avg-Web"].mean(),
            "video": df["mos-avg-Video"].mean(),
            "msg": df["mos-avg-Messaging"].mean(),
            "qlen_Mb": df["qlen-Mb-avg (active)"].mean(),
        }
    return rows


results = {}
try:
    # ---------------- patch notebook ----------------
    nb = json.loads(nb_template.decode("utf-8"))

    # cell 0: + cvxpy (early) and blocka_optimizer imports
    src0 = "".join(nb["cells"][0]["source"])
    src0 = patch_once(src0, "import os\n", "import cvxpy   # imported first: avoids loader-order conflicts\nimport os\n", "cell0 cvxpy")
    src0 = patch_once(src0, "from traffic_model import *",
                      "from traffic_model import *\nfrom blocka_optimizer import *   # BLOCKA variable-bandwidth allocator",
                      "cell0 blocka")
    nb["cells"][0]["source"] = src0.splitlines(keepends=True)

    # cell 4: SFM mobility
    src4 = "".join(nb["cells"][4]["source"])
    assert src4.count("bound={boundary}.xlsx") == 3, src4.count("bound={boundary}.xlsx")
    src4 = src4.replace("bound={boundary}.xlsx", "bound={boundary}-sfm.xlsx")
    i0 = src4.index("    # Gen user movements using the Gauss-Markov model")
    i1 = src4.index("speed_avg=speed_user_avg\n        )") + len("speed_avg=speed_user_avg\n        )")
    src4 = (src4[:i0]
            + "    # Gen user movements using the Social Force Model (SFM)  [comparison run]\n"
              "    xaxis_all, yaxis_all = generate_mobility_SFM(\n"
              "        n_users=n_users, n_slots=n_slots, slot_len=slot_len, boundary=boundary,\n"
              "        speed_avg=speed_user_avg, seed=11\n"
              "    )"
            + src4[i1:])
    nb["cells"][4]["source"] = src4.splitlines(keepends=True)

    # cell 6: Weibull arrival
    nb["cells"][6]["source"] = NEW_CELL6.splitlines(keepends=True)

    # cell 8: load SFM locations + Weibull arrival
    src8 = "".join(nb["cells"][8]["source"])
    assert src8.count("bound={boundary}.xlsx") == 2, src8.count("bound={boundary}.xlsx")
    src8 = src8.replace("bound={boundary}.xlsx", "bound={boundary}-sfm.xlsx")
    assert src8.count("arrival traffic, nusers=") == 1, src8.count("arrival traffic, nusers=")
    src8 = src8.replace("arrival traffic, nusers=", "arrival traffic, dist=weibull, nusers=")
    nb["cells"][8]["source"] = src8.splitlines(keepends=True)

    # cells 22/24: Stationary / Centroid -> BLOCKA (identical to run_blocka_comparison)
    for idx in (22, 24):
        src = "".join(nb["cells"][idx]["source"])
        src = patch_once(src, OLD_BW_BLOCK, NEW_BW_BLOCK, f"cell{idx} bw")
        src = patch_once(src, OLD_BH_BLOCK, NEW_BH_BLOCK, f"cell{idx} bh")
        src = patch_once(src, OLD_DELIV, NEW_DELIV, f"cell{idx} deliv")
        nb["cells"][idx]["source"] = src.splitlines(keepends=True)

    # cell 50: metadata references the old controller's V_lyap attribute
    src50 = "".join(nb["cells"][50]["source"])
    src50 = patch_once(src50, "    'Lyapunov parameter V': ctrl.V_lyap, ",
                       "    'psi_E (energy weight)': ctrl.psi_E, \n"
                       "    'airship movable': ctrl.airship_movable, \n"
                       "    'critic candidates K': ctrl.n_candidates, ", "cell50 metadata")
    nb["cells"][50]["source"] = src50.splitlines(keepends=True)

    # cell 26: full replacement by the joint actor-critic
    nb["cells"][26]["source"] = (NEW_CELL26
                                 .replace("__AIRSHIP_MOVABLE__", MOVABLE)
                                 .replace("__MODE__", repr(MODE))
                                 .splitlines(keepends=True))

    for idx in (4, 6, 8, 22, 24, 26, 50):      # cell 0 has %magic, skip compile
        compile("".join(nb["cells"][idx]["source"]), f"<cell {idx}>", "exec")

    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"notebook patched (SFM + Weibull + BLOCKA + joint AC, airship_movable={MOVABLE})", flush=True)

    # ---------------- patch output folder tag ----------------
    ps = params_template.decode("utf-8")
    old_t = "-pureLoS-K20-s1.1, n_slots={n_slots}, "
    assert old_t in ps, "folder tag not found"
    ps = ps.replace(old_t, f"-pureLoS-K20-s1.1{TAG}, n_slots={{n_slots}}, ")
    open(PARAMS, "w", encoding="utf-8").write(ps)
    print(f"params patched (folder tag {TAG})", flush=True)

    if DRYRUN:
        print("[dryrun] patch OK, restoring without execution", flush=True)
        results = {}
        raise SystemExit(0)

    # ---------------- run the full notebook ----------------
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    r = subprocess.run([PY, "-m", "nbconvert", "--to", "notebook", "--execute", "main.ipynb",
                        "--output", "main_executed.ipynb",
                        "--ExecutePreprocessor.timeout=144000",
                        "--ExecutePreprocessor.kernel_name=uav-drl"],
                       cwd=PROJ, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"NOTEBOOK FAILED\n--- stderr tail ---\n{r.stderr[-6000:]}")
        sys.exit(1)
    print("notebook executed OK", flush=True)

    # scan executed notebook for silent BLOCKA solver fallbacks
    try:
        ex = json.load(open(os.path.join(PROJ, "main_executed.ipynb"), encoding="utf-8"))
        warns = 0
        for c in ex.get("cells", []):
            for o in c.get("outputs", []):
                txt = "".join(o.get("text", []))
                warns += txt.count("[blocka] solve failed") + txt.count("[blocka] solver")
        print(f"BLOCKA solver warnings in executed notebook: {warns}", flush=True)
    except Exception as e:
        print(f"(could not scan warnings: {e})")

    # ---------------- collect results (whatever folders exist) --------------
    def find_dir(tag):
        g = glob.glob(os.path.join(PROJ, rf"simulation\sim-output\*K20-s1.1{tag}, n_slots=*"))
        return g[0] if g else None

    for label, tag in [("joint AC (movable airship)", "-sfm-wb-jac"),
                       ("joint AC (airship fixed, ablation)", "-sfm-wb-jac-fa"),
                       ("joint AC directed-only, no actor", "-sfm-wb-jac-nd"),
                       ("joint AC directed+local search", "-sfm-wb-jac-ndls"),
                       ("hybrid REINFORCE + BLOCKA (before)", "-sfm-wb-blocka")]:
        d = find_dir(tag)
        if d and all(os.path.exists(os.path.join(d, f"{m}-kpis-over-time.xlsx"))
                     for m in ["Stationary", "Centroid", "Proposed"]):
            results[label] = kpi_of(d)
        else:
            print(f"[skip] {label}: {tag} not ready")

finally:
    open(NB, "wb").write(nb_template)
    open(PARAMS, "wb").write(params_template)
    print("baseline notebook/params restored", flush=True)

# ---------------- summary ----------------
if results:
    rows = []
    for scheme, d in results.items():
        for m, v in d.items():
            rows.append({"scheme": scheme, "method": m, **{k: round(x, 3) for k, x in v.items()}})
    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(os.path.dirname(PROJ), "jointac_vs_baseline_results.xlsx"), index=False)

    print(f"\n===== Joint actor-critic ({TAG}) vs previous methods (SFM + Weibull) =====")
    print(df.pivot_table(index="method", columns="scheme",
                         values=["weighted_mos", "qlen_Mb"]).round(3).to_string())
    print("\nsaved jointac_vs_baseline_results.xlsx")
else:
    print("no result folders found yet")

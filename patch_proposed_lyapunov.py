# -*- coding: utf-8 -*-
'''Replace the Proposed cell (cell 26) with the Lyapunov-relay implementation:
   actor-decided 3D movement + inner delivery subproblem (bisection + vertex
   rounding) + backhaul virtual queues + REINFORCE on drift-minus-reward.
'''
import json, sys

PROJ = r"D:\Learn\work\drl-zcode\uav-bs-placement-drl-modified"
NB = PROJ + r"\main.ipynb"

nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

src26 = "".join(cells[26]["source"])
assert "uav_swarm_proposed" in src26 and "users_proposed" in src26, "cell 26 mismatch"

NEW26 = '''# Modified (v6): Proposed method = Lyapunov controller with backhaul virtual queues
#   - movement: a shared PyTorch actor senses (traffic heatmap, own position, airship
#     position, all backhaul virtual queues Z) and outputs normalized (vx, vy, vz)
#   - delivery: per slot the rates D_i solve the inner subproblem (critical-price
#     bisection + vertex rounding) under the hard backhaul constraint tau*R_bh_u;
#     bandwidth allocation itself is NOT compressed
#   - learning: reward = -(drift_Q + drift_Z) + V * MOS, actor updated by REINFORCE
from relay_lyapunov_controller import RelayLyapunovController

# Initiate the UAV Swarm (relay drones with the original 3D movement) 
uav_swarm_proposed = [UAV(name= f'UAV {i} (Proposed)', x_init=init_locations_uav[i][0], y_init=init_locations_uav[i][1], z_init=z_init_proposed, 
                        xlim = xlim_uav[i], ylim = ylim_uav[i], speedxy_max=uav_speedxy_max, speedz_max=uav_speedz_max) for i in range(0,n_uavs)]

# Lyapunov relay controller (see relay_lyapunov_controller.py)
ctrl = RelayLyapunovController(
    n_users=n_users, n_uavs=n_uavs, boundary=boundary, grid_size=grid_size,
    queue_norm=queue_norm_Mb, V_lyap=lyapunov_param, tau=slot_len,
    ss_min_types=ss_min_types, ss_max_types=ss_max_types,
    mos_weights=mos_type_weights, type_rates=ON_data_arrival_mean_Mb, d0=d0,
    sigma_max=std_var_vxvyvz_explore, sigma_min=std_var_vxvyvz_exploit,
    sigma_decay_slots=n_slots, seed=0
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
    
    # (2) Access bandwidth pools (same rule as the baseline methods: 30/60/10 by type)
    bw_alloc_Hz_all = np.zeros(shape=(n_users,))
    ttype_all_t = traffic_type[:, t]
    for k in range(len(bw_type_pool_Hz)): 
        ids_k = np.where(ttype_all_t == k)[0]
        active_k = ids_k[isActive_all[ids_k]]
        if len(active_k) > 0: 
            bw_alloc_Hz_all[active_k] = bw_type_pool_Hz[k] / len(active_k)
    bw_alloc_all = bw_alloc_Hz_all / bw_per_user      # fraction of the reference per-user bw
    
    # (3) UAV movement decided by the actor (heatmap + positions + backhaul virtual queues)
    heatmap_t = gen_heatmap(xaxis_all[:,t], yaxis_all[:,t], queue_length_Mb + traffic_Mb, queue_norm_Mb)
    uav_locs = np.array([[uav.x[t-1], uav.y[t-1], uav.z[t-1]] for uav in uavs]) if t>0 \\
               else np.array([[uav.x[0], uav.y[0], uav.z[0]] for uav in uavs])
    actions_t, logps_t = ctrl.select_actions(
        uav_locs=uav_locs, airship_loc=np.array([airship_x, airship_y, airship_z]),
        heatmap=heatmap_t, t=t
    )
    for k, uav in enumerate(uavs):
        vx, vy, vz = actions_t[k]
        xt = uav.x[t-1] + vx*slot_len
        yt = uav.y[t-1] + vy*slot_len
        zt = uav.z[t-1] + vz*slot_len
        uav.x[t] = adjust_location(xt, bound=boundary)
        uav.y[t] = adjust_location(yt, bound=boundary)
        uav.z[t] = adjust_altitude(zt, zmin=z_min, zmax=z_max)
        uav.velocity[t] = np.sqrt(vx**2 + vy**2 + vz**2)
        uav.pw_prop_W[t] = cal_uav_propulsion_energy(V=uav.velocity[t])[0]
        uav.vqlen_prop[t] = update_vqlen_prop(uav.vqlen_prop[t-1], uav.pw_prop_W[t]) if t > 0 else 0.0
    
    # (3.1) Periodic actor update (REINFORCE on the Lyapunov drift-minus-reward)
    if (t >= t_training_start) and (t % training_interval == 0): 
        ctrl.train_step()
    
    # (4) Access rates at the new positions (Mb per slot)
    R_access = np.zeros(shape=(n_users,))
    for k in range(n_uavs): 
        ids_k = np.where(cids == k)[0]
        if len(ids_k) == 0: 
            continue
        g_k = cal_channel_fading_gain(xaxis_all[ids_k, t], yaxis_all[ids_k, t], uavs[k].x[t], uavs[k].y[t], uavs[k].z[t])
        R_access[ids_k] = bw_alloc_Hz_all[ids_k] * np.log2(1 + pTx_downlink * g_k / noise_pw_total) * slot_len / 1e6
    
    # (5) Backhaul rates and the inner delivery subproblem (bisection + vertex rounding)
    R_bh = np.array([cal_backhaul_capacity_Mb(uav.x[t], uav.y[t], uav.z[t]) for uav in uavs])
    D_opt, S_u = ctrl.solve_delivery(
        R_access=R_access, R_backhaul=R_bh, Q=queue_length_Mb, A=traffic_Mb,
        cids=cids, active=isActive_all, ld=ld_Mbps, types=traffic_type[:, t]
    )
    bh_capa_log.append(np.mean(R_bh))
    bh_util_log.append(np.sum(S_u) / max(np.sum(R_bh), 1e-9))
    
    # (6) User updates with the delivery rates D
    mos_post = np.full(shape=(n_users,), fill_value=np.nan)
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
        # post-delivery MOS used inside the Lyapunov reward
        if user.traffic_state[t] == True: 
            qlen_post = np.max([qlen_prev_Mb + user.arrival_traffic_Mb[t] - D_opt[id], 0])
            mos_post[id] = mos_func(qlen=qlen_post, lambda_=ld_type_Mbps, ss_min_arr=ss_min_types[ttype_t], ss_max_arr=ss_max_types[ttype_t])
    
    # (7) Lyapunov reward, virtual-queue update, store the transition for REINFORCE
    reward_t = ctrl.compute_reward(Q=queue_length_Mb, A=traffic_Mb, D=D_opt, S_u=S_u,
                                   R_bh=R_bh, mos_mean=np.nanmean(mos_post))
    ctrl.update_virtual_queues(S_u=S_u, R_bh=R_bh)
    ctrl.store_transition(
        heatmap=heatmap_t, uav_locs=uav_locs,
        airship_loc=np.array([airship_x, airship_y, airship_z]),
        actions=actions_t, logps=logps_t, reward=reward_t
    )
    reward_log.append(reward_t)

print(f"Backhaul [Proposed-Lyapunov]: avg capacity/drone = {np.mean(bh_capa_log):.1f} Mb/slot, avg utilization = {np.mean(bh_util_log):.3f}")
print(f"Lyapunov reward: mean = {np.mean(reward_log):.2f}, last-100 avg = {np.mean(reward_log[-100:]):.2f}")
print(f"Backhaul virtual queue Z (Mb): final = {np.round(ctrl.Z, 2)}, max ever = {np.round(ctrl.Z_max, 2)}")
'''

cells[26]["source"] = NEW26.splitlines(keepends=True)

# syntax-check every code cell (skip IPython magics)
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        src_check = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        try:
            compile(src_check, f"<cell {i}>", "exec")
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {i}: {e}")
            sys.exit(1)

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("cell 26 replaced with Lyapunov-relay implementation")

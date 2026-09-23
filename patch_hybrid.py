# -*- coding: utf-8 -*-
'''Patch main.ipynb cell 26: switch the Proposed method to the hybrid candidate-evaluation
   controller (actor proposes K candidates, true-model scoring, best executed).'''
import json, sys

PROJ = r"D:\Learn\work\drl-zcode\uav-bs-placement-drl-modified"
NB = PROJ + r"\main.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

src = "".join(cells[26]["source"])
assert "uav_swarm_proposed" in src and "RelayLyapunovController" in src

def rep(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, f"expected {count} of {old!r}, found {n}"
    src = src.replace(old, new)

# 1) header comment + import
rep("# Modified (v6): Proposed method = Lyapunov controller with backhaul virtual queues\n"
    "#   - movement: a shared PyTorch actor senses (traffic heatmap, own position, airship\n"
    "#     position, all backhaul virtual queues Z) and outputs normalized (vx, vy, vz)\n"
    "#   - delivery: per slot the rates D_i solve the inner subproblem (critical-price\n"
    "#     bisection + vertex rounding) under the hard backhaul constraint tau*R_bh_u;\n"
    "#     bandwidth allocation itself is NOT compressed\n"
    "#   - learning: reward = -(drift_Q + drift_Z) + V * MOS, actor updated by REINFORCE\n"
    "from relay_lyapunov_controller import RelayLyapunovController",
    "# Modified (v13): Proposed method = hybrid candidate evaluation + Lyapunov relay\n"
    "#   - movement: per drone the actor proposes K candidate moves; EACH candidate is\n"
    "#     scored with the true system model (channel -> inner delivery subproblem ->\n"
    "#     Lyapunov score) and the best is executed (works with zero training)\n"
    "#   - delivery: per slot the rates D_i solve the inner subproblem (critical-price\n"
    "#     bisection + vertex rounding) under the hard backhaul constraint tau*R_bh_u;\n"
    "#     bandwidth allocation itself is NOT compressed\n"
    "#   - learning: REINFORCE reinforces every sampled candidate with advantage\n"
    "#     (J_cand - mean(J)) / std, i.e. self-critical style; reward log kept as before\n"
    "from relay_hybrid_controller import HybridCandidateController")

# 2) controller instantiation
rep("""# Lyapunov relay controller (see relay_lyapunov_controller.py)
ctrl = RelayLyapunovController(
    n_users=n_users, n_uavs=n_uavs, boundary=boundary, grid_size=grid_size,
    queue_norm=queue_norm_Mb, V_lyap=lyapunov_param, tau=slot_len,
    ss_min_types=ss_min_types, ss_max_types=ss_max_types,
    mos_weights=mos_type_weights, type_rates=ON_data_arrival_mean_Mb, d0=d0,
    sigma_max=std_var_vxvyvz_explore, sigma_min=std_var_vxvyvz_exploit,
    sigma_decay_slots=n_slots, seed=0
)""",
"""# Hybrid Lyapunov relay controller (see relay_hybrid_controller.py)
ctrl = HybridCandidateController(
    n_users=n_users, n_uavs=n_uavs, boundary=boundary, grid_size=grid_size,
    queue_norm=queue_norm_Mb, V_lyap=lyapunov_param, tau=slot_len,
    ss_min_types=ss_min_types, ss_max_types=ss_max_types,
    mos_weights=mos_type_weights, type_rates=ON_data_arrival_mean_Mb, d0=d0,
    sigma_decay_slots=n_slots, seed=0,
    n_candidates=10, eval_sigma=std_var_vxvyvz_explore
)
ctrl.bind_models(cal_channel_fading_gain, cal_backhaul_capacity_Mb, pTx_downlink, noise_pw_total)""")

# 3) movement block: direct actor action -> candidate generation + scoring
rep("""    # (3) UAV movement decided by the actor (heatmap + positions + backhaul virtual queues)
    heatmap_t = gen_heatmap(xaxis_all[:,t], yaxis_all[:,t], queue_length_Mb + traffic_Mb, queue_norm_Mb)
    uav_locs = np.array([[uav.x[t-1], uav.y[t-1], uav.z[t-1]] for uav in uavs]) if t>0 \\
               else np.array([[uav.x[0], uav.y[0], uav.z[0]] for uav in uavs])
    actions_t, logps_t = ctrl.select_actions(
        uav_locs=uav_locs, airship_loc=np.array([airship_x, airship_y, airship_z]),
        heatmap=heatmap_t, t=t
    )
    for k, uav in enumerate(uavs):
        vx, vy, vz = actions_t[k]""",
"""    # (3) UAV movement (hybrid): actor proposes K candidates per drone, each scored with
    # the true system model (channel -> D-subproblem -> Lyapunov score); best is executed
    heatmap_t = gen_heatmap(xaxis_all[:,t], yaxis_all[:,t], queue_length_Mb + traffic_Mb, queue_norm_Mb)
    uav_locs = np.array([[uav.x[t-1], uav.y[t-1], uav.z[t-1]] for uav in uavs]) if t>0 \\
               else np.array([[uav.x[0], uav.y[0], uav.z[0]] for uav in uavs])
    ctrl.begin_slot(heatmap_t, uav_locs, np.array([airship_x, airship_y, airship_z]))
    for k, uav in enumerate(uavs):
        best_a, best_J = ctrl.decide_and_score(
            k=k, uav_loc=uav_locs[k], airship_loc=np.array([airship_x, airship_y, airship_z]),
            heatmap=heatmap_t, user_x=xaxis_all[:,t], user_y=yaxis_all[:,t],
            bw_alloc_Hz=bw_alloc_Hz_all, Q=queue_length_Mb, A=traffic_Mb, ld=ld_Mbps,
            types=traffic_type[:,t], cids=cids,
            speed_xy=uav_speedxy_max, speed_z=uav_speedz_max, zmin=z_min, zmax=z_max,
            boundary=boundary)
        vx, vy, vz = best_a""")

# 4) store transition -> commit slot candidates
rep("""    # store the transition BEFORE updating Z, so the stored state Z matches the state
    # in which the action was actually selected (Z at decision time, not post-update)
    ctrl.store_transition(
        heatmap=heatmap_t, uav_locs=uav_locs,
        airship_loc=np.array([airship_x, airship_y, airship_z]),
        actions=actions_t, logps=logps_t, reward=reward_t
    )
    ctrl.update_virtual_queues(S_u=S_u, R_bh=R_bh)""",
"""    # commit the candidate records (state snapshot taken in begin_slot at decision time),
    # then update the backhaul virtual queues
    ctrl.commit_slot(reward_t)
    ctrl.update_virtual_queues(S_u=S_u, R_bh=R_bh)""")

# 5) also refresh the header line "# Simulation starts!" context: add best-J log lists
rep("bh_capa_log, bh_util_log, reward_log = [], [], []",
    "bh_capa_log, bh_util_log, reward_log = [], [], []\nbestJ_log = []")
rep("""        boundary=boundary)
        vx, vy, vz = best_a""",
    """        boundary=boundary)
        bestJ_log.append(best_J)
        vx, vy, vz = best_a""")
rep("""print(f"Backhaul virtual queue Z (Mb): final = {np.round(ctrl.Z, 2)}, max ever = {np.round(ctrl.Z_max, 2)}")""",
    """print(f"Backhaul virtual queue Z (Mb): final = {np.round(ctrl.Z, 2)}, max ever = {np.round(ctrl.Z_max, 2)}")
print(f"Candidate evaluation score J: mean = {np.mean(bestJ_log):.2f}, last-100 avg = {np.mean(bestJ_log[-100:]):.2f}")""")

# write the patched source back into the cell
cells[26]["source"] = src.splitlines(keepends=True)

# syntax-check every code cell (skip IPython magics)
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        s = "".join(c["source"])
        s = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith(("%", "!")))
        try:
            compile(s, f"<cell {i}>", "exec")
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {i}: {e}")
            sys.exit(1)

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("cell 26 switched to hybrid candidate-evaluation controller")

# -*- coding: utf-8 -*-
'''SFM + Weibull main scheme: variable-bandwidth BLOCKA (convex P_R) vs the
   fixed type-pool bandwidth model (30/60/10), all three placement methods.

   Patches main.ipynb:
     cell 0 : + cvxpy / blocka_optimizer import
     cell 4 : SFM mobility (reuses bound=250-sfm.xlsx)
     cell 6 : Weibull arrival from the existing ON/OFF schedule
     cell 8 : load SFM locations + Weibull arrival
     cell 22/24 (Stationary/Centroid): type-pool bw + bh_scale hard cap
              replaced by one BLOCKA solve per slot (W_ui, W_ua, D)
     cell 26 (Proposed): candidate scoring keeps the cheap type-pool proxy,
              executed allocation + delivery replaced by BLOCKA
     params_sys.py folder tag: -sfm-wb-blocka
   Restores main.ipynb and params_sys.py afterwards; compares against the
   existing fixed-bw SFM+Weibull folder (-sfm-wb).
'''
import json, os, glob, subprocess, sys
import numpy as np
import pandas as pd

PROJ = r"D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11"
NB = os.path.join(PROJ, "main.ipynb")
PARAMS = os.path.join(PROJ, "params_sys.py")
PY = sys.executable
W = {0: 0.35, 1: 0.30, 2: 0.35}          # MOS weights Web/Video/Messaging

nb_template = open(NB, "rb").read()
params_template = open(PARAMS, "rb").read()

# ---------------- shared cell sources ----------------
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

# (2) type-pool block (identical in cells 22/24) -> keep only the type alias
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

# (3.5) backhaul hard-cap block (identical in cells 22/24) -> BLOCKA solve
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

# (4) per-user delivery lines (identical in cells 22/24) -> use BLOCKA output
OLD_DELIV = '''        user.channel_capacity_Mb[t] = cal_channel_capacity_Mb(user.chgains_downlink[t], user.alpha[t])[0]
        
        qlen_prev_Mb = user.queue_length_Mb[t-1] if t > 0 else 0
        depart_Mb = cal_downlink_throughput_Mb(qlen_prev_Mb, user.arrival_traffic_Mb[t], user.channel_capacity_Mb[t]) * bh_scale[cid]
'''
NEW_DELIV = '''        user.channel_capacity_Mb[t] = R_access[id]
        
        qlen_prev_Mb = user.queue_length_Mb[t-1] if t > 0 else 0
        depart_Mb = D_opt[id]
'''

# cell 26: (4)+(5) -> BLOCKA; (2) type-pool block stays as the scoring proxy
OLD_P26 = '''    # (4) Access rates at the new positions (Mb per slot)
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
'''
NEW_P26 = '''    # (4-5) BLOCKA: joint variable-bandwidth + delivery optimisation (convex P_R)
    # at the executed positions; the type-pool bw from (2) is only the scoring proxy
    _res = blocka_allocate(
        cids=cids, user_x=xaxis_all[:, t], user_y=yaxis_all[:, t],
        uav_locs=np.array([[uav.x[t], uav.y[t], uav.z[t]] for uav in uavs]),
        airship_loc=np.array([airship_x, airship_y, airship_z]),
        queue_Mb=queue_length_Mb, traffic_Mb=traffic_Mb, types=traffic_type[:, t])
    R_access = _res['R_access_Mb']
    D_opt = _res['D_Mb']
    S_u = _res['S_u']
    R_bh = _res['R_bh_Mb']
    bw_alloc_Hz_all = _res['W_ui_Hz']          # for alpha logging only
    bw_alloc_all = bw_alloc_Hz_all / bw_per_user
    bh_capa_log.append(np.mean(R_bh))
    bh_util_log.append(np.sum(S_u) / max(np.sum(R_bh), 1e-9))
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

    # cell 4: SFM mobility (same patch as run_sfm_comparison.py)
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

    # cell 6: Weibull arrival from the existing schedule
    nb["cells"][6]["source"] = NEW_CELL6.splitlines(keepends=True)

    # cell 8: load SFM locations + Weibull arrival (state/type unchanged)
    src8 = "".join(nb["cells"][8]["source"])
    assert src8.count("bound={boundary}.xlsx") == 2, src8.count("bound={boundary}.xlsx")
    src8 = src8.replace("bound={boundary}.xlsx", "bound={boundary}-sfm.xlsx")
    assert src8.count("arrival traffic, nusers=") == 1, src8.count("arrival traffic, nusers=")
    src8 = src8.replace("arrival traffic, nusers=", "arrival traffic, dist=weibull, nusers=")
    nb["cells"][8]["source"] = src8.splitlines(keepends=True)

    # cells 22/24: Stationary / Centroid -> BLOCKA replaces type-pool + bh_scale
    for idx in (22, 24):
        src = "".join(nb["cells"][idx]["source"])
        src = patch_once(src, OLD_BW_BLOCK, NEW_BW_BLOCK, f"cell{idx} bw")
        src = patch_once(src, OLD_BH_BLOCK, NEW_BH_BLOCK, f"cell{idx} bh")
        src = patch_once(src, OLD_DELIV, NEW_DELIV, f"cell{idx} deliv")
        nb["cells"][idx]["source"] = src.splitlines(keepends=True)

    # cell 26: Proposed -> BLOCKA for the executed allocation/delivery
    src26 = "".join(nb["cells"][26]["source"])
    src26 = patch_once(src26, OLD_P26, NEW_P26, "cell26")
    nb["cells"][26]["source"] = src26.splitlines(keepends=True)

    for idx in (4, 6, 8, 22, 24, 26):      # cell 0 has %magic, skip compile
        compile("".join(nb["cells"][idx]["source"]), f"<cell {idx}>", "exec")

    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("notebook patched (SFM + Weibull + BLOCKA var-bw)", flush=True)

    # ---------------- patch output folder tag ----------------
    ps = params_template.decode("utf-8")
    old_t = "-pureLoS-K20-s1.1, n_slots={n_slots}, "
    assert old_t in ps, "folder tag not found"
    ps = ps.replace(old_t, "-pureLoS-K20-s1.1-sfm-wb-blocka, n_slots={n_slots}, ")
    open(PARAMS, "w", encoding="utf-8").write(ps)
    print("params patched (folder tag -sfm-wb-blocka)", flush=True)

    # ---------------- run the full notebook ----------------
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    r = subprocess.run([PY, "-m", "nbconvert", "--to", "notebook", "--execute", "main.ipynb",
                        "--output", "main_executed.ipynb",
                        "--ExecutePreprocessor.timeout=144000",
                        "--ExecutePreprocessor.kernel_name=uav-drl"],
                       cwd=PROJ, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"NOTEBOOK FAILED\n{r.stderr[-2000:]}")
        sys.exit(1)
    print("notebook executed OK", flush=True)

    # ---------------- collect results ----------------
    blk_dir = glob.glob(os.path.join(PROJ, r"simulation\sim-output\*K20-s1.1-sfm-wb-blocka, n_slots=*"))[0]
    fix_dir = glob.glob(os.path.join(PROJ, r"simulation\sim-output\*K20-s1.1-sfm-wb, n_slots=*"))[0]
    results["var-bw BLOCKA"] = kpi_of(blk_dir)
    results["fixed-bw (30/60/10)"] = kpi_of(fix_dir)

finally:
    open(NB, "wb").write(nb_template)
    open(PARAMS, "wb").write(params_template)
    print("baseline notebook/params restored", flush=True)

# ---------------- summary ----------------
rows = []
for scheme, d in results.items():
    for m, v in d.items():
        rows.append({"scheme": scheme, "method": m, **{k: round(x, 3) for k, x in v.items()}})
df = pd.DataFrame(rows)
df.to_excel(os.path.join(os.path.dirname(PROJ), "blocka_varbw_vs_fixedbw_results.xlsx"), index=False)

print("\n===== BLOCKA variable bw vs fixed bw (SFM + Weibull) =====")
print(df.pivot_table(index="method", columns="scheme",
                     values=["weighted_mos", "qlen_Mb"]).round(3).to_string())
print("\nsaved blocka_varbw_vs_fixedbw_results.xlsx")

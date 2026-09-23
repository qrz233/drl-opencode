# -*- coding: utf-8 -*-
'''Patch main.ipynb in uav-bs-placement-drl-modified:
   - input traffic file names get a "ntype=3" tag (forces regeneration with 3 service types)
   - altitude analysis list includes 300 m
   - 5 simulation cells: BSs pinned at the area center (no movement), per-type MOS & delay
   - KPI export: weighted-average MOS over service types (replaces the MOS sum)
   - QoE plot y-limit lowered
'''
import json, shutil, sys

PROJ = r"D:\Learn\work\drl-zcode\uav-bs-placement-drl-modified"
NB = PROJ + r"\main.ipynb"
shutil.copyfile(NB, PROJ + r"\main.ipynb.orig")   # backup of the pristine copy

nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

def get_src(i):
    return "".join(cells[i]["source"])

def set_src(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)

def replace_in_cell(i, old, new, count=1, label=""):
    src = get_src(i)
    n = src.count(old)
    assert n == count, f"cell {i} [{label}]: expected {count} occurrence(s) of {old!r}, found {n}"
    set_src(i, src.replace(old, new))

def replace_lines(i, matcher, new_lines, count=1, label=""):
    '''Replace whole lines whose stripped content matches `matcher`.
    new_lines are used verbatim (they must carry their own indentation).'''
    lines = get_src(i).splitlines(keepends=True)
    hits = [k for k, l in enumerate(lines) if matcher(l)]
    assert len(hits) == count, f"cell {i} [{label}]: expected {count} line hit(s), found {len(hits)}"
    for k in hits:
        lines[k] = "".join(nl + "\n" for nl in new_lines)
    set_src(i, "".join(lines))

def insert_before_line(i, matcher, block_lines, label=""):
    lines = get_src(i).splitlines(keepends=True)
    hits = [k for k, l in enumerate(lines) if matcher(l)]
    assert len(hits) == 1, f"cell {i} [{label}]: expected 1 anchor line, found {len(hits)}"
    k = hits[0]
    lines[k:k] = [bl + "\n" for bl in block_lines]
    set_src(i, "".join(lines))

# ---------------- cells 6 & 8: input file tag ----------------
for i, cnt in [(6, 4), (8, 3)]:
    replace_in_cell(
        i,
        ", nslots={n_slots}, lambda_Mb=",
        ", nslots={n_slots}, ntype=3, lambda_Mb=",
        count=cnt,
        label="ntype tag",
    )

# ---------------- cell 17: altitude list ----------------
replace_in_cell(17, "zuav_list = [100, 150, 200]", "zuav_list = [100, 150, 200, 300]", label="zuav_list")

# ---------------- cells 22/24/26/31/33: simulation loops ----------------
SIM_CELLS = [22, 24, 26, 31, 33]

PIN_BLOCK = [
    "    # Modified: aerial BSs are pinned at the center of the area (altitude fixed at 300 m)",
    "    if uav_fixed_placement: ",
    "        for uav in uavs: ",
    "            uav.x[t], uav.y[t], uav.z[t] = uav.x[0], uav.y[0], uav.z[0]",
    "            uav.velocity[t] = 0",
    "            uav.pw_prop_W[t] = cal_uav_propulsion_energy(V=0)[0]",
]

MOS_LINES = [
    "        ttype_t = int(user.traffic_type[t])   # service type active in this slot (-1 if OFF)",
    "        ld_type_Mbps = ON_data_arrival_mean_Mb[ttype_t] if ttype_t >= 0 else user.lambd_Mb",
    "        user.mos[t] = mos_func(qlen=user.ema_qlen_Mb[t], lambda_=ld_type_Mbps, ss_min_arr=ss_min_types[ttype_t], ss_max_arr=ss_max_types[ttype_t]) if (user.traffic_state[t]==True) else np.nan",
]
DELAY_LINE = ["        user.delay[t] = d0 + user.ema_qlen_Mb[t] / ld_type_Mbps"]

for i in SIM_CELLS:
    # (a) clustering: BSs are co-located, so do not seed KMeans with (degenerate) BS locations
    if i in (22, 24, 26):
        replace_in_cell(i, "kmeans_clustering(loc_users, loc_uavs)", "kmeans_clustering(loc_users)", label="kmeans init")
    elif i == 31:
        replace_in_cell(i, "kmeans_clustering(loc_users, loc_uavs, n_clusters=n_uavs)",
                        "kmeans_clustering(loc_users, n_clusters=n_uavs)", label="kmeans init")
    # cell 33 uses zone_clustering -> no change (asserts removed in utils_sys.py)

    # (b) disable the movement block when BSs are pinned
    replace_lines(i, lambda l: l.strip() == "if t > 0:",
                  ["    if t > 0 and (not uav_fixed_placement): "], label="movement guard")

    # (c) per-type MOS & delay
    replace_lines(i, lambda l: "user.mos[t] = mos_func(qlen=user.ema_qlen_Mb[t], lambda_=" in l,
                  MOS_LINES, label="mos line")
    replace_lines(i, lambda l: "user.delay[t] = d0 + user.ema_qlen_Mb[t] / user.lambd_Mb" in l,
                  DELAY_LINE, label="delay line")

    # (d) pin BS positions each slot
    insert_before_line(i, lambda l: l.startswith("    # (4) User Updates"), PIN_BLOCK, label="pin block")

# ---------------- cell 42: QoE plot y-limit ----------------
replace_in_cell(42, "plt.ylim(bottom=2.5, top=5)", "plt.ylim(bottom=1, top=5)", label="ylim")

# ---------------- cell 53: weighted-average MOS ----------------
replace_lines(
    53,
    lambda l: "mos_all = np.array([user.mos for user in users])" in l,
    [
        "    mos_all = np.array([user.mos for user in users])                            # shape=(n_users,n_slots)",
        "    traffic_type_all = np.array([user.traffic_type for user in users])          # shape=(n_users,n_slots)",
    ],
    label="collect traffic types",
)

replace_lines(
    53,
    lambda l: "mos_total = np.nansum(mos_all, axis=0)" in l,
    [
        "    # Modified: weighted-average MOS over the 3 service types (0.35 Web / 0.25 Video / 0.4",
        "    # Messaging) replaces the previous sum over users. In each slot only types having",
        "    # active users are counted, with the weights re-normalized over those types.",
        "    n_types = len(mos_type_weights)",
        "    mos_weighted = np.full(shape=(mos_all.shape[1],), fill_value=np.nan)",
        "    mos_per_type = {tname: np.full(shape=(mos_all.shape[1],), fill_value=np.nan) for tname in traffic_type_names}",
        "    for tt in range(mos_all.shape[1]): ",
        "        w_sum, val_sum = 0.0, 0.0",
        "        for k in range(n_types): ",
        "            ids = np.where(traffic_type_all[:, tt] == k)[0]",
        "            if len(ids) > 0 and np.any(~np.isnan(mos_all[ids, tt])): ",
        "                mos_k = np.nanmean(mos_all[ids, tt])",
        "                mos_per_type[traffic_type_names[k]][tt] = mos_k",
        "                val_sum += mos_type_weights[k] * mos_k",
        "                w_sum += mos_type_weights[k]",
        "        if w_sum > 0: ",
        "            mos_weighted[tt] = val_sum / w_sum",
    ],
    label="weighted mos block",
)

replace_in_cell(53, "'mos-total':mos_total,", "'mos-weighted-avg':mos_weighted,", label="kpi dict")
replace_in_cell(
    53,
    "'mos_percentile25':mos_percentile25,",
    "'mos_percentile25':mos_percentile25,\n        **{f'mos-avg-{tname}': mos_per_type[tname] for tname in traffic_type_names},",
    label="per-type kpi columns",
)

# ---------------- syntax-check every code cell, then save ----------------
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        src_check = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        try:
            compile(src_check, f"<cell {i}>", "exec")
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {i}: {e}")
            print(src[:800])
            sys.exit(1)

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("main.ipynb patched OK (backup: main.ipynb.orig)")

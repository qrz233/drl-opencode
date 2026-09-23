# -*- coding: utf-8 -*-
'''Patch main.ipynb (modified project) for the relay architecture:
   airship (0,0,250) -> 4 relay drones (3D-moving, 100-200 m) -> users.
   Adds a hard backhaul constraint: each drone's total served downlink rate to its
   users cannot exceed its airship->drone backhaul capacity (40 MHz, Tx 30 dBm,
   pure LoS). Excess stays queued; the bandwidth allocation itself is untouched.
'''
import json, sys

PROJ = r"D:\Learn\work\drl-zcode\uav-bs-placement-drl-modified"
NB = PROJ + r"\main.ipynb"
LABELS = {22: "Stationary", 24: "Centroid", 26: "Proposed", 31: "A2C", 33: "Majority"}

nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

def get_src(i):
    return "".join(cells[i]["source"])

def set_src(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)

def replace_lines(i, matcher, new_lines, count=1, label=""):
    lines = get_src(i).splitlines(keepends=True)
    hits = [k for k, l in enumerate(lines) if matcher(l)]
    assert len(hits) == count, f"cell {i} [{label}]: expected {count} hit(s), found {len(hits)}"
    for k in hits:
        lines[k] = "".join(nl + "\n" for nl in new_lines)
    set_src(i, "".join(lines))

def insert_before_line(i, matcher, block_lines, label=""):
    lines = get_src(i).splitlines(keepends=True)
    hits = [k for k, l in enumerate(lines) if matcher(l)]
    assert len(hits) == 1, f"cell {i} [{label}]: expected 1 anchor, found {len(hits)}"
    k = hits[0]
    lines[k:k] = [bl + "\n" for bl in block_lines]
    set_src(i, "".join(lines))

BH_INIT = ["bh_capa_log, bh_scale_log = [], []"]

BH_BLOCK = [
    "    # (3.5) Modified: backhaul constraint - airship -> drone k hard-caps the drone's",
    "    # total served downlink rate (excess stays queued; bandwidth allocation untouched).",
    "    bh_scale = np.ones(shape=(n_uavs,))",
    "    bh_capa_k = np.zeros(shape=(n_uavs,))",
    "    for k in range(n_uavs): ",
    "        ids_k = np.where(cids == k)[0]",
    "        if len(ids_k) == 0: ",
    "            continue",
    "        g_k = cal_channel_fading_gain(xaxis_all[ids_k, t], yaxis_all[ids_k, t], uavs[k].x[t], uavs[k].y[t], uavs[k].z[t])",
    "        capa_k = bw_alloc_Hz_all[ids_k] * np.log2(1 + pTx_downlink * g_k / noise_pw_total) * slot_len / 1e6   # Mb/slot",
    "        offered_k = np.sum(np.minimum(capa_k, queue_length_Mb[ids_k] + traffic_Mb[ids_k]))",
    "        bh_capa_k[k] = cal_backhaul_capacity_Mb(uavs[k].x[t], uavs[k].y[t], uavs[k].z[t])",
    "        if offered_k > bh_capa_k[k]: ",
    "            bh_scale[k] = bh_capa_k[k] / offered_k",
    "    bh_capa_log.append(np.mean(bh_capa_k))",
    "    bh_scale_log.append(np.min(bh_scale))",
]

QUEUE_LINES = [
    "        depart_Mb = cal_downlink_throughput_Mb(qlen_prev_Mb, user.arrival_traffic_Mb[t], user.channel_capacity_Mb[t]) * bh_scale[cid]",
    "        user.queue_length_Mb[t] = update_queue_Mb(qlen_prev_Mb, user.arrival_traffic_Mb[t], depart_Mb)",
]
THROUGHPUT_LINE = ["        user.downlink_throughput_Mb[t] = depart_Mb"]

for i, name in LABELS.items():
    # (1) log lists before the time loop
    insert_before_line(i, lambda l: l.startswith("for t in range(0, n_slots):"), BH_INIT, label="bh init")

    # (2) backhaul-constraint step right before the user updates
    insert_before_line(i, lambda l: l.startswith("    # (4) User Updates"), BH_BLOCK, label="bh block")

    # (3) served rate = min(capacity, load) * bh_scale; queue drains with the served rate
    replace_lines(i, lambda l: "user.queue_length_Mb[t] = update_queue_Mb(qlen_prev_Mb, user.arrival_traffic_Mb[t], user.channel_capacity_Mb[t])" in l,
                  QUEUE_LINES, label="queue line")
    replace_lines(i, lambda l: "user.downlink_throughput_Mb[t] = cal_downlink_throughput_Mb(" in l,
                  THROUGHPUT_LINE, label="throughput line")

    # (4) summary print at the end of the cell
    src = get_src(i)
    src += (f"\nprint('Backhaul [{name}]: avg capacity/drone = ' +\n"
            f"      f'{{np.mean(bh_capa_log):.1f}} Mb/slot, time-avg of min bh_scale = {{np.mean(bh_scale_log):.3f}} (1.0 = never throttled)')\n")
    set_src(i, src)

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
print("relay backhaul patch OK in cells", list(LABELS))

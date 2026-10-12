# -*- coding: utf-8 -*-
'''SFM + Weibull: rate-proportional type pools vs the other three rules.

   Rate-proportional arm (this run): type k's pool share = mu_k / sum(mu)
   (Web 2.0 / Video 5.0 / Messaging 0.1 -> 28.2% / 70.4% / 1.4%), each pool
   shared equally by that type's active users; backhaul unchanged (5 MHz/drone
   hard cap, same as the alpha and type-pool arms).

   Patches main.ipynb cells 4/6/8 (SFM + Weibull) and the (2) block of
   cells 22/24/26; params folder tag -sfm-wb-ratepool. Restores everything and
   merges into a 4-way table with the existing -sfm-wb-alpha / -sfm-wb /
   -sfm-wb-blocka folders.
'''
import json, os, glob, subprocess, sys
import numpy as np
import pandas as pd

PROJ = r"D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11"
NB = os.path.join(PROJ, "main.ipynb")
PARAMS = os.path.join(PROJ, "params_sys.py")
PY = sys.executable
W = {0: 0.35, 1: 0.30, 2: 0.35}

nb_template = open(NB, "rb").read()
params_template = open(PARAMS, "rb").read()

NEW_CELL6 = '''fpath = os.path.join(sim_in_path, f"arrival traffic, dist=weibull, nusers={n_users}, nslots={n_slots}, ntype=3, lambda_Mb={lambd_Mb:.1f}, uON={ON_duration_mean_tslot:.0f}, uOFF={OFF_duration_mean_tslot:.0f}.xlsx")

if os.path.exists(fpath)==False:
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
NEW_BW_BLOCK = '''    # (2) Bandwidth Allocation (rate-proportional pools): type k's pool share =
    # mu_k / sum(mu) (arrival-rate ratio 2.0 : 5.0 : 0.1 -> 28.2% / 70.4% / 1.4%);
    # each pool is shared equally by that type's active users.
    _mu = np.asarray(ON_data_arrival_mean_Mb, dtype=float)
    _shares = _mu / np.sum(_mu)
    bw_alloc_Hz_all = np.zeros(shape=(n_users,))
    ttype_all_t = traffic_type[:, t]
    for k in range(len(_shares)):
        ids_k = np.where(ttype_all_t == k)[0]
        active_k = ids_k[isActive_all[ids_k]]
        if len(active_k) > 0: 
            bw_alloc_Hz_all[active_k] = _shares[k] * bw_total_Hz / len(active_k)
    bw_alloc_all = bw_alloc_Hz_all / bw_per_user      # fraction of the reference per-user bw
'''

OLD_P26_BW = '''    # (2) Access bandwidth pools (same rule as the baseline methods: 30/60/10 by type)
    bw_alloc_Hz_all = np.zeros(shape=(n_users,))
    ttype_all_t = traffic_type[:, t]
    for k in range(len(bw_type_pool_Hz)): 
        ids_k = np.where(ttype_all_t == k)[0]
        active_k = ids_k[isActive_all[ids_k]]
        if len(active_k) > 0: 
            bw_alloc_Hz_all[active_k] = bw_type_pool_Hz[k] / len(active_k)
    bw_alloc_all = bw_alloc_Hz_all / bw_per_user      # fraction of the reference per-user bw
'''
NEW_P26_BW = '''    # (2) Access bandwidth (rate-proportional pools, see baseline cells); also the
    # scoring proxy for the candidate evaluation
    _mu = np.asarray(ON_data_arrival_mean_Mb, dtype=float)
    _shares = _mu / np.sum(_mu)
    bw_alloc_Hz_all = np.zeros(shape=(n_users,))
    ttype_all_t = traffic_type[:, t]
    for k in range(len(_shares)):
        ids_k = np.where(ttype_all_t == k)[0]
        active_k = ids_k[isActive_all[ids_k]]
        if len(active_k) > 0: 
            bw_alloc_Hz_all[active_k] = _shares[k] * bw_total_Hz / len(active_k)
    bw_alloc_all = bw_alloc_Hz_all / bw_per_user      # fraction of the reference per-user bw
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
    nb = json.loads(nb_template.decode("utf-8"))

    # cell 4: SFM mobility
    src4 = "".join(nb["cells"][4]["source"])
    assert src4.count("bound={boundary}.xlsx") == 3
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

    # cell 8: load SFM + Weibull
    src8 = "".join(nb["cells"][8]["source"])
    assert src8.count("bound={boundary}.xlsx") == 2
    src8 = src8.replace("bound={boundary}.xlsx", "bound={boundary}-sfm.xlsx")
    assert src8.count("arrival traffic, nusers=") == 1
    src8 = src8.replace("arrival traffic, nusers=", "arrival traffic, dist=weibull, nusers=")
    nb["cells"][8]["source"] = src8.splitlines(keepends=True)

    # cells 22/24: rate-proportional pools
    for idx in (22, 24):
        src = "".join(nb["cells"][idx]["source"])
        src = patch_once(src, OLD_BW_BLOCK, NEW_BW_BLOCK, f"cell{idx} bw")
        nb["cells"][idx]["source"] = src.splitlines(keepends=True)

    # cell 26
    src26 = "".join(nb["cells"][26]["source"])
    src26 = patch_once(src26, OLD_P26_BW, NEW_P26_BW, "cell26 bw")
    nb["cells"][26]["source"] = src26.splitlines(keepends=True)

    for idx in (4, 6, 8, 22, 24, 26):
        compile("".join(nb["cells"][idx]["source"]), f"<cell {idx}>", "exec")

    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("notebook patched (SFM + Weibull + rate-proportional pools)", flush=True)

    ps = params_template.decode("utf-8")
    old_t = "-pureLoS-K20-s1.1, n_slots={n_slots}, "
    assert old_t in ps, "folder tag not found"
    ps = ps.replace(old_t, "-pureLoS-K20-s1.1-sfm-wb-ratepool, n_slots={n_slots}, ")
    open(PARAMS, "w", encoding="utf-8").write(ps)
    print("params patched (folder tag -sfm-wb-ratepool)", flush=True)

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

    def find(tag):
        return glob.glob(os.path.join(PROJ, rf"simulation\sim-output\*K20-s1.1-{tag}, n_slots=*"))[0]
    results["rate-proportional pools"] = kpi_of(find("sfm-wb-ratepool"))
    results["alpha equal-split"] = kpi_of(find("sfm-wb-alpha"))
    results["type-pool 30/60/10"] = kpi_of(find("sfm-wb"))
    results["BLOCKA var-bw"] = kpi_of(find("sfm-wb-blocka"))

finally:
    open(NB, "wb").write(nb_template)
    open(PARAMS, "wb").write(params_template)
    print("baseline notebook/params restored", flush=True)

rows = []
for scheme, d in results.items():
    for m, v in d.items():
        rows.append({"scheme": scheme, "method": m, **{k: round(x, 3) for k, x in v.items()}})
df = pd.DataFrame(rows)
df.to_excel(os.path.join(os.path.dirname(PROJ), "ratepool_vs_others_results.xlsx"), index=False)

print("\n===== 4-way: rate-proportional vs alpha vs type-pool vs BLOCKA (SFM+Weibull) =====")
print(df.pivot_table(index="method", columns="scheme",
                     values=["weighted_mos", "qlen_Mb"]).round(3).to_string())
print("\nsaved ratepool_vs_others_results.xlsx")

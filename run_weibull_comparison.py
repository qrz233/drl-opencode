# -*- coding: utf-8 -*-
'''SFM mobility + Weibull arrival (vs bounded Pareto): run the full simulation
   and compare KPIs against SFM+Pareto and the GM+Pareto baseline.

   Controlled design:
     - unchanged: SFM mobility (reuses bound=250-sfm.xlsx), ON/OFF schedule and
       service-type assignment (reuses baseline traffic state/type xlsx)
     - changed:   ON-slot file-size marginal: bounded Pareto -> Weibull
                  (WEIBULL_K/SCALE matched to Pareto mean & CV, same 3*mu cap)

   Patches main.ipynb cell 4 (SFM mobility), cell 6 (Weibull arrival gen from
   the existing schedule), cell 8 (load SFM locations + Weibull arrival), and
   params_sys.py folder tag ("-sfm-wb"). Restores everything afterwards.
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

    for idx in (4, 6, 8):
        compile("".join(nb["cells"][idx]["source"]), f"<cell {idx}>", "exec")

    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("notebook patched (SFM mobility + Weibull arrival)", flush=True)

    # ---------------- patch output folder tag ----------------
    ps = params_template.decode("utf-8")
    old_t = "-pureLoS-K20-s1.1, n_slots={n_slots}, "
    assert old_t in ps, "folder tag not found"
    ps = ps.replace(old_t, "-pureLoS-K20-s1.1-sfm-wb, n_slots={n_slots}, ")
    open(PARAMS, "w", encoding="utf-8").write(ps)
    print("params patched (folder tag -sfm-wb)", flush=True)

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
    wb_dir = glob.glob(os.path.join(PROJ, r"simulation\sim-output\*K20-s1.1-sfm-wb, n_slots=*"))[0]
    sfm_dir = glob.glob(os.path.join(PROJ, r"simulation\sim-output\*K20-s1.1-sfm, n_slots=*"))[0]
    gm_dir = glob.glob(os.path.join(PROJ, r"simulation\sim-output\*K20-s1.1, n_slots=*"))[0]
    results["SFM+Weibull"] = kpi_of(wb_dir)
    results["SFM+Pareto"] = kpi_of(sfm_dir)
    results["GM+Pareto"] = kpi_of(gm_dir)

    # arrival-distribution sanity check
    tag = f"nusers=100, nslots=501, ntype=3, lambda_Mb=2.4, uON=501, uOFF=1"
    def _load(fname):
        return np.transpose(pd.read_excel(os.path.join(PROJ, "simulation", "sim-input", fname),
                                          index_col=0).to_numpy())
    st = _load(f"traffic state, {tag}.xlsx").astype(bool)
    for label, fname in [("Pareto", f"arrival traffic, {tag}.xlsx"),
                         ("Weibull", f"arrival traffic, dist=weibull, {tag}.xlsx")]:
        v = _load(fname)[st]
        print(f"arrival[{label}]: mean={v.mean():.3f} cv={v.std()/v.mean():.3f} "
              f"p99={np.percentile(v,99):.3f} max={v.max():.3f}", flush=True)

finally:
    open(NB, "wb").write(nb_template)
    open(PARAMS, "wb").write(params_template)
    print("baseline notebook/params restored", flush=True)

# ---------------- summary ----------------
rows = []
for traffic, d in results.items():
    for m, v in d.items():
        rows.append({"traffic": traffic, "method": m, **{k: round(x, 3) for k, x in v.items()}})
df = pd.DataFrame(rows)
df.to_excel(os.path.join(os.path.dirname(PROJ), "weibull_vs_pareto_results.xlsx"), index=False)

print("\n===== Weibull vs Pareto (SFM) vs baseline (GM) =====")
print(df.pivot_table(index="method", columns="traffic",
                     values=["weighted_mos", "qlen_Mb"]).round(3).to_string())
print("\nsaved weibull_vs_pareto_results.xlsx")

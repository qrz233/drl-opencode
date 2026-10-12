# -*- coding: utf-8 -*-
'''SFM (Social Force Model) user-mobility vs Gauss-Markov: run the full
   simulation with SFM trajectories and compare KPIs against the GM baseline.

   - patches main.ipynb cell 4 (generate_mobility_GM -> generate_mobility_SFM,
     input filename tag +"-sfm") and cell 8 (load the SFM xlsx files)
   - patches params_sys.py output folder tag (+ "-sfm")
   - runs the notebook via nbconvert
   - collects weighted-MOS / per-type KPIs for Stationary/Centroid/Proposed,
     compares against the existing GM result folder
   - restores pristine notebook/params afterwards
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

    # cell 4: filename tags + replace GM generation loop with SFM
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

    # cell 8: load the SFM location files
    src8 = "".join(nb["cells"][8]["source"])
    assert src8.count("bound={boundary}.xlsx") == 2, src8.count("bound={boundary}.xlsx")
    src8 = src8.replace("bound={boundary}.xlsx", "bound={boundary}-sfm.xlsx")
    nb["cells"][8]["source"] = src8.splitlines(keepends=True)

    # syntax-check the two patched cells
    for idx in (4, 8):
        compile("".join(nb["cells"][idx]["source"]), f"<cell {idx}>", "exec")

    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("notebook patched (SFM mobility)", flush=True)

    # ---------------- patch output folder tag ----------------
    ps = params_template.decode("utf-8")
    old_t = "-pureLoS-K20-s1.1, n_slots={n_slots}, "
    assert old_t in ps, "folder tag not found"
    ps = ps.replace(old_t, "-pureLoS-K20-s1.1-sfm, n_slots={n_slots}, ")
    open(PARAMS, "w", encoding="utf-8").write(ps)
    print("params patched (folder tag -sfm)", flush=True)

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
    sfm_dir = glob.glob(os.path.join(PROJ, r"simulation\sim-output\*K20-s1.1-sfm, n_slots=*"))[0]
    gm_dir = glob.glob(os.path.join(PROJ, r"simulation\sim-output\*K20-s1.1, n_slots=*"))[0]
    results["SFM"] = kpi_of(sfm_dir)
    results["GM"] = kpi_of(gm_dir)

finally:
    open(NB, "wb").write(nb_template)
    open(PARAMS, "wb").write(params_template)
    print("baseline notebook/params restored", flush=True)

# ---------------- summary ----------------
rows = []
for mob, d in results.items():
    for m, v in d.items():
        rows.append({"mobility": mob, "method": m, **{k: round(x, 3) for k, x in v.items()}})
df = pd.DataFrame(rows)
df.to_excel(os.path.join(os.path.dirname(PROJ), "sfm_vs_gm_results.xlsx"), index=False)

print("\n===== SFM vs Gauss-Markov =====")
print(df.pivot_table(index="method", columns="mobility",
                     values=["weighted_mos", "qlen_Mb"]).round(3).to_string())
print("\nsaved sfm_vs_gm_results.xlsx")

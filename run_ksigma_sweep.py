# -*- coding: utf-8 -*-
'''K / sigma sweep for the hybrid Proposed method (single episode).
   For each (K, sigma) config: patch cell 26 + output folder tag, run the full
   notebook via nbconvert, then collect the weighted MOS and per-type KPIs.
   Baseline files are restored afterwards; results also saved to ksigma_results_round3.xlsx.
'''
import json, os, shutil, subprocess, sys, glob
import pandas as pd

PROJ = r"D:\Learn\work\drl-zcode\uav-bs-placement-drl-ksigma-sweep"
NB = PROJ + r"\main.ipynb"
PARAMS = PROJ + r"\params_sys.py"
PY = sys.executable
W = {0: 0.35, 1: 0.30, 2: 0.35}          # MOS weights Web/Video/Messaging

CONFIGS = [(20, 1.1), (20, 1.3)]

# pristine templates
nb_template = open(NB, "rb").read()
params_template = open(PARAMS, "rb").read()

results = []
try:
    for K, s in CONFIGS:
        tag = f"K{K}-s{s}"
        print(f"\n========== config {tag} ==========", flush=True)

        # patch notebook cell 26
        nb = json.loads(nb_template.decode("utf-8"))
        src26 = "".join(nb["cells"][26]["source"])
        old_c = "n_candidates=10, eval_sigma=std_var_vxvyvz_explore"
        assert old_c in src26, "candidate line not found"
        src26 = src26.replace(old_c, f"n_candidates={K}, eval_sigma={s}")
        nb["cells"][26]["source"] = src26.splitlines(keepends=True)
        json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

        # patch params_sys.py folder tag
        ps = params_template.decode("utf-8")
        old_t = "-pureLoS, n_slots={n_slots}, "
        assert old_t in ps, "folder tag not found"
        ps = ps.replace(old_t, f"-pureLoS-{tag}, n_slots={{n_slots}}, ")
        open(PARAMS, "w", encoding="utf-8").write(ps)

        # run the full notebook
        env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
        r = subprocess.run([PY, "-m", "nbconvert", "--to", "notebook", "--execute", "main.ipynb",
                            "--output", "main_executed.ipynb",
                            "--ExecutePreprocessor.timeout=14400",
                            "--ExecutePreprocessor.kernel_name=uav-drl"],
                           cwd=PROJ, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{tag}: NOTEBOOK FAILED\n{r.stderr[-1500:]}")
            results.append({"config": tag, "weighted_mos": None, "error": True})
            continue

        # collect KPIs
        base = glob.glob(PROJ + r"\simulation\sim-output\*-%s, n_slots=*" % tag)[0]
        rows = {}
        for m in ["Stationary", "Centroid", "Proposed"]:
            df = pd.read_excel(os.path.join(base, f"{m}-kpis-over-time.xlsx"))
            wmos = sum(W[k] * df[f"mos-avg-{n}"].mean() for k, n in enumerate(["Web", "Video", "Messaging"]))
            rows[m] = {"weighted_mos": wmos,
                       "web": df["mos-avg-Web"].mean(), "video": df["mos-avg-Video"].mean(),
                       "msg": df["mos-avg-Messaging"].mean(), "qlen": df["qlen-Mb-avg (active)"].mean()}
        results.append({"config": tag, **{f"{m}_{k}": v for m, d in rows.items() for k, v in d.items()}, "error": False})
        print(f"{tag}: Proposed weighted MOS = {rows['Proposed']['weighted_mos']:.3f} "
              f"(Centroid {rows['Centroid']['weighted_mos']:.3f}), qlen = {rows['Proposed']['qlen']:.1f}", flush=True)
finally:
    # restore pristine baseline files (K=10, sigma=std_var_vxvyvz_explore)
    open(NB, "wb").write(nb_template)
    open(PARAMS, "wb").write(params_template)
    print("\nbaseline notebook/params restored")

df = pd.DataFrame(results)
df.to_excel(PROJ + r"\ksigma_results_round3.xlsx", index=False)
print("\n===== SWEEP SUMMARY =====")
print(df[["config", "Proposed_weighted_mos", "Proposed_qlen", "Centroid_weighted_mos", "error"]].to_string(index=False))

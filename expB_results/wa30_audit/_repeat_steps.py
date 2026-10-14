# -*- coding: utf-8 -*-
"""Disambiguate task-C residual: from K5-best, repeat the SAME K5 directions
for a 2nd step vs take one arbitrary grid step. If same-direction repeat also
recovers most residual -> step-budget issue, not direction deficiency.
"""
import os, sys, time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
import numpy as np
import pandas as pd

OUT = r'D:\Learn\work\drl-opencode\expB_results\wa30_audit'
src = open(os.path.join(OUT, '_grid_residual.py'), encoding='utf-8').read()
cut = src.index('rows = []')
ns = {}
exec(compile(src[:cut], '_grid_prefix', 'exec'), ns)
solve_full, gammas, evalJ = ns['solve_full'], ns['gammas'], ns['evalJ']
beta_dirs, unit, SNAP = ns['beta_dirs'], ns['unit'], ns['SNAP']
joint_eval, DIRS, RHOS = ns['joint_eval'], ns['DIRS'], ns['RHOS']
move_single = ns['move_single']
SPD_XY, TAU, BOUNDARY = ns['SPD_XY'], ns['TAU'], ns['BOUNDARY']
NU_EPS, C_NU = ns['NU_EPS'], ns['C_NU']

TARGET = [13, 14, 18, 19, 23, 25, 28, 29, 32]
rows = []
t0 = time.time()
for sn in SNAP:
    if sn['si'] not in TARGET:
        continue
    beta, types, cids = sn['beta'], sn['types'], sn['cids']
    locs0, air0, E0 = sn['locs0'], sn['air0'], sn['E0']
    ga0, gb0, _ = gammas(sn['ux'], sn['uy'], cids, locs0, air0)
    base = solve_full(ga0, gb0, sn['demand'], cids, beta)
    nu = np.maximum(base['nu_bh'], 0.0)
    e_u_beta, e_a_beta = beta_dirs(beta, types, cids, locs0, air0, sn['ux'], sn['uy'])
    e_bh = np.array([unit(air0[:2] - locs0[u, :2]) for u in range(4)])
    w_d = np.where(nu > NU_EPS, nu / (C_NU + nu), 0.0)
    e_u_dual = np.array([unit((1 - w_d[u]) * e_u_beta[u] + w_d[u] * e_bh[u])
                         for u in range(4)])
    J0, _ = evalJ(locs0.copy(), air0, sn, E0, np.zeros((4, 2)))

    cands = (('b25', .25, e_u_beta), ('b50', .5, e_u_beta),
             ('du25', .25, e_u_dual), ('du50', .5, e_u_dual))
    best_J, best_tag, best_rho, best_eu = J0, 'stay', 0.0, e_u_beta
    evals = {}
    for tag, rho, eu in cands:
        J, _ = joint_eval(locs0, air0, sn, E0, eu, rho)
        evals[tag] = J
        if J < best_J - 1e-12:
            best_J, best_tag, best_rho, best_eu = J, tag, rho, eu

    # reconstruct K5 best locs
    locs_k5 = locs0.copy()
    if best_tag != 'stay':
        vxy = SPD_XY * best_rho * best_eu
        for u in range(4):
            locs_k5[u, 0] = np.clip(locs0[u, 0] + vxy[u, 0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
            locs_k5[u, 1] = np.clip(locs0[u, 1] + vxy[u, 1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    J_k5, _ = evalJ(locs_k5.copy(), air0, sn, E0,
                    SPD_XY * best_rho * best_eu if best_tag != 'stay' else np.zeros((4, 2)))

    # R1: repeat the same candidate direction from K5 best (same rho)
    J_same = np.inf
    if best_tag != 'stay':
        J_same, _ = joint_eval(locs_k5, air0, sn, E0, best_eu, best_rho)
    # R2: best of all 4 K5 candidates applied again from K5 best
    J_repeat2 = np.inf
    for tag, rho, eu in cands:
        J, _ = joint_eval(locs_k5, air0, sn, E0, eu, rho)
        J_repeat2 = min(J_repeat2, J)
    # R3: best single-UAV arbitrary step from K5 best
    J_grid1 = np.inf
    for u in range(4):
        for d in DIRS:
            for rho in RHOS:
                loc2, v = move_single(locs_k5, u, d, rho)
                J, _ = evalJ(loc2, air0, sn, E0, v)
                J_grid1 = min(J_grid1, J)

    rows.append(dict(
        si=sn['si'], dJ_k5=J0 - best_J, k5_tag=best_tag,
        same_dir_gain=(best_J - J_same) if np.isfinite(J_same) else 0.0,
        repeat2_gain=best_J - J_repeat2,
        grid1_gain=best_J - J_grid1,
    ))
    print(f"si={sn['si']:>3} tag={best_tag:<5} dJ_k5={J0-best_J:6.3f} "
          f"same_dir_gain={rows[-1]['same_dir_gain']:.3f} "
          f"repeat2_gain={rows[-1]['repeat2_gain']:.3f} "
          f"grid1_gain={rows[-1]['grid1_gain']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'taskC_repeat_steps.csv'), index=False)
print('\n', df.describe().to_string(), flush=True)
print(f'elapsed {time.time()-t0:.0f}s', flush=True)

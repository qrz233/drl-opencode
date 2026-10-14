# -*- coding: utf-8 -*-
"""Focused C-check on the 9 access-binding / simultaneous snapshots at W_A=30.

Budget-matched comparisons from STAY and from K5-best:
  S1  best single-UAV single-step move (1 UAV, 16 dir x 2 rho)
      -> direction coverage test under step budget <= K5's joint step
  S2  first grid-improving move direction + angle vs beta/bh directions
  S3  full 2-pass residual (already in taskC_grid_residual.csv) split by pass
"""
import os, sys, time, json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
import numpy as np
import pandas as pd

OUT = r'D:\Learn\work\drl-opencode\expB_results\wa30_audit'
sys.path.insert(0, OUT)
sys.path.insert(0, r'D:\Learn\work\drl-opencode')
sys.path.insert(0, r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11')

# reuse grid script machinery via exec of its top part (up to DIRS)
import importlib.util
spec = importlib.util.spec_from_file_location('gr', os.path.join(OUT, '_grid_residual.py'))
# can't import (executes main). Re-declare minimal functions by importing module pieces:
# Instead: exec only definitions. Simplest: copy needed helpers via runpy on a trimmed copy.
# We take the pragmatic path: define thin wrappers around solve_full etc. by
# exec'ing the file with __name__ != '__main__' guarded main... file has no guard.
# So: read source, cut at "rows = []" marker, exec that prefix.
src = open(os.path.join(OUT, '_grid_residual.py'), encoding='utf-8').read()
cut = src.index('rows = []')
ns = {}
exec(compile(src[:cut], '_grid_prefix', 'exec'), ns)
solve_full = ns['solve_full']
gammas = ns['gammas']
evalJ = ns['evalJ']
beta_dirs = ns['beta_dirs']
unit = ns['unit']
SNAP = ns['SNAP']
move_single = ns['move_single']
joint_eval = ns['joint_eval']
DIRS = ns['DIRS']
RHOS = ns['RHOS']
SPD_XY, TAU, BOUNDARY = ns['SPD_XY'], ns['TAU'], ns['BOUNDARY']
EPS_ACT, NU_EPS, C_NU = ns['EPS_ACT'], ns['NU_EPS'], ns['C_NU']

TARGET = [13, 14, 18, 19, 23, 25, 28, 29, 32]  # acc-binding / simultaneous
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
    # K5 best joint
    best_J, best_tag = J0, 'stay'
    for tag, rho, eu in (('b25', .25, e_u_beta), ('b50', .5, e_u_beta),
                         ('du25', .25, e_u_dual), ('du50', .5, e_u_dual)):
        J, _ = joint_eval(locs0, air0, sn, E0, eu, rho)
        if J < best_J - 1e-12:
            best_J, best_tag = J, tag

    # S1: best single-UAV single-step from STAY (budget: 1 UAV x step <= 15m)
    s1_J, s1_info = np.inf, None
    for u in range(4):
        for d in DIRS:
            for rho in RHOS:
                loc2, v = move_single(locs0, u, d, rho)
                J, _ = evalJ(loc2, air0, sn, E0, v)
                if J < s1_J - 1e-12:
                    s1_J, s1_info = J, (u, d, rho)
    # angle of S1 direction vs beta/bh of that UAV
    u_star, d_star, rho_star = s1_info
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(
        float(np.dot(unit(a), unit(b))), -1, 1)))) if (np.linalg.norm(a) > 1e-9
                                                       and np.linalg.norm(b) > 1e-9) else np.nan
    ang_beta = ang(d_star, e_u_beta[u_star])
    ang_bh = ang(d_star, e_bh[u_star])

    # S1b: best single-UAV single-step from K5-best position
    # reconstruct K5 locs
    K5_LOC = None
    vxy = SPD_XY * dict(b25=.25, b50=.5, du25=.25, du50=.5, stay=0.0)[best_tag] * \
        (e_u_beta if best_tag in ('b25', 'b50') else (e_u_dual if best_tag.startswith('du') else e_u_beta))
    locs_k5 = locs0.copy()
    if best_tag != 'stay':
        for u in range(4):
            locs_k5[u, 0] = np.clip(locs0[u, 0] + vxy[u, 0] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
            locs_k5[u, 1] = np.clip(locs0[u, 1] + vxy[u, 1] * TAU, -BOUNDARY + 1, BOUNDARY - 1)
    s1b_J, s1b_info = np.inf, None
    for u in range(4):
        for d in DIRS:
            for rho in RHOS:
                loc2, v = move_single(locs_k5, u, d, rho)
                J, _ = evalJ(loc2, air0, sn, E0, v)
                if J < s1b_J - 1e-12:
                    s1b_J, s1b_info = J, (u, d, rho)

    rows.append(dict(
        si=sn['si'], J0=J0, k5_best=best_J, k5_tag=best_tag,
        dJ_k5=J0 - best_J,
        s1_best=s1_J, dJ_s1=J0 - s1_J, s1_u=int(s1_star[0]) if (s1_star := s1_info) else -1,
        s1_rho=float(s1_info[2]), s1_ang_beta=ang_beta, s1_ang_bh=ang_bh,
        s1_beats_k5=bool(s1_J < best_J - 1e-9),
        s1b_gain=(best_J - s1b_J), s1b_u=int(s1b_info[0]), s1b_rho=float(s1b_info[2]),
        n_mu=int((base['lam_acc'] > EPS_ACT)[
            (sn['demand'] > 0) & (cids >= 0) & (beta > 0)].sum()),
    ))
    print(f"si={sn['si']:>3} dJ_k5={J0-best_J:7.3f} dJ_s1={J0-s1_J:7.3f} "
          f"s1_beats_k5={rows[-1]['s1_beats_k5']} ang_beta={ang_beta:5.1f} "
          f"ang_bh={ang_bh:5.1f} s1b_gain={rows[-1]['s1b_gain']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'taskC_singlestep.csv'), index=False)
print('\nsummary:', flush=True)
print(df[['dJ_k5', 'dJ_s1', 's1_beats_k5', 's1_ang_beta', 's1_ang_bh',
          's1b_gain']].describe().to_string(), flush=True)
print('s1 beats k5 rate:', df['s1_beats_k5'].mean(), flush=True)
print(f'elapsed {time.time()-t0:.0f}s', flush=True)

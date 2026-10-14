# -*- coding: utf-8 -*-
"""Task 1 audit: constraints & statistical scope of repeat-step / grid residual.

Part 1: scope of residual mean=2.31 and recovered gain=5.02.
Part 2: constraint audit of the second-step candidates (9 coexistence snaps),
        under both interpretations: same-slot double execution vs next-slot
        continuation probe. Invalid candidates excluded from valid gains.
"""
import os, sys, json
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
joint_eval = ns['joint_eval']
SPD_XY, TAU, BOUNDARY = ns['SPD_XY'], ns['TAU'], ns['BOUNDARY']
NU_EPS, C_NU = ns['NU_EPS'], ns['C_NU']
z_min, z_max = ns['z_min'], ns['z_max']

from utils_sys import cal_uav_propulsion_energy

PW_MAX = 683.0
TARGET = [13, 14, 18, 19, 23, 25, 28, 29, 32]

# ---------- Part 1: statistical scope ----------
g = pd.read_csv(os.path.join(OUT, 'taskC_grid_residual.csv'))
r = pd.read_csv(os.path.join(OUT, 'taskC_repeat_steps.csv'))
scope = dict(
    residual_2_31=dict(
        source='taskC_grid_residual.csv::residual',
        quantity=('J(K5 best) - J(after 2-pass greedy single-UAV grid descent, '
                  'up to 8 single-UAV steps from K5 best position)'),
        sample_n=int(len(g)),
        sample_set='all 40 stratified snapshots',
        mean=round(float(g['residual'].mean()), 4)),
    residual_on_9=dict(
        mean=round(float(g[g['si'].isin(TARGET)]['residual'].mean()), 4),
        n=int(g['si'].isin(TARGET).sum())),
    residual_on_31=dict(
        mean=round(float(g[~g['si'].isin(TARGET)]['residual'].mean()), 4),
        n=int((~g['si'].isin(TARGET)).sum())),
    gain_5_02=dict(
        source='taskC_repeat_steps.csv::same_dir_gain',
        quantity=('J(K5 best) - J(K5 best + ONE extra same-direction joint step '
                  'with the SAME rho)'),
        sample_n=int(len(r)),
        sample_set='9 coexistence snapshots only (not the 40)',
        mean=round(float(r['same_dir_gain'].mean()), 4)),
)
print('=== Part 1: scope ===')
print(json.dumps(scope, indent=1, ensure_ascii=False))

# ---------- Part 2: constraint audit of second-step candidates ----------
rows = []
for sn in SNAP:
    if sn['si'] not in TARGET:
        continue
    beta, types, cids = sn['beta'], sn['types'], sn['cids']
    locs0, air0, E0 = sn['locs0'], sn['air0'], sn['E0']
    z0 = locs0[:, 2].copy()
    ga0, gb0, _ = gammas(sn['ux'], sn['uy'], cids, locs0, air0)
    base = solve_full(ga0, gb0, sn['demand'], cids, beta)
    nu = np.maximum(base['nu_bh'], 0.0)
    e_u_beta, _ = beta_dirs(beta, types, cids, locs0, air0, sn['ux'], sn['uy'])
    e_bh = np.array([unit(air0[:2] - locs0[u, :2]) for u in range(4)])
    w_d = np.where(nu > NU_EPS, nu / (C_NU + nu), 0.0)
    e_u_dual = np.array([unit((1 - w_d[u]) * e_u_beta[u] + w_d[u] * e_bh[u])
                         for u in range(4)])
    cands = (('b25', .25, e_u_beta), ('b50', .5, e_u_beta),
             ('du25', .25, e_u_dual), ('du50', .5, e_u_dual))
    J0, _ = evalJ(locs0.copy(), air0, sn, E0, np.zeros((4, 2)))
    best_J, best_tag, best_rho, best_eu = J0, 'stay', 0.0, e_u_beta
    for tag, rho, eu in cands:
        J, _ = joint_eval(locs0, air0, sn, E0, eu, rho)
        if J < best_J - 1e-12:
            best_J, best_tag, best_rho, best_eu = J, tag, rho, eu
    if best_tag == 'stay':
        continue

    v1 = SPD_XY * best_rho * best_eu
    locs1 = locs0.copy()
    locs1[:, 0] = np.clip(locs0[:, 0] + v1[:, 0] * TAU, -BOUNDARY, BOUNDARY)
    locs1[:, 1] = np.clip(locs0[:, 1] + v1[:, 1] * TAU, -BOUNDARY, BOUNDARY)
    locs2 = locs1.copy()
    locs2[:, 0] = np.clip(locs1[:, 0] + v1[:, 0] * TAU, -BOUNDARY, BOUNDARY)
    locs2[:, 1] = np.clip(locs1[:, 1] + v1[:, 1] * TAU, -BOUNDARY, BOUNDARY)

    d1 = locs1 - locs0
    d2 = locs2 - locs1
    net2 = locs2 - locs0
    clip1 = bool(np.any(np.abs(d1[:, :2] - v1[:, :2] * TAU) > 1e-9))
    clip2 = bool(np.any(np.abs(d2[:, :2] - v1[:, :2] * TAU) > 1e-9))
    v_cmd = float(np.max(np.linalg.norm(v1[:, :2], axis=1)))
    v_net_step = float(np.max(np.linalg.norm(d2[:, :2], axis=1) / TAU))
    v_net_sameslot = float(np.max(np.linalg.norm(net2[:, :2], axis=1) / TAU))
    P_cmd = float(max(cal_uav_propulsion_energy(
        V=np.linalg.norm(v1[u, :2]), Vz=0.0)[0] for u in range(4)))
    P_net_sameslot = float(max(cal_uav_propulsion_energy(
        V=np.linalg.norm(net2[u, :2]) / TAU, Vz=0.0)[0] for u in range(4)))

    ok_speed_step = v_cmd <= 30.0 + 1e-9
    ok_p_step = P_cmd <= PW_MAX + 1e-9
    ok_speed_same = v_net_sameslot <= 30.0 + 1e-9
    ok_p_same = P_net_sameslot <= PW_MAX + 1e-9
    ok_region = bool(np.all(np.abs(locs2[:, 0]) <= BOUNDARY + 1e-9) and
                     np.all(np.abs(locs2[:, 1]) <= BOUNDARY + 1e-9))
    ok_alt = bool(np.all(locs2[:, 2] >= z_min - 1e-9) and
                  np.all(locs2[:, 2] <= z_max + 1e-9))

    rows.append(dict(
        si=sn['si'], t=sn['t'], tag=best_tag, rho=best_rho,
        dJ_k5=J0 - best_J,
        max_disp_step1=round(float(np.max(np.linalg.norm(d1[:, :2], axis=1))), 3),
        max_disp_total_2step=round(float(np.max(np.linalg.norm(net2[:, :2], axis=1))), 3),
        v_cmd=round(v_cmd, 3), v_net_step=round(v_net_step, 3),
        v_net_sameslot=round(v_net_sameslot, 3),
        P_cmd=round(P_cmd, 2), P_net_sameslot=round(P_net_sameslot, 2),
        clip_step1=int(clip1), clip_step2=int(clip2),
        z_unchanged=int(np.allclose(locs2[:, 2], z0)),
        ok_speed_step=int(ok_speed_step), ok_p_step=int(ok_p_step),
        ok_speed_sameslot=int(ok_speed_same), ok_p_sameslot=int(ok_p_same),
        ok_region=int(ok_region), ok_alt=int(ok_alt),
        valid_probe=int(ok_speed_step and ok_p_step and ok_region and ok_alt),
        valid_sameslot=int(ok_speed_same and ok_p_same and ok_region and ok_alt),
    ))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'task1_repeat_constraint_audit.csv'), index=False)
print('\n=== Part 2: constraint audit (9 repeat-step candidates) ===')
print(df.to_string(index=False))
print('\nvalid as next-slot probe: %d/9   valid as same-slot double: %d/9'
      % (df['valid_probe'].sum(), df['valid_sameslot'].sum()))
print('clipping occurred: step1=%d step2=%d' % (df['clip_step1'].sum(),
                                                df['clip_step2'].sum()))
print('z unchanged: %d/9   region ok: %d/9   alt ok: %d/9'
      % (df['z_unchanged'].sum(), df['ok_region'].sum(), df['ok_alt'].sum()))
print('max v_cmd=%.3f (lim 30)  max P_cmd=%.2f (lim %g)'
      % (df['v_cmd'].max(), df['P_cmd'].max(), PW_MAX))
print('max v_net_sameslot=%.3f  max P_net_sameslot=%.2f'
      % (df['v_net_sameslot'].max(), df['P_net_sameslot'].max()))

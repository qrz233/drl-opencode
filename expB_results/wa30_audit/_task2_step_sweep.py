# -*- coding: utf-8 -*-
"""Task 2: fixed-direction extended step-length sweep at W_A=30, 40 snapshots.

rho in {0, 0.1, 0.25, 0.5, 0.75, 1.0} x directions {beta, binding}.
All candidates built from the same slot-initial position.

Hard constraints enforced/checked per candidate:
  - UAV horizontal speed: net displacement / tau <= 30 m/s
  - UAV vertical: z unchanged -> vz=0 <= 5 (always ok)
  - altitude in [z_min, z_max]
  - region |x|,|y| <= boundary (production clip)
  - airship: fixed (no motion) -> airship limits trivially satisfied
  - instantaneous propulsion power from ACTUAL NET displacement <= PW_MAX=683
  - Block A full constraints (solver status optimal)
Invalid candidates excluded from valid gains.

Outputs -> expB_results\\wa30_audit\\task2_step_sweep.csv (+summary print)
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
SPD_XY, TAU, BOUNDARY = ns['SPD_XY'], ns['TAU'], ns['BOUNDARY']
NU_EPS, BIND_EPS = ns['NU_EPS'], ns['BIND_EPS']
z_min, z_max, PSI_E = ns['z_min'], ns['z_max'], ns['PSI_E']

from utils_sys import cal_uav_propulsion_energy

PW_MAX = 683.0
V_MAX = 30.0
RHOS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

d2 = pd.read_csv(os.path.join(OUT, 'task2_wa30_methods.csv'))
pick = {int(s): row for s, row in
        zip(d2['si'], d2.to_dict('records'))}


def eval_cand(locs, air, sn, E0, beta):
    """Move all UAVs to locs (already clipped); power from actual net disp."""
    net = locs - sn['locs0']
    v_net = np.linalg.norm(net[:, :2], axis=1) / TAU
    vz = np.abs(net[:, 2]) / TAU
    P = np.array([float(cal_uav_propulsion_energy(V=v_net[u], Vz=vz[u])[0])
                  for u in range(4)])
    ga, gb, g = gammas(sn['ux'], sn['uy'], sn['cids'], locs, air)
    t0 = time.time()
    res = solve_full(ga, gb, sn['demand'], sn['cids'], beta)
    dt = time.time() - t0
    J = PSI_E * float(E0 @ P) - float(beta @ res['D_Mb'])
    ok_status = res['status'] in ('optimal', 'optimal_inaccurate')
    ok_speed = bool(np.all(v_net <= V_MAX + 1e-9))
    ok_power = bool(np.all(P <= PW_MAX + 1e-9))
    ok_region = bool(np.all(np.abs(locs[:, 0]) <= BOUNDARY + 1e-9) and
                     np.all(np.abs(locs[:, 1]) <= BOUNDARY + 1e-9))
    ok_alt = bool(np.all(locs[:, 2] >= z_min - 1e-9) and
                  np.all(locs[:, 2] <= z_max + 1e-9))
    ok = ok_status and ok_speed and ok_power and ok_region and ok_alt
    return dict(J=J, P=P, res=res, dt=dt, ok=ok, ok_status=ok_status,
                ok_speed=ok_speed, ok_power=ok_power, ok_region=ok_region,
                ok_alt=ok_alt, v_net=v_net, D=float(beta @ res['D_Mb']),
                Eterm=PSI_E * float(E0 @ P))


rows = []
t_start = time.time()
for sn in SNAP:
    si = sn['si']
    if si not in pick:
        continue
    beta, types, cids = sn['beta'], sn['types'], sn['cids']
    locs0, air0, E0 = sn['locs0'], sn['air0'], sn['E0']
    ga0, gb0, _ = gammas(sn['ux'], sn['uy'], cids, locs0, air0)
    base = solve_full(ga0, gb0, sn['demand'], cids, beta)
    nu = np.maximum(base['nu_bh'], 0.0)
    e_u_beta, _ = beta_dirs(beta, types, cids, locs0, air0, sn['ux'], sn['uy'])
    e_bh = np.array([unit(air0[:2] - locs0[u, :2]) for u in range(4)])
    w_b = (nu > BIND_EPS).astype(float)
    e_u_bind = np.array([unit((1 - w_b[u]) * e_u_beta[u] + w_b[u] * e_bh[u])
                         for u in range(4)])

    for dname, e_u in (('beta', e_u_beta), ('bind', e_u_bind)):
        for rho in RHOS:
            locs = locs0.copy()
            if rho > 0:
                disp = SPD_XY * rho * e_u * TAU
                locs[:, 0] = np.clip(locs0[:, 0] + disp[:, 0], -BOUNDARY, BOUNDARY)
                locs[:, 1] = np.clip(locs0[:, 1] + disp[:, 1], -BOUNDARY, BOUNDARY)
            r = eval_cand(locs, air0, sn, E0, beta)
            rows.append(dict(
                si=si, t=sn['t'], qlen=sn['qlen'], dir=dname, rho=rho,
                J=r['J'], D=r['D'], Eterm=r['Eterm'],
                P_max=float(r['P'].max()), P_sum=float(r['P'].sum()),
                v_max=float(r['v_net'].max()),
                valid=int(r['ok']), ok_status=int(r['ok_status']),
                ok_speed=int(r['ok_speed']), ok_power=int(r['ok_power']),
                ok_region=int(r['ok_region']), ok_alt=int(r['ok_alt']),
                solve_s=round(r['dt'], 3),
                status=r['res']['status'],
                n_mu=int((r['res']['lam_acc'] > 1e-6)[
                    (sn['demand'] > 0) & (cids >= 0) & (beta > 0)].sum()),
                n_nu=int((r['res']['nu_bh'] > 1e-6).sum()),
                pool_acc=int(r['res']['nu_acc_pool'] > 1e-6),
                pool_bh=int(r['res']['nu_bh_pool'] > 1e-6),
            ))
    if len(rows) % 240 < 12:
        print(f"  {si} done, elapsed {time.time()-t_start:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'task2_step_sweep.csv'), index=False)
print(f'\nswept {df["si"].nunique()} snaps x 2 dirs x {len(RHOS)} rho = '
      f'{len(df)} candidates in {time.time()-t_start:.0f}s')
print('valid rate: %.3f' % df['valid'].mean())
print('invalid breakdown: status=%d speed=%d power=%d region=%d alt=%d'
      % ((~df['ok_status'].astype(bool)).sum(),
         (~df['ok_speed'].astype(bool)).sum(),
         (~df['ok_power'].astype(bool)).sum(),
         (~df['ok_region'].astype(bool)).sum(),
         (~df['ok_alt'].astype(bool)).sum()))
print('solve_s: mean=%.3f p95=%.3f max=%.3f  statuses: %s'
      % (df['solve_s'].mean(), df['solve_s'].quantile(.95),
         df['solve_s'].max(), df['status'].value_counts().to_dict()))

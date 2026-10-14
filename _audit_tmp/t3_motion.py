# -*- coding: utf-8 -*-
"""Audit test 3: motion / speed / power-cap compliance for both controllers."""
import os, sys
import numpy as np
PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
from params_sys import *
from utils_sys import cal_uav_propulsion_energy
from mobility_model import adjust_location, adjust_altitude

print(f'PW_MAX={PW_MAX}  uav_speedxy_max={uav_speedxy_max}  uav_speedz_max={uav_speedz_max}')
print(f'airship fixed at ({airship_x},{airship_y},{airship_z})\n')

# ---------- relay_hybrid_controller.motion ----------
from relay_hybrid_controller import HybridCandidateController
hc = HybridCandidateController(
    n_users=n_users, n_uavs=n_uavs, boundary=boundary, grid_size=grid_size,
    queue_norm=queue_norm_Mb, V_lyap=lyapunov_param, tau=slot_len,
    ss_min_types=ss_min_types, ss_max_types=ss_max_types,
    mos_weights=mos_type_weights, type_rates=ON_data_arrival_mean_Mb, d0=d0,
    sigma_decay_slots=n_slots, seed=0, n_candidates=20, eval_sigma=1.1)
hc._mp = (float(uav_speedxy_max), float(uav_speedz_max), z_min, z_max, float(boundary))

loc0 = np.array([0.0, 0.0, 150.0])
print('--- relay_hybrid_controller.motion ---')
for name, c in [('|c_xy|=1 (unit)', np.array([1.0, 0.0, 0.0])),
                ('|c_xy|=1.414 (corner)', np.array([1.0, 1.0, 0.0])),
                ('|c_xy|=2 (overshoot)', np.array([2.0, 0.0, 0.0])),
                ('c_z=1 (climb)', np.array([0.0, 0.0, 1.0]))]:
    p, v = hc.motion(c, loc0)
    vxy = np.linalg.norm(v[:2]); P = cal_uav_propulsion_energy(V=vxy, Vz=v[2])[0]
    flag = 'VIOLATION' if P > PW_MAX + 1e-9 else 'ok'
    print(f'  {name:22s} v_xy={vxy:7.4f}  v_z={v[2]:6.3f}  P={P:9.4f}  {flag}')

# ---------- joint_ac_controller._to_motion ----------
from joint_ac_controller import JointACCritic
jc = JointACCritic(n_users=n_users, n_uavs=n_uavs, n_grids=int(2*boundary/grid_size),
                   ss_min_types=ss_min_types, ss_max_types=ss_max_types,
                   mos_weights=mos_type_weights, type_rates=ON_data_arrival_mean_Mb,
                   d0=d0, V_lyap=lyapunov_param, queue_norm=queue_norm_Mb,
                   n_candidates=10, eval_sigma=0.5, airship_movable=True,
                   mode='directed', seed=0)
st = dict(uav_locs=np.array([[0.,0.,150.]]*n_uavs),
          airship_pos=np.array([airship_x, airship_y, airship_z]))
print('\n--- joint_ac_controller._to_motion (actor/noise actions are clipped to [-1,1]) ---')
for name, bxy in [('unit', (1.0, 0.0)), ('corner 1.414', (1.0, 1.0))]:
    a = np.zeros(3*n_uavs+2); a[0] = bxy[0]; a[1] = bxy[1]
    locs, v, air = jc._to_motion(a, st, uav_speedxy_max, uav_speedz_max)
    vxy = np.linalg.norm(v[0, :2]); P = cal_uav_propulsion_energy(V=vxy, Vz=v[0,2])[0]
    flag = 'VIOLATION' if P > PW_MAX + 1e-9 else 'ok'
    print(f'  b_xy={name:12s} v_xy={vxy:7.4f}  P={P:9.4f}  {flag}')

# ---------- eval vs exec boundary clip ----------
print('\n--- eval/exec boundary consistency (start at edge x=+249) ---')
loc_edge = np.array([249.0, 0.0, 150.0])
a = np.zeros(3*n_uavs+2); a[0] = 1.0; a[1] = 0.0
locs_j, v_j, _ = jc._to_motion(a, dict(uav_locs=np.array([loc_edge]*n_uavs),
                                       airship_pos=np.array([0.,0.,250.])),
                               uav_speedxy_max, uav_speedz_max)
print(f'  joint_ac evaluated x = {locs_j[0,0]:.4f} (clips to +boundary={boundary})')
print(f'  expA/notebook executed x = adjust_location({locs_j[0,0]:.4f}) = '
      f'{adjust_location(locs_j[0,0], bound=boundary):.4f}')
p_h, v_h = hc.motion(np.array([1.0, 0.0, 0.0]), loc_edge)
print(f'  hybrid evaluated x = {p_h[0]:.4f} ; executed = adjust_location(...) = '
      f'{adjust_location(p_h[0], bound=boundary):.4f}  (consistent)')

# ---------- vertical speed cap check ----------
print('\n--- vertical: is v_z capped at speed_z? ---')
for cz in [0.5, 1.0, 5.0]:
    a = np.zeros(3*n_uavs+2); a[2] = cz
    _, v, _ = jc._to_motion(a, st, uav_speedxy_max, uav_speedz_max)
    print(f'  joint_ac b_z={cz}: v_z={v[0,2]:.3f} (cap {uav_speedz_max})')
for cz in [0.5, 1.0, 5.0]:
    p, v = hc.motion(np.array([0.,0.,cz]), loc0)
    print(f'  hybrid   c_z={cz}: v_z={v[2]:.3f} (cap {uav_speedz_max})')

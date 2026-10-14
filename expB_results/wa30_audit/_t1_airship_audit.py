# -*- coding: utf-8 -*-
"""Task 1.3: airship-mapping / motion-compliance audit of all existing data.

Static inventory (script -> airship mapping -> legality -> affected outputs /
claims) + dynamic checks on stored trajectories (expA snapshots, expB closed-
loop npz). No new Block A solves, no new simulation.

Legality rules (system params): UAV vxy <= 30 m/s, vz <= 5 m/s,
airship |v| <= SPD_AIR = 5.0 m/s, P <= PW_MAX = 683 W,
|x|,|y| <= boundary = 250, z in [z_min, z_max].
"""
import os
import re
import glob
import numpy as np
import pandas as pd

ROOT = r'D:\Learn\work\drl-opencode'
OUT = os.path.join(ROOT, 'expB_results', 'wa30_audit')

V_XY, V_Z, V_AIR, P_MAX, BOUNDARY = 30.0, 5.0, 5.0, 683.0, 250.0

# ---------------------------------------------------------------- inventory
INV = [
    # (script, locus, mapping, legality, affected outputs, affected claims, status)
    ('expB_wa30_audit.py', 'move_air L405-410; calls L469/476',
     'airship step = SPD_XY*rho (7.5-15 m/s)', 'ILLEGAL (>5)',
     'task2_wa30_methods.csv::dJ_dual_air,air_tag,air_gain_over_uav',
     'wa30_diagnostic_report.md L113/L114/L147 "+0.093", run.log L108',
     'NON-CITABLE'),
    ('expB_wa30_audit.py', 'sets k3/bind/dual loop L452-462',
     'UAV-only; airship fixed at air0', 'LEGAL',
     'task2_wa30_methods.csv::J0,dJ_k3,dJ_bind,dJ_dual,dJ_acc (+tags)',
     'wa30_diagnostic_report.md K5 table; adaptive_step_report.md K5/ADP2b',
     'VALID'),
    ('expB_task3_frozen.py', 'eval_action_dual L380-391 (va=SPD_XY*rho*e_a)',
     'airship step = SPD_XY*rho (7.5/15 m/s)', 'ILLEGAL (>5)',
     'expA_native_slots.csv::J_du_best,dJ_du,dJ_du_vs_beta,dual_selected,'
     'du_D_sum,du_P_sum,du_vxy_mean,J_old_betaK3 (b25/b50 rows move airship)',
     'task3_frozen_run.log L59 "Hybrid win-vs-dual 0.270"; no md cites',
     'NON-CITABLE'),
    ('expB_task3_frozen.py', 'eval_action_hybrid L363-378; ExpB L500-521',
     'airship fixed (air0.copy()); UAV v clipped <=30', 'LEGAL',
     'expA_native_slots.csv::J_hy_*,dJ_hy*; expB_unified_speed_slots.csv',
     'none in md reports', 'VALID (UAV-side only)'),
    ('expB_selective_mobility.py', 'apply_move L176-182 / apply_joint L194-197',
     'illegal SPD_XY*rho branch exists BUT call sites L262 move_air=False, '
     'L278 air_fixed=True', 'LEGAL in actual runs (dead code)',
     'selective_candidates.csv, selective_slots.csv, selective_report.md',
     'selective_report.md rho 0.25/0.5 conclusions', 'VALID'),
    ('expB_scenario_audit.py', 'airship_move_eval L343-356',
     'SPD_AIR=5.0, 8 compass dirs, clip +-249', 'LEGAL',
     'task3_airship_geom.csv (dU 0.056/0.080)',
     'scenario_audit_report.md L75-77, L89-93', 'VALID'),
    ('expB_stage0.py', 'L196-199 SPD_AIR=5.0', 'airship 5 m/s directed',
     'LEGAL', 'stage0 csv/npz', 'stage0 report claims', 'VALID'),
    ('expB_stage1.py', 'L214-217 SPD_AIR=5.0', 'airship 5 m/s directed',
     'LEGAL', 'stage1_runs.npz, stage1 csv', 'stage1 report claims', 'VALID'),
    ('expB_stage2.py', 'L244-247 SPD_AIR=5.0', 'airship 5 m/s directed',
     'LEGAL', 'stage2_seed0-4.npz, stage2 csv',
     'stage2 / multi-seed report claims', 'VALID'),
    ('expB_task2_long.py', 'L208-211 SPD_AIR=5.0', 'airship 5 m/s directed',
     'LEGAL', 'task2_long_runs.npz', 'task2_long claims', 'VALID'),
    ('expB_task3_binding_closed.py', 'L229-232 SPD_AIR=5.0',
     'airship 5 m/s directed', 'LEGAL', 'task3_binding.npz',
     'task3 binding closed-loop claims', 'VALID'),
    ('joint_ac_controller.py (src)', 'AIRSHIP_SPEED_MAX=5.0 L64; clip L353',
     'production airship limit 5 m/s', 'LEGAL (reference)',
     'expA_dual_diagnostics.py -> expA_results/snapshots.npz',
     'ALL W_A=30 audit work (snapshots are the single full-state source)',
     'VALID'),
    ('expA_dual_diagnostics.py', 'via JointACCritic (AIRSHIP_SPEED_MAX=5.0)',
     'airship <=2.5 m/s measured', 'LEGAL',
     'expA_results/snapshots.npz (152 snaps, slots 4..500)',
     'everything built on snapshots', 'VALID'),
    ('expB_task3_hybrid.py', 'hybrid actor: vxy<=30, vz<=5, airship fixed',
     'airship fixed; UAV speed at limits', 'LEGAL on speed; '
     'P reaches 745.6 W > PW_MAX=683 (combined xy+z)',
     'task3_seed0/2.npz::hybrid_P_hist, task3_kpi.csv::pw_max=745.57',
     'no md report cites hybrid KPI (only ablation design L42 = Hybrid-slow)',
     'CAVEAT: power over PW_MAX; uncited'),
    ('expB_task3b_slow.py', 'fast variant actor same as above; slow v<=4.8',
     'airship fixed', 'LEGAL on speed; fast P reaches 745.6 W > 683',
     'task3b_seed0/2.npz::fast_P_hist, task3b_kpi.csv::slots_over_683=1214/1188',
     'no md report cites fast/slow KPI', 'CAVEAT: power over PW_MAX; uncited'),
    ('expB_task3_adaptive / step_sweep / _grid_residual / _task2_*',
     'airship fixed at air0 for every candidate',
     'airship fixed (rho_air=0)', 'LEGAL',
     'task2_step_sweep.csv, task2_extended_vs_k5.csv, task3_adaptive_replay.csv',
     'adaptive_step_report.md (K5/ADP2b/oracle numbers)', 'VALID'),
    ('expB_hybrid_*.py (task2_repro)', 'airship fixed; vxy<=30 (lookahead)',
     'airship fixed', 'LEGAL',
     'task2_hybrid_lookahead.npz (P_hist max 139.2)',
     'task1_audit_report.md Hybrid MOS=4.9967 baseline', 'VALID'),
    ('expB_ablation_guidance*.py', 'airship fixed (design uses hybrid-slow)',
     'airship fixed', 'LEGAL', 'ablation_guidance/*',
     'ablation_guidance_design.md (design only, not executed)', 'VALID'),
]
inv = pd.DataFrame(INV, columns=['script', 'locus', 'airship_mapping',
                                 'legality', 'affected_outputs',
                                 'affected_claims', 'status'])
inv.to_csv(os.path.join(OUT, 'task1_airship_audit.csv'), index=False)
print('=== 1.3 static inventory ===')
print(inv.groupby(['status']).size().to_string())
print()

# ------------------------------------------------- dynamic: expA snapshots
z = np.load(os.path.join(ROOT, 'expA_results', 'snapshots.npz'))
sl = z['slots'].astype(int)
gap = np.diff(sl)
ld, le = z['locs_dec'], z['locs_exec']
ad, ae = z['air_dec'], z['air_exec']
uav_slot = np.linalg.norm((le - ld)[:, :, [0, 1]], axis=2)          # m / slot
uav_chain = np.linalg.norm((le[1:] - le[:-1])[:, :, [0, 1]], axis=2) / gap[:, None]
air_slot = np.linalg.norm((ae - ad)[:, :2], axis=1)
air_chain = np.linalg.norm((ae[1:] - ae[:-1])[:, :2], axis=1) / gap
print('=== 1.3 dynamic: expA snapshots (source of all W_A=30 audits) ===')
print('  slots=%d..%d gap=%s' % (sl[0], sl[-1], dict(zip(*np.unique(gap, return_counts=True)))))
print('  UAV within-slot max=%.3f m/s (lim %.0f)  chain/gap max=%.3f  n>30=%d'
      % (uav_slot.max(), V_XY, uav_chain.max(), int((uav_chain > V_XY).sum())))
print('  AIR within-slot max=%.3f m/s (lim %.0f)  chain/gap max=%.3f  n>5=%d'
      % (air_slot.max(), V_AIR, air_chain.max(), int((air_chain > V_AIR).sum())))
print('  UAV |xy|max=%.1f  AIR |xy|max=%.1f (lim %.0f)  z=[%.0f,%.0f]'
      % (np.abs(le[:, :, :2]).max(), np.abs(ae[:, :2]).max(), BOUNDARY,
         le[:, :, 2].min(), le[:, :, 2].max()))
ok_expA = (uav_slot.max() <= V_XY + 1e-9 and air_slot.max() <= V_AIR + 1e-9
           and uav_chain.max() <= V_XY + 1e-9 and air_chain.max() <= V_AIR + 1e-9
           and np.abs(le[:, :, :2]).max() <= BOUNDARY + 1e-9
           and np.abs(ae[:, :2]).max() <= BOUNDARY + 1e-9)
print('  -> expA snapshot trajectory COMPLIANT: %s' % ok_expA)
print()

# --------------------------------------------- dynamic: expB closed loops
print('=== 1.3 dynamic: expB closed-loop npz compliance ===')
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, 'expB_results', '**', '*.npz'),
                          recursive=True)):
    zf = np.load(f)
    rel = os.path.relpath(f, ROOT)
    for k in sorted(zf.files):
        m = re.match(r'^(.*)_(vxy|vz|vair|P_hist|pw_hist)$', k)
        if not m:
            continue
        a = np.asarray(zf[k], dtype=float)
        kind = m.group(2)
        lim = {'vxy': V_XY, 'vz': V_Z, 'vair': V_AIR,
               'P_hist': P_MAX, 'pw_hist': P_MAX}[kind]
        mx = float(np.nanmax(a))
        rows.append(dict(file=rel, series=k, kind=kind, limit=lim,
                         max_val=mx, ok=bool(mx <= lim + 1e-6),
                         n_over=int(np.sum(a > lim + 1e-6))))
c = pd.DataFrame(rows)
c.to_csv(os.path.join(OUT, 'task1_motion_compliance.csv'), index=False)
bad = c[~c['ok']]
print('  series checked: %d   violations: %d' % (len(c), len(bad)))
if len(bad):
    print(bad[['file', 'series', 'limit', 'max_val', 'n_over']].to_string(index=False))
legal_closed = c[c['series'].str.contains('k3|k5|seed[02]P|dualk5|betak5|slow')]
print('  K5/Dual/Binding closed-loop series all ok: %s'
      % bool(legal_closed['ok'].all()))
print()

# ------------------------------------------------------- non-citable list
print('=== 1.3 NON-CITABLE / CAVEAT list ===')
for _, r in inv[inv['status'] != 'VALID'].iterrows():
    print('  [%s] %s @ %s' % (r['status'], r['script'], r['locus']))
    print('      outputs: %s' % r['affected_outputs'])
    print('      claims : %s' % r['affected_claims'])
print()
print('written: task1_airship_audit.csv, task1_motion_compliance.csv')

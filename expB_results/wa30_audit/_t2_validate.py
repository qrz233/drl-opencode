# -*- coding: utf-8 -*-
"""Task 2: independent validation of the FROZEN ADP2b rule vs fixed K5 vs
extended-grid oracle at W_A=20 and W_A=30 MHz (new snapshots, no tuning).

Selection (frozen BEFORE any evaluation):
  exclude design-40 (task2_step_sweep si); stratify remaining 112 into
  mixed    = both pools bound at either bandwidth,
  early_cong = qlen >= median, not mixed,
  stable   = qlen < median, not mixed;
  sample 16/stratum, random_state=0 -> 48 snapshots.
  All 152 snapshots come from ONE trajectory (slots 4..500): results are
  descriptive, NOT independent-seed statistics.

Per snapshot x bandwidth:
  recompute base solve AT that bandwidth (binding dir depends on duals);
  sweep rho {0,0.1,0.25,0.5,0.75,1.0} x dir {beta,bind}, UAV-only, airship
  fixed (legal SPD_AIR mapping); legality flags incl. rho=1.0 power check.
Calibers (thresholds frozen in task 1.1: EPS=1e-9, tie 1e-6):
  K5(fix5) = min{stay, beta/bind x rho 0.25,0.5}
  ADP2b    = frozen rule: stay; b25; bd25; if <stay -> d*50; if J50<J25 ->
             d*75 (d* = better dir at 0.25, tie->beta; argmin incl. stay)
  oracle   = min over ALL valid sweep rows (rho=1.0 judged per candidate)
Outputs -> task2_validation_selection.csv / _sweep.csv / _calibers.csv + print
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
import numpy as np
import pandas as pd

PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
os.chdir(PROJ)   # params_sys creates relative sim-output dir at import
OUT = r'D:\Learn\work\drl-opencode\expB_results\wa30_audit'
ROOT = r'D:\Learn\work\drl-opencode'
SRC = open(os.path.join(OUT, '_grid_residual.py'), encoding='utf-8').read()
cut = SRC.index('rows = []')
ns = {}
exec(compile(SRC[:cut], '_grid_prefix', 'exec'), ns)
solve_full, gammas, beta_dirs, unit, SNAP = (
    ns['solve_full'], ns['gammas'], ns['beta_dirs'], ns['unit'], ns['SNAP'])
SPD_XY, TAU, BOUNDARY = ns['SPD_XY'], ns['TAU'], ns['BOUNDARY']
NU_EPS, BIND_EPS = ns['NU_EPS'], ns['BIND_EPS']
z_min, z_max, PSI_E = ns['z_min'], ns['z_max'], ns['PSI_E']
cal_uav_propulsion_energy = ns['cal_uav_propulsion_energy']

EPS, EPS_TIE = 1e-9, 1e-6
V_MAX, PW_MAX = 30.0, 683.0
RHOS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
DIRS = ('beta', 'bind')
BWS = (20.0, 30.0)
N_PER_STRATUM = 16

# ---------------- selection (frozen) ----------------
a20 = pd.read_csv(os.path.join(ROOT, 'expB_results', 'scenario_audit',
                               'task1_audit.csv'))
a30 = pd.read_csv(os.path.join(ROOT, 'expB_results', 'wa30_audit',
                               'task1_wa30_stats.csv'))
des = pd.read_csv(os.path.join(OUT, 'task2_step_sweep.csv'))
d40 = set(int(s) for s in des['si'].unique())
m = a20[['si', 't', 'qlen', 'pool_acc_bind', 'pool_bh_bind']].merge(
    a30[['si', 'pool_acc_bind', 'pool_bh_bind']], on='si',
    suffixes=('_20', '_30'))
rem = m[~m['si'].isin(d40)].copy()
qmed = float(rem['qlen'].median())
mixed = (((rem['pool_acc_bind_20'] == 1) & (rem['pool_bh_bind_20'] == 1)) |
         ((rem['pool_acc_bind_30'] == 1) & (rem['pool_bh_bind_30'] == 1)))
rem['stratum'] = np.where(mixed, 'mixed',
                          np.where(rem['qlen'] >= qmed, 'early_cong', 'stable'))
parts = [g.sample(n=min(N_PER_STRATUM, len(g)), random_state=0)
         for _, g in rem.groupby('stratum')]
sel = pd.concat(parts).sort_values('si').reset_index(drop=True)
sel.to_csv(os.path.join(OUT, 'task2_validation_selection.csv'), index=False)
SEL = {int(r.si): str(r.stratum) for r in sel.itertuples()}
assert not (set(SEL) & d40), 'selection overlaps design-40'
print('selection: %s  qmed=%.4f  excluded design-40=%d  n=%d'
      % (sel['stratum'].value_counts().to_dict(), qmed, len(d40), len(sel)),
      flush=True)

# ---------------- eval helpers ----------------
def eval_cand(locs, air, sn, E0, beta, W_A):
    net = locs - sn['locs0']
    v_net = np.linalg.norm(net[:, :2], axis=1) / TAU
    vz = np.abs(net[:, 2]) / TAU
    P = np.array([float(cal_uav_propulsion_energy(V=v_net[u], Vz=vz[u])[0])
                  for u in range(4)])
    ga, gb, g = gammas(sn['ux'], sn['uy'], sn['cids'], locs, air)
    t0 = time.time()
    res = solve_full(ga, gb, sn['demand'], sn['cids'], beta, W_A=W_A)
    dt = time.time() - t0
    J = PSI_E * float(E0 @ P) - float(beta @ res['D_Mb'])
    ok_status = res['status'] in ('optimal', 'optimal_inaccurate')
    ok_speed = bool(np.all(v_net <= V_MAX + 1e-9))
    ok_power = bool(np.all(P <= PW_MAX + 1e-9))
    ok_region = bool(np.all(np.abs(locs[:, 0]) <= BOUNDARY + 1e-9) and
                     np.all(np.abs(locs[:, 1]) <= BOUNDARY + 1e-9))
    ok_alt = bool(np.all(locs[:, 2] >= z_min - 1e-9) and
                  np.all(locs[:, 2] <= z_max + 1e-9))
    return dict(J=J, P=P, res=res, dt=dt, ok=bool(ok_status and ok_speed and
                 ok_power and ok_region and ok_alt),
                ok_status=ok_status, ok_speed=ok_speed, ok_power=ok_power,
                ok_region=ok_region, ok_alt=ok_alt, v_net=v_net,
                D=float(beta @ res['D_Mb']),
                Eterm=PSI_E * float(E0 @ P), P_sum=float(P.sum()),
                P_max=float(P.max()))


def move(locs0, e_u, rho):
    locs = locs0.copy()
    if rho > 0:
        disp = SPD_XY * rho * e_u * TAU
        locs[:, 0] = np.clip(locs0[:, 0] + disp[:, 0], -BOUNDARY, BOUNDARY)
        locs[:, 1] = np.clip(locs0[:, 1] + disp[:, 1], -BOUNDARY, BOUNDARY)
    return locs


# ---------------- sweep + calibers ----------------
by_si = {sn['si']: sn for sn in SNAP}
crows, frows = [], []
t_start = time.time()
for W_A in BWS:
    for si in sorted(SEL):
        sn = by_si[si]
        beta, types, cids = sn['beta'], sn['types'], sn['cids']
        locs0, air0, E0 = sn['locs0'], sn['air0'], sn['E0']
        ga0, gb0, _ = gammas(sn['ux'], sn['uy'], cids, locs0, air0)
        base = solve_full(ga0, gb0, sn['demand'], cids, beta, W_A=W_A)
        nu = np.maximum(base['nu_bh'], 0.0)
        e_u_beta, _ = beta_dirs(beta, types, cids, locs0, air0,
                                sn['ux'], sn['uy'])
        e_bh = np.array([unit(air0[:2] - locs0[u, :2]) for u in range(4)])
        w_b = (nu > BIND_EPS).astype(float)
        e_u_bind = np.array([unit((1 - w_b[u]) * e_u_beta[u] +
                                  w_b[u] * e_bh[u]) for u in range(4)])

        meta = {}
        for dname, e_u in (('beta', e_u_beta), ('bind', e_u_bind)):
            for rho in RHOS:
                r = eval_cand(move(locs0, e_u, rho), air0.copy(), sn, E0,
                              beta, W_A)
                meta[(dname, rho)] = r
                crows.append(dict(
                    bw=W_A, si=si, stratum=SEL[si], t=sn['t'], qlen=sn['qlen'],
                    dir=dname, rho=rho, J=r['J'], D=r['D'], Eterm=r['Eterm'],
                    P_max=r['P_max'], P_sum=r['P_sum'],
                    v_max=float(r['v_net'].max()), valid=int(r['ok']),
                    ok_status=int(r['ok_status']), ok_speed=int(r['ok_speed']),
                    ok_power=int(r['ok_power']), ok_region=int(r['ok_region']),
                    ok_alt=int(r['ok_alt']), solve_s=round(r['dt'], 3),
                    status=r['res']['status'],
                    pool_bh=int(r['res']['nu_bh_pool'] > 1e-6),
                    pool_acc=int(r['res']['nu_acc_pool'] > 1e-6)))

        J = {(d, r): meta[(d, r)]['J'] for d in DIRS for r in RHOS}
        ok = {(d, r): meta[(d, r)]['ok'] for d in DIRS for r in RHOS}
        J0 = 0.5 * (J[('beta', 0.0)] + J[('bind', 0.0)])
        fix5_keys = [('beta', 0.0), ('beta', 0.25), ('beta', 0.5),
                     ('bind', 0.25), ('bind', 0.5)]
        fix5 = min(J[k] for k in fix5_keys)
        fkey = min(fix5_keys, key=lambda k: J[k])
        oracle = min(J[k] for k in J if ok[k])
        okey = min((k for k in J if ok[k]), key=lambda k: J[k])

        Jb, Jd = J[('beta', 0.25)], J[('bind', 0.25)]
        dstar = 'beta' if Jb <= Jd + EPS else 'bind'
        J25 = Jb if dstar == 'beta' else Jd
        improved25 = J25 < J0 - EPS
        keys = [('beta', 0.0), ('beta', 0.25), ('bind', 0.25)]
        trace = ['stay', 'b25', 'bd25']
        if improved25:
            keys.append((dstar, 0.5)); trace.append('d*50')
            if J[(dstar, 0.5)] < J25 - EPS:
                keys.append((dstar, 0.75)); trace.append('d*75')
        adp2b = min(J[k] for k in keys)
        akey = min(keys, key=lambda k: J[k])
        calls = len(keys)
        stop3 = int(calls == 3)

        def pk(k, col):
            return meta[k][col]

        stayE, stayP = pk(('beta', 0.0), 'Eterm'), pk(('beta', 0.0), 'P_sum')
        frows.append(dict(
            bw=W_A, si=si, stratum=SEL[si], t=sn['t'], qlen=sn['qlen'],
            J0=J0, J_fix5=fix5, J_adp2b=adp2b, J_oracle=oracle,
            gain_fix5=J0 - fix5, gain_adp2b=J0 - adp2b,
            gain_oracle=J0 - oracle,
            adp2b_minus_fix5=fix5 - adp2b, oracle_minus_fix5=fix5 - oracle,
            leak_adp2b_oracle=adp2b - oracle,
            calls_adp2b=calls, adp2b_trace='+'.join(trace),
            adp2b_dstar=dstar, stop3=stop3,
            akey_dir=akey[0], akey_rho=akey[1],
            fkey_dir=fkey[0], fkey_rho=fkey[1],
            okey_dir=okey[0], okey_rho=okey[1],
            rho1_valid=int(ok[('beta', 1.0)] or ok[('bind', 1.0)]),
            rho1_power_ok=int(ok[('beta', 1.0)] and ok[('bind', 1.0)]),
            dP_adp2b=pk(akey, 'P_sum') - stayP,
            dE_adp2b=pk(akey, 'Eterm') - stayE,
            dP_fix5=pk(fkey, 'P_sum') - stayP,
            dE_fix5=pk(fkey, 'Eterm') - stayE,
            dP_oracle=pk(okey, 'P_sum') - stayP,
            dE_oracle=pk(okey, 'Eterm') - stayE,
            solve_s_adp2b=float(sum(pk(k, 'dt') for k in keys)),
            solve_s_fix5=float(sum(pk(k, 'dt') for k in fix5_keys)),
            solve_s_oracle=float(sum(pk(k, 'dt') for k in J if ok[k])),
            n_valid=int(sum(ok.values())),
            pool_acc_base=int(base['nu_acc_pool'] > 1e-6),
            pool_bh_base=int(base['nu_bh_pool'] > 1e-6)))
    print('  bw=%.0f done, elapsed %.0fs' % (W_A, time.time() - t_start),
          flush=True)

cw = pd.DataFrame(crows)
fw = pd.DataFrame(frows)
cw.to_csv(os.path.join(OUT, 'task2_validation_sweep.csv'), index=False)
fw.to_csv(os.path.join(OUT, 'task2_validation_calibers.csv'), index=False)

# ---------------- summary ----------------
print('\n=== task 2 validation: %d snaps x %d bw = %d caliber rows ==='
      % (len(sel), len(BWS), len(fw)), flush=True)
print('non-independence: all snapshots come from ONE trajectory '
      '(expA snapshots.npz, slots 4..500) -- descriptive statistics only, '
      'not independent seeds.', flush=True)
for bw in BWS:
    g = fw[fw['bw'] == bw]
    print('\n--- W_A=%.0f MHz  n=%d  strata=%s ---'
          % (bw, len(g), g['stratum'].value_counts().to_dict()))
    for c in ('gain_fix5', 'gain_adp2b', 'gain_oracle'):
        print('  %-12s mean=%+.4f med=%+.4f sum=%+.4f'
              % (c, g[c].mean(), g[c].median(), g[c].sum()))
    d = g['adp2b_minus_fix5'].values
    for eps, tag in ((EPS, '1e-9'), (EPS_TIE, '1e-6')):
        print('  ADP2b vs K5 eps=%s win=%d tie=%d lose=%d mean=%+.4f'
              % (tag, int((d > eps).sum()), int((np.abs(d) <= eps).sum()),
                 int((d < -eps).sum()), d.mean()))
    lk = g['leak_adp2b_oracle']
    print('  oracle<=adp2b all: %s | leak>1e-9: %d (max %.2e sum %.2e) | '
          'oracle<=K5 all: %s'
          % (bool((lk >= -EPS).all()), int((lk > EPS).sum()), lk.max(),
             lk.sum(), bool((g['oracle_minus_fix5'] >= -EPS).all())))
    print('  calls: %s mean=%.3f  stop3=%d'
          % (g['calls_adp2b'].value_counts().sort_index().to_dict(),
             g['calls_adp2b'].mean(), int(g['stop3'].sum())))
    st3 = g[g['stop3'] == 1]
    if len(st3):
        print('  stop3 leak>1e-9: %d/%d snaps (si=%s)'
              % (int((st3['leak_adp2b_oracle'] > EPS).sum()), len(st3),
                 st3.loc[st3['leak_adp2b_oracle'] > EPS, 'si'].tolist()))
    print('  rho=1.0 valid: %d/%d (power-only failures expected)'
          % (int(g['rho1_valid'].sum()), len(g)))
    print('  dP_sum adp2b=%+.1fW fix5=%+.1fW oracle=%+.1fW | '
          'dE_term adp2b=%.4f fix5=%.4f oracle=%.4f'
          % (g['dP_adp2b'].mean(), g['dP_fix5'].mean(),
             g['dP_oracle'].mean(), g['dE_adp2b'].mean(),
             g['dE_fix5'].mean(), g['dE_oracle'].mean()))
    print('  solve_s/call: adp2b=%.3f fix5=%.3f oracle=%.3f | '
          'n_valid=%s'
          % (g['solve_s_adp2b'].mean(), g['solve_s_fix5'].mean(),
             g['solve_s_oracle'].mean(), sorted(g['n_valid'].unique())))
    st = g.groupby('stratum')[['gain_fix5', 'gain_adp2b', 'gain_oracle']].mean()
    print('  per-stratum mean gains:')
    print(st.to_string(float_format=lambda x: '%+.4f' % x))

g20 = fw[fw['bw'] == 20.0].set_index('si')
g30 = fw[fw['bw'] == 30.0].set_index('si')
print('\n--- cross-bandwidth (paired, same 48 snaps) ---')
for c in ('gain_fix5', 'gain_adp2b', 'gain_oracle'):
    d = g30[c] - g20[c]
    print('  mean(%s @30 - @20) = %+.4f  (n=%d, paired not independent)'
          % (c, d.mean(), len(d)))
print('\nK5 rho tags @30: %s' % g30['fkey_rho'].value_counts().to_dict())
print('ADP2b chosen rho @30: %s' % g30['akey_rho'].value_counts().to_dict())
print('oracle rho @30: %s' % g30['okey_rho'].value_counts().to_dict())
print('oracle rho @20: %s' % g20['okey_rho'].value_counts().to_dict())
print('\ntotal sweep solves=%d in %.0fs; written: task2_validation_selection'
      '/_sweep/_calibers.csv' % (len(cw), time.time() - t_start))

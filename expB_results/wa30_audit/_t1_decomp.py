# -*- coding: utf-8 -*-
"""Task 1.2: unified delivery/energy decomposition on the W_A=30 design set.

Pure replay of task2_step_sweep.csv (no new Block A solves).

For every snapshot x direction, with two baselines (Stay = rho 0, and rho 0.5):
  dU     = D_cand - D_base          [Mb, beta-weighted delivered bits]
  dC_E   = Eterm_cand - Eterm_base  [psi_E-scaled energy cost, J-equivalent]
  dJ     = J_base - J_cand = dU - dC_E   (identity check; J = Eterm - D)

Reported per dir, per baseline, for candidate rho in {0.1, 0.25, 0.5, 0.75},
plus the conditional subset where the extended-grid best rho* == 0.75.
Sample = 40 design snapshots, 10 valid candidates each (rho=1.0 excluded,
power-infeasible). Units: D/Mb is beta-weighted sum over served UAVs;
Eterm = psi_E * E0 . P with psi_E = 1/683 and E0, P in W-slot/W -> J-equivalent.
"""
import os
import numpy as np
import pandas as pd

OUT = r'D:\Learn\work\drl-opencode\expB_results\wa30_audit'
EPS = 1e-9

sw = pd.read_csv(os.path.join(OUT, 'task2_step_sweep.csv'))
ext = pd.read_csv(os.path.join(OUT, 'task2_extended_vs_k5.csv')).set_index('si')
v = sw[sw['valid'] == 1].copy()

# per-si/dir rho* (extended best within that dir) and overall rho*
best_dir = v.loc[v.groupby(['si', 'dir'])['J'].idxmin()][['si', 'dir', 'rho']]
star75 = set(best_dir.loc[best_dir['rho'] == 0.75, 'si'])
star_dir75 = set(map(tuple, best_dir.loc[best_dir['rho'] == 0.75,
                                          ['si', 'dir']].values))
overall75 = set(ext.loc[ext['rho'] == 0.75].index)

piv = {k: v.pivot_table(index='si', columns=['dir', 'rho'], values=k)
       for k in ('J', 'D', 'Eterm')}

rows = []
for dname in ('beta', 'bind'):
    for base_rho in (0.0, 0.5):
        for cand_rho in (0.1, 0.25, 0.5, 0.75):
            if cand_rho == base_rho:
                continue
            key_b, key_c = (dname, base_rho), (dname, cand_rho)
            if key_b not in piv['J'].columns or key_c not in piv['J'].columns:
                continue
            si = piv['J'].index
            dU = (piv['D'][key_c] - piv['D'][key_b]).values
            dC = (piv['Eterm'][key_c] - piv['Eterm'][key_b]).values
            dJ = (piv['J'][key_b] - piv['J'][key_c]).values
            resid = dJ - (dU - dC)
            for i, s in enumerate(si):
                rows.append(dict(
                    si=int(s), dir=dname, base_rho=base_rho, cand_rho=cand_rho,
                    dU=dU[i], dC_E=dC[i], dJ_gain=dJ[i],
                    in_star75_overall=int(s in overall75),
                    in_star75_dir=int((s, dname) in star_dir75),
                ))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'task1_decomp.csv'), index=False)

print('=== task 1.2 decomposition: identity dJ = dU - dC_E ===')
print('  max|resid| = %.3e   rows=%d  snaps=%d  dirs=%s'
      % (abs(df['dJ_gain'] - (df['dU'] - df['dC_E'])).max(),
         len(df), df['si'].nunique(), sorted(df['dir'].unique())))
print('  sample: 40 design snaps; valid rho in {0.1,0.25,0.5,0.75} '
      '(rho=1.0 power-invalid, excluded); baselines rho in {0, 0.5}')
print()

def blk(sub, tag):
    if not len(sub):
        return
    print('  %-38s n=%3d  dU mean=%+8.4f [%+7.3f,%+7.3f]  '
          'dC_E mean=%+8.4f  dJ mean=%+8.4f  med=%+7.4f'
          % (tag, len(sub), sub['dU'].mean(), sub['dU'].min(), sub['dU'].max(),
             sub['dC_E'].mean(), sub['dJ_gain'].mean(),
             np.median(sub['dJ_gain'])))

print('=== A. candidate vs Stay (rho 0), split by dir ===')
for dname in ('beta', 'bind'):
    for r in (0.1, 0.25, 0.5, 0.75):
        blk(df[(df['dir'] == dname) & (df['base_rho'] == 0.0) &
               (df['cand_rho'] == r)], 'cand rho=%.2f  dir=%s' % (r, dname))
print()

print('=== B. marginal candidate vs rho 0.5 (same dir) ===')
for dname in ('beta', 'bind'):
    for r in (0.1, 0.25, 0.75):
        blk(df[(df['dir'] == dname) & (df['base_rho'] == 0.5) &
               (df['cand_rho'] == r)], 'rho=%.2f - rho=0.50  dir=%s' % (r, dname))
print()

print('=== C. conditional: snaps whose extended-grid rho* == 0.75 ===')
print('  overall n=%d (si=%s); per-dir n(beta)=%d n(bind)=%d'
      % (len(overall75), sorted(overall75),
         sum(1 for s in overall75 if (s, 'beta') in star_dir75),
         sum(1 for s in overall75 if (s, 'bind') in star_dir75)))
sub = df[(df['in_star75_overall'] == 1) & (df['base_rho'] == 0.0) &
         (df['cand_rho'] == 0.75)]
blk(sub, 'rho=0.75 vs Stay, star75 snaps (all dirs)')
for dname in ('beta', 'bind'):
    blk(sub[sub['dir'] == dname], '  dir=%s only' % dname)
sub = df[(df['in_star75_overall'] == 1) & (df['base_rho'] == 0.5) &
         (df['cand_rho'] == 0.75)]
blk(sub, 'rho=0.75 - rho=0.50, star75 snaps (all dirs)')
for dname in ('beta', 'bind'):
    blk(sub[sub['dir'] == dname], '  dir=%s only' % dname)
print()

print('=== D. same rows on NON-star75 snaps (contrast) ===')
sub = df[(df['in_star75_overall'] == 0) & (df['base_rho'] == 0.0) &
         (df['cand_rho'] == 0.75)]
blk(sub, 'rho=0.75 vs Stay, non-star75 (all dirs)')
sub = df[(df['in_star75_overall'] == 0) & (df['base_rho'] == 0.5) &
         (df['cand_rho'] == 0.75)]
blk(sub, 'rho=0.75 - rho=0.50, non-star75 (all dirs)')
print()

print('=== E. energy-vs-delivery scaling check (report claims) ===')
m = df[(df['base_rho'] == 0.5) & (df['cand_rho'] == 0.75)]
print('  marginal 0.75 vs 0.5: dC_E mean=%.3f  dU mean=%.4f..%.4f (by dir) '
      '-> report claim dEterm=+13.5 >> dD=+0.3~0.7'
      % (m['dC_E'].mean(),
         m[m['dir'] == 'beta']['dU'].mean(), m[m['dir'] == 'bind']['dU'].mean()))
s = df[(df['base_rho'] == 0.0) & (df['cand_rho'] == 0.75)]
print('  0.75 vs Stay: dC_E mean=%.3f  dU mean=%.4f (report +17.8 / +1.1~+2.3)'
      % (s['dC_E'].mean(), s['dU'].mean()))
neg = s[s['dJ_gain'] < 0]
print('  0.75 vs Stay: negative dJ share = %d/%d (mean dJ of those = %.2f) -> '
      'all-sample mean dominated by high-power candidates'
      % (len(neg), len(s), neg['dJ_gain'].mean()))
print()
print('written: task1_decomp.csv')

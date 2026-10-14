# -*- coding: utf-8 -*-
"""Task 1.1: unify evaluation calibers on the W_A=30 40-snapshot design set.

Pure replay of stored CSVs (no new Block A solves). Produces one per-snapshot
table with both K5 calibers, frozen-rule ADP2b, extended-grid oracle, legal
masks and call counts, then checks the identities that explain why
  1.654 (Binding-K5 gain) + 0.369 (k5best->ext gap) != 2.028 (oracle gain).

Calibers:
  fix5      = min over sweep rows {rho in 0,0.25,0.5} x dir in {beta,bind}   [sweep grid]
  k5_bind   = J0 - dJ_bind  (methods eval path)
  k5_dual   = J0 - dJ_dual  (methods; dual dir NOT in sweep grid)
  k5_k3     = J0 - dJ_k3
  k5best    = min(k5_dual, k5_bind, k5_k3)                                  [methods]
  ADP2b     = frozen rule (early-stop trend-extend), candidates subset of sweep grid
  oracle    = min over ALL valid sweep rows (= task2_extended_vs_k5 J_ext)
"""
import os
import numpy as np
import pandas as pd

OUT = r'D:\Learn\work\drl-opencode\expB_results\wa30_audit'
EPS = 1e-9
EPS_TIE = 1e-6          # win/tie/lose counting threshold (matches report)
RHOS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
DIRS = ('beta', 'bind')

sw = pd.read_csv(os.path.join(OUT, 'task2_step_sweep.csv'))
me = pd.read_csv(os.path.join(OUT, 'task2_wa30_methods.csv')).set_index('si')
ext = pd.read_csv(os.path.join(OUT, 'task2_extended_vs_k5.csv')).set_index('si')
rep = pd.read_csv(os.path.join(OUT, 'task3_adaptive_replay.csv')).set_index('si')

# --- valid mask per snapshot: 12 rows ordered dir x rho ---
sw['ord'] = sw['dir'].map({'beta': 0, 'bind': 1}) * 6 + sw['rho'].map(
    {r: i for i, r in enumerate(RHOS)})
sw = sw.sort_values(['si', 'ord'])

rows = []
for si, g in sw.groupby('si'):
    g = g.set_index(['dir', 'rho'])
    m = {(d, r): g.loc[(d, r)] for d in DIRS for r in RHOS}
    J = {k: float(v['J']) for k, v in m.items()}
    valid = np.array([int(m[(d, r)]['valid']) for d in DIRS for r in RHOS])
    ok_bits = ''.join(str(int(m[(d, r)]['ok_status'])) for d in DIRS for r in RHOS)

    J0 = 0.5 * (J[('beta', 0.0)] + J[('bind', 0.0)])          # stay, both rows identical
    J0_spread = abs(J[('beta', 0.0)] - J[('bind', 0.0)])

    # fix5 (sweep caliber, binding K5)
    fix5 = min(J[('beta', 0.0)], J[('beta', 0.25)], J[('beta', 0.5)],
               J[('bind', 0.25)], J[('bind', 0.5)])
    # extended oracle = min over valid sweep rows
    oracle = min(J[k] for k in J if int(m[k]['valid']))

    # methods calibers
    m0 = me.loc[si]
    J0m = float(m0['J0'])
    k5_bind = J0m - float(m0['dJ_bind'])
    k5_dual = J0m - float(m0['dJ_dual'])
    k5_k3 = J0m - float(m0['dJ_k3'])
    k5best = min(k5_bind, k5_dual, k5_k3)

    # ---- frozen ADP2b rule (recomputed from sweep grid) ----
    Jb, Jd = J[('beta', 0.25)], J[('bind', 0.25)]
    dstar = 'beta' if Jb <= Jd + EPS else 'bind'
    J25 = Jb if dstar == 'beta' else Jd
    dother = 'bind' if dstar == 'beta' else 'beta'
    improved25 = J25 < J0 - EPS
    cand, calls = [J0, Jb, Jd], 3
    trace = ['stay', 'b25', 'bd25']
    if improved25:
        J50 = J[(dstar, 0.5)]
        cand.append(J50); calls += 1; trace.append('d*50')
        if J50 < J25 - EPS:
            cand.append(J[(dstar, 0.75)]); calls += 1; trace.append('d*75')
    adp2b = min(cand)

    # early-stop leak candidates (used by task 1.4 too)
    stop_leak = min(J0, Jb, Jd) - min(J[(dd, rr)] for dd in DIRS
                                      for rr in RHOS if int(m[(dd, rr)]['valid']))
    other_dir_50 = J[(dother, 0.5)] - J[(dstar, 0.5)]

    rows.append(dict(
        si=si, t=int(m0['t']), qlen=float(m0['qlen']),
        n_valid=int(valid.sum()), valid_bits=''.join(map(str, valid)),
        ok_status_bits=ok_bits,
        J0=J0, J0_spread=J0_spread, J0_methods_vs_sweep=J0m - J0,
        J_k5_fix5=fix5, J_k5_bind=k5_bind, J_k5_dual=k5_dual, J_k5_k3=k5_k3,
        J_k5best=k5best, J_adp2b=adp2b, J_oracle=oracle,
        calls_fix5=5, calls_adp2b=calls, calls_oracle=11,
        adp2b_trace='+'.join(trace), adp2b_dstar=dstar,
        adp2b_improved25=int(improved25),
        gap_fix5_oracle=fix5 - oracle,
        gap_k5best_oracle=k5best - oracle,
        gap_adp2b_oracle=adp2b - oracle,
        gain_fix5=J0 - fix5, gain_k5best=J0 - k5best,
        gain_adp2b=J0 - adp2b, gain_oracle=J0 - oracle,
        adp2b_minus_fix5=fix5 - adp2b,
        adp2b_minus_k5dual=k5_dual - adp2b,
        stop_leak=stop_leak, other_dir_50=other_dir_50,
        dJ_bind_methods=float(m0['dJ_bind']),
    ))

df = pd.DataFrame(rows).sort_values('si').reset_index(drop=True)
df.to_csv(os.path.join(OUT, 'task1_caliber.csv'), index=False)

n = len(df)
print('=== task 1.1 caliber table: %d snaps ===' % n)
print('rows/si=%s  valid per si=%s  ok_status all=1: %s'
      % (sw.groupby('si').size().unique().tolist(),
         sorted(df['n_valid'].unique().tolist()),
         bool((df['ok_status_bits'] == '1' * 12).all())))
print('invalid rows are rho=1.0 only: %s'
      % set(zip(sw.loc[sw['valid'] == 0, 'dir'], sw.loc[sw['valid'] == 0, 'rho'])))
print('J0: max|sweep(stay rows spread)|=%.3e  max|methods-sweep|=%.3e'
      % (df['J0_spread'].max(), df['J0_methods_vs_sweep'].abs().max()))
print()

def stat(x, name):
    x = np.asarray(x, float)
    print('  %-22s mean=%+.6f  med=%+.6f  sum=%+.4f' %
          (name, x.mean(), np.median(x), x.sum()))

print('=== gains vs Stay (dJ = J0 - J) ===')
for c in ('J_k5_fix5', 'J_k5_dual', 'J_k5best', 'J_adp2b', 'J_oracle'):
    stat(df['J0'] - df[c], c)
print()

print('=== identity check (must sum exactly per caliber) ===')
lhs1 = df['gain_fix5'] + df['gap_fix5_oracle']
print('  fix5 caliber  : gain_fix5 + gap_fix5_oracle - gain_oracle  max|.|=%.3e'
      % (lhs1 - df['gain_oracle']).abs().max())
lhs2 = df['gain_k5best'] + df['gap_k5best_oracle']
print('  k5best caliber: gain_k5best + gap_k5best_oracle - gain_oracle max|.|=%.3e'
      % (lhs2 - df['gain_oracle']).abs().max())
mixed = df['gain_fix5'] + df['gap_k5best_oracle']
print('  MIXED (wrong) : gain_fix5 + gap_k5best_oracle - gain_oracle  max|.|=%.3e'
      % (mixed - df['gain_oracle']).abs().max())
print('  mean values: gain_fix5=%.4f  gain_k5best=%.4f  gap_fix5=%.4f  '
      'gap_k5best=%.4f  gain_oracle=%.4f'
      % (df['gain_fix5'].mean(), df['gain_k5best'].mean(),
         df['gap_fix5_oracle'].mean(), df['gap_k5best_oracle'].mean(),
         df['gain_oracle'].mean()))
print('  report arithmetic: 1.6538+0.3741=%.4f (fix5, OK)   '
      '1.6588+0.3691=%.4f (k5best, OK)   1.6538+0.3691=%.4f (MIXED, off by %.4f)'
      % (1.6538 + 0.3741, 1.6588 + 0.3691, 1.6538 + 0.3691,
         1.6588 + 0.3691 - 1.6538 - 0.3691))
print('  k5best - fix5: mean=%.4f  n<0=%s  si=%s'
      % ((df['J_k5best'] - df['J_k5_fix5']).mean(),
         int((df['J_k5best'] - df['J_k5_fix5'] < -EPS).sum()),
         df.loc[df['J_k5best'] - df['J_k5_fix5'] < -EPS, 'si'].tolist()))
print('  gap_k5best_oracle < 0 snaps: %s (dual dir outside sweep grid)'
      % df.loc[df['gap_k5best_oracle'] < -EPS, 'si'].tolist())
print()

print('=== methods vs sweep eval-path cross-check ===')
print('  max|k5_bind(sweep-fix5)| = %.3e   (expect ~1e-13)'
      % (df['J_k5_bind'] - df['J_k5_fix5']).abs().max())
print('  max|J0(sweep) - J0(methods)| = %.3e' % df['J0_methods_vs_sweep'].abs().max())
print('  max|oracle - ext.J_ext| = %.3e'
      % np.abs(df['J_oracle'].values - ext.loc[df['si'].values, 'J_ext'].values).max())
print()

print('=== ADP2b frozen-rule recomputation vs report ===')
print('  calls distribution: %s   mean=%.4f (report: {3:16,4:17,5:7} mean 3.77)'
      % (df['calls_adp2b'].value_counts().sort_index().to_dict(),
         df['calls_adp2b'].mean()))
print('  gain_adp2b mean=%.4f (report 2.0279)' % df['gain_adp2b'].mean())
for eps, tag in ((EPS, '1e-9'), (EPS_TIE, '1e-6')):
    g5 = df['adp2b_minus_fix5'].values
    gd = df['adp2b_minus_k5dual'].values
    print('  eps=%s  vs fix5: win=%d tie=%d lose=%d mean=%.4f | '
          'vs k5dual: win=%d tie=%d lose=%d mean=%.4f sum_neg=%.4f'
          % (tag,
             int((g5 > eps).sum()), int((np.abs(g5) <= eps).sum()),
             int((g5 < -eps).sum()), g5.mean(),
             int((gd > eps).sum()), int((np.abs(gd) <= eps).sum()),
             int((gd < -eps).sum()), gd.mean(), gd[gd < -eps].sum()))
print('  report: vs fix5 6/33/1 (lose=1.5e-9)  vs k5dual 8/30/2 sum_neg=-0.0435'
      '  -> fix5@1e-9 and k5dual@1e-6 both reproduce')
print('  vs replay CSV adp2 column: max|diff|=%.3e (adp2 = same rule WITHOUT '
      'early-stop)' % (df.set_index('si')['J_adp2b'] - rep.loc[df['si'], 'adp2']).abs().max())
print()

print('=== oracle superset assertions ===')
viol = df[df['J_oracle'] > df['J_adp2b'] + 1e-6]
print('  oracle <= adp2b for all snaps: %s (violations: %d)'
      % (len(viol) == 0, len(viol)))
print('  oracle <= fix5 for all snaps: %s'
      % bool((df['J_oracle'] <= df['J_k5_fix5'] + 1e-6).all()))
below = df[df['J_oracle'] > df['J_k5best'] + 1e-6]
print('  oracle <= k5best for all snaps: %s  (failing si=%s, dual dir outside '
      'grid -> k5best NOT a subset of sweep)'
      % (len(below) == 0, below['si'].tolist()))
print()

print('=== call counts ===')
print('  fix5: 5 (stay + 2 dirs x rho 0.25/0.5)')
print('  adp2b: %s' % df['calls_adp2b'].value_counts().sort_index().to_dict())
print('  oracle: rows=%d, distinct candidates=11 (stay x2 deduped), '
      'valid rows=%d' % (12, int(df['n_valid'].iloc[0])))
print('  solve time from sweep: sum=%.2fs mean/cand=%.3fs'
      % (sw['solve_s'].sum(), sw['solve_s'].mean()))
print()

print('=== where calibers disagree (per-snap) ===')
d = df[(df['J_k5best'] < df['J_k5_fix5'] - EPS) |
       (df['gap_k5best_oracle'] < -EPS)]
print(d[['si', 'gain_fix5', 'gain_k5best', 'gap_fix5_oracle',
         'gap_k5best_oracle', 'gain_oracle']].to_string(index=False))
print()
print('written: task1_caliber.csv')

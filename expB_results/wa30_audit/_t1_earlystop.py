# -*- coding: utf-8 -*-
"""Task 1.4: early-stop risk of the frozen ADP2b rule (pure replay).

For the 40-snapshot W_A=30 design set (task2_step_sweep.csv + task1_caliber.csv):
  1. Branch analysis of the frozen rule (stop@3 / stop@4 / full@5).
  2. In the J25>=J0 branch (stop@3): count snaps where rho 0.1/0.5/0.75 still
     beats Stay (per direction) -> real early-stop leak.
  3. Direction reversal: does the 0.25-loser direction win at larger rho
     (vs Stay and vs the 0.25-winner at same rho)? ADP2b never evaluates it.
  4. Non-unimodality measured, NOT assumed: interior argmin counts, trend
     resume after a flat/rising step (J75<J50 when rule stopped at 4), and
     rho=0.1 (never evaluated by the rule) as global-argmin blind spot.
  5. ADP2b vs fix5 / k5dual win-tie-lose: no dominance claim.
"""
import os
import numpy as np
import pandas as pd

OUT = r'D:\Learn\work\drl-opencode\expB_results\wa30_audit'
EPS = 1e-9
EPS_TIE = 1e-6
RHOS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
DIRS = ('beta', 'bind')

sw = pd.read_csv(os.path.join(OUT, 'task2_step_sweep.csv'))
ca = pd.read_csv(os.path.join(OUT, 'task1_caliber.csv')).set_index('si')

rows = []
for si, g in sw.groupby('si'):
    g = g.set_index(['dir', 'rho'])
    J = {(d, r): float(g.loc[(d, r), 'J']) for d in DIRS for r in RHOS}
    val = {(d, r): int(g.loc[(d, r), 'valid']) for d in DIRS for r in RHOS}
    J0 = 0.5 * (J[('beta', 0.0)] + J[('bind', 0.0)])
    c = ca.loc[si]

    Jb, Jd = J[('beta', 0.25)], J[('bind', 0.25)]
    dstar = 'beta' if Jb <= Jd + EPS else 'bind'
    J25 = Jb if dstar == 'beta' else Jd
    dother = 'bind' if dstar == 'beta' else 'beta'
    calls = int(c['calls_adp2b'])
    branch = 'stop3' if calls == 3 else ('stop4' if calls == 4 else 'full5')

    adp2b, oracle = float(c['J_adp2b']), float(c['J_oracle'])
    min3 = min(J0, Jb, Jd)                       # stop@3 baseline value

    # --- 2. stop@3 branch: would any rho still beat Stay? (real leak) ---
    #    rho 0.1 is never evaluated by the rule at any branch.
    beat_stop3 = {}
    for rr in (0.1, 0.5, 0.75):
        for dd in DIRS:
            beat_stop3[(dd, rr)] = bool(
                val[(dd, rr)] and J[(dd, rr)] < min3 - EPS)

    # --- 3. direction reversal (0.25-loser at larger rho) ---
    rev_vs_stay = {}      # dother at rho beats Stay (min3)
    rev_vs_dstar = {}     # dother at rho beats dstar at same rho
    for rr in (0.5, 0.75):
        rev_vs_stay[rr] = bool(val[(dother, rr)]
                               and J[(dother, rr)] < min3 - EPS)
        rev_vs_dstar[rr] = bool(val[(dother, rr)] and val[(dstar, rr)]
                                and J[(dother, rr)] < J[(dstar, rr)] - EPS)

    # --- 4a. interior argmin per direction over the full valid grid ---
    interior = {}
    for dd in DIRS:
        rs = [r for r in RHOS if val[(dd, r)]]
        best = min(rs, key=lambda r: J[(dd, r)])
        interior[dd] = int(best not in (0.0, 0.75))   # 0.75 = furthest legal rho
    # rho=0.1 global-argmin blind spot
    g_pairs = [(J[(d, r)], d, r) for d in DIRS for r in RHOS if val[(d, r)]]
    g_pairs.sort()
    rho01_is_argmin = int(g_pairs[0][2] == 0.1)

    # trend profile: gain(rho)=J0-J(rho) along dstar & dother
    def nonmono(dd):
        seq = [J[(dd, r)] for r in (0.0, 0.1, 0.25, 0.5, 0.75) if val[(dd, r)]]
        # count rises (J larger = worse) followed by a later fall => non-unimodal
        d = np.diff(seq)                       # >0 = step got worse
        return int((d > EPS).any() and (d < -EPS).any())
    nm_dstar, nm_dother = nonmono(dstar), nonmono(dother)

    # trend resume in stop@4 branch: rule stopped because J50>=J25-EPS,
    # but does J75 improve on J50 (or on the evaluated best)?
    d50, d75 = J[(dstar, 0.5)], J[(dstar, 0.75)]
    resume75_vs50 = bool(val[(dstar, 0.75)] and d75 < d50 - EPS)
    resume75_vs_best = bool(val[(dstar, 0.75)] and d75 < adp2b - EPS)

    # --- 1. actual leak + stop@3 hypothetical (from caliber) ---
    leak = adp2b - oracle                       # >=0 expected; >EPS = real leak
    stop3_would_leak = min3 - oracle            # gain left on table if stop@3 always

    rows.append(dict(
        si=int(si), t=int(c['t']), branch=branch, calls=calls, dstar=dstar,
        dother=dother, improved25=int(c['adp2b_improved25']),
        J0=J0, J_adp2b=adp2b, J_oracle=oracle, leak=leak,
        min3=min3, stop3_would_leak=stop3_would_leak,
        ext_contrib=min3 - adp2b,               # gain added by calls 4-5
        n_beat_stop3=int(sum(beat_stop3.values())),
        beat_stop3_b10=int(beat_stop3[('beta', 0.1)]),
        beat_stop3_d10=int(beat_stop3[('bind', 0.1)]),
        beat_stop3_b50=int(beat_stop3[('beta', 0.5)]),
        beat_stop3_d50=int(beat_stop3[('bind', 0.5)]),
        beat_stop3_b75=int(beat_stop3[('beta', 0.75)]),
        beat_stop3_d75=int(beat_stop3[('bind', 0.75)]),
        rev_vs_stay_50=int(rev_vs_stay[0.5]), rev_vs_stay_75=int(rev_vs_stay[0.75]),
        rev_vs_dstar_50=int(rev_vs_dstar[0.5]), rev_vs_dstar_75=int(rev_vs_dstar[0.75]),
        interior_dstar=interior[dstar], interior_dother=interior[dother],
        rho01_is_argmin=rho01_is_argmin,
        nonmono_dstar=nm_dstar, nonmono_dother=nm_dother,
        resume75_vs50=int(resume75_vs50), resume75_vs_best=int(resume75_vs_best),
        adp2b_minus_fix5=float(c['adp2b_minus_fix5']),
        adp2b_minus_k5dual=float(c['adp2b_minus_k5dual']),
        stop_leak_caliber=float(c['stop_leak']),
        other_dir_50=float(c['other_dir_50']),
    ))

df = pd.DataFrame(rows).sort_values('si').reset_index(drop=True)
df.to_csv(os.path.join(OUT, 'task1_earlystop.csv'), index=False)
n = len(df)

print('=== task 1.4 early-stop risk: n=%d snapshots ===' % n)
print()
print('[1] frozen-rule branches: %s'
      % df['branch'].value_counts().reindex(['stop3', 'stop4', 'full5'])
      .dropna().astype(int).to_dict())
print('    (report calls dist {3:16,4:17,5:7}) match: %s'
      % (df['calls'].value_counts().sort_index().to_dict()
         == {3: 16, 4: 17, 5: 7}))
print()

lk = df['leak']
print('[2] ACTUAL leak (J_adp2b - J_oracle): >1e-9: %d | >1e-6: %d | '
      'max=%.3e (si=%s) | sum=%.3e'
      % (int((lk > EPS).sum()), int((lk > EPS_TIE).sum()),
         lk.max(), int(df.loc[lk.idxmax(), 'si']), lk.sum()))
if (lk > EPS).any():
    print('    leaking snaps: %s'
          % df.loc[lk > EPS, ['si', 'branch', 'leak']]
              .to_dict('records'))
print('    -> early stop is lossless on this set (up to %.2e numerical)' % lk.max())
print()

s3 = df[df['branch'] == 'stop3']
print('[3] stop@3 branch (J25>=J0), n=%d:' % len(s3))
print('    any unevaluated rho (0.1/0.5/0.75, both dirs) beats the stop@3 '
      'baseline: %d snaps' % int((s3['n_beat_stop3'] > 0).sum()))
for col, lab in (('beat_stop3_b10', 'beta@0.1'), ('beat_stop3_d10', 'bind@0.1'),
                 ('beat_stop3_b50', 'beta@0.5'), ('beat_stop3_d50', 'bind@0.5'),
                 ('beat_stop3_b75', 'beta@0.75'), ('beat_stop3_d75', 'bind@0.75')):
    print('      beats Stay @ %s : %d' % (lab, int(s3[col].sum())))
print('    stop3_would_leak (grid gain missed IF rule always stopped at 3): '
      'mean=%.4f max=%.4f  >1e-9: %d snaps'
      % (s3['stop3_would_leak'].mean(), s3['stop3_would_leak'].max(),
         int((s3['stop3_would_leak'] > EPS).sum())))
print()

print('[4] direction reversal (0.25-loser dir, never evaluated at rho>0.25 '
      'by ADP2b), all %d snaps:' % n)
print('    dother@0.5 beats stop@3 baseline : %d | beats dstar@0.5 : %d '
      '(mean J diff other-star @0.5 = %+.4f)'
      % (int(df['rev_vs_stay_50'].sum()), int(df['rev_vs_dstar_50'].sum()),
         df['other_dir_50'].mean()))
print('    dother@0.75 beats stop@3 baseline: %d | beats dstar@0.75 : %d'
      % (int(df['rev_vs_stay_75'].sum()), int(df['rev_vs_dstar_75'].sum())))
print('    reversal that changes the answer (beats ADP2b result): %d snaps'
      % int((df['rev_vs_stay_50'] & (df['branch'] == 'stop3')).sum()
            + (df['rev_vs_stay_75'] & (df['branch'] == 'stop3')).sum()))
print()

print('[5] non-unimodality measured (no monotonicity assumed):')
print('    interior argmin (best rho in {0.1,0.25,0.5}) on full valid grid: '
      'dstar-dir=%d/%d  dother-dir=%d/%d snaps'
      % (int(df['interior_dstar'].sum()), n,
         int(df['interior_dother'].sum()), n))
print('    J(rho) has both rises and falls (non-unimodal profile): '
      'dstar-dir=%d/%d  dother-dir=%d/%d'
      % (int(df['nonmono_dstar'].sum()), n,
         int(df['nonmono_dother'].sum()), n))
print('    rho=0.1 (never evaluated by rule) is the global argmin: %d snaps'
      % int(df['rho01_is_argmin'].sum()))
s4 = df[df['branch'] == 'stop4']
print('    stop@4 branch n=%d: rule stopped (J50>=J25) but J75<J50: %d | '
      'J75 beats evaluated best: %d'
      % (len(s4), int(s4['resume75_vs50'].sum()),
         int(s4['resume75_vs_best'].sum())))
print('    trend extension (calls 4-5) contributed: mean=%.4f  '
      'total=%.4f  >1e-9 in %d snaps'
      % (df['ext_contrib'].mean(), df['ext_contrib'].sum(),
         int((df['ext_contrib'] > EPS).sum())))
print()

print('[6] ADP2b vs fixed baselines -- NO dominance claim (counts only):')
for col, ref in (('adp2b_minus_fix5', 'fix5'), ('adp2b_minus_k5dual', 'k5dual')):
    for eps, tag in ((EPS, '1e-9'), (EPS_TIE, '1e-6')):
        v = df[col].values
        w, t, l = int((v > eps).sum()), int((np.abs(v) <= eps).sum()), \
                  int((v < -eps).sum())
        lose_si = df.loc[v < -eps, 'si'].tolist()
        print('    eps=%s vs %-8s win=%d tie=%d lose=%d  mean=%+.4f  '
              'lose si=%s' % (tag, ref, w, t, l, v.mean(), lose_si))
print('    (ADP2b can be worse than fix5/k5dual on individual snaps; '
      'dual dir is outside the sweep grid for k5dual)')
print()
print('written: task1_earlystop.csv')

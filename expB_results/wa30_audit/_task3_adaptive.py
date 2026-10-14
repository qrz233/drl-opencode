# -*- coding: utf-8 -*-
"""Task 3: adaptive step-length search within a 5-call Block A budget.

All policies are replayed offline on the task-2 sweep grid
(dirs {beta, bind} x rho {0, 0.1, 0.25, 0.5, 0.75}; rho=1.0 power-infeasible),
so every query an adaptive policy makes already has a stored J -> exact
evaluation with zero extra Block A cost.

Policies (all retain Stay as call 1, and beta+bind dirs as calls 2-3):
  FIX5   fixed K5-binding: {stay, b25, b50, bd25, bd50}          (5 calls)
  ADP1   shrink-extend bracket:
           4: J25<J0 ? d*@0.75 : d*@0.1
           5: (0.75 & improved) ? d*@0.5 : (0.75 & worse) ? d*@0.1 : stop
  ADP2   trend-extend:
           4: d*@0.5 ; 5: J50<J25 ? d*@0.75 : stop              (<=5 calls)
  ADP3   both-tails: 4: d*@0.75 ; 5: d*@0.1                     (5 calls)
  ORACLE min over all valid sweep candidates                    (11 calls)
d* = direction with lower J at rho=0.25 (tie -> beta).
"""
import os, json
import numpy as np
import pandas as pd

OUT = r'D:\Learn\work\drl-opencode\expB_results\wa30_audit'
df = pd.read_csv(os.path.join(OUT, 'task2_step_sweep.csv'))
t2 = pd.read_csv(os.path.join(OUT, 'task2_wa30_methods.csv')).set_index('si')

v = df[df['valid'] == 1]
# J lookup: J[si][dir][rho]
J = {}
for si, grp in v.groupby('si'):
    J[si] = {(r['dir'], r['rho']): r['J'] for _, r in grp.iterrows()}

EPS = 1e-9
rows = []
for si, m in J.items():
    J0 = m[('beta', 0.0)]  # stay (same for both dirs)
    Jb, Jd = m[('beta', 0.25)], m[('bind', 0.25)]
    if Jb <= Jd + EPS:
        dstar, J25 = 'beta', Jb
    else:
        dstar, J25 = 'bind', Jd
    dother = 'bind' if dstar == 'beta' else 'beta'
    improved25 = J25 < J0 - EPS
    base = min(J0, Jb, Jd)

    # FIX5
    fix5 = min(J0, Jb, Jd, m[('beta', 0.5)], m[('bind', 0.5)])

    # ADP1: shrink-extend bracket
    ap1 = [J0, Jb, Jd]
    r1 = []
    if improved25:
        j = m[(dstar, 0.75)]; ap1.append(j); r1.append(0.75)
        if j < J25 - EPS:
            ap1.append(m[(dstar, 0.5)]); r1.append(0.5)
        else:
            ap1.append(m[(dstar, 0.1)]); r1.append(0.1)
    else:
        ap1.append(m[(dstar, 0.1)]); r1.append(0.1)
        # if 0.1 did not beat stay either -> stay is best, stop at 4 calls
        if m[(dstar, 0.1)] < J0 - EPS:
            pass  # 0.1 improved over stay but not over 0.25; no useful 5th probe
    adp1, n1 = min(ap1), len(ap1)

    # ADP2: trend-extend
    ap2 = [J0, Jb, Jd, m[(dstar, 0.5)]]
    if m[(dstar, 0.5)] < J25 - EPS:
        ap2.append(m[(dstar, 0.75)])
    adp2, n2 = min(ap2), len(ap2)

    # ADP3: both tails from 0.25
    ap3 = [J0, Jb, Jd, m[(dstar, 0.75)], m[(dstar, 0.1)]]
    adp3, n3 = min(ap3), len(ap3)

    oracle = min(m.values())
    rows.append(dict(
        si=si, t=t2.loc[si, 't'], qlen=t2.loc[si, 'qlen'],
        coexist=int(t2.loc[si, 'base_sim_uav'] > 0),
        acc_bind=int(t2.loc[si, 'base_pool_acc_bind']),
        J0=J0, fix5=fix5, adp1=adp1, adp2=adp2, adp3=adp3, oracle=oracle,
        n1=n1, n2=n2, n3=n3,
        dstar=dstar, rho25_improved=int(improved25),
        gap_fix5_oracle=fix5 - oracle,
        gap_adp1_fix5=fix5 - adp1,
        gap_adp2_fix5=fix5 - adp2,
        gap_adp3_fix5=fix5 - adp3,
        gap_adp1_oracle=adp1 - oracle,
    ))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'task3_adaptive_replay.csv'), index=False)


def d(x):
    x = np.asarray(x, float)
    return ('mean=%.4f med=%.4f p90=%.4f max=%.4f' %
            (x.mean(), np.median(x), np.percentile(x, 90), x.max()))


print('=== gains vs Stay (dJ = J0 - Jpolicy, 40 snaps) ===')
for c in ('fix5', 'adp1', 'adp2', 'adp3', 'oracle'):
    print('  %-7s %s' % (c, d(res['J0'] - res[c])))
print()
print('=== vs fixed K5 (positive = adaptive better) ===')
for c in ('adp1', 'adp2', 'adp3'):
    g = res['gap_%s_fix5' % c]
    print('  %-5s %s  win=%d tie=%d lose=%d  calls(mean)=%.2f'
          % (c, d(g), int((g > EPS).sum()), int((abs(g) <= EPS).sum()),
             int((g < -EPS).sum()), res['n%d' % int(c[-1])].mean()))
print()
print('=== oracle gap captured (fix5 gap to oracle vs adp gap) ===')
print('  fix5  %s  sum=%.3f' % (d(res['gap_fix5_oracle']), res['gap_fix5_oracle'].sum()))
print('  adp1  %s  sum=%.3f  captured=%.1f%%'
      % (d(res['gap_adp1_oracle']), res['gap_adp1_oracle'].sum(),
         100 * (1 - res['gap_adp1_oracle'].sum() / res['gap_fix5_oracle'].sum())))
print('  adp2  captured=%.1f%%' %
      (100 * (1 - (res['adp2'] - res['oracle']).clip(lower=0).sum()
              / res['gap_fix5_oracle'].sum())))
print()
print('=== where does fix5 miss? (gap_fix5_oracle>0 snaps) ===')
miss = res[res['gap_fix5_oracle'] > EPS]
print('n=%d  coexist=%d acc_bind=%d t<=40=%d' %
      (len(miss), miss['coexist'].sum(), miss['acc_bind'].sum(),
       (miss['t'] <= 40).sum()))
print(miss[['si', 't', 'coexist', 'gap_fix5_oracle', 'gap_adp1_fix5',
            'gap_adp2_fix5', 'gap_adp3_fix5']].to_string(index=False))
print()
print('=== ADP1 regret detail vs oracle ===')
r1 = res[res['gap_adp1_oracle'] > EPS]
print(r1[['si', 'coexist', 'J0', 'adp1', 'oracle', 'gap_adp1_oracle']].to_string(index=False))
print()
# coexist subset
co = res['coexist'] == 1
print('=== coexist subset (n=%d): dJ vs stay ===' % co.sum())
for c in ('fix5', 'adp1', 'adp2', 'adp3', 'oracle'):
    print('  %-7s %s' % (c, d((res.loc[co, 'J0'] - res.loc[co, c]))))
print('  fix5-oracle gap:', d(res.loc[co, 'gap_fix5_oracle']))
print('  adp1 - fix5    :', d(res.loc[co, 'gap_adp1_fix5']))

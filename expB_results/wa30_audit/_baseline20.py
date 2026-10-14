# -*- coding: utf-8 -*-
"""Baselines from prior runs: 20 MHz airship dU + 20 MHz binding stats."""
import os
import pandas as pd

SA = r'D:\Learn\work\drl-opencode\expB_results\scenario_audit'
d3 = pd.read_csv(os.path.join(SA, 'task3_airship_geom.csv'))
g0 = d3[d3['geom'] == 'G0']
print('=== 20 MHz airship (G0, scenario_audit task3) ===')
print('dU: mean=%.6f med=%.6f max=%.6f move>0 rate=%.3f'
      % (g0['dU'].mean(), g0['dU'].median(), g0['dU'].max(), (g0['dU'] > 1e-9).mean()))
print('U0 mean=%.3f' % g0['U0'].mean())

d1 = pd.read_csv(os.path.join(SA, 'task1_audit.csv'))
print('\n=== 20 MHz baseline (task1_audit n=%d) columns ===' % len(d1))
print(d1.columns.tolist())
for c in d1.columns:
    lc = c.lower()
    if any(k in lc for k in ('acc', 'bh', 'bind', 'dual', 'nu', 'mu')):
        try:
            print('  %-24s mean=%.4f' % (c, float(d1[c].mean())))
        except Exception:
            pass

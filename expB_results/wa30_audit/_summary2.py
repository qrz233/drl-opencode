# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
OUT = r'D:\Learn\work\drl-opencode\expB_results\wa30_audit'
d = pd.read_csv(OUT + r'\task2_wa30_methods.csv')
bd = d['bind_minus_dual']
print('max|bind-dual| = %.4f  min=%.4f max=%.4f' % (bd.abs().max(), bd.min(), bd.max()))
print('acc-binding snaps:')
ab = d[d.base_pool_acc_bind == 1]
print(ab[['si', 'dJ_k3', 'dJ_bind', 'dJ_dual', 'dJ_acc', 'dJ_dual_air',
          'air_gain_over_uav']].to_string(index=False))
print('mean dJ acc-binding: dual=%.3f k3=%.3f' % (ab.dJ_dual.mean(), ab.dJ_k3.mean()))
print('mean dJ non-binding: dual=%.3f k3=%.3f'
      % (d[d.base_pool_acc_bind == 0].dJ_dual.mean(),
         d[d.base_pool_acc_bind == 0].dJ_k3.mean()))
print('median J0=%.2f' % d.J0.median())
print('n_mu_pos on acc-binding snaps: %s' % ab.base_n_mu.tolist())
# relative scale: report dJ / P0 (energy term) and dJ vs beta*D term? use dJ/P0 as percent
print('dJ_dual/P0 mean=%.4f%%' % (100 * (d.dJ_dual / d.P0).mean()))

# -*- coding: utf-8 -*-
import os, sys
import numpy as np
PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
from params_sys import *
from utils_sys import cal_channel_fading_gain
import blocka_optimizer as B
MHZ = B.MHZ; TAU = float(slot_len)
z = np.load(r'D:\Learn\work\drl-opencode\expA_results\snapshots.npz', allow_pickle=True)
bd = np.load(r'D:\Learn\work\drl-opencode\expA_results\base_diagnostics.npz', allow_pickle=True)
print('base_diagnostics keys:', list(bd.files))
demand = np.maximum(z['D'] + z['dem_slack'], 0.0)

si = 10
cids = z['cids'][si]; beta = z['beta'][si]
g = np.zeros(100)
for u in np.unique(cids):
    ids = np.where(cids == u)[0]
    g[ids] = cal_channel_fading_gain(z['user_x'][si][ids], z['user_y'][si][ids],
                                     z['locs_dec'][si][u,0], z['locs_dec'][si][u,1], z['locs_dec'][si][u,2])
gb, _ = B.cal_gamma_backhaul_MHz(z['locs_dec'][si], z['air_dec'][si])
ga = B.cal_gamma_access_MHz(g)

# t index in expA full-slot arrays
t_full = int(z['slots'][si])
print(f'\nsi={si}  slot t={t_full}')
print('expA stored: acc_resid =', bd['acc_resid'][t_full], ' bh_resid =', bd['bh_resid'][t_full],
      ' status =', bd['base_status'][t_full], ' solver =', bd['base_solver'][t_full])
print('expA stored D[si,14] =', z['D'][si][14], ' my demand[si,14] =', demand[si][14])

r = B.solve_blocka(ga, gb, demand[si], cids, beta, W_U_MHz=bw_total_Hz/MHZ,
                   W_A_MHz=20.0, tau=TAU)
print('my status =', r['status'])
print('my D[14] =', r['D_Mb'][14], ' my W_ui[14] MHz =', r['W_ui_Hz'][14]/MHZ,
      ' my R_acc[14] =', r['R_access_Mb'][14])
print('g[14] =', g[14], ' gamma_acc_MHz[14] =', ga[14])
# true concave rate at the returned W
w = r['W_ui_Hz'][14]/MHZ
print('recomputed tau*W*log2(1+G/W) with W=%.3e -> %.6f' % (w, TAU*w*np.log2(1+ga[14]/max(w,1e-12))))
print('limit as W->0: tau*G/ln2 = %.6f Mb' % (TAU*ga[14]/np.log(2)))
print('sum W (MHz) =', r['W_ui_Hz'].sum()/MHZ)
print('stored D at si=10 vs mine: max|diff| =', np.max(np.abs(z['D'][si] - r['D_Mb'])),
      ' at user', int(np.argmax(np.abs(z['D'][si]-r['D_Mb']))))
print('expA D_diag[si,14] =', bd['D'][t_full][14])
print('expA acc_slack[si,14] =', bd['acc_slack'][t_full][14])

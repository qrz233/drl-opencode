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

si = 0
cids = z['cids'][si]; beta = z['beta'][si]; types = z['types'][si]
locs = z['locs_dec'][si]; air = z['air_dec'][si]
Q = z['Q'][si]; A = z['A'][si]
demand = np.maximum(Q, 0) + np.maximum(A, 0)
recon = z['D'][si] + z['dem_slack'][si]
active = (demand > 0) & (types >= 0) & (beta > 0)
print('slot t =', z['slots'][si])
print('n active =', int(active.sum()), ' cids uniq =', np.unique(cids))
print('beta>0 count =', int((beta > 0).sum()), ' types>=0 count =', int((types >= 0).sum()))
print('demand sum =', demand.sum(), ' recon sum =', recon.sum())
print('inactive-with-demand users =', int(((demand > 0) & ~active).sum()))
print('beta of inactive-with-demand (first 5):', beta[(demand > 0) & ~active][:5])

# my gammas
g = np.zeros(len(cids))
for u in np.unique(cids):
    ids = np.where(cids == u)[0]
    g[ids] = cal_channel_fading_gain(z['user_x'][si][ids], z['user_y'][si][ids], locs[u,0], locs[u,1], locs[u,2])
gb, _ = B.cal_gamma_backhaul_MHz(locs, air)
r = B.solve_blocka(B.cal_gamma_access_MHz(g), gb, demand, cids, beta,
                   W_U_MHz=bw_total_Hz/MHZ, W_A_MHz=bw_bh_total_Hz/MHZ, tau=TAU)
print('\nmy status =', r['status'])
print('my  D sum =', r['D_Mb'].sum(), ' stored D sum =', z['D'][si].sum())
print('my  D[:8] =', np.round(r['D_Mb'][:8], 4))
print('sto D[:8] =', np.round(z['D'][si][:8], 4))
print('my  R_bh =', np.round(r['R_bh_Mb'], 4), ' stored C =', np.round(z['C'][si], 4))
print('my  S_u  =', np.round(r['S_u'], 4), ' stored S =', np.round(z['S'][si], 4))
print('my  W_ua/1e6 =', np.round(r['W_ua_Hz']/1e6, 4))
print('stored nu =', np.round(z['nu'][si], 6))

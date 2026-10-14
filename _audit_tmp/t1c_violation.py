# -*- coding: utf-8 -*-
"""Pin down the Block A solver-fallback primal violation + Q snapshot semantics."""
import os, sys
import numpy as np
PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
from params_sys import *
from utils_sys import cal_channel_fading_gain
import blocka_optimizer as B
MHZ = B.MHZ; TAU = float(slot_len)
z = np.load(r'D:\Learn\work\drl-opencode\expA_results\snapshots.npz', allow_pickle=True)
S = len(z['slots'])
demand = np.maximum(z['D'] + z['dem_slack'], 0.0)

# ---- Q semantics ----
Q = z['Q']; A = z['A']; D = z['D']
print('[Q-semantics] max|Q_stored - (demand - A)| =',
      np.max(np.abs(Q - (demand - A))), ' (Q_prev hypothesis)')
print('[Q-semantics] max|Q_stored - max(demand-D,0)| =',
      np.max(np.abs(Q - np.maximum(demand - D, 0))), ' (post-update w/ diag D)')
print('[Q-semantics] max|Q_stored - (demand-A)| on si=0 =',
      np.max(np.abs(Q[0] - (demand[0] - A[0]))))

def gammas(ux, uy, cids, locs, air):
    g = np.zeros(len(cids))
    for u in np.unique(cids):
        ids = np.where(cids == u)[0]
        g[ids] = cal_channel_fading_gain(ux[ids], uy[ids], locs[u,0], locs[u,1], locs[u,2])
    gb, _ = B.cal_gamma_backhaul_MHz(locs, air)
    return B.cal_gamma_access_MHz(g), gb

# ---- find violating snapshots, with solver attribution ----
for si in range(S):
    cids = z['cids'][si]; beta = z['beta'][si]
    ga, gb = gammas(z['user_x'][si], z['user_y'][si], cids, z['locs_dec'][si], z['air_dec'][si])
    for WA, tag in ((20.0, '20MHz'), (30.0, '30MHz')):
        r = B.solve_blocka(ga, gb, demand[si], cids, beta, W_U_MHz=bw_total_Hz/MHZ,
                           W_A_MHz=WA, tau=TAU)
        Dv = r['D_Mb']
        sw = r['W_ui_Hz'].sum()/MHZ
        acc_v = float((Dv - r['R_access_Mb']).max())
        if sw - 40.0 > 1e-6 or acc_v > 1e-6:
            u = int(np.argmax(Dv - r['R_access_Mb']))
            print(f'\n[VIOL] si={si} t={int(z["slots"][si])} {tag} status={r["status"]}')
            print(f'   sum(W)={sw:.6f} MHz  excess={sw-40.0:.3e}')
            print(f'   max(D - tau*R_access)={acc_v:.6e} Mb  at user {u} '
                  f'(D={Dv[u]:.6f}, Racc={r["R_access_Mb"][u]:.6f})')
            print(f'   n_active={int(((demand[si]>0)&(cids>=0)&(beta>0)).sum())}')

# -*- coding: utf-8 -*-
"""Independent audit test 1+2 (v2): correct demand = D + dem_slack."""
import os, sys
import numpy as np
import pandas as pd
PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
from params_sys import *
from utils_sys import cal_channel_fading_gain, cal_uav_propulsion_energy
import blocka_optimizer as B
MHZ = B.MHZ; TAU = float(slot_len); PSI_E = 1.0 / PW_MAX
z = np.load(r'D:\Learn\work\drl-opencode\expA_results\snapshots.npz', allow_pickle=True)
S = len(z['slots'])

D = z['D']; ds = z['dem_slack']; Q = z['Q']; A = z['A']
demand = np.maximum(D + ds, 0.0)              # expB's reconstruction

# --- is stored Q the post-update queue? test Q == max(demand - D, 0) ---
pred_post = np.maximum(demand - D, 0.0)
print(f'[T0a] max|Q_stored - max(demand-D,0)| = {np.max(np.abs(Q - pred_post)):.3e}  '
      f'(small => Q_stored is Q(t+1))')

def gammas(ux, uy, cids, locs, air):
    g = np.zeros(len(cids))
    for u in np.unique(cids):
        ids = np.where(cids == u)[0]
        g[ids] = cal_channel_fading_gain(ux[ids], uy[ids], locs[u,0], locs[u,1], locs[u,2])
    gb, _ = B.cal_gamma_backhaul_MHz(locs, air)
    return B.cal_gamma_access_MHz(g), gb

viol = dict(demand=0, bh_uav=0, acc_rate=0, pool_acc=0, pool_bh=0, neg=0)
mx = dict(dem=0.0, bh=0.0, pa=0.0, pb=0.0, acc=0.0)
statuses = {}
J0_mine = {}
Dmax20 = 0.0
for si in range(S):
    cids = z['cids'][si]; beta = z['beta'][si]
    locs = z['locs_dec'][si]; air = z['air_dec'][si]; dem = demand[si]
    ga, gb = gammas(z['user_x'][si], z['user_y'][si], cids, locs, air)
    r = B.solve_blocka(ga, gb, dem, cids, beta, W_U_MHz=bw_total_Hz/MHZ,
                       W_A_MHz=bw_bh_total_Hz/MHZ, tau=TAU)
    statuses[r['status']] = statuses.get(r['status'], 0) + 1
    Dv = r['D_Mb']
    Dmax20 = max(Dmax20, float(np.max(np.abs(Dv - D[si]))))
    if (Dv < -1e-12).any(): viol['neg'] += 1
    e = (Dv - dem).max()
    if e > 1e-6: viol['demand'] += 1; mx['dem'] = max(mx['dem'], float(e))
    for u in range(n_uavs):
        su = float(Dv[cids == u].sum()); cap = TAU * r['R_bh_Mb'][u]
        if su - cap > 1e-6: viol['bh_uav'] += 1; mx['bh'] = max(mx['bh'], su - cap)
    ea = (Dv - r['R_access_Mb']).max()
    if ea > 1e-6: viol['acc_rate'] += 1; mx['acc'] = max(mx['acc'], float(ea))
    sw = r['W_ui_Hz'].sum()/MHZ; swb = r['W_ua_Hz'].sum()/MHZ
    if sw - 40.0 > 1e-6: viol['pool_acc'] += 1; mx['pa'] = max(mx['pa'], sw)
    if swb - 20.0 > 1e-6: viol['pool_bh'] += 1; mx['pb'] = max(mx['pb'], swb)
    P = np.full(n_uavs, float(cal_uav_propulsion_energy(V=0.0, Vz=0.0)[0]))
    J0_mine[si] = PSI_E * float(z['E_dec'][si] @ P) - float(beta @ Dv)

print(f'\n[T1] statuses (W_A=20, decision geom, correct demand): {statuses}')
print(f'[T1] violations: {viol}   max excess {mx}')
print(f'[T1] max|myD - storedD| @20MHz = {Dmax20:.3e}')

# --- T2: J0 vs expB CSV (W_A=30) ---
J0_30 = {}
for si in range(S):
    cids = z['cids'][si]; beta = z['beta'][si]
    ga, gb = gammas(z['user_x'][si], z['user_y'][si], cids, z['locs_dec'][si], z['air_dec'][si])
    r = B.solve_blocka(ga, gb, demand[si], cids, beta, W_U_MHz=bw_total_Hz/MHZ,
                       W_A_MHz=30.0, tau=TAU)
    P = np.full(n_uavs, float(cal_uav_propulsion_energy(V=0.0, Vz=0.0)[0]))
    J0_30[si] = PSI_E * float(z['E_dec'][si] @ P) - float(beta @ r['D_Mb'])

csv = pd.read_csv(r'D:\Learn\work\drl-opencode\expB_results\wa30_audit\task2_wa30_methods.csv')
mine = np.array([J0_30[int(s)] for s in csv['si']])
diff = np.abs(mine - csv['J0'].to_numpy())
print(f'\n[T2] my stay-J(30MHz) vs expB CSV J0: max|diff|={diff.max():.3e}  mean={diff.mean():.3e}')
print(f'     expB J0 range [{csv["J0"].min():.3f}, {csv["J0"].max():.3f}]')

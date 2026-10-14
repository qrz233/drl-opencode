# -*- coding: utf-8 -*-
"""Independent audit test 1+2: Block A feasibility & J reproducibility.

Read-only. Does NOT touch production code or existing results.
"""
import os, sys
import numpy as np
import pandas as pd

PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
from params_sys import *
from utils_sys import cal_channel_fading_gain, cal_uav_propulsion_energy
import blocka_optimizer as B

MHZ = B.MHZ
TAU = float(slot_len)
PSI_E = 1.0 / PW_MAX
W_A30 = 30.0
z = np.load(r'D:\Learn\work\drl-opencode\expA_results\snapshots.npz', allow_pickle=True)
S = len(z['slots'])
print(f'loaded {S} snapshots\n')

# ---- check demand reconstruction: D + dem_slack == Q + A ? ----
D = z['D']; ds = z['dem_slack']; Q = z['Q']; A = z['A']
recon = D + ds
target = np.maximum(Q, 0) + np.maximum(A, 0)
err = np.max(np.abs(recon - target))
print(f'[T0] max|(D+dem_slack) - (Q+A)| = {err:.3e}')
active_snap = z['active']
active_calc = (z['types'] >= 0) & (target > 0)
print(f'[T0] active mask mismatch count = {int((active_snap != active_calc).sum())}')
print(f'[T0] dem_slack min = {ds.min():.3e}  (negative would mean D>demand)')

# ---- T1: independent Block A at decision geometry, W_A=30, feasibility ----
def gammas(ux, uy, cids, locs, air):
    g = np.zeros(len(cids))
    for u in np.unique(cids):
        ids = np.where(cids == u)[0]
        g[ids] = cal_channel_fading_gain(ux[ids], uy[ids], locs[u, 0], locs[u, 1], locs[u, 2])
    gb, _ = B.cal_gamma_backhaul_MHz(locs, air)
    return B.cal_gamma_access_MHz(g), gb, g

viol = dict(demand=0, bh_uav=0, acc_uav=0, pool_acc=0, pool_bh=0, acc_rate=0, neg=0)
max_dem_excess = 0.0; max_bh_excess = 0.0; max_pool_acc = 0.0; max_pool_bh = 0.0
statuses = {}
J0_mine = {}
for si in range(S):
    cids = z['cids'][si]; beta = z['beta'][si]; types = z['types'][si]
    locs = z['locs_dec'][si]; air = z['air_dec'][si]
    demand = target[si]
    ga, gb, _ = gammas(z['user_x'][si], z['user_y'][si], cids, locs, air)
    r = B.solve_blocka(ga, gb, demand, cids, beta,
                       W_U_MHz=bw_total_Hz / MHZ, W_A_MHz=W_A30, tau=TAU)
    statuses[r['status']] = statuses.get(r['status'], 0) + 1
    Dv = r['D_Mb']
    if (Dv < -1e-12).any():
        viol['neg'] += 1
    ex = Dv - demand
    if ex.max() > 1e-6:
        viol['demand'] += 1; max_dem_excess = max(max_dem_excess, float(ex.max()))
    # per-UAV backhaul
    for u in range(n_uavs):
        su = float(Dv[cids == u].sum())
        cap = TAU * r['R_bh_Mb'][u]
        if su - cap > 1e-6:
            viol['bh_uav'] += 1; max_bh_excess = max(max_bh_excess, su - cap)
    # access rate per user
    if (Dv - r['R_access_Mb'] > 1e-6).any():
        viol['acc_rate'] += 1
    # pools
    sw = r['W_ui_Hz'].sum() / MHZ
    swb = r['W_ua_Hz'].sum() / MHZ
    if sw - bw_total_Hz / MHZ > 1e-6:
        viol['pool_acc'] += 1; max_pool_acc = max(max_pool_acc, sw)
    if swb - W_A30 > 1e-6:
        viol['pool_bh'] += 1; max_pool_bh = max(max_pool_bh, swb)
    # J at stay (my own formula)
    P = np.array([float(cal_uav_propulsion_energy(V=0.0, Vz=0.0)[0]) for _ in range(n_uavs)])
    J0_mine[si] = PSI_E * float(z['E_dec'][si] @ P) - float(beta @ Dv)

print(f'\n[T1] statuses (W_A=30, decision geom): {statuses}')
print(f'[T1] violations: {viol}')
print(f'[T1] max demand excess={max_dem_excess:.3e}  max bh/uav excess={max_bh_excess:.3e}')
print(f'[T1] max pool_acc used={max_pool_acc:.4f} (<=40)  max pool_bh used={max_pool_bh:.4f} (<=30)')

# ---- T2: compare my stay-J (W_A=30) with expB task2 CSV J0 ----
csv = pd.read_csv(r'D:\Learn\work\drl-opencode\expB_results\wa30_audit\task2_wa30_methods.csv')
mine = []
for _, row in csv.iterrows():
    mine.append(J0_mine[int(row['si'])])
mine = np.array(mine)
diff = np.abs(mine - csv['J0'].to_numpy())
print(f'\n[T2] compare my stay-J vs expB J0 (40 snapshots, W_A=30):')
print(f'     max|diff| = {diff.max():.3e}   mean = {diff.mean():.3e}')

# ---- T3: compare my stay-D with stored snapshot D (20 MHz baseline) ----
r20 = []
for si in range(S):
    ga, gb, _ = gammas(z['user_x'][si], z['user_y'][si], z['cids'][si], z['locs_dec'][si], z['air_dec'][si])
    r = B.solve_blocka(ga, gb, target[si], z['cids'][si], z['beta'][si],
                       W_U_MHz=bw_total_Hz / MHZ, W_A_MHz=bw_bh_total_Hz / MHZ, tau=TAU)
    r20.append(np.max(np.abs(r['D_Mb'] - D[si])))
r20 = np.array(r20)
print(f'\n[T3] my 20MHz stay-D vs stored snapshot D: max|diff|={r20.max():.3e} mean={r20.mean():.3e}')
print(f'     (snapshots generated at 20 MHz default W_A)')

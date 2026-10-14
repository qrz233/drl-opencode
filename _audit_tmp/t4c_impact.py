# -*- coding: utf-8 -*-
"""Quantify impact of unseeded KMeans label permutation on the Stationary baseline."""
import os, sys
import numpy as np
import pandas as pd
PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
from params_sys import *
from utils_sys import kmeans_clustering, cal_channel_fading_gain

sim_in = os.path.join(PROJ, 'simulation', 'sim-input')
def from_exel(f):
    return np.transpose(pd.read_excel(f, index_col=0).to_numpy())
xa = from_exel(os.path.join(sim_in, f"xlocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx"))
ya = from_exel(os.path.join(sim_in, f"ylocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx"))

# Stationary: UAVs never move, stay at init corners, z = z_init
uav_xy = init_locations_uav.astype(float)
z0 = float(z_init)
print('Stationary UAV corners:', uav_xy.tolist())

for t in [4, 20, 100, 300, 440]:
    loc = np.hstack((xa[:, [t]], ya[:, [t]], np.zeros((n_users, 1))))
    res = {}
    for s in (0, 7, 12345):
        np.random.seed(s)
        cids = kmeans_clustering(loc)[1]
        g = np.zeros(n_users)
        for k in range(n_uavs):
            ids = np.where(cids == k)[0]
            if len(ids):
                g[ids] = cal_channel_fading_gain(xa[ids, t], ya[ids, t], uav_xy[k, 0], uav_xy[k, 1], z0)
        # mean access SNR-based spectral efficiency proxy
        se = np.log2(1 + pTx_downlink * g / noise_pw_total)
        res[s] = se
    print(f't={t:3d}: mean log2(1+SNR) per seed -> ' +
          '  '.join(f's{s}={res[s].mean():.3f}' for s in res) +
          f'   (spread={max(r.mean() for r in res.values())-min(r.mean() for r in res.values()):.3f})')

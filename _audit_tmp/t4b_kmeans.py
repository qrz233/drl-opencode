# -*- coding: utf-8 -*-
"""Is the KMeans difference a label permutation or a real re-partition?"""
import os, sys
import numpy as np
import pandas as pd
PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
from params_sys import *
from utils_sys import kmeans_clustering

sim_in = os.path.join(PROJ, 'simulation', 'sim-input')
def from_exel(f):
    return np.transpose(pd.read_excel(f, index_col=0).to_numpy())
xa = from_exel(os.path.join(sim_in, f"xlocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx"))
ya = from_exel(os.path.join(sim_in, f"ylocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx"))

print('UAV initial positions (corner deployment):')
print(init_locations_uav)

for t in [4, 20, 100, 440]:
    loc = np.hstack((xa[:, [t]], ya[:, [t]], np.zeros((n_users, 1))))
    np.random.seed(0)
    cm1, c1, cent1 = kmeans_clustering(loc)
    np.random.seed(7)
    cm2, c2, cent2 = kmeans_clustering(loc)
    # match clusters by nearest centroid
    perm = []
    for k in range(n_uavs):
        j = int(np.argmin(np.linalg.norm(cent2 - cent1[k], axis=1)))
        perm.append(j)
    c2_remap = np.array([perm[c] for c in c2])
    same_partition = np.array_equal(c1, c2_remap)
    # is the label->UAV-position assignment stable?
    def nearest_uav(cent):
        d = np.linalg.norm(cent[:, :2][:, None, :] - init_locations_uav[None, :, :], axis=2)
        return d.argmin(axis=1)
    a1, a2 = nearest_uav(cent1), nearest_uav(cent2)
    print(f'\nt={t}: label-permutation-only = {same_partition}; '
          f'n_diff={int((c1!=c2).sum())}')
    print(f'   cluster k -> nearest UAV (seed0): {a1.tolist()}   (seed7): {a2.tolist()}')
    print(f'   seed0 centroids: {np.round(cent1[:,:2],1).tolist()}')
    print(f'   seed7 centroids: {np.round(cent2[:,:2],1).tolist()}')

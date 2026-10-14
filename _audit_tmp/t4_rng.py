# -*- coding: utf-8 -*-
"""Audit test 4: RNG coupling of KMeans across methods; heatmap noise reproducibility."""
import os, sys
import numpy as np
import pandas as pd
PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
from params_sys import *
from utils_sys import kmeans_clustering, gen_heatmap

sim_in = os.path.join(PROJ, 'simulation', 'sim-input')
def from_exel(f):
    return np.transpose(pd.read_excel(f, index_col=0).to_numpy())
xa = from_exel(os.path.join(sim_in, f"xlocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx"))
ya = from_exel(os.path.join(sim_in, f"ylocation, nusers={n_users}, nslots={n_slots}, speed_avg={speed_user_avg:.1f}, bound={boundary}-sfm.xlsx"))
print('user location input loaded', xa.shape)

loc = np.hstack((xa[:, [20]], ya[:, [20]], np.zeros((n_users, 1))))

# (a) same input, two calls with different intervening RNG consumption
np.random.seed(0)
c1 = kmeans_clustering(loc)[1]
np.random.seed(0)
_ = np.random.randn(500)              # simulate other methods' RNG consumption
c2 = kmeans_clustering(loc)[1]
print(f'\n[KMeans] labels identical across different RNG states: {np.array_equal(c1, c2)} '
      f'(n_diff={int((c1!=c2).sum())})')

# (b) across a few slots
diff_slots = 0
for t in [4, 20, 100, 200, 440]:
    loc_t = np.hstack((xa[:, [t]], ya[:, [t]], np.zeros((n_users, 1))))
    np.random.seed(0)
    a = kmeans_clustering(loc_t)[1]
    np.random.seed(1)
    b = kmeans_clustering(loc_t)[1]
    d = int((a != b).sum())
    diff_slots += (d > 0)
    print(f'  t={t:3d}: label diff between seed0 vs seed1 = {d}')
print(f'[KMeans] slots with any label difference: {diff_slots}/5')

# (c) heatmap noise reproducibility (params rng is default_rng() unseeded)
h1 = gen_heatmap(xa[:, 20], ya[:, 20], np.ones(n_users), 1.0)
h2 = gen_heatmap(xa[:, 20], ya[:, 20], np.ones(n_users), 1.0)
print(f'\n[heatmap] two consecutive gen_heatmap calls identical: {np.array_equal(h1, h2)}')
print(f'[heatmap] max|h1-h2| = {np.abs(h1-h2).max():.3e}   (noise floor {1e-6/np.max(np.abs(h1)):.3e} rel)')
print(f'[heatmap] rng seeded? params_sys line 5: rng = np.random.default_rng()  -> unseeded')

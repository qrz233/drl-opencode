# -*- coding: utf-8 -*-
"""Audit test 5: (a) baseline vs BlockA noise-model divergence; (b) oracle/subset ordering."""
import os, sys
import numpy as np
import pandas as pd
PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
sys.path.insert(0, PROJ)
from params_sys import *
import blocka_optimizer as B

print('=== (a) rate model divergence ===')
print(f'bw_per_user (BlockA noise-ref W_ref) = {bw_per_user/1e6} MHz')
print(f'access pool W_U = {bw_total_Hz/1e6} MHz ; avg per active user if 100 active = '
      f'{bw_total_Hz/1e6/100:.3f} MHz')
# baseline model: rate = W*log2(1+SNR)  (noise fixed = noise_pw_total)
# blockA model:  rate = W*log2(1+G/W), G = g*pTx/(noise_pw_total/W_ref)
SNR = 10 ** 1.5     # representative 15 dB
for W_MHz in [0.2, 0.4, 0.5, 0.8, 1.5, 5.0]:
    W = W_MHz * 1e6
    base = W * np.log2(1 + SNR) / 1e6           # Mb/s
    G = SNR * (bw_per_user / 1e6)               # in MHz
    blk = W_MHz * np.log2(1 + G / W_MHz)        # Mb/s
    print(f'  W={W_MHz:4.2f} MHz : baseline={base:8.4f} Mb/s  blockA={blk:8.4f}  '
          f'ratio={blk/base:6.3f}')

print('\n=== (b) oracle vs subset ordering (task2_validation_sweep.csv) ===')
p = r'D:\Learn\work\drl-opencode\expB_results\wa30_audit\task2_validation_sweep.csv'
if os.path.exists(p):
    d = pd.read_csv(p)
    print('cols:', d.columns.tolist())
    print(d.head(3).to_string())
else:
    print('missing', p)

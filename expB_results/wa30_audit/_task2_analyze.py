# -*- coding: utf-8 -*-
"""Analyze step sweep (task 2 outputs) + reuse 20 MHz data for step-limitation."""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = r'D:\Learn\work\drl-opencode\expB_results\wa30_audit'
df = pd.read_csv(os.path.join(OUT, 'task2_step_sweep.csv'))
t2 = pd.read_csv(os.path.join(OUT, 'task2_wa30_methods.csv'))
g = pd.read_csv(os.path.join(OUT, 'taskC_grid_residual.csv'))

# coexistence flag from task2
co_set = set(t2.loc[t2['base_sim_uav'] > 0, 'si'])
early_set = set(t2.loc[t2['t'] <= 40, 'si'])   # early slots (t<=40 ~ first 8%)
acc_bind = set(t2.loc[t2['base_pool_acc_bind'] == 1, 'si'])

# ---- 1. rho* distribution (per dir, valid only) ----
v = df[df['valid'] == 1].copy()
print('=== invalid candidates (excluded) ===')
inv = df[df['valid'] == 0]
print('n=%d, rho values: %s' % (len(inv), inv['rho'].value_counts().to_dict()))
print('power fail: P_max range %.2f..%.2f' % (inv['P_max'].min(), inv['P_max'].max()))
print()

# J0 reference (rho=0)
j0 = df[df['rho'] == 0].set_index('si')['J']
res = {}
for dname in ('beta', 'bind'):
    sub = v[v['dir'] == dname].copy()
    best = sub.loc[sub.groupby('si')['J'].idxmin()]
    rho_star = best['rho'].value_counts().sort_index()
    print(f'=== rho* distribution, dir={dname} (valid snaps n={len(best)}) ===')
    print(rho_star.to_dict())
    # how many snaps excluded because rho=1 invalid -> their best might be 0.75
    res[dname] = best
print()

# rho=1.0 all invalid -> for snaps where J still decreasing at 0.75, flag "censored at 0.75"
for dname in ('beta', 'bind'):
    sub = v[v['dir'] == dname]
    piv = sub.pivot_table(index='si', columns='rho', values='J')
    decreasing = piv[0.75] < piv[0.5]   # still improving at last valid rho
    print(f'{dname}: snaps where J(0.75)<J(0.5) [censored, true rho* may be >0.75]: '
          f'{int(decreasing.sum())}/{len(piv)}')
print()

# ---- 2. extended-grid best vs fixed K5 ----
# fixed K5 = best of {stay, b25, b50, bd25, bd50} from task2 (uses rho 0/0.25/0.5, dir beta/bind)
k5 = t2.set_index('si')
# extended best per dir and overall
ext_overall = []
for si, grp in v.groupby('si'):
    b = grp.loc[grp['J'].idxmin()]
    ext_overall.append(dict(si=si, dir=b['dir'], rho=b['rho'], J_ext=b['J'],
                            J_k5_dual=k5.loc[si, 'J0'] - k5.loc[si, 'dJ_dual'],
                            J_k5_bind=k5.loc[si, 'J0'] - k5.loc[si, 'dJ_bind'],
                            J_k5_k3=k5.loc[si, 'J0'] - k5.loc[si, 'dJ_k3']))
ext = pd.DataFrame(ext_overall)
ext['J_k5best'] = ext[['J_k5_dual', 'J_k5_bind', 'J_k5_k3']].min(axis=1)
ext['gap_k5_to_ext'] = ext['J_k5best'] - ext['J_ext']   # >=0 expected
ext = ext.set_index('si')
ext['coexist'] = ext.index.isin(co_set)
ext['early'] = ext.index.isin(early_set)
ext['acc_bind'] = ext.index.isin(acc_bind)
ext['qlen'] = t2.set_index('si').loc[ext.index, 'qlen']
ext['t'] = t2.set_index('si').loc[ext.index, 't']

print('=== extended grid vs fixed K5 (gap = J_k5best - J_ext, valid only) ===')
print('mean=%.4f med=%.4f p90=%.4f max=%.4f  >0 rate=%.3f  >0.5 rate=%.3f'
      % (ext['gap_k5_to_ext'].mean(), ext['gap_k5_to_ext'].median(),
         ext['gap_k5_to_ext'].quantile(.9), ext['gap_k5_to_ext'].max(),
         (ext['gap_k5_to_ext'] > 1e-9).mean(),
         (ext['gap_k5_to_ext'] > 0.5).mean()))
print('total gap sum=%.2f over %d snaps' % (ext['gap_k5_to_ext'].sum(), len(ext)))
print()
print('gap by subset:')
for name, mask in (('coexist', ext['coexist']), ('non-coexist', ~ext['coexist']),
                   ('t<=40 (early)', ext['early']), ('t>40', ~ext['early']),
                   ('acc-binding', ext['acc_bind']), ('non-acc', ~ext['acc_bind'])):
    s = ext.loc[mask, 'gap_k5_to_ext']
    print('  %-16s n=%2d  mean=%.4f  sum=%.2f' % (name, len(s), s.mean(), s.sum()))
print()

# rho* of extended best
print('extended-best rho* distribution:', ext['rho'].value_counts().sort_index().to_dict())
print('extended-best dir distribution:', ext['dir'].value_counts().to_dict())
print()

# ---- 3. delivery vs energy decomposition for large rho ----
print('=== decomposition: J = Eterm - D  (Eterm=psi_E*E0.P, D=beta.D) ===')
piv_D = v.pivot_table(index='si', columns=['dir', 'rho'], values='D')
piv_E = v.pivot_table(index='si', columns=['dir', 'rho'], values='Eterm')
for dname in ('beta', 'bind'):
    for rho in (0.75,):
        dD = (piv_D[(dname, rho)] - piv_D[(dname, 0.0)]).mean()
        dE = (piv_E[(dname, rho)] - piv_E[(dname, 0.0)]).mean()
        gain = dD - dE   # dJ vs stay = (D-D0) - (Eterm-Eterm0)
        print(f'  {dname} rho={rho} vs stay: dD=+{dD:.3f}  dEterm=+{dE:.3f}  gain dJ={gain:.4f}')
    for rho in (0.5,):
        dD = (piv_D[(dname, rho)] - piv_D[(dname, 0.0)]).mean()
        dE = (piv_E[(dname, rho)] - piv_E[(dname, 0.0)]).mean()
        print(f'  {dname} rho={rho} vs stay: dD=+{dD:.3f}  dEterm=+{dE:.3f}  gain dJ={dD-dE:.4f}')
    # 0.75 vs 0.5 marginal
    dD = (piv_D[(dname, 0.75)] - piv_D[(dname, 0.5)]).mean()
    dE = (piv_E[(dname, 0.75)] - piv_E[(dname, 0.5)]).mean()
    print(f'  {dname} marginal 0.75 vs 0.5: dD=+{dD:.4f}  dEterm=+{dE:.4f}  gain={dD-dE:.4f}')
print()

# ---- 4. constraint / anomaly stats per rho ----
print('=== per-rho stats ===')
stat = df.groupby(['dir', 'rho']).agg(
    n=('si', 'count'), valid_rate=('valid', 'mean'),
    J_mean=('J', 'mean'), P_max_mean=('P_max', 'mean'),
    v_max_mean=('v_max', 'mean'), solve_mean=('solve_s', 'mean'),
    n_inacc=('status', lambda s: int((s == 'optimal_inaccurate').sum()))
).round(4)
print(stat.to_string())
print()

# ---- 5. reuse: 20 MHz step-limited evidence from selective_mobility ----
print('=== 20 MHz reuse: selective_mobility candidates (rho 0.25 vs 0.5) ===')
sel = pd.read_csv(r'D:\Learn\work\drl-opencode\expB_results\selective_mobility\selective_candidates.csv')
if 'rho' in sel.columns:
    piv2 = sel.pivot_table(index=['slot', 'u'], columns='rho', values='J')
    if 0.5 in piv2.columns and 0.25 in piv2.columns:
        better50 = (piv2[0.5] < piv2[0.25] - 1e-9).mean()
        print('20 MHz (W_A=20): fraction J(rho=0.5) < J(rho=0.25): %.3f (n=%d pairs)'
              % (better50, len(piv2)))
        print('=> at 20 MHz the tested grid was {0.25,0.5}; if 0.5 wins majority, '
              'the optimum was already at the grid edge (step-limited, censored).')
        # dJ distribution at edge
        dJ_edge = (piv2[0.25] - piv2[0.5])
        print('dJ(0.25->0.5): mean=%.4f med=%.4f p90=%.4f' %
              (dJ_edge.mean(), dJ_edge.median(), dJ_edge.quantile(.9)))
print()

# also: 30MHz with only K5's rhos {0.25,0.5} vs full valid grid -> what K5 missed
sub_k5rho = v[v['rho'].isin([0.0, 0.25, 0.5])]
k5rho_best = sub_k5rho.loc[sub_k5rho.groupby('si')['J'].idxmin()].set_index('si')
gap_extra_rho = k5rho_best['J'] - ext['J_ext']
print('=== 30 MHz: extra-rho effect (rho in {0.1,0.75} beyond K5 {0.25,0.5}) ===')
print('mean gap=%.4f  >0 rate=%.3f  sum=%.2f'
      % (gap_extra_rho.mean(), (gap_extra_rho > 1e-9).mean(), gap_extra_rho.sum()))
print()

# save
ext.to_csv(os.path.join(OUT, 'task2_extended_vs_k5.csv'))
summary = dict(
    invalid_n=int((df['valid'] == 0).sum()),
    invalid_rho=df.loc[df['valid'] == 0, 'rho'].value_counts().to_dict(),
    gap_k5_ext_mean=float(ext['gap_k5_to_ext'].mean()),
    gap_k5_ext_sum=float(ext['gap_k5_to_ext'].sum()),
    gap_gt0_rate=float((ext['gap_k5_to_ext'] > 1e-9).mean()),
    gap_coexist_mean=float(ext.loc[ext['coexist'], 'gap_k5_to_ext'].mean()),
    gap_noncoexist_mean=float(ext.loc[~ext['coexist'], 'gap_k5_to_ext'].mean()),
    gap_early_mean=float(ext.loc[ext['early'], 'gap_k5_to_ext'].mean()),
    gap_late_mean=float(ext.loc[~ext['early'], 'gap_k5_to_ext'].mean()),
    extra_rho_gap_mean=float(gap_extra_rho.mean()),
    extra_rho_gap_gt0=float((gap_extra_rho > 1e-9).mean()),
)
with open(os.path.join(OUT, 'task2_sweep_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1)
print('summary written')

# plot
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
ax = axes[0, 0]
for dname, c in (('beta', 'tab:blue'), ('bind', 'tab:orange')):
    m = v[v['dir'] == dname].groupby('rho')['J'].mean()
    ax.plot(m.index, m.values, 'o-', color=c, label=dname)
ax.set_xlabel('rho'); ax.set_ylabel('mean J'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('J vs rho (valid, 30 MHz)')
ax = axes[0, 1]
ax.hist(ext.loc[ext['gap_k5_to_ext'] > 1e-9, 'gap_k5_to_ext'], bins=15,
        color='tab:green', alpha=.8)
ax.set_xlabel('gap: K5 best - extended best'); ax.set_title('gap distribution')
ax.grid(alpha=0.3)
ax = axes[0, 2]
colors = np.where(ext['coexist'], 'tab:red', 'tab:gray')
ax.scatter(ext['t'], ext['gap_k5_to_ext'], c=colors, s=28, alpha=.8)
ax.set_xlabel('slot t'); ax.set_ylabel('gap'); ax.grid(alpha=0.3)
ax.set_title('gap vs t (red=coexist)')
ax = axes[1, 0]
ax.scatter(ext['qlen'], ext['gap_k5_to_ext'], c=np.where(ext['coexist'], 'tab:red', 'tab:gray'),
           s=28, alpha=.8)
ax.set_xlabel('qlen'); ax.set_ylabel('gap'); ax.grid(alpha=0.3)
ax.set_title('gap vs qlen')
ax = axes[1, 1]
rho_star = ext['rho'].value_counts().sort_index()
ax.bar([str(x) for x in rho_star.index], rho_star.values, color='tab:purple', alpha=.8)
ax.set_title('extended-best rho* distribution'); ax.grid(alpha=0.3)
ax = axes[1, 2]
width = 0.35
for dname, off in (('beta', -width/2), ('bind', width/2)):
    m = v[v['dir'] == dname].groupby('rho')[['D', 'Eterm']].mean()
    xs = np.arange(len(m))
    ax.bar(xs + off, m['D'], width, alpha=.6, label=f'D {dname}')
ax2 = ax.twinx()
m = v[v['dir'] == 'bind'].groupby('rho')['Eterm'].mean()
ax2.plot(range(len(m)), m.values, 'k-o', label='Eterm(bind)')
ax.set_xticks(range(len(m))); ax.set_xticklabels([str(x) for x in m.index])
ax.set_xlabel('rho'); ax.set_ylabel('mean D'); ax2.set_ylabel('mean Eterm')
ax.set_title('delivery vs energy term'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'task2_step_sweep.png'), dpi=130)
print('plot written')

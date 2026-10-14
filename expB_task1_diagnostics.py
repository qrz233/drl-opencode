# -*- coding: utf-8
'''Task-1 diagnostics: late-horizon stability from existing stage-2 runs.
No new simulation; reads expB_results/stage2_multiseed/stage2_seed{0-4}.npz.

Questions answered:
  1. Window KPIs t in [0,120), [120,300), [300,501), [120,501):
     wMOS, Web queue (all), backhaul util/capacity.
  2. Late-window linear trends of Q_web(t) and E_u(t) (slope per 100 slots).
  3. Dual win rate & mean dJ_gain on dual-win slots, per window.
  4. Cumulative wMOS gap dual-k3 vs time: share of full-horizon gap
     earned in [0,120), [120,300), [300,501).
'''
import os
import numpy as np
import pandas as pd

IN = r'D:\Learn\work\drl-opencode\expB_results\stage2_multiseed'
OUT = r'D:\Learn\work\drl-opencode\expB_results\task1_diagnostics'
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ('k3', 'betak5', 'dualk5')
WINDOWS = [('full', 0, 501), ('early', 0, 120), ('mid', 120, 300),
           ('late', 300, 501), ('post120', 120, 501)]

runs = {}
for s in SEEDS:
    z = np.load(os.path.join(IN, f'stage2_seed{s}.npz'))
    runs[s] = {}
    for v in VARIANTS:
        pre = v + '_'
        runs[s][v] = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}

rows = []
trend_rows = []
cum_rows = []
for s in SEEDS:
    d0 = runs[s]['dualk5']
    for v in VARIANTS:
        r = runs[s][v]
        for name, a, b in WINDOWS:
            sl = slice(a, b)
            rows.append(dict(
                seed=s, variant=v, window=name,
                wmos=float(np.nanmean(r['wmos'][sl])),
                web_q=float(np.nanmean(r['web_qlen_all'][sl])),
                web_q_end=float(r['web_qlen_all'][b - 1]),
                bh_util=float(np.nanmean(r['bh_util'][sl])),
                bh_cap=float(np.nanmean(r['sum_Rbh'][sl])),
                qlen_act=float(np.nanmean(r['qlen_act'][sl])),
                win_dual=float((r['win'][sl] == 2).mean()) if v == 'dualk5' else np.nan,
            ))
        # late-window trends: OLS slope per 100 slots on [300,501)
        t = np.arange(300, 501)
        q = r['web_qlen_all'][300:501]
        m = np.isfinite(q)
        q_slope = (np.polyfit(t[m], q[m], 1)[0] * 100) if m.sum() > 10 else np.nan
        E_mean = r['E_hist'][300:501].mean(axis=1)
        E_slope = np.polyfit(t, E_mean, 1)[0] * 100
        trend_rows.append(dict(
            seed=s, variant=v,
            qweb_slope_late=q_slope,
            qweb_end=r['web_qlen_all'][-1],
            E_slope_late=E_slope,
            E_final_max=float(np.max(r['E_final'])),
            E_hist_max=float(r['E_hist'].max()),
        ))
    # dual-specific per window
    for name, a, b in WINDOWS:
        sl = slice(a, b)
        dw = d0['dual_win'][sl] & np.isfinite(d0['decomp_err'][sl])
        trend_rows[-1].setdefault('x', None)
        rows.append(dict(
            seed=s, variant='dualk5_only', window=name,
            wmos=np.nan, web_q=np.nan, web_q_end=np.nan, bh_util=np.nan,
            bh_cap=np.nan, qlen_act=np.nan,
            win_dual=float(d0['dual_win'][sl].mean()),
            dualwin_dJ=float(np.nanmean(d0['dJ_g'][sl][dw])) if dw.any() else np.nan,
            dualwin_n=int(dw.sum()),
        ))
    # cumulative wMOS gap dual-k3: per-slot contribution (NaN-safe)
    w_d = d0['wmos']
    w_k = runs[s]['k3']['wmos']
    both = np.isfinite(w_d) & np.isfinite(w_k)
    contrib = np.where(both, w_d - w_k, 0.0)
    cum = np.cumsum(contrib)
    total = cum[-1]
    for name, a, b in WINDOWS:
        cum_rows.append(dict(
            seed=s, window=name,
            cum_gap_sum=float(cum[min(b, 501) - 1] - (cum[a - 1] if a > 0 else 0.0)),
            share_of_total=float((cum[min(b, 501) - 1] - (cum[a - 1] if a > 0 else 0.0)) / total)
            if abs(total) > 1e-9 else np.nan,
        ))

kdf = pd.DataFrame(rows)
tdf = pd.DataFrame(trend_rows)
cdf = pd.DataFrame(cum_rows)
kdf.to_csv(os.path.join(OUT, 'window_kpis.csv'), index=False)
tdf.to_csv(os.path.join(OUT, 'late_trends.csv'), index=False)
cdf.to_csv(os.path.join(OUT, 'cum_gap.csv'), index=False)

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

print('==== 1. Window KPIs (mean over seeds) ====')
pv = kdf[kdf.variant.isin(VARIANTS)].pivot_table(
    index='window', columns='variant', values=['wmos', 'web_q', 'bh_util'])
for metric in ('wmos', 'web_q', 'bh_util'):
    print(f'\n-- {metric} --')
    print(pv[metric].loc[[w[0] for w in WINDOWS]].round(4))

print('\n==== 2. Late-window trends (slope per 100 slots on [300,501)) ====')
for v in VARIANTS:
    sub = tdf[tdf.variant == v]
    print(f'\n[{v}]')
    print(sub[['seed', 'qweb_slope_late', 'qweb_end', 'E_slope_late',
               'E_final_max', 'E_hist_max']].round(3).to_string(index=False))

print('\n==== 3. Dual win rate & dJ on dual-win (per window, per seed) ====')
donly = kdf[kdf.variant == 'dualk5_only']
for name in ('full', 'early', 'mid', 'late', 'post120'):
    sub = donly[donly.window == name]
    print(f'\n[{name}]')
    print(sub[['seed', 'win_dual', 'dualwin_dJ', 'dualwin_n']].round(4).to_string(index=False))

print('\n==== 4. Cumulative wMOS gap (dual - k3): share by window ====')
print(cdf.pivot(index='seed', columns='window', values='share_of_total')
      .round(3)[['full', 'early', 'mid', 'late', 'post120']])
print('\nmean share:', cdf.groupby('window')['share_of_total'].mean()
      .round(3)[['full', 'early', 'mid', 'late', 'post120']].to_dict())

# decision helper
print('\n==== decision flags ====')
late = kdf[(kdf.variant.isin(VARIANTS)) & (kdf.window == 'late')]
print('late-window web_q mean per scheme:')
print(late.groupby('variant')['web_q'].mean().round(3))
tl = tdf[tdf.variant.isin(VARIANTS)]
print('max late qweb slope:', tl.groupby('variant')['qweb_slope_late'].max().round(4).to_dict())
print('max late E slope:', tl.groupby('variant')['E_slope_late'].max().round(4).to_dict())
print('max E_final:', tl.groupby('variant')['E_final_max'].max().round(1).to_dict())
print(f'\nwrote {OUT}')

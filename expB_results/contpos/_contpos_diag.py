# -*- coding: utf-8 -*-
"""Task 2 (contpos): small-scale continuous position search diagnosis.

Schemes per snapshot x bandwidth (same slot inputs, airship & altitude FIXED,
exact Block A via solve_full):

  A = Binding K5 (safe fallback): stay + {beta,bind} x {0.25,0.5}   [base 1 + 5 evals]
  B = 2-var continuous (lam, rho): e_u(lam)=unit((1-lam)*e_beta+lam*e_bh), global
      lam in [0,1], rho in [0, v_pw*tau/30]; grid (rho=0 once; else 11 lam x 8 rho)
      + Nelder-Mead polish (maxfev=40).
  C = 8-dim continuous local opt: warm start at K5 best position, SLSQP with
      analytic dual gradient (contpos_model.md §5.2), box bounds + per-UAV disk
      radius R_MAX = v_pw*tau (power binding per Lemma 1); 3 starts
      (k5 / stay / rand) -> multi-start sensitivity. Failed/illegal start -> K5.

Safety: J_comp = min legal{A,B,C} -> never worse than K5, never illegal.
Gradient check: central FD (h=0.5) vs analytic dual gradient at K5 & stay points.

Selection frozen BEFORE evaluation: 16 snaps from the frozen 48
(mixed=6, early_cong=5, stable=5, random_state=1), each run at W_A in {20,30}.

Outputs -> contpos_selection.csv / contpos_results.csv / contpos_cstarts.csv /
           contpos_gradcheck.csv
Run: python _contpos_diag.py [--smoke]
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

ROOT = r'D:\Learn\work\drl-opencode'
OUTB = os.path.join(ROOT, 'expB_results', 'wa30_audit')       # prefix + prior phase
OUT = os.path.join(ROOT, 'expB_results', 'contpos')
PROJ = r'D:\Learn\work\drl-opencode\uav-bs-placement-drl-modified-v13b-K20-s11'
os.makedirs(OUT, exist_ok=True)
os.chdir(PROJ)                                                # params_sys import side effect

SRC = open(os.path.join(OUTB, '_grid_residual.py'), encoding='utf-8').read()
cut = SRC.index('rows = []')
ns = {}
exec(compile(SRC[:cut], '_grid_prefix', 'exec'), ns)
solve_full = ns['solve_full']
gammas = ns['gammas']
beta_dirs = ns['beta_dirs']
unit = ns['unit']
SNAP = ns['SNAP']
cal_channel_fading_gain = ns['cal_channel_fading_gain']
cal_uav_propulsion_energy = ns['cal_uav_propulsion_energy']
PSI_E, SPD_XY, TAU, BOUNDARY = ns['PSI_E'], ns['SPD_XY'], ns['TAU'], ns['BOUNDARY']
z_min, z_max = ns['z_min'], ns['z_max']
DIR_EPS, BIND_EPS = ns['DIR_EPS'], ns['BIND_EPS']
LN2, MHZ = ns['LN2'], ns['MHZ']
Bmod = ns['B']
from params_sys import pTx_downlink, pTx_airship
from blocka_optimizer import NOISE_PSD_ACC, NOISE_PSD_BH

EPS, EPS_TIE = 1e-9, 1e-6
V_MAX, PW_MAX = 30.0, 683.0
BWS = (20.0, 30.0)
H_FD = 0.5
BIG, PEN_TOL = 1e6, 1e-9
SMOKE = '--smoke' in sys.argv

# ---- power radius (Lemma 1): {v: P(v)<=683} = [0, v_pw] ----
Pfun = lambda v: float(cal_uav_propulsion_energy(V=float(v), Vz=0.0)[0])
v_pw = brentq(lambda v: Pfun(v) - PW_MAX, 6.0, 30.0)
R_MAX = v_pw * TAU
assert Pfun(R_MAX / TAU) <= PW_MAX + 1e-6 and v_pw < V_MAX
print('R_MAX (power-binding disk radius) = %.6f m  (v_pw=%.6f < %.0f)'
      % (R_MAX, v_pw, V_MAX), flush=True)


def Pprime(V):
    """dP/dV of cal_uav_propulsion_energy (Vz=0), analytic (contpos_model.md §4.1)."""
    if V <= 0.0:
        return 0.0
    omega, R, v0, d0, s, rho, A = 200, 0.4, 4.03, 0.6, 0.12, 1.225, 3.14 * 0.4 ** 2
    W, delta, k = 10, 0.012, 0.1
    Utip = omega * R
    P0 = delta / 8 * rho * s * A * omega ** 3 * R ** 3
    Pi = (1 + k) * W ** 1.5 / np.sqrt(2 * rho * A)
    root = np.sqrt(1.0 + V ** 4 / (4 * v0 ** 4))
    u = root - V ** 2 / (2 * v0 ** 2)
    up = V ** 3 / (2 * v0 ** 4) / root - V / v0 ** 2
    return 6 * P0 * V / Utip ** 2 + Pi * up / (2 * np.sqrt(u)) + 1.5 * d0 * rho * s * A * V ** 2


# Pprime self-check vs FD
for _v in (0.5, 5.0, 10.0, 25.0):
    _fd = (Pfun(_v + 1e-4) - Pfun(_v - 1e-4)) / 2e-4
    assert abs(Pprime(_v) - _fd) <= 1e-4 * max(1.0, abs(_fd)), (_v, Pprime(_v), _fd)

CNT = dict(n=0, bad=0, inacc=0)


def cnt_delta():
    return dict(n=CNT['n'], bad=CNT['bad'], inacc=CNT['inacc'])


def eval_cand(locs, air, sn, E0, beta, W_A):
    """Exact Block A + actual-net-displacement power (same as _t2_validate)."""
    CNT['n'] += 1
    net = locs - sn['locs0']
    v_net = np.linalg.norm(net[:, :2], axis=1) / TAU
    vz = np.abs(net[:, 2]) / TAU
    P = np.array([float(cal_uav_propulsion_energy(V=v_net[u], Vz=vz[u])[0])
                  for u in range(4)])
    ga, gb, g = gammas(sn['ux'], sn['uy'], sn['cids'], locs, air)
    t0 = time.time()
    res = solve_full(ga, gb, sn['demand'], sn['cids'], beta, W_A=W_A)
    dt = time.time() - t0
    J = PSI_E * float(E0 @ P) - float(beta @ res['D_Mb'])
    ok_status = res['status'] in ('optimal', 'optimal_inaccurate')
    if not ok_status:
        CNT['bad'] += 1
    elif res['status'] == 'optimal_inaccurate':
        CNT['inacc'] += 1
    ok_speed = bool(np.all(v_net <= V_MAX + 1e-9))
    ok_power = bool(np.all(P <= PW_MAX + 1e-9))
    ok_region = bool(np.all(np.abs(locs[:, 0]) <= BOUNDARY + 1e-9) and
                     np.all(np.abs(locs[:, 1]) <= BOUNDARY + 1e-9))
    ok_alt = bool(np.all(locs[:, 2] >= z_min - 1e-9) and
                  np.all(locs[:, 2] <= z_max + 1e-9))
    return dict(J=J, P=P, res=res, dt=dt, ok=bool(ok_status and ok_speed and
                 ok_power and ok_region and ok_alt),
                 ok_status=ok_status, ok_speed=ok_speed, ok_power=ok_power,
                 ok_region=ok_region, ok_alt=ok_alt, v_net=v_net,
                 D=float(beta @ res['D_Mb']), Eterm=PSI_E * float(E0 @ P),
                 P_sum=float(P.sum()), P_max=float(P.max()))


def move(locs0, e_u, rho):
    locs = locs0.copy()
    if rho > 0:
        disp = SPD_XY * rho * e_u * TAU
        locs[:, 0] = np.clip(locs0[:, 0] + disp[:, 0], -BOUNDARY, BOUNDARY)
        locs[:, 1] = np.clip(locs0[:, 1] + disp[:, 1], -BOUNDARY, BOUNDARY)
    return locs


def dual_grad(res, locs, air, sn, E0):
    """Analytic grad of J wrt horizontal positions (4,2): contpos_model.md §5.2."""
    q = locs[:, :2]
    net = q - sn['locs0'][:, :2]
    v = np.linalg.norm(net, axis=1) / TAU
    gJ = np.zeros((4, 2))
    for u in range(4):
        if E0[u] != 0.0 and v[u] > DIR_EPS:
            gJ[u] += PSI_E * E0[u] * Pprime(v[u]) / TAU * net[u] / v[u]
    px, py, cids = sn['ux'], sn['uy'], sn['cids']
    g_raw = np.zeros(len(px))
    for u in np.unique(cids):
        m = cids == u
        g_raw[m] = cal_channel_fading_gain(px[m], py[m], locs[u, 0], locs[u, 1], locs[u, 2])
    ga = g_raw * pTx_downlink / NOISE_PSD_ACC / MHZ
    gb, graw_b = Bmod.cal_gamma_backhaul_MHz(locs, air)
    mu, Wi = res['lam_acc'], res['W_ui_Hz'] / MHZ
    nu, Wb = res['nu_bh'], res['W_ua_Hz'] / MHZ
    for i in range(len(px)):
        if mu[i] <= 0.0 or Wi[i] <= 0.0 or ga[i] <= 0.0:
            continue
        u = int(cids[i])
        coef = mu[i] * TAU * Wi[i] / (LN2 * (Wi[i] + ga[i])) * pTx_downlink / NOISE_PSD_ACC / MHZ
        r2 = (px[i] - q[u, 0]) ** 2 + (py[i] - q[u, 1]) ** 2 + locs[u, 2] ** 2
        grad_g = 2.0 * g_raw[i] * np.array([px[i] - q[u, 0], py[i] - q[u, 1]]) / r2
        gJ[u] -= coef * grad_g
    for u in range(4):
        if nu[u] <= 0.0 or Wb[u] <= 0.0 or gb[u] <= 0.0:
            continue
        coef = nu[u] * TAU * Wb[u] / (LN2 * (Wb[u] + gb[u])) * pTx_airship / NOISE_PSD_BH / MHZ
        T = float(np.sum((locs[u] - air) ** 2))
        if T <= 1e-12:
            continue
        grad_g = -2.0 * graw_b[u] * (q[u] - air[:2]) / T
        gJ[u] -= coef * grad_g
    return gJ


def fd_vs_dual(q, res, locs0, air, sn, E0, beta, W_A, point, rows, si):
    locs_pt = locs0.copy()
    locs_pt[:, :2] = q
    du = dual_grad(res, locs_pt, air, sn, E0).ravel()
    fd = np.zeros(8)
    for j in range(8):
        flat = q.ravel().copy()
        flat[j] += H_FD
        locp = locs0.copy()
        locp[:, :2] = flat.reshape(4, 2)
        flat[j] -= 2 * H_FD
        locm = locs0.copy()
        locm[:, :2] = flat.reshape(4, 2)
        Jp = eval_cand(locp, air, sn, E0, beta, W_A)['J']
        Jm = eval_cand(locm, air, sn, E0, beta, W_A)['J']
        fd[j] = (Jp - Jm) / (2 * H_FD)
        rows.append(dict(bw=W_A, si=si, point=point, coord=j, uav=j // 2,
                         axis='x' if j % 2 == 0 else 'y', h=H_FD,
                         fd=fd[j], dual=du[j], abs_err=abs(fd[j] - du[j]),
                         sign_ok=int(np.sign(fd[j]) == np.sign(du[j]) or
                                     (abs(fd[j]) < 1e-9 and abs(du[j]) < 1e-9)),
                         Jp=Jp, Jm=Jm))
    nf, nd = float(np.linalg.norm(fd)), float(np.linalg.norm(du))
    l2 = float(np.linalg.norm(fd - du)) / max(nf, 1e-12)
    cos = float(fd @ du / (nf * nd)) if nf * nd > 0 else float('nan')
    maxabs = float(np.max(np.abs(fd - du)))
    return dict(fd_l2=l2, fd_cos=cos, fd_maxabs=maxabs, fd_norm=nf, dual_norm=nd,
                fd_sign_frac=float(np.mean([r['sign_ok'] for r in rows[-8:]])))


# ---------------- selection (frozen before evaluation) ----------------
sel48 = pd.read_csv(os.path.join(OUTB, 'task2_validation_selection.csv'))
parts = [sel48[sel48['stratum'] == s].sample(n=n, random_state=1)
         for s, n in (('mixed', 6), ('early_cong', 5), ('stable', 5))]
sel = pd.concat(parts).sort_values('si').reset_index(drop=True)
sel.to_csv(os.path.join(OUT, 'contpos_selection.csv'), index=False)
assert len(sel) == 16 and sel['si'].nunique() == 16
print('selection frozen: n=%d %s  si=%s'
      % (len(sel), sel['stratum'].value_counts().to_dict(), sel['si'].tolist()),
      flush=True)

prior = pd.read_csv(os.path.join(OUTB, 'task2_validation_calibers.csv'))
prior = prior.set_index(['bw', 'si'])

by_si = {sn['si']: sn for sn in SNAP}
crows, srows, frows = [], [], []
t_start = time.time()
for W_A in BWS:
    sis = sorted(int(x) for x in sel['si'])
    if SMOKE:
        sis = sis[:1]
    for si in sis:
        sn = by_si[si]
        beta, types, cids = sn['beta'], sn['types'], sn['cids']
        locs0, air0, E0 = sn['locs0'], sn['air0'], sn['E0']
        stratum = str(sel.loc[sel['si'] == si, 'stratum'].iloc[0])

        # ---- base solve (analysis ctx; counted as calls_base) ----
        c0 = cnt_delta(); t0 = time.time()
        ga0, gb0, _ = gammas(sn['ux'], sn['uy'], cids, locs0, air0)
        base = solve_full(ga0, gb0, sn['demand'], cids, beta, W_A=W_A)
        CNT['n'] += 1
        if base['status'] not in ('optimal', 'optimal_inaccurate'):
            CNT['bad'] += 1
        elif base['status'] == 'optimal_inaccurate':
            CNT['inacc'] += 1
        time_base = time.time() - t0
        n_base = cnt_delta()['n'] - c0['n']
        nu = np.maximum(base['nu_bh'], 0.0)
        e_u_beta, _ = beta_dirs(beta, types, cids, locs0, air0, sn['ux'], sn['uy'])
        e_bh = np.array([unit(air0[:2] - locs0[u, :2]) for u in range(4)])
        w_b = (nu > BIND_EPS).astype(float)
        e_u_bind = np.array([unit((1 - w_b[u]) * e_u_beta[u] + w_b[u] * e_bh[u])
                             for u in range(4)])

        # ---- scheme A: Binding K5 ----
        c0 = cnt_delta(); t0 = time.time()
        keys_A = [('beta', 0.0), ('beta', 0.25), ('beta', 0.5),
                  ('bind', 0.25), ('bind', 0.5)]
        meta = {}
        for dname, e_u in (('beta', e_u_beta), ('bind', e_u_bind)):
            for rho in (0.0, 0.25, 0.5):
                if (dname, rho) in keys_A:
                    meta[(dname, rho)] = eval_cand(move(locs0, e_u, rho),
                                                   air0.copy(), sn, E0, beta, W_A)
        time_A = time.time() - t0
        cA = cnt_delta()
        n_A, bad_A, inacc_A = cA['n'] - c0['n'], cA['bad'] - c0['bad'], cA['inacc'] - c0['inacc']
        legal_A = [k for k in keys_A if meta[k]['ok']]
        assert len(legal_A) == 5, ('K5 illegality', si, W_A)
        key_A = min(legal_A, key=lambda k: meta[k]['J'])
        J_A, q_K5 = meta[key_A]['J'], move(locs0, e_u_beta if key_A[0] == 'beta'
                                           else e_u_bind, key_A[1])
        res_A = meta[key_A]['res']
        J0 = meta[('beta', 0.0)]['J']
        locs_K5 = q_K5

        # ---- scheme B: 2-var continuous (lam, rho) ----
        c0 = cnt_delta(); t0 = time.time()
        rho_max = R_MAX / (SPD_XY * TAU)

        def B_ev(th):
            lam = float(np.clip(th[0], 0.0, 1.0))
            rho = float(np.clip(th[1], 0.0, rho_max))
            e_u = np.array([unit((1 - lam) * e_u_beta[u] + lam * e_bh[u])
                            for u in range(4)])
            return eval_cand(move(locs0, e_u, rho), air0.copy(), sn, E0, beta, W_A)

        grid_best = None
        for rho in np.linspace(0.0, rho_max, 9):
            lams = [0.0] if rho == 0.0 else list(np.linspace(0.0, 1.0, 11))
            for lam in lams:
                r = B_ev((lam, rho))
                key = (lam, rho)
                if r['ok'] and (grid_best is None or r['J'] < grid_best[0]):
                    grid_best = (r['J'], key)
        J_B, key_B, nm_ok = float('nan'), None, 0
        if grid_best is not None:
            try:
                sol = minimize(lambda th: B_ev(th)['J'], np.array(grid_best[1]),
                               method='Nelder-Mead',
                               bounds=[(0.0, 1.0), (0.0, rho_max)],
                               options=dict(maxfev=40, xatol=1e-4, fatol=1e-7))
                rf = B_ev(sol.x)
                nm_ok = 1
                if rf['ok'] and rf['J'] < grid_best[0]:
                    J_B, key_B = rf['J'], (float(sol.x[0]), float(sol.x[1]))
            except Exception as e:
                print('  B NM failed si=%d bw=%.0f: %r' % (si, W_A, e), flush=True)
            if key_B is None:
                J_B, key_B = grid_best[0], grid_best[1]
        if not np.isfinite(J_B):                       # fallback safety
            J_B, key_B = J_A, ('fallback_K5', 0.0)
        time_B = time.time() - t0
        cB = cnt_delta()
        n_B, bad_B, inacc_B = cB['n'] - c0['n'], cB['bad'] - c0['bad'], cB['inacc'] - c0['inacc']

        # ---- scheme C: 8-dim continuous, 3 starts, SLSQP + dual gradient ----
        c0 = cnt_delta(); t0 = time.time()
        bounds = [(-BOUNDARY, BOUNDARY)] * 8

        def mk_cons():
            cons = []
            for u in range(4):
                def cf(f, u=u):
                    d = f.reshape(4, 2)[u] - locs0[u, :2]
                    return R_MAX ** 2 - float(d @ d)
                def cj(f, u=u):
                    g = np.zeros(8)
                    d = f.reshape(4, 2)[u] - locs0[u, :2]
                    g[2 * u:2 * u + 2] = -2.0 * d
                    return g
                cons.append(dict(type='ineq', fun=cf, jac=cj))
            return cons
        cons = mk_cons()

        def C_ev(flat):
            q = flat.reshape(4, 2)
            locs = locs0.copy()
            locs[:, :2] = q
            if float(np.min(np.linalg.norm(locs - air0, axis=1))) < 1.0:
                return BIG, np.zeros(8)
            r = eval_cand(locs, air0, sn, E0, beta, W_A)
            if not r['ok_status']:
                return BIG, np.zeros(8)
            return r['J'], dual_grad(r['res'], locs, air0, sn, E0).ravel()

        rng = np.random.default_rng(900000 + si * 10 + int(W_A))
        d_r = rng.uniform(-1.0, 1.0, (4, 2))
        nrm = np.linalg.norm(d_r, axis=1, keepdims=True)
        d_r = d_r * np.minimum(1.0, R_MAX / np.maximum(nrm, 1e-12))
        starts = [('k5', locs_K5[:, :2].copy(), J_A),
                  ('stay', locs0[:, :2].copy(), J0),
                  ('rand', np.clip(locs0[:, :2] + d_r, -BOUNDARY, BOUNDARY), None)]
        cstart_rows, finals = [], []
        for st_name, q_x0, J_x0 in starts:
            if J_x0 is None:
                lc = locs0.copy(); lc[:, :2] = q_x0
                J_x0 = eval_cand(lc, air0.copy(), sn, E0, beta, W_A)['J']
            nb = cnt_delta()['n']; tb = time.time()
            rec = dict(bw=W_A, si=si, start=st_name, J_start=J_x0,
                       legal=0, status='exception', success=0, nit=0, nfev=0,
                       time_s=0.0, improved=float('nan'), J_final=float('nan'),
                       gradnorm_dual=float('nan'), projected=0)
            try:
                sol = minimize(C_ev, q_x0.ravel(), jac=True, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options=dict(maxiter=60, ftol=1e-9))
            except Exception as e:
                import traceback
                rec['status'] = 'exception:%s' % type(e).__name__
                if not cstart_rows and st_name == starts[0][0]:
                    traceback.print_exc()
                sol = None
            if sol is not None:
                qf = sol.x.reshape(4, 2)
                d = qf - locs0[:, :2]
                n = np.linalg.norm(d, axis=1)
                over = n > R_MAX
                if np.any(over):            # project onto feasible disk (like move()'s clip)
                    d = d.copy()
                    d[over] *= (R_MAX / n[over])[:, None]
                    qf = locs0[:, :2] + d
                    rec['projected'] = 1
                qf = np.clip(qf, -BOUNDARY, BOUNDARY)
                locs_f = locs0.copy()
                locs_f[:, :2] = qf
                rf = eval_cand(locs_f, air0.copy(), sn, E0, beta, W_A)
                rec.update(status=rf['res']['status'], success=int(bool(sol.success)),
                           nit=int(getattr(sol, 'nit', 0)),
                           legal=int(rf['ok']), J_final=rf['J'],
                           improved=J_x0 - rf['J'],
                           gradnorm_dual=float(np.linalg.norm(
                               dual_grad(rf['res'], locs_f, air0, sn, E0))))
                if rf['ok']:
                    finals.append((rf['J'], qf, st_name))
                else:
                    print('  C start %s illegal si=%d bw=%.0f (spd=%s pw=%s reg=%s)'
                          % (st_name, si, W_A, rf['ok_speed'], rf['ok_power'],
                             rf['ok_region']), flush=True)
            rec['nfev'] = cnt_delta()['n'] - nb
            rec['time_s'] = time.time() - tb
            cstart_rows.append(rec)
        time_C = time.time() - t0
        cC = cnt_delta()
        n_C, bad_C, inacc_C = cC['n'] - c0['n'], cC['bad'] - c0['bad'], cC['inacc'] - c0['inacc']
        if finals:
            J_C, q_C, best_start = min(finals, key=lambda t: t[0])
            c_fallback = 0
        else:
            J_C, q_C, best_start, c_fallback = J_A, locs_K5, 'fallback_K5', 1
        c_vals = [r['J_final'] for r in cstart_rows if np.isfinite(r['J_final'])
                  and r['legal']]
        c_spread = float(max(c_vals) - min(c_vals)) if len(c_vals) >= 2 else float('nan')
        c_std = float(np.std(c_vals)) if len(c_vals) >= 2 else float('nan')

        # ---- gradient check at K5 & stay points ----
        gc0 = cnt_delta(); tf0 = time.time()
        gc_k5 = fd_vs_dual(locs_K5[:, :2], res_A, locs0, air0, sn, E0, beta, W_A,
                           'k5', frows, si)
        gc_stay = fd_vs_dual(locs0[:, :2], meta[('beta', 0.0)]['res'], locs0,
                             air0, sn, E0, beta, W_A, 'stay', frows, si)
        time_gc = time.time() - tf0
        cG = cnt_delta()
        n_G = cG['n'] - gc0['n']

        # ---- prior-phase joins + asserts ----
        pr = prior.loc[(W_A, si)]
        assert abs(J_A - float(pr['J_fix5'])) < 1e-6, (si, W_A, J_A, pr['J_fix5'])
        assert abs(J0 - float(pr['J0'])) < 1e-6, (si, W_A, J0, pr['J0'])
        J_orc = float(pr['J_oracle'])

        J_comp = min([J_A, J_B, J_C])
        assert J_comp <= J_A + EPS_TIE

        crows.append(dict(
            bw=W_A, si=si, stratum=stratum, t=sn['t'], qlen=sn['qlen'],
            E0_sum=float(E0.sum()), E0_allzero=int(np.all(E0 == 0)),
            n_mu=int(np.sum(base['lam_acc'] > 1e-6)), n_nu=int(np.sum(nu > 1e-6)),
            pool_acc=int(base['nu_acc_pool'] > 1e-6),
            pool_bh=int(base['nu_bh_pool'] > 1e-6),
            mixed=int(base['nu_acc_pool'] > 1e-6 and base['nu_bh_pool'] > 1e-6),
            J0=J0, J_A=J_A, J_B=J_B, J_C=J_C, J_comp=J_comp,
            gain_A=J0 - J_A, gain_B=J0 - J_B, gain_C=J0 - J_C,
            gain_comp=J0 - J_comp,
            B_minus_A=J_A - J_B, C_minus_B=J_B - J_C, C_minus_A=J_A - J_C,
            J_oracle_prior=J_orc,
            B_minus_oracle=J_orc - J_B, C_minus_oracle=J_orc - J_C,
            b_beats_oracle=int(J_B < J_orc - 1e-6),
            c_beats_oracle=int(J_C < J_orc - 1e-6),
            key_A_dir=key_A[0], key_A_rho=key_A[1],
            B_lam=float(key_B[0]) if key_B and key_B[0] != 'fallback_K5' else float('nan'),
            B_rho=float(key_B[1]) if key_B else float('nan'),
            B_nm_ok=nm_ok, c_fallback=c_fallback, c_best_start=best_start,
            c_spread=c_spread, c_std=c_std,
            c_n_failed=int(sum(1 for r in cstart_rows if not r['legal'])),
            c_n_proj=int(sum(r['projected'] for r in cstart_rows)),
            c_n_success0=int(sum(1 for r in cstart_rows if not
                                 str(r['status']).startswith('exception') and
                                 not r['success'])),
            calls_base=n_base, calls_A=n_A, calls_B=n_B, calls_C=n_C,
            calls_G=n_G, calls_total=n_base + n_A + n_B + n_C + n_G,
            time_base=round(time_base, 3), time_A=round(time_A, 3),
            time_B=round(time_B, 3), time_C=round(time_C, 3),
            time_gc=round(time_gc, 3),
            bad_A=bad_A, bad_B=bad_B, bad_C=bad_C,
            inacc_A=inacc_A, inacc_B=inacc_B, inacc_C=inacc_C,
            fd_l2_k5=gc_k5['fd_l2'], fd_cos_k5=gc_k5['fd_cos'],
            fd_maxabs_k5=gc_k5['fd_maxabs'], fd_sign_k5=gc_k5['fd_sign_frac'],
            fd_norm_k5=gc_k5['fd_norm'], dual_norm_k5=gc_k5['dual_norm'],
            fd_l2_stay=gc_stay['fd_l2'], fd_cos_stay=gc_stay['fd_cos'],
            fd_maxabs_stay=gc_stay['fd_maxabs'], fd_sign_stay=gc_stay['fd_sign_frac'],
            fd_norm_stay=gc_stay['fd_norm'], dual_norm_stay=gc_stay['dual_norm']))
        for r in cstart_rows:
            srows.append(r)
        print('  bw=%.0f si=%3d %s | J0=%+8.4f A=%+8.4f B=%+8.4f C=%+8.4f '
              '| calls A/B/C/G=%d/%d/%d/%d fd_k5 l2=%.2e cos=%.4f'
              % (W_A, si, stratum, J0, J_A, J_B, J_C, n_A, n_B, n_C, n_G,
                 gc_k5['fd_l2'], gc_k5['fd_cos']), flush=True)
    print('bw=%.0f done, elapsed %.0fs' % (W_A, time.time() - t_start), flush=True)
    if SMOKE:
        break

cw = pd.DataFrame(crows)
sw = pd.DataFrame(srows)
fw = pd.DataFrame(frows)
cw.to_csv(os.path.join(OUT, 'contpos_results.csv'), index=False)
sw.to_csv(os.path.join(OUT, 'contpos_cstarts.csv'), index=False)
fw.to_csv(os.path.join(OUT, 'contpos_gradcheck.csv'), index=False)

print('\n=== contpos diagnostic: %d rows ===' % len(cw), flush=True)
print('total solves=%d  bad=%d  inacc=%d  wall=%.0fs'
      % (CNT['n'], CNT['bad'], CNT['inacc'], time.time() - t_start), flush=True)
for bw in cw['bw'].unique():
    g = cw[cw['bw'] == bw]
    print('\n--- W_A=%.0f n=%d strata=%s ---' % (bw, len(g),
          g['stratum'].value_counts().to_dict()))
    for c in ('gain_A', 'gain_B', 'gain_C', 'gain_comp', 'B_minus_A',
              'C_minus_B', 'C_minus_oracle'):
        print('  %-16s mean=%+.5f med=%+.5f sum=%+.5f'
              % (c, g[c].mean(), g[c].median(), g[c].sum()))
    print('  b_beats_oracle=%d c_beats_oracle=%d | C fails=%d fallbacks=%d'
          % (g['b_beats_oracle'].sum(), g['c_beats_oracle'].sum(),
             g['c_n_failed'].sum(), int(g['c_fallback'].sum())))
    print('  calls mean: base=%.1f A=%.1f B=%.1f C=%.1f G=%.1f total=%.1f'
          % (g['calls_base'].mean(), g['calls_A'].mean(), g['calls_B'].mean(),
             g['calls_C'].mean(), g['calls_G'].mean(), g['calls_total'].mean()))
    print('  time_s mean: A=%.2f B=%.2f C=%.2f gc=%.2f'
          % (g['time_A'].mean(), g['time_B'].mean(), g['time_C'].mean(),
             g['time_gc'].mean()))
    print('  bad evals A/B/C=%d/%d/%d inacc A/B/C=%d/%d/%d'
          % (g['bad_A'].sum(), g['bad_B'].sum(), g['bad_C'].sum(),
             g['inacc_A'].sum(), g['inacc_B'].sum(), g['inacc_C'].sum()))
    print('  FD vs dual: k5 l2 med=%.3e max=%.3e cos med=%.4f | stay l2 med=%.3e '
          'cos med=%.4f' % (g['fd_l2_k5'].median(), g['fd_l2_k5'].max(),
                            g['fd_cos_k5'].median(), g['fd_l2_stay'].median(),
                            g['fd_cos_stay'].median()))
    print('  C multistart spread med=%.4f max=%.4f | best_start=%s'
          % (g['c_spread'].median(), g['c_spread'].max(),
             g['c_best_start'].value_counts().to_dict()))
    st = g.groupby('stratum')[['gain_A', 'gain_B', 'gain_C', 'C_minus_A']].mean()
    print('  per-stratum mean gains:')
    print(st.to_string(float_format=lambda x: '%+.4f' % x))
print('\nwritten: contpos_selection/_results/_cstarts/_gradcheck.csv  '
      'total solves=%d in %.0fs' % (CNT['n'], time.time() - t_start), flush=True)

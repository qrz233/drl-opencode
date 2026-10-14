# -*- coding: utf-8 -*-
'''
Joint centralized multi-head actor-critic controller (movable airship).

Implements actor-critic部分.md:

Critic (Lyapunov Analytic Critic)
    Per slot a joint candidate set is generated (K total, K = 10):
        a^(0) = 0                    (no-movement guard)
        a^(1) = actor output         (greedy)
        a^(2), a^(3) = directed beta-centroid moves at rho in {0.25, 0.5}:
            uav u moves at rho * v_xy_max toward its beta-weighted cluster
            centroid c_u^beta, airship at rho * v_a_max toward the B_u-weighted
            drone centroid c_a (B_u = sum_{i in I_u} beta_i); z unchanged
        a^(k) = clip(a^(1) + eps),   eps ~ N(0, 0.5^2 I),  k = 4..K-1
    For every candidate the FULL Block A (convex P_R) is solved at the
    candidate positions, then scored
        J^(k) = psi_E * sum_u E_u(t) * P_u^(k)  -  sum_i beta_i(t) * D_i^(k)
    and k* = argmin J is executed: geometry + bandwidth + delivery together.
    beta_i(t) = psi_Q * Q_i(t) + V * w_k / (lambda_k * sum_{j in I+} w_kj),
    psi_Q = 1 / queue_norm (fixed Lyapunov normalisation),
    I+ = the BlockA active set (type >= 0 and Q + A > 0).

Ablation modes (mode=...)
    'full'         K=10 candidates above (learning; default).
    'directed'     no actor: K=3 = [stay, dir0.25, dir0.5].
    'directed_ls'  no actor: K=3 + one compass-search round of +-ls_delta
                   on every action dim around the best base candidate.
    All modes record per-slot stats: win_counts (stay/actor/directed/noise/
    local), gap_best2nd / gap_stay (J differences), a_norm |a*| and v_mean
    (executed horizontal speed).

Energy queue (replaces the old backhaul virtual queue Z)
    E_u(t+1) = max(E_u(t) + P_u*(t) - PW_THRES, 0),   E(0) = 0,
    psi_E = 1 / PW_MAX,  P_u = cal_uav_propulsion_energy(V_xy, V_z)
    (horizontal Zeng model + vertical climb/descend power).

Actor (Centralized Multi-Head Joint Actor)
    Shared CNN encoder phi_H over the per-UAV dual-channel map
    H_u = [backlog channel, pressure channel] (G x G x 2),
    local feature  f_u = [h_u, p_u/boundary, E_u/PW_MAX, (p_a-p_u)/boundary],
    global feature f_G = MLP(concat_u f_u, p_a/boundary),
    heads: mu_U(f_u, f_G) -> (vx,vy,vz) normalized (shared over UAVs),
           mu_A(f_G, p_a) -> (vx,vy) normalized (airship, height fixed).

Training
    Every `training_interval` slots, sample a mini-batch from replay and take
    an Adam step on the supervised MSE between mu(s) and the critic-selected
    normalized joint action a* (3M+2 dims), plus L2 (weight decay).
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from params_sys import (boundary as BOUNDARY, z_min, z_max,
                        uav_speedxy_max, uav_speedz_max,
                        PW_MAX, PW_THRES,
                        slot_len as TAU_DEFAULT)
from utils_sys import cal_uav_propulsion_energy
from blocka_optimizer import blocka_allocate

AIRSHIP_SPEED_MAX = 5.0        # m/s, horizontal only (height fixed)


# ---------------------------------------------------------------------- actor
class JointActor(nn.Module):
    def __init__(self, n_grids=25, n_uavs=4, conv_ch=16, h_dim=32, g_dim=64):
        super().__init__()
        self.n_uavs = n_uavs
        self.h_dim = h_dim
        self.enc = nn.Sequential(
            nn.Conv2d(2, conv_ch, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(conv_ch, conv_ch * 2, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.h_fc = nn.Linear(conv_ch * 2 * 4 * 4, h_dim)          # phi_H -> h_u
        self.f_fc = nn.Linear(h_dim + 3 + 1 + 3, h_dim)            # local feature
        self.g_fc = nn.Sequential(                                 # global feature
            nn.Linear(n_uavs * h_dim + 3, g_dim), nn.ReLU(), nn.Linear(g_dim, g_dim))
        self.mu_u = nn.Sequential(                                 # UAV head (shared)
            nn.Linear(h_dim + g_dim, 32), nn.ReLU(), nn.Linear(32, 3), nn.Tanh())
        self.mu_a = nn.Sequential(                                 # airship head
            nn.Linear(g_dim + 3, 32), nn.ReLU(), nn.Linear(32, 2), nn.Tanh())

    def forward(self, H, uav_locs, airship_loc, E):
        '''
        H          : (B,M,2,G,G) dual-channel maps
        uav_locs   : (B,M,3)
        airship_loc: (B,3)
        E          : (B,M) energy queues
        Returns mu_U (B*M,3), mu_A (B,2) both in [-1,1].
        '''
        B, M = H.shape[0], H.shape[1]
        h = torch.relu(self.h_fc(self.enc(H.view(B * M, *H.shape[2:])).view(B * M, -1)))
        p_u = uav_locs.view(B * M, 3) / BOUNDARY
        p_a = airship_loc / BOUNDARY                                        # (B,3)
        e_hat = (E.view(B * M, 1) / PW_MAX)
        local = torch.cat([h, p_u, e_hat,
                           (p_a.unsqueeze(1) - uav_locs / BOUNDARY).reshape(B * M, 3)], dim=1)
        f_u = torch.relu(self.f_fc(local))                                  # (B*M,h)
        g = self.g_fc(torch.cat([f_u.view(B, -1), p_a], dim=1))             # (B,g)
        g_rep = g.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)
        mu_U = self.mu_u(torch.cat([f_u, g_rep], dim=1))
        mu_A = self.mu_a(torch.cat([g, p_a], dim=1))
        return mu_U, mu_A


# -------------------------------------------------------------------- critic
class JointACCritic:
    def __init__(self, n_users, n_uavs, n_grids,
                 ss_min_types, ss_max_types, mos_weights, type_rates, d0,
                 psi_E=None, E_thres=PW_THRES, tau=TAU_DEFAULT,
        V_lyap=25.0, queue_norm=16.0,
        n_candidates=10, eval_sigma=0.5, airship_movable=True,
        mode='full', ls_delta=0.1,
                 lr=1e-4, batch_size=64, min_samples=16, buffer_size=10000,
                 lambda_a=1.0, weight_decay=1e-5, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.n_users, self.n_uavs, self.n_grids = n_users, n_uavs, n_grids
        self.tau = float(tau)
        self.d0 = float(d0)
        self.V_lyap = float(V_lyap)
        self.queue_norm = float(queue_norm)
        self.psi_Q = 1.0 / float(queue_norm)          # fixed Lyapunov normalisation
        self.ss_min = np.asarray(ss_min_types, dtype=float)
        self.ss_max = np.asarray(ss_max_types, dtype=float)
        self.w = np.asarray(mos_weights, dtype=float)
        self.ld = np.asarray(type_rates, dtype=float)
        self.C1 = (-4.0) / np.log(self.ss_min / self.ss_max)     # per-type log slope
        self.psi_E = (1.0 / PW_MAX) if psi_E is None else float(psi_E)
        self.E_thres = float(E_thres)
        self.n_candidates = int(n_candidates)
        self.eval_sigma = float(eval_sigma)
        self.airship_movable = bool(airship_movable)
        # ablation modes: 'full' = actor + directed + noise (learning);
        # 'directed' = no actor, candidates [stay, dir025, dir05];
        # 'directed_ls' = same base + one compass-search round (no learning)
        assert mode in ('full', 'directed', 'directed_ls'), f'bad mode {mode}'
        self.mode = str(mode)
        self.ls_delta = float(ls_delta)
        self.win_counts = dict(stay=0, actor=0, directed=0, noise=0, local=0)
        self.gap_best2nd, self.gap_stay = [], []   # J gaps per slot
        self.a_norm, self.v_mean = [], []           # |a*| and exec. speed

        self.actor = JointActor(n_grids=n_grids, n_uavs=n_uavs)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size, self.min_samples = batch_size, min_samples
        self.lambda_a = lambda_a

        self.E = np.zeros(n_uavs)            # energy queues (W*slot)
        self.airship_pos = np.zeros(3)       # current airship position
        self._ctx = None                     # per-slot context
        self._state = None                   # decision-time state snapshot

    # ------------------------------------------------------------ slot setup
    def begin_slot(self, user_x, user_y, Q, A, types, cids,
                   uav_locs, airship_pos, E=None):
        '''Compute the Lyapunov priority vector, the dual-channel maps, and
        snapshot the decision-time state.  E defaults to the controller's own
        energy queues.

        beta_i(t) = psi_Q * Q_i(t) + V * w_k / (lambda_k * sum_{j in I+} w_kj),
        psi_Q = 1 / queue_norm (fixed Lyapunov normalisation),
        I+ = the BlockA active set (type >= 0 and Q + A > 0).'''
        types = np.asarray(types)
        demand = np.maximum(Q, 0) + np.maximum(A, 0)
        lam = np.where(types >= 0, self.ld[np.maximum(types, 0).astype(int)], 1.0)
        active = (types >= 0) & (demand > 0)
        if np.any(active):
            w_sum = max(float(np.sum(self.w[types[active].astype(int)])), 1e-12)
            beta = np.where(active,
                            self.psi_Q * np.maximum(Q, 0)
                            + self.V_lyap * self.w[np.maximum(types, 0).astype(int)]
                            / (lam * w_sum),
                            0.0)
        else:
            beta = np.zeros(self.n_users)

        # beta-weighted centroids for the directed candidates:
        #   c_u^beta = sum_{i in I_u, active} beta_i q_i / sum beta_i   (per uav)
        #   c_a      = sum_u B_u q_u / sum_u B_u,   B_u = sum_{i in I_u} beta_i
        user_xy = np.stack([np.asarray(user_x, dtype=float),
                            np.asarray(user_y, dtype=float)], axis=1)   # (N,2)
        q_xy = np.asarray(uav_locs, dtype=float)[:, :2]                 # (M,2)
        cent = q_xy.copy()
        for u in range(self.n_uavs):
            m = (np.asarray(cids) == u) & active
            if not np.any(m):
                continue
            bw = float(beta[m].sum())
            if bw > 1e-12:
                cent[u] = (beta[m, None] * user_xy[m]).sum(axis=0) / bw
        B_u = np.array([float(beta[(np.asarray(cids) == u) & active].sum())
                        for u in range(self.n_uavs)])
        if B_u.sum() > 1e-12:
            air_target = (B_u[:, None] * q_xy).sum(axis=0) / B_u.sum()
        else:
            air_target = np.asarray(airship_pos, dtype=float)[:2].copy()

        self._ctx = dict(user_x=np.asarray(user_x, dtype=float),
                         user_y=np.asarray(user_y, dtype=float),
                         Q=np.asarray(Q, dtype=float),
                         A=np.asarray(A, dtype=float),
                         types=types, cids=np.asarray(cids),
                         demand=demand, beta=beta,
                         centroids=cent, air_target=air_target)
        self._state = dict(H=self._build_maps(user_x, user_y, demand, beta, cids),
                           uav_locs=np.asarray(uav_locs, dtype=float).copy(),
                           airship_pos=np.asarray(airship_pos, dtype=float).copy(),
                           E=(self.E if E is None else np.asarray(E, dtype=float)).copy())
        self.airship_pos = self._state['airship_pos'].copy()
        return beta

    def _build_maps(self, user_x, user_y, demand, beta, cids):
        '''H (M,2,G,G): channel 0 = backlog (Q+A)/queue_norm, channel 1 =
        Lyapunov pressure beta normalised by the slot max.'''
        G = self.n_grids
        H = np.zeros((self.n_uavs, 2, G, G), dtype=np.float32)
        xs = np.clip(np.asarray(user_x), -BOUNDARY + 1, BOUNDARY - 1)
        ys = np.clip(np.asarray(user_y), -BOUNDARY + 1, BOUNDARY - 1)
        gs = 2.0 * BOUNDARY / G
        cols = np.floor((xs + BOUNDARY) / gs).astype(int)
        rows = np.floor((BOUNDARY - ys) / gs).astype(int)
        for u in range(self.n_uavs):
            m = (cids == u) & (demand > 0)
            if not np.any(m):
                continue
            np.add.at(H[u, 0], (rows[m], cols[m]), demand[m] / 16.0)   # queue_norm=16Mb
            np.add.at(H[u, 1], (rows[m], cols[m]), beta[m])
        bmax = float(H[:, 1].max())
        if bmax > 0:
            H[:, 1] /= bmax
        H[:, 0] = np.clip(H[:, 0], 0.0, 10.0)
        return H

    # ------------------------------------------------------------- decisions
    def _eval_candidate(self, a, st, ctx, speed_xy, speed_z):
        '''Solve Block A + score J for one joint candidate action.'''
        locs_k, vel_k, airship_k = self._to_motion(a, st, speed_xy, speed_z)
        P_k = np.array([float(cal_uav_propulsion_energy(V=np.linalg.norm(vel_k[u, :2]),
                                                        Vz=vel_k[u, 2])[0])
                        for u in range(self.n_uavs)])
        res = blocka_allocate(cids=ctx['cids'], user_x=ctx['user_x'],
                              user_y=ctx['user_y'], uav_locs=locs_k,
                              airship_loc=airship_k, queue_Mb=ctx['Q'],
                              traffic_Mb=ctx['A'], types=ctx['types'],
                              beta=ctx['beta'], tau=self.tau)
        power = self.psi_E * float(st['E'] @ P_k)
        betaD = float(ctx['beta'] @ res['D_Mb'])
        return dict(J=power - betaD, power=power, betaD=betaD, a=a,
                    uav_locs=locs_k, velocities=vel_k, airship_pos=airship_k,
                    P=P_k, res=res)

    def decide_and_score(self, speed_xy=uav_speedxy_max, speed_z=uav_speedz_max):
        '''Generate the joint candidates (mode-dependent), solve Block A for
        each, score J, return the best candidate record.

        full:         [stay, actor, dir0.25, dir0.5, noise x6]  (K=10, learning)
        directed:     [stay, dir0.25, dir0.5]                   (no actor)
        directed_ls:  same base + one compass-search round of
                      +-ls_delta on every action dim around the best base
        '''
        ctx, st = self._ctx, self._state
        M = self.n_uavs
        # directed directions: beta-weighted centroid directions with paired
        # amplitudes rho in {0.25, 0.5} (uav toward its beta-centroid of the
        # cluster, airship toward the B_u-weighted drone centroid, B_u = sum beta)
        d_xy = ctx['centroids'] - st['uav_locs'][:, :2]                 # (M,2)
        nrm = np.linalg.norm(d_xy, axis=1, keepdims=True)
        e_u = np.where(nrm > 1e-6, d_xy / np.maximum(nrm, 1e-6), 0.0)
        d_a = ctx['air_target'] - st['airship_pos'][:2]
        na = float(np.linalg.norm(d_a))
        e_a = d_a / na if na > 1e-6 else np.zeros(2)

        cands, cats = [np.zeros(3 * M + 2, dtype=float)], ['stay']
        if self.mode == 'full':
            mu_U, mu_A = self._forward(st)
            a_greedy = np.clip(np.concatenate([mu_U.numpy().ravel(),
                                               mu_A.numpy().ravel()]), -1, 1)
            cands.append(a_greedy)
            cats.append('actor')
        for rho in (0.25, 0.5):
            a_dir = np.zeros(3 * M + 2, dtype=float)
            a_dir[:3 * M] = np.hstack([rho * e_u, np.zeros((M, 1))]).ravel()
            a_dir[3 * M:] = rho * e_a
            cands.append(a_dir)
            cats.append('directed')
        if self.mode == 'full':
            for _ in range(self.n_candidates - 4):
                cands.append(np.clip(a_greedy + self.eval_sigma *
                                     np.random.randn(*a_greedy.shape), -1, 1))
                cats.append('noise')

        evals = [self._eval_candidate(a, st, ctx, speed_xy, speed_z)
                 for a in cands]

        if self.mode == 'directed_ls':
            Js0 = np.array([e['J'] for e in evals])
            center = cands[int(np.argmin(Js0))]
            n_dim = 3 * M + 2
            for i in range(n_dim):
                for s in (+1.0, -1.0):
                    off = np.zeros(n_dim)
                    off[i] = s * self.ls_delta
                    a_ls = np.clip(center + off, -1, 1)
                    cands.append(a_ls)
                    cats.append('local')
                    evals.append(self._eval_candidate(a_ls, st, ctx,
                                                      speed_xy, speed_z))

        Js = np.array([e['J'] for e in evals])
        k_best = int(np.argmin(Js))
        order = np.argsort(Js)
        best = dict(evals[k_best])
        best['k'] = k_best
        best['all_J'] = Js.tolist()
        best['all_comp'] = [(e['power'], e['betaD']) for e in evals]
        best['cands'] = np.stack(cands)
        best['cats'] = list(cats)

        # slot statistics: winner category, J gaps, action magnitude
        self.win_counts[cats[k_best]] += 1
        if len(order) > 1:
            self.gap_best2nd.append(float(Js[order[1]] - Js[order[0]]))
        self.gap_stay.append(float(Js[cats.index('stay')] - Js[k_best]))
        self.a_norm.append(float(np.linalg.norm(best['a'])))
        self.v_mean.append(float(np.mean(np.linalg.norm(
            best['velocities'][:, :2], axis=1))))
        self.last_best = best
        return best

    def _to_motion(self, a, st, speed_xy, speed_z):
        '''normalized joint action (3M+2,) -> candidate positions + velocities.'''
        M = self.n_uavs
        bU = a[:3 * M].reshape(M, 3)
        bA = a[3 * M:]
        v = np.zeros((M, 3))
        for u in range(M):
            bxy = bU[u, :2]
            v[u, :2] = speed_xy * bxy / max(1.0, np.linalg.norm(bxy))
            v[u, 2] = speed_z * bU[u, 2]
        locs = st['uav_locs'] + self.tau * v
        locs[:, 0] = np.clip(locs[:, 0], -BOUNDARY, BOUNDARY)
        locs[:, 1] = np.clip(locs[:, 1], -BOUNDARY, BOUNDARY)
        locs[:, 2] = np.clip(locs[:, 2], z_min, z_max)

        if self.airship_movable:
            va = AIRSHIP_SPEED_MAX * bA / max(1.0, np.linalg.norm(bA))
        else:
            va = np.zeros(2)
        airship = st['airship_pos'] + self.tau * np.array([va[0], va[1], 0.0])
        airship[0] = np.clip(airship[0], -BOUNDARY, BOUNDARY)
        airship[1] = np.clip(airship[1], -BOUNDARY, BOUNDARY)
        airship[2] = st['airship_pos'][2]                      # height fixed
        return locs, v, airship

    def _forward(self, st):
        H = torch.as_tensor(st['H']).unsqueeze(0)
        locs = torch.as_tensor(st['uav_locs'], dtype=torch.float32).unsqueeze(0)
        ap = torch.as_tensor(st['airship_pos'], dtype=torch.float32).unsqueeze(0)
        E = torch.as_tensor(st['E'], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mu_U, mu_A = self.actor(H, locs, ap, E)
        return mu_U, mu_A

    # ----------------------------------------------------------- slot commit
    def commit_slot(self, best, reward=None):
        '''Store (state, critic-selected normalized action) for actor training.
        No-op in ablation modes (no learning).'''
        if self._state is None or self.mode != 'full':
            return
        self.buffer.append(dict(H=self._state['H'],
                                uav_locs=self._state['uav_locs'],
                                airship_pos=self._state['airship_pos'],
                                E=self._state['E'],
                                a_star=best['a'].astype(np.float32)))
        self._state = None
        self._ctx = None

    def update_energy(self, P_exec):
        '''E_u(t+1) = max(E_u + P_u* - P_thres, 0).'''
        self.E = np.maximum(self.E + np.asarray(P_exec, dtype=float) - self.E_thres, 0.0)

    # --------------------------------------------------------------- training
    def train_step(self):
        '''Periodic supervised update: MSE(mu(s), a*) on both heads.'''
        if len(self.buffer) < self.min_samples:
            return None
        idx = np.random.choice(len(self.buffer), size=self.batch_size, replace=True)
        batch = [self.buffer[i] for i in idx]
        M = self.n_uavs

        H = torch.as_tensor(np.stack([b['H'] for b in batch]))            # (B,M,2,G,G)
        B = H.shape[0]
        locs = torch.as_tensor(np.stack([b['uav_locs'] for b in batch]), dtype=torch.float32)
        ap = torch.as_tensor(np.stack([b['airship_pos'] for b in batch]), dtype=torch.float32)
        E = torch.as_tensor(np.stack([b['E'] for b in batch]), dtype=torch.float32)
        a_star = torch.as_tensor(np.stack([b['a_star'] for b in batch]), dtype=torch.float32)

        mu_U, mu_A = self.actor(H, locs, ap, E)
        loss_U = torch.mean((mu_U - a_star[:, :3 * M].reshape(B * M, 3)) ** 2)
        loss_A = torch.mean((mu_A - a_star[:, 3 * M:]) ** 2)
        loss = loss_U + self.lambda_a * loss_A

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

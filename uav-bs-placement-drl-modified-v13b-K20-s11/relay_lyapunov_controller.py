# -*- coding: utf-8 -*-
'''
Modified: Lyapunov-based relay controller for the airship-drone-user architecture.

Components
----------
1. RelayActor (PyTorch): senses the traffic heatmap, its own 3D position, the
   airship position and all backhaul virtual queues Z, outputs (vx, vy, vz)
   normalized actions in [-1, 1]^3 (tanh).
2. Delivery subproblem solver: per drone and per slot, allocate the delivery
   rates D_i to its users under the hard backhaul constraint sum(D) <= tau * R_bh,
   with the upper bound U_i = min(tau * R_access_i, Q_i + A_i). Solved with the
   critical-price bisection + vertex rounding (top up by switching benefit).
3. Backhaul virtual queues Z_u and the Lyapunov drift-minus-reward
       reward = -(drift_Q + drift_Z) + V * MOS,
   the actor is trained by REINFORCE (Gaussian exploration, reward baseline).

Adaptations w.r.t. the original sketch:
- queues/rates are normalized by `queue_norm` so that the drift terms and the
  MOS term are of comparable magnitude;
- the per-user cost g_i(D) uses the same per-type MOS anchors (ss_min/ss_max)
  as the system KPIs: C1_k = b_k = -4/ln(ss_min_k/ss_max_k) > 0, so that
  g_i(D) = (Z_u - Q_i) * D + V * w_k * C1_k / N_active * log(u_val(D))
  is a cost that DECREASES when the user is served (u_val = session time);
- the actor update is REINFORCE with the stored actions re-evaluated under the
  current policy (the sketch did not specify the actor update rule).
'''
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RelayActor(nn.Module):
    '''Actor: heatmap CNN + FC on (uav pos, airship pos, virtual queues Z) -> 3 actions.'''

    def __init__(self, heatmap_size=25, uav_pos_dim=3, airship_dim=3, z_dim=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten()
        )
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, 1, heatmap_size, heatmap_size)).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(conv_out + uav_pos_dim + airship_dim + z_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3), nn.Tanh()
        )

    def forward(self, heatmap, uav_pos, airship_pos, z):
        h = self.conv(heatmap)
        x = torch.cat([h, uav_pos, airship_pos, z], dim=1)
        return self.fc(x)


class RelayLyapunovController:
    '''One controller instance shared by all M relay drones (parameter sharing).'''

    def __init__(self, n_users, n_uavs, boundary, grid_size, queue_norm, V_lyap, tau,
                 ss_min_types, ss_max_types, mos_weights, type_rates, d0,
                 lr=1e-4, batch_size=64, min_samples=256, buffer_size=20000,
                 sigma_max=0.5, sigma_min=0.1, sigma_decay_slots=500, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.n_users, self.n_uavs = n_users, n_uavs
        self.boundary, self.grid_size = float(boundary), float(grid_size)
        self.qn = float(queue_norm)             # normalization for queues/rates
        self.V_lyap = float(V_lyap)
        self.tau = float(tau)
        self.d0 = float(d0)
        self.sigma_max, self.sigma_min = sigma_max, sigma_min
        self.sigma_decay_slots = sigma_decay_slots
        self.ss_min = np.asarray(ss_min_types, dtype=float)
        self.ss_max = np.asarray(ss_max_types, dtype=float)
        self.mos_weights = np.asarray(mos_weights, dtype=float)
        self.type_rates = np.asarray(type_rates, dtype=float)
        # MOS log-slope per type: mos = a - b*log(session)  ->  C1_k = b_k > 0
        self.C1 = (-4.0) / np.log(self.ss_min / self.ss_max)

        self.actor = RelayActor(heatmap_size=grid_size_to_ngrids(boundary, grid_size))
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size, self.min_samples = batch_size, min_samples
        self.sigma_max, self.sigma_min = sigma_max, sigma_min
        self.n_slots_seen = 0

        self.Z = np.zeros(shape=(n_uavs,))      # backhaul virtual queues (Mb)
        self.Z_max = np.zeros(shape=(n_uavs,))  # tracking only

    # ------------------------------------------------------------------ state / action
    def select_actions(self, uav_locs, airship_loc, heatmap, t):
        '''Sample (vx, vy, vz)-normalized actions for all drones from the actor.
        Returns (actions (M,3) raw pre-clip, logps (M,)).'''
        sigma = max(self.sigma_min, self.sigma_max * (1.0 - t / float(self.sigma_decay_slots)))
        self._last_sigma = sigma
        mu = self._forward_actor(uav_locs, airship_loc, heatmap)             # (M, 3)
        noise = sigma * np.random.randn(*mu.shape)
        actions = mu + noise
        logps = -0.5 * np.sum(((actions - mu) / sigma) ** 2, axis=1)         # const dropped
        return actions, logps

    def _forward_actor(self, uav_locs, airship_loc, heatmap):
        hm = torch.as_tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(self.n_uavs, 1, 1, 1)
        up = torch.as_tensor(np.asarray(uav_locs, dtype=np.float32) / self.boundary)
        ap = torch.as_tensor(np.asarray(airship_loc, dtype=np.float32) / self.boundary).unsqueeze(0).repeat(self.n_uavs, 1)
        z = torch.as_tensor((self.Z / self.qn).astype(np.float32)).unsqueeze(0).repeat(self.n_uavs, 1)  # (M, n_uavs): every drone sees all virtual queues
        with torch.no_grad():
            out = self.actor(hm, up, ap, z)
        return out.numpy()

    # ------------------------------------------------------------------ delivery subproblem
    def solve_delivery(self, R_access, R_backhaul, Q, A, cids, active, ld, types):
        '''Inner D subproblem per drone: bisection on the critical price + vertex rounding.

        Returns (D_opt (n_users,), S_u (n_uavs,)) with D in Mb and
        sum(D over drone u) <= tau * R_backhaul[u].'''
        D = np.zeros(shape=(self.n_users,))
        S = np.zeros(shape=(self.n_uavs,))
        for u in range(self.n_uavs):
            ids_u = np.where(cids == u)[0]
            if len(ids_u) == 0:
                continue
            D_u, S_u = self._solve_cluster(u, ids_u, R_access[ids_u], self.tau * R_backhaul[u], Q, A, ld, types)
            D[ids_u] = D_u
            S[u] = S_u
        return D, S

    def _solve_cluster(self, u, ids_u, R_acc_u, cap_u, Q, A, ld, types):
        '''Delivery rates for ONE cluster: U_i = min(tau*R_acc_i, Q_i + A_i), critical-price
        bisection + vertex rounding under cap_u = tau * R_bh. Returns (D_u (n_u,), S_u).'''
        U = np.minimum(self.tau * R_acc_u, Q[ids_u] + A[ids_u])   # (n_u,)
        g0 = self._g_cost(np.zeros_like(U), Q[ids_u], A[ids_u], ld[ids_u], types[ids_u], u, len(ids_u))
        gU = self._g_cost(U, Q[ids_u], A[ids_u], ld[ids_u], types[ids_u], u, len(ids_u))
        mu = np.where(U > 1e-12, (g0 - gU) / np.maximum(U, 1e-12), 0.0)   # cost reduction per Mb

        cap = cap_u
        lo, hi = 0.0, max(np.max(mu), 0.0)
        for _ in range(30):                                               # bisection on mu*
            mid = 0.5 * (lo + hi)
            if np.sum(U[mu > mid]) > cap:
                lo = mid
            else:
                hi = mid
        mu_star = hi
        take = mu > mu_star
        D_u = np.zeros(shape=(len(ids_u),))
        D_u[take] = U[take]

        # vertex rounding: spend leftover backhaul on the best remaining users
        leftover = cap - np.sum(D_u)
        if leftover > 1e-9:
            order = np.argsort(-mu)
            for j in order:
                if D_u[j] > 0 or U[j] <= 1e-12 or mu[j] <= 0:
                    continue
                d_j = min(U[j], leftover)
                D_u[j] = d_j
                leftover -= d_j
                if leftover <= 1e-9:
                    break
        return D_u, float(np.sum(D_u))

    def _g_cost(self, D, Q_i, A_i, ld_i, types_i, u, n_act):
        '''Per-user delivery cost g_i(D) (lower = better). Normalized by queue_norm.'''
        Z_u_n = self.Z[u] / self.qn
        Q_n = Q_i / self.qn
        u_val = np.maximum(self.d0 + (Q_i + A_i - D) / np.maximum(ld_i, 1e-4), 1e-6)
        w = self.mos_weights[types_i.astype(int)]
        c1 = self.C1[types_i.astype(int)]
        return (Z_u_n - Q_n) * (D / self.qn) \
            + self.V_lyap * w * c1 / max(n_act, 1) * np.log(u_val)

    # ------------------------------------------------------------------ queues / reward
    def update_virtual_queues(self, S_u, R_bh):
        '''Z_u <- max(0, Z_u + S_u - tau * R_bh_u). Returns the pre-update Z (for the drift).'''
        Z_prev = self.Z.copy()
        self.Z = np.maximum(self.Z + S_u - self.tau * R_bh, 0.0)
        self.Z_max = np.maximum(self.Z_max, self.Z)
        return Z_prev

    def compute_reward(self, Q, A, D, S_u, R_bh, mos_mean, Z_prev=None):
        '''reward = -(drift_Q + drift_Z) + V * MOS (queues normalized by queue_norm).'''
        if Z_prev is None:
            Z_prev = self.Z
        drift_Q = np.sum((Q / self.qn) * ((A - D) / self.qn))
        drift_Z = np.sum((Z_prev / self.qn) * ((S_u - self.tau * R_bh) / self.qn))
        mos = 0.0 if (mos_mean is None or np.isnan(mos_mean)) else float(mos_mean)
        return -(drift_Q + drift_Z) + self.V_lyap * mos

    # ------------------------------------------------------------------ training
    def store_transition(self, heatmap, uav_locs, airship_loc, actions, logps, reward):
        self.buffer.append({
            'heatmap': np.asarray(heatmap, dtype=np.float32),
            'uav_locs': np.asarray(uav_locs, dtype=np.float32),
            'airship_loc': np.asarray(airship_loc, dtype=np.float32),
            'Z': self.Z.copy(),            # state Z snapshot (post-update state of slot t)
            'actions': np.asarray(actions, dtype=np.float32),
            'logps': np.asarray(logps, dtype=np.float32),
            'sigma': float(getattr(self, '_last_sigma', self.sigma_max)),
            'reward': float(reward),
        })
        self.n_slots_seen += 1

    def train_step(self):
        '''REINFORCE: maximize (R - baseline) * log pi(a|s) over stored slot transitions.'''
        if len(self.buffer) < self.min_samples:
            return None
        idx = np.random.choice(len(self.buffer), size=self.batch_size, replace=True)
        batch = [self.buffer[i] for i in idx]
        rewards = np.array([b['reward'] for b in batch], dtype=np.float32)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        hm = torch.as_tensor(np.stack([b['heatmap'] for b in batch]), dtype=torch.float32).unsqueeze(1)  # (B,1,H,W)
        B, M = hm.shape[0], self.n_uavs
        hm = hm.repeat_interleave(M, dim=0)                                   # (B*M,1,H,W)
        up = torch.as_tensor(np.stack([b['uav_locs'] for b in batch]) / self.boundary, dtype=torch.float32).view(B * M, 3)
        ap = torch.as_tensor(np.stack([b['airship_loc'] for b in batch]) / self.boundary, dtype=torch.float32).unsqueeze(1).repeat_interleave(M, dim=1).view(B * M, 3)
        z = torch.as_tensor(np.stack([b['Z'] for b in batch]) / self.qn, dtype=torch.float32).unsqueeze(1).repeat_interleave(M, dim=1).view(B * M, M)
        a_old = torch.as_tensor(np.stack([b['actions'] for b in batch]), dtype=torch.float32).view(B * M, 3)
        adv_t = torch.as_tensor(np.repeat(adv, M), dtype=torch.float32)

        mu = self.actor(hm, up, ap, z)
        # Gaussian log-prob of the stored actions under the current policy, using the
        # per-transition exploration sigma that was in effect when the action was drawn
        sig = torch.as_tensor(np.repeat(np.array([b['sigma'] for b in batch]), M), dtype=torch.float32).unsqueeze(1)
        logp = -0.5 * torch.sum(((a_old - mu) / sig) ** 2, dim=1)
        loss = -torch.mean(adv_t * logp)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())


def grid_size_to_ngrids(boundary, grid_size):
    '''Same formula as params_sys: n_grids = 1 + floor((2*boundary - 1)/grid_size).'''
    return int(1 + np.floor((2 * boundary - 1) / grid_size))

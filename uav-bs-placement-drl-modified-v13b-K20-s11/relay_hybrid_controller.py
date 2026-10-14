# -*- coding: utf-8 -*-
'''
Modified (v13): hybrid candidate-evaluation controller for the Proposed method.

Per slot and per drone:
  1. the actor proposes K candidate moves (1 greedy + K-1 Gaussian perturbations);
  2. each candidate is scored with the TRUE system model: move the drone to the
     candidate position -> per-user channel gains -> access rates -> inner delivery
     subproblem (bisection + vertex rounding) -> Lyapunov score
         J = -(drift_Q + drift_Z of this drone's cluster) + V * MOS(post-delivery);
  3. the best candidate is EXECUTED (zero-training online optimization, the winning
     mechanism of the original paper, now with the relay/backhaul objective);
  4. training: every sampled candidate is reinforced with advantage
         A_j = (J_j - mean(J)) / (std(J) + eps)      (self-critical style baseline),
     so the actor learns to propose candidates whose score beats its own average.

The scoring state (heatmap, positions, virtual queues Z) is snapshotted at decision
time; the Z used inside the evaluation is the decision-time Z (pre-update).
'''
import numpy as np
import torch
import torch.nn as nn

from relay_lyapunov_controller import RelayLyapunovController


class HybridCandidateController(RelayLyapunovController):

    def __init__(self, n_candidates=10, eval_sigma=0.5, **kwargs):
        super().__init__(**kwargs)
        self.n_candidates = n_candidates
        self.eval_sigma = eval_sigma
        # bound by the notebook (see bind_models)
        self.cal_channel_fading_gain = None
        self.cal_backhaul_capacity_Mb = None
        self.pTx_downlink = None
        self.noise_pw_total = None
        self._slot_entry = None

    # ------------------------------------------------------------------ setup
    def bind_models(self, channel_gain_fn, backhaul_fn, pTx_downlink, noise_pw_total):
        self.cal_channel_fading_gain = channel_gain_fn
        self.cal_backhaul_capacity_Mb = backhaul_fn
        self.pTx_downlink = pTx_downlink
        self.noise_pw_total = noise_pw_total
        self._mp = None          # motion params (set at each decide_and_score)

    # ------------------------------------------------------------------ per-slot flow
    def begin_slot(self, heatmap, uav_locs, airship_loc):
        '''Snapshot the decision-time state once per slot (before any drone moves).'''
        self._slot_entry = {
            'heatmap': np.asarray(heatmap, dtype=np.float32),
            'uav_locs': np.asarray(uav_locs, dtype=np.float32),
            'airship_loc': np.asarray(airship_loc, dtype=np.float32),
            'Z': self.Z.copy(),                                        # decision-time Z
            'sigma': float(self.eval_sigma),
            'cand_actions': np.zeros(shape=(self.n_uavs, self.n_candidates, 3), dtype=np.float32),
            'advs': np.zeros(shape=(self.n_uavs, self.n_candidates), dtype=np.float32),
            'reward': 0.0,
        }

    def decide_and_score(self, k, uav_loc, airship_loc, heatmap, user_x, user_y,
                         bw_alloc_Hz, Q, A, ld, types, cids,
                         speed_xy, speed_z, zmin, zmax, boundary):
        '''Propose K candidates for drone k, score each with the true model, execute best.
        Returns the best raw action (3,).'''
        self._mp = (speed_xy, speed_z, zmin, zmax, boundary)
        mu = self._forward_single(uav_loc, airship_loc, heatmap)          # (3,)
        cands = np.tile(mu, (self.n_candidates, 1))
        cands[1:] += self.eval_sigma * np.random.randn(self.n_candidates - 1, 3)

        ids_k = np.where(cids == k)[0]
        J = np.zeros(shape=(self.n_candidates,))
        if len(ids_k) == 0:
            best = 0                                                      # nothing to serve
            self._slot_entry['cand_actions'][k] = cands                   # advs stay 0 (no signal)
            return cands[best], 0.0
        else:
            for j, c in enumerate(cands):
                (x, y, z), _ = self.motion(c, uav_loc)                    # eval == exec motion
                g = self.cal_channel_fading_gain(user_x[ids_k], user_y[ids_k], x, y, z)
                R_acc = bw_alloc_Hz[ids_k] * np.log2(1 + self.pTx_downlink * g / self.noise_pw_total) \
                    * self.tau / 1e6                                      # Mb per slot
                R_bh_c = self.cal_backhaul_capacity_Mb(x, y, z)
                D_u, S_c = self._solve_cluster(k, ids_k, R_acc, self.tau * R_bh_c, Q, A, ld, types)
                qlen_post = np.maximum(Q[ids_k] + A[ids_k] - D_u, 0.0)
                mos_k = self._mos_post_vec(qlen_post, ld[ids_k], types[ids_k])
                drift_Q = np.sum((Q[ids_k] / self.qn) * ((A[ids_k] - D_u) / self.qn))
                drift_Z = (self.Z[k] / self.qn) * ((S_c - self.tau * R_bh_c) / self.qn)
                J[j] = -(drift_Q + drift_Z) + self.V_lyap * float(np.nanmean(mos_k))
            best = int(np.argmax(J))
            advs = (J - J.mean()) / (J.std() + 1e-6)
            self._slot_entry['cand_actions'][k] = cands
            self._slot_entry['advs'][k] = advs
        return cands[best], float(J[best])

    def commit_slot(self, reward):
        '''Append the slot's candidate record to the replay buffer (call after all drones moved).'''
        if self._slot_entry is not None:
            self._slot_entry['reward'] = float(reward)
            self.buffer.append(self._slot_entry)
            self.n_slots_seen += 1
            self._slot_entry = None

    # ------------------------------------------------------------------ helpers
    def motion(self, c, loc):
        '''Normalized candidate action c (3,) -> (position, velocity).

        Single source of truth used by BOTH candidate evaluation and notebook
        execution, so evaluated and executed positions are identical.
        Hard speed caps: ||v_xy|| <= speed_xy, |v_z| <= speed_z (physical limits);
        position clipped exactly like adjust_location / adjust_altitude.'''
        speed_xy, speed_z, zmin, zmax, boundary = self._mp
        c = np.asarray(c, dtype=float)
        v = np.empty(3)
        nxy = float(np.linalg.norm(c[:2]))
        v[:2] = speed_xy * c[:2] / max(1.0, nxy)
        v[2] = speed_z * c[2] / max(1.0, abs(c[2]))
        p = np.array([np.clip(loc[0] + v[0] * self.tau, -boundary + 1, boundary - 1),
                      np.clip(loc[1] + v[1] * self.tau, -boundary + 1, boundary - 1),
                      np.clip(loc[2] + v[2] * self.tau, zmin, zmax)])
        return p, v

    def _forward_single(self, uav_loc, airship_loc, heatmap):
        hm = torch.as_tensor(heatmap, dtype=torch.float32).view(1, 1, *np.shape(heatmap))
        up = torch.as_tensor(np.asarray(uav_loc, dtype=np.float32) / self.boundary).view(1, 3)
        ap = torch.as_tensor(np.asarray(airship_loc, dtype=np.float32) / self.boundary).view(1, 3)
        z = torch.as_tensor((self.Z / self.qn).astype(np.float32)).view(1, self.n_uavs)
        with torch.no_grad():
            return self.actor(hm, up, ap, z).numpy()[0]

    def _mos_post_vec(self, qlen_post, ld, types_i):
        '''Per-type post-delivery MOS, same anchors as the system KPI mos_func.'''
        ti = types_i.astype(int)
        a = 4.0 * (-np.log(self.ss_min[ti])) / np.log(self.ss_min[ti] / self.ss_max[ti]) + 5.0
        u = np.maximum(self.d0 + qlen_post / np.maximum(ld, 1e-4), 1e-6)
        return np.clip(a - self.C1[ti] * np.log(u), 1.0, 5.0)

    # ------------------------------------------------------------------ training
    def train_step(self):
        '''REINFORCE over stored candidates: maximize A_j * log pi(a_j | s).
        Each slot contributes M*K (drone, candidate) rows sharing one state.'''
        if len(self.buffer) < self.min_samples:
            return None
        idx = np.random.choice(len(self.buffer), size=self.batch_size, replace=True)
        batch = [self.buffer[i] for i in idx]
        B, M, K = len(batch), self.n_uavs, self.n_candidates

        hm_np = np.stack([b['heatmap'] for b in batch])                   # (B,H,W)
        H, W = hm_np.shape[1], hm_np.shape[2]
        hm = torch.as_tensor(np.repeat(hm_np, M * K, axis=0), dtype=torch.float32).view(B * M * K, 1, H, W)
        up = torch.as_tensor(np.stack([b['uav_locs'] for b in batch]).repeat(K, axis=0) / self.boundary,
                             dtype=torch.float32).view(B * M * K, 3)
        ap = torch.as_tensor(np.repeat(np.stack([b['airship_loc'] for b in batch]), M * K, axis=0) / self.boundary,
                             dtype=torch.float32).view(B * M * K, 3)
        z = torch.as_tensor(np.repeat(np.stack([b['Z'] for b in batch]), M * K, axis=0) / self.qn,
                            dtype=torch.float32)                              # (B*M*K, M): full Z vector per row
        a_all = torch.as_tensor(np.stack([b['cand_actions'] for b in batch]), dtype=torch.float32).view(B * M * K, 3)
        advs = np.stack([b['advs'] for b in batch]).ravel()               # (B*M*K,) order (b,m,k)
        sig = np.repeat(np.array([b['sigma'] for b in batch]), M * K)
        adv_t = torch.as_tensor(advs, dtype=torch.float32)
        sig_t = torch.as_tensor(sig, dtype=torch.float32).unsqueeze(1)

        mu = self.actor(hm, up, ap, z)
        logp = -0.5 * torch.sum(((a_all - mu) / sig_t) ** 2, dim=1)
        loss = -torch.mean(adv_t * logp)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

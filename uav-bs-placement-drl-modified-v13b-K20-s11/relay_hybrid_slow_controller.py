# -*- coding: utf-8
'''Hybrid-slow-consistent controller.

Audit finding: the working-tree fix chose option B (exec matched the fast
eval ~30*c). This variant chooses option A:
  - KEEP the original low-speed execution mapping  Δp = c * Δt
    (candidate action treated directly as velocity in m/s);
  - FIX evaluation to use the SAME mapping (eval == exec);
  - unified motion projection enforces physical caps
    (||v_xy|| <= speed_xy, |v_z| <= speed_z, position clips).

Candidate generation, bandwidth pools, delivery subproblem, Lyapunov J and
online REINFORCE training are UNCHANGED from HybridCandidateController.
'''
import numpy as np

from relay_hybrid_controller import HybridCandidateController


class HybridSlowConsistentController(HybridCandidateController):
    '''Hybrid with slow (original-exec) motion, eval-consistent scoring.'''

    def motion(self, c, loc):
        '''Normalized candidate c (3,) -> (position, velocity).

        Slow mapping: v = c (no speed_xy multiplier), then hard-project:
          ||v_xy|| <= speed_xy, |v_z| <= speed_z,
          position clipped like adjust_location / adjust_altitude.
        With tanh actor outputs |c|<=1 this yields |v|<=1 m/s, far below caps.
        '''

        speed_xy, speed_z, zmin, zmax, boundary = self._mp
        c = np.asarray(c, dtype=float)
        v = np.empty(3)
        nxy = float(np.linalg.norm(c[:2]))
        if nxy > speed_xy:                       # rare: noise can exceed cap
            v[:2] = speed_xy * c[:2] / nxy
        else:
            v[:2] = c[:2]
        if abs(c[2]) > speed_z:
            v[2] = speed_z * np.sign(c[2])
        else:
            v[2] = c[2]
        p = np.array([np.clip(loc[0] + v[0] * self.tau, -boundary + 1, boundary - 1),
                      np.clip(loc[1] + v[1] * self.tau, -boundary + 1, boundary - 1),
                      np.clip(loc[2] + v[2] * self.tau, zmin, zmax)])
        return p, v

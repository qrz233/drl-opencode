# -*- coding: utf-8 -*-
'''Patch the lookahead-H3 project: switch cell 26 to LookaheadHybridController (H=3)
and tag the params_sys.py output folder name with -H3.'''
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uav-bs-placement-drl-lookahead-H3')


def replace_once(src, old, new):
    assert src.count(old) == 1, f'pattern not found exactly once: {old!r} (count={src.count(old)})'
    return src.replace(old, new)


# ---------------------------------------------------------------- notebook cell 26
nb_path = os.path.join(ROOT, 'main.ipynb')
nb = json.load(open(nb_path, encoding='utf-8'))
cell = nb['cells'][26]
src = ''.join(cell['source'])

src = replace_once(
    src,
    "# Modified (v13): Proposed method = hybrid candidate evaluation + Lyapunov relay",
    "# Modified (v15): Proposed method = H=3 lookahead candidate evaluation + Lyapunov relay")

src = replace_once(
    src,
    "from relay_hybrid_controller import HybridCandidateController",
    "from relay_lookahead_controller import LookaheadHybridController")

# expected per-slot arrival per type over the frozen-world horizon:
# P(ON) * E[bounded Pareto] with Pareto alpha=2 clipped at 3*mu => E[min(Y,3mu)] = 11/12 * mu
src = replace_once(
    src,
    """# Hybrid Lyapunov relay controller (see relay_hybrid_controller.py)
ctrl = HybridCandidateController(""",
    """# H-step lookahead hybrid controller (see relay_lookahead_controller.py)
# Expected per-slot arrival per type for the frozen-world rollout: P(ON)*E[bounded Pareto],
# with Pareto alpha=2 clipped at 3*mu  =>  E[min(Y, 3*mu)] = 11/12 * mu
p_on = ON_duration_mean_tslot / (ON_duration_mean_tslot + OFF_duration_mean_tslot)
exp_arr_by_type = p_on * (11.0 / 12.0) * np.asarray(ON_data_arrival_mean_Mb, dtype=float)
ctrl = LookaheadHybridController(""")

src = replace_once(
    src,
    "    n_candidates=20, eval_sigma=1.1\n",
    "    n_candidates=20, eval_sigma=1.1,\n    lookahead_H=3, expected_arrival_by_type=exp_arr_by_type\n")

cell['source'] = src.splitlines(keepends=True)
json.dump(nb, open(nb_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('patched notebook cell 26 OK')

# ---------------------------------------------------------------- params_sys.py folder name
p_path = os.path.join(ROOT, 'params_sys.py')
psrc = open(p_path, encoding='utf-8').read()
psrc = replace_once(psrc, 'pureLoS-K20-s1.1, n_slots=', 'pureLoS-K20-s1.1-H3, n_slots=')
open(p_path, 'w', encoding='utf-8').write(psrc)
print('patched params_sys folder_name OK')

# -*- coding: utf-8 -*-
'''Patch main.ipynb (modified project): replace the fixed-per-user / channel-limited
   bandwidth allocation with a per-service-type pool allocation
   (30% Web / 60% Video / 10% Messaging of the 50 MHz pool).
'''
import json, sys

PROJ = r"D:\Learn\work\drl-zcode\uav-bs-placement-drl-modified"
NB = PROJ + r"\main.ipynb"

nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

OLD = """    # (2) Bandwidth Allocation
    bw_alloc_all = allocate_bandwidth_limited(
        mos=mos_all, cids=cids, n_channels=n_channels, n_uavs=n_uavs
    )"""

NEW = """    # (2) Bandwidth Allocation (modified): the 50 MHz pool is split across the 3 service
    # types with shares 30% (Web) / 60% (Video) / 10% (Messaging); in each slot a type's
    # pool is shared equally by its active users. alpha = user bw / bw_per_user (reference),
    # so channel capacity = alpha * bw_per_user * log2(1+SNR) = user_bw * log2(1+SNR).
    bw_alloc_Hz_all = np.zeros(shape=(n_users,))
    ttype_all_t = traffic_type[:, t]
    for k in range(len(bw_type_pool_Hz)): 
        ids_k = np.where(ttype_all_t == k)[0]
        active_k = ids_k[isActive_all[ids_k]]
        if len(active_k) > 0: 
            bw_alloc_Hz_all[active_k] = bw_type_pool_Hz[k] / len(active_k)
    bw_alloc_all = bw_alloc_Hz_all / bw_per_user      # fraction of the reference per-user bw"""

SIM_CELLS = [22, 24, 26, 31, 33]

for i in SIM_CELLS:
    src = "".join(cells[i]["source"])
    n = src.count(OLD)
    assert n == 1, f"cell {i}: expected 1 allocation block, found {n}"
    cells[i]["source"] = src.replace(OLD, NEW).splitlines(keepends=True)

# syntax-check every code cell (skip IPython magics)
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        src_check = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        try:
            compile(src_check, f"<cell {i}>", "exec")
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {i}: {e}")
            sys.exit(1)

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("bandwidth allocation patched OK in cells", SIM_CELLS)

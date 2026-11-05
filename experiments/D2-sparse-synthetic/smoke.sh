#!/bin/bash
cd /tmp/dp-pfn
PY=/home/wlwuser/anaconda3/bin/python
PYTHONUNBUFFERED=1 "$PY" - <<'PY'
import sys
sys.path.insert(0, "experiments/D2-sparse-synthetic")
sys.path.insert(0, "experiments")
import sparse_hist as m
import numpy as np
rng = np.random.default_rng(0)
res = m.run(rng, n=50, eps=0.5, tasks=4, nq=100)
for k in ("oracle", "naive", "noise_aware"):
    v = res[k]
    print(k, "acc %.3f nll %.3f ece %.3f over %+.3f" % (v["acc"], v["nll"], v["ece"], v["overconf"]))
print("occupied", round(res["occupied_cell_fraction"], 3))
PY

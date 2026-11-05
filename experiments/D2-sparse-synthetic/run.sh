#!/bin/bash
cd /tmp/dp-pfn
mkdir -p logs
PY=/home/wlwuser/anaconda3/bin/python
nohup env OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 \
  "$PY" experiments/D2-sparse-synthetic/sparse_hist.py \
  > logs/d2sparse.log 2>&1 &
echo "d2sparse $!"

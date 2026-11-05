#!/bin/bash
cd /tmp/dp-pfn
mkdir -p logs
PY=/home/wlwuser/anaconda3/bin/python
nohup env OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  "$PY" experiments/E4c-lite-mst-margins/e4c_mst.py train --mode aware --steps 4000 --threads 4 \
  > logs/e4c-aware.log 2>&1 &
echo "aware $!"
nohup env OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  "$PY" experiments/E4c-lite-mst-margins/e4c_mst.py train --mode naive --steps 4000 --threads 4 \
  > logs/e4c-naive.log 2>&1 &
echo "naive $!"

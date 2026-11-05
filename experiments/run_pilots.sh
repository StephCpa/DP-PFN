#!/bin/bash
set -u
cd /tmp/dp-pfn
mkdir -p logs
PY=/home/wlwuser/anaconda3/bin/python

nohup env CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 \
  "$PY" experiments/E1-lite-amortization-gap/train.py --steps 6000 --threads 4 \
  > logs/e1-train.log 2>&1 &
echo "e1-train $!"

nohup env OMP_NUM_THREADS=4 \
  "$PY" experiments/X0-theory-audit/audit_constants.py \
  > logs/x0.log 2>&1 &
echo "x0 $!"

nohup env OMP_NUM_THREADS=4 \
  "$PY" experiments/D2-noise-aware-calibration/d2_calibration.py \
  > logs/d2.log 2>&1 &
echo "d2 $!"

nohup env OMP_NUM_THREADS=2 \
  "$PY" experiments/G3-rr-sbm-threshold/rr_sbm.py \
  > logs/g3.log 2>&1 &
echo "g3 $!"

nohup env OMP_NUM_THREADS=8 \
  "$PY" experiments/E3-lite-vote-sample-complexity/vote_c2.py \
  --n 500 --tag _n500 --tasks 16 --queries 200 --reps 40 --fresh 100 \
  > logs/e3.log 2>&1 &
echo "e3 $!"

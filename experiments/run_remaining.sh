#!/bin/bash
set -u
cd /tmp/dp-pfn
PY=/home/wlwuser/anaconda3/bin/python

nohup env CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  "$PY" experiments/E1-lite-amortization-gap/train.py --steps 6000 --threads 4 \
  > logs/e1-train.log 2>&1 &
echo "e1-train $!"

nohup env OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 \
  "$PY" experiments/D2-noise-aware-calibration/d2_calibration.py \
  > logs/d2.log 2>&1 &
echo "d2 $!"

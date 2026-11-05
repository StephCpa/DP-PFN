#!/bin/bash
set -u
cd /tmp/dp-pfn
PY=/home/wlwuser/anaconda3/bin/python
nohup env OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 \
  "$PY" experiments/E1-lite-amortization-gap/evaluate.py \
  --steps-list 300 6000 \
  --etas 0.5 0.1 \
  --ns 4 16 \
  --reps 32 \
  --restarts 6 \
  --search-steps 40 \
  --adv-checkpoints 300 6000 \
  --adv-ns 4 16 \
  --threads 8 \
  > logs/e1-eval.log 2>&1 &
echo "e1-eval $!"

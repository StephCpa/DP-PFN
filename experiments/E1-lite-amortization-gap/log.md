# E1-lite log

Run: gpu-gfkd (`gfkd-NF5468M6`), `/tmp/dp-pfn`, commit `f7ea51f`, 2026-11-05.
Python 3.12.4, torch 2.4.0+cu121, numpy 1.26.4. Training and evaluation ran on CPU
(GPU 6 was reserved but the script does not move the model to CUDA).

## Training

`train.py --steps 6000 --threads 4`. 580,866 parameters. Exact-Bayes validation NLL
0.6667. Final validation NLL 0.6673, excess 0.00057 nat at step 6000 (1845 s).
Matches the previously committed curve. Artifact: `artifacts/E1-lite-amortization-gap/training_curve.json`.

## Evaluation (reduced)

`evaluate.py --steps-list 300 6000 --etas 0.5 0.1 --ns 4 16 --reps 32 --restarts 6 --search-steps 40`.
Full design was 5 checkpoints, reps 64, restarts 12, search-steps 80. This run is smaller
and is a pilot, not the planned E1. Artifact: `evaluation.json`.

Privacy loss below is the max |log ratio| over the flipped row, query and label.
The certificate bound is 2 B_y: 2.20 at eta 0.5, 5.89 at eta 0.1.

| step | eta | n | Lambda median / max | network s, median / max | exact PPD, median / max |
| --- | --- | --- | --- | --- | --- |
| 300 | 0.5 | 4 | 0.141 / 0.320 | 0.381 / 0.541 | 0.206 / 0.223 |
| 300 | 0.5 | 16 | 0.121 / 0.329 | 0.199 / 0.273 | 0.181 / 0.193 |
| 300 | 0.1 | 4 | 0.346 / 0.660 | 0.882 / 1.097 | 0.702 / 0.806 |
| 300 | 0.1 | 16 | 0.494 / 1.097 | 0.500 / 0.606 | 0.514 / 0.645 |
| 6000 | 0.5 | 4 | 0.055 / 0.199 | 0.261 / 0.411 | 0.206 / 0.223 |
| 6000 | 0.5 | 16 | 0.079 / 0.166 | 0.265 / 0.361 | 0.181 / 0.193 |
| 6000 | 0.1 | 4 | 0.236 / 0.634 | 0.913 / 1.527 | 0.702 / 0.806 |
| 6000 | 0.1 | 16 | 0.156 / 0.285 | 0.614 / 0.918 | 0.514 / 0.645 |

Adversarial search, step 6000, max found (6 restarts x 40 Adam steps; a lower bound only):

| eta | n | in-support s_hat | out-of-support s_hat | out-of-support Lambda |
| --- | --- | --- | --- | --- |
| 0.5 | 4 | 0.53 | 1.06 | 0.72 |
| 0.5 | 16 | 0.54 | 0.63 | 0.72 |
| 0.1 | 4 | 1.53 | 1.70 | 1.89 |
| 0.1 | 16 | 1.10 | 1.16 | 2.19 |

## Reading

1. Nothing found exceeds 2 B_y. The floor certificate still holds for this network, on these searches.
2. The exact PPD is far inside that certificate (max about 0.22 at eta 0.5). The trained network's own sensitivity is larger than the exact mechanism's on the same typical datasets: at eta 0.5, n = 16, step 6000, median 0.265 against 0.181.
3. Training tighter does not uniformly reduce leakage. At eta 0.1, n = 4 the max sensitivity on typical data rose from 1.097 at step 300 to 1.527 at step 6000, while median Lambda fell from 0.346 to 0.236. This is the direction of Observation B5, in one cell only.
4. Out-of-support datasets (the case Proposition B2 is about) raise both Lambda and sensitivity. At eta 0.1, n = 16, Lambda reached 2.19 against a typical max of 0.285.

Not measured: n = 64, the other checkpoints, clean-target training (A5), continuous-feature row-DP.

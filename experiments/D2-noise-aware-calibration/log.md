# D2-check log

Run: gpu-gfkd, `/tmp/dp-pfn`, commit `f7ea51f`, 2026-11-05. CPU.
`d2_calibration.py`, seed 0, 200 tasks per cell, 400 queries. Exact grid-Bayes on a 1-D logistic prior, so this measures the target, not a network.

The job was still computing n = 1000 at eps = 1 and eps = 2 when this was written.
Seven of the nine cells had been printed. No `results.json` yet; numbers below are from `logs/d2.log`.

Over-confidence is mean confidence minus accuracy. Proposition D2 predicts it is negative for naive use of label randomized response (under-confident) and positive for naive use of bulk synthetic rows (over-confident).

## Completed cells

| n | eps | RR naive NLL / ECE / overconf | RR noise-aware NLL / ECE / overconf | SYN naive overconf | SYN noise-aware overconf |
| --- | --- | --- | --- | --- | --- |
| 50 | 0.5 | 0.6504 / 0.1373 / -0.0264 | 0.6470 / 0.1364 / +0.0069 | +0.0317 | -0.0220 |
| 50 | 1 | 0.6129 / 0.1199 / -0.0639 | 0.5999 / 0.1023 / -0.0019 | +0.0148 | +0.0018 |
| 50 | 2 | 0.5751 / 0.0879 / -0.0352 | 0.5699 / 0.0825 / +0.0035 | +0.0017 | +0.0013 |
| 200 | 0.5 | 0.6306 / 0.1484 / -0.1270 | 0.5829 / 0.0991 / -0.0069 | -0.0013 | +0.0009 |
| 200 | 1 | 0.5912 / 0.1249 / -0.1087 | 0.5510 / 0.0719 / +0.0015 | +0.0029 | +0.0011 |
| 200 | 2 | 0.5525 / 0.0753 / -0.0561 | 0.5402 / 0.0551 / -0.0040 | +0.0007 | -0.0010 |
| 1000 | 0.5 | 0.6358 / 0.1654 / -0.1586 | 0.5589 / 0.0677 / -0.0010 | -0.0029 | -0.0018 |

Accuracy is essentially unchanged by noise-aware inference (RR at n = 200, eps = 0.5: 0.696 vs 0.698). The gain is calibration and likelihood.

## Reading

Label RR: the prediction holds in all seven cells. Naive over-confidence is between -0.026 and -0.159, and it gets worse as n grows, because the model treats flipped labels as evidence. Noise-aware over-confidence stays within 0.007 of zero, and NLL drops (at n = 1000, eps = 0.5: 0.636 to 0.559).

Synthetic rows: the predicted over-confidence appears only at n = 50 (naive +0.032 and +0.015). By n = 200 it is gone (within 0.003). So D2's synthetic half is not supported in this histogram-release model except at the smallest n. The RR half is the robust finding.

These are exact targets. A trained noise-aware PFN still has to be shown to reach them. That is the actual E4.

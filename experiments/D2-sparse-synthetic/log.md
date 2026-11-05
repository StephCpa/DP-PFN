# D2-sparse log

Run: gpu-gfkd, `/tmp/dp-pfn`, 2026-11-05. CPU, 8 threads.
`sparse_hist.py`, seed 0. Artifact: `artifacts/D2-sparse-synthetic/results.json`.

## Design

512 cells: one binary label and eight binary covariates, of which four are active.
Laplace(2/eps) on the histogram, then 5n synthetic rows. Exact grid Bayes on a
61 x 41 grid over (bias, shared weight). Tasks: 80 at n = 50, 60 at n = 200,
40 at n = 1000. Queries: 400 per task.

The question was whether the over-confidence of naive synthetic-data use, which
died by n = 200 in the 20-cell D2-check, would persist when most cells are empty.

## Results

Occupied-cell fraction: 9% at n = 50, 30% at n = 200, 72% at n = 1000.

| n | eps | naive NLL / overconf | noise-aware NLL / overconf | oracle NLL |
| --- | --- | --- | --- | --- |
| 50 | 0.5 | 0.695 / +0.040 | 0.679 / +0.008 | 0.480 |
| 50 | 1 | 0.671 / -0.061 | 0.629 / -0.042 | 0.477 |
| 50 | 2 | 0.651 / -0.075 | 0.580 / +0.021 | 0.443 |
| 200 | 0.5 | 0.655 / -0.120 | 0.591 / +0.027 | 0.443 |
| 200 | 1 | 0.626 / -0.188 | 0.528 / +0.026 | 0.464 |
| 200 | 2 | 0.567 / -0.157 | 0.475 / +0.003 | 0.450 |
| 1000 | 0.5 | 0.558 / -0.153 | 0.475 / +0.014 | 0.454 |
| 1000 | 1 | 0.501 / -0.115 | 0.443 / -0.008 | 0.435 |
| 1000 | 2 | 0.481 / -0.050 | 0.462 / +0.008 | 0.459 |

## Reading

Naive use is over-confident in one cell only (n = 50, eps = 0.5). In the other
eight it is under-confident, and the bias grows with n before shrinking again
only when eps = 2 and n = 1000. The prediction from the 20-cell model does not
carry over.

The noise-aware target has lower NLL in all nine cells (largest drop 0.098, at
n = 200, eps = 1) and its over-confidence stays inside +/-0.04. Accuracy differs
by at most 0.03. So noise-aware inference still wins, but on likelihood, not on
the sign of the calibration error.

A first smoke run had the label bit reversed in the predictive and gave an oracle
NLL of 0.987. The fix is in `predictive`: bit 0 is the label, cleared for y = 0.
After the fix the oracle NLL is about 0.45 and its over-confidence is about zero,
which is the check that the indexing is right.

# E4c-lite log

Run: gpu-gfkd, `/tmp/dp-pfn`, 2026-11-05. CPU.
Both networks trained 4,000 steps (the E4b checkpoints were not on this machine), then
evaluated on the same 12 cells and seeds as E4b. Artifact: `artifacts/E4c-lite-mst-margins/evaluation.json`.

## Mechanism

Total ε split in half. Feature-label tables use Laplace scale 4d/ε (E4b used 2d/ε).
The 15 tree edges use scale 4(d−1)/ε. The tree is one shared Prim tree per batch,
scored by the product of released label-conditional gaps, so selecting it is
post-processing of the first release.

The first implementation rebuilt a tree inside Python loops for every dataset and
every pair. The aware network made no progress in four minutes. Scoring the pairs
from the released margins and sharing one tree per batch brought it to about 0.34 s
per step.

## Check that the tree carries correlation

With ε = 1e6 (noise negligible) on 2,000 rows: mean absolute feature correlation
0.176 in the real data, 0.034 in the tree sample. The first edge alone recovers
0.143 against a real 0.154. The low average is the rank-2 factor model, in which
most pairs are nearly independent. Label rate matched to 0.001.

## Result

Naive over-confidence against the E4b value, same cell:

| n | eps | k | E4b | E4c | change |
| --- | --- | --- | --- | --- | --- |
| 32 | 1 | 1 | +0.181 | +0.218 | +0.037 |
| 32 | 4 | 1 | +0.113 | +0.174 | +0.061 |
| 32 | 16 | 1 | +0.081 | +0.098 | +0.018 |
| 128 | 1 | 1 | +0.128 | +0.183 | +0.055 |
| 128 | 4 | 1 | +0.094 | +0.103 | +0.009 |
| 128 | 16 | 4 | +0.089 | +0.090 | +0.001 |

It rises in all 12 cells. The aware network's over-confidence stays within ±0.025
(8 of 12 within ±0.01) and its NLL is lower in every cell, by 0.026 to 0.205.

## Reading

At a fixed budget the tree is not worth it for a naive downstream model. The
correlation it could recover is real, but the noise added by splitting the budget
is larger at these ε. Whether MST's own edge selection, which spends budget only on
the strongest pairs, would change this is not tested.

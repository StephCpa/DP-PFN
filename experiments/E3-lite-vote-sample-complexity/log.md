# E3-lite log

Run: gpu-gfkd, `/tmp/dp-pfn`, commit `f7ea51f`, 2026-11-05. CPU, OMP threads 8.
Command: `vote_c2.py --n 500 --tag _n500 --tasks 16 --queries 200 --reps 40 --fresh 100`.
Artifact: `artifacts/E3-lite-vote-sample-complexity/results_n500.json`.
Bayes-optimal accuracy with the true task known: about 0.81 (printed by the script).

## Teacher error at shard size m

Mean pointwise disagreement with the true task's Bayes label, and the infinite-T floor
(share of queries with gamma_m(x*) <= 0):

| teacher | m=4 e / floor | m=16 e / floor | m=64 e / floor |
| --- | --- | --- | --- |
| bayes (idealised PFN) | 0.368 / 0.093 | 0.227 / 0.050 | 0.118 / 0.018 |
| bayes_flat (weaker prior) | 0.393 / 0.103 | 0.233 / 0.033 | 0.118 / 0.014 |
| logreg (trained PATE teacher) | 0.415 / 0.118 | 0.260 / 0.044 | 0.157 / 0.016 |

The floor shrinks with m but does not vanish. At m = 4 a Bayes teacher still has 9% of queries where infinite voting cannot recover the reference label.

## Best shard size, observed vs the C2 formula (n = 500)

| teacher | eps | observed best m | m minimising the C2 bound | m minimising n* |
| --- | --- | --- | --- | --- |
| bayes | 0.25 | 4 | 4 | 4 |
| bayes | 1 | 16 | 8 | 16 |
| bayes | 4 | 32 | 16 | 16 |
| logreg | 0.25 | 8 | 8 | 8 |
| logreg | 1 | 16 | 8 | 16 |
| logreg | 4 | 16 | 16 | 16 |

Agreement is good at small eps and off by one grid step at large eps. The formula is useful but not exact, consistent with the review's note that the Laplace tail bound has 2x slack.

n* at eps = 1: bayes 1356, bayes_flat 1453, logreg 1908. The idealised PFN teacher does lower the predicted sample size relative to logistic regression. The gap is moderate, not an order of magnitude, in this 16-dimensional Gaussian task.

## Vote vs LogPool, best over m, accuracy against true labels

| teacher | eps=0.25 vote / margin / logpool .02 .1 .3 | eps=1 vote / margin / logpool |
| --- | --- | --- |
| bayes | 0.778 / 0.790 / 0.684, 0.718, 0.736 | 0.806 / 0.806 / 0.788, 0.797, 0.800 |
| bayes_hot | 0.778 / 0.790 / 0.765, 0.773, 0.777 | 0.805 / 0.807 / 0.805, 0.807, 0.806 |
| logreg | 0.752 / 0.773 / 0.676, 0.705, 0.720 | 0.801 / 0.804 / 0.784, 0.792, 0.796 |

LogPool does not beat Vote anywhere in this grid. It only ties once the teacher is made over-confident (bayes_hot) and eps is large (eps = 4: 0.812 vs Vote 0.809). The binary margin vote beats report-noisy-max at low eps for free, as the review predicted.

## Reading

H3's qualitative claim holds here: the Bayes teacher, which is more accurate at small m, has a smaller predicted n* than the trained teacher (1356 vs 1908 at eps = 1). The mechanism of Theorem C2 is not just mean error. The infinite-T floor is several points at the shard sizes C2 prefers, so T6 should use the pointwise restatement C2-prime.

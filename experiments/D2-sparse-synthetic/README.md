# D2-sparse: does naive over-confidence survive in a sparse DP histogram?

D2-check found the synthetic-data half of Proposition D2 only at n = 50. By n = 200 it had vanished, in a model with 20 histogram cells and a 2-parameter task. The review (section 4.4) says that setting is too small to decide the question, and asks for sparse cells.

**Question.** When the released histogram has far more cells than records, does treating the resampled synthetic rows as real data stay over-confident as n grows, and does conditioning on the noisy histogram remove it?

**Release.** One binary feature and eight public binary covariates give 2^9 = 512 cells. Each record changes two cells of the one-way-per-label histogram by 1, so the replace-one sensitivity is 2 and the mechanism adds Laplace(2/eps) to every cell. Synthetic rows: draw 5n rows from the clipped, normalised noisy histogram. Empty mass falls back to uniform.

**Predictors**, all exact, so this is about the target and not a network:

- `naive`: grid Bayes on the synthetic rows, using the true logistic task prior. This is "TabPFN on synthetic data".
- `noise_aware`: the same grid Bayes with a Gaussian likelihood for the noisy histogram itself (multinomial variance plus the Laplace variance). The synthetic rows are a post-processing of that histogram, so this is the legal conditioning.
- `oracle`: grid Bayes on the clean records.

**Prediction from D2.** Naive over-confidence (mean confidence minus accuracy) stays positive as n grows whenever the typical cell count stays small, and the noise-aware target brings it near zero at equal or better NLL. If over-confidence still dies by n = 200 here, the synthetic half of D2 is not robust to sparsity either.

```bash
python experiments/D2-sparse-synthetic/sparse_hist.py
```

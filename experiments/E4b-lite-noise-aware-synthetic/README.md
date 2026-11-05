# E4b-lite: trained noise-aware PFN on high-dimensional, marginal-based DP synthetic data (Prop. D2, H4, Gate 2)

**Question.** D2-sparse released a noisy *full* histogram. Laplace noise flattened it toward uniform, so naive use came out under-confident. Its log names the untested case: a generator that keeps low-order margins, as AIM, MST and PrivBayes do, together with a *trained* network. That family drops higher-order structure and emits many synthetic rows, the two mechanisms that could make naive use over-confident. Does naive use turn over-confident here, and does a noise-aware PFN fix it?

**Task prior.** d = 16 binary features, correlated through a 2-factor latent Gaussian (mean |corr| ≈ 0.17). Logistic label with w_j ~ N(0, 0.55²) and b ~ N(0, 0.5²). Large-n logistic accuracy 0.79, naive Bayes 0.73, majority 0.59. The full joint has 2¹⁷ cells.

**Release (ε-DP, replace-one rows).** The d two-way tables (x_j, y) get Laplace(2d/ε) per cell; their joint L1 sensitivity is 2d. Then clip at 0, add 0.5, form the naive-Bayes model p(y) Π_j p(x_j | y), and sample k·n synthetic rows. Feature–feature dependence is not released.

**Networks.** A vector-input version of the E1 PFN (4 layers, d_model = 128), 4,000 steps × 32 tasks:

- `naive`: trained on clean contexts of 16–512 rows, then fed the synthetic rows. This is "TabPFN on DP synthetic data".
- `aware`: trained with the synthesizer inside the prior. Inputs: n ~ U{16..128}, k ∈ {1, 2, 4}, ε ~ log-U[0.5, 32], with (log ε, log₂ k, log n) as tokens. Targets are **clean** query labels.

**Reference predictors.**

- `oracle_pfn`: the naive net on the real records (non-private).
- `nb_plugin`: naive Bayes on the noisy tables, i.e. what the rows encode as k → ∞.
- `nb_clean`: non-private naive Bayes.

No exact target exists here, since the posterior given S is intractable.

**Evaluation.** n ∈ {32, 128}, ε ∈ {1, 4, 16}, k ∈ {1, 4}; 200 tasks × 400 clean test labels per cell. Metrics: accuracy, NLL, ECE, over-confidence.

```bash
python experiments/E4b-lite-noise-aware-synthetic/e4b_syn.py train --mode aware --steps 4000
python experiments/E4b-lite-noise-aware-synthetic/e4b_syn.py train --mode naive --steps 4000
python experiments/E4b-lite-noise-aware-synthetic/e4b_syn.py eval
```

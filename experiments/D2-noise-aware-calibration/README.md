# D2-check: naive vs noise-aware use of a DP release (Thm. D1, Prop. D2, H4, Gate 2)

**Question.** Is the *target* of a noise-aware PFN better calibrated than naive plug-in on the same DP release, in the directions Prop. D2 predicts? Label RR should make naive under-confident; bulk DP synthetic rows should make it over-confident.

All predictors are exact grid-Bayes posteriors on a 1-D logistic prior. This isolates the target from network approximation error. A noise-aware PFN approximates the `noise_aware` column; naive TabPFN on the release approximates `naive`.

- **RR**: label randomized response with ρ = 1/(1+e^ε). Noise-aware likelihood: (1−ρ)σ + ρ(1−σ).
- **SYN**: Laplace(2/ε) on a 10-bin × 2-label histogram, then 5n synthetic rows resampled from it. Noise-aware: a Gaussian-approximate likelihood of the noisy histogram, which the synthetic rows are a post-processing of.

Metrics on clean test labels: accuracy, NLL, ECE and over-confidence (mean confidence − accuracy). n ∈ {50, 200, 1000}, ε ∈ {0.5, 1, 2}, 200 tasks each.

# E3-lite: Theorem C2, shard size, and Vote vs LogPool (H3, T4, T6)

**Question.** Theorem C2 says a teacher that is more accurate at small shard size m lowers the DP sample size n*, and that the best m can be predicted. Is mean teacher accuracy the right quantity? Does C2 predict the best m? When does LogPool beat Vote?

**Setting.** Binary task with a closed-form Bayes posterior predictive:
μ₀, μ₁ ~ N(0, τ²I₁₆), τ = 0.35; x | y ~ N(μ_y, I). The Bayes-optimal error with known μ is about 0.18.

Teachers on T = n/m disjoint shards:

- `bayes`: exact PPD under the prior. This is the idealised PFN: Bayes-optimal *per shard*, calibrated.
- `bayes_hot`: logits ×3, over-confident (same votes; differs only in LogPool).
- `bayes_flat`: Bayes under a much weaker prior (τ = 10). Worse per shard, less shrinkage bias.
- `logreg`: sklearn logistic regression, C = 1. A PATE-style trained teacher with default tuning.

Aggregators, ε-DP per answer under replace-one rows:

- `vote`: report-noisy-max with Lap(2/ε) per count (C1).
- `vote_margin`: Lap(2/ε) on the count difference (binary only; also ε-DP, less noise).
- `logpool_eta`: exponential mechanism on summed floored log-odds (C3), with sensitivity log(1 + 2(1−η)/η).

C2 quantities are measured **pointwise**. e(m; x*) is a teacher's disagreement with the true task's Bayes label at x*, estimated from 100 fresh shards. As T → ∞ at fixed m, Vote returns that label exactly where γ_m(x*) > 0. The *infinite-T floor* is the share of queries with γ_m(x*) ≤ 0.

Sweep: n ∈ {250, 500, 1000}, m ∈ {2, …, 125}, ε ∈ {0.1, 0.25, 0.5, 1, 2}, 16 tasks × 200 queries × 40 noise draws.

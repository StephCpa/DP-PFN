# DP-PFN proposal — reading notes

Source: `DP-PFN Differentially Private Prior-Data Fitted Networks.pdf` (25 pages, 5 Nov 2026, Karen Eze). Extracted text: `dp-pfn-proposal.txt`. Background citations in the PDF are marked "from memory, not re-checked."

## What the proposal is

A 14-month, three-paper plan (tabular, graph, theory) for differentially private prior-data fitted networks. It is a research design, not a completed study. No experiments have been run.

## Setup that everything else depends on

A record is z = (x, y). Context D = (z_1, ..., z_n). Query x*. Prior: φ ~ π, then records i.i.d. from p(z | φ). The PFN q_θ is trained to match the posterior predictive P(y | x*, D) **on average** over prior-drawn datasets (Müller et al. 2022). That average-case objective is why a loss bound cannot certify worst-case DP.

Neighbour relations that must be stated separately:

| Setting | Neighbours differ in | Protects |
| --- | --- | --- |
| Tabular row-DP (replace-one) | one row (x_i, y_i) | a whole record |
| Tabular label-DP | one label y_i | labels; features treated as public |
| Graph edge-DP | one edge | a relationship |
| Graph node-DP | one node, features, label, incident edges | everything about one person |
| Graph label-DP | one node label | labels only |

Proposition 0 fails if (1) weights were fine-tuned on real data (avoid Real-TabPFN-2.5) or (2) preprocessing (standardization, quantiles, PCA, SVD, degree-normalized adjacency) is fit on the private context without being counted.

## Theory map

### A — exact PPD sampling is already DP

Oscillation B = sup_φ (sup_z log p(z|φ) − inf_z log p(z|φ)). If B is finite, one draw of φ ~ π(·|D) or y ~ P(·|x*, D) is 2B-DP (Theorem A2). Standard PFN priors have B = ∞ (unbounded feature densities; class probabilities can approach 0). A floored mixture p_α = (1−α)p + α r, or label floor p_η = (1−η)p + η/K, makes B finite and **can be simulated**, so a PFN can be trained on it. Tempering cannot.

Worked numbers (K = 2): η = 0.5 gives flip rate 25%, B_y = 1.10, ε = 2B_y = 2.20, and after m/n = 0.1 subsampling ε ≈ 0.59 (Theorem A7, Balle et al. 2018).

- **A5.** Train on η-noisy context labels but clean query targets. Sampling the clean-target predictive is post-processing of one posterior draw: 2B_y-label-DP, and the released label is not itself η-noisy.
- **A6.** Chain clean pseudo-labels at public query points. The whole student context costs 2B_y once, independent of m. This is the exact-case contrast with PATE, which pays per label.

These ε values hold only for the **exact** posterior.

### B — amortization gap (the central caveat)

Λ(D) is a sup-log gap between q_θ and the true PPD; Δ(D) is a TV gap. Uniform control transfers the exact DP guarantee (Lemma B1), but training loss never controls that supremum (Proposition B2): on continuous features every dataset is prior-null, so a predictor can match the loss and still reveal a chosen row. Certificate must be the network's own row sensitivity s(θ, n) (Proposition B3), which is expected to be loose at TabPFN scale. Design rule (Corollary B4): put the PFN inside a black-box mechanism, or let it read only a privatized summary.

Observation B5: under a low-noise prior the exact PPD already depends strongly on single rows, so a faithful PFN leaks more. Membership attacks partly measure faithfulness, not a bug. Prediction for E1: better PFNs leak more unless the target itself is private.

### C — black-box mechanisms (guarantee holds for any network)

- **C1 Vote.** Random split into T shards of size m. Each teacher is one forward pass. Vote histogram has ℓ₂ sensitivity √2. Report-noisy-max with Laplace(2/ε) is ε-DP per answer. Data-dependent PATE accounting charges less when teachers agree.
- **C2.** For binary labels, teacher error e(m) and margin γ_m = 1 − 2e(m) set the DP sample size n* = min_m m · T(m), with T ≥ 8 log(2/β) max(1/γ_m², 1/(ε γ_m)). Privacy enters only through T. PFNs are built for small m, which is where this bound rewards accuracy.
- **C3 LogPool.** Bayesian committee machine on floored teacher probabilities, sampled by an exponential mechanism. Sensitivity is log(K/η) per shard change. Keeps confidence; best when teachers are calibrated. When it beats Vote is open (T4).
- **C4 Student.** Label N public or prior-drawn queries once (R2). Later predictions are free. In the exact regime, A6 replaces per-label votes.

### D — noise-aware PFN (cleanest combination)

Put the public mechanism M inside the prior: train q_θ(y* | x*, S) with S ~ M(D). Theorem D1: the minimizer is the private-data PPD; every network inherits M's (ε, δ) by post-processing, so later queries are free; excess risk versus the non-private Bayes predictor is exactly I(y*; D | x*, S). Proposition D2: naive plug-in is miscalibrated — label RR makes a standard PFN under-confident; bulk DP synthetic rows make it over-confident.

Simulatable mechanisms: label RR, feature LDP, noisy marginals, DP synthetic data (AIM/MST/PrivBayes, costly — use a surrogate), noisy graph aggregates and edge RR.

T2 comparison at K = 2, ε ≈ 2.2: central one-draw assumes 25% flips (η = 0.5); local RR actually flips ~10%. Compare on accuracy, not flip rates. OPS randomness also shrinks with n.

### E — graphs

NodePFN layer: H^{ℓ+1} = LayerNorm(H^ℓ + Attn + GCN(H^ℓ, Ã)), Ã symmetric-normalized. Also: optional edge feature-smoothing, 32-member ensemble.

- **G1 label-DP.** Shard labels only. Every teacher keeps the full graph and features. C1/C2 apply. Structural signal is not lost the way tabular sharding loses rows.
- **G2 central edge-DP.** Replace GCN with sum of unit-norm states plus Gaussian noise. One edge changes two rows by at most 1, ℓ₂ sensitivity √2. One forward pass costs L · α/σ² (Rényi) and predicts every node. Four required changes: drop symmetric normalization; reuse one noisy aggregate across the 32 ensemble members; privatize smoothing; retrain noise-aware. This is GAP's aggregation perturbation inside an in-context learner.
- **G3 edge-LDP.** Adjacency RR with ρ = 1/(1+e^ε) maps a cSBM to a cSBM with p' = ρ + (1−2ρ)p. Cost is density: N = 1024, ε = 2 adds ~61,000 fake edges vs ~12,700 real. Heuristic detectability (Prop. G3, not yet rigorous): N ≲ (1−2ρ)²(a−b)² / (K² ρ). Example K=2, a=20, b=4, ε=2 gives N below ~310.
- **G4 node-DP.** Open and hard. Sensitivity grows with d_max. Proposed route: degree-bounded noisy aggregation + shard voting + shard-local attention. Must state whether a node's own features may be used for its own prediction.

## Backbones (Q2)

| | TabPFN v2 / 2.5 | NodePFN (ICLR 2026) |
| --- | --- | --- |
| Synthetic-only weights | default checkpoints yes; not Real-TabPFN-2.5 | yes (~250k synthetic graphs) |
| Context | v2: 10k rows / 500 features; 2.5: 50k / 2,000 | pretraining graphs have 1,024 nodes |
| C1–C4 without retrain | yes, if each teacher fits preprocessing on its own shard | label-DP yes; edge/node no |
| Retrain needed for | floored / clean-target (A5, A6); noise-aware | G2, G3, G4 |

Cheap baseline: GAP noisy aggregates as features for unmodified TabPFN ("GAP features + TabPFN"), to isolate the value of noise-aware NodePFN training.

## Advantages and weaknesses (Q3)

Claimed stronger (status in parentheses): no privacy on training (proved, Prop. 0); no privacy on tuning (by construction; E2 must charge baselines); public knowledge without public data (by construction); smaller DP sample size on small data (proved via C2, measured in E3); one forward pass per teacher; calibrated use of noisy releases (proved under the prior, D1; tested E4); graph label-DP keeps the whole graph; no per-graph training under edge-DP; learned edge-LDP denoising (hypothesis, E6).

Weaker: certification gap; per-query cost in R1; T forward passes until C4 distillation; context limits (sharding only helps wrappers); row-DP needs floored continuous densities; node-DP unsolved; prior misspecification, and adapting the prior to private data would itself cost privacy.

## Plan

Theory order: T4 and T6 first (they set E2/E3); then a clean theorem from B2/B3; graph paper led by T7; theory paper is T1, T2, T3, T5, T8, T9.

Experiments: E1 exact vs amortized (go/no-go for Section 2); E2 main tabular table; E3 sample-complexity vs C2; E4 noise-aware calibration; E5 graph label-DP; E6 edge-DP / edge-LDP; E7 audits (cheap because shadow "models" are forward passes); E8 ablations including prior misspecification.

Gates: (1) E1 shows the predicted gap **and** Vote beats trained-teacher PATE on small tables; (2) E4 improves calibration over naive use; (3) E6 DP-NodePFN competitive with GAP/ProGAP on at least small graphs.

Timeline anchor: about 14 months from November 2026. Tabular paper first. Venues: NeurIPS/ICML/ICLR; USENIX Security/CCS/S&P/PETS/SaTML; COLT/ALT/JPC; LoG/KDD/WWW.

## Risks called out by the authors

Amortization gap likely makes one-draw mechanisms unusable in practice (design already rests on wrappers). TabPFN-scale retrain may be infeasible (use small PFNs or synthetic-only continued pretraining). TabPATE already covers PATE-style tabular ICL — differentiation is B2, C2, D1, LogPool, noise-aware PFNs, and graphs.

## Points to treat carefully

1. The PDF's background reference list was not re-checked by the authors. Constants (2B vs the 4B in Dimitrakakis / Wang et al.) need verification before they are cited as proved.
2. Proposition G3 is explicitly a heuristic. The N ≲ ... threshold is not a theorem yet (that is T7).
3. Theorem C2's Hoeffding argument treats teachers as independent given the data distribution. The split must be data-independent. The bound is binary-label.
4. "Privacy for free" here means weights are free and one exact posterior draw has a finite ε. It does **not** mean a released TabPFN checkpoint is DP on a private table.
5. Gate 1 can fail the Section 2 route while the wrapper route remains the paper. That is the intended fallback, not a collapse of the project.

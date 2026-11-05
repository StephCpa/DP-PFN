# DP-PFN research plan: review and first-round research

Status: **draft**; the pilot results section (§4) is filled in when the runs finish.

Date: 5 Nov 2026. Reviews `DP-PFN Differentially Private Prior-Data Fitted Networks.pdf` (Karen Eze, 5 Nov 2026), as extracted to `dp-pfn-proposal.txt`.

Companion files:
- `literature/related-work-review.md`: citation audit, missing work, novelty.
- `literature/references.bib`: verified bibliography.
- `experiments/*`: code for five pilots.
- `artifacts/*`: results with provenance.

<!-- SUMMARY -->

## 1. What was done

1. **Theory audit.** Every proposition was re-derived by hand. Every number was recomputed by `experiments/X0-theory-audit/audit_constants.py`. Exact privacy losses were computed in a closed-form floored Beta–Bernoulli model.
2. **Literature.** All 40+ references were checked by web search. Closest prior and concurrent work was searched for. The two most load-bearing items, NodePFN and the newly found PrivTab, were checked against their official code.
3. **Four pilots**, each the cheapest test that could falsify a claim:
   - E1-lite: amortization gap.
   - E3-lite: Theorem C2 and Vote vs LogPool.
   - D2-check: noise-aware vs naive calibration.
   - G3-check: edge-RR threshold.

   These are CPU-only with small in-house PFNs or exact Bayes. HuggingFace and arXiv were blocked by the container's network policy, so TabPFN/NodePFN checkpoints and paper PDFs were unavailable.

## 2. Theory audit, claim by claim

Verdicts: **OK** correct as stated · **OK, loose** correct but not tight · **Gap** argument incomplete for what it is used for · **Error** wrong as stated.

| Item | Verdict | Finding |
| --- | --- | --- |
| Prop. 0 (free weights) | OK | Also requires η, T, m, σ and the floor to be fixed without looking at D. The plan says so for T, σ, η. NodePFN's released pipeline violates it (per-dataset settings, see §3). |
| Thm. A2 (2B-DP) | OK, tight for φ | In the floored Beta–Bernoulli model the exact loss of releasing φ ~ π(·\|D) is 2.193 at n = 1000, against 2B_y = 2.197. The bound is attained for the posterior draw. |
| "4B in Dimitrakakis / Wang et al." remark | OK | Consistent with Wang, Fienberg & Smola (2015); PDF not re-read. |
| Lemma A4 | OK, tight | Attained by point-mass p. |
| Table in Sec. 2 (η, flip, B_y, ε, subsampled ε) | OK | All nine numbers reproduce to 3 decimals. |
| Thm. A5 (clean target, 2B_y) | **OK, loose ×2** | The exact loss of the clean-target release tends to **B_y**, not 2B_y (1.097 vs 2.197 at η = 0.5; 2.18 vs 4.39 at η = 0.2). So the ε column overstates A5's cost about 2× in this model. T1 should derive the exact constant. |
| PPD sampling (A2, applied to y) | **OK, very loose** | Releasing one noisy-PPD label costs 0.167 at n = 1 and **0.0034 at n = 1000**. It decays like 1/n because the PPD is a smooth function of the sufficient statistic. The 2B_y worst case needs features that make one row dominate the prediction (E1-lite measures this). |
| Thm. A6 (many labels for one draw) | OK in the exact case | The chain rule argument is right. Once q_θ is approximate, each of the m conditionals adds its own gap, so the "independent of m" cost does not survive amortization. E8 should measure gap growth in m. |
| Thm. A7 (amplification) | OK | Formula is right for sampling without replacement under substitution. Requires the subsample to stay secret and be drawn fresh. |
| Lemma B1 (both parts) | OK | — |
| Prop. B2 | OK | Correct about the *objective*. It says nothing about a particular trained network, which is what E1 must measure. The second paragraph (ν log 1/κ) is the version that matters for continuous networks. |
| Prop. B3 | OK | The attention bound (weights ≤ e^{2R}/n) is right for one readout. **PrivTab** (§3) certifies by a different route: bounded *sums* plus Gaussian noise, with sensitivity O(1). That is simpler than proving O(C/n) for mean attention and is already Lean-verified. |
| Mech. C1 | OK | Report-noisy-max with Lap(2/ε) is ε-DP under replace-one. For binary labels, one Lap(2/ε) on the count margin is also ε-DP with half the noise variance. |
| Thm. C2 | **Gap** | The algebra is right; the Laplace tail even has 2× slack (X0). But the proof needs a *pointwise* error e(m; x*) relative to a fixed reference label, while the text and T6 use a mean error e(m). At fixed m, Vote converges to the reference label only where γ_m(x*) > 0. That leaves a bias floor C2 does not show: in E3-lite it is 13% of queries at m = 2 and 8% at m = 8, even for the exact-Bayes teacher. Corrected statement in §5. The C2 conclusion (better small-m teachers lower n*) does hold in E3-lite, because the Bayes teacher also has the lower floor. But that is an empirical fact about the prior, not a consequence of lower mean error. Binary only, per query. |
| Mech. C3 (LogPool) | OK, loose; overclaim | Privacy is right. The stated width log(K/η) is a valid bound, but the exact width is log(1 + K(1−η)/η), which needs less noise: at K = 2, η = 0.5 it is log 3 vs log 4 (26% less). "Bayes-consistent" overclaims: the committee-machine identity is exact for parameter posteriors, not for predictives, and the floor breaks it further. |
| Mech. C4 | OK | Same design as TabPATE. |
| Thm. D1 (1)–(3) | OK | The conditional-mutual-information identity re-derived. "The PFN attains this price" holds for the loss minimiser only; the network's own gap adds to it. |
| Prop. D2 | OK (direction) | Confirmed in closed form (D2-check, §4). The synthetic-data half is prior art (Räisä et al. 2023, 2025). |
| T2 numbers (10% vs 25%) | OK | Foulds et al. (2016) already show one-posterior-sample is asymptotically inefficient against noisy sufficient statistics plus Bayes. |
| G1 (graph-shard voting) | **Gap** | Privacy is right (one label moves one teacher). "C2 applies unchanged" is wrong for utility: all teachers share the graph and features, and labels are dependent through the graph, so C2's independence step fails. A version conditional on (G, X) is needed. |
| G2 (noisy sum aggregation) | OK | √2 sensitivity is right for an undirected graph with unit-norm states; α/σ² per layer and L layers per pass are right. Scale check (X0): L = 2, ε = 1, δ = 1e-5 needs σ ≈ 9.8 against unit-norm messages. Nodes of degree ≲ 10 get mostly noise, so the noise-aware PFN must learn to fall back on features. |
| G3 numbers (ρ, p′, 61k fake edges) | OK | ρ = 0.1192; 60,922 fake edges added and 1,514 real edges removed at N = 1024, ε = 2. |
| Prop. G3 threshold | **Error in form** | The RR graph is dense (degree ≈ Nρ). The Kesten–Stigum form needs the p′(1−p′) variance, and the dropped lower-order term matters. For the proposal's example the thresholds are 311 (heuristic), 235 (sparse KS, all terms) and **290 (dense)**; at ε = 1 they are 51 / 30 / 59. G3-check (§4) tests which one simulation follows. |
| NodePFN facts (Sec. 7) | **Error** (2 items) | See §3: the "homophily 0.1–0.9" parameter is not edge homophily, and RR moves graphs outside the prior's density range. |
| TREDBench "keeps Prop. 0 intact" | **Gap** | Its synthetic tasks are *selected* by similarity to real datasets. Prop. 0 holds only if the private table is not in that reference set. |
| Sec. 9 "Proposition E3" | Typo | Should be Proposition G3. |

## 3. Literature: what changes

Details: `literature/related-work-review.md`.

1. **PrivTab** (Alrawajfeh, …, Räisä, Honkela; arXiv 2610.10068, Oct 2026). This is the closest concurrent work, and the proposal misses it. Checked in its code:
   - weights trained only on a synthetic prior;
   - three private Perceiver cross-attention layers with tanh-bounded scores and unit-norm values;
   - Lean-proved replace-one sensitivity, then Gaussian noise, composed as μ-GDP;
   - noise-aware pretraining over μ ∈ [0.16, 64];
   - a reusable DP summary with free predictions.

   It is a working, certified answer to T3 and an instance of Section 5. It is also the tabular version of G2. It must be a baseline in E2 and E4, and Corollary B4 needs a third branch: *certified by architecture*.
2. **TabPATE** (cited) covers C1 + C4. **Private PoEtry** (arXiv 2602.05012) covers the product-of-experts aggregator behind C3.
3. **Prior art the proposal should cite:**
   - Foulds et al. 2016 for T2.
   - Räisä et al. 2023/2025 for D2.
   - ALIBI and RR-with-prior as label-DP baselines.
   - Nagler 2023 for B3.
   - Hehir et al. 2022 and Lu & Sen 2023 for T7.
   - LinkTeller for E7.
   - Naor et al. 2023 for R1/R2.
   - Subsample-and-aggregate and Bassily et al. 2018 for C1/C2.
   - G2T-FM: the "GAP features + TabPFN" baseline is a DP G2T-FM.
   - GraphPFN, a second graph PFN backbone.
4. **NodePFN, checked in code:**
   - Pretraining graphs have 1,024 nodes (as stated).
   - The SBM parameter h ∈ [0.1, 0.9] sets p_out = p_in(1−h), so every SBM draw is **assortative**; it is not edge homophily.
   - Prior density ≤ 0.08 in our reconstruction. RR at ε = 2 gives density ≥ 0.119, **outside the prior**.
   - The released inference pipeline **tunes per dataset**: smoothing steps (0–4), TruncatedSVD rank (10–50, fit on all nodes), and ensemble size (1–32). It also normalises with test features. Under DP these are private choices; E5/E6 must fix one public configuration or charge the selection.
5. **Citation fixes:**
   - "Panda et al. 2023" → Wu, Panda, Wang & Mittal, ICLR 2024 (same arXiv id).
   - AMIA's full title adds "…and Protection of High-Risk Queries".
   - Add the TREDBench caveat.

## 4. Pilot results

All numbers come from `artifacts/<id>/*.json`, each with a provenance block (commit, seed, library versions, argv). Provenance is stamped when the file is written. The experiment scripts have not changed since commit `75cb31d` (E1: `32a76df`).

### 4.1 X0: constants and exact one-draw privacy (Sections 2, 4–6)

All proposal constants reproduce: Table A, T2 flip rates, the 60,922 fake edges, ρ = 0.1192. Exact label-DP loss in the floored Beta–Bernoulli model (2B_y = 2.197 at η = 0.5):

| n | posterior draw | clean-target label (A5) | noisy-PPD label |
| ---: | ---: | ---: | ---: |
| 1 | 1.099 | 0.511 | 0.167 |
| 10 | 1.861 | 0.944 | 0.123 |
| 100 | 2.158 | 1.079 | 0.026 |
| 1000 | 2.193 | 1.097 | 0.0034 |

Theorem A2 is tight for the posterior draw. The clean-target release (A5) costs ≈ B_y, half the stated bound. A single noisy-PPD label decays like 1/n when there are no features.

### 4.2 E1-lite: amortization gap (Prop. B2, Obs. B5, H2)

<!-- E1 -->

### 4.3 E3-lite: Theorem C2, PFN vs trained teachers, Vote vs LogPool (H3, T4, T6)

Gaussian class-conditional prior, d = 16. Bayes-optimal accuracy with the task known is 0.82–0.84. 16 tasks × 200 queries × 40 noise draws per cell. Figure: `artifacts/E3-lite-vote-sample-complexity/vote_vs_m_n500.png`.

| | n = 250 | n = 500 | n = 1000 |
| --- | --- | --- | --- |
| Best DP-Vote accuracy, Bayes (ideal PFN) vs trained log. reg., ε = 0.1 | 0.664 vs 0.624 (**+4.1**) | 0.728 vs 0.691 (**+3.7**) | 0.798 vs 0.774 (**+2.3**) |
| same, ε = 0.5 | 0.781 vs 0.751 (+3.0) | 0.799 vs 0.792 (+0.7) | 0.830 vs 0.827 (+0.3) |
| same, ε = 2 | +0.8 | −0.2 | +0.2 |
| Vote floor P(γ_m ≤ 0) at m = 2, Bayes vs log. reg. | 0.14 vs 0.18 | 0.13 vs 0.18 | 0.16 vs 0.20 |
| C2′ argmax m = observed best m (all teachers, 5 ε) | 10/15 | 13/15 | 8/15 |
| C2 n* at β = 0.05 (min over m) | 1,314–4,818 | 1,380–5,077 | 1,176–4,919 |
| LogPool − Vote, calibrated Bayes teachers, ε = 0.1 / 0.5 | −7.5 / −3.3 | −7.3 / −1.0 | −5.1 / −0.7 |
| LogPool − Vote, over-confident teachers, ε = 0.1 / 0.5 | −1.7 / −0.2 | −1.0 / +0.3 | −0.3 / −0.1 |
| Vote with Laplace on the margin − report-noisy-max, ε = 0.1 | +4.2 | +2.9 | +1.2 |

Findings:

- **H3 / Gate 1b direction holds in the idealised setting.** Exact-Bayes (PFN-like) teachers beat trained teachers exactly where DP hurts most: small n·ε. The advantage vanishes once n·ε is large. The trained teacher here uses default C = 1 and is not charged for tuning; tuning would cost budget, and E2 must charge it.
- **C2 needs its pointwise restatement (C2′).** Even exact-Bayes teachers leave a 13–16% floor of queries at m = 2 where Vote converges to the wrong label for any T. The PFN advantage shows up as a *lower floor* at small m, which C2's mean-error form cannot express.
- **C2 is useful for choosing m, not for sizing n.** Its argmax is within a factor of 2 of the observed best m in 42 of 45 configurations, exact in 31. But its n* is 1,176–5,077 at β = 0.05, while Vote is within 1–2 points of its best at n = 250–1,000.
- **T4, first answer.** With *calibrated* small-shard teachers, LogPool loses to Vote at every ε (by up to 7.5 points). The exponential mechanism's noise is scaled to the worst-case log-odds log(1 + K(1−η)/η), and calibrated small-m teachers report small log-odds, so they waste the sensitivity budget. A hard vote always uses its full ±1. Over-confident teachers bring LogPool level with Vote. Calibration, the property the proposal says should favour LogPool, works against it under DP noise. That is the opposite of the proposal's intuition and needs theory (T4). Private PoEtry's reported gains are on LLMs and against different baselines, so they do not contradict this.
- **Free improvement:** for binary labels, Laplace noise on the single vote margin beats report-noisy-max by 1–4 points at ε = 0.1.

### 4.4 D2-check: naive vs noise-aware use of a DP release (Prop. D2, H4, Gate 2)

Exact grid-Bayes, 1-D logistic prior, 200 tasks per cell, clean test labels. "Over-conf." = mean confidence − accuracy.

| Release | n | ε | Naive NLL / ECE / over-conf. | Noise-aware NLL / ECE / over-conf. | Clean oracle NLL |
| --- | ---: | ---: | --- | --- | ---: |
| label RR | 50 | 1 | 0.613 / 0.120 / −0.064 | 0.600 / 0.102 / −0.002 | 0.553 |
| label RR | 200 | 0.5 | 0.631 / 0.148 / **−0.127** | 0.583 / 0.099 / −0.007 | 0.528 |
| label RR | 1000 | 0.5 | 0.636 / 0.165 / **−0.159** | 0.559 / 0.068 / −0.001 | 0.541 |
| label RR | 1000 | 2 | 0.545 / 0.071 / −0.057 | 0.530 / 0.046 / −0.000 | 0.529 |
| DP synthetic (5n rows) | 50 | 0.5 | 0.636 / 0.130 / **+0.032** | 0.625 / 0.124 / −0.022 | 0.540 |
| DP synthetic | 50 | 1 | 0.598 / 0.099 / +0.015 | 0.593 / 0.102 / +0.002 | 0.553 |
| DP synthetic | 1000 | 0.5 | 0.542 / 0.046 / −0.003 | 0.542 / 0.046 / −0.002 | 0.541 |

- **Label RR: Prop. D2 confirmed, strongly.** Naive use is under-confident, and the effect *grows* with n, since bias does not average out. The noise-aware target removes it at equal accuracy and cuts NLL by up to 0.077 nats. Gate 2 should pass for RR, provided the network approximates the noise-aware target.
- **DP synthetic data: direction confirmed, but small here.** Naive use is over-confident only at n = 50, and the effect vanishes by n = 200 in a 2-parameter model with 10 bins. E4 must use settings where it bites: high-dimensional marginals, sparse cells, and many synthetic rows per real row. Otherwise Gate 2 can "fail" for the wrong reason.

### 4.5 G3-check: edge RR and community detectability (Prop. G3, T7)

Spectral recovery on edge-RR SBMs, a = 20, b = 4. Figure: `artifacts/G3-rr-sbm-threshold/threshold_eps2.png`.

| ε | Proposal heuristic | Sparse KS (all terms) | **Dense SNR** | Simulated corr² (N): near / at / past dense threshold |
| ---: | ---: | ---: | ---: | --- |
| 1 | 51 | 30 | **59** | 0.40 (40) / 0.23 (60) / 0.09 (100) |
| 2 | 311 | 235 | **290** | 0.30 (200) / 0.12 (300) / 0.03 (500) |
| 4 | 3,307 | 2,664 | **2,739** | 0.23 (2000) / 0.10 (2500) / 0.01 (3500) |

- **The dense-regime SNR fits.** Simulated corr² tracks 1 − 1/SNR_dense (ε = 2, N = 100: 0.552 vs 0.552), up to finite-size smoothing near the threshold. The sparse KS form is too pessimistic at small ε. The proposal's heuristic is close only by cancellation of dropped terms.
- **Consequence for E6.** At NodePFN's training size (N = 1024) and ε = 2, structure-only signal is essentially gone (overlap 0.04). Edge-LDP DP-NodePFN can only win through node features. The "learned denoising beats Blink" hypothesis must be tested where structure survives: ε ≥ 4, small or dense graphs.

## 5. Recommended changes to the plan

In priority order. Each item says what to change and why.

1. **Reposition the tabular paper against PrivTab, TabPATE and Private PoEtry.**
   - Prop. 0, C1, C3 and C4 are now shared ground.
   - The distinct tabular contribution is the **noise-aware PFN for externally fixed releases**: RR labels, noisy marginals, AIM/MST synthetic data. In that setting the analyst did not choose the mechanism; PrivTab instead designs its own.
   - Put PrivTab in E2 and E4.
   - Consider leading with E4 plus the graph work (G2/G3), where no overlapping work was found. Keep Vote, LogPool and Student as baselines and ablations.
2. **Restate Theorem C2 pointwise and with a bias floor; rebuild T6 and E3 on it.** A corrected form that the existing proof already supports:

   > **C2′.** Fix x* and a reference label ℓ(x*), e.g. the Bayes label of the true task. Let γ_m(x*) = 1 − 2 P_{D_t}[teacher(x*) ≠ ℓ(x*)]. Then
   > P[Vote(x*) ≠ ℓ(x*)] ≤ 1{γ_m(x*) ≤ 0} + exp(−Tγ_m(x*)²/8) + exp(−εTγ_m(x*)/8).
   > Averaging over x*, the excess error of Vote over the reference is at most
   > P_x[γ_m(x) ≤ 0] (bias floor, independent of T and ε) + E_x[exp(−Tγ_m²/8) + exp(−εTγ_m/8); γ_m > 0] (finite-T and noise term).

   n* is governed by the *lower tail* of γ_m(x*), not by the mean error e(m). Because the prior is an input dial (η, prior strength), a PFN teacher's bias can be traded against variance **data-independently, at no privacy cost**. This is a genuine PFN-specific angle for T6.

   In E3-lite, C2′'s bound picks the observed best shard size in 18/20 configurations at n = 500. Its n* is still 3–10× larger than the n at which Vote is already near its best, so T6 should report the bound's slack, not only its argmin.
3. **Fix G1's utility claim.** C2 needs teachers that are independent given the data distribution. Graph teachers share G and X. State the result conditional on (G, X) with a random label partition. Concentration then comes from random-partition (sampling-without-replacement) arguments, not Hoeffding over independent shards.
4. **Replace Prop. G3's heuristic** with the dense-regime signal-to-noise ratio (variance p′(1−p′), all terms kept). Build T7 on Deshpande et al. 2018 and Lu & Sen 2023. Edge RR places graphs in the large-degree regime where those thresholds are rigorous. Cite Hehir et al. 2022 as the closest prior work.
5. **NodePFN pipeline under DP.**
   - Fix one public preprocessing and ensemble configuration for every dataset in E5/E6, or charge the selection (Papernot & Steinke).
   - Fit no SVD or normalisation on query or test features under node- or row-level DP.
   - Correct the homophily statement.
   - Note that RR at ε ≤ 4 leaves the prior's density range, which strengthens the case for noise-aware retraining.
6. **LogPool:** use the exact width log(1 + K(1−η)/η) and drop "Bayes-consistent". Also report Vote with Laplace on the margin (binary), which is free extra utility.
7. **One-draw route (Sec. 2):** report exact constants. The clean-target release costs ≈ B_y here, and a noisy-PPD label decays like 1/n without features. Frame T2 around Foulds et al. And note that A6's m-independence holds only for exact conditionals.
8. **Gate 1 is two gates.** Split it into (a) the E1 amortization gap and (b) Vote beating trained-teacher PATE. They test different routes: Section 2 and Section 4. Either can fail independently. E1-lite results bear on (a); E3-lite on (b).
9. **Citations:** apply the fixes in §3.5 and add the missing work from `literature/related-work-review.md`.

# DP-PFN — related work and citation audit

Date: 2026-11-05. Scope: every reference in the proposal, plus a search for closest prior and concurrent work. BibTeX with per-entry verification status: `references.bib`.

**Method and limits.** Titles, authors and venues were confirmed by web search. arXiv, OpenReview and conference sites could not be fetched from the research container (network policy), so paper full texts were not read. Two key items were instead checked against their **official code**, cloned from GitHub: NodePFN (commit `cac883d`) and PrivTab (commit `26c6b6d`). Statements below say which evidence they rest on.

## 1. Citation audit of the proposal

| Proposal citation | Status | Finding |
| --- | --- | --- |
| Müller et al., PFNs, ICLR 2022 | ok | — |
| Choi, Kim, Kang & Park, NodePFN, ICLR 2026 | **verified, code-checked** | arXiv 2604.19028. Code confirms 1,024-node pretraining graphs and a 50/50 SBM/ER prior. Code contradicts two proposal statements; see §1.1. |
| Prior Labs, TabPFN-2.5 report | verified | arXiv 2511.08667. 50k rows / 2k features confirmed. Real-TabPFN-2.5 is real-data fine-tuned (confirmed). "64 learned thinking rows" not confirmed. |
| GraphPFN (attention masking) | verified | arXiv 2509.21489 (Eremeev et al.). It is a full graph PFN on a LimiX backbone and should be treated as a **second graph backbone and competitor**, not only a masking reference. |
| Wahdany, Jagielski, Cresswell, Dziedzic & Boenisch, TabPATE | verified | arXiv 2606.31474; 2nd ICML Workshop on Foundation Models for Structured Data (2026). Author list as in the proposal. |
| Carvalho & Cordy, AMIA, 2026 | verified | arXiv 2601.15874. Full title adds "…and Protection of High-Risk Queries". |
| Risk In Context, 2025 | verified | arXiv 2507.17066 (Byun, Lin, Ward, Cheng). |
| Xiong, Ju & Zhang, TMLR 2025 | verified | arXiv 2310.12781. |
| Panda et al., DP-ICL, 2023 | **fix** | Same arXiv id 2305.01639. It was published as Wu, Panda, Wang & Mittal, *Privacy-Preserving In-Context Learning for LLMs*, ICLR 2024. Cite that version. |
| Cautionary Tale on DP-ICL evaluation, NeurIPS 2024 workshop | verified | — |
| Carey et al., DP-TabICL, 2024 | verified | arXiv 2403.05681, IEEE BigData 2024. |
| Tran et al., Tab-PE, ICML 2026 | verified | arXiv 2606.08259. |
| GAP, ProGAP, DPDGC, Blink, Kim et al. | ok | — |
| TFM-Playground dataset-inference study | verified | ICML 2026, *Dataset Inference for Data Provenance and Privacy Auditing in Tabular Foundation Models*. |
| TREDBench | **verified, caveat** | arXiv 2603.04692 (Regenwetter et al.). The 200 synthetic pretraining tasks are **selected from 10,000 by embedding similarity to real engineering datasets**. So the weights depend on real reference data. Proposition 0 survives only if the private table is disjoint from that reference set. The proposal says it "keeps Proposition 0 intact" without this condition. Regression only. |
| Background refs (Dimitrakakis; Wang–Fienberg–Smola; Geumlek; Williams–McSherry; Bernstein–Sheldon; Ju; Dwork–Feldman; PATE; Balle; Papernot–Steinke; Kasiviswanathan–Smith; Steinke; Tresp; Fong; Deshpande; Daigavane; PrivGNN; LPGNN; AIM/MST/PrivBayes; DP-EBM) | ok | All exist as cited. The 4B constant of Wang et al. is consistent with the proposal's remark, but the PDF was not re-read. |

Internal typo: Section 9 "Order of work" refers to "Proposition E3"; it means **Proposition G3**.

### 1.1 NodePFN facts the proposal relies on, checked against code

- **Pretraining graph size 1,024: correct.** `bptt = 1024`. Classes are drawn from U{2..20}.
- **"Prior spans homophily 0.1–0.9": misleading.** The parameter `homophily_rate` h ~ U(0.1, 0.9) sets p_out = p_in(1−h), with p_in ~ U(0.01, 0.1). Every SBM draw is therefore *assortative* (block probabilities off the diagonal ≤ p_out ≤ diagonal). Low *edge* homophily arises only through many classes. ER graphs use p ~ U(0.01, 0.05).
- **Edge randomized response leaves the prior.** RR at ε = 2 makes every pair an edge with probability ≥ ρ = 0.119. The densest prior draw has density about 0.08 (artifact `X0-theory-audit/audit.json`). So "RR-shifted homophily stays inside a familiar prior family" is false for density: G3 needs retraining, as the G3 row of the Q2 table already says.
- **"~12,700 real edges": about right.** Reconstructing the prior with balanced classes gives a mean of about 14,600 and a median of about 13,800.
- **Data-dependent, per-dataset-tuned preprocessing** (`run.sh`, `node_classification.py`). Feature smoothing over edges uses `gcn_norm` with self-loops, 0–4 steps. TruncatedSVD is fit on **all** nodes' features, with 10–50 components. Normalisation uses test features (`normalize_with_test=True`). Ensemble size is 1–32. **All four are set per dataset.** Under DP these are tuning choices on private data. The proposal's "no privacy spent on tuning" claim only holds if E5/E6 fix one public configuration, or charge the selection.

## 2. Closest prior and concurrent work missing from the proposal

### 2.1 Direct overlap — must be cited and compared against

1. **PrivTab** (Alrawajfeh, …, Diaconu, Räisä, Honkela et al., arXiv 2610.10068, Oct 2026; code checked). PrivTab is a tabular foundation model trained only on synthetic data, so Proposition 0 applies. Its context encoder is itself a Gaussian mechanism:
   - three Perceiver cross-attention layers with **tanh-bounded scores and unit-norm values**;
   - replace-one sensitivity 2√128 per layer, **formally proved in Lean**;
   - noise σ = 2√384/μ, composed as μ-GDP over three adaptive layers.

   It is pretrained with μ ~ log-U[0.16, 64], i.e. *noise-aware*. It releases a reusable DP summary, and every prediction is post-processing. It is evaluated on 33 TabArena datasets against DP-LR and DP-MLP.

   This occupies a large part of the proposal's space:
   - It answers **T3** ("can a constrained PFN be certified?") positively, by a different route: noisy *sum* aggregation with O(1) sensitivity, instead of the O(C/n) mean-attention bound of Prop. B3.
   - It is a learned-mechanism instance of **Section 5**.
   - It is the tabular analogue of **G2** (noisy sum aggregation inside the PFN).

   Consequences:
   - The design rule of Corollary B4 needs a third branch: *certified-by-architecture* PFNs.
   - PrivTab must be a baseline in E2 and E4.
   - The novelty claims of Sections 3 and 5 must be restated relative to it.
2. **TabPATE** (cited). It covers C1+C4 for tabular ICL, with synthetic queries from the schema.
3. **Private PoEtry** (arXiv 2602.05012, 2026). Private ICL by **product of experts**: clipped log-probabilities summed in log space, (ε, δ)-DP per in-context example. This is the same aggregator family as **LogPool (C3)**. C3's novelty reduces to the PFN-specific parts: the prior-correction term, floors set from the PFN's calibration, and the T4 analysis.

### 2.2 Prior work that answers or sharpens an open question

- **Foulds, Geumlek, Welling & Chaudhuri (UAI 2016).** One posterior sample is asymptotically data-inefficient. Laplace noise on sufficient statistics followed by Bayesian inference is asymptotically efficient. This is prior evidence for **T2**: central one-draw loses to "noisy statistic + noise-aware inference". Combined with our X0 result (the clean-target release costs about B_y, not 2B_y), T2 should be framed as a constant-factor comparison, not an open qualitative question.
- **Räisä, Jälkö, Kaski & Honkela (AISTATS 2023, NAPSU-MQ)**, **Räisä et al. (JMLR 2025)**, and Wilde et al. (AISTATS 2021). These show naive analysis of DP synthetic data is over-confident, and develop noise-aware fixes. This is prior art for **Prop. D2**'s synthetic-data half; D2 should cite it and position the PFN as *amortising* noise-aware inference.
- **Malek Esmaeili et al. (NeurIPS 2021, ALIBI)**: Laplace label noise plus Bayesian denoising, i.e. noise-aware label-DP. **Ghazi et al. (NeurIPS 2021)**: RR-with-prior. Both are label-DP baselines for E4 and E5.
- **Nagler (ICML 2023)** proves a PFN's variance vanishes iff its per-sample sensitivity does. This is the statistical counterpart of **Prop. B3**: s(θ, n) = O(1/n) is both a privacy and a consistency property.
- **Hehir, Slavković & Niu (JPC 2022)**: edge-flip RR plus spectral clustering under edge-LDP. Also **Koskela, Seif & Goldsmith (2025)** and **Lu & Sen (JMLR 2023)**, who settle the cSBM threshold conjecture. This is prior work for **T7**. Deshpande et al.'s cSBM threshold is rigorous in the large-degree limit. Edge RR puts the graph exactly there (degree ≈ Nρ). So the "rigorous version of G3" is closer to a corollary than the proposal suggests, provided the dense-regime variance factor is kept (see `G3-rr-sbm-threshold`).
- **LinkTeller (Wu et al., S&P 2022)**: the standard link-inference attack for E7, plus LapGraph and EdgeRand-style DP-GCN baselines.
- **Private Everlasting Prediction (Naor, Nissim, Stemmer & Yan, NeurIPS 2023)**: theory for answering unboundedly many queries, which frames the R1-vs-R2 trade-off.
- **Subsample-and-aggregate (Nissim, Raskhodnikova & Smith, STOC 2007)** and **model-agnostic private learning (Bassily, Thakkar & Thakurta, NeurIPS 2018)** are the lineage of C1/C2. Theorem C2 is a standard SAA argument; its novelty lies only in the PFN instantiation.
- **G2T-FM (Eremeev et al., NeurIPS 2025)**: TabPFN or LimiX on aggregated neighbour features plus structural encodings. The proposal's "GAP features + TabPFN" baseline is a DP G2T-FM and should be named so.
- **PromptPATE (Duan et al., NeurIPS 2023)**: the design TabPATE follows. Also TuneTables (suggests DP prompt tuning for PFNs) and *Probing Memorization of Tabular ICL* (arXiv 2606.31208, content not checked).

## 3. What remains novel, and a suggested re-positioning

| Proposal component | Status after this review |
| --- | --- |
| Prop. 0 (free weights) | Folklore. Stated by TabPATE and PrivTab. Keep it as setup, not as a contribution. |
| A2/A5/A6 exact one-draw route | Novel in PFN form, but practically weak. Foulds et al. show OPS is inefficient, and E1-lite measures the amortization gap. Best kept as the theory paper's test bed. |
| B2/B3 "no certification from loss" | Conceptually useful and short. PrivTab sidesteps it by architecture, so it no longer carries a paper alone. |
| C1/C2 Vote + sample complexity | C1 is TabPATE/PATE. C2 is a standard SAA bound **with a gap**: it is pointwise, and PFN teachers are biased at small m (see E3-lite). A bias-aware C2 would be new. |
| C3 LogPool | Overlaps Private PoEtry. The T4 analysis and the PFN prior correction remain new. |
| C4 Student | TabPATE. |
| **Section 5 noise-aware PFN for externally fixed releases** (RR labels, noisy marginals, DP synthetic data) | **Still distinct.** PrivTab learns its own mechanism; it does not analyse data someone else released with AIM/MST or RR. This "amortised Bayesian analyst of DP releases" is the strongest remaining tabular contribution, and D2-check supports its premise. |
| **Graphs (G1–G3)** | **No overlapping work found.** PrivTab de-risks G2: noisy sum aggregation plus μ-curriculum training works at TFM scale. G2 is "PrivTab for edges inside NodePFN". |
| G4 node-DP | Open, as the proposal says. |

Suggested ordering: lead the first paper with **noise-aware PFNs for released DP data** (E4) plus **DP-NodePFN under edge-DP / edge-LDP** (E6). Keep Vote/LogPool/Student as baselines and ablations against TabPATE, Private PoEtry and PrivTab, rather than as headline contributions.

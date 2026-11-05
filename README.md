# DP-PFN: Differentially Private Prior-Data Fitted Networks

Research project for **DP-PFN**: using prior-data fitted networks (TabPFN, NodePFN) as differentially private predictors for tabular and graph data.

The source document is `DP-PFN Differentially Private Prior-Data Fitted Networks.pdf` (extracted text: `dp-pfn-proposal.txt`). Dated 5 Nov 2026, author Karen Eze.

## Core claim

PFN weights trained only on synthetic prior data cost **no privacy budget** (Proposition 0). Privacy is spent only on the private context and on what is released from it. A trained PFN **cannot** be certified from its training loss (Proposition B2). Guarantees come from black-box wrappers (Vote, LogPool, Student) or from noise-aware use of an already-private release.

## Research question

How can TabPFN and NodePFN support differentially private tabular and graph prediction so that weights are free, while certificates come from mechanisms that do not depend on amortization error?

## Layout

| Path | Role |
| --- | --- |
| `research-manifest.json` | Research loop state (hypotheses, experiments, phase) |
| `literature/` | Notes, `references.bib`, `related-work-review.md` |
| `reviews/` | Plan reviews (iteration 1: `2026-11-05-plan-review.md`) |
| `experiments/` | One directory per experiment (design, code, results) |
| `artifacts/` | Versioned outputs with provenance |
| `data/` | Local data only; not committed |
| `manuscript/` | Drafts |

## Three release modes

- **R1** prediction API: each query spends budget (composes over k).
- **R2** student-context release: one privately labelled context, then free post-processing.
- **R3** noise-aware release: publish S = M(D), then a PFN trained to read S.

## Status of the plan

Iteration 1 (5 Nov 2026) is done: a plan review plus first-round research. Start with **`reviews/2026-11-05-plan-review.md`**.

- **Theory audit** (`experiments/X0-theory-audit`): all constants reproduce. Fixes are needed to Thm. C2 (pointwise restatement with a bias floor), Prop. G3 (dense-regime threshold), G1 (teacher independence), and the LogPool width. Thm. A5's ε is 2× loose.
- **Literature** (`literature/`): verified bibliography. Closest concurrent work: **PrivTab** (a certified, noise-aware tabular foundation model), TabPATE and Private PoEtry. NodePFN and TabPFN facts were checked against their code.
- **Pilots** (CPU, closed-form priors), results in `artifacts/`:
  - E1-lite: amortization gap.
  - E3-lite: C2, PFN vs trained teachers, Vote vs LogPool.
  - D2-check: noise-aware calibration.
  - G3-check: edge-RR threshold.
  - E4-lite: a trained noise-aware PFN on label RR. It recovers 71–86% of the exact calibration gain over naive TabPFN-style use, so Gate 2 passes in pilot form for label RR.
  - D2-sparse and E4b-lite: on DP synthetic data, naive use is under-confident for a noisy full histogram but over-confident for a marginal-based synthesizer (12/12 cells). A trained noise-aware PFN is calibrated and lowers NLL in every E4b cell.

T1–T9 and the full E1–E8 (TabPFN/NodePFN scale, real data) remain to do. Three go/no-go gates (E1, E4, E6) decide whether later phases proceed; the review recommends splitting Gate 1 in two.

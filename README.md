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
| `literature/` | Notes and `references.bib` |
| `experiments/` | One directory per experiment (design, code, results) |
| `artifacts/` | Versioned outputs with provenance |
| `data/` | Local data only; not committed |
| `manuscript/` | Drafts |

## Three release modes

- **R1** prediction API: each query spends budget (composes over k).
- **R2** student-context release: one privately labelled context, then free post-processing.
- **R3** noise-aware release: publish S = M(D), then a PFN trained to read S.

## Status of the plan

This repository currently holds the research proposal and the project scaffold. Theory items T1–T9 and experiments E1–E8 are planned, not yet executed. Three go/no-go gates (E1, E4, E6) decide whether later phases proceed.

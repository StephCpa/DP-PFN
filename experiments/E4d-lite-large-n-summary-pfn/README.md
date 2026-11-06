# E4d-lite: a noise-aware PFN for large n that reads the release itself (Thm. D1, H4)

**Question.** The noise-aware PFNs of E4b and E4c′ read synthetic rows and were trained for n ≤ 128. The n·ε sweep (E4c′, §4.11) found that a private MST tree starts to pay for the *release's own classifier* at n·ε ≈ 8,000. What does a noise-aware analyst achieve at large n? Does it beat the release's own classifier, does it remove the structural over-confidence of plain margins, and which release does it prefer?

**Why tables, not rows.** The releases are noisy count tables; synthetic rows are a post-processing of them. A row-reading PFN costs O(rows²) in attention, and at large n a few hundred synthetic rows carry less information than the tables. Conditioning on the tables has the same target, the private-data PPD of Thm. D1, and costs the same at any n. This is the "statistic tokens" reader that proposal Section 5 lists for noisy marginals.

**Releases.** As in E4c′:

- `nb`: Laplace(2d/ε) on the 16 (x_j, y) tables.
- `mst_dp` (tree): ε/3 on the tables, ε/3 on a private per-dataset Prim tree, ε/3 on the 15 (x_parent, x_child, y) tables.

The raw noisy counts, unclipped, are what the PFN reads.

**Network (`SumPFN`, one model for both releases).**

- Memory tokens:
  - a meta token (log ε, log n, release type);
  - 16 label-table tokens (counts/n, noise scale/n, feature-id embedding);
  - 15 edge tokens (counts/n, scale/n, parent- and child-id embeddings; masked for `nb`);
  - 2 self-attention layers over all of them.
- Each query is 16 per-feature tokens (value plus feature-id embedding) and a CLS token. 3 decoder layers cross-attend to the memory.
- Training: 5,000 steps × 32 tasks × 16 clean query labels; n ~ log-U[32, 8192], ε ~ log-U[0.5, 64], release type 50/50.

**Evaluation.** The sweep's grid, n ∈ {32 … 8192} × ε ∈ {1, 4, 16, 64}, with 200 tasks × 400 clean test labels per cell. Predictors:

- `sum_aware_{nb,tree}`: the new network on each release.
- `plugin_{nb,tree}`: each release's own classifier (naive Bayes / TAN).
- `clean_{nb,tan}`: noise-free ceilings.
- `clean_pfn` (n ≤ 512): E4b's PFN on the real records.
- `row_aware_{nb,tree}` (n ≤ 128, k = 1): the earlier row-reading noise-aware PFNs, as a continuity check.

```bash
python experiments/E4d-lite-large-n-summary-pfn/e4d_sum.py train --steps 5000 --n-qry 16
python experiments/E4d-lite-large-n-summary-pfn/e4d_sum.py eval
```

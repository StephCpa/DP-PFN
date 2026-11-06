# E4d-lite: a noise-aware PFN for large n that reads the release itself (Thm. D1, H4)

**Question.** The noise-aware PFNs of E4b and E4c′ read synthetic rows and were trained for n ≤ 128. The n·ε sweep (E4c′, §4.11) found that a private MST tree starts to pay for the *release's own classifier* at n·ε ≈ 8,000. What does a noise-aware analyst achieve at large n? Does it beat the release's own classifier, does it remove the structural over-confidence of plain margins, and which release does it prefer?

**Why tables, not rows.** The releases are noisy count tables; synthetic rows are a post-processing of them. A row-reading PFN costs O(rows²) in attention, and at large n a few hundred synthetic rows carry less information than the tables. Conditioning on the tables has the same target, the private-data PPD of Thm. D1, and costs the same at any n. This is the "statistic tokens" reader that proposal Section 5 lists for noisy marginals.

**Releases.** As in E4c′:

- `nb`: Laplace(2d/ε) on the 16 (x_j, y) tables.
- `mst_dp` (tree): ε/3 on the tables, ε/3 on a private per-dataset Prim tree, ε/3 on the 15 (x_parent, x_child, y) tables.

The raw noisy counts, unclipped, are what the PFN reads.

**Network.** One model covers both releases. Two architectures were tried (`--arch`):

- **`xattn` (v1, failed).**
  - Memory: a meta token, 16 label-table tokens and 15 edge tokens, matched to the query only by learned feature-id embeddings.
  - Each query: 16 per-feature tokens plus a CLS token, cross-attending to the memory.
  - It did not learn the routing in 5,000 steps. NLL stayed near the base rate (0.665–0.693) in every cell: calibrated but uninformative.
  - Kept for the record: `artifacts/E4d-lite-large-n-summary-pfn/{training,evaluation}_v1_cross_attention.json`.
- **`aligned` (v2, used).** One token per feature j containing:
  - the query value x*_j;
  - label table j (counts/n through asinh, log noise scale);
  - for a tree release, the parent's value x*_par(j) and the (parent, j, y) table;
  - the release's meta features.

  A meta token and a CLS token are added, then 3 self-attention layers run across the 18 tokens. This builds in the naive-Bayes / TAN factorisation over features and is equivariant to feature order, like the prior. 6,000 steps × 32 tasks × 16 clean query labels; n ~ log-U[32, 8192], ε ~ log-U[0.5, 64], release type 50/50.

**Evaluation.** The sweep's grid, n ∈ {32 … 8192} × ε ∈ {1, 4, 16, 64}, with 200 tasks × 400 clean test labels per cell. Predictors:

- `sum_aware_{nb,tree}`: the new network on each release.
- `plugin_{nb,tree}`: each release's own classifier (naive Bayes / TAN).
- `clean_{nb,tan}`: noise-free ceilings.
- `clean_pfn` (n ≤ 512): E4b's PFN on the real records.
- `row_aware_{nb,tree}` (n ≤ 128, k = 1): the earlier row-reading noise-aware PFNs, as a continuity check.

```bash
python experiments/E4d-lite-large-n-summary-pfn/e4d_sum.py train --arch aligned --steps 6000 --n-qry 16
python experiments/E4d-lite-large-n-summary-pfn/e4d_sum.py eval --arch aligned
```

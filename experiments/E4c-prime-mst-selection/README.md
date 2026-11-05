# E4c′: a per-dataset, privately selected MST tree of feature pairs (Prop. D2, H4)

**Question.** E4c-lite added 15 feature-pair tables at a fixed budget, and naive over-confidence rose in 8 of 12 cells. But its tree was a star scored by label informativeness, shared across a batch, and its edges were no more correlated than random pairs. Does a tree chosen *per dataset by feature–feature dependence*, privately as MST does, change that? And would a perfectly chosen tree help if selection were free?

**Releases.** Total budget ε, replace-one rows, built from the *same data* in every cell:

| Release | Budget use | DP? |
| --- | --- | --- |
| `nb` (E4b) | ε on the 16 (x_j, y) tables; naive-Bayes synthesis | yes |
| `star` (E4c) | ε/2 on (x_j, y) tables; ε/2 on 15 (x_u, x_v, y) tables along a batch-shared star scored by label gaps | yes |
| `mst_dp` (new) | ε/3 on (x_j, y) tables; ε/3 on selection (Prim from node 0, 15 exponential-mechanism picks at ε/45, utility q(u,v) = 4·\|N₁₁ − N₁·N·₁/n\|, sensitivity 4); ε/3 on the 15 selected (x_u, x_v, y) tables | yes |
| `tree_oracle` (diagnostic) | per-dataset maximum spanning tree on the clean q, then ε/2 + ε/2 on tables | **no**: the selection uses the clean data |

Tree releases synthesize as a tree-augmented naive Bayes: the label, then the root given y, then each new node given its tree parent and y, in Prim order.

The sensitivity of q is 4. For a 2×2 table, q = 4·|D₁₁| with D₁₁ = N₁₁ − N₁·N·₁/n, and replacing one row moves D₁₁ by at most 1. A brute-force check over 20,000 random replacements per n found a maximum of 3.84.

**Networks.**

- `naive`: E4b's clean-trained network, reused. Its `naive_nb` numbers reproduce E4b's artifact exactly, which checks that the cells are paired.
- `aware_mst`: trained here with `mst_dp` in the prior (E4b recipe, 4,000 steps).
- `aware_nb`: E4b's noise-aware network.

**Selection quality.** For each release, the mean population |corr| of its chosen edges (from 4,000 fresh rows per task). It is compared with all pairs and with the best population tree.

```bash
python experiments/E4c-prime-mst-selection/e4cp_mst.py train --steps 4000
python experiments/E4c-prime-mst-selection/e4cp_mst.py eval
```

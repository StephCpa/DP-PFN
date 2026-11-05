# E4c-lite: does releasing feature-feature margins shrink naive over-confidence?

E4b-lite released only the (feature, label) tables. A learner then treats features as
independent, and naive use was over-confident in all 12 cells. This run gives the
generator one more thing: a maximum spanning tree of feature pairs, which is the
structure MST uses, at the **same total privacy budget**.

**Budget split.** Total ε is fixed. Half the budget pays for the d feature-label
tables (scale 4d/ε instead of E4b's 2d/ε). The other half pays for the d − 1 tree
edges. Each selected pair table has 8 cells and replace-one sensitivity 2, so with
d − 1 of them released together the noise scale is 4(d − 1)/ε.

**Tree.** Edges are scored by the total variation between the noisy pair table and
the product of its noisy single-feature margins. The maximum spanning tree is then
released. Edge selection uses noisy scores, so it is part of the mechanism and is
replayed inside the prior. It is not a privacy leak.

**Sampling.** Labels come from the feature-label tables as in E4b. Features are then
drawn from the tree: the first feature from its label-conditional margin, each later
feature from the released pair table conditioned on its parent and on the label.

**Prediction.** Naive over-confidence should fall as the tree captures the
correlation that E4b dropped. It should not fall to zero, because a tree cannot
represent the rank-2 factor model that generates the features. The noise-aware
network, trained with this generator inside its prior, should stay calibrated and
keep a lower NLL. If the over-confidence instead grows, the extra noise from
splitting the budget dominates the recovered correlation.

```bash
python experiments/E4c-lite-mst-margins/e4c_mst.py train --mode aware --steps 4000 --threads 4
python experiments/E4c-lite-mst-margins/e4c_mst.py train --mode naive --steps 4000 --threads 4
python experiments/E4c-lite-mst-margins/e4c_mst.py eval
```

The naive network is trained on clean data, exactly as E4b's naive network is, and
is then shown the tree-based synthetic rows. The aware network sees the tree release
during training.

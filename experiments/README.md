# Experiments

One directory per experiment. Code is committed. Raw outputs and checkpoints go to `results/` (gitignored). Summaries with provenance (commit, seed, library versions, argv) go to `artifacts/<id>/`.

These are **"-lite" pilots**. Each tests one claim of the proposal in the cheapest setting that can falsify it: closed-form priors, CPU only, no released checkpoints. HuggingFace was blocked from the container, so TabPFN/NodePFN weights could not be downloaded. They de-risk the planned experiments E1–E8. They do not replace them.

| id | Tests | Proposal item | Runtime (4 CPU) |
| --- | --- | --- | --- |
| `X0-theory-audit` | every constant; exact privacy loss of one-draw releases | Sec. 2, 4, 5, 6 | <1 min |
| `E1-lite-amortization-gap` | Λ(D) and s(θ, n) of a trained PFN vs the exact mechanism | Prop. B2, Obs. B5, H2, Gate 1a | ~30 min train + ~15 min eval |
| `E3-lite-vote-sample-complexity` | Thm. C2 prediction of n* and best shard size; Vote vs LogPool | Thm. C2, T4, T6, H3 | ~30 min |
| `D2-noise-aware-calibration` | naive vs noise-aware use of label-RR and DP synthetic data | Thm. D1, Prop. D2, H4, Gate 2 | ~10 min |
| `G3-rr-sbm-threshold` | where edge RR destroys community signal | Prop. G3, T7 | ~10 min |
| `E4-lite-noise-aware-rr` | does a *trained* noise-aware PFN reach the calibrated target on label-RR data? | Thm. D1, Prop. D2, H4, Gate 2 | ~30 min train + ~5 min eval |
| `D2-sparse-synthetic` | does naive over-confidence survive in a 512-cell DP histogram? | Prop. D2, H4 | ~15 min |
| `E4b-lite-noise-aware-synthetic` | trained noise-aware vs naive PFN on high-dimensional, marginal-based DP synthetic data | Prop. D2, H4, Gate 2 | ~30 min train + ~10 min eval |

Environment: `python -m venv venv && pip install numpy scipy scikit-learn torch networkx matplotlib`. Run `OMP_NUM_THREADS=1` for the numpy jobs when they share a machine with training.

Run order:

```bash
python experiments/X0-theory-audit/audit_constants.py
python experiments/E1-lite-amortization-gap/train.py --steps 6000 --threads 3
python experiments/E1-lite-amortization-gap/evaluate.py
python experiments/E3-lite-vote-sample-complexity/vote_c2.py --n 500 --tag _n500
python experiments/D2-noise-aware-calibration/d2_calibration.py
python experiments/G3-rr-sbm-threshold/rr_sbm.py
python experiments/E4-lite-noise-aware-rr/e4_rr.py train --mode aware
python experiments/E4-lite-noise-aware-rr/e4_rr.py train --mode naive
python experiments/E4-lite-noise-aware-rr/e4_rr.py eval
python experiments/D2-sparse-synthetic/sparse_hist.py
```

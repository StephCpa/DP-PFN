# E4-lite: a trained noise-aware PFN on label randomized response (Thm. D1, Prop. D2, H4, Gate 2)

**Question.** D2-check showed that the *target* of a noise-aware PFN fixes the under-confidence of naive plug-in on label-RR data. Does a *trained* network reach that target? This is Gate 2 in its cheapest form.

**Prior.** Same as D2-check: w ~ N(0, 2²), b ~ N(0, 1), x ~ U[−1, 1], y ~ Bern(σ(wx + b)).
**Release.** Label randomized response with ρ = 1/(1+e^ε). This is ε-label-LDP, so every downstream prediction is post-processing.

**Networks.** E1-lite architecture: 4 layers, d = 128, permutation-invariant. The noise-level slot carries ρ. Each is trained 4,000 steps × 64 tasks, n ~ U{8..128}, 32 queries.

- `aware`: RR contexts with ε ~ log-U[0.25, 6] (10% clean), ρ as input, **clean** query targets. Its loss minimiser is P(y* | x*, S) by Thm. D1.
- `naive`: trained on clean contexts (ρ input = 0), then given the RR release at test time. This is the "TabPFN on released data" baseline.

**Evaluation.** n ∈ {32, 128}, ε ∈ {0.5, 1, 2, ∞}, 200 tasks × 400 clean test labels. The networks are compared against three exact grid-Bayes targets: oracle (clean data), naive (RR labels treated as clean), and noise-aware. Metrics: accuracy, NLL, ECE, over-confidence, and KL(exact noise-aware ‖ predictor). The ε = ∞ row checks that noise-awareness costs nothing on clean data.

```bash
python experiments/E4-lite-noise-aware-rr/e4_rr.py train --mode aware --threads 2
python experiments/E4-lite-noise-aware-rr/e4_rr.py train --mode naive --threads 2
python experiments/E4-lite-noise-aware-rr/e4_rr.py eval
```

**Longer training** (review §4.6.1). Warm start from the 4,000-step model, 12,000 more steps, same paired evaluation:

```bash
python experiments/E4-lite-noise-aware-rr/e4_rr.py train --mode aware --init pfn_aware.pt --tag _16k --steps 12000 --save-at 4000 8000 --lr 3e-4 --seed 7 --threads 4
python experiments/E4-lite-noise-aware-rr/e4_rr.py eval --aware-ckpt pfn_aware_16k_step4000.pt --out evaluation_8k.json
python experiments/E4-lite-noise-aware-rr/e4_rr.py eval --aware-ckpt pfn_aware_16k_step8000.pt --out evaluation_12k.json
python experiments/E4-lite-noise-aware-rr/e4_rr.py eval --aware-ckpt pfn_aware_16k.pt --out evaluation_16k.json
python experiments/E4-lite-noise-aware-rr/compare_steps.py
```

**Training range** (review §4.6.2). In the runs above n = 128 is the top of the training range U{8..128}. Two more continuations of the same 4,000-step model, identical to the `_16k` run except for the context sizes: n ~ U{8..256}, and n ~ log-U[8, 256] (equal weight per octave, which keeps small n well represented). One evaluation pass scores all ten aware checkpoints on the same draws, over n ∈ {32, 64, 128, 256}. Its 4k and n ≤ 128 numbers reproduce the committed single-checkpoint files exactly.

```bash
python experiments/E4-lite-noise-aware-rr/e4_rr.py train --mode aware --init pfn_aware.pt --tag _n256 --steps 12000 --save-at 4000 8000 --lr 3e-4 --seed 7 --threads 2 --n-max 256
python experiments/E4-lite-noise-aware-rr/e4_rr.py train --mode aware --init pfn_aware.pt --tag _n256log --steps 12000 --save-at 4000 8000 --lr 3e-4 --seed 7 --threads 2 --n-max 256 --n-dist loguniform
python experiments/E4-lite-noise-aware-rr/e4_rr.py eval --ns 32 64 128 256 --out evaluation_range.json --extra-aware \
    pfn_aware_16k_step4000.pt pfn_aware_16k_step8000.pt pfn_aware_16k.pt \
    pfn_aware_n256_step4000.pt pfn_aware_n256_step8000.pt pfn_aware_n256.pt \
    pfn_aware_n256log_step4000.pt pfn_aware_n256log_step8000.pt pfn_aware_n256log.pt
python experiments/E4-lite-noise-aware-rr/compare_range.py
```

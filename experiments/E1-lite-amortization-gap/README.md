# E1-lite: exact vs amortized privacy (Prop. B2, Obs. B5, H2)

**Question.** A PFN is trained on a prior whose exact PPD has finite privacy loss (floored labels, Theorem A2). Does the network inherit that loss? Or can its own row sensitivity s(θ, n) exceed the exact mechanism's, on typical or on adversarial datasets?

**Prior** (label-DP: features public, one context label flips):
w ~ N(0, 2²), b ~ N(0, 1), x ~ U[−1, 1], y ~ Bern((1−η)σ(wx+b) + η/2).
η ~ U(0.05, 0.9) is an input to the network, so one network covers every η. Exact PPD: grid posterior over (w, b) (241 × 81). Theorem A2 bound for the exact PPD: 2B_y(η) = 2 log(1 + 2(1−η)/η).

**Model.** 4-layer, d = 128 PFN with standard masking (context↔context, query→context) and no positional encoding, so it is permutation-invariant. Trained 6,000 steps × 128 datasets, n ~ U{1..64}, 32 queries each. Checkpoints at 100, 300, 1000, 3000 and 6000 steps track how faithfulness changes leakage (Obs. B5).

**Measurements** (`evaluate.py`), per checkpoint, η ∈ {0.1, 0.5}, n ∈ {4, 16, 64}. Adversarial searches run at checkpoints 300 and 6000 for n ∈ {4, 16}, with 12 restarts × 80 Adam steps:

- *typical* — 64 prior datasets: Λ(D) = max over x*, y of |log q − log P|. Leakage of q and of the exact P: max over flipped row j, x* and y of the log-ratio.
- *adv_in* — gradient search over datasets with x ∈ [−1, 1] (prior support).
- *adv_out* — gradient search with x ∈ [−5, 5]. Such datasets have prior probability 0: the case Prop. B2 is about.

Each search maximises one of three objectives: the network's leakage (a lower bound on s(θ, n)), the exact mechanism's leakage, or the gap Λ. On the dataset found, all three quantities are reported.

**Predictions from the proposal.** Λ is small on typical data and large on adversarial data. s(θ, n) can exceed 2B_y on adversarial pairs. Better-trained PFNs leak more.

**Not covered.** TabPFN scale; continuous-feature row-DP (B = ∞ there unless features are floored); the clean-target network of A5.

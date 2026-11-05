# G3-check: edge randomized response and community detectability (Prop. G3, T7)

**Question.** Prop. G3 gives a heuristic N ≲ (1−2ρ)²(a−b)²/(K²ρ) beyond which structure-only community signal disappears under ε-edge RR. Is it right, and which exact form should T7 prove?

Edge RR makes the graph **dense**: edge probability ≥ ρ, average degree ≈ Nρ. So the Kesten–Stigum form (sparse, variance ≈ p) should be replaced by the dense spiked-matrix form, with variance p(1−p):

SNR = (a'−b')² / (K (a'+(K−1)b') (1−p̄')),  with a' = Nρ + (1−2ρ)a.

`X0-theory-audit` computes three thresholds: the heuristic, sparse KS without dropped terms, and dense BBP. This experiment checks them with spectral recovery (top eigenvector of the centred adjacency) on simulated RR graphs. Settings: K = 2, a = 20, b = 4, ε ∈ {1, 2, 4}. The theory curve corr² = 1 − 1/SNR is the spiked-Wigner eigenvector overlap.

Not covered: node features (the cSBM part of T7). Deshpande et al. 2018 give λ² + μ²/γ > 1 in the large-degree limit, which is exactly the regime RR produces.

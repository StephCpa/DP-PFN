# G3-check log

Run: gpu-gfkd, `/tmp/dp-pfn`, commit `f7ea51f`, 2026-11-05. CPU.
`rr_sbm.py`, seed 0. K = 2, a = 20, b = 4. Spectral recovery on the centred adjacency of the RR graph.
Reps: 60 at eps = 1, 30 at eps = 2, 8 at eps = 4. Artifact: `artifacts/G3-rr-sbm-threshold/results.json`.
Provenance records `git_dirty: true` because the server tree contained the untracked run scripts.

Thresholds on N (above this, dense SNR drops below 1):

| eps | heuristic (proposal) | sparse KS, all terms | dense BBP |
| --- | --- | --- | --- |
| 1 | 51 | 30 | 59 |
| 2 | 311 | 235 | 290 |
| 4 | 3307 | 2664 | 2739 |

Observed correlation squared against the dense spiked-Wigner curve 1 - 1/SNR:

| eps | N | overlap | corr2 observed | corr2 theory | SNR |
| --- | --- | --- | --- | --- | --- |
| 2 | 200 | 0.414 (at N=2000; see file for the curve) | — | — | — |
| 4 | 1000 | 0.687 | 0.493 | 0.518 | 2.07 |
| 4 | 1500 | 0.524 | 0.341 | 0.369 | 1.58 |
| 4 | 2000 | 0.414 | 0.230 | 0.220 | 1.28 |
| 4 | 2500 | 0.244 | 0.098 | 0.071 | 1.08 |
| 4 | 3000 | 0.123 | 0.033 | 0.000 | 0.93 |
| 4 | 3500 | 0.063 | 0.010 | 0.000 | 0.82 |

At eps = 4 the observed overlap collapses between N = 2500 and N = 3000. The dense threshold (2739) sits in that interval. The proposal's heuristic (3307) is late: at N = 3000 the measured corr2 is already 0.033. The sparse KS value (2664) is close but uses the wrong variance, since RR makes the graph dense (degree about N rho).

At eps = 2 the same pattern holds qualitatively: overlap falls from 0.159 at N = 400 to 0.038 at N = 1000, i.e. the signal is already gone well before the heuristic threshold would suggest, and the dense curve tracks it.

## Reading

Proposition G3's formula should be replaced by the dense signal-to-noise ratio before T7 is proved. The simulation follows that curve; it does not follow the printed heuristic. Node features are not in this run, so the contextual-SBM half of T7 is untouched.

"""Write the first research-loop iteration into research-manifest.json.

python experiments/common/update_manifest.py

Idempotent: re-running replaces the iteration-1 records rather than appending
duplicates.  Hypothesis statuses use: proposed | supported | partially-supported |
challenged | refuted | pending.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research-manifest.json"
NOW = "2026-11-05T11:00:00.000Z"


def main() -> None:
    m = json.loads(MANIFEST.read_text())
    e1 = json.loads((ROOT / "artifacts/E1-lite-amortization-gap/summary.json").read_text()) \
        if (ROOT / "artifacts/E1-lite-amortization-gap/summary.json").exists() else None

    experiments = [
        dict(id="X0-theory-audit", title="Numerical audit of all proposal constants; exact one-draw privacy loss",
             tests=["H1"], status="done", code="experiments/X0-theory-audit/",
             artifacts=["artifacts/X0-theory-audit/audit.json"],
             summary="All constants reproduce. Thm A2 tight for the posterior draw (2.193 vs 2B_y=2.197 at n=1000); "
                     "the clean-target release (A5) costs ~B_y, half the bound; a noisy-PPD label costs 0.0034 at n=1000. "
                     "Prop. G3 threshold for the worked example is 290 (dense SNR), not 311."),
        dict(id="E1-lite-amortization-gap", title="Exact vs amortized privacy loss of a small PFN on a floored logistic prior",
             tests=["H2"], status="done" if e1 else "running", code="experiments/E1-lite-amortization-gap/",
             artifacts=["artifacts/E1-lite-amortization-gap/training_curve.json",
                        "artifacts/E1-lite-amortization-gap/evaluation.json",
                        "artifacts/E1-lite-amortization-gap/summary.json",
                        "artifacts/E1-lite-amortization-gap/leakage_vs_training.png"],
             summary=(e1 or {}).get("headline", "pending")),
        dict(id="E3-lite-vote-sample-complexity", title="Theorem C2 vs simulation; Bayes vs trained teachers; Vote vs LogPool",
             tests=["H3"], status="done", code="experiments/E3-lite-vote-sample-complexity/",
             artifacts=[f"artifacts/E3-lite-vote-sample-complexity/results_n{n}.json" for n in (250, 500, 1000)]
                       + ["artifacts/E3-lite-vote-sample-complexity/vote_vs_m_n500.png"],
             summary="Exact-Bayes (PFN-like) teachers beat trained logistic-regression teachers under DP-Vote by "
                     "+4.1/+3.7/+2.3 points at eps=0.1 (n=250/500/1000); the gap vanishes at large n*eps. Even Bayes "
                     "teachers leave a 13-16% infinite-T error floor at m=2, so C2 must be pointwise. C2 argmax m is "
                     "within 2x of the observed best in 42/45 configs, but its n* (1,176-5,077) is very conservative. "
                     "LogPool loses to Vote by up to 7.5 points with calibrated teachers."),
        dict(id="D2-noise-aware-calibration", title="Naive vs noise-aware use of label-RR and DP synthetic data (exact Bayes)",
             tests=["H4"], status="done", code="experiments/D2-noise-aware-calibration/",
             artifacts=["artifacts/D2-noise-aware-calibration/results.json"],
             summary="Label RR: naive is under-confident, growing with n (n=1000, eps=0.5: ECE 0.165 vs 0.068, "
                     "over-confidence -0.159 vs -0.001); noise-aware fixes it at equal accuracy. DP synthetic data: "
                     "naive over-confident only at n=50 (+0.032) in a 2-parameter model."),
        dict(id="G3-rr-sbm-threshold", title="Spectral recovery on edge-RR SBMs vs threshold formulas",
             tests=[], status="done", code="experiments/G3-rr-sbm-threshold/",
             artifacts=["artifacts/G3-rr-sbm-threshold/results.json", "artifacts/G3-rr-sbm-threshold/threshold_eps2.png"],
             summary="Simulation follows the dense spiked-matrix SNR (variance p'(1-p')); thresholds 59/290/2739 at "
                     "eps=1/2/4. At N=1024, eps=2 structural signal is gone (overlap 0.04)."),
    ]
    status = {
        "H1": ("partially-supported",
               "Prop. 0 and the wrapper/post-processing guarantees are correct (X0). Must add a third route: certified-by-"
               "architecture PFNs (PrivTab, arXiv 2610.10068). TREDBench caveat: synthetic-task selection uses real data."),
        "H2": ((e1 or {}).get("h2_status", "pending"), (e1 or {}).get("h2_note", "E1-lite evaluation running.")),
        "H3": ("partially-supported",
               "E3-lite (idealised Gaussian prior, exact-Bayes teacher): PFN-like teachers lower DP-Vote error at small "
               "n*eps, and C2 picks m within 2x. But C2 must be pointwise (bias floor 13-16% at m=2) and its n* is "
               "conservative by ~3-10x. Real PFN teachers and tuned/charged trained teachers still to test (E2/E3)."),
        "H4": ("partially-supported",
               "Target-level (exact Bayes) D2-check: confirmed strongly for label RR; synthetic-data over-confidence is "
               "small in low-dimensional settings. A trained noise-aware PFN has not been tested yet (E4)."),
    }
    for h in m["hypotheses"]:
        if h["id"] in status:
            h["status"], h["evidence"] = status[h["id"]]
            h["experiments"] = [e["id"] for e in experiments if h["id"] in e["tests"]]
    m["experiments"] = experiments
    m["artifacts"] = sorted({a for e in experiments for a in e["artifacts"] if (ROOT / a).exists()})
    m["reviews"] = [dict(id="review-2026-11-05", path="reviews/2026-11-05-plan-review.md", at=NOW,
                         kind="plan review + first-round research",
                         companion=["literature/related-work-review.md", "literature/references.bib"])]
    m["loop"]["phase"] = "analysis"
    m["loop"]["iteration"] = 1
    m["loop"]["history"] = [x for x in m["loop"]["history"] if x.get("iteration") != 1]
    m["loop"]["history"].append(dict(
        at=NOW, iteration=1, kind="review",
        text="Plan reviewed: theory audit (X0), citation audit + related work (PrivTab, Private PoEtry, TabPATE overlap), "
             "and four CPU pilots (E1-lite, E3-lite, D2-check, G3-check). See reviews/2026-11-05-plan-review.md."))
    m["updatedAt"] = NOW
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print("updated", MANIFEST)


if __name__ == "__main__":
    main()

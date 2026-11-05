"""Condense artifacts/E4b-lite-noise-aware-synthetic/evaluation.json into summary.json.

python experiments/E4b-lite-noise-aware-synthetic/summarize.py

A cell counts as over-/under-confident when |mean confidence - accuracy| > 0.01.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.provenance import ARTIFACTS, write_json  # noqa: E402

EXP = "E4b-lite-noise-aware-synthetic"
TOL = 0.01


def sign(v):
    return "over" if v > TOL else ("under" if v < -TOL else "calibrated")


def main() -> None:
    ev = json.loads((ARTIFACTS / EXP / "evaluation.json").read_text())
    rows = []
    for r in ev["rows"]:
        R = r["results"]
        rows.append(dict(
            n=r["n"], eps=r["eps"], k=r["k"],
            nll={k: R[k]["nll"] for k in R}, ece={k: R[k]["ece"] for k in R},
            overconf={k: R[k]["overconf"] for k in R}, acc={k: R[k]["acc"] for k in R},
            naive_sign=sign(R["naive_pfn"]["overconf"]), aware_sign=sign(R["aware_pfn"]["overconf"]),
            nll_gain_aware_vs_naive=R["naive_pfn"]["nll"] - R["aware_pfn"]["nll"],
            acc_diff_aware_minus_naive=R["aware_pfn"]["acc"] - R["naive_pfn"]["acc"]))
    cnt = {s: sum(x["naive_sign"] == s for x in rows) for s in ("over", "under", "calibrated")}
    acnt = {s: sum(x["aware_sign"] == s for x in rows) for s in ("over", "under", "calibrated")}
    gains = [x["nll_gain_aware_vs_naive"] for x in rows]
    wins = sum(g > 0 for g in gains)
    # effect of bulk rows: naive over-confidence at k=4 minus k=1, same (n, eps)
    dk = []
    for x in rows:
        if x["k"] == max(r["k"] for r in rows):
            y = next(z for z in rows if z["n"] == x["n"] and z["eps"] == x["eps"] and z["k"] == min(r["k"] for r in rows))
            dk.append(x["overconf"]["naive_pfn"] - y["overconf"]["naive_pfn"])
    worst_naive = max(rows, key=lambda x: abs(x["overconf"]["naive_pfn"]))
    headline = (
        "Marginal-based DP synthesizer (d=16, %d cells): naive PFN on synthetic rows is over-confident in %d, under-confident "
        "in %d and calibrated in %d cells (largest %+.3f at n=%d, eps=%g, k=%d); quadrupling the synthetic rows changes its "
        "over-confidence by %+.3f to %+.3f. The noise-aware PFN lowers NLL in %d/%d cells (gain %+.3f to %+.3f nats), with "
        "|over-confidence| <= %.3f and accuracy within %.3f of naive." % (
            len(rows), cnt["over"], cnt["under"], cnt["calibrated"], worst_naive["overconf"]["naive_pfn"], worst_naive["n"],
            worst_naive["eps"], worst_naive["k"], min(dk), max(dk), wins, len(rows), min(gains), max(gains),
            max(abs(x["overconf"]["aware_pfn"]) for x in rows), max(abs(x["acc_diff_aware_minus_naive"]) for x in rows)))
    nb_clean_oc = [x["overconf"]["nb_clean"] for x in rows]
    h4_note = (
        "Label RR: a trained noise-aware PFN recovers 71-86%% of the exact NLL gain over naive plug-in (E4-lite). "
        "Synthetic data: the sign of naive mis-calibration depends on the generator. It is under-confident for a noisy full "
        "histogram (D2-sparse, exact targets) and over-confident in %d/%d cells for a marginal-based, naive-Bayes-structured "
        "synthesizer (E4b-lite, trained networks: %+.3f to %+.3f), where non-private naive Bayes is already over-confident "
        "(%+.3f to %+.3f) and extra synthetic rows add little (%+.3f to %+.3f). In E4b-lite the trained noise-aware PFN is "
        "calibrated (|over-confidence| <= %.3f) and lowers NLL in %d/%d cells. Open: richer-marginal generators (AIM/MST), "
        "real data, TabPFN scale." % (
            cnt["over"], len(rows), min(x["overconf"]["naive_pfn"] for x in rows), max(x["overconf"]["naive_pfn"] for x in rows),
            min(nb_clean_oc), max(nb_clean_oc), min(dk), max(dk),
            max(abs(x["overconf"]["aware_pfn"]) for x in rows), wins, len(rows)))
    out = dict(rows=rows, naive_sign_counts=cnt, aware_sign_counts=acnt, nll_wins=wins, k_effect_on_naive_overconf=dk,
               headline=headline, h4_status="partially-supported", h4_note=h4_note)
    path = write_json(EXP, "summary.json", out, __file__)
    print(headline)
    print("wrote", path)


if __name__ == "__main__":
    main()

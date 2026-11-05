"""Condense artifacts/E4c-prime-mst-selection/evaluation.json into summary.json.

python experiments/E4c-prime-mst-selection/summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.provenance import ARTIFACTS, write_json  # noqa: E402

EXP = "E4c-prime-mst-selection"
RELS = ("nb", "star", "mst_dp", "tree_oracle")


def rng(v):
    return "%+.3f to %+.3f" % (min(v), max(v))


def main() -> None:
    ev = json.loads((ARTIFACTS / EXP / "evaluation.json").read_text())
    e4b = {(r["n"], r["eps"], r["k"]): r["results"] for r in
           json.loads((ARTIFACTS / "E4b-lite-noise-aware-synthetic/evaluation.json").read_text())["rows"]}
    rows, paired = [], 0
    for r in ev["rows"]:
        R, S, key = r["results"], r["edge_abs_corr"], (r["n"], r["eps"], r["k"])
        paired += int(abs(R["naive_nb"]["overconf"] - e4b[key]["naive_pfn"]["overconf"]) < 1e-9)
        rows.append(dict(
            n=r["n"], eps=r["eps"], k=r["k"], edge_abs_corr=S,
            naive_overconf={x: R["naive_" + x]["overconf"] for x in RELS},
            naive_nll={x: R["naive_" + x]["nll"] for x in RELS},
            naive_acc={x: R["naive_" + x]["acc"] for x in RELS},
            aware_mst=dict(overconf=R["aware_mst"]["overconf"], nll=R["aware_mst"]["nll"], acc=R["aware_mst"]["acc"]),
            aware_nb=dict(overconf=R["aware_nb"]["overconf"], nll=R["aware_nb"]["nll"], acc=R["aware_nb"]["acc"])))
    lower = {x: sum(z["naive_overconf"][x] < z["naive_overconf"]["nb"] for z in rows) for x in RELS[1:]}
    sel_gap = [z["edge_abs_corr"]["mst_dp"] - z["edge_abs_corr"]["all_pairs"] for z in rows]
    big = [z for z in rows if z["n"] == 128 and z["eps"] == 16.0]
    gain = [z["naive_nll"]["mst_dp"] - z["aware_mst"]["nll"] for z in rows]
    vs_nb = [z["aware_mst"]["nll"] - z["aware_nb"]["nll"] for z in rows]
    acc_drop = [z["naive_acc"]["nb"] - z["naive_acc"]["mst_dp"] for z in rows]
    headline = (
        "Per-dataset DP MST selection (eps/3 each for label tables, selection and pair tables; 12 cells paired with E4b, "
        "%d/12 verified): private selection picks edges barely more correlated than random pairs (%+.3f to %+.3f |corr|; "
        "%.2f vs oracle %.2f at n=128, eps=16). Relative to E4b's release, naive over-confidence is lower in only %d/12 "
        "cells for the DP tree, %d/12 for the star and %d/12 even for a non-private oracle tree; at n=128, eps=16 the "
        "oracle tree lowers it by %.3f-%.3f. The DP tree costs naive use %.3f-%.3f accuracy. The noise-aware PFN on the "
        "DP-MST release stays calibrated (%s) and beats naive in NLL in %d/12 cells, but is worse than E4b's noise-aware "
        "PFN on the simpler release in %d/12 cells (by %.3f-%.3f nats)." % (
            paired, min(sel_gap), max(sel_gap), max(z["edge_abs_corr"]["mst_dp"] for z in big),
            max(z["edge_abs_corr"]["tree_oracle"] for z in big), lower["mst_dp"], lower["star"], lower["tree_oracle"],
            min(z["naive_overconf"]["nb"] - z["naive_overconf"]["tree_oracle"] for z in big),
            max(z["naive_overconf"]["nb"] - z["naive_overconf"]["tree_oracle"] for z in big),
            min(acc_drop), max(acc_drop), rng([z["aware_mst"]["overconf"] for z in rows]), sum(g > 0 for g in gain),
            sum(v > 0 for v in vs_nb), min(vs_nb), max(vs_nb)))
    h4_note = (
        " E4c-prime selects a per-dataset tree privately, as MST does: at n<=128, eps<=16 the selection is close to random, "
        "and even a non-private oracle tree lowers naive over-confidence only at n=128 with eps=4-16 (%d/12 cells), "
        "because splitting the budget adds more noise than the tree removes. The noise-aware PFN stays calibrated on the "
        "DP-MST release and beats naive in NLL in %d/12 cells, though that release is less informative than E4b's." % (
            lower["tree_oracle"], sum(g > 0 for g in gain)))
    out = dict(rows=rows, paired_cells=paired, lower_than_nb=lower, headline=headline, h4_note_suffix=h4_note,
               aware_mst_nll_gain_over_naive=gain, aware_mst_minus_aware_nb_nll=vs_nb)
    path = write_json(EXP, "summary.json", out, __file__)
    print(headline)
    print("wrote", path)


if __name__ == "__main__":
    main()

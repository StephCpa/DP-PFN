"""Compare the E4d-lite aligned reader after 6k, 12k and 18k training steps (paired cells, eval seed 3).

python experiments/E4d-lite-large-n-summary-pfn/compare_steps.py

Inputs: evaluation.json (6k), evaluation_12k_high_neps.json (12k, n*eps >= 8192 only),
evaluation_18k.json (18k, all 20 cells).  The releases and plug-ins are identical across files
(same seed), so differences are the network's alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.provenance import ARTIFACTS, write_json  # noqa: E402

EXP = "E4d-lite-large-n-summary-pfn"


def load(name):
    d = json.loads((ARTIFACTS / EXP / name).read_text())
    return {(r["n"], r["eps"]): r for r in d["rows"]}


def main() -> None:
    runs = {"6k": load("evaluation.json"), "12k": load("evaluation_12k_high_neps.json"), "18k": load("evaluation_18k.json")}
    base = runs["6k"]
    # check that the releases' own classifiers are identical across runs (same data and releases)
    same = all(abs(runs[k][c]["results"]["plugin_tree"]["nll"] - base[c]["results"]["plugin_tree"]["nll"]) < 1e-12
               for k in ("12k", "18k") for c in runs[k])
    cells = []
    for c in sorted(base, key=lambda c: (c[0] * c[1], c[0])):
        row = dict(n=c[0], eps=c[1], n_eps=c[0] * c[1],
                   plugin_tree=base[c]["results"]["plugin_tree"]["nll"], plugin_nb=base[c]["results"]["plugin_nb"]["nll"],
                   clean_tan=base[c]["results"]["clean_tan"]["nll"])
        for k, r in runs.items():
            if c in r:
                R = r[c]["results"]
                row[k] = dict(tree=R["sum_aware_tree"]["nll"], nb=R["sum_aware_nb"]["nll"],
                              oc_tree=R["sum_aware_tree"]["overconf"], oc_nb=R["sum_aware_nb"]["overconf"])
        cells.append(row)
    hi = [c for c in cells if c["n_eps"] >= 32768]
    gap = {k: [c[k]["tree"] - c["plugin_tree"] for c in hi if k in c] for k in runs}
    full18 = [c for c in cells if "18k" in c]
    out = dict(releases_identical=same, cells=cells,
               gap_to_plugin_tree_at_high_neps={k: [min(v), max(v)] for k, v in gap.items()},
               closed_cells_18k=sum(g <= 0 for g in gap["18k"]),
               high_cells=len(hi),
               beats_plugin_tree_18k=sum(c["18k"]["tree"] < c["plugin_tree"] for c in full18),
               beats_plugin_nb_18k=sum(c["18k"]["nb"] < c["plugin_nb"] for c in full18),
               prefers_tree_18k=[c["n_eps"] for c in full18 if c["18k"]["tree"] < c["18k"]["nb"]],
               overconf_range_18k=[min(min(c["18k"]["oc_tree"], c["18k"]["oc_nb"]) for c in full18),
                                   max(max(c["18k"]["oc_tree"], c["18k"]["oc_nb"]) for c in full18)],
               nll_change_18k_vs_6k=dict(
                   tree=[min(c["18k"]["tree"] - c["6k"]["tree"] for c in full18), max(c["18k"]["tree"] - c["6k"]["tree"] for c in full18)],
                   nb=[min(c["18k"]["nb"] - c["6k"]["nb"] for c in full18), max(c["18k"]["nb"] - c["6k"]["nb"] for c in full18)]))
    out["headline"] = (
        "Training the table-reading noise-aware PFN 3x longer (6k -> 18k steps, warm start): at n*eps >= 32,768 its gap to "
        "the tree release's own classifier goes from %+.3f..%+.3f (6k) to %+.3f..%+.3f (12k) to %+.3f..%+.3f (18k) nats; "
        "closed in %d/%d cells. Over all 20 cells at 18k: beats the plain release's classifier in %d/20 and the tree "
        "release's in %d/20; over-confidence %+.3f..%+.3f; NLL change vs 6k %+.3f..%+.3f (tree) and %+.3f..%+.3f (plain)." % (
            *out["gap_to_plugin_tree_at_high_neps"]["6k"], *out["gap_to_plugin_tree_at_high_neps"]["12k"],
            *out["gap_to_plugin_tree_at_high_neps"]["18k"], out["closed_cells_18k"], out["high_cells"],
            out["beats_plugin_nb_18k"], out["beats_plugin_tree_18k"], *out["overconf_range_18k"],
            *out["nll_change_18k_vs_6k"]["tree"], *out["nll_change_18k_vs_6k"]["nb"]))
    write_json(EXP, "continued_training_summary.json", out, __file__)
    print("releases identical across runs:", same)
    print(out["headline"])
    print("cell            n*eps   plug_tree  aware_tree 6k / 12k / 18k   | aware_nb 6k / 18k | clean_tan")
    for c in cells:
        t12 = "%.4f" % c["12k"]["tree"] if "12k" in c else "  -   "
        print("%-14s %7g   %.4f     %.4f / %s / %.4f   | %.4f / %.4f | %.4f" % (
            (c["n"], c["eps"]), c["n_eps"], c["plugin_tree"], c["6k"]["tree"], t12, c["18k"]["tree"],
            c["6k"]["nb"], c["18k"]["nb"], c["clean_tan"]))


if __name__ == "__main__":
    main()

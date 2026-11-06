"""Summarize artifacts/E4d-lite-large-n-summary-pfn/evaluation.json -> summary.json and e4d.png.

python experiments/E4d-lite-large-n-summary-pfn/summarize.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.provenance import ARTIFACTS, write_json  # noqa: E402
from common.make_figures import INK2, REF, S1, S2, S3  # noqa: E402

EXP = "E4d-lite-large-n-summary-pfn"
S4 = "#eda100"  # reference palette slot 4 (yellow); lines are direct-labelled
SERIES = (("plugin_nb", "plain release: own classifier", S1, "o"),
          ("sum_aware_nb", "plain release: noise-aware PFN", S2, "o"),
          ("plugin_tree", "tree release: own classifier", S3, "s"),
          ("sum_aware_tree", "tree release: noise-aware PFN", S4, "s"))


def main() -> None:
    ev = json.loads((ARTIFACTS / EXP / "evaluation.json").read_text())
    rows = ev["rows"]
    by = defaultdict(list)
    for r in rows:
        by[r["n_eps"]].append(r["results"])
    levels = []
    for ne in sorted(by):
        R = by[ne]
        med = {k: dict(nll=float(np.median([x[k]["nll"] for x in R])), overconf=float(np.median([x[k]["overconf"] for x in R])))
               for k in ("plugin_nb", "sum_aware_nb", "plugin_tree", "sum_aware_tree", "clean_nb", "clean_tan")}
        levels.append(dict(n_eps=ne, cells=len(R), median=med,
                           aware_beats_plugin_nb=sum(x["sum_aware_nb"]["nll"] < x["plugin_nb"]["nll"] for x in R),
                           aware_beats_plugin_tree=sum(x["sum_aware_tree"]["nll"] < x["plugin_tree"]["nll"] for x in R),
                           aware_prefers_tree=sum(x["sum_aware_tree"]["nll"] < x["sum_aware_nb"]["nll"] for x in R)))
    allR = [r["results"] for r in rows]
    oc_aw = [x[k]["overconf"] for x in allR for k in ("sum_aware_nb", "sum_aware_tree")]
    big = [r["results"] for r in rows if r["n_eps"] >= 32768]
    small = [r for r in rows if r["n"] <= 128]
    cont = [(r["results"]["sum_aware_nb"]["nll"] - r["results"]["row_aware_nb"]["nll"],
             r["results"]["sum_aware_tree"]["nll"] - r["results"]["row_aware_tree"]["nll"]) for r in small if "row_aware_nb" in r["results"]]
    first_pref = next((l["n_eps"] for l in levels if l["aware_prefers_tree"] == l["cells"]
                       and all(m["aware_prefers_tree"] == m["cells"] for m in levels if m["n_eps"] >= l["n_eps"])), None)
    out = dict(levels=levels,
               aware_overconf_range=[min(oc_aw), max(oc_aw)],
               aware_beats_plugin_nb_cells=sum(x["sum_aware_nb"]["nll"] < x["plugin_nb"]["nll"] for x in allR),
               aware_beats_plugin_tree_cells=sum(x["sum_aware_tree"]["nll"] < x["plugin_tree"]["nll"] for x in allR),
               cells=len(allR),
               large_neps_overconf=dict(plugin_nb=[min(x["plugin_nb"]["overconf"] for x in big), max(x["plugin_nb"]["overconf"] for x in big)],
                                        sum_aware_nb=[min(x["sum_aware_nb"]["overconf"] for x in big), max(x["sum_aware_nb"]["overconf"] for x in big)]),
               large_neps_gap_to_clean_tan=dict(
                   sum_aware_tree=[min(x["sum_aware_tree"]["nll"] - x["clean_tan"]["nll"] for x in big), max(x["sum_aware_tree"]["nll"] - x["clean_tan"]["nll"] for x in big)],
                   sum_aware_nb=[min(x["sum_aware_nb"]["nll"] - x["clean_tan"]["nll"] for x in big), max(x["sum_aware_nb"]["nll"] - x["clean_tan"]["nll"] for x in big)],
                   plugin_tree=[min(x["plugin_tree"]["nll"] - x["clean_tan"]["nll"] for x in big), max(x["plugin_tree"]["nll"] - x["clean_tan"]["nll"] for x in big)]),
               continuity_vs_row_aware=dict(nb=[min(c[0] for c in cont), max(c[0] for c in cont)],
                                            tree=[min(c[1] for c in cont), max(c[1] for c in cont)]) if cont else None,
               aware_prefers_tree_from_n_eps=first_pref)
    out["headline"] = (
        "Noise-aware PFN reading the noisy tables (n 32-8192, 20 cells): calibrated everywhere (over-confidence %+.3f to "
        "%+.3f) and better in NLL than the release's own classifier in %d/%d cells (plain release) and %d/%d (tree release). "
        "At n*eps >= 32,768 it removes plain margins' structural over-confidence (%+.3f..%+.3f vs %+.3f..%+.3f for the "
        "naive-Bayes plug-in). The noise-aware analyst prefers the tree release in every cell from n*eps = %s; at n <= 128 the "
        "table reader is within %+.3f..%+.3f (plain) and %+.3f..%+.3f (tree) nats of the row-reading noise-aware PFNs." % (
            out["aware_overconf_range"][0], out["aware_overconf_range"][1], out["aware_beats_plugin_nb_cells"], out["cells"],
            out["aware_beats_plugin_tree_cells"], out["cells"], *out["large_neps_overconf"]["sum_aware_nb"],
            *out["large_neps_overconf"]["plugin_nb"], first_pref, *(out["continuity_vs_row_aware"] or {"nb": [0, 0]})["nb"],
            *(out["continuity_vs_row_aware"] or {"tree": [0, 0]})["tree"]))
    write_json(EXP, "summary.json", out, __file__)
    print(out["headline"])
    for l in levels:
        md = l["median"]
        print("%7d (%d) NLL plug nb %.3f aware nb %.3f | plug tree %.3f aware tree %.3f | clean tan %.3f | oc plug nb %+.3f aware nb %+.3f plug tree %+.3f aware tree %+.3f | aware>plug nb %d tree %d | prefers tree %d" % (
            l["n_eps"], l["cells"], md["plugin_nb"]["nll"], md["sum_aware_nb"]["nll"], md["plugin_tree"]["nll"],
            md["sum_aware_tree"]["nll"], md["clean_tan"]["nll"], md["plugin_nb"]["overconf"], md["sum_aware_nb"]["overconf"],
            md["plugin_tree"]["overconf"], md["sum_aware_tree"]["overconf"], l["aware_beats_plugin_nb"], l["aware_beats_plugin_tree"],
            l["aware_prefers_tree"]))

    # ---------------- figure: median per n*eps level, two panels, direct labels
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    xs = [l["n_eps"] for l in levels]
    for key, lab, col, mk in SERIES:
        for ax, metric in zip(axes, ("nll", "overconf")):
            ys = [l["median"][key][metric] for l in levels]
            ax.plot(xs, ys, "-" + mk, color=col, label=lab)
    axes[0].plot(xs, [l["median"]["clean_tan"]["nll"] for l in levels], "--", color=REF, lw=1.5, label="noise-free TAN plug-in (non-private)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("NLL on clean test labels (median over cells)")
    axes[0].set_title("Log-loss")
    axes[1].axhline(0, color=REF, lw=1)
    axes[1].set_ylabel("over-confidence (conf − acc)")
    axes[1].set_title("Calibration")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n · ε")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(ARTIFACTS / EXP / "e4d.png", dpi=200)
    print("wrote", ARTIFACTS / EXP / "e4d.png")


if __name__ == "__main__":
    main()

"""Summarize artifacts/E4c-prime-mst-selection/sweep.json -> sweep_summary.json and sweep.png.

python experiments/E4c-prime-mst-selection/summarize_sweep.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.provenance import ARTIFACTS, write_json  # noqa: E402
from common.make_figures import GRID, INK2, REF, S1, S2, S3  # noqa: E402  (shared palette and rcParams)

EXP = "E4c-prime-mst-selection"


def main() -> None:
    sw = json.loads((ARTIFACTS / EXP / "sweep.json").read_text())
    rows = sw["rows"]
    by = defaultdict(list)
    for r in rows:
        by[r["n_eps"]].append(r)
    levels = []
    for ne in sorted(by):
        rs = by[ne]
        R = [r["results"] for r in rs]
        lev = dict(n_eps=ne, cells=[(r["n"], r["eps"]) for r in rs],
                   dp_edge_corr=[r["edge_abs_corr"]["mst_dp"] for r in rs],
                   random_edge_corr=[r["edge_abs_corr"]["all_pairs"] for r in rs],
                   oracle_edge_corr=[r["edge_abs_corr"]["tree_oracle"] for r in rs],
                   plugin_mst_beats_nb_nll=sum(x["mst_dp_plugin"]["nll"] < x["nb_plugin"]["nll"] for x in R),
                   plugin_oracle_beats_nb_nll=sum(x["tree_oracle_plugin"]["nll"] < x["nb_plugin"]["nll"] for x in R),
                   plugin_mst_less_overconf=sum(x["mst_dp_plugin"]["overconf"] < x["nb_plugin"]["overconf"] for x in R),
                   n_cells=len(rs))
        pf = [x for x in R if "nb_pfn" in x]
        if pf:
            lev["pfn_cells"] = len(pf)
            lev["pfn_mst_less_overconf"] = sum(x["mst_dp_pfn"]["overconf"] < x["nb_pfn"]["overconf"] for x in pf)
        levels.append(lev)
    nb_struct = [r["results"]["nb_plugin"]["overconf"] for r in rows if r["n_eps"] >= 32768]
    tan_gain = [r["results"]["nb_clean"]["nll"] - r["results"]["tan_clean"]["nll"] for r in rows if r["n"] >= 128]
    first_all = next((l["n_eps"] for l in levels if l["plugin_mst_beats_nb_nll"] == l["n_cells"]
                      and all(m["plugin_mst_beats_nb_nll"] == m["n_cells"] for m in levels if m["n_eps"] >= l["n_eps"])), None)
    first_oracle = next((l["n_eps"] for l in levels if l["plugin_oracle_beats_nb_nll"] == l["n_cells"]
                         and all(m["plugin_oracle_beats_nb_nll"] == m["n_cells"] for m in levels if m["n_eps"] >= l["n_eps"])), None)
    low = [r for r in rows if r["n_eps"] <= 128 and "nb_pfn" in r["results"]]
    headline = (
        "n*eps sweep (n 32-8192, eps 1-64, 20 cells): the release's own classifier (k -> inf limit of naive use) gains from "
        "a tree once n*eps >= %d for the non-private oracle tree and >= %d for the private per-dataset MST tree (all cells "
        "at and above, NLL); below ~2,000 every tree release is worse than plain feature-label margins. Private selection "
        "becomes informative near n*eps 2,000 and near-oracle by 8,000-33,000. Plain marginals keep a structural "
        "over-confidence of %+.3f to %+.3f at n*eps >= 32,768 that only the tree removes (noise-free TAN beats noise-free "
        "NB by %.3f-%.3f nats for n >= 128). At n*eps <= 128 the plug-in is far more over-confident than the PFN on k=1 rows "
        "(%+.3f to %+.3f vs %+.3f to %+.3f), so bulk synthetic rows matter when k is large." % (
            first_oracle, first_all, min(nb_struct), max(nb_struct), min(tan_gain), max(tan_gain),
            min(r["results"]["nb_plugin"]["overconf"] for r in low), max(r["results"]["nb_plugin"]["overconf"] for r in low),
            min(r["results"]["nb_pfn"]["overconf"] for r in low), max(r["results"]["nb_pfn"]["overconf"] for r in low)))
    out = dict(levels=levels, crossover_oracle_nll=first_oracle, crossover_mst_dp_nll=first_all,
               nb_structural_overconf=[min(nb_struct), max(nb_struct)], tan_vs_nb_clean_gain=[min(tan_gain), max(tan_gain)],
               headline=headline)
    write_json(EXP, "sweep_summary.json", out, __file__)
    print(headline)
    for l in levels:
        print(l["n_eps"], l["cells"], "dp corr %.3f-%.3f" % (min(l["dp_edge_corr"]), max(l["dp_edge_corr"])),
              "NLL mst<nb %d/%d oracle<nb %d/%d | overconf mst<nb %d/%d" % (
                  l["plugin_mst_beats_nb_nll"], l["n_cells"], l["plugin_oracle_beats_nb_nll"], l["n_cells"],
                  l["plugin_mst_less_overconf"], l["n_cells"]),
              ("| PFN mst<nb %d/%d" % (l["pfn_mst_less_overconf"], l["pfn_cells"])) if "pfn_cells" in l else "")

    # ---------------- figure: three panels against n*eps
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    xs = [r["n_eps"] for r in rows]
    ax = axes[0]
    ax.scatter(xs, [r["edge_abs_corr"]["mst_dp"] for r in rows], color=S2, s=40, label="private MST tree", zorder=3)
    ax.scatter(xs, [r["edge_abs_corr"]["tree_oracle"] for r in rows], color=S3, s=40, marker="s", label="oracle tree (not DP)", zorder=3)
    ax.scatter(xs, [r["edge_abs_corr"]["all_pairs"] for r in rows], color=REF, s=22, marker="x", label="random pair")
    ax.set_ylabel("mean |corr| of chosen edges")
    ax.set_title("Tree selection quality")
    ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    ax = axes[1]
    ax.scatter(xs, [r["results"]["nb_plugin"]["overconf"] for r in rows], color=S1, s=40, label="feature–label margins only", zorder=3)
    ax.scatter(xs, [r["results"]["mst_dp_plugin"]["overconf"] for r in rows], color=S2, s=40, label="+ private MST tree", zorder=3)
    ax.scatter(xs, [r["results"]["tree_oracle_plugin"]["overconf"] for r in rows], color=S3, s=40, marker="s", label="+ oracle tree (not DP)", zorder=3)
    ax.axhline(0, color=REF, lw=1)
    ax.set_ylabel("over-confidence (conf − acc)")
    ax.set_title("Release's own classifier: calibration")
    ax.legend(fontsize=8, loc="upper right")
    ax = axes[2]
    ax.scatter(xs, [r["results"]["mst_dp_plugin"]["nll"] - r["results"]["nb_plugin"]["nll"] for r in rows], color=S2, s=40, label="private MST tree", zorder=3)
    ax.scatter(xs, [r["results"]["tree_oracle_plugin"]["nll"] - r["results"]["nb_plugin"]["nll"] for r in rows], color=S3, s=40, marker="s", label="oracle tree (not DP)", zorder=3)
    ax.axhline(0, color=REF, lw=1)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel("NLL(tree release) − NLL(margins only)")
    ax.set_title("Does the tree pay? (< 0 = yes)")
    ax.legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n · ε")
        ax.grid(True, color=GRID)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / EXP / "sweep.png", dpi=200)
    print("wrote", ARTIFACTS / EXP / "sweep.png")


if __name__ == "__main__":
    main()

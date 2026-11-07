"""E4-lite: does training on contexts up to n = 256 fix the trade-off between the noisiest cell and small n?

python experiments/E4-lite-noise-aware-rr/compare_range.py

Input: evaluation_range.json, one evaluation pass (eval seed 1) that scores every aware checkpoint
on the same draws, n in {32, 64, 128, 256} x eps in {0.5, 1, 2, inf}.  Three continuations of the
same 4,000-step model, identical except for the context-size distribution during training
(12,000 steps, lr 3e-4, seed 7, checkpoints after 4k and 8k more steps):
  n128 : n ~ U{8..128}   (the original range; the run of section 4.6.1)
  u256 : n ~ U{8..256}
  log256 : n ~ log-U[8, 256] (equal weight per octave of n)

gap_recovered = (NLL naive PFN - NLL aware PFN) / (NLL naive PFN - NLL exact noise-aware target), as in
compare_steps.py.  Because the exact target is the Bayes predictor under the prior, the expected NLL excess
of any predictor over it equals its KL from the target, so gap_kl = 1 - KL(aware) / KL(naive PFN) estimates
the same quantity without the sampling noise of the test labels (which is larger than the whole noise-aware
gain at n = 32).  The naive PFN was trained on n <= 128, so at n = 256 it extrapolates; there the gap is also
reported against the exact naive plug-in (gap_vs_exact_naive), which needs no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.provenance import ARTIFACTS, write_json  # noqa: E402

EXP = "E4-lite-noise-aware-rr"
INF = float("inf")
NETS = {"4k": "aware_pfn"}
for arm, tag in (("n128", "16k"), ("u256", "n256"), ("log256", "n256log")):
    for total, suffix in (("8k", "_step4000"), ("12k", "_step8000"), ("16k", "")):
        NETS[f"{arm}@{total}"] = f"aware_pfn@pfn_aware_{tag}{suffix}"
COMMITTED = {"4k": "evaluation.json", "n128@8k": "evaluation_8k.json", "n128@12k": "evaluation_12k.json",
             "n128@16k": "evaluation_16k.json"}


def rows(name):
    return {(r["n"], r["eps"]): r["results"] for r in json.loads((ARTIFACTS / EXP / name).read_text())["rows"]}


def main() -> None:
    ev = rows("evaluation_range.json")
    # the new pass must reproduce the committed single-checkpoint evaluations exactly (same draws)
    repro = {}
    for k, f in COMMITTED.items():
        old = rows(f)
        repro[k] = max(abs(ev[c][NETS[k]][m] - old[c]["aware_pfn"][m]) for c in old for m in ("nll", "overconf", "ece"))
    cells = []
    for c in sorted(ev, key=lambda c: (c[0], -c[1] if c[1] != INF else -1e9)):
        R = ev[c]
        ex, nv, nx = R["aware_exact"], R["naive_pfn"], R["naive_exact"]
        row = dict(n=c[0], eps=c[1], exact=dict(nll=ex["nll"], overconf=ex["overconf"]),
                   naive_pfn=dict(nll=nv["nll"], overconf=nv["overconf"]), naive_exact=dict(nll=nx["nll"], overconf=nx["overconf"]))
        for k, key in NETS.items():
            a = R[key]
            d_pfn, d_ex = nv["nll"] - ex["nll"], nx["nll"] - ex["nll"]
            row[k] = dict(nll=a["nll"], overconf=a["overconf"], ece=a["ece"], acc=a["acc"], kl=a["kl_from_aware_exact"],
                          gap_recovered=(nv["nll"] - a["nll"]) / d_pfn if d_pfn > 0.002 else None,
                          gap_kl=1 - a["kl_from_aware_exact"] / nv["kl_from_aware_exact"] if c[1] != INF else None,
                          gap_vs_exact_naive=(nx["nll"] - a["nll"]) / d_ex if d_ex > 0.002 else None)
        cells.append(row)

    def cell(n, eps):
        return next(c for c in cells if c["n"] == n and c["eps"] == eps)

    old8 = [c for c in cells if c["n"] in (32, 128)]  # the 8 cells of sections 4.6 / 4.6.1
    rr = [c for c in cells if c["eps"] != INF]
    agg = {}
    for k in NETS:
        agg[k] = dict(
            noisiest_gap=cell(128, 0.5)[k]["gap_recovered"],
            noisiest_gap_kl=cell(128, 0.5)[k]["gap_kl"],
            noisiest_overconf=cell(128, 0.5)[k]["overconf"],
            noisiest_kl=cell(128, 0.5)[k]["kl"],
            n128_gaps=[cell(128, e)[k]["gap_recovered"] for e in (0.5, 1.0, 2.0)],
            n256_gaps_vs_exact_naive=[cell(256, e)[k]["gap_vs_exact_naive"] for e in (0.5, 1.0, 2.0)],
            n32_nll_minus_naive=[cell(32, e)[k]["nll"] - cell(32, e)["naive_pfn"]["nll"] for e in (0.5, 1.0, 2.0)],
            n32_overconf=[cell(32, e)[k]["overconf"] for e in (0.5, 1.0, 2.0)],
            n32_kl=[cell(32, e)[k]["kl"] for e in (0.5, 1.0, 2.0, INF)],
            gap_kl_by_n={n: [cell(n, e)[k]["gap_kl"] for e in (0.5, 1.0, 2.0)] for n in (32, 64, 128, 256)},
            kl_worse_than_naive_pfn_rr=sum(c[k]["kl"] > c["naive_pfn"]["kl"] for c in rr),
            mean_kl_old8=sum(c[k]["kl"] for c in old8) / len(old8),
            mean_kl_all16=sum(c[k]["kl"] for c in cells) / len(cells),
            max_kl_all16=max(c[k]["kl"] for c in cells),
            max_abs_overconf_minus_exact_rr=max(abs(c[k]["overconf"] - c["exact"]["overconf"]) for c in rr),
            beats_naive_pfn_rr=sum(c[k]["nll"] < c["naive_pfn"]["nll"] for c in rr),
            beats_exact_naive_rr=sum(c[k]["nll"] < c["naive_exact"]["nll"] for c in rr),
            clean_nll_minus_exact=[c[k]["nll"] - c["exact"]["nll"] for c in cells if c["eps"] == INF])
    out = dict(reproduces_committed_max_abs_diff=repro, networks=NETS, aggregates=agg, cells=cells, n_rr_cells=len(rr))
    a, b, g = agg["n128@16k"], agg["u256@16k"], agg["log256@16k"]
    out["headline"] = (
        "Same 4k start, 12k more steps, only the training range of n changed. Noisiest cell (n=128, eps=0.5), "
        "share of the exact noise-aware NLL gain: n<=128 %.0f%%, U{8..256} %.0f%%, log-U[8,256] %.0f%% "
        "(KL to exact %.4f / %.4f / %.4f). Mean KL over 16 cells %.4f / %.4f / %.4f; max |over-confidence - exact| "
        "%.3f / %.3f / %.3f. Largest KL at n=32: %.4f / %.4f / %.4f." % (
            100 * a["noisiest_gap"], 100 * b["noisiest_gap"], 100 * g["noisiest_gap"],
            a["noisiest_kl"], b["noisiest_kl"], g["noisiest_kl"],
            a["mean_kl_all16"], b["mean_kl_all16"], g["mean_kl_all16"],
            a["max_abs_overconf_minus_exact_rr"], b["max_abs_overconf_minus_exact_rr"], g["max_abs_overconf_minus_exact_rr"],
            max(a["n32_kl"]), max(b["n32_kl"]), max(g["n32_kl"])))
    write_json(EXP, "range_summary.json", out, __file__)

    print("reproduces committed evaluations (max |diff|):", {k: "%.1e" % v for k, v in repro.items()})
    print(out["headline"])
    print("\n%-11s %8s %8s %8s %8s %8s %10s %8s %8s %6s %8s" % (
        "network", "gap128.5", "gapKL", "oc128.5", "KL128.5", "meanKL16", "max|oc-ex|", "maxKL32", "n32 oc", "beats", "clean"))
    for k, v in agg.items():
        print("%-11s %7.0f%% %7.0f%% %+8.3f %8.4f %8.4f %10.3f %8.4f %+8.3f %3d/%d %+8.4f" % (
            k, 100 * v["noisiest_gap"], 100 * v["noisiest_gap_kl"], v["noisiest_overconf"], v["noisiest_kl"], v["mean_kl_all16"],
            v["max_abs_overconf_minus_exact_rr"], max(v["n32_kl"]), max(v["n32_overconf"], key=abs),
            v["beats_naive_pfn_rr"], len(rr), max(v["clean_nll_minus_exact"], key=abs)))
    print("\ngap_kl by n (eps 0.5/1/2):")
    for k, v in agg.items():
        print("%-11s %s" % (k, "  ".join("n=%d %s" % (n, "/".join("%3.0f%%" % (100 * x) for x in g)) for n, g in v["gap_kl_by_n"].items())))
    print("\ncell         exact NLL/oc     naive pfn/exact |  aware NLL: 4k  n128  u256  log256 (16k) | KL: 4k n128 u256 log256")
    for c in cells:
        ks = ("4k", "n128@16k", "u256@16k", "log256@16k")
        print("%-12s %.4f/%+.3f  %.4f/%.4f | %s | %s" % (
            (c["n"], c["eps"]), c["exact"]["nll"], c["exact"]["overconf"], c["naive_pfn"]["nll"], c["naive_exact"]["nll"],
            " ".join("%.4f" % c[k]["nll"] for k in ks), " ".join("%.4f" % c[k]["kl"] for k in ks)))


if __name__ == "__main__":
    main()

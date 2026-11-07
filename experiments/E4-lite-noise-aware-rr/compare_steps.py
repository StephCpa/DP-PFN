"""Compare E4-lite's noise-aware PFN after 4k (original), 8k, 12k and 16k steps (paired cells, eval seed 1).

python experiments/E4-lite-noise-aware-rr/compare_steps.py

The data, releases and exact targets are identical across the evaluation files (same seed; the
evaluation reproduces the committed 4k numbers exactly), so differences are the network's alone.
gap_recovered = (NLL naive PFN - NLL aware PFN) / (NLL naive PFN - NLL exact noise-aware target).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.provenance import ARTIFACTS, write_json  # noqa: E402

EXP = "E4-lite-noise-aware-rr"
FILES = {"4k": "evaluation.json", "8k": "evaluation_8k.json", "12k": "evaluation_12k.json", "16k": "evaluation_16k.json"}


def main() -> None:
    runs = {k: {(r["n"], r["eps"]): r["results"] for r in json.loads((ARTIFACTS / EXP / f).read_text())["rows"]}
            for k, f in FILES.items()}
    base = runs["4k"]
    same = all(abs(runs[k][c]["aware_exact"]["nll"] - base[c]["aware_exact"]["nll"]) < 1e-12 and
               abs(runs[k][c]["naive_pfn"]["nll"] - base[c]["naive_pfn"]["nll"]) < 1e-12 for k in runs for c in base)
    cells = []
    for c in sorted(base, key=lambda c: (c[0], -c[1] if c[1] != float("inf") else -1e9)):
        ex, nv = base[c]["aware_exact"], base[c]["naive_pfn"]
        row = dict(n=c[0], eps=c[1], exact=dict(nll=ex["nll"], overconf=ex["overconf"], ece=ex["ece"]),
                   naive=dict(nll=nv["nll"], overconf=nv["overconf"]))
        for k, r in runs.items():
            a = r[c]["aware_pfn"]
            denom = nv["nll"] - ex["nll"]
            row[k] = dict(nll=a["nll"], overconf=a["overconf"], ece=a["ece"], acc=a["acc"], kl=a["kl_from_aware_exact"],
                          gap_recovered=(nv["nll"] - a["nll"]) / denom if denom > 0.002 else None)
        cells.append(row)
    noisy = next(c for c in cells if c["n"] == 128 and c["eps"] == 0.5)
    rr = [c for c in cells if c["eps"] != float("inf")]
    out = dict(identical_targets=same, cells=cells,
               noisiest_cell={k: noisy[k] for k in ("exact", "naive", *FILES)},
               max_abs_overconf_minus_exact={k: max(abs(c[k]["overconf"] - c["exact"]["overconf"]) for c in rr) for k in FILES},
               mean_kl={k: sum(c[k]["kl"] for c in cells) / len(cells) for k in FILES},
               clean_cells_nll_change_16k=[c["16k"]["nll"] - c["4k"]["nll"] for c in cells if c["eps"] == float("inf")])
    n4, n16 = noisy["4k"], noisy["16k"]
    out["headline"] = (
        "Noisiest label-RR cell (n=128, eps=0.5): training the noise-aware PFN 4x longer (4k -> 16k steps, warm start) "
        "moves it from %.0f%% to %.0f%% of the exact noise-aware NLL gain (NLL %.4f -> %.4f; exact %.4f), over-confidence "
        "%+.3f -> %+.3f (exact %+.3f), KL to the exact target %.4f -> %.4f. Across the 8 cells, mean KL %.4f -> %.4f and "
        "max |over-confidence - exact| %.3f -> %.3f." % (
            100 * n4["gap_recovered"], 100 * n16["gap_recovered"], n4["nll"], n16["nll"], noisy["exact"]["nll"],
            n4["overconf"], n16["overconf"], noisy["exact"]["overconf"], n4["kl"], n16["kl"], out["mean_kl"]["4k"],
            out["mean_kl"]["16k"], out["max_abs_overconf_minus_exact"]["4k"], out["max_abs_overconf_minus_exact"]["16k"]))
    write_json(EXP, "continued_training_summary.json", out, __file__)
    print("targets identical across runs:", same)
    print(out["headline"])
    print("cell         exact NLL/oc     naive NLL |  aware NLL 4k/8k/12k/16k  |  oc 4k/8k/12k/16k  |  gap rec. 4k/16k | KL 4k/16k")
    for c in cells:
        g = lambda k: ("%3.0f%%" % (100 * c[k]["gap_recovered"])) if c[k]["gap_recovered"] is not None else "  - "
        print("%-12s %.4f/%+.3f  %.4f | %.4f/%.4f/%.4f/%.4f | %+.3f/%+.3f/%+.3f/%+.3f | %s/%s | %.4f/%.4f" % (
            (c["n"], c["eps"]), c["exact"]["nll"], c["exact"]["overconf"], c["naive"]["nll"],
            *[c[k]["nll"] for k in FILES], *[c[k]["overconf"] for k in FILES], g("4k"), g("16k"), c["4k"]["kl"], c["16k"]["kl"]))


if __name__ == "__main__":
    main()

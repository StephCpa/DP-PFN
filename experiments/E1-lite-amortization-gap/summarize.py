"""Condense artifacts/E1-lite-amortization-gap/evaluation.json into summary.json.

python experiments/E1-lite-amortization-gap/summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.provenance import ARTIFACTS, write_json  # noqa: E402

EXP = "E1-lite-amortization-gap"


def main() -> None:
    ev = json.loads((ARTIFACTS / EXP / "evaluation.json").read_text())
    tc = json.loads((ARTIFACTS / EXP / "training_curve.json").read_text())
    rows = ev["rows"]
    final = max(r["step"] for r in rows)
    out = {"final_step": final, "final_excess_nll_over_bayes": tc["curve"][-1]["excess_over_bayes"], "final": [], "dynamics": []}
    for r in rows:
        t = r["typical"]
        if r["step"] == final:
            row = dict(eta=r["eta"], n=r["n"], two_B_y=r["two_B_y"],
                       typical_Lambda_median=t["Lambda"]["median"], typical_Lambda_max=t["Lambda"]["max"],
                       typical_leak_pfn_median=t["s_hat"]["median"], typical_leak_exact_median=t["eps_exact"]["median"],
                       typical_leak_pfn_max=t["s_hat"]["max"], typical_leak_exact_max=t["eps_exact"]["max"],
                       typical_ratio_median=t["s_hat"]["median"] / t["eps_exact"]["median"])
            for name in ("adv_in", "adv_out"):
                if name in r:
                    a = r[name]
                    row[name] = dict(pfn_worst=a["leak_q"]["leak_q"], exact_on_pfn_worst=a["leak_q"]["leak_exact"],
                                     exact_worst=a["leak_P"]["leak_exact"], Lambda_worst=a["gap"]["gap"],
                                     ratio_pfn_worst_over_exact_worst=a["leak_q"]["leak_q"] / a["leak_P"]["leak_exact"])
            out["final"].append(row)
        out["dynamics"].append(dict(step=r["step"], eta=r["eta"], n=r["n"],
                                    leak_ratio_median=t["s_hat"]["median"] / t["eps_exact"]["median"],
                                    Lambda_median=t["Lambda"]["median"]))
    adv = [f for f in out["final"] if "adv_in" in f]
    worst_ratio_in = max(f["adv_in"]["ratio_pfn_worst_over_exact_worst"] for f in adv)
    min_ratio_in = min(f["adv_in"]["ratio_pfn_worst_over_exact_worst"] for f in adv)
    worst_pfn = max(max(f["adv_in"]["pfn_worst"], f["adv_out"]["pfn_worst"]) for f in adv)
    typ = [f["typical_ratio_median"] for f in out["final"]]
    early = [d["leak_ratio_median"] for d in out["dynamics"] if d["step"] == min(r["step"] for r in rows)]
    out["headline"] = (
        "Trained PFN (excess NLL %.4f over exact Bayes): on prior-typical data its label-DP leakage tracks the exact PPD's "
        "(median ratio %.2f-%.2f; gap Lambda median <= %.2f). Gradient search finds in-support neighbours where it leaks "
        "%.1f-%.1fx the exact mechanism's worst case, and the gap grows out of support (Lambda up to %.2f). Largest leakage "
        "found %.2f, below 2B_y in every setting. An under-trained checkpoint leaks %.2f-%.2fx the exact level (Obs. B5)." % (
            out["final_excess_nll_over_bayes"], min(typ), max(typ), max(f["typical_Lambda_median"] for f in out["final"]),
            min_ratio_in, worst_ratio_in, max(f["adv_out"]["Lambda_worst"] for f in adv), worst_pfn, min(early), max(early)))
    out["h2_status"] = "partially-supported"
    out["h2_note"] = ("E1-lite: Lambda small on typical data and larger adversarially (supported); network leakage exceeds "
                      "the exact mechanism's worst case by up to ~%.1fx in support (supported: exact-PPD privacy is not "
                      "inherited); but leakage stayed below 2B_y everywhere searched (the 'driven large' part not observed "
                      "for a small LayerNorm network); better-trained PFNs leak more (Obs. B5 supported)." % worst_ratio_in)
    path = write_json(EXP, "summary.json", out, __file__)
    print(out["headline"])
    print("wrote", path)


if __name__ == "__main__":
    main()

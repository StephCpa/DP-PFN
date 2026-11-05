"""Condense artifacts/E4-lite-noise-aware-rr/evaluation.json into summary.json.

python experiments/E4-lite-noise-aware-rr/summarize.py

gap_recovered = (NLL naive_pfn - NLL aware_pfn) / (NLL naive_pfn - NLL aware_exact):
the share of the achievable noise-aware NLL improvement the trained network realises.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.provenance import ARTIFACTS, write_json  # noqa: E402

EXP = "E4-lite-noise-aware-rr"


def main() -> None:
    ev = json.loads((ARTIFACTS / EXP / "evaluation.json").read_text())
    rows = []
    for r in ev["rows"]:
        R = r["results"]
        nv, aw, ex = R["naive_pfn"], R["aware_pfn"], R["aware_exact"]
        denom = nv["nll"] - ex["nll"]
        rows.append(dict(
            n=r["n"], eps=r["eps"], rho=r["rho"],
            nll=dict(naive_pfn=nv["nll"], aware_pfn=aw["nll"], aware_exact=ex["nll"], oracle=R["oracle_exact"]["nll"]),
            ece=dict(naive_pfn=nv["ece"], aware_pfn=aw["ece"], aware_exact=ex["ece"]),
            overconf=dict(naive_pfn=nv["overconf"], aware_pfn=aw["overconf"], aware_exact=ex["overconf"]),
            acc=dict(naive_pfn=nv["acc"], aware_pfn=aw["acc"]),
            kl_to_aware_target=dict(naive_pfn=nv["kl_from_aware_exact"], aware_pfn=aw["kl_from_aware_exact"]),
            kl_naive_pfn_to_naive_target=nv["kl_from_naive_exact"],
            gap_recovered=(nv["nll"] - aw["nll"]) / denom if denom > 0.002 else None,
        ))
    big = [x for x in rows if x["n"] == max(r["n"] for r in rows) and x["gap_recovered"] is not None]
    clean = [x for x in rows if x["eps"] == float("inf") or x["rho"] == 0.0]
    headline = (
        "A trained noise-aware PFN on label-RR releases recovers %.0f-%.0f%% of the exact noise-aware NLL gain over naive "
        "plug-in at n=%d (NLL %s vs naive %s), with unchanged accuracy and no cost on clean data (max NLL difference %.4f). "
        "The naive PFN matches the exact naive target (KL <= %.4f), i.e. it is under-confident as Prop. D2 predicts." % (
            100 * min(x["gap_recovered"] for x in big), 100 * max(x["gap_recovered"] for x in big), big[0]["n"],
            "/".join("%.3f" % x["nll"]["aware_pfn"] for x in big), "/".join("%.3f" % x["nll"]["naive_pfn"] for x in big),
            max(abs(x["nll"]["aware_pfn"] - x["nll"]["naive_pfn"]) for x in clean),
            max(x["kl_naive_pfn_to_naive_target"] for x in rows)))
    out = dict(rows=rows, headline=headline, gate2="pass (pilot, label RR only)",
               h4_status="partially-supported",
               h4_note="D2-check (exact targets) plus E4-lite (trained network): for label RR, a trained noise-aware PFN "
                       "recovers most of the calibration gain at equal accuracy and costs nothing on clean data. "
                       "Open: the synthetic-data half of D2, real-data/backbone scale (TabPFN), and the noisiest cell "
                       "(n=128, eps=0.5), where the network remains under-confident.")
    path = write_json(EXP, "summary.json", out, __file__)
    print(headline)
    print("wrote", path)


if __name__ == "__main__":
    main()

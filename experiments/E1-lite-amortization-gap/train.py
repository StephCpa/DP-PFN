"""Train the E1-lite PFN on the floored logistic prior (CPU, ~15 min).

python experiments/E1-lite-amortization-gap/train.py --steps 6000 --threads 3

Checkpoints go to experiments/E1-lite-amortization-gap/results/ (gitignored);
the training curve, with the exact-Bayes reference loss, goes to artifacts/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from pfn import PFN, ExactPPD, sample_batch  # noqa: E402
from common.provenance import write_json  # noqa: E402

EXP = "E1-lite-amortization-gap"


def bayes_reference(val, exact: ExactPPD) -> float:
    """Mean NLL of the exact PPD on the validation set = the loss floor."""
    tot, cnt = 0.0, 0
    for xc, yc, xq, yq, eta in val:
        for i in range(xc.shape[0]):
            lp = exact.log_ppd(xc[i], yc[i], xq[i], float(eta[i]))
            tot -= lp.gather(1, yq[i].view(-1, 1)).sum().item()
            cnt += yq.shape[1]
    return tot / cnt


def evaluate(model, val) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for xc, yc, xq, yq, eta in val:
            lp = model(xc, yc, xq, eta)
            tot -= lp.gather(2, yq.unsqueeze(-1)).sum().item()
            cnt += yq.numel()
    model.train()
    return tot / cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-max", type=int, default=64)
    ap.add_argument("--n-qry", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save-at", type=int, nargs="*", default=[100, 300, 1000, 3000])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)
    vgen = torch.Generator().manual_seed(10_000 + args.seed)
    out = HERE / "results"
    out.mkdir(exist_ok=True)

    # fixed validation set at several context sizes
    val = [sample_batch(64, n, args.n_qry, gen=vgen) for n in (2, 8, 16, 32, 64)]
    exact = ExactPPD()
    ref = bayes_reference(val, exact)
    print("exact-Bayes validation NLL (loss floor): %.4f" % ref, flush=True)

    model = PFN()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    curve = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        n = int(torch.randint(1, args.n_max + 1, (1,), generator=gen))
        xc, yc, xq, yq, eta = sample_batch(args.batch, n, args.n_qry, gen=gen)
        lp = model(xc, yc, xq, eta)
        loss = -lp.gather(2, yq.unsqueeze(-1)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step in args.save_at or step == args.steps:
            torch.save(model.state_dict(), out / f"pfn_step{step}.pt")
        if step % 250 == 0 or step in args.save_at or step == args.steps:
            v = evaluate(model, val)
            curve.append(dict(step=step, val_nll=v, excess_over_bayes=v - ref, seconds=time.time() - t0))
            print("step %5d  val NLL %.4f  excess %.4f  (%.0fs)" % (step, v, v - ref, time.time() - t0), flush=True)
    write_json(EXP, "training_curve.json",
               dict(bayes_val_nll=ref, curve=curve, config=vars(args), n_params=sum(p.numel() for p in model.parameters())),
               __file__, args.seed)


if __name__ == "__main__":
    main()

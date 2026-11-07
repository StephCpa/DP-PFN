"""E4-lite: a *trained* noise-aware PFN on label randomized response (Thm. D1, Prop. D2, H4, Gate 2).

  python experiments/E4-lite-noise-aware-rr/e4_rr.py train --mode aware --threads 2
  python experiments/E4-lite-noise-aware-rr/e4_rr.py train --mode naive --threads 2
  python experiments/E4-lite-noise-aware-rr/e4_rr.py eval

Prior (same as D2-check): w ~ N(0, 2^2), b ~ N(0, 1), x ~ U[-1, 1], y ~ Bern(sigmoid(w x + b)).
Release M: label randomized response, flip probability rho = 1 / (1 + e^eps) (eps-label-LDP).

  aware : trained on (x_i, M(y_i)) contexts with rho given as an input token, clean query
          targets y*.  Its loss minimiser is P(y* | x*, S) (Thm. D1).  eps ~ log-U[0.25, 6];
          10% of training tasks are clean (rho = 0).
  naive : the same architecture trained on clean contexts (rho input fixed at 0), then fed
          the RR release at test time, as off-the-shelf TabPFN would be.

Evaluation on clean test labels, against exact grid-Bayes targets:
  oracle (clean data), exact naive (RR labels treated as clean), exact noise-aware.
Metrics: accuracy, NLL, ECE, over-confidence (mean confidence - accuracy), and the mean
KL(exact noise-aware target || network) over queries.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "E1-lite-amortization-gap"))
from common.provenance import write_json  # noqa: E402
from pfn import PFN  # noqa: E402  (E1-lite architecture; its eta slot carries rho here)

EXP = "E4-lite-noise-aware-rr"
W_SD, B_SD = 2.0, 1.0


def rho_of(eps):
    return 1.0 / (1.0 + torch.exp(eps))


def sample(batch, n_ctx, n_qry, mode, gen, eps=None, clean_frac=0.1):
    """Returns xc, released labels, xq, clean yq, rho (input), clean yc."""
    w = torch.randn(batch, 1, generator=gen) * W_SD
    b = torch.randn(batch, 1, generator=gen) * B_SD
    x = torch.rand(batch, n_ctx + n_qry, generator=gen) * 2 - 1
    y = (torch.rand(batch, n_ctx + n_qry, generator=gen) < torch.sigmoid(w * x + b)).long()
    xc, yc, xq, yq = x[:, :n_ctx], y[:, :n_ctx], x[:, n_ctx:], y[:, n_ctx:]
    if eps is None:
        if mode == "naive":
            rho = torch.zeros(batch)
        else:
            le = torch.empty(batch).uniform_(math.log(0.25), math.log(6.0), generator=gen)
            rho = rho_of(le.exp())
            rho = torch.where(torch.rand(batch, generator=gen) < clean_frac, torch.zeros(batch), rho)
    else:
        rho = torch.full((batch,), float(rho_of(torch.tensor(float(eps)))) if eps != float("inf") else 0.0)
    flip = torch.rand(batch, n_ctx, generator=gen) < rho[:, None]
    yrel = torch.where(flip, 1 - yc, yc)
    return xc, yrel, xq, yq, rho, yc


# ------------------------------------------------------------------ exact targets
class Grid:
    def __init__(self, nw=161, nb=61):
        w = torch.linspace(-10, 10, nw, dtype=torch.float64)
        b = torch.linspace(-4.5, 4.5, nb, dtype=torch.float64)
        W, Bm = torch.meshgrid(w, b, indexing="ij")
        self.w, self.b = W.reshape(-1), Bm.reshape(-1)
        lp = -0.5 * (self.w / W_SD) ** 2 - 0.5 * (self.b / B_SD) ** 2
        self.lp = lp - torch.logsumexp(lp, 0)

    def ppd(self, xc, yl, xq, rho):
        """xc, yl: (B, n); xq: (B, m); rho: (B,) likelihood flip prob (0 = treat as clean)."""
        s = torch.sigmoid(xc.double().unsqueeze(-1) * self.w + self.b)  # (B, n, G)
        r = rho.double().view(-1, 1, 1)
        p1 = (1 - r) * s + r * (1 - s)
        ll = torch.where(yl.unsqueeze(-1).bool(), torch.log(p1), torch.log1p(-p1)).sum(1)
        post = self.lp + ll
        post = torch.exp(post - torch.logsumexp(post, -1, keepdim=True))  # (B, G)
        sq = torch.sigmoid(xq.double().unsqueeze(-1) * self.w + self.b)  # (B, m, G)
        return (sq * post.unsqueeze(1)).sum(-1)  # P(y*=1), (B, m)


def metrics(p1, y):
    """Per-task metrics averaged over tasks. p1, y: (B, m) numpy."""
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    conf = np.maximum(p1, 1 - p1)
    correct = ((p1 > 0.5).astype(int) == y).astype(float)
    nll = -(y * np.log(p1) + (1 - y) * np.log(1 - p1)).mean(1)
    edges = np.linspace(0.5, 1.0, 11)
    ece = np.zeros(p1.shape[0])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        cnt = m.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = np.abs((conf * m).sum(1) / np.maximum(cnt, 1) - (correct * m).sum(1) / np.maximum(cnt, 1))
        ece += cnt / p1.shape[1] * gap
    return dict(acc=float(correct.mean()), nll=float(nll.mean()), ece=float(ece.mean()),
                overconf=float((conf.mean(1) - correct.mean(1)).mean()))


def kl_bern(p, q):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return float((p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))).mean())


# ------------------------------------------------------------------ train
def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed + (0 if args.mode == "aware" else 1))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    model = PFN()
    if args.init:  # warm start (continued training); optimizer state starts fresh
        model.load_state_dict(torch.load(out / args.init, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    curve, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        n = int(torch.randint(args.n_min, args.n_max + 1, (1,), generator=gen))
        xc, yrel, xq, yq, rho, _ = sample(args.batch, n, args.n_qry, args.mode, gen)
        loss = -model(xc, yrel, xq, rho).gather(2, yq.unsqueeze(-1)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step in args.save_at:
            torch.save(model.state_dict(), out / f"pfn_{args.mode}{args.tag}_step{step}.pt")
        if step % 250 == 0 or step == args.steps:
            curve.append(dict(step=step, train_loss=float(loss), seconds=time.time() - t0))
            print("%s step %5d loss %.4f (%.0fs)" % (args.mode, step, float(loss), time.time() - t0), flush=True)
    torch.save(model.state_dict(), out / f"pfn_{args.mode}{args.tag}.pt")
    write_json(EXP, args.out or f"training_{args.mode}{args.tag}.json", dict(config=vars(args), curve=curve), __file__, args.seed)


# ------------------------------------------------------------------ evaluate
@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(args.threads)
    grid = Grid()
    nets = {}
    for mode in ("aware", "naive"):
        net = PFN()
        ckpt = args.aware_ckpt if (mode == "aware" and args.aware_ckpt) else f"pfn_{mode}.pt"
        net.load_state_dict(torch.load(HERE / "results" / ckpt, weights_only=True))
        nets[mode] = net.eval()
    rows = []
    for n in args.ns:
        for eps in args.epss:
            gen = torch.Generator().manual_seed(args.seed * 7919 + n * 31 + int(eps * 10) if eps != float("inf") else 99)
            acc = {}
            for chunk in range(args.tasks // args.chunk):
                xc, yrel, xq, yq, rho, yc = sample(args.chunk, n, args.queries, "aware", gen, eps=eps)
                zero = torch.zeros_like(rho)
                preds = {
                    "oracle_exact": grid.ppd(xc, yc, xq, zero),
                    "naive_exact": grid.ppd(xc, yrel, xq, zero),
                    "aware_exact": grid.ppd(xc, yrel, xq, rho),
                    "naive_pfn": nets["naive"](xc, yrel, xq, zero)[..., 1].exp().double(),
                    "aware_pfn": nets["aware"](xc, yrel, xq, rho)[..., 1].exp().double(),
                }
                y = yq.numpy()
                for k, p in preds.items():
                    for mk, mv in metrics(p.numpy(), y).items():
                        acc.setdefault(k, {}).setdefault(mk, []).append(mv)
                for k in ("aware_pfn", "naive_pfn", "naive_exact"):
                    acc[k].setdefault("kl_from_aware_exact", []).append(kl_bern(preds["aware_exact"].numpy(), preds[k].numpy()))
                acc["naive_pfn"].setdefault("kl_from_naive_exact", []).append(kl_bern(preds["naive_exact"].numpy(), preds["naive_pfn"].numpy()))
            res = {k: {mk: float(np.mean(v)) for mk, v in d.items()} for k, d in acc.items()}
            rows.append(dict(n=n, eps=eps, rho=float(rho[0]), results=res))
            print("n=%4d eps=%s rho=%.3f" % (n, eps, float(rho[0])))
            for k, v in res.items():
                extra = "".join("  %s %.4f" % (kk, vv) for kk, vv in v.items() if kk.startswith("kl"))
                print("   %-13s acc %.3f  NLL %.4f  ECE %.4f  overconf %+.4f%s" % (k, v["acc"], v["nll"], v["ece"], v["overconf"], extra), flush=True)
    write_json(EXP, args.out or "evaluation.json", dict(config=vars(args), rows=rows), __file__, args.seed)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--mode", choices=["aware", "naive"], required=True)
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--n-min", type=int, default=8)
    t.add_argument("--n-max", type=int, default=128)
    t.add_argument("--n-qry", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--init", type=str, default="", help="checkpoint in results/ to warm-start from")
    t.add_argument("--tag", type=str, default="", help="suffix for checkpoint and curve names")
    t.add_argument("--save-at", type=int, nargs="*", default=[], help="extra checkpoints at these (local) steps")
    t.add_argument("--out", type=str, default="")
    e = sub.add_parser("eval")
    e.add_argument("--ns", type=int, nargs="*", default=[32, 128])
    e.add_argument("--epss", type=float, nargs="*", default=[0.5, 1.0, 2.0, float("inf")])
    e.add_argument("--tasks", type=int, default=200)
    e.add_argument("--chunk", type=int, default=25)
    e.add_argument("--queries", type=int, default=400)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--aware-ckpt", type=str, default="", help="aware checkpoint in results/ (default pfn_aware.pt)")
    e.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()

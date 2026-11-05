"""E4b-lite: trained noise-aware PFN on high-dimensional, marginal-based DP synthetic data.

  python experiments/E4b-lite-noise-aware-synthetic/e4b_syn.py train --mode aware --threads 2
  python experiments/E4b-lite-noise-aware-synthetic/e4b_syn.py train --mode naive --threads 2
  python experiments/E4b-lite-noise-aware-synthetic/e4b_syn.py eval

D2-sparse released a noisy *full* histogram, which flattens toward uniform and made naive
use under-confident. This run tests the untested case it names: a *marginal-based*
generator (PrivBayes/MST family) and a *trained* network.

Task prior (d = 16 correlated binary features, logistic label):
    A ~ N(0, 0.8^2)^{d x 2}, f ~ N(0, I_2), z = A f + N(0, I_d), t ~ N(0, 0.5^2)^d, x_j = 1[z_j > t_j]
    y ~ Bern(sigmoid(b + sum_j w_j (2 x_j - 1))),  w_j ~ N(0, 0.55^2),  b ~ N(0, 0.5^2)
Release M (eps-DP, replace-one rows): Laplace(2d/eps) on every cell of the d tables
    T_j[a, c] = #{x_j = a, y = c} (joint L1 sensitivity 2d); clip at 0, add 0.5;
    p(y) from the averaged column sums, p(x_j | y) per table; then N_syn = k n rows drawn
    from that naive-Bayes model.  Feature-feature dependence is not released.
Networks (same architecture, vector-input PFN):
    naive : trained on clean contexts of size 16..512 (noise tokens 0); fed S at test time,
            i.e. "TabPFN on DP synthetic data".
    aware : trained on S = M(D) with tokens (log eps, log2 k, log n) and CLEAN query targets;
            its loss minimiser is P(y* | x*, S) (Thm. D1).
Reference predictors: oracle = naive net on the real records D; nb_plugin = naive Bayes on
the noisy tables (what the rows encode as k -> inf); nb_clean = non-private naive Bayes on D.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "E4-lite-noise-aware-rr"))
from common.provenance import write_json  # noqa: E402
from e4_rr import metrics  # noqa: E402

EXP = "E4b-lite-noise-aware-synthetic"
D = 16


# ------------------------------------------------------------------ prior and release
def draw_tasks(B, gen):
    A = torch.randn(B, D, 2, generator=gen) * 0.8
    t = torch.randn(B, D, generator=gen) * 0.5
    w = torch.randn(B, D, generator=gen) * 0.55
    b = torch.randn(B, generator=gen) * 0.5
    return A, t, w, b


def draw_rows(task, m, gen):
    A, t, w, b = task
    B = A.shape[0]
    f = torch.randn(B, m, 2, generator=gen)
    z = torch.einsum("bdr,bmr->bmd", A, f) + torch.randn(B, m, D, generator=gen)
    x = (z > t[:, None, :]).float()
    logit = b[:, None] + ((2 * x - 1) * w[:, None, :]).sum(-1)
    y = (torch.rand(B, m, generator=gen) < torch.sigmoid(logit)).long()
    return x, y


def noisy_tables(x, y, eps, gen):
    """x (B, n, D), y (B, n); eps (B,). Returns noisy p(y=1) (B,), p(x_j=1|y=c) (B, D, 2)."""
    yc = torch.stack([(y == 0).float(), (y == 1).float()], -1)  # (B, n, 2)
    T1 = torch.einsum("bnd,bnc->bdc", x, yc)  # counts x_j=1, y=c
    T0 = torch.einsum("bnd,bnc->bdc", 1 - x, yc)  # counts x_j=0, y=c
    scale = (2 * D / eps).view(-1, 1, 1)
    lap = torch.distributions.Laplace(0.0, 1.0)
    T1 = (T1 + lap.sample(T1.shape) * scale).clamp_min(0) + 0.5
    T0 = (T0 + lap.sample(T0.shape) * scale).clamp_min(0) + 0.5
    col = T1 + T0  # (B, D, 2) noisy count of y=c, per table
    py1 = (col[..., 1] / col.sum(-1)).mean(-1)  # (B,)
    px1 = T1 / col  # p(x_j=1 | y=c)
    return py1, px1


def synth(py1, px1, N, gen):
    B = py1.shape[0]
    ys = (torch.rand(B, N, generator=gen) < py1[:, None]).long()
    p = torch.where(ys.bool().unsqueeze(-1), px1[:, None, :, 1], px1[:, None, :, 0])  # (B, N, D)
    xs = (torch.rand(B, N, D, generator=gen) < p).float()
    return xs, ys


def nb_predict(py1, px1, xq):
    """Naive-Bayes P(y=1 | xq) from p(y), p(x_j|y). xq (B, m, D)."""
    l1 = torch.log(py1)[:, None] + (xq * torch.log(px1[:, None, :, 1]) + (1 - xq) * torch.log1p(-px1[:, None, :, 1])).sum(-1)
    l0 = torch.log1p(-py1)[:, None] + (xq * torch.log(px1[:, None, :, 0]) + (1 - xq) * torch.log1p(-px1[:, None, :, 0])).sum(-1)
    return torch.sigmoid(l1 - l0)


def nb_clean_tables(x, y):
    yc = torch.stack([(y == 0).float(), (y == 1).float()], -1)
    T1 = torch.einsum("bnd,bnc->bdc", x, yc) + 0.5
    T0 = torch.einsum("bnd,bnc->bdc", 1 - x, yc) + 0.5
    col = T1 + T0
    return (col[..., 1] / col.sum(-1)).mean(-1), T1 / col


def tokens(eps, k, n):
    return torch.stack([torch.log(eps) / 3, torch.log2(k) / 2, torch.log(n) / 5], -1)


# ------------------------------------------------------------------ network
class VecPFN(nn.Module):
    """E1-lite PFN with a D-dimensional row encoder and a 3-scalar release token."""

    def __init__(self, d=128, heads=4, layers=4, ff=256):
        super().__init__()
        self.x_enc = nn.Sequential(nn.Linear(D, d), nn.GELU(), nn.Linear(d, d))
        self.t_enc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.y_emb = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, xc, yc, xq, tok):
        B, n, _ = xc.shape
        m = xq.shape[1]
        e = self.t_enc(tok).unsqueeze(1)
        ctx = self.x_enc(2 * xc - 1) + self.y_emb(yc) + e
        qry = self.x_enc(2 * xq - 1) + self.y_emb(torch.full((B, m), 2, dtype=torch.long)) + e
        mask = torch.zeros(n + m, n + m, dtype=torch.bool)
        mask[:, n:] = True
        h = self.body(torch.cat([ctx, qry], 1), mask=mask)
        return torch.log_softmax(self.head(h[:, n:]), -1)


# ------------------------------------------------------------------ train
def train(args):
    torch.manual_seed(args.seed + (0 if args.mode == "aware" else 1))
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed + (0 if args.mode == "aware" else 1))
    (HERE / "results").mkdir(exist_ok=True)
    model = VecPFN()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    curve, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        task = draw_tasks(args.batch, gen)
        if args.mode == "aware":
            n = int(torch.randint(16, 129, (1,), generator=gen))
            k = float([1, 2, 4][int(torch.randint(0, 3, (1,), generator=gen))])
            x, y = draw_rows(task, n + args.n_qry, gen)
            xd, yd, xq, yq = x[:, :n], y[:, :n], x[:, n:], y[:, n:]
            eps = torch.exp(torch.empty(args.batch).uniform_(math.log(0.5), math.log(32.0), generator=gen))
            py1, px1 = noisy_tables(xd, yd, eps, gen)
            xc, yc = synth(py1, px1, int(k * n), gen)
            tok = tokens(eps, torch.full((args.batch,), k), torch.full((args.batch,), float(n)))
        else:
            N = int(torch.randint(16, 513, (1,), generator=gen))
            x, y = draw_rows(task, N + args.n_qry, gen)
            xc, yc, xq, yq = x[:, :N], y[:, :N], x[:, N:], y[:, N:]
            tok = torch.zeros(args.batch, 3)
        loss = -model(xc, yc, xq, tok).gather(2, yq.unsqueeze(-1)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps:
            curve.append(dict(step=step, train_loss=float(loss), seconds=time.time() - t0))
            print("%s step %5d loss %.4f (%.0fs)" % (args.mode, step, float(loss), time.time() - t0), flush=True)
    torch.save(model.state_dict(), HERE / "results" / f"pfn_{args.mode}.pt")
    write_json(EXP, f"training_{args.mode}.json", dict(config=vars(args), curve=curve), __file__, args.seed)


# ------------------------------------------------------------------ evaluate
@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(args.threads)
    nets = {}
    for mode in ("aware", "naive"):
        net = VecPFN()
        net.load_state_dict(torch.load(HERE / "results" / f"pfn_{mode}.pt", weights_only=True))
        nets[mode] = net.eval()
    rows = []
    for n in args.ns:
        for eps_v in args.epss:
            for k in args.ks:
                gen = torch.Generator().manual_seed(args.seed * 100003 + n * 101 + int(eps_v * 10) * 7 + k)
                torch.manual_seed(args.seed * 100003 + n * 101 + int(eps_v * 10) * 7 + k)  # Laplace sampler
                acc = {}
                for _ in range(args.tasks // args.chunk):
                    task = draw_tasks(args.chunk, gen)
                    x, y = draw_rows(task, n + args.queries, gen)
                    xd, yd, xq, yq = x[:, :n], y[:, :n], x[:, n:], y[:, n:]
                    eps = torch.full((args.chunk,), float(eps_v))
                    py1, px1 = noisy_tables(xd, yd, eps, gen)
                    xs, ys = synth(py1, px1, k * n, gen)
                    tok = tokens(eps, torch.full((args.chunk,), float(k)), torch.full((args.chunk,), float(n)))
                    zero = torch.zeros(args.chunk, 3)
                    cpy, cpx = nb_clean_tables(xd, yd)
                    preds = {
                        "oracle_pfn": nets["naive"](xd, yd, xq, zero)[..., 1].exp(),
                        "naive_pfn": nets["naive"](xs, ys, xq, zero)[..., 1].exp(),
                        "aware_pfn": nets["aware"](xs, ys, xq, tok)[..., 1].exp(),
                        "nb_plugin": nb_predict(py1, px1, xq),
                        "nb_clean": nb_predict(cpy, cpx, xq),
                    }
                    yn = yq.numpy()
                    for key, p in preds.items():
                        for mk, mv in metrics(p.double().numpy(), yn).items():
                            acc.setdefault(key, {}).setdefault(mk, []).append(mv)
                res = {key: {mk: float(np.mean(v)) for mk, v in d.items()} for key, d in acc.items()}
                rows.append(dict(n=n, eps=eps_v, k=k, results=res))
                print("n=%4d eps=%-4g k=%d" % (n, eps_v, k))
                for key, v in res.items():
                    print("   %-11s acc %.3f  NLL %.4f  ECE %.4f  overconf %+.4f" % (key, v["acc"], v["nll"], v["ece"], v["overconf"]), flush=True)
    write_json(EXP, "evaluation.json", dict(config=vars(args), rows=rows), __file__, args.seed)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--mode", choices=["aware", "naive"], required=True)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--n-qry", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--ns", type=int, nargs="*", default=[32, 128])
    e.add_argument("--epss", type=float, nargs="*", default=[1.0, 4.0, 16.0])
    e.add_argument("--ks", type=int, nargs="*", default=[1, 4])
    e.add_argument("--tasks", type=int, default=200)
    e.add_argument("--chunk", type=int, default=25)
    e.add_argument("--queries", type=int, default=400)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()

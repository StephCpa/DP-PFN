"""E1-lite evaluation: exact vs amortized privacy loss (tests Prop. B2, Obs. B5, H2).

python experiments/E1-lite-amortization-gap/evaluate.py

For each checkpoint, label floor eta and context size n it reports, under
label-DP (flip one context label, features fixed):

  typical      datasets drawn from the prior:
               Lambda(D)  = max_{x*, y} |log q - log P|             (approximation gap)
               eps_exact(D) = max_{j, x*, y} |log P(y|x*,D) - log P(y|x*,D^j)|
               s_hat(D)   = the same for the network q_theta
  adv_in       gradient search over in-support datasets x in [-1,1]^n, x* in [-1,1]
  adv_out      gradient search over prior-null datasets x in [-5,5]^n, x* in [-5,5]

Each search is run twice: once maximising the network's leakage (a lower bound on
s(theta, n)) and once maximising the approximation gap Lambda(D).  On every
found dataset the exact mechanism's leakage is evaluated too, so the two can be
compared on the same input.  2 B_y(eta) is the Theorem A2 bound for the exact PPD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from pfn import PFN, ExactPPD, sample_batch, two_B_y  # noqa: E402
from common.provenance import write_json  # noqa: E402

EXP = "E1-lite-amortization-gap"
XQ_GRID_IN = torch.linspace(-1, 1, 41)


def exact_batch(ex: ExactPPD, xc, yc, xq, eta: float):
    """Batched exact log PPD. xc, yc: (B, n); xq: (B, m) -> (B, m, 2) float64."""
    l1, l0 = ex._logp1(xc, eta)  # (B, n, G)
    ll = torch.where(yc.unsqueeze(-1).bool(), l1, l0).sum(1)  # (B, G)
    post = ex.log_prior + ll
    post = post - torch.logsumexp(post, -1, keepdim=True)
    q1, q0 = ex._logp1(xq, eta)  # (B, m, G)
    lp1 = torch.logsumexp(post.unsqueeze(1) + q1, -1)
    lp0 = torch.logsumexp(post.unsqueeze(1) + q0, -1)
    return torch.stack([lp0, lp1], -1)


def flip(yc, j):
    y2 = yc.clone()
    y2[:, j] = 1 - y2[:, j]
    return y2


def leakage_all_rows(fn, xc, yc, xq):
    """max over rows j, queries and labels of |log f(D) - log f(D^j)|, per dataset."""
    base = fn(xc, yc, xq)
    worst = torch.zeros(xc.shape[0], dtype=torch.float64)
    for j in range(xc.shape[1]):
        other = fn(xc, flip(yc, j), xq)
        worst = torch.maximum(worst, (base - other).abs().amax((1, 2)).double())
    return worst


def typical(model, ex, eta, n, reps, gen):
    xc, yc, _, _, _ = sample_batch(reps, n, 1, eta_range=(eta, eta), gen=gen)
    xq = XQ_GRID_IN.expand(reps, -1)
    et = torch.full((reps,), eta)
    with torch.no_grad():
        q = lambda a, b, c: model(a, b, c, et)
        P = lambda a, b, c: exact_batch(ex, a, b, c, eta)
        lam = (q(xc, yc, xq).double() - P(xc, yc, xq)).abs().amax((1, 2))
        s_hat = leakage_all_rows(q, xc, yc, xq)
        e_ex = leakage_all_rows(P, xc, yc, xq)
    return lam.numpy(), s_hat.numpy(), e_ex.numpy()


def search(model, ex, eta, n, radius, objective, restarts, steps, gen):
    """Gradient ascent over (x_1..x_n, x*) with random labels; row 0 is flipped.

    objective = "leak_q"  : |log q(D) - log q(D^0)|   (lower bound on s(theta, n))
                "leak_P"  : |log P(D) - log P(D^0)|   (exact mechanism)
                "gap"     : |log q(D) - log P(D)|     (Lambda(D))
    """
    et = torch.full((restarts,), eta)
    u = (torch.randn(restarts, n, generator=gen) * 1.0).requires_grad_(True)
    uq = (torch.randn(restarts, 1, generator=gen) * 1.0).requires_grad_(True)
    yc = torch.randint(0, 2, (restarts, n), generator=gen)
    opt = torch.optim.Adam([u, uq], lr=0.08)
    for p in model.parameters():
        p.requires_grad_(False)

    def terms(xc, xq):
        lq = model(xc, yc, xq, et).double()
        lq2 = model(xc, flip(yc, 0), xq, et).double()
        lP = exact_batch(ex, xc, yc, xq, eta)
        lP2 = exact_batch(ex, xc, flip(yc, 0), xq, eta)
        return lq, lq2, lP, lP2

    for _ in range(steps):
        xc, xq = radius * torch.tanh(u), radius * torch.tanh(uq)
        lq, lq2, lP, lP2 = terms(xc, xq)
        if objective == "leak_q":
            f = (lq - lq2).abs().amax((1, 2))
        elif objective == "leak_P":
            f = (lP - lP2).abs().amax((1, 2))
        else:
            f = (lq - lP).abs().amax((1, 2))
        opt.zero_grad()
        (-f.sum()).backward()
        opt.step()
    with torch.no_grad():
        xc, xq = radius * torch.tanh(u), radius * torch.tanh(uq)
        # evaluate on the found x* plus the whole in-support query grid
        grid = torch.cat([xq, XQ_GRID_IN.expand(restarts, -1) * radius], 1)
        q = lambda a, b, c: model(a, b, c, et)
        P = lambda a, b, c: exact_batch(ex, a, b, c, eta)
        leak_q = leakage_all_rows(q, xc, yc, grid)
        leak_P = leakage_all_rows(P, xc, yc, grid)
        gap = (q(xc, yc, grid).double() - P(xc, yc, grid)).abs().amax((1, 2))
    best = int(torch.argmax({"leak_q": leak_q, "leak_P": leak_P, "gap": gap}[objective]))
    return dict(leak_q=float(leak_q[best]), leak_exact=float(leak_P[best]), gap=float(gap[best]),
                leak_q_median=float(leak_q.median()), leak_exact_median=float(leak_P.median()),
                gap_median=float(gap.median()))


def summarize(a):
    a = np.asarray(a)
    return dict(median=float(np.median(a)), p90=float(np.quantile(a, 0.9)), max=float(a.max()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-list", type=int, nargs="*", default=[100, 300, 1000, 3000, 6000])
    ap.add_argument("--etas", type=float, nargs="*", default=[0.1, 0.5])
    ap.add_argument("--ns", type=int, nargs="*", default=[4, 16, 64])
    ap.add_argument("--reps", type=int, default=64)
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--search-steps", type=int, default=80)
    ap.add_argument("--adv-checkpoints", type=int, nargs="*", default=[300, 6000])
    ap.add_argument("--adv-ns", type=int, nargs="*", default=[4, 16])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ex = ExactPPD()
    rows = []
    for step in args.steps_list:
        model = PFN()
        model.load_state_dict(torch.load(HERE / "results" / f"pfn_step{step}.pt"))
        model.eval()
        for eta in args.etas:
            for n in args.ns:
                gen = torch.Generator().manual_seed(args.seed * 1000 + n)
                lam, s_hat, e_ex = typical(model, ex, eta, n, args.reps, gen)
                row = dict(step=step, eta=eta, n=n, two_B_y=two_B_y(eta),
                           typical=dict(Lambda=summarize(lam), s_hat=summarize(s_hat), eps_exact=summarize(e_ex)))
                if step in args.adv_checkpoints and n in args.adv_ns:
                    for name, radius in (("adv_in", 1.0), ("adv_out", 5.0)):
                        row[name] = {obj: search(model, ex, eta, n, radius, obj, args.restarts, args.search_steps, gen)
                                     for obj in ("leak_q", "leak_P", "gap")}
                rows.append(row)
                t = row["typical"]
                msg = "step %5d eta %.1f n %2d | typical: Lambda med %.3f max %.3f | s_hat med %.3f max %.3f | exact med %.3f max %.3f | 2B_y %.2f" % (
                    step, eta, n, t["Lambda"]["median"], t["Lambda"]["max"], t["s_hat"]["median"], t["s_hat"]["max"],
                    t["eps_exact"]["median"], t["eps_exact"]["max"], two_B_y(eta))
                print(msg, flush=True)
                for name in ("adv_in", "adv_out"):
                    if name in row:
                        r = row[name]
                        print("    %s: max s_hat %.2f (exact there %.2f) | max exact %.2f (s_hat there %.2f) | max Lambda %.2f" % (
                            name, r["leak_q"]["leak_q"], r["leak_q"]["leak_exact"], r["leak_P"]["leak_exact"],
                            r["leak_P"]["leak_q"], r["gap"]["gap"]), flush=True)
    write_json(EXP, "evaluation.json", dict(rows=rows, config=vars(args)), __file__, args.seed)


if __name__ == "__main__":
    main()

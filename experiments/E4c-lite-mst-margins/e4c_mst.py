"""E4c-lite: E4b's synthesizer plus a maximum spanning tree of feature pairs.

python experiments/E4c-lite-mst-margins/e4c_mst.py train --mode aware --steps 4000
python experiments/E4c-lite-mst-margins/e4c_mst.py eval

The total privacy budget equals E4b's. Half pays for the feature-label tables and
half for the d - 1 tree edges, so each query is noisier than in E4b. See README.md.
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
sys.path.insert(0, str(HERE.parent / "E4b-lite-noise-aware-synthetic"))
sys.path.insert(0, str(HERE.parent / "E4-lite-noise-aware-rr"))
from common.provenance import write_json  # noqa: E402
from e4_rr import metrics  # noqa: E402
from e4b_syn import D, VecPFN, draw_rows, draw_tasks, nb_clean_tables, nb_predict, tokens  # noqa: E402

EXP = "E4c-lite-mst-margins"


def _laplace(shape, scale):
    return torch.distributions.Laplace(0.0, 1.0).sample(shape) * scale


def release(x, y, eps):
    """Release feature-label tables and one MST of feature pairs. Total budget eps.

    The budget is split in half. The d feature-label tables have joint
    sensitivity 2d, so at budget eps/2 their Laplace scale is 4d/eps. The same
    accounting gives scale 4(d-1)/eps for the d-1 pair tables. The tree is chosen
    from the already-released single-feature margins, which is post-processing.

    Returns py1 (B,), px1 (B, D, 2), pair (B, d-1, 2, 2, 2) and edges.
    pair[:, e, a, b, c] is the noisy probability of (x_u=a, x_v=b, y=c).
    """
    B = x.shape[0]
    yc = torch.stack([(y == 0).float(), (y == 1).float()], -1)
    scale_lab = (4 * D / eps).view(-1, 1, 1)
    T1 = torch.einsum("bnd,bnc->bdc", x, yc)
    T0 = torch.einsum("bnd,bnc->bdc", 1 - x, yc)
    T1 = (T1 + _laplace(T1.shape, scale_lab)).clamp_min(0) + 0.5
    T0 = (T0 + _laplace(T0.shape, scale_lab)).clamp_min(0) + 0.5
    col = T1 + T0
    py1 = (col[..., 1] / col.sum(-1)).mean(-1)
    px1 = T1 / col

    # score pairs by dependence implied by the released margins, then release
    # only the chosen tree. Selection is post-processing of the first release.
    # dependence score from the released label-conditional margins: a feature whose
    # two class-conditionals differ is informative, and pairs of such features are
    # the ones the factor model correlates. Used only to pick the tree.
    # One shared tree per call, from the batch-averaged margins. Recomputing a
    # tree per dataset would make training hundreds of times slower, and the
    # margins are already noisy, so the choice is still data-dependent only
    # through the released statistics.
    gap = (px1[:, :, 1] - px1[:, :, 0]).abs().mean(0)
    scores = (gap[:, None] * gap[None, :]).expand(x.shape[0], -1, -1)
    edges = _mst_edges(scores)

    scale_pair = (4 * (D - 1) / eps).view(-1)
    pair = torch.zeros(B, D - 1, 2, 2, 2)
    for e, (u, v) in enumerate(edges):
        for a in (0, 1):
            for b in (0, 1):
                ind = ((x[:, :, u] == a) & (x[:, :, v] == b)).float().unsqueeze(-1) * yc
                count = ind.sum(1)
                pair[:, e, a, b] = (count + _laplace(count.shape, scale_pair[:, None])).clamp_min(0) + 0.5
        pair[:, e] = pair[:, e] / pair[:, e].sum(dim=(1, 2), keepdim=True)
    return py1, px1, pair, edges


def _mst_edges(scores):
    """Prim's algorithm on one shared score matrix averaged over the batch.

    The tree is computed from the batch-averaged scores. That keeps the edge set
    identical for every row in a training batch, which the sampler needs. The
    scores are noisy, so the choice is still a function of the mechanism only.
    """
    s = scores.mean(0)
    D_ = s.shape[0]
    chosen = [0]
    edges = []
    left = set(range(1, D_))
    while left:
        best, bu, bv = -1.0, None, None
        for u in chosen:
            for v in left:
                a, b = (u, v) if u < v else (v, u)
                val = float(s[a, b])
                if val > best:
                    best, bu, bv = val, u, v
        edges.append((bu, bv))
        chosen.append(bv)
        left.remove(bv)
    return edges


def _mst(scores):
    return _mst_edges(scores)


def synth_tree(py1, px1, pair, edges, N, gen):
    """Draw N rows: label first, root feature from its margin, then along the tree."""
    B = py1.shape[0]
    ys = (torch.rand(B, N, generator=gen) < py1[:, None]).long()
    xs = torch.zeros(B, N, D)
    degree = torch.zeros(D)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    root = int(degree.argmax())
    parent = {root: None}
    for u, v in edges:
        if u in parent and v not in parent:
            parent[v] = u
        elif v in parent and u not in parent:
            parent[u] = v
    order = [root]
    while len(order) < D:
        for node, par in list(parent.items()):
            if node not in order and (par is None or par in order):
                order.append(node)
    p_root = torch.where(ys.bool(), px1[:, root, 1][:, None], px1[:, root, 0][:, None])
    xs[:, :, root] = (torch.rand(B, N, generator=gen) < p_root).float()
    edge_index = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(edges)}
    for node in order[1:]:
        par = parent[node]
        e = edge_index[(min(node, par), max(node, par))]
        u, v = edges[e]
        par_val = xs[:, :, par].long()
        tab = pair[:, e]
        # P(x_node=1 | x_parent=a, y=c), tabulated once per dataset
        cond = torch.zeros(B, 2, 2)
        for a in (0, 1):
            for c in (0, 1):
                cell = tab[:, a, :, c] if par == u else tab[:, :, a, c]
                cond[:, a, c] = cell[:, 1] / cell.sum(-1).clamp_min(1e-6)
        p = cond[torch.arange(B)[:, None], par_val, ys]
        xs[:, :, node] = (torch.rand(B, N, generator=gen) < p).float()
    return xs, ys


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
            py1, px1, pair, edges = release(xd, yd, eps)
            xc, yc = synth_tree(py1, px1, pair, edges, int(k * n), gen)
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
                seed = args.seed * 100003 + n * 101 + int(eps_v * 10) * 7 + k
                gen = torch.Generator().manual_seed(seed)
                torch.manual_seed(seed)
                acc = {}
                for _ in range(args.tasks // args.chunk):
                    task = draw_tasks(args.chunk, gen)
                    x, y = draw_rows(task, n + args.queries, gen)
                    xd, yd, xq, yq = x[:, :n], y[:, :n], x[:, n:], y[:, n:]
                    eps = torch.full((args.chunk,), float(eps_v))
                    py1, px1, pair, edges = release(xd, yd, eps)
                    xs, ys = synth_tree(py1, px1, pair, edges, k * n, gen)
                    tok = tokens(eps, torch.full((args.chunk,), float(k)), torch.full((args.chunk,), float(n)))
                    zero = torch.zeros(args.chunk, 3)
                    cpy, cpx = nb_clean_tables(xd, yd)
                    preds = {
                        "oracle_pfn": nets["naive"](xd, yd, xq, zero)[..., 1].exp(),
                        "naive_pfn": nets["naive"](xs, ys, xq, zero)[..., 1].exp(),
                        "aware_pfn": nets["aware"](xs, ys, xq, tok)[..., 1].exp(),
                        "nb_clean": nb_predict(cpy, cpx, xq),
                    }
                    yn = yq.numpy()
                    for key, p in preds.items():
                        for mk, mv in metrics(p.double().numpy(), yn).items():
                            acc.setdefault(key, {}).setdefault(mk, []).append(mv)
                res = {key: {mk: float(np.mean(v)) for mk, v in d.items()} for key, d in acc.items()}
                rows.append(dict(n=n, eps=eps_v, k=k, results=res))
                print("n=%4d eps=%-4g k=%d" % (n, eps_v, k), flush=True)
                for key, v in res.items():
                    print("   %-11s acc %.3f  NLL %.4f  ECE %.4f  overconf %+.4f" % (
                        key, v["acc"], v["nll"], v["ece"], v["overconf"]), flush=True)
    write_json(EXP, "evaluation.json", dict(config=vars(args), rows=rows), __file__, args.seed)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--mode", choices=["aware", "naive"], required=True)
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--n-qry", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--threads", type=int, default=4)
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

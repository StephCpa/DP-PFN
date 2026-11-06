"""E4c-prime: per-dataset, privately selected MST tree of feature pairs (Prop. D2, H4).

  python experiments/E4c-prime-mst-selection/e4cp_mst.py train --threads 3
  python experiments/E4c-prime-mst-selection/e4cp_mst.py eval

E4c-lite released 15 feature-pair tables along a star chosen by label informativeness,
shared across a batch.  This run selects a tree *per dataset* by feature-feature
dependence, privately, the way MST does, and compares four releases built from the
same data with E4b's naive network.

Releases (total budget eps, replace-one rows):
  nb          E4b: Laplace(2d/eps) on the d (x_j, y) tables; naive-Bayes synthesis.
  star        E4c: eps/2 on (x_j, y) tables, eps/2 on 15 (x_u, x_v, y) tables along a
              batch-shared star scored by label gaps.
  mst_dp      E4c-prime, eps split in thirds as in MST:
                eps/3  (x_j, y) tables, Laplace scale 2d/(eps/3) = 6d/eps;
                eps/3  tree selection: Prim from node 0, 15 exponential-mechanism picks at
                       eps/45 each, utility q(u,v) = sum_ab |N_ab - N_a. N_.b / n| on the
                       (x_u, x_v) counts. q = 4 |N_11 - N_1. N_.1 / n| and one replaced row
                       moves N_11 - N_1. N_.1 / n by at most 1, so sensitivity is 4;
                eps/3  the 15 selected (x_u, x_v, y) tables, Laplace scale 2(d-1)/(eps/3).
              Synthesis: label, root x_0 | y, then each new node | (its tree parent, y),
              i.e. a tree-augmented naive Bayes.  Pure eps-DP by basic composition.
  tree_oracle DIAGNOSTIC, NOT DP: the per-dataset maximum spanning tree on the clean q,
              then eps/2 label tables and eps/2 pair tables.  Shows what a good tree buys
              when selection is free.
Networks: naive = E4b's clean-trained network (reused, so cells are paired with E4b);
aware_mst = trained here with mst_dp inside the prior (E4b recipe); aware_nb = E4b's.
Selection quality: mean population |corr| of the chosen edges (from 4,000 fresh rows per task).
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
EXPS = HERE.parent
for sub in ("", "E4b-lite-noise-aware-synthetic", "E4-lite-noise-aware-rr", "E4c-lite-mst-margins"):
    sys.path.insert(0, str(EXPS / sub))
from common.provenance import write_json  # noqa: E402
from e4_rr import metrics  # noqa: E402
from e4b_syn import D, VecPFN, draw_rows, draw_tasks, nb_clean_tables, nb_predict, noisy_tables, synth, tokens  # noqa: E402
import e4c_mst  # noqa: E402

EXP = "E4c-prime-mst-selection"
SENS_Q = 4.0


def lap(shape, scale, gen):
    """Laplace(0, scale) from an explicit generator (inverse CDF)."""
    u = torch.rand(shape, generator=gen) - 0.5
    # 0.5 - 2**-25 is the largest float32 below 0.5: this only changes the case rand == 0 exactly,
    # which otherwise gives log(0) = -inf.  (0.5 - 1e-12 rounds to 0.5 in float32 and did not guard it.)
    return -scale * torch.sign(u) * torch.log1p(-2 * u.abs().clamp(max=0.5 - 2.0 ** -25))


def dependence(x):
    """q(u, v) for all pairs: 4 |N11 - N1. N.1 / n|. x (B, n, D) -> (B, D, D)."""
    n = x.shape[1]
    N11 = torch.einsum("bni,bnj->bij", x, x)
    N1 = x.sum(1)
    return 4 * (N11 - N1[:, :, None] * N1[:, None, :] / n).abs()


def prim(q, eps_sel, gen, private=True):
    """Per-dataset Prim tree. Returns parent (B, D-1), child (B, D-1) in insertion order (root 0).

    private: each pick samples an edge (u in tree, v outside) with probability
    proportional to exp(eps_sel q / (2 * SENS_Q)) via Gumbel-max; else argmax.
    """
    B = q.shape[0]
    in_tree = torch.zeros(B, D, dtype=torch.bool)
    in_tree[:, 0] = True
    par, chi = [], []
    for _ in range(D - 1):
        cand = in_tree[:, :, None] & ~in_tree[:, None, :]  # (B, u, v)
        score = q * (eps_sel.view(-1, 1, 1) / (2 * SENS_Q)) if private else q.clone()
        if private:
            g = -torch.log(-torch.log(torch.rand(score.shape, generator=gen).clamp_min(1e-12)))
            score = score + g
        score = score.masked_fill(~cand, float("-inf")).view(B, -1)
        idx = score.argmax(-1)
        u, v = idx // D, idx % D
        par.append(u)
        chi.append(v)
        in_tree[torch.arange(B), v] = True
    return torch.stack(par, 1), torch.stack(chi, 1)


def label_tables(x, y, eps_part, gen):
    """Noisy (x_j, y) tables at budget eps_part. Returns py1 (B,), px1 (B, D, 2)."""
    yc = torch.stack([(y == 0).float(), (y == 1).float()], -1)
    T1 = torch.einsum("bnd,bnc->bdc", x, yc)
    T0 = torch.einsum("bnd,bnc->bdc", 1 - x, yc)
    scale = (2 * D / eps_part).view(-1, 1, 1)
    T1 = (T1 + lap(T1.shape, 1.0, gen) * scale).clamp_min(0) + 0.5
    T0 = (T0 + lap(T0.shape, 1.0, gen) * scale).clamp_min(0) + 0.5
    col = T1 + T0
    return (col[..., 1] / col.sum(-1)).mean(-1), T1 / col


def pair_tables(x, y, par, chi, eps_part, gen):
    """Noisy (x_parent, x_child, y) tables for each edge -> P(x_child = 1 | x_parent, y), (B, D-1, 2, 2)."""
    B, n, _ = x.shape
    xp = torch.gather(x, 2, par[:, None, :].expand(B, n, D - 1))
    xc = torch.gather(x, 2, chi[:, None, :].expand(B, n, D - 1))
    yc = torch.stack([(y == 0).float(), (y == 1).float()], -1)  # (B, n, 2)
    scale = (2 * (D - 1) / eps_part).view(-1, 1, 1, 1, 1)
    T = torch.zeros(B, D - 1, 2, 2, 2)  # [edge, parent a, child b, label c]
    for a in (0, 1):
        for b in (0, 1):
            ind = ((xp == a) & (xc == b)).float()  # (B, n, E)
            T[:, :, a, b, :] = torch.einsum("bne,bnc->bec", ind, yc)
    T = (T + lap(T.shape, 1.0, gen) * scale).clamp_min(0) + 0.5
    return T[:, :, :, 1, :] / T.sum(3)  # P(child=1 | parent=a, y=c): (B, E, 2, 2)


def tree_synth(py1, px1, cond, par, chi, N, gen):
    """Label, root x_0 | y, then children in Prim order | (parent, y)."""
    B = py1.shape[0]
    ys = (torch.rand(B, N, generator=gen) < py1[:, None]).long()
    xs = torch.zeros(B, N, D)
    p0 = torch.where(ys.bool(), px1[:, 0, 1][:, None], px1[:, 0, 0][:, None])
    xs[:, :, 0] = (torch.rand(B, N, generator=gen) < p0).float()
    ar = torch.arange(B)
    for e in range(D - 1):
        pv = xs[ar, :, par[:, e]].long()  # (B, N) parent values
        p = cond[ar[:, None], e, pv, ys]
        xs[ar, :, chi[:, e]] = (torch.rand(B, N, generator=gen) < p).float()
    return xs, ys


def release_mst(x, y, eps, gen, oracle=False):
    if oracle:  # NOT DP: tree from clean data; budget halves on tables only
        py1, px1 = label_tables(x, y, eps / 2, gen)
        par, chi = prim(dependence(x), eps, gen, private=False)
        cond = pair_tables(x, y, par, chi, eps / 2, gen)
    else:
        py1, px1 = label_tables(x, y, eps / 3, gen)
        par, chi = prim(dependence(x), eps / 3 / (D - 1), gen, private=True)
        cond = pair_tables(x, y, par, chi, eps / 3, gen)
    return py1, px1, cond, par, chi


# ------------------------------------------------------------------ train (aware, mst_dp)
def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)
    (HERE / "results").mkdir(exist_ok=True)
    model = VecPFN()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    curve, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        task = draw_tasks(args.batch, gen)
        n = int(torch.randint(16, 129, (1,), generator=gen))
        k = float([1, 2, 4][int(torch.randint(0, 3, (1,), generator=gen))])
        x, y = draw_rows(task, n + args.n_qry, gen)
        xd, yd, xq, yq = x[:, :n], y[:, :n], x[:, n:], y[:, n:]
        eps = torch.exp(torch.empty(args.batch).uniform_(math.log(0.5), math.log(32.0), generator=gen))
        py1, px1, cond, par, chi = release_mst(xd, yd, eps, gen)
        xc, yc = tree_synth(py1, px1, cond, par, chi, int(k * n), gen)
        tok = tokens(eps, torch.full((args.batch,), k), torch.full((args.batch,), float(n)))
        loss = -model(xc, yc, xq, tok).gather(2, yq.unsqueeze(-1)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps:
            curve.append(dict(step=step, train_loss=float(loss), seconds=time.time() - t0))
            print("aware_mst step %5d loss %.4f (%.0fs)" % (step, float(loss), time.time() - t0), flush=True)
    torch.save(model.state_dict(), HERE / "results" / "pfn_aware_mst.pt")
    write_json(EXP, "training_aware_mst.json", dict(config=vars(args), curve=curve), __file__, args.seed)


# ------------------------------------------------------------------ evaluate (paired with E4b)
def edge_corr(task, par, chi, gen):
    """Mean population |corr| of chosen edges, from 4,000 fresh rows per task."""
    xb, _ = draw_rows(task, 4000, gen)
    xb = xb - xb.mean(1, keepdim=True)
    cov = torch.einsum("bni,bnj->bij", xb, xb) / xb.shape[1]
    sd = cov.diagonal(dim1=1, dim2=2).clamp_min(1e-9).sqrt()
    corr = (cov / (sd[:, :, None] * sd[:, None, :])).abs()
    ar = torch.arange(task[0].shape[0])[:, None]
    chosen = corr[ar, par, chi].mean(1)
    off = ~torch.eye(D, dtype=torch.bool)
    allp = corr[:, off].mean(1)
    best_par, best_chi = prim(corr, torch.zeros(corr.shape[0]), gen, private=False)
    best = corr[ar, best_par, best_chi].mean(1)
    return chosen, allp, best


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(args.threads)
    nets = {}
    for name, path in (("naive", EXPS / "E4b-lite-noise-aware-synthetic/results/pfn_naive.pt"),
                       ("aware_nb", EXPS / "E4b-lite-noise-aware-synthetic/results/pfn_aware.pt"),
                       ("aware_mst", HERE / "results/pfn_aware_mst.pt")):
        net = VecPFN()
        net.load_state_dict(torch.load(path, weights_only=True))
        nets[name] = net.eval()
    rows = []
    for n in args.ns:
        for eps_v in args.epss:
            for k in args.ks:
                seed = args.seed * 100003 + n * 101 + int(eps_v * 10) * 7 + k  # E4b's per-cell seed
                gen = torch.Generator().manual_seed(seed)
                torch.manual_seed(seed)
                gen2 = torch.Generator().manual_seed(seed + 7777)  # releases other than nb
                acc, sel = {}, {}
                for _ in range(args.tasks // args.chunk):
                    task = draw_tasks(args.chunk, gen)
                    x, y = draw_rows(task, n + args.queries, gen)
                    xd, yd, xq, yq = x[:, :n], y[:, :n], x[:, n:], y[:, n:]
                    eps = torch.full((args.chunk,), float(eps_v))
                    # E4b's release, reproduced exactly (global RNG for Laplace, gen for rows)
                    py1, px1 = noisy_tables(xd, yd, eps, gen)
                    xs_nb, ys_nb = synth(py1, px1, k * n, gen)
                    state = torch.get_rng_state()
                    spy, spx, spair, sedges = e4c_mst.release(xd, yd, eps)  # E4c star (uses global RNG)
                    xs_st, ys_st = e4c_mst.synth_tree(spy, spx, spair, sedges, k * n, gen2)
                    torch.set_rng_state(state)  # keep E4b's Laplace stream aligned for the next chunk
                    rel = {"mst_dp": release_mst(xd, yd, eps, gen2), "tree_oracle": release_mst(xd, yd, eps, gen2, oracle=True)}
                    syn = {"nb": (xs_nb, ys_nb), "star": (xs_st, ys_st)}
                    for name, (a, b_, c_, par, chi) in rel.items():
                        syn[name] = tree_synth(a, b_, c_, par, chi, k * n, gen2)
                    tok = tokens(eps, torch.full((args.chunk,), float(k)), torch.full((args.chunk,), float(n)))
                    zero = torch.zeros(args.chunk, 3)
                    cpy, cpx = nb_clean_tables(xd, yd)
                    preds = {"oracle_pfn": nets["naive"](xd, yd, xq, zero)[..., 1].exp(),
                             "nb_clean": nb_predict(cpy, cpx, xq),
                             "aware_nb": nets["aware_nb"](*syn["nb"], xq, tok)[..., 1].exp(),
                             "aware_mst": nets["aware_mst"](*syn["mst_dp"], xq, tok)[..., 1].exp()}
                    for name, (xs, ys) in syn.items():
                        preds["naive_" + name] = nets["naive"](xs, ys, xq, zero)[..., 1].exp()
                    yn = yq.numpy()
                    for key, p in preds.items():
                        for mk, mv in metrics(p.double().numpy(), yn).items():
                            acc.setdefault(key, {}).setdefault(mk, []).append(mv)
                    star_par = torch.tensor([u for u, v in sedges])[None].expand(args.chunk, -1)
                    star_chi = torch.tensor([v for u, v in sedges])[None].expand(args.chunk, -1)
                    gq = torch.Generator().manual_seed(seed + 999)
                    for name, (pp, cc) in (("star", (star_par, star_chi)), ("mst_dp", rel["mst_dp"][3:]),
                                           ("tree_oracle", rel["tree_oracle"][3:])):
                        chosen, allp, best = edge_corr(task, pp, cc, torch.Generator().manual_seed(seed + 999))
                        sel.setdefault(name, []).append(float(chosen.mean()))
                    sel.setdefault("all_pairs", []).append(float(allp.mean()))
                    sel.setdefault("best_tree_population", []).append(float(best.mean()))
                res = {key: {mk: float(np.mean(v)) for mk, v in d.items()} for key, d in acc.items()}
                selq = {key: float(np.mean(v)) for key, v in sel.items()}
                rows.append(dict(n=n, eps=eps_v, k=k, results=res, edge_abs_corr=selq))
                print("n=%4d eps=%-4g k=%d | edge |corr|: star %.3f mst_dp %.3f oracle %.3f (all %.3f, best %.3f)" % (
                    n, eps_v, k, selq["star"], selq["mst_dp"], selq["tree_oracle"], selq["all_pairs"], selq["best_tree_population"]))
                for key, v in res.items():
                    print("   %-17s acc %.3f  NLL %.4f  ECE %.4f  overconf %+.4f" % (key, v["acc"], v["nll"], v["ece"], v["overconf"]), flush=True)
    write_json(EXP, "evaluation.json", dict(config=vars(args), rows=rows), __file__, args.seed)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--n-qry", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--threads", type=int, default=3)
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

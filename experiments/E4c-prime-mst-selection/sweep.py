"""E4c-prime n*eps sweep: at what n*eps does releasing an MST tree of feature pairs start to pay?

python experiments/E4c-prime-mst-selection/sweep.py

Grid n in {32, 128, 512, 2048, 8192} x eps in {1, 4, 16, 64}; 200 tasks x 400 clean test labels
per cell.  Releases (see e4cp_mst.py): nb (eps on the (x_j, y) tables), mst_dp (eps/3 tables,
eps/3 private per-dataset Prim selection, eps/3 pair tables), tree_oracle (NOT DP: clean tree,
eps/2 + eps/2).  For each release:

  plugin      the classifier the release itself encodes: naive Bayes for nb, tree-augmented
              naive Bayes (TAN) for the tree releases, computed exactly from the noisy tables.
              A naive learner fed many synthetic rows converges to it (the k -> inf limit of
              naive use), so it scales to any n.
  naive_pfn   E4b's clean-trained PFN on k = 1 synthetic rows, only where the context is in its
              training range (n <= 512).

References without privacy noise: nb_clean (naive Bayes) and tan_clean (TAN on the clean tree);
they bound what each structure can do on this prior.  Selection quality: mean population |corr|
of the private tree's edges vs random pairs and the oracle tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e4cp_mst import (D, EXPS, VecPFN, dependence, draw_rows, draw_tasks, edge_corr, label_tables, metrics,  # noqa: E402
                      nb_predict, pair_tables, prim, release_mst, tree_synth)
from common.provenance import write_json  # noqa: E402
from e4b_syn import synth  # noqa: E402

EXP = "E4c-prime-mst-selection"


def tan_predict(py1, px1, cond, par, chi, xq):
    """P(y=1 | xq) under P(y) P(x_0 | y) prod_e P(x_chi | x_par, y). xq (B, m, D)."""
    B, m, _ = xq.shape
    logit = torch.log(py1)[:, None] - torch.log1p(-py1)[:, None]
    x0 = xq[:, :, 0]
    for c, sgn in ((1, 1.0), (0, -1.0)):
        p0 = px1[:, 0, c][:, None]
        logit = logit + sgn * (x0 * torch.log(p0) + (1 - x0) * torch.log1p(-p0))
    for e in range(D - 1):
        xp = torch.gather(xq, 2, par[:, e][:, None, None].expand(B, m, 1)).squeeze(-1).long()
        xc = torch.gather(xq, 2, chi[:, e][:, None, None].expand(B, m, 1)).squeeze(-1)
        for c, sgn in ((1, 1.0), (0, -1.0)):
            p = cond[torch.arange(B)[:, None], e, xp, c]  # (B, m)
            logit = logit + sgn * (xc * torch.log(p) + (1 - xc) * torch.log1p(-p))
    return torch.sigmoid(logit)


@torch.no_grad()
def main(ns=(32, 128, 512, 2048, 8192), epss=(1.0, 4.0, 16.0, 64.0), tasks=200, chunk=25, queries=400, pfn_max_n=512, seed=2):
    torch.set_num_threads(4)
    naive = VecPFN()
    naive.load_state_dict(torch.load(EXPS / "E4b-lite-noise-aware-synthetic/results/pfn_naive.pt", weights_only=True))
    naive.eval()
    rows = []
    for n in ns:
        for eps_v in epss:
            cs = seed * 1000003 + n * 131 + int(eps_v * 10)
            gen = torch.Generator().manual_seed(cs)
            acc, sel = {}, {}
            ch = chunk if n <= 2048 else 10
            for _ in range(tasks // ch):
                task = draw_tasks(ch, gen)
                x, y = draw_rows(task, n + queries, gen)
                xd, yd, xq, yq = x[:, :n], y[:, :n], x[:, n:], y[:, n:]
                eps = torch.full((ch,), float(eps_v))
                big = torch.full((ch,), 1e12)
                py_nb, px_nb = label_tables(xd, yd, eps, gen)
                rel = {"mst_dp": release_mst(xd, yd, eps, gen), "tree_oracle": release_mst(xd, yd, eps, gen, oracle=True)}
                clean_tree = release_mst(xd, yd, big, gen, oracle=True)  # noise-free TAN on the clean tree
                cpy, cpx = label_tables(xd, yd, big, gen)
                preds = {"nb_plugin": nb_predict(py_nb, px_nb, xq), "nb_clean": nb_predict(cpy, cpx, xq),
                         "tan_clean": tan_predict(*clean_tree, xq)}
                for name, r in rel.items():
                    preds[name + "_plugin"] = tan_predict(*r, xq)
                if n <= pfn_max_n:
                    zero = torch.zeros(ch, 3)
                    preds["oracle_pfn"] = naive(xd, yd, xq, zero)[..., 1].exp()
                    xs, ys = synth(py_nb, px_nb, n, gen)
                    preds["nb_pfn"] = naive(xs, ys, xq, zero)[..., 1].exp()
                    for name, r in rel.items():
                        xs, ys = tree_synth(*r, n, gen)
                        preds[name + "_pfn"] = naive(xs, ys, xq, zero)[..., 1].exp()
                yn = yq.numpy()
                for key, p in preds.items():
                    for mk, mv in metrics(p.double().clamp(1e-6, 1 - 1e-6).numpy(), yn).items():
                        acc.setdefault(key, {}).setdefault(mk, []).append(mv)
                for name in ("mst_dp", "tree_oracle"):
                    chosen, allp, best = edge_corr(task, rel[name][3], rel[name][4], torch.Generator().manual_seed(cs + 1))
                    sel.setdefault(name, []).append(float(chosen.mean()))
                sel.setdefault("all_pairs", []).append(float(allp.mean()))
                sel.setdefault("best_tree_population", []).append(float(best.mean()))
            res = {k: {mk: float(np.mean(v)) for mk, v in d.items()} for k, d in acc.items()}
            sq = {k: float(np.mean(v)) for k, v in sel.items()}
            rows.append(dict(n=n, eps=eps_v, n_eps=n * eps_v, results=res, edge_abs_corr=sq))
            line = "n=%5d eps=%-4g n*eps=%7g | edge|corr| dp %.3f oracle %.3f all %.3f | plugin NLL nb %.4f mst %.4f orc %.4f (clean nb %.4f tan %.4f) | overconf nb %+.3f mst %+.3f orc %+.3f" % (
                n, eps_v, n * eps_v, sq["mst_dp"], sq["tree_oracle"], sq["all_pairs"], res["nb_plugin"]["nll"],
                res["mst_dp_plugin"]["nll"], res["tree_oracle_plugin"]["nll"], res["nb_clean"]["nll"], res["tan_clean"]["nll"],
                res["nb_plugin"]["overconf"], res["mst_dp_plugin"]["overconf"], res["tree_oracle_plugin"]["overconf"])
            if "nb_pfn" in res:
                line += " | PFN overconf nb %+.3f mst %+.3f orc %+.3f" % (
                    res["nb_pfn"]["overconf"], res["mst_dp_pfn"]["overconf"], res["tree_oracle_pfn"]["overconf"])
            print(line, flush=True)
    write_json(EXP, "sweep.json", dict(config=dict(ns=list(ns), epss=list(epss), tasks=tasks, queries=queries,
                                                   pfn_max_n=pfn_max_n, seed=seed), rows=rows), __file__, seed)


if __name__ == "__main__":
    main()

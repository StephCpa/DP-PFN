"""E4d-lite: a noise-aware PFN for large n that reads the DP release itself (noisy tables as tokens).

  python experiments/E4d-lite-large-n-summary-pfn/e4d_sum.py train --steps 6000
  python experiments/E4d-lite-large-n-summary-pfn/e4d_sum.py eval

Why tables, not synthetic rows.  The releases of E4b/E4c' are noisy count tables; synthetic rows
are a post-processing of them.  A row-reading PFN costs O(rows^2) and, at large n, a few hundred
synthetic rows carry less than the tables themselves.  Conditioning on the tables has the same
target as conditioning on any post-processing that keeps them (Thm. D1: the private-data PPD),
and its cost does not depend on n.  This is the "statistic tokens" reader of proposal Sec. 5.

Releases (same mechanisms and noise as e4cp_mst.py; raw noisy counts, unclipped, are the release):
  nb      : Laplace(2d/eps) on the 16 (x_j, y) tables.
  mst_dp  : eps/3 (x_j, y) tables, eps/3 private per-dataset Prim tree (exponential mechanism,
            sensitivity 4), eps/3 on the 15 (x_parent, x_child, y) tables.
Network (one model for both releases):
  memory tokens  meta (log eps, log n, release type) + 16 label-table tokens (4 counts / n, scale / n,
                 + feature-id embedding) + 15 edge tokens (8 counts / n, scale / n, + parent-id and
                 child-id embeddings; masked for nb); 2 self-attention layers.
  query          one token per query feature (value + feature-id embedding) + a CLS token;
                 3 decoder layers (self-attention among the query's tokens, cross-attention to memory).
Trained on clean query labels; n ~ log-U[32, 8192], eps ~ log-U[0.5, 64], release type 50/50.
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
EXPS = HERE.parent
for sub in ("", "E4b-lite-noise-aware-synthetic", "E4-lite-noise-aware-rr", "E4c-prime-mst-selection"):
    sys.path.insert(0, str(EXPS / sub))
from common.provenance import write_json  # noqa: E402
from e4_rr import metrics  # noqa: E402
from e4b_syn import D, VecPFN, draw_rows, draw_tasks, nb_predict, synth, tokens  # noqa: E402
from e4cp_mst import dependence, lap, prim, tree_synth  # noqa: E402
from sweep import tan_predict  # noqa: E402

EXP = "E4d-lite-large-n-summary-pfn"


# ------------------------------------------------------------------ releases (raw noisy tables)
def raw_label_tables(x, y, eps_part, gen, noise=True):
    """Noisy counts T[b, j, a, c] = #{x_j = a, y = c} + Laplace(2d/eps_part). Returns T, scale (B,)."""
    yc = torch.stack([(y == 0).float(), (y == 1).float()], -1)
    B = x.shape[0]
    T = torch.zeros(B, D, 2, 2)
    T[:, :, 1, :] = torch.einsum("bnd,bnc->bdc", x, yc)
    T[:, :, 0, :] = torch.einsum("bnd,bnc->bdc", 1 - x, yc)
    scale = 2 * D / eps_part
    if noise:
        T = T + lap(T.shape, 1.0, gen) * scale.view(-1, 1, 1, 1)
    return T, scale


def raw_pair_tables(x, y, par, chi, eps_part, gen, noise=True):
    """Noisy counts T[b, e, a, b', c] = #{x_par = a, x_chi = b', y = c} + Laplace(2(d-1)/eps_part)."""
    B, n, _ = x.shape
    xp = torch.gather(x, 2, par[:, None, :].expand(B, n, D - 1))
    xc = torch.gather(x, 2, chi[:, None, :].expand(B, n, D - 1))
    yc = torch.stack([(y == 0).float(), (y == 1).float()], -1)
    T = torch.zeros(B, D - 1, 2, 2, 2)
    for a in (0, 1):
        for b in (0, 1):
            T[:, :, a, b, :] = torch.einsum("bne,bnc->bec", ((xp == a) & (xc == b)).float(), yc)
    scale = 2 * (D - 1) / eps_part
    if noise:
        T = T + lap(T.shape, 1.0, gen) * scale.view(-1, 1, 1, 1, 1)
    return T, scale


def plugin_label(T):
    Tc = T.clamp_min(0) + 0.5
    col = Tc.sum(2)  # (B, D, 2)
    return (col[..., 1] / col.sum(-1)).mean(-1), Tc[:, :, 1, :] / col


def plugin_pair(T):
    Tc = T.clamp_min(0) + 0.5
    return Tc[:, :, :, 1, :] / Tc.sum(3)  # P(child = 1 | parent = a, y = c), (B, E, 2, 2)


def release(x, y, eps, gen, tree, noise=True):
    """Returns a dict with raw tables, scales, tree (or None) and the plug-in model parameters."""
    if not tree:
        TL, sL = raw_label_tables(x, y, eps, gen, noise)
        return dict(TL=TL, sL=sL, TP=None, sP=None, par=None, chi=None)
    TL, sL = raw_label_tables(x, y, eps / 3, gen, noise)
    if noise:
        par, chi = prim(dependence(x), eps / 3 / (D - 1), gen, private=True)
    else:
        par, chi = prim(dependence(x), eps, gen, private=False)
    TP, sP = raw_pair_tables(x, y, par, chi, eps / 3, gen, noise)
    return dict(TL=TL, sL=sL, TP=TP, sP=sP, par=par, chi=chi)


def memory_inputs(rel, n, eps):
    B = rel["TL"].shape[0]
    nf = float(n)
    # asinh tames heavy Laplace tails at small n*eps; noise scales enter on a log scale
    lab = torch.cat([torch.asinh(4 * rel["TL"].reshape(B, D, 4) / nf),
                     (torch.log(rel["sL"] / nf) / 5).view(-1, 1, 1).expand(B, D, 1)], -1)
    if rel["TP"] is None:
        pair = torch.zeros(B, D - 1, 9)
        pmask = torch.ones(B, D - 1, dtype=torch.bool)
        par = chi = torch.zeros(B, D - 1, dtype=torch.long)
        flag = torch.zeros(B)
    else:
        pair = torch.cat([torch.asinh(4 * rel["TP"].reshape(B, D - 1, 8) / nf),
                          (torch.log(rel["sP"] / nf) / 5).view(-1, 1, 1).expand(B, D - 1, 1)], -1)
        pmask = torch.zeros(B, D - 1, dtype=torch.bool)
        par, chi, flag = rel["par"], rel["chi"], torch.ones(B)
    meta = torch.stack([torch.log(eps) / 4, torch.full((B,), math.log(nf) / 9), flag], -1)
    return lab, pair, pmask, par, chi, meta


# ------------------------------------------------------------------ network
def mlp(i, d):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, d))


class SumPFN(nn.Module):
    def __init__(self, d=128, heads=4, enc_layers=2, dec_layers=3, ff=256):
        super().__init__()
        self.lab_enc, self.pair_enc, self.meta_enc, self.x_enc = mlp(5, d), mlp(9, d), mlp(3, d), mlp(1, d)
        self.fid, self.fpar, self.fchi = nn.Embedding(D, d), nn.Embedding(D, d), nn.Embedding(D, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, 0.0, batch_first=True, norm_first=True),
                                         enc_layers, enable_nested_tensor=False)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, ff, 0.0, batch_first=True, norm_first=True),
                                         dec_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, lab, pair, pmask, par, chi, meta, xq):
        B, m = xq.shape[:2]
        ids = torch.arange(D)
        mem = torch.cat([self.meta_enc(meta)[:, None], self.lab_enc(lab) + self.fid(ids)[None],
                         self.pair_enc(pair) + self.fpar(par) + self.fchi(chi)], 1)
        mask = torch.cat([torch.zeros(B, 1 + D, dtype=torch.bool), pmask], 1)
        mem = self.enc(mem, src_key_padding_mask=mask)
        q = self.x_enc((2 * xq - 1).reshape(B * m, D, 1)) + self.fid(ids)[None]
        q = torch.cat([self.cls.expand(B * m, 1, -1), q], 1)
        h = self.dec(q, mem.repeat_interleave(m, 0), memory_key_padding_mask=mask.repeat_interleave(m, 0))
        return torch.log_softmax(self.head(h[:, 0]), -1).view(B, m, 2)


# ------------------------------------------------------------------ train
def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)
    (HERE / "results").mkdir(exist_ok=True)
    model = SumPFN()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    curve, t0, skipped = [], time.time(), 0
    for step in range(1, args.steps + 1):
        task = draw_tasks(args.batch, gen)
        n = int(math.exp(float(torch.empty(1).uniform_(math.log(32), math.log(8192), generator=gen))))
        x, y = draw_rows(task, n + args.n_qry, gen)
        xd, yd, xq, yq = x[:, :n], y[:, :n], x[:, n:], y[:, n:]
        eps = torch.exp(torch.empty(args.batch).uniform_(math.log(0.5), math.log(64.0), generator=gen))
        tree = bool(torch.rand(1, generator=gen) < 0.5)
        rel = release(xd, yd, eps, gen, tree)
        loss = -model(*memory_inputs(rel, n, eps), xq).gather(2, yq.unsqueeze(-1)).mean()
        if not torch.isfinite(loss):
            skipped += 1
            print("non-finite loss at step %d (n=%d, tree=%s); step skipped" % (step, n, tree), flush=True)
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps:
            curve.append(dict(step=step, train_loss=float(loss), seconds=time.time() - t0))
            print("sumpfn step %5d loss %.4f (%.0fs)" % (step, float(loss), time.time() - t0), flush=True)
    torch.save(model.state_dict(), HERE / "results" / "sumpfn.pt")
    write_json(EXP, "training.json", dict(config=vars(args), curve=curve, skipped_steps=skipped), __file__, args.seed)


# ------------------------------------------------------------------ evaluate
@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(args.threads)
    sp = SumPFN()
    sp.load_state_dict(torch.load(HERE / "results" / "sumpfn.pt", weights_only=True))
    sp.eval()
    rows_nets = {}
    for name, path in (("clean_pfn", EXPS / "E4b-lite-noise-aware-synthetic/results/pfn_naive.pt"),
                       ("row_aware_nb", EXPS / "E4b-lite-noise-aware-synthetic/results/pfn_aware.pt"),
                       ("row_aware_tree", EXPS / "E4c-prime-mst-selection/results/pfn_aware_mst.pt")):
        net = VecPFN()
        net.load_state_dict(torch.load(path, weights_only=True))
        rows_nets[name] = net.eval()
    out = []
    for n in args.ns:
        for eps_v in args.epss:
            cs = args.seed * 1000003 + n * 131 + int(eps_v * 10)
            gen = torch.Generator().manual_seed(cs)
            acc = {}
            ch = args.chunk if n <= 2048 else 10
            for _ in range(args.tasks // ch):
                task = draw_tasks(ch, gen)
                x, y = draw_rows(task, n + args.queries, gen)
                xd, yd, xq, yq = x[:, :n], y[:, :n], x[:, n:], y[:, n:]
                eps = torch.full((ch,), float(eps_v))
                rnb, rtr = release(xd, yd, eps, gen, False), release(xd, yd, eps, gen, True)
                cnb, ctr = release(xd, yd, eps, gen, False, noise=False), release(xd, yd, eps, gen, True, noise=False)
                preds = {}
                for name, rel in (("nb", rnb), ("tree", rtr)):
                    preds["sum_aware_" + name] = torch.cat(
                        [sp(*memory_inputs(rel, n, eps), xq[:, i:i + 100])[..., 1].exp() for i in range(0, args.queries, 100)], 1)
                py, px = plugin_label(rnb["TL"])
                preds["plugin_nb"] = nb_predict(py, px, xq)
                pyt, pxt = plugin_label(rtr["TL"])
                preds["plugin_tree"] = tan_predict(pyt, pxt, plugin_pair(rtr["TP"]), rtr["par"], rtr["chi"], xq)
                cpy, cpx = plugin_label(cnb["TL"])
                preds["clean_nb"] = nb_predict(cpy, cpx, xq)
                tpy, tpx = plugin_label(ctr["TL"])
                preds["clean_tan"] = tan_predict(tpy, tpx, plugin_pair(ctr["TP"]), ctr["par"], ctr["chi"], xq)
                if n <= args.row_max_n:
                    zero = torch.zeros(ch, 3)
                    preds["clean_pfn"] = rows_nets["clean_pfn"](xd, yd, xq, zero)[..., 1].exp()
                if n <= args.row_aware_max_n:
                    tok = tokens(eps, torch.ones(ch), torch.full((ch,), float(n)))
                    xs, ys = synth(py, px, n, gen)
                    preds["row_aware_nb"] = rows_nets["row_aware_nb"](xs, ys, xq, tok)[..., 1].exp()
                    xs, ys = tree_synth(pyt, pxt, plugin_pair(rtr["TP"]), rtr["par"], rtr["chi"], n, gen)
                    preds["row_aware_tree"] = rows_nets["row_aware_tree"](xs, ys, xq, tok)[..., 1].exp()
                yn = yq.numpy()
                for key, p in preds.items():
                    for mk, mv in metrics(p.double().clamp(1e-6, 1 - 1e-6).numpy(), yn).items():
                        acc.setdefault(key, {}).setdefault(mk, []).append(mv)
            res = {k: {mk: float(np.mean(v)) for mk, v in d.items()} for k, d in acc.items()}
            out.append(dict(n=n, eps=eps_v, n_eps=n * eps_v, results=res))
            line = "n=%5d eps=%-4g n*eps=%7g | NLL sum_nb %.4f sum_tree %.4f plug_nb %.4f plug_tree %.4f clean_nb %.4f clean_tan %.4f | oc sum_nb %+.3f sum_tree %+.3f plug_nb %+.3f plug_tree %+.3f" % (
                n, eps_v, n * eps_v, res["sum_aware_nb"]["nll"], res["sum_aware_tree"]["nll"], res["plugin_nb"]["nll"],
                res["plugin_tree"]["nll"], res["clean_nb"]["nll"], res["clean_tan"]["nll"], res["sum_aware_nb"]["overconf"],
                res["sum_aware_tree"]["overconf"], res["plugin_nb"]["overconf"], res["plugin_tree"]["overconf"])
            if "row_aware_nb" in res:
                line += " | row-aware NLL nb %.4f tree %.4f" % (res["row_aware_nb"]["nll"], res["row_aware_tree"]["nll"])
            if "clean_pfn" in res:
                line += " | clean_pfn %.4f" % res["clean_pfn"]["nll"]
            print(line, flush=True)
    write_json(EXP, "evaluation.json", dict(config=vars(args), rows=out), __file__, args.seed)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--n-qry", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--ns", type=int, nargs="*", default=[32, 128, 512, 2048, 8192])
    e.add_argument("--epss", type=float, nargs="*", default=[1.0, 4.0, 16.0, 64.0])
    e.add_argument("--tasks", type=int, default=200)
    e.add_argument("--chunk", type=int, default=25)
    e.add_argument("--queries", type=int, default=400)
    e.add_argument("--row-max-n", type=int, default=512)
    e.add_argument("--row-aware-max-n", type=int, default=128)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()

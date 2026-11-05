"""D2-sparse: naive vs noise-aware use of a sparse high-dimensional DP histogram.

python experiments/D2-sparse-synthetic/sparse_hist.py

A binary label and D = 8 binary covariates give 512 cells. The task is a sparse
logistic model: the intercept and four of the eight weights are drawn, the rest
are zero. The release is Laplace noise on the full histogram (replace-one
sensitivity 2), then 5n synthetic rows drawn from it.

This is the setting section 4.4 of the review asked for. D2-check used 20 cells
and the over-confidence died by n = 200. Here most cells are empty at the n we
test, so the Laplace noise is not averaged away.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.provenance import write_json  # noqa: E402

EXP = "D2-sparse-synthetic"
SEED = 0
D = 8
CELLS = 1 << (D + 1)  # 512
ACTIVE = 4


def bits(idx):
    """idx (m,) -> array (m, D+1) with column 0 the label and 1..D the covariates."""
    shifts = np.arange(D + 1)
    return ((idx[:, None] >> shifts) & 1).astype(float)


IDX = np.arange(CELLS)
FEAT = bits(IDX)                       # (512, 9): y, x1..x8
COV = FEAT[:, 1:]                      # (512, 8)
LAB = FEAT[:, 0]                       # (512,)


def task_prior_grid():
    """Grid over (bias, weight), shared by the four active coefficients.

    The four active features get the same weight. That keeps the grid at
    61 x 41 = 2501 points, small enough to score exactly.
    """
    bias = np.linspace(-3, 3, 61)
    weight = np.linspace(-2.5, 2.5, 41)
    B, W = np.meshgrid(bias, weight, indexing="ij")
    B, W = B.ravel(), W.ravel()
    log_prior = -0.5 * (B / 1.0) ** 2 - 0.5 * (W / 1.0) ** 2
    log_prior -= logsumexp(log_prior)
    active = COV[:, :ACTIVE].sum(1)                       # (512,)
    logit = B[None, :] + W[None, :] * active[:, None]     # (512, G)
    p_y1 = expit(logit)
    # cell probability: covariates are uniform over {0,1}^D, so each covariate
    # pattern has mass 2^-D, split by the label probability.
    p_cell = np.where(LAB[:, None] == 1, p_y1, 1 - p_y1) / (1 << D)
    return B, W, log_prior, p_cell


B, W, LOG_PRIOR, P_CELL = task_prior_grid()


def draw_task(rng):
    b = rng.normal(0, 1)
    w = rng.normal(0, 1)
    return b, w


def cell_prob(b, w):
    active = COV[:, :ACTIVE].sum(1)
    p_y1 = expit(b + w * active)
    return np.where(LAB == 1, p_y1, 1 - p_y1) / (1 << D)


def draw_cells(rng, p, n):
    return rng.choice(CELLS, size=n, p=p)


def histogram(cells):
    return np.bincount(cells, minlength=CELLS).astype(float)


def sample_from_histogram(rng, h, nsyn):
    h = np.clip(h, 0, None)
    if h.sum() <= 0:
        p = np.full(CELLS, 1 / CELLS)
    else:
        p = h / h.sum()
    return rng.choice(CELLS, size=nsyn, p=p)


def posterior(loglik):
    lp = LOG_PRIOR + loglik
    return np.exp(lp - logsumexp(lp))


def loglik_rows(cells):
    """Multinomial-style product of cell probabilities for exact rows."""
    return np.log(np.clip(P_CELL[cells], 1e-300, None)).sum(0)


def loglik_noisy_hist(h, n, eps):
    """Gaussian likelihood of the released histogram, per grid point."""
    var_lap = 2 * (2 / eps) ** 2
    mean = n * P_CELL
    var = n * P_CELL * (1 - P_CELL) + var_lap
    return (-0.5 * ((h[:, None] - mean) ** 2 / var + np.log(var))).sum(0)


def predictive(post, q_cells):
    """P(y=1 | covariates of the query cells), averaged over the posterior.

    Bit 0 of a cell index is the label, so the y=0 partner of a cell is the
    index with bit 0 cleared and the y=1 partner has bit 0 set.
    """
    p_cell = P_CELL @ post
    x0 = q_cells & ~1
    p0 = p_cell[x0]
    p1 = p_cell[x0 + 1]
    return p1 / np.clip(p0 + p1, 1e-300, None)


def metrics(p1, y):
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    pred = (p1 > 0.5).astype(int)
    conf = np.maximum(p1, 1 - p1)
    correct = (pred == y).astype(float)
    nll = -np.mean(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    ece = 0.0
    edges = np.linspace(0.5, 1.0, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi + (1e-12 if hi == 1.0 else 0))
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return dict(acc=float(correct.mean()), nll=float(nll), ece=float(ece),
                overconf=float(conf.mean() - correct.mean()))


def run(rng, n, eps, tasks, nq):
    acc = {}
    occupied = []
    for _ in range(tasks):
        b, w = draw_task(rng)
        p = cell_prob(b, w)
        cells = draw_cells(rng, p, n)
        occupied.append(np.count_nonzero(histogram(cells)) / CELLS)
        h = histogram(cells) + rng.laplace(0, 2 / eps, CELLS)
        syn = sample_from_histogram(rng, h, 5 * n)
        q = draw_cells(rng, p, nq)
        yq = (q & 1).astype(int)
        preds = {
            "oracle": predictive(posterior(loglik_rows(cells)), q),
            "naive": predictive(posterior(loglik_rows(syn)), q),
            "noise_aware": predictive(posterior(loglik_noisy_hist(h, n, eps)), q),
        }
        for k, p1 in preds.items():
            for mk, mv in metrics(p1, yq).items():
                acc.setdefault(k, {}).setdefault(mk, []).append(mv)
    out = {k: {mk: float(np.mean(v)) for mk, v in d.items()} for k, d in acc.items()}
    out["occupied_cell_fraction"] = float(np.mean(occupied))
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    plan = [(50, 80), (200, 60), (1000, 40)]
    rows = []
    for n, tasks in plan:
        for eps in (0.5, 1.0, 2.0):
            res = run(rng, n, eps, tasks, nq=400)
            rows.append(dict(n=n, eps=eps, cells=CELLS, nsyn_factor=5, results=res))
            print("n=%4d eps=%.1f occupied=%.3f" % (n, eps, res["occupied_cell_fraction"]), flush=True)
            for k in ("oracle", "naive", "noise_aware"):
                v = res[k]
                print("   %-12s acc %.3f  NLL %.4f  ECE %.4f  overconf %+.4f" % (
                    k, v["acc"], v["nll"], v["ece"], v["overconf"]), flush=True)
    path = write_json(EXP, "results.json", dict(rows=rows, cells=CELLS, active_features=ACTIVE), __file__, SEED)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()

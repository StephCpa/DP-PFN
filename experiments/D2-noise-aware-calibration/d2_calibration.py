"""D2 / H4 / Gate 2 (closed form): naive vs noise-aware use of a DP release.

python experiments/D2-noise-aware-calibration/d2_calibration.py

Task prior: w ~ N(0, 2^2), b ~ N(0, 1), x ~ U[-1, 1], y ~ Bernoulli(sigmoid(w x + b)).
All predictors are exact grid-Bayes posteriors, so the comparison isolates the
*target* (Theorem D1, Proposition D2), not network approximation error.  A
noise-aware PFN would approximate the noise-aware column; naive TabPFN on the
release approximates the naive column.

Releases at privacy level eps:
  RR      label randomized response, flip prob rho = 1 / (1 + e^eps)   (label-DP, local)
  SYN     DP synthetic data: Laplace(2/eps) noise on the 2-way (x-bin, y)
          histogram (10 public bins; replace-one sensitivity 2), clipped,
          normalised, then N_syn = 5 n synthetic rows sampled from it
          (row-DP, central).  The x of a synthetic row is uniform within its bin.
Predictors:
  oracle      posterior from the clean data (non-private reference)
  naive       treats the release as clean data
  noise_aware RR: likelihood (1-rho) s + rho (1-s).
              SYN: likelihood of the noisy histogram (Gaussian approximation to
              multinomial + Laplace: var = n p (1-p) + 2 (2/eps)^2 per cell),
              i.e. it conditions on the released statistic, which the synthetic
              rows are a post-processing of.
Metrics on clean test labels: accuracy, NLL, ECE (10 bins), and over-confidence
= mean(confidence) - accuracy (> 0 over-confident, < 0 under-confident).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.provenance import write_json  # noqa: E402

EXP = "D2-noise-aware-calibration"
SEED = 0
W, B = np.meshgrid(np.linspace(-10, 10, 161), np.linspace(-4.5, 4.5, 61), indexing="ij")
W, B = W.ravel(), B.ravel()
LOG_PRIOR = -0.5 * (W / 2.0) ** 2 - 0.5 * B**2
LOG_PRIOR -= logsumexp(LOG_PRIOR)
BINS = np.linspace(-1, 1, 11)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])


def sig(x):
    return expit(np.outer(x, W) + B)  # (n, G)


def post_from_loglik(ll):
    lp = LOG_PRIOR + ll
    return np.exp(lp - logsumexp(lp))


def ppd(post, xq):
    return sig(xq) @ post  # P(y=1 | xq)


def loglik_clean(x, y):
    s = np.clip(sig(x), 1e-12, 1 - 1e-12)
    return (y[:, None] * np.log(s) + (1 - y[:, None]) * np.log1p(-s)).sum(0)


def loglik_rr(x, ytil, rho):
    s = sig(x)
    p1 = (1 - rho) * s + rho * (1 - s)
    return (ytil[:, None] * np.log(p1) + (1 - ytil[:, None]) * np.log1p(-p1)).sum(0)


def bin_probs():
    """P(bin j, y=1 | w, b) and P(bin j, y=0) with x uniform; (10, G) each (midpoint rule, 5 pts/bin)."""
    p1 = np.zeros((10, W.size))
    for j in range(10):
        xs = np.linspace(BINS[j], BINS[j + 1], 7)[1:-1]
        p1[j] = sig(xs).mean(0) / 10
    return p1, 0.1 - p1


P1_BIN, P0_BIN = bin_probs()


def loglik_noisy_hist(h1, h0, n, eps):
    """Gaussian approx. log-likelihood of the noisy histogram (h1, h0 each (10,))."""
    var_lap = 2 * (2 / eps) ** 2
    ll = 0.0
    for h, p in ((h1, P1_BIN), (h0, P0_BIN)):
        mean = n * p
        var = n * p * (1 - p) + var_lap
        ll = ll - 0.5 * ((h[:, None] - mean) ** 2 / var + np.log(var)).sum(0)
    return ll


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
    return dict(acc=correct.mean(), nll=nll, ece=ece, overconf=conf.mean() - correct.mean())


def run(rng, n, eps, tasks=200, nq=400):
    rho = 1 / (1 + np.exp(eps))
    acc = {}
    for _ in range(tasks):
        w, b = rng.normal(0, 2), rng.normal(0, 1)
        x = rng.uniform(-1, 1, n)
        y = (rng.random(n) < expit(w * x + b)).astype(int)
        xq = rng.uniform(-1, 1, nq)
        yq = (rng.random(nq) < expit(w * xq + b)).astype(int)
        preds = {"oracle": ppd(post_from_loglik(loglik_clean(x, y)), xq)}
        # --- RR release
        ytil = np.where(rng.random(n) < rho, 1 - y, y)
        preds["RR_naive"] = ppd(post_from_loglik(loglik_clean(x, ytil)), xq)
        preds["RR_noise_aware"] = ppd(post_from_loglik(loglik_rr(x, ytil, rho)), xq)
        # --- DP synthetic data release
        j = np.clip(np.digitize(x, BINS) - 1, 0, 9)
        h1 = np.bincount(j[y == 1], minlength=10) + rng.laplace(0, 2 / eps, 10)
        h0 = np.bincount(j[y == 0], minlength=10) + rng.laplace(0, 2 / eps, 10)
        probs = np.clip(np.concatenate([h1, h0]), 0, None)
        probs = probs / probs.sum() if probs.sum() > 0 else np.full(20, 1 / 20)
        nsyn = 5 * n
        cell = rng.choice(20, size=nsyn, p=probs)
        ysyn = (cell < 10).astype(int)
        xsyn = BINS[cell % 10] + rng.random(nsyn) * 0.2
        preds["SYN_naive"] = ppd(post_from_loglik(loglik_clean(xsyn, ysyn)), xq)
        preds["SYN_noise_aware"] = ppd(post_from_loglik(loglik_noisy_hist(h1, h0, n, eps)), xq)
        for k, p in preds.items():
            for mk, mv in metrics(p, yq).items():
                acc.setdefault(k, {}).setdefault(mk, []).append(mv)
    return {k: {mk: float(np.mean(v)) for mk, v in d.items()} for k, d in acc.items()}


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for n in (50, 200, 1000):
        for eps in (0.5, 1.0, 2.0):
            res = run(rng, n, eps)
            rows.append(dict(n=n, eps=eps, results=res))
            print("n=%4d eps=%.1f" % (n, eps))
            for k, v in res.items():
                print("   %-16s acc %.3f  NLL %.4f  ECE %.4f  overconf %+.4f" % (k, v["acc"], v["nll"], v["ece"], v["overconf"]), flush=True)
    path = write_json(EXP, "results.json", dict(rows=rows), __file__, SEED)
    print("wrote", path)


if __name__ == "__main__":
    main()

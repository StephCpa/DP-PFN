"""E3-lite / T6 / T4: Theorem C2 vs simulation, and Vote vs LogPool.

python experiments/E3-lite-vote-sample-complexity/vote_c2.py

Task prior (binary, closed-form Bayes):
    mu_0, mu_1 ~ N(0, tau^2 I_d);  y ~ Bernoulli(1/2);  x | y ~ N(mu_y, I_d)
Teachers on disjoint shards of size m (T = n / m teachers):
    bayes      exact posterior predictive under the prior (an idealised PFN; calibrated)
    bayes_hot  the same, logits x3 (over-confident; for T4)
    bayes_flat Bayes predictive under a much weaker prior (tau_teacher = 10): higher
               individual error at small m, but less shrinkage bias
    logreg     sklearn LogisticRegression, default C=1 (a PATE-style trained teacher)
Aggregators, epsilon-DP per answered query (replace-one row neighbours):
    vote       report-noisy-max, Laplace(2/eps) on each count (Mechanism C1)
    vote_margin Laplace(2/eps) on the single count difference (binary only; also eps-DP)
    logpool    exponential mechanism on summed floored log-probs (Mechanism C3),
               sensitivity Delta = log(1 + K(1-eta)/eta)
C2 is pointwise: e(m; x*) is the probability a teacher disagrees with the Bayes label
of the *true* task at x*.  As T -> infinity at fixed m, Vote returns that label iff
gamma_m(x*) > 0, so the fraction of queries with gamma_m(x*) <= 0 is Vote's error floor
("infinite-T floor"); it is set by teacher *bias*, not by the teacher's mean error.  We compare C2's lower bound on P(Vote returns that label)
with the simulated frequency, and C2's predicted best shard size with the observed one.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.provenance import write_json  # noqa: E402

EXP = "E3-lite-vote-sample-complexity"
warnings.filterwarnings("ignore")


def draw_task(rng, d, tau):
    return rng.normal(0, tau, (2, d))


def draw_data(rng, mu, n):
    y = rng.integers(0, 2, n)
    x = mu[y] + rng.normal(size=(n, mu.shape[1]))
    return x, y


def bayes_label(mu, xq):
    """Bayes-optimal label of the TRUE task (equal class priors, identity covariance)."""
    d0 = ((xq - mu[0]) ** 2).sum(1)
    d1 = ((xq - mu[1]) ** 2).sum(1)
    return (d1 < d0).astype(int)


def bayes_teacher(xs, ys, xq, tau, temp=1.0):
    """Posterior predictive P(y=1 | x*, shard) under the Gaussian prior. xs: (T, m, d)."""
    out = []
    for c in (0, 1):
        mask = (ys == c)[..., None]
        nc = mask.sum(1)  # (T, 1)
        prec = nc + 1 / tau**2
        mean = (xs * mask).sum(1) / prec  # (T, d)
        var = 1 + 1 / prec  # predictive variance per dim, (T, 1)
        d = xq.shape[1]
        sq = ((xq[None, :, :] - mean[:, None, :]) ** 2).sum(-1)  # (T, Q)
        loglik = -0.5 * sq / var - 0.5 * d * np.log(var)
        logprior = np.log((nc + 1) / (ys.shape[1] + 2))  # Beta(1,1) class prior
        out.append(loglik + logprior)
    logit = (out[1] - out[0]) * temp
    return 1 / (1 + np.exp(-logit))  # (T, Q)


def logreg_teacher(xs, ys, xq):
    T = xs.shape[0]
    p = np.empty((T, xq.shape[0]))
    for t in range(T):
        if ys[t].min() == ys[t].max():
            p[t] = 0.99 if ys[t][0] == 1 else 0.01
        else:
            p[t] = LogisticRegression(C=1.0, max_iter=200).fit(xs[t], ys[t]).predict_proba(xq)[:, 1]
    return p


def all_teachers(xs, ys, xq, tau):
    return {"bayes": bayes_teacher(xs, ys, xq, tau),
            "bayes_hot": bayes_teacher(xs, ys, xq, tau, temp=3.0),
            "bayes_flat": bayes_teacher(xs, ys, xq, 10.0),
            "logreg": logreg_teacher(xs, ys, xq)}


def aggregate(p, eps, rng, reps, eta_floor=0.1):
    """p: (T, Q) teacher P(y=1). Returns dict of (reps, Q) predicted labels."""
    T, Q = p.shape
    votes1 = (p > 0.5).sum(0)
    votes0 = T - votes1
    b = 2 / eps
    n1 = votes1[None] + rng.laplace(0, b, (reps, Q))
    n0 = votes0[None] + rng.laplace(0, b, (reps, Q))
    out = {"vote": (n1 > n0).astype(int)}
    out["vote_margin"] = ((votes1 - votes0)[None] + rng.laplace(0, b, (reps, Q)) > 0).astype(int)
    for eta in (0.02, eta_floor, 0.3):
        K = 2
        pf = (1 - eta) * p + eta / K
        S = (np.log(pf) - np.log(1 - pf)).sum(0)  # S_1 - S_0; prior term is constant (1/2)
        delta = math.log1p(K * (1 - eta) / eta)
        prob1 = 1 / (1 + np.exp(-eps * S / (2 * delta)))
        out[f"logpool_eta{eta}"] = (rng.random((reps, Q)) < prob1[None]).astype(int)
    return out


def c2_bound(gamma, T, eps):
    """C2 lower bound on P(Vote returns the reference label), pointwise; 0 if gamma <= 0."""
    g = np.clip(gamma, 1e-12, None)
    fail = np.exp(-T * g**2 / 8) + np.exp(-eps * T * g / 8)
    return np.where(gamma > 0, np.clip(1 - fail, 0, 1), 0.0)


def n_star(gamma, eps, beta=0.05):
    if gamma <= 0:
        return math.inf
    return 8 * math.log(2 / beta) * max(1 / gamma**2, 1 / (eps * gamma))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--ms", type=int, nargs="*", default=[4, 8, 16, 32, 64, 125, 250, 500])
    ap.add_argument("--epss", type=float, nargs="*", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--tasks", type=int, default=24)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--fresh", type=int, default=200, help="fresh shards for population e(m; x*)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    teachers = ("bayes", "bayes_hot", "bayes_flat", "logreg")
    acc = {}  # (teacher, agg, eps, m) -> list of accuracy vs true labels
    agree = {}  # agreement with the Bayes label of the true task
    c2 = {}  # (teacher, eps, m) -> list of C2 bound values
    emean = {}  # (teacher, m) -> mean pointwise teacher error
    gmed = {}
    nonpriv = {}
    floor = {}
    bayes_opt = []
    for task in range(args.tasks):
        mu = draw_task(rng, args.d, args.tau)
        x, y = draw_data(rng, mu, args.n)
        xq, yq = draw_data(rng, mu, args.queries)
        ref = bayes_label(mu, xq)
        bayes_opt.append((ref == yq).mean())
        perm = rng.permutation(args.n)
        for m in args.ms:
            T = args.n // m
            idx = perm[: T * m].reshape(T, m)
            xs, ys = x[idx], y[idx]
            probs = all_teachers(xs, ys, xq, args.tau)
            # population pointwise error e(m; x*): fresh shards from the true task
            xf, yf = draw_data(rng, mu, args.fresh * m)
            pop = all_teachers(xf.reshape(args.fresh, m, -1), yf.reshape(args.fresh, m), xq, args.tau)
            for t in teachers:
                p = probs[t]
                e_pt = ((pop[t] > 0.5).astype(int) != ref[None]).mean(0)  # pointwise teacher error
                gamma = 1 - 2 * e_pt
                emean.setdefault((t, m), []).append(e_pt.mean())
                gmed.setdefault((t, m), []).append(np.median(gamma))
                floor.setdefault((t, m), []).append(float((gamma <= 0).mean()))
                nonpriv.setdefault((t, m), []).append((((p > 0.5).sum(0) > T / 2).astype(int) == yq).mean())
                for eps in args.epss:
                    c2.setdefault((t, eps, m), []).append(c2_bound(gamma, T, eps).mean())
                    for agg, lab in aggregate(p, eps, rng, args.reps).items():
                        acc.setdefault((t, agg, eps, m), []).append((lab == yq[None]).mean())
                        agree.setdefault((t, agg, eps, m), []).append((lab == ref[None]).mean())
        print("task %d done" % task, flush=True)

    def mean(v):
        return float(np.mean(v))

    table = []
    for (t, agg, eps, m), v in sorted(acc.items()):
        table.append(dict(teacher=t, agg=agg, eps=eps, m=m, T=args.n // m, acc=mean(v),
                          agree_bayes_label=mean(agree[(t, agg, eps, m)]),
                          c2_bound=mean(c2[(t, eps, m)]) if agg == "vote" else None))
    teacher_curves = [dict(teacher=t, m=m, mean_error=mean(emean[(t, m)]), median_gamma=mean(gmed[(t, m)]),
                           infinite_T_floor=mean(floor[(t, m)]), nonprivate_majority_acc=mean(nonpriv[(t, m)]))
                      for (t, m) in sorted(emean)]
    # best m per (teacher, agg, eps): observed vs C2-predicted
    best = []
    for t in teachers:
        for eps in args.epss:
            obs = {m: mean(agree[(t, "vote", eps, m)]) for m in args.ms}
            pred = {m: mean(c2[(t, eps, m)]) for m in args.ms}
            nstar = {m: m * n_star(mean(gmed[(t, m)]), eps) for m in args.ms}
            best.append(dict(teacher=t, eps=eps, observed_best_m=max(obs, key=obs.get),
                             c2_best_m=max(pred, key=pred.get), n_star_best_m=min(nstar, key=nstar.get),
                             n_star_min=min(nstar.values()), observed=obs, c2=pred))
    # vote vs logpool summary
    duel = []
    for t in teachers:
        for eps in args.epss:
            bv = max(mean(acc[(t, "vote", eps, m)]) for m in args.ms)
            bvm = max(mean(acc[(t, "vote_margin", eps, m)]) for m in args.ms)
            row = dict(teacher=t, eps=eps, best_vote=bv, best_vote_margin=bvm)
            for eta in (0.02, 0.1, 0.3):
                row[f"best_logpool_eta{eta}"] = max(mean(acc[(t, f"logpool_eta{eta}", eps, m)]) for m in args.ms)
            duel.append(row)
    res = dict(config=vars(args), bayes_optimal_acc=mean(bayes_opt), table=table,
               teacher_curves=teacher_curves, best_m=best, vote_vs_logpool=duel)
    path = write_json(EXP, f"results{args.tag}.json", res, __file__, args.seed)

    print("Bayes-optimal accuracy (true task known): %.3f" % mean(bayes_opt))
    print("\nTeacher error e(m) (mean pointwise disagreement with true-task Bayes label)")
    for r in teacher_curves:
        print("  %-10s m=%4d e=%.3f median gamma=%.3f inf-T floor=%.3f nonprivate-majority acc=%.3f" % (
            r["teacher"], r["m"], r["mean_error"], r["median_gamma"], r["infinite_T_floor"], r["nonprivate_majority_acc"]))
    print("\nBest shard size: observed (Vote agreement) vs C2 bound vs n* formula")
    for r in best:
        print("  %-10s eps=%.2f observed m=%d  C2-bound m=%d  n*-argmin m=%d (n*=%.0f)" % (
            r["teacher"], r["eps"], r["observed_best_m"], r["c2_best_m"], r["n_star_best_m"], r["n_star_min"]))
    print("\nVote vs LogPool (best over m, accuracy vs true labels)")
    for r in duel:
        print("  %-10s eps=%.2f vote=%.3f vote_margin=%.3f logpool(.02/.1/.3)=%.3f/%.3f/%.3f" % (
            r["teacher"], r["eps"], r["best_vote"], r["best_vote_margin"], r["best_logpool_eta0.02"],
            r["best_logpool_eta0.1"], r["best_logpool_eta0.3"]))
    print("wrote", path)


if __name__ == "__main__":
    main()

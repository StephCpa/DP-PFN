"""X0: numerical audit of every constant and closed-form claim in the proposal.

Run:  python experiments/X0-theory-audit/audit_constants.py
Output: artifacts/X0-theory-audit/audit.json and a printed report.

Sections mirror the proposal: A (exact PPD), C (wrappers), T2, G2, G3.
Nothing here trains a network; all numbers are exact or Monte Carlo with a
fixed seed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, betaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.provenance import write_json  # noqa: E402

EXP = "X0-theory-audit"
SEED = 0


# ---------------------------------------------------------------- Section A
def label_floor_B(eta: float, K: int) -> float:
    """Lemma A4: B_y(p_eta) = log(1 + K(1-eta)/eta) (tight for point-mass p)."""
    return math.log1p(K * (1 - eta) / eta)


def amplify_wor(eps: float, q: float) -> float:
    """Theorem A7 (Balle, Barthe & Gaboardi 2018), substitution relation."""
    return math.log1p(q * math.expm1(eps))


def table_A(K: int = 2, q: float = 0.1) -> list[dict]:
    rows = []
    for eta in (0.2, 0.5, 0.8):
        B = label_floor_B(eta, K)
        eps = 2 * B
        rows.append(dict(eta=eta, flip_rate=eta * (K - 1) / K, B_y=B, eps=eps,
                         eps_subsampled=amplify_wor(eps, q)))
    return rows


def exact_beta_bernoulli(eta: float, ns, a: float = 1.0, b: float = 1.0, grid: int = 40001):
    """Exact label-DP privacy loss of three releases under a floored Beta-Bernoulli prior.

    Model: phi ~ Beta(a, b); y ~ Bernoulli((1-eta) phi + eta/2).  No features, so
    D is summarised by its count k of ones and label neighbours are k vs k+1.

    Releases compared (all single draws):
      posterior : phi ~ pi(.|D)                      (Thm A2: <= 2 B_y)
      noisy PPD : y ~ P(.|D)                         (Thm A2: <= 2 B_y)
      clean     : y = 1[phi > 1/2], phi ~ pi(.|D)    (Thm A5: <= 2 B_y)
    """
    phi = (np.arange(grid) + 0.5) / grid
    log_prior = (a - 1) * np.log(phi) + (b - 1) * np.log1p(-phi)
    p1 = (1 - eta) * phi + eta / 2
    l1, l0 = np.log(p1), np.log1p(-p1)
    above = phi > 0.5
    out = []
    for n in ns:
        k = np.arange(n + 1)
        # log unnormalised posterior for every k: shape (n+1, grid)
        lp = log_prior[None, :] + k[:, None] * l1[None, :] + (n - k)[:, None] * l0[None, :]
        lZ = logsumexp(lp, axis=1)
        post = lp - lZ[:, None]  # log posterior mass per grid cell (up to const)
        eps_post = np.max(np.abs(post[1:] - post[:-1]))
        # noisy PPD
        ppd1 = np.exp(logsumexp(post + l1[None, :], axis=1))
        ppd = np.stack([1 - ppd1, ppd1], 1)
        eps_ppd = np.max(np.abs(np.log(ppd[1:]) - np.log(ppd[:-1])))
        # clean target Q(1|D) = pi(phi > 1/2 | D), in log space
        lq1 = logsumexp(post[:, above], axis=1)
        lq0 = logsumexp(post[:, ~above], axis=1)
        lq = np.stack([lq0, lq1], 1)
        eps_clean = np.max(np.abs(lq[1:] - lq[:-1]))
        out.append(dict(n=int(n), eps_posterior=float(eps_post), eps_noisy_ppd=float(eps_ppd),
                        eps_clean_target=float(eps_clean)))
    return out


# ---------------------------------------------------------------- Section C
def laplace_diff_tail(t: float, b: float) -> float:
    """P(L1 - L2 >= t) for iid Laplace(b): (1/2 + t/(4b)) exp(-t/b)."""
    return (0.5 + t / (4 * b)) * math.exp(-t / b)


def check_C2_tail() -> dict:
    """Thm C2 needs P(noise overturns margin T*gamma/2) <= exp(-eps*T*gamma/8)."""
    worst = -np.inf
    for eps in (0.1, 0.5, 1, 2, 4, 8):
        b = 2 / eps
        for margin in np.linspace(0, 200, 2001):
            exact = laplace_diff_tail(margin, b)
            bound = math.exp(-eps * margin / 4)  # margin = T*gamma/2 -> exp(-eps*T*gamma/8)
            worst = max(worst, exact / bound)
    return dict(max_ratio_exact_over_bound=worst, holds=bool(worst <= 1 + 1e-12))


def logpool_width() -> list[dict]:
    """C3 says floored log-probs lie in an interval of width <= log(K/eta).

    Exact width of log((1-eta)q + eta/K) over q in [0,1] is log(1 + K(1-eta)/eta).
    """
    rows = []
    for K in (2, 5, 10):
        for eta in (0.01, 0.05, 0.2, 0.5):
            exact = math.log1p(K * (1 - eta) / eta)
            rows.append(dict(K=K, eta=eta, exact_width=exact, claimed_width=math.log(K / eta),
                             claim_is_upper_bound=bool(exact <= math.log(K / eta) + 1e-12)))
    return rows


# ---------------------------------------------------------------- T2 / Sec 5
def t2_flip_rates() -> dict:
    eps = 2 * label_floor_B(0.5, 2)
    return dict(eps=eps, central_one_draw_flip=0.25, local_rr_flip=1 / (math.exp(eps) + 1))


# ---------------------------------------------------------------- G2
def gaussian_rdp_to_dp(L: int, sigma: float, delta: float) -> float:
    """L adaptive Gaussian releases with l2 sensitivity sqrt(2): RDP L*alpha/sigma^2.

    Standard conversion eps = min_alpha L alpha/sigma^2 + log(1/delta)/(alpha-1)
    (Mironov 2017).  Closed form at the optimum alpha.
    """
    alphas = np.linspace(1.01, 500, 200000)
    return float(np.min(L * alphas / sigma**2 + math.log(1 / delta) / (alphas - 1)))


def g2_sigma_table(delta: float = 1e-5) -> list[dict]:
    rows = []
    for L in (1, 2, 3):
        for target in (1.0, 2.0, 4.0, 8.0):
            lo, hi = 0.05, 200.0
            for _ in range(80):
                mid = (lo + hi) / 2
                if gaussian_rdp_to_dp(L, mid, delta) > target:
                    lo = mid
                else:
                    hi = mid
            rows.append(dict(L=L, target_eps=target, delta=delta, sigma=hi))
    return rows


# ---------------------------------------------------------------- G3
def rr_rho(eps: float) -> float:
    return 1 / (1 + math.exp(eps))


def g3_numbers(N: int = 1024, eps: float = 2.0, real_edges: float = 12700) -> dict:
    rho = rr_rho(eps)
    pairs = N * (N - 1) / 2
    fake = rho * (pairs - real_edges)
    removed = rho * real_edges
    return dict(N=N, eps=eps, rho=rho, node_pairs=pairs, fake_edges_added=fake,
                real_edges_removed=removed, edges_after_rr=real_edges - removed + fake)


def g3_thresholds(a: float = 20, b: float = 4, K: int = 2, eps: float = 2.0) -> dict:
    """Three versions of the Prop. G3 detectability threshold on N.

    heuristic  : proposal's N < (1-2rho)^2 (a-b)^2 / (K^2 rho)   (drops lower-order terms)
    KS_full    : Kesten-Stigum on the RR graph without dropping terms,
                 (a'-b')^2 > K (a' + (K-1) b'),  a' = N rho + (1-2rho) a.
    dense_BBP  : the RR graph is dense (p' = Theta(1)), so the edge variance is
                 p'(1-p'), not p'.  SNR = (a'-b')^2 / (K (a'+(K-1)b') (1 - pbar')).
    """
    rho = rr_rho(eps)
    c = 1 - 2 * rho
    heuristic = c**2 * (a - b) ** 2 / (K**2 * rho)

    def snr(N, dense):
        ap = N * rho + c * a
        bp = N * rho + c * b
        denom = K * (ap + (K - 1) * bp)
        if dense:
            pbar = (ap + (K - 1) * bp) / (K * N)
            denom *= 1 - pbar
        return (ap - bp) ** 2 / denom

    def solve(dense):
        lo, hi = 1.0, 1e6
        for _ in range(200):
            mid = (lo + hi) / 2
            if snr(mid, dense) > 1:
                lo = mid
            else:
                hi = mid
        return lo

    return dict(a=a, b=b, K=K, eps=eps, rho=rho, heuristic_N=heuristic,
                ks_full_N=solve(False), dense_bbp_N=solve(True))


def nodepfn_prior_edges(N: int = 1024, draws: int = 4000, seed: int = SEED) -> dict:
    """Expected edge count of NodePFN's released graph prior.

    Read from the official repository (nodepfn/priors/network_utils.py and
    nodepfn/scripts/model_configs.py): graph_type in {sbm, random} (uniform);
    ER edge_prob ~ U(0.01, 0.05); SBM p_in ~ U(0.01, 0.1), homophily_rate h ~ U(0.1, 0.9),
    p_out = p_in (1-h), off-diagonal block probs = Power(5) * p_out, diagonal =
    p_out + Power(2) * (p_in - p_out), then symmetrised; classes ~ U{2..20}
    (pretrain.py).  Classes are taken as balanced here (rank-based multiclass
    labels), which is an approximation.
    """
    rng = np.random.default_rng(seed)
    pairs = N * (N - 1) / 2
    edges, dens = [], []
    for _ in range(draws):
        if rng.random() < 0.5:
            p = rng.uniform(0.01, 0.05)
            e = p * pairs
            dens.append(p)
        else:
            K = rng.integers(2, 21)
            p_in = rng.uniform(0.01, 0.1)
            h = rng.uniform(0.1, 0.9)
            p_out = p_in * (1 - h)
            P = rng.power(5, (K, K)) * p_out
            np.fill_diagonal(P, p_out + rng.power(2, K) * (p_in - p_out))
            P = (P + P.T) / 2
            size = N / K
            within = K * size * (size - 1) / 2
            e = np.trace(P) * size * (size - 1) / 2 + (P.sum() - np.trace(P)) / 2 * size * size
            dens.append(e / pairs)
            del within
        edges.append(e)
    edges = np.asarray(edges)
    dens = np.asarray(dens)
    rho = rr_rho(2.0)
    return dict(N=N, mean_edges=float(edges.mean()), median_edges=float(np.median(edges)),
                p95_edges=float(np.quantile(edges, 0.95)), max_density=float(dens.max()),
                rr_eps2_min_density=float(rho + (1 - 2 * rho) * dens.min()),
                note="RR at eps=2 gives density >= rho = %.3f, above every prior draw" % rho)


def main() -> None:
    res = {}
    res["table_A"] = table_A()
    res["exact_beta_bernoulli_eta0.5"] = exact_beta_bernoulli(0.5, [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    res["exact_beta_bernoulli_eta0.2"] = exact_beta_bernoulli(0.2, [1, 5, 20, 100, 500])
    res["two_B_y"] = {"eta0.5": 2 * label_floor_B(0.5, 2), "eta0.2": 2 * label_floor_B(0.2, 2)}
    res["C2_laplace_tail"] = check_C2_tail()
    res["C3_logpool_width"] = logpool_width()
    res["T2_flip_rates"] = t2_flip_rates()
    res["G2_sigma_for_target_eps"] = g2_sigma_table()
    res["G3_density"] = g3_numbers()
    res["G3_thresholds_example"] = g3_thresholds()
    res["G3_thresholds_eps"] = [g3_thresholds(eps=e) for e in (1.0, 2.0, 4.0, 8.0)]
    res["nodepfn_prior_edges"] = nodepfn_prior_edges()
    path = write_json(EXP, "audit.json", res, __file__, SEED)

    # ---------------- printed report
    print("Table A (K=2, m/n=0.1)")
    for r in res["table_A"]:
        print("  eta=%.1f flip=%.0f%% B_y=%.3f eps=%.3f eps_sub=%.3f" % (
            r["eta"], 100 * r["flip_rate"], r["B_y"], r["eps"], r["eps_subsampled"]))
    for key in ("exact_beta_bernoulli_eta0.5", "exact_beta_bernoulli_eta0.2"):
        eta = key.rsplit("eta", 1)[1]
        print("Exact label-DP loss, floored Beta-Bernoulli, eta=%s (2B_y=%.3f)" % (eta, res["two_B_y"]["eta" + eta]))
        for r in res[key]:
            print("  n=%4d posterior=%.3f noisyPPD=%.4f clean=%.3f" % (
                r["n"], r["eps_posterior"], r["eps_noisy_ppd"], r["eps_clean_target"]))
    print("C2 Laplace tail bound:", res["C2_laplace_tail"])
    print("T2:", res["T2_flip_rates"])
    print("G3 density:", res["G3_density"])
    print("G3 thresholds:", res["G3_thresholds_example"])
    for r in res["G3_thresholds_eps"]:
        print("  eps=%.0f heuristic=%.0f ks=%.0f dense=%.0f" % (r["eps"], r["heuristic_N"], r["ks_full_N"], r["dense_bbp_N"]))
    print("NodePFN prior edges:", res["nodepfn_prior_edges"])
    print("G2 sigma table (delta=1e-5):")
    for r in res["G2_sigma_for_target_eps"]:
        print("  L=%d eps=%.0f sigma=%.2f" % (r["L"], r["target_eps"], r["sigma"]))
    print("wrote", path)


if __name__ == "__main__":
    main()

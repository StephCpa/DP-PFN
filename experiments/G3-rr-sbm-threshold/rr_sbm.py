"""G3 / T7: where does edge randomized response destroy SBM community signal?

python experiments/G3-rr-sbm-threshold/rr_sbm.py

Balanced 2-community SBM with fixed expected degrees a = N p_in, b = N p_out
(the proposal's example: a = 20, b = 4).  Every node pair is flipped with
rho = 1 / (1 + e^eps) (edge-LDP randomized response).  We run spectral recovery
(top eigenvector of the centred adjacency) and record
  overlap   = |2 * accuracy - 1| of sign rounding (0 = chance, 1 = perfect)
  corr2     = squared correlation of the eigenvector with the planted labels.
Spiked-Wigner theory (dense regime) predicts corr2 ~ max(0, 1 - 1/SNR) with
SNR = (a'-b')^2 / (K (a'+(K-1)b') (1 - pbar')), which vanishes at N = dense_bbp_N.
Compared with the proposal's heuristic threshold and the sparse Kesten-Stigum form.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "X0-theory-audit"))
from common.provenance import write_json  # noqa: E402
from audit_constants import g3_thresholds, rr_rho  # noqa: E402

EXP = "G3-rr-sbm-threshold"
SEED = 0


def rr_sbm(rng, N, a, b, eps):
    labels = np.repeat([1, -1], N // 2)
    rng.shuffle(labels)
    same = labels[:, None] == labels[None, :]
    P = np.where(same, a / N, b / N)
    rho = rr_rho(eps)
    Pp = rho + (1 - 2 * rho) * P  # edge probability after RR (exact)
    U = rng.random((N, N))
    A = np.triu(U < Pp, 1)
    A = (A | A.T).astype(np.float32)
    return A, labels, Pp


def spectral(A):
    N = A.shape[0]
    pbar = A.sum() / (N * (N - 1))
    M = A - pbar * (np.ones_like(A) - np.eye(N, dtype=A.dtype))
    _, v = eigsh(M.astype(np.float64), k=1, which="LA")
    return v[:, 0]


def snr_dense(N, a, b, eps, K=2):
    rho = rr_rho(eps)
    c = 1 - 2 * rho
    ap, bp = N * rho + c * a, N * rho + c * b
    pbar = (ap + (K - 1) * bp) / (K * N)
    return (ap - bp) ** 2 / (K * (ap + (K - 1) * bp) * (1 - pbar))


def main() -> None:
    rng = np.random.default_rng(SEED)
    a, b = 20.0, 4.0
    plan = {1.0: [20, 30, 40, 50, 60, 80, 100, 150],
            2.0: [100, 150, 200, 250, 300, 350, 400, 500, 700, 1000],
            4.0: [1000, 1500, 2000, 2500, 3000, 3500, 4500]}
    reps = {1.0: 60, 2.0: 30, 4.0: 8}
    out = []
    for eps, Ns in plan.items():
        th = g3_thresholds(a=a, b=b, K=2, eps=eps)
        print("eps=%.0f thresholds: heuristic %.0f  KS(sparse) %.0f  dense-BBP %.0f" % (
            eps, th["heuristic_N"], th["ks_full_N"], th["dense_bbp_N"]), flush=True)
        for N in Ns:
            ov, c2 = [], []
            for _ in range(reps[eps]):
                A, lab, _ = rr_sbm(rng, N, a, b, eps)
                v = spectral(A)
                acc = np.mean(np.sign(v) == lab)
                ov.append(abs(2 * acc - 1))
                c2.append(float((v @ lab) ** 2 / (np.sum(v**2) * N)))
            s = snr_dense(N, a, b, eps)
            row = dict(eps=eps, N=N, rho=rr_rho(eps), reps=reps[eps], overlap_mean=float(np.mean(ov)),
                       overlap_se=float(np.std(ov) / math.sqrt(len(ov))), corr2_mean=float(np.mean(c2)),
                       snr_dense=s, corr2_theory=max(0.0, 1 - 1 / s), **{k: th[k] for k in ("heuristic_N", "ks_full_N", "dense_bbp_N")})
            out.append(row)
            print("  N=%5d overlap %.3f +- %.3f  corr2 %.3f  theory corr2 %.3f  SNR %.2f" % (
                N, row["overlap_mean"], row["overlap_se"], row["corr2_mean"], row["corr2_theory"], s), flush=True)
    path = write_json(EXP, "results.json", dict(a=a, b=b, rows=out), __file__, SEED)
    print("wrote", path)


if __name__ == "__main__":
    main()

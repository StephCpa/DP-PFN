"""Render the review figures from artifacts/*.json (static PNG, light surface).

python experiments/common/make_figures.py

Palette: reference categorical slots 1-3 (blue, orange, aqua), which validate
all-pairs in light mode; grey for references/thresholds. Thin 2px lines, >= 8px
markers, recessive grid, one y-axis per panel, direct labels + legend.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 8,
})


def load(path):
    return json.loads((ART / path).read_text())


def fig_e1():
    rows = load("E1-lite-amortization-gap/evaluation.json")["rows"]
    etas = sorted({r["eta"] for r in rows})
    n_show = 16
    fig, axes = plt.subplots(1, len(etas), figsize=(4.2 * len(etas), 3.4), sharey=False)
    for ax, eta in zip(axes, etas):
        rr = sorted([r for r in rows if r["eta"] == eta and r["n"] == n_show], key=lambda r: r["step"])
        steps = [r["step"] for r in rr]
        ax.plot(steps, [r["typical"]["s_hat"]["median"] for r in rr], "-o", color=S1, label="PFN leakage (median)")
        ax.plot(steps, [r["typical"]["eps_exact"]["median"] for r in rr], "--", color=REF, label="exact PPD leakage (median)")
        ax.plot(steps, [r["typical"]["Lambda"]["median"] for r in rr], "-s", color=S2, label="gap Λ(D) (median)")
        ax.set_xscale("log")
        ax.set_xlabel("training step")
        ax.set_title(f"η = {eta}, n = {n_show}, prior datasets")
        ax.set_ylabel("log-ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(ART / "E1-lite-amortization-gap" / "leakage_vs_training.png", dpi=200)
    plt.close(fig)


def fig_e3(tag="_n500", eps=0.5):
    res = load(f"E3-lite-vote-sample-complexity/results{tag}.json")
    tab = res["table"]
    curves = res["teacher_curves"]
    teachers = [("bayes", "Bayes teacher (ideal PFN)", S1), ("bayes_flat", "weak-prior Bayes", S3), ("logreg", "trained log. regression", S2)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for key, lab, col in teachers:
        rr = sorted([r for r in tab if r["teacher"] == key and r["agg"] == "vote" and r["eps"] == eps], key=lambda r: r["m"])
        axes[0].plot([r["m"] for r in rr], [r["acc"] for r in rr], "-o", color=col, label=lab)
        cc = sorted([c for c in curves if c["teacher"] == key], key=lambda c: c["m"])
        axes[1].plot([c["m"] for c in cc], [c["infinite_T_floor"] for c in cc], "-o", color=col, label=lab)
    axes[0].axhline(res["bayes_optimal_acc"], color=REF, ls="--", lw=1.5)
    axes[0].text(2, res["bayes_optimal_acc"] + 0.004, "Bayes-optimal (task known)", color=INK2, fontsize=8)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("shard size m")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title(f"DP-PFN-Vote, n = {res['config']['n']}, ε = {eps} per query")
    axes[1].set_ylabel("share of queries with γ_m(x*) ≤ 0")
    axes[1].set_title("Vote error floor as T → ∞")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(ART / "E3-lite-vote-sample-complexity" / f"vote_vs_m{tag}.png", dpi=200)
    plt.close(fig)


def fig_g3(eps=2.0):
    res = load("G3-rr-sbm-threshold/results.json")
    rr = sorted([r for r in res["rows"] if r["eps"] == eps], key=lambda r: r["N"])
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    Ns = [r["N"] for r in rr]
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT / "experiments" / "G3-rr-sbm-threshold"))
    sys.path.insert(0, str(ROOT / "experiments"))
    from rr_sbm import snr_dense  # noqa: E402

    ax.plot(Ns, [r["corr2_mean"] for r in rr], "-o", color=S1, label="simulated (spectral, mean)")
    grid = np.linspace(min(Ns), max(Ns), 400)
    ax.plot(grid, [max(0.0, 1 - 1 / snr_dense(N, res["a"], res["b"], eps)) for N in grid], "--", color=S2,
            label="dense spiked-matrix theory")
    top = max(r["corr2_mean"] for r in rr)
    marks = (("ks_full_N", "sparse KS", "right", 0.95), ("dense_bbp_N", "dense", "right", 0.80),
             ("heuristic_N", "proposal heuristic", "left", 0.65))
    for key, lab, ha, frac in marks:
        x = rr[0][key]
        ax.axvline(x, color=REF, lw=1, ls=":")
        ax.text(x + (6 if ha == "left" else -6), top * frac, f"{lab} {x:.0f}", color=INK2, fontsize=7, ha=ha, va="center")
    ax.set_xlabel("number of nodes N  (a = 20, b = 4 fixed)")
    ax.set_ylabel("eigenvector corr² with labels")
    ax.set_title(f"Edge RR at ε = {eps:g}: community signal vs N")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(ART / "G3-rr-sbm-threshold" / f"threshold_eps{eps:g}.png", dpi=200)
    plt.close(fig)


def main():
    for fn in (fig_e1, fig_e3, fig_g3):
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skip", fn.__name__, e)


if __name__ == "__main__":
    main()

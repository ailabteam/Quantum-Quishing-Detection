"""Regenerate the two conceptual/illustrative figures with HONEST labels.

The originals baked in claims that contradict the revised, honest narrative:
  - Robustness_Margin_Illustration.pdf labelled panel (b) "Quantum VQC Kernel:
    Wide Robustness Margin" and panel (a) with sigma=0.4. The margin effect is
    due to nonlinearity, not the quantum circuit, so we relabel accordingly.
  - Feature_Space_TSNE.pdf labelled panel (b) "Q-ResNet Quantum Feature Space";
    that 4-D space is the classical bottleneck output (pre-VQC), so we relabel it.

Usage:
  python -m revision.plot_concept_figures --tsne figures_for_paper/tsne_features.csv --out <paper_dir>
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300,
})


def plot_margin(out):
    x = np.linspace(-3, 3, 400)
    steep = np.tanh(4.0 * x)   # linear/brittle head: high gradient near boundary
    gentle = np.tanh(0.8 * x)  # nonlinear head: low gradient, wider margin
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)

    # (a) narrow margin
    ax[0].plot(x, steep, color="#c0392b", lw=2)
    ax[0].axhline(0, ls="--", color="0.5", lw=1)
    ax[0].axvspan(-0.25, 0.25, color="#c0392b", alpha=0.12)  # narrow margin band
    ax[0].plot(0.7, np.tanh(4*0.7), "ko", ms=5)              # clean sample (+)
    ax[0].plot(-0.2, np.tanh(4*-0.2), "x", color="#c0392b", ms=9, mew=2)  # after noise: flipped
    ax[0].annotate(r"$\delta x$ (noise)", xy=(-0.2, -0.1), xytext=(0.8, -0.55),
                   arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=8)
    ax[0].set_title("(a) Linear head: narrow margin", fontsize=9)
    ax[0].text(0.02, 0.92, r"high gradient $\Rightarrow$ flips", transform=ax[0].transAxes,
               fontsize=8, color="#c0392b")
    ax[0].set_xlabel(r"Compressed feature $x$")
    ax[0].set_ylabel(r"Prediction score $f(x)$")

    # (b) wide margin
    ax[1].plot(x, gentle, color="#1f7a3a", lw=2)
    ax[1].axhline(0, ls="--", color="0.5", lw=1)
    ax[1].axvspan(-1.1, 1.1, color="#1f7a3a", alpha=0.10)    # wide margin band
    ax[1].plot(1.6, np.tanh(0.8*1.6), "ko", ms=5)            # clean sample (+)
    ax[1].plot(1.1, np.tanh(0.8*1.1), "o", color="#1f7a3a", ms=6)  # after noise: still correct
    ax[1].annotate(r"$\delta x$ (noise)", xy=(1.1, np.tanh(0.8*1.1)), xytext=(-2.6, 0.55),
                   arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=8)
    ax[1].set_title("(b) Nonlinear head: wide margin", fontsize=9)
    ax[1].text(0.02, 0.92, r"low gradient $\Rightarrow$ absorbs $\delta x$", transform=ax[1].transAxes,
               fontsize=8, color="#1f7a3a")
    ax[1].set_xlabel(r"Compressed feature $x$")

    for a in ax:
        a.set_ylim(-1.15, 1.15)
        a.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def plot_tsne(csv, out):
    df = pd.read_csv(csv)
    lab = df.iloc[:, 2].values  # first 'Label' column
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for cls, color, name in [(0, "#34495e", "Benign"), (1, "#2ecc71", "Malicious")]:
        m = lab == cls
        ax[0].scatter(df["C_x"][m], df["C_y"][m], s=8, alpha=0.6, color=color, label=name)
        ax[1].scatter(df["Q_x"][m], df["Q_y"][m], s=8, alpha=0.6, color=color, label=name)
    ax[0].set_title("(a) Backbone feature space (512-D)", fontsize=9)
    ax[1].set_title("(b) Bottleneck feature space (4-D)", fontsize=9)
    for a in ax:
        a.set_xlabel("t-SNE 1"); a.set_ylabel("t-SNE 2"); a.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsne", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plot_margin(os.path.join(a.out, "Robustness_Margin_Illustration.pdf"))
    plot_tsne(a.tsne, os.path.join(a.out, "Feature_Space_TSNE.pdf"))


if __name__ == "__main__":
    main()

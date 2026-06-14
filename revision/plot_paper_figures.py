"""Regenerate the paper's result figures from the revised robustness CSVs.

Reads experiments_revision/robustness_raw_summary.csv (4 parameter-matched heads,
clean and noise-aware) and writes IEEE-style PDFs:
  Noise_Robustness.pdf   - 4 clean heads, accuracy vs Gaussian sigma (with std)
  Occlusion_Robustness.pdf - 4 clean heads, accuracy vs occlusion block size
  Defense_Noise.pdf      - clean vs noise-aware training (the defense)

Usage:
  python -m revision.plot_paper_figures --summary experiments_revision/robustness_raw_summary.csv --out <paper_dir>
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 300, "lines.linewidth": 1.4, "lines.markersize": 4,
})

# display label, csv model key, marker
HEADS = [
    ("Linear FC", "classic_fc", "o"),
    ("Bottleneck+tanh", "bottleneck_fc", "s"),
    ("MLP head", "mlp_head", "^"),
    ("VQC (Q-ResNet)", "qresnet_q4l2strong", "D"),
]


def _series(df, model, na, threat):
    d = df[(df.model == model) & (df.noise_aware == na) & (df.threat == threat)].sort_values("level")
    return d.level.values, d.acc_mean.values, d.acc_std.values


def plot_noise(df, out):
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for label, key, mk in HEADS:
        x, y, e = _series(df, key, False, "noise")
        if len(x):
            ax.errorbar(x, y, yerr=e, marker=mk, capsize=2, label=label)
    ax.set_xlabel(r"Gaussian noise std $\sigma$ (on $[0,1]$ image)")
    ax.set_ylabel("Accuracy (\\%)")
    ax.set_ylim(45, 101)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def plot_occlusion(df, out):
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for label, key, mk in HEADS:
        x, y, e = _series(df, key, False, "occlusion")
        if len(x):
            ax.errorbar(x, y, yerr=e, marker=mk, capsize=2, label=label)
    ax.set_xlabel("Occlusion block size (pixels)")
    ax.set_ylabel("Accuracy (\\%)")
    ax.set_ylim(90, 101)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def plot_defense(df, out):
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    pairs = [("classic_fc", "Linear FC", "o"), ("mlp_head", "MLP head", "^")]
    for key, label, mk in pairs:
        x, y, _ = _series(df, key, False, "noise")
        if len(x):
            ax.plot(x, y, marker=mk, ls="--", color="gray" if key == "classic_fc" else "black",
                    alpha=0.7, label=f"{label} (clean-trained)")
        xa, ya, _ = _series(df, key, True, "noise")
        if len(xa):
            ax.plot(xa, ya, marker=mk, ls="-", label=f"{label} (noise-aware)")
    ax.set_xlabel(r"Gaussian noise std $\sigma$ (on $[0,1]$ image)")
    ax.set_ylabel("Accuracy (\\%)")
    ax.set_ylim(45, 101)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True, help="paper directory to write the PDFs into")
    a = ap.parse_args()
    df = pd.read_csv(a.summary)
    df.noise_aware = df.noise_aware.astype(str).str.lower().map({"true": True, "false": False})
    plot_noise(df, os.path.join(a.out, "Noise_Robustness.pdf"))
    plot_occlusion(df, os.path.join(a.out, "Occlusion_Robustness.pdf"))
    plot_defense(df, os.path.join(a.out, "Defense_Noise.pdf"))


if __name__ == "__main__":
    main()

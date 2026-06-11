"""Core ablation experiment orchestrator (the make-or-break for R1-2 / R2-3).

Trains the four head variants (classic_fc, bottleneck_fc, mlp_head, qresnet)
across several seeds, then runs the dense multi-seed robustness sweep over all of
them on the shared test set. This is the single command to reproduce the main
revision result.

Run (server, RTX 4090):
    python -m revision.run_ablation --data data/raw/kaggle_qr --seeds 0,1,2 --epochs 5
Then inspect:
    experiments_revision/robustness_raw_summary.csv   (acc mean/std per level)
    experiments_revision/robustness_raw_metrics.csv   (AURC, sigma* per model)
"""

import argparse

from .train import train_model
from . import robustness

ABLATION_MODELS = ["classic_fc", "bottleneck_fc", "mlp_head", "qresnet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="experiments_revision")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--noise-aware", action="store_true",
                    help="also train a noise-aware Q-ResNet and ResNet (R2-2)")
    ap.add_argument("--pert-seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    for seed in seeds:
        for name in ABLATION_MODELS:
            train_model(name, a.data, a.out, seed=seed, epochs=a.epochs,
                        batch_size=a.batch_size, num_workers=a.num_workers)
        if a.noise_aware:
            for name in ["classic_fc", "qresnet"]:
                train_model(name, a.data, a.out, seed=seed, epochs=a.epochs,
                            batch_size=a.batch_size, num_workers=a.num_workers,
                            noise_aware=True, noise_sigma_max=0.3)

    robustness.run(a.out, a.data, f"{a.out}/robustness_raw.csv",
                   threats={"noise", "occlusion"},
                   noise_levels=robustness.DEFAULT_NOISE,
                   occ_levels=robustness.DEFAULT_OCCLUSION,
                   pert_seeds=[int(s) for s in a.pert_seeds.split(",")],
                   num_workers=a.num_workers)


if __name__ == "__main__":
    main()

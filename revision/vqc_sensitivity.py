"""VQC sensitivity study (R2-4): qubit count, circuit depth, ansatz choice.

Trains a Q-ResNet for each (n_qubits, n_layers, ansatz) configuration and writes
one checkpoint per config into --out. Robustness can then be evaluated with
`python -m revision.robustness --exp-dir <out> --data <data>`.

Run (server):
  python -m revision.vqc_sensitivity --data data/raw/kaggle_qr --out experiments_vqc_sens
"""

import argparse
import itertools

from .train import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="experiments_vqc_sens")
    ap.add_argument("--qubits", default="2,4,6,8")
    ap.add_argument("--layers", default="1,2,3")
    ap.add_argument("--ansatze", default="strong,basic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    a = ap.parse_args()
    from .runlog import start_logging
    start_logging(a.out, "vqc_sensitivity")

    qubits = [int(x) for x in a.qubits.split(",")]
    layers = [int(x) for x in a.layers.split(",")]
    ansatze = a.ansatze.split(",")

    for nq, nl, ans in itertools.product(qubits, layers, ansatze):
        print(f"\n#### VQC config: qubits={nq} layers={nl} ansatz={ans} ####")
        train_model("qresnet", a.data, a.out, seed=a.seed, epochs=a.epochs,
                    batch_size=a.batch_size, num_workers=a.num_workers,
                    model_kwargs={"n_qubits": nq, "n_layers": nl, "ansatz": ans})


if __name__ == "__main__":
    main()

"""Train any model variant, with optional noise-aware augmentation.

Addresses:
  - R2-3 / R1-2: trains the ablation heads with an identical protocol so the
    only difference is the head.
  - R2-2: `--noise-aware` injects Gaussian noise (on [0,1] images) during
    training, so we can report clean vs noise-aware trained models.
  - reproducibility: every run is seeded.

Usage:
  python -m revision.train --model qresnet --seed 0 --data data/raw/kaggle_qr
  python -m revision.train --model bottleneck_fc --seed 0 --freeze-backbone
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .data import load_image_data
from .models import build_model, head_param_count, count_params
from .perturbations import add_gaussian_noise
from .seeding import set_seed, perturbation_generator


def _noise_aware_batch(x, sigma_max, gen):
    """Apply a random-strength Gaussian corruption to a fraction of the batch."""
    sigma = float(torch.rand(1, generator=gen).item()) * sigma_max
    return add_gaussian_noise(x, sigma, gen)


def evaluate(model, loader, device):
    model.eval()
    probs, preds, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = torch.softmax(out, dim=1)[:, 1]
            probs.extend(p.cpu().numpy())
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    acc = 100 * accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    f1 = f1_score(labels, preds)
    return {"acc": acc, "auc": auc, "f1": f1}


def train_model(model_name, data_dir, out_dir, seed=0, epochs=5, lr=1e-4,
                batch_size=128, freeze_backbone=False, noise_aware=False,
                noise_sigma_max=0.3, max_per_class=None, num_workers=8, device=None,
                model_kwargs=None, pretrained=True):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = model_kwargs or {}

    train_loader, val_loader, test_loader = load_image_data(
        data_dir, batch_size=batch_size, seed=42, num_workers=num_workers,
        max_per_class=max_per_class)

    model = build_model(model_name, pretrained=pretrained, **model_kwargs).to(device)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    aug_gen = perturbation_generator(seed + 10_000)

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{model_name}_seed{seed}" + ("_noiseaware" if noise_aware else "")
    log_rows = []
    best_acc, best_path = -1.0, os.path.join(out_dir, f"best_{tag}.pth")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if noise_aware:
                x = _noise_aware_batch(x, noise_sigma_max, aug_gen).to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        val = evaluate(model, val_loader, device)
        train_loss = running / max(1, len(train_loader))
        print(f"[{tag}] epoch {epoch+1}/{epochs} loss={train_loss:.4f} "
              f"val_acc={val['acc']:.2f} val_auc={val['auc']:.4f}")
        log_rows.append({"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(model.state_dict(), best_path)

    # reload best and report clean test metrics
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test = evaluate(model, test_loader, device)
    meta = {
        "model": model_name, "seed": seed, "epochs": epochs, "lr": lr,
        "freeze_backbone": freeze_backbone, "noise_aware": noise_aware,
        "head_params": head_param_count(model),
        "trainable_params": count_params(model),
        "best_val_acc": best_acc, "test": test, "train_time_s": time.time() - t0,
        "model_kwargs": model_kwargs, "best_path": best_path,
    }
    with open(os.path.join(out_dir, f"meta_{tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    import csv
    with open(os.path.join(out_dir, f"trainlog_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        w.writeheader(); w.writerows(log_rows)
    print(f"[{tag}] DONE. clean test: {test}")
    return meta


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="experiments_revision")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--noise-aware", action="store_true")
    ap.add_argument("--noise-sigma-max", type=float, default=0.3)
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--ansatz", default="strong")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    a = ap.parse_args()
    train_model(a.model, a.data, a.out, seed=a.seed, epochs=a.epochs, lr=a.lr,
                batch_size=a.batch_size, freeze_backbone=a.freeze_backbone,
                noise_aware=a.noise_aware, noise_sigma_max=a.noise_sigma_max,
                max_per_class=a.max_per_class, num_workers=a.num_workers,
                model_kwargs={"n_qubits": a.n_qubits, "n_layers": a.n_layers, "ansatz": a.ansatz})


if __name__ == "__main__":
    _cli()

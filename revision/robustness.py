"""Dense, multi-seed robustness evaluation.

Addresses R2-1 (the sigma=0.4 point looks like an outlier): instead of a sparse
3-point table we sweep sigma on a fine grid and repeat each corruption with
several independent perturbation seeds, reporting mean +/- std. From the curve
we derive two summary metrics that do not hinge on a single sigma:

  AURC  : Area Under the Robustness Curve (mean accuracy integrated over the
          sigma range, normalized to [0,100]); higher = more robust overall.
  sigma*: critical severity, the smallest sigma at which mean accuracy first
          drops below a threshold (default 70%); larger = more robust.

It evaluates EVERY trained checkpoint found under --exp-dir (meta_*.json) on the
same held-out test set, so the ablation heads are compared on identical data and
identical corrupted images.
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

from .data import load_image_data
from .models import build_model
from .perturbations import apply_threat
from .seeding import perturbation_generator

DEFAULT_NOISE = [round(s, 3) for s in np.arange(0.0, 0.701, 0.05)]
DEFAULT_OCCLUSION = list(range(0, 121, 20))


def _load_checkpoints(exp_dir):
    out = []
    for mp in sorted(glob.glob(os.path.join(exp_dir, "meta_*.json"))):
        with open(mp) as fh:
            meta = json.load(fh)
        out.append(meta)
    return out


def _eval_corrupted(model, loader, threat, level, device, pert_seeds):
    accs = []
    for ps in pert_seeds:
        gen = perturbation_generator(ps)
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x = apply_threat(x, threat, level, generator=gen).to(device)
                pred = model(x).argmax(1).cpu()
                correct += (pred == y).sum().item()
                total += y.size(0)
        accs.append(100 * correct / total)
    return accs


def run(exp_dir, data_dir, out_csv, threats, noise_levels, occ_levels,
        pert_seeds, device=None, max_per_class=None, num_workers=8,
        crit_threshold=70.0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = load_image_data(data_dir, batch_size=64, seed=42,
                                        num_workers=num_workers, max_per_class=max_per_class)
    metas = _load_checkpoints(exp_dir)
    if not metas:
        raise SystemExit(f"No meta_*.json checkpoints found in {exp_dir}")

    rows = []
    for meta in metas:
        model = build_model(meta["model"], pretrained=False, **(meta.get("model_kwargs") or {})).to(device)
        model.load_state_dict(torch.load(meta["best_path"], map_location=device))
        model.eval()
        label = os.path.splitext(os.path.basename(meta["best_path"]))[0].replace("best_", "")
        print(f"== evaluating {label} ==")
        plan = []
        if "noise" in threats:
            plan += [("noise", lv) for lv in noise_levels]
        if "occlusion" in threats:
            plan += [("occlusion", lv) for lv in occ_levels]
        for threat, level in plan:
            accs = _eval_corrupted(model, test_loader, threat, level, device, pert_seeds)
            for ps, a in zip(pert_seeds, accs):
                rows.append({"label": label, "model": meta["model"], "train_seed": meta["seed"],
                             "noise_aware": meta["noise_aware"], "threat": threat,
                             "level": level, "pert_seed": ps, "acc": a})
            print(f"   {threat} {level}: {np.mean(accs):.2f} +/- {np.std(accs):.2f}")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[SUCCESS] raw per-seed results -> {out_csv}")

    _summarize(rows, out_csv.replace(".csv", "_summary.csv"), crit_threshold)
    return rows


def _summarize(rows, out_csv, crit_threshold):
    # aggregate over pert_seed AND train_seed, per (model, noise_aware, threat, level)
    import collections
    bucket = collections.defaultdict(list)
    for r in rows:
        bucket[(r["model"], r["noise_aware"], r["threat"], r["level"])].append(r["acc"])
    summ = []
    for (model, na, threat, level), accs in sorted(bucket.items()):
        summ.append({"model": model, "noise_aware": na, "threat": threat, "level": level,
                     "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                     "n": len(accs)})
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    print(f"[SUCCESS] summary -> {out_csv}")

    # derived metrics per (model, noise_aware, threat)
    import collections as c
    by = c.defaultdict(list)
    for s in summ:
        by[(s["model"], s["noise_aware"], s["threat"])].append((s["level"], s["acc_mean"]))
    print("\n--- derived robustness metrics ---")
    deriv = []
    for (model, na, threat), pts in sorted(by.items()):
        pts.sort()
        xs = np.array([p[0] for p in pts], float)
        ys = np.array([p[1] for p in pts], float)
        aurc = float(np.trapz(ys, xs) / (xs.max() - xs.min())) if xs.max() > xs.min() else float(ys.mean())
        below = xs[ys < crit_threshold]
        sigma_star = float(below.min()) if below.size else float(xs.max())
        deriv.append({"model": model, "noise_aware": na, "threat": threat,
                      "AURC": aurc, "sigma_star": sigma_star})
        print(f"   {model:14} na={na} {threat:9} AURC={aurc:6.2f}  sigma*={sigma_star}")
    with open(out_csv.replace("_summary.csv", "_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(deriv[0].keys()))
        w.writeheader(); w.writerows(deriv)


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="experiments_revision")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="experiments_revision/robustness_raw.csv")
    ap.add_argument("--threats", default="noise,occlusion")
    ap.add_argument("--pert-seeds", default="0,1,2,3,4")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--crit-threshold", type=float, default=70.0)
    a = ap.parse_args()
    run(a.exp_dir, a.data, a.out,
        threats=set(a.threats.split(",")),
        noise_levels=DEFAULT_NOISE, occ_levels=DEFAULT_OCCLUSION,
        pert_seeds=[int(s) for s in a.pert_seeds.split(",")],
        max_per_class=a.max_per_class, num_workers=a.num_workers,
        crit_threshold=a.crit_threshold)


if __name__ == "__main__":
    _cli()

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


def _eval_model_single_pass(model, loader, device, plan, pert_seeds, batch_size):
    """Evaluate every (threat, level, pert_seed) in ONE pass over the test set.

    The loader (hence the disk) is iterated exactly once; for each batch we apply
    all corruption variants in memory. The perturbation RNG is seeded per
    (pert_seed, batch_index) so results are deterministic and identical across
    models (test loader has shuffle=False). Returns {(threat,level,ps): acc}.
    """
    correct = {(t, lv, ps): 0 for (t, lv) in plan for ps in pert_seeds}
    total = {k: 0 for k in correct}
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            yc = y
            for (threat, level) in plan:
                for ps in pert_seeds:
                    gen = perturbation_generator(ps * 100003 + bi)
                    xp = apply_threat(x, threat, level, generator=gen).to(device)
                    pred = model(xp).argmax(1).cpu()
                    k = (threat, level, ps)
                    correct[k] += (pred == yc).sum().item()
                    total[k] += yc.size(0)
            if bi % 50 == 0:
                print(f"   ...batch {bi}", flush=True)
    return {k: 100 * correct[k] / max(1, total[k]) for k in correct}


def run(exp_dir, data_dir, out_csv, threats, noise_levels, occ_levels,
        pert_seeds, device=None, max_per_class=None, num_workers=8,
        crit_threshold=70.0, batch_size=128):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = load_image_data(data_dir, batch_size=batch_size, seed=42,
                                        num_workers=num_workers, max_per_class=max_per_class)
    metas = _load_checkpoints(exp_dir)
    if not metas:
        raise SystemExit(f"No meta_*.json checkpoints found in {exp_dir}")

    plan = []
    if "noise" in threats:
        plan += [("noise", lv) for lv in noise_levels]
    if "occlusion" in threats:
        plan += [("occlusion", lv) for lv in occ_levels]

    rows = []
    for meta in metas:
        model = build_model(meta["model"], pretrained=False, **(meta.get("model_kwargs") or {})).to(device)
        model.load_state_dict(torch.load(meta["best_path"], map_location=device, weights_only=True))
        model.eval()
        label = os.path.splitext(os.path.basename(meta["best_path"]))[0].replace("best_", "")
        print(f"== evaluating {label} (single disk pass, {len(plan)} levels x {len(pert_seeds)} seeds) ==")
        accs = _eval_model_single_pass(model, test_loader, device, plan, pert_seeds, batch_size)
        for (threat, level) in plan:
            vals = [accs[(threat, level, ps)] for ps in pert_seeds]
            for ps in pert_seeds:
                rows.append({"label": label, "model": meta["model"], "train_seed": meta["seed"],
                             "noise_aware": meta["noise_aware"], "threat": threat,
                             "level": level, "pert_seed": ps, "acc": accs[(threat, level, ps)]})
            print(f"   {threat} {level}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}")

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
    ap.add_argument("--pert-seeds", default="0,1,2")
    ap.add_argument("--noise-levels", default=None,
                    help="comma list overriding the default dense grid, e.g. 0,0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--occ-levels", default=None, help="comma list, e.g. 0,40,80,100")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--crit-threshold", type=float, default=70.0)
    a = ap.parse_args()
    from .runlog import start_logging, write_report
    start_logging(a.exp_dir, "robustness")
    noise_levels = [float(x) for x in a.noise_levels.split(",")] if a.noise_levels else DEFAULT_NOISE
    occ_levels = [int(x) for x in a.occ_levels.split(",")] if a.occ_levels else DEFAULT_OCCLUSION
    run(a.exp_dir, a.data, a.out,
        threats=set(a.threats.split(",")),
        noise_levels=noise_levels, occ_levels=occ_levels,
        pert_seeds=[int(s) for s in a.pert_seeds.split(",")],
        max_per_class=a.max_per_class, num_workers=a.num_workers,
        crit_threshold=a.crit_threshold, batch_size=a.batch_size)
    write_report(a.exp_dir)


if __name__ == "__main__":
    _cli()

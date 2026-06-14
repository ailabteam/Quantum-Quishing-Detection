"""Evaluate trained models on a shortcut-controlled subset (Cua 1).

Reviewer 1 suspects the near-perfect accuracy is a shortcut: QR images are
rendered from URLs, so QR density correlates with payload length, which may
correlate with the malicious/benign label. revision.audit_dataset exports a
payload-length-matched subset (equal class counts per length bin). This script
evaluates the trained models on that subset: if clean accuracy DROPS sharply
relative to the full test set, the shortcut is confirmed.

Run (after audit):
  python -m revision.eval_subset --exp-dir experiments_revision --data data/raw/qrset \
      --subset experiments_revision/audit/length_matched_subset.csv
"""

import argparse
import csv
import glob
import json
import os

import torch

from .data import load_subset_from_csv
from .models import build_model
from .robustness import config_group
from .train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="experiments_revision")
    ap.add_argument("--data", required=True, help="ImageFolder root the CSV paths are relative to")
    ap.add_argument("--subset", required=True, help="CSV with columns file,class")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args()
    from .runlog import start_logging
    start_logging(a.exp_dir, "eval_subset")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = load_subset_from_csv(a.data, a.subset, batch_size=a.batch_size, num_workers=a.num_workers)

    rows = []
    for mp in sorted(glob.glob(os.path.join(a.exp_dir, "meta_*.json"))):
        with open(mp) as fh:
            meta = json.load(fh)
        model = build_model(meta["model"], pretrained=False, **(meta.get("model_kwargs") or {})).to(device)
        model.load_state_dict(torch.load(meta["best_path"], map_location=device, weights_only=True))
        m = evaluate(model, loader, device)
        full = meta.get("test", {})
        group = config_group(meta)
        print(f"{group} seed{meta['seed']}: subset acc={m['acc']:.2f} auc={m['auc']:.4f} f1={m['f1']:.4f} "
              f"(full-test acc was {full.get('acc', float('nan')):.2f}) -> drop {full.get('acc', 0) - m['acc']:+.2f}")
        rows.append({"group": group, "seed": meta["seed"],
                     "subset_acc": m["acc"], "subset_auc": m["auc"], "subset_f1": m["f1"],
                     "fulltest_acc": full.get("acc"), "acc_drop": (full.get("acc") or 0) - m["acc"]})

    out = a.out or os.path.join(a.exp_dir, "subset_eval.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[SUCCESS] subset evaluation -> {out}")
    print("If subset_acc is much lower than fulltest_acc, the payload-length shortcut is confirmed.")


if __name__ == "__main__":
    main()

"""Dataset bias / shortcut audit (R1-1).

Reviewer 1 worries that near-perfect clean accuracy may reflect a shortcut: the
model could separate classes by QR *density / payload length / payload type /
generator artifacts* rather than by anything related to maliciousness. The labels
in the Kaggle set carry no payload metadata, so this script RE-DECODES each QR
image and reports per-class distributions of the obvious shortcut variables. It
also exports a payload-length-matched subset so the model can be re-evaluated
with the most obvious shortcut removed.

Requires a QR decoder. Tries pyzbar first (gives symbol type), falls back to
OpenCV's QRCodeDetector. Install on the server:
    pip install pyzbar opencv-python   (and the zbar shared lib for pyzbar)

Run (server):
    python -m revision.audit_dataset --data data/raw/kaggle_qr --out experiments_revision/audit
"""

import argparse
import csv
import os
import re
from collections import Counter, defaultdict

import numpy as np

_URL_RE = re.compile(r"^(https?://|www\.)", re.I)


def _decoders():
    pyzbar = cv2 = None
    try:
        from pyzbar import pyzbar as _pz
        pyzbar = _pz
    except Exception:
        pass
    try:
        import cv2 as _cv2
        cv2 = _cv2
    except Exception:
        pass
    if pyzbar is None and cv2 is None:
        raise SystemExit("Need pyzbar or opencv-python to decode QR codes. "
                         "pip install pyzbar opencv-python")
    return pyzbar, cv2


def _payload_type(s: str) -> str:
    if not s:
        return "undecoded"
    low = s.lower()
    if _URL_RE.match(s):
        return "url"
    for scheme in ("wifi:", "tel:", "mailto:", "smsto:", "geo:", "bitcoin:", "matmsg:"):
        if low.startswith(scheme):
            return scheme.rstrip(":")
    return "text"


def _decode_one(path, pyzbar, cv2):
    """Return (payload_str_or_None)."""
    if pyzbar is not None:
        try:
            from PIL import Image
            res = pyzbar.decode(Image.open(path).convert("RGB"))
            if res:
                return res[0].data.decode("utf-8", "replace")
        except Exception:
            pass
    if cv2 is not None:
        try:
            img = cv2.imread(path)
            data, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
            if data:
                return data
        except Exception:
            pass
    return None


def audit(data_dir, out_dir, limit_per_class=None):
    pyzbar, cv2 = _decoders()
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    per_class_len = defaultdict(list)
    per_class_type = defaultdict(Counter)
    per_class_decoded = defaultdict(lambda: [0, 0])  # [decoded, total]

    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    for cls in classes:
        cdir = os.path.join(data_dir, cls)
        files = sorted(f for f in os.listdir(cdir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
        if limit_per_class:
            files = files[:limit_per_class]
        for fn in files:
            path = os.path.join(cdir, fn)
            payload = _decode_one(path, pyzbar, cv2)
            ptype = _payload_type(payload)
            plen = len(payload) if payload else 0
            per_class_decoded[cls][1] += 1
            if payload:
                per_class_decoded[cls][0] += 1
                per_class_len[cls].append(plen)
            per_class_type[cls][ptype] += 1
            rows.append({"file": os.path.join(cls, fn), "class": cls,
                         "decoded": int(payload is not None),
                         "payload_len": plen, "payload_type": ptype})
        print(f"[audit] {cls}: {per_class_decoded[cls][0]}/{per_class_decoded[cls][1]} decoded")

    with open(os.path.join(out_dir, "audit_per_image.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n=== PER-CLASS SUMMARY ===")
    summ = []
    for cls in classes:
        lens = np.array(per_class_len[cls], float)
        dec, tot = per_class_decoded[cls]
        s = {"class": cls, "n": tot, "decode_rate": round(dec / max(1, tot), 4),
             "len_mean": float(lens.mean()) if lens.size else 0.0,
             "len_std": float(lens.std()) if lens.size else 0.0,
             "len_median": float(np.median(lens)) if lens.size else 0.0,
             "types": dict(per_class_type[cls])}
        summ.append(s)
        print(f" {cls}: decode={s['decode_rate']:.2%} len {s['len_mean']:.1f}+/-{s['len_std']:.1f} "
              f"(median {s['len_median']:.0f}) types={s['types']}")
    with open(os.path.join(out_dir, "audit_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["class", "n", "decode_rate", "len_mean",
                                           "len_std", "len_median", "types"])
        w.writeheader()
        for s in summ:
            r = dict(s); r["types"] = str(r["types"]); w.writerow(r)

    _export_length_matched(rows, classes, out_dir)
    print(f"\n[SUCCESS] audit written to {out_dir}/")
    print("Interpretation: if payload_len distributions or decode_rate differ "
          "sharply across classes, near-perfect clean accuracy is a likely shortcut.")


def _export_length_matched(rows, classes, out_dir, n_bins=20):
    """Sample a payload-length-matched subset (equal class counts per length bin)."""
    if len(classes) != 2:
        return
    decoded = [r for r in rows if r["decoded"]]
    if not decoded:
        print("[audit] no decoded payloads; skipping length-matched subset.")
        return
    lens = np.array([r["payload_len"] for r in decoded], float)
    edges = np.quantile(lens, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1
    by_bin = defaultdict(lambda: defaultdict(list))
    for r in decoded:
        b = int(np.searchsorted(edges, r["payload_len"], side="right") - 1)
        by_bin[b][r["class"]].append(r["file"])
    matched = []
    for b, perc in by_bin.items():
        if all(perc.get(c) for c in classes):
            k = min(len(perc[c]) for c in classes)
            for c in classes:
                matched.extend({"file": f, "class": c} for f in perc[c][:k])
    with open(os.path.join(out_dir, "length_matched_subset.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "class"])
        w.writeheader(); w.writerows(matched)
    print(f"[audit] length-matched subset: {len(matched)} images "
          f"({sum(m['class']==classes[0] for m in matched)} vs "
          f"{sum(m['class']==classes[1] for m in matched)})")


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="experiments_revision/audit")
    ap.add_argument("--limit-per-class", type=int, default=None)
    a = ap.parse_args()
    from .runlog import start_logging
    start_logging(a.out, "audit")
    audit(a.data, a.out, a.limit_per_class)


if __name__ == "__main__":
    _cli()

"""Automatic file logging + a consolidated human-readable REPORT.md.

Goal: the user runs an experiment, pushes `experiments_revision/`, and the
reviewer pulls a single `REPORT.md` instead of reading pasted console output.

  - start_logging(out_dir, name): tee stdout/stderr to out_dir/logs/<name>_<ts>.log
  - write_report(out_dir): scan meta_*.json + *_metrics.csv + *_summary.csv and
    emit out_dir/REPORT.md with clean-performance and robustness tables, plus the
    key ablation comparison (qresnet vs bottleneck_fc vs mlp_head).
"""

import csv
import datetime
import glob
import json
import os
import sys
from collections import defaultdict


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def start_logging(out_dir, name):
    logdir = os.path.join(out_dir, "logs")
    os.makedirs(logdir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(logdir, f"{name}_{ts}.log")
    fh = open(path, "a", buffering=1, encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"[log] {datetime.datetime.now().isoformat(timespec='seconds')} writing console to {path}")
    return path


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, var ** 0.5


def write_report(out_dir, out_path=None):
    out_path = out_path or os.path.join(out_dir, "REPORT.md")
    metas = []
    for p in sorted(glob.glob(os.path.join(out_dir, "meta_*.json"))):
        try:
            with open(p) as fh:
                metas.append(json.load(fh))
        except Exception as e:
            print(f"[report] skip {p}: {e}")

    lines = ["# Revision experiment report",
             f"_generated {datetime.datetime.now().isoformat(timespec='seconds')}_", ""]

    # ---- clean performance, aggregated across seeds ----
    from .robustness import config_group
    lines += ["## Clean test performance (aggregated across seeds)", ""]
    if metas:
        agg = defaultdict(lambda: {"acc": [], "auc": [], "f1": [], "seeds": [],
                                   "head_params": None, "trainable_params": None,
                                   "train_time_s": []})
        for m in metas:
            key = (config_group(m), m.get("noise_aware", False))
            a = agg[key]
            t = m.get("test", {})
            a["acc"].append(t.get("acc")); a["auc"].append(t.get("auc")); a["f1"].append(t.get("f1"))
            a["seeds"].append(m.get("seed"))
            a["head_params"] = m.get("head_params")
            a["trainable_params"] = m.get("trainable_params")
            a["train_time_s"].append(m.get("train_time_s"))
        lines += ["| model | noise_aware | seeds | head params | trainable | acc (mean±std) | auc | f1 |",
                  "|---|---|---|---|---|---|---|---|"]
        for (model, na), a in sorted(agg.items()):
            am, asd = _mean_std(a["acc"]); aucm, _ = _mean_std(a["auc"]); f1m, _ = _mean_std(a["f1"])
            seeds = ",".join(str(s) for s in sorted(x for x in a["seeds"] if x is not None))
            lines.append(f"| {model} | {na} | {seeds} | {a['head_params']} | {a['trainable_params']} | "
                         f"{am:.2f}±{asd:.2f} | {aucm:.4f} | {f1m:.4f} |")
    else:
        lines.append("_no meta_*.json found_")
    lines.append("")

    # ---- robustness derived metrics (AURC, sigma*) ----
    lines += ["## Robustness summary (AURC, sigma*)", ""]
    metric_rows = _read_csvs(out_dir, "*_metrics.csv")
    if metric_rows:
        lines += ["| model | noise_aware | threat | AURC | sigma* |", "|---|---|---|---|---|"]
        for r in sorted(metric_rows, key=lambda r: (r.get("threat", ""), r.get("model", ""))):
            lines.append(f"| {r.get('model')} | {r.get('noise_aware')} | {r.get('threat')} | "
                         f"{float(r.get('AURC', 'nan')):.2f} | {r.get('sigma_star')} |")
        lines += ["", _ablation_verdict(metric_rows)]
    else:
        lines.append("_no *_metrics.csv found (run revision.robustness)_")
    lines.append("")

    # ---- full accuracy-vs-level curves ----
    summ_rows = _read_csvs(out_dir, "*_summary.csv")
    if summ_rows:
        lines += ["## Accuracy vs severity (mean±std)", ""]
        by_threat = defaultdict(list)
        for r in summ_rows:
            by_threat[r["threat"]].append(r)
        for threat, rows in sorted(by_threat.items()):
            lines += [f"### {threat}", ""]
            models = sorted({r["model"] + ("*" if r["noise_aware"] in ("True", True) else "") for r in rows})
            levels = sorted({float(r["level"]) for r in rows})
            lines.append("| level | " + " | ".join(models) + " |")
            lines.append("|" + "---|" * (len(models) + 1))
            idx = {(r["model"] + ("*" if r["noise_aware"] in ("True", True) else ""), float(r["level"])):
                   (float(r["acc_mean"]), float(r["acc_std"])) for r in rows}
            for lv in levels:
                cells = []
                for mdl in models:
                    v = idx.get((mdl, lv))
                    cells.append(f"{v[0]:.1f}±{v[1]:.1f}" if v else "-")
                lines.append(f"| {lv:g} | " + " | ".join(cells) + " |")
            lines.append("")
        lines.append("_models marked * are noise-aware trained._")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[report] wrote {out_path}")
    return out_path


def _read_csvs(out_dir, pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, pattern))):
        with open(p, encoding="utf-8") as fh:
            rows.extend(list(csv.DictReader(fh)))
    return rows


def _ablation_verdict(metric_rows):
    """Auto-summarize the make-or-break comparison for the noise threat."""
    noise = {r["model"]: float(r["AURC"]) for r in metric_rows
             if r.get("threat") == "noise" and r.get("noise_aware") in ("False", False)}
    qgroups = {k: v for k, v in noise.items() if k.startswith("qresnet")}
    if not qgroups:
        return "_qresnet noise AURC not available for verdict._"
    best_q_name = max(qgroups, key=qgroups.get)
    q = qgroups[best_q_name]
    classical = {k: v for k, v in noise.items() if k in ("bottleneck_fc", "mlp_head", "classic_fc")}
    if not classical:
        return "_no classical ablation heads for verdict._"
    best_classical_name = max(classical, key=classical.get)
    best_classical = classical[best_classical_name]
    delta = q - best_classical
    verdict = (f"**Ablation verdict (noise AURC):** best VQC `{best_q_name}`={q:.2f} vs best classical head "
               f"`{best_classical_name}`={best_classical:.2f} -> gap {delta:+.2f}. ")
    if delta > 2.0:
        verdict += "Quantum head leads; check this exceeds cross-seed std before claiming advantage."
    elif delta < -2.0:
        verdict += "Classical head leads; the quantum advantage does NOT hold here, reframe honestly."
    else:
        verdict += "Within ~2 pts of the best classical head; likely no clear quantum advantage, reframe honestly."
    return verdict

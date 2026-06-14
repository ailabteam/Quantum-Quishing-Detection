# Revision experiments (TDSC-2025-11-2261, major revision)

Self-contained package addressing the round-1 reviewer requests. It does not
modify the original `src/` pipeline; it adds a corrected, reproducible harness.

## What changed vs the original code
- **Noise model is now pre-normalization.** Perturbations act on raw `[0,1]`
  images (clamped), then ImageNet normalization happens inside the model wrapper
  (`NormalizedModel`). The original added noise on top of already-normalized
  tensors, so the reported sigma was on the normalized scale and undocumented.
- **Everything is seeded** (training and perturbation sampling), and the
  robustness sweep repeats each corruption over several perturbation seeds and
  reports mean +/- std.
- **R_y angle encoding** (`rotation="Y"`) to match Eq. (2). The original used
  `AngleEmbedding`'s default `RX`.
- **Class-stratified 70/10/20 split** (the paper claimed stratification; the
  original used a plain random split).

## Reviewer mapping
| Request | Module |
|---|---|
| R1-2 / R2-3 ablation isolating the VQC | `models.py` (4 param-matched heads), `run_ablation.py` |
| R2-1 dense sigma curve + multi-seed | `robustness.py` (fine grid, mean+/-std, AURC, sigma*) |
| R2-2 noise-aware training | `train.py --noise-aware`, `run_ablation.py --noise-aware` |
| R2-4 VQC sensitivity (qubits/layers/ansatz) | `vqc_sensitivity.py` |
| R1-1 dataset bias / shortcut audit | `audit_dataset.py` |

## The four ablation heads (identical ResNet-18 backbone)
| name | head | head params |
|---|---|---|
| `classic_fc` | FC 512->2 | 1026 |
| `bottleneck_fc` | 512->4 -> tanh -> 4->2 (no VQC) | 2062 |
| `mlp_head` | 512->4 -> tanh -> 4->4 -> ReLU -> 4->2 | 2082 |
| `qresnet` | 512->4 -> tanh -> VQC(4q,2L) -> 4->2 | 2086 |

`bottleneck_fc` / `mlp_head` / `qresnet` are parameter-matched to within ~24
params, so a robustness gap between them isolates the VQC's contribution.

## Environment (conda-only server, no sudo)
Recommended (robust): minimal conda env + pip wheels, which avoids the libmamba
`pytorch-cuda` solver stall and needs no system CUDA:
```bash
bash revision/setup_env.sh
conda activate quishing-rev
```
Fallback (conda-native, may stall on some solvers):
```bash
conda env create -f revision/environment-revision.yml && conda activate quishing-rev
```
No system libraries are needed: the VQC runs on PennyLane's `default.qubit`
simulator (no lightning.gpu / cuQuantum / CUDA toolkit), and the bias audit
decodes QR codes with OpenCV (headless wheel, so no system `libGL` or `zbar`).

## Quick start (server, RTX 4090)
```bash
# 0) sanity: pipeline runs end-to-end on synthetic data (CPU, ~1 min)
python -m revision.smoke_test

# 1) MAIN RESULT: ablation across seeds + dense robustness sweep
python -m revision.run_ablation --data data/raw/kaggle_qr --seeds 0,1,2 --epochs 5
#    optionally add noise-aware training (R2-2):
python -m revision.run_ablation --data data/raw/kaggle_qr --seeds 0,1,2 --noise-aware

# 2) VQC sensitivity (R2-4)
python -m revision.vqc_sensitivity --data data/raw/kaggle_qr --out experiments_vqc_sens
python -m revision.robustness --exp-dir experiments_vqc_sens --data data/raw/kaggle_qr \
    --out experiments_vqc_sens/robustness_raw.csv

# 3) dataset bias audit (R1-1) -- uses conda-forge opencv (already in the env)
python -m revision.audit_dataset --data data/raw/kaggle_qr
```

## Reporting workflow (no log pasting)
Every CLI tees its console to `experiments_revision/logs/<name>_<ts>.log` and the
ablation / robustness runs regenerate a single consolidated `REPORT.md` with the
clean-performance table, the AURC/sigma* summary, an automatic ablation verdict,
and the accuracy-vs-severity curves. Regenerate it anytime:
```bash
python -m revision.report --exp-dir experiments_revision
```
Then just push the lightweight artifacts (the `.gitignore` in `experiments_revision/`
excludes the large `.pth` checkpoints automatically):
```bash
git add experiments_revision && git commit -m "round-1 results" && git push
```

## Outputs to push back for analysis
- `experiments_revision/REPORT.md` (read this first: everything consolidated)
- `experiments_revision/meta_*.json` (clean test acc/auc/f1 + param counts per run)
- `experiments_revision/robustness_raw*.csv` (per-seed, summary mean/std, metrics AURC/sigma*)
- `experiments_revision/logs/*.log` (full console trace incl. timings)
- `experiments_revision/audit/audit_summary.csv` + `length_matched_subset.csv`

## Reading the result
- If `qresnet` beats `bottleneck_fc` and `mlp_head` on AURC / sigma* by a margin
  larger than the cross-seed std, the quantum contribution is real -> strong paper.
- If the three are within noise of each other, the gain is classical
  (tanh + bottleneck). Do not fudge: reframe the contribution honestly
  (e.g. same robustness at fewer params, or wider verified margin).

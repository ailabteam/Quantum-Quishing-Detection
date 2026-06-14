# Quantum-Quishing-Detection

## Dependability of Quishing Detection under Realistic Corruption: A Rigorous Robustness Study and a Simple Effective Defense

This repository contains the code and experimental results for the paper (under major revision at IEEE Transactions on Dependable and Secure Computing, TDSC).

> We study the dependability of image-based quishing (QR-phishing) detection under realistic corruption. Using a single ResNet-18 backbone with **parameter-matched classification heads**, we show that the **classifier-head design, not a quantum circuit, governs robustness**. A low-qubit Variational Quantum Circuit (VQC) head provides **no dependable robustness advantage** over parameter-matched classical heads, and its apparent single-seed gains are **training-instability artifacts** that vanish under multi-seed control. We further show that the near-perfect clean accuracy is **not** explained by a payload-length shortcut, and that a simple **noise-aware training** defense restores dependability for every head.

> **Note on a revised conclusion.** An earlier version of this work reported a large "quantum advantage" under Gaussian noise. Under a corrected, standardized evaluation protocol (noise applied pre-normalization) and parameter-matched, multi-seed ablations requested by the reviewers, that advantage **does not hold**: it was attributable to the classical nonlinear bottleneck and to single-seed training instability. This repository reflects the corrected, honest findings.

---

## Key results (corrected)

Noise robustness, measured by AURC (area under the accuracy-vs-$\sigma$ curve over $\sigma\in[0,0.2]$ on the $[0,1]$ image scale; higher = more robust). Heads share one ResNet-18 backbone and are parameter-matched to within ~24 parameters.

| Classifier head | clean acc | AURC (noise) |
| :--- | :---: | :---: |
| Linear FC ($512\to2$) | 100% | 74.9 |
| Bottleneck+tanh ($512\to4\to2$, no VQC) | 100% | 83.5 |
| Classical MLP head | 100% | **88.8** |
| VQC head (Q-ResNet, 4q/2L) | 100% | 80.1 |

The VQC head is **below** both parameter-matched classical nonlinear heads. A 3-seed control at a matched bottleneck width of 6 confirms this (VQC 79.4 vs classical MLP 88.3) and shows the VQC's robustness is unstable across seeds (per-seed std ≈ 19).

**Defense (noise-aware training).** Training with per-sample Gaussian augmentation restores dependability for every head:

| head | AURC clean-trained | AURC noise-aware |
| :--- | :---: | :---: |
| Linear FC | 74.9 | **99.98** |
| Classical MLP | 88.8 | **99.2** |

**Dataset validity.** The QR images are rendered from URLs. The payload-length distributions are nearly identical across classes (benign 80.1±5.8 vs malicious 79.7±7.6 characters) and accuracy persists at ~100% on a payload-length-matched subset, so the high clean accuracy is not a payload-length/density shortcut.

---

## Reproducing the revision results

The revision harness lives in [`revision/`](revision/). It is self-contained and does not modify the original `src/`. See [`revision/README.md`](revision/README.md) for details.

```bash
# environment (conda-only, no sudo)
bash revision/setup_env.sh && conda activate quishing-rev

# 0) sanity check on synthetic data
python -m revision.smoke_test

# 1) main ablation (4 parameter-matched heads) + dense, multi-seed robustness sweep
python -m revision.run_ablation --data data/raw/qrset --seeds 0,1,2 --epochs 5

# 2) VQC sensitivity (qubits/layers/ansatz)
python -m revision.vqc_sensitivity --data data/raw/qrset --out experiments_vqc_sens
python -m revision.robustness --exp-dir experiments_vqc_sens --data data/raw/qrset --out experiments_vqc_sens/robustness_raw.csv

# 3) noise-aware defense
python -m revision.train --model classic_fc --data data/raw/qrset --noise-aware --noise-sigma-max 0.15
python -m revision.train --model mlp_head   --data data/raw/qrset --noise-aware --noise-sigma-max 0.15

# 4) dataset bias / shortcut audit
python -m revision.audit_dataset --data data/raw/qrset --out experiments_revision/audit
python -m revision.eval_subset  --exp-dir experiments_revision --data data/raw/qrset \
    --subset experiments_revision/audit/length_matched_subset.csv

# regenerate the paper figures from the result CSVs
python -m revision.plot_paper_figures --summary experiments_revision/robustness_raw_summary.csv --out figures_for_paper
```

Each run writes logs to `experiments_*/logs/` and a consolidated `REPORT.md` (clean-performance table, AURC/sigma* summary, accuracy-vs-severity curves, and an automatic ablation verdict).

### Dataset

```bash
kaggle datasets download -d samahsadiq/benign-and-malicious-qr-codes -p data/raw --unzip
# the images live under data/raw/qrset/{benign,malicious}/ (ImageFolder layout)
```

---

## Takeaways

- The brittleness of QR detectors under noise is real and is governed by the **classifier head**: a linear head collapses, a nonlinear bottleneck head degrades gracefully.
- A low-qubit **VQC head gives no dependable advantage** over a parameter-matched classical head, and single-seed/single-config evaluation can manufacture a spurious "quantum advantage." Always compare against parameter-matched baselines across multiple seeds.
- The practical recipe for dependable QR detection is a **classical nonlinear head + noise-aware training**.

## Citation

```bibtex
@article{do2026dependability,
  title={Dependability of Quishing Detection under Realistic Corruption: A Rigorous Robustness Study and a Simple Effective Defense},
  author={Do, Phuc Hao and [Co-Authors]},
  journal={IEEE Transactions on Dependable and Secure Computing (TDSC)},
  year={2026},
  note={Under major revision}
}
```

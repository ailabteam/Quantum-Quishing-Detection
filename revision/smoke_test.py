"""End-to-end smoke test on tiny synthetic data (CPU, no download, no real dataset).

Validates that every model variant trains, saves, reloads, and that the dense
multi-seed robustness sweep runs and produces the summary/metrics CSVs. It does
NOT validate scientific results: the synthetic images are random noise, so
accuracies are meaningless. It only proves the pipeline executes end to end.

Run:  python -m revision.smoke_test
"""

import os
import shutil
import tempfile

import numpy as np
from PIL import Image

from .models import build_model, head_param_count, MODEL_NAMES
from .train import train_model
from . import robustness


def _make_synthetic_dataset(root, per_class=24, size=64, seed=0):
    rng = np.random.default_rng(seed)
    for cls in ["benign", "malicious"]:
        d = os.path.join(root, cls)
        os.makedirs(d, exist_ok=True)
        # give the two classes a faint mean offset so training is not pathological
        offset = 0 if cls == "benign" else 40
        for i in range(per_class):
            arr = np.clip(rng.integers(0, 180, (size, size, 3)) + offset, 0, 255).astype("uint8")
            Image.fromarray(arr).save(os.path.join(d, f"{cls}_{i}.png"))


def main():
    tmp = tempfile.mkdtemp(prefix="qresnet_smoke_")
    data_dir = os.path.join(tmp, "data")
    exp_dir = os.path.join(tmp, "exp")
    try:
        print(f"[smoke] workspace: {tmp}")
        _make_synthetic_dataset(data_dir, per_class=24, size=64)

        # 1) parameter-budget sanity (the fair-comparison claim)
        print("\n[smoke] head parameter counts:")
        for name in MODEL_NAMES:
            m = build_model(name, pretrained=False)
            print(f"   {name:14}: {head_param_count(m)} head params")

        # 2) train every variant, 1 epoch, tiny, CPU, no pretrained download
        for name in MODEL_NAMES:
            train_model(name, data_dir, exp_dir, seed=0, epochs=1, lr=1e-3,
                        batch_size=8, max_per_class=24, num_workers=0,
                        device="cpu", pretrained=False)
        # one noise-aware run to exercise that path (R2-2)
        train_model("qresnet", data_dir, exp_dir, seed=0, epochs=1, lr=1e-3,
                    batch_size=8, max_per_class=24, num_workers=0, device="cpu",
                    pretrained=False, noise_aware=True, noise_sigma_max=0.2)

        # 3) dense robustness sweep, small grid, 2 perturbation seeds
        out_csv = os.path.join(exp_dir, "robustness_raw.csv")
        robustness.run(exp_dir, data_dir, out_csv,
                       threats={"noise", "occlusion"},
                       noise_levels=[0.0, 0.1, 0.2, 0.3],
                       occ_levels=[0, 20, 40],
                       pert_seeds=[0, 1], device="cpu",
                       max_per_class=24, num_workers=0, crit_threshold=70.0)

        for f in ["robustness_raw.csv", "robustness_raw_summary.csv", "robustness_raw_metrics.csv"]:
            p = os.path.join(exp_dir, f)
            assert os.path.exists(p), f"missing output {p}"
            print(f"[smoke] OK -> {f}")
        print("\n[smoke] PASSED: full pipeline executed end to end.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

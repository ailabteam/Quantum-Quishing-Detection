#!/usr/bin/env bash
# Robust environment setup for a conda-only, no-sudo server (RTX 4090).
#
# We avoid `conda env create` with the pytorch-cuda metapackage: on libmamba it
# often stalls on SOLVER_RULE_STRICT_REPO_PRIORITY. Instead we make a minimal
# conda env (python + pip) and install the heavy packages from pip wheels, which
# bundle their own CUDA runtime (no system CUDA, no sudo).
#
# opencv-python-headless is used (not opencv-python) so the QR-decode audit does
# not need a system libGL.
#
# Usage:  bash revision/setup_env.sh   then   conda activate quishing-rev
set -e

ENV=quishing-rev
conda create -n "$ENV" python=3.10 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pennylane numpy pandas scikit-learn matplotlib pillow tqdm opencv-python-headless

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), \
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "Done. Run:  conda activate $ENV  &&  python -m revision.smoke_test"

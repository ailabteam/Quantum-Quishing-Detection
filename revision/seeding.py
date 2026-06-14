"""Reproducibility helpers.

Reviewer 2 flagged that the headline sigma=0.4 gain looks like an outlier and is
not reproducible. The original scripts seeded neither training nor the
perturbation sampling. Every revision experiment routes its randomness through
these helpers so that runs are deterministic and can be repeated across seeds.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for a single run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def perturbation_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """A dedicated RNG for perturbation sampling.

    Keeping perturbation noise on its own generator means the corrupted test set
    for a given (model, seed) pair is identical regardless of model internals,
    so every model is evaluated on exactly the same corrupted images.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen

"""Data loading for revision experiments.

Differences from the original src/data_loader_img.py:
  - No ImageNet normalization in the transform: images stay in [0,1] so that
    perturbations are applied pre-normalization (normalization lives in the model).
  - The 70/10/20 split is class-stratified (the paper claims stratification but
    the original used a plain random_split).
"""

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _stratified_indices(targets, val_split, test_split, seed):
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(targets):
        idx = np.where(targets == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])
    rng.shuffle(train_idx)
    return train_idx, val_idx, test_idx


def load_image_data(root_dir, batch_size=128, val_split=0.1, test_split=0.2,
                    seed=42, num_workers=8, img_size=224, max_per_class=None):
    """Return (train_loader, val_loader, test_loader) over [0,1] images.

    `max_per_class` truncates each class (used by the smoke test for speed).
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # -> [0,1], NO normalization here
    ])
    full = datasets.ImageFolder(root=root_dir, transform=transform)
    targets = [s[1] for s in full.samples]
    print(f"--> [INFO] Found {len(full)} images. Classes: {full.classes}")

    if max_per_class is not None:
        targets_arr = np.asarray(targets)
        keep = []
        for cls in np.unique(targets_arr):
            keep.extend(np.where(targets_arr == cls)[0][:max_per_class])
        full = Subset(full, keep)
        targets = [targets[i] for i in keep]

    tr, va, te = _stratified_indices(targets, val_split, test_split, seed)
    print(f"--> [INFO] Stratified split: train={len(tr)} val={len(va)} test={len(te)}")

    def mk(idx, shuffle):
        return DataLoader(Subset(full, idx), batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return mk(tr, True), mk(va, False), mk(te, False)

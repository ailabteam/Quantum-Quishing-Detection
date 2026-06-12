"""Data loading for revision experiments.

Differences from the original src/data_loader_img.py:
  - No ImageNet normalization in the transform: images stay in [0,1] so that
    perturbations are applied pre-normalization (normalization lives in the model).
  - The 70/10/20 split is class-stratified (the paper claims stratification but
    the original used a plain random_split).
"""

import csv
import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def eval_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # -> [0,1], normalization happens inside the model
    ])


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
    transform = eval_transform(img_size)
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


class _CSVSubset(Dataset):
    """Images listed in a CSV with columns file,class (file relative to root_dir).

    Used to evaluate on the shortcut-controlled (payload-length-matched) subset
    exported by revision.audit_dataset, so we can test whether accuracy persists
    once the obvious payload-length shortcut is removed (Cua 1).
    """

    def __init__(self, root_dir, csv_path, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.items = []
        with open(csv_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.items.append((row["file"], row["class"]))
        classes = sorted({c for _, c in self.items})
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, cls = self.items[i]
        img = Image.open(os.path.join(self.root_dir, rel)).convert("RGB")
        return self.transform(img), self.class_to_idx[cls]


def load_subset_from_csv(root_dir, csv_path, batch_size=128, num_workers=4, img_size=224):
    ds = _CSVSubset(root_dir, csv_path, eval_transform(img_size))
    print(f"--> [INFO] CSV subset: {len(ds)} images, classes {ds.class_to_idx}")
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class QRDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Image shape: (69, 69) -> Add channel dim -> (1, 69, 69)
        image = self.images[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        # Normalize to [0, 1] if not already
        if image.max() > 1.0:
            image = image / 255.0
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

def load_data(data_path, labels_path, test_size=0.2, val_size=0.1, batch_size=32, seed=42):
    """
    Load data from pickle files and split into Train/Val/Test.
    """
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Data files not found at {data_path} or {labels_path}")

    print(f"--> [INFO] Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        X = pickle.load(f)
    with open(labels_path, 'rb') as f:
        y = pickle.load(f)

    # Split 1: Train + Val vs Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Split 2: Train vs Val
    # Val size relative to original (e.g. 0.1 of total)
    # Adjust val_size relative to X_train_val length
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=seed, stratify=y_train_val
    )

    print(f"--> [INFO] Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Create Datasets
    train_dataset = QRDataset(X_train, y_train)
    val_dataset = QRDataset(X_val, y_val)
    test_dataset = QRDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

# Hàm phụ để load dạng numpy cho XGBoost
def load_data_numpy(data_path, labels_path, test_size=0.2, seed=42):
    print(f"--> [INFO] Loading numpy data for XGBoost...")
    with open(data_path, 'rb') as f:
        X = pickle.load(f)
    with open(labels_path, 'rb') as f:
        y = pickle.load(f)
        
    # Flatten images: (N, 69, 69) -> (N, 4761)
    X_flat = X.reshape(X.shape[0], -1) / 255.0
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test

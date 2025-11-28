import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class QRDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Features đã được flatten sẵn (4761 chiều)
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def load_data_dense(data_path, labels_path, batch_size=64): # Batch lớn hơn để ổn định
    with open(data_path, 'rb') as f: X = pickle.load(f)
    with open(labels_path, 'rb') as f: y = pickle.load(f)

    # 1. FLATTEN NGAY LẬP TỨC (Giống XGBoost)
    # X shape: (N, 69, 69) -> (N, 4761)
    X_flat = X.reshape(X.shape[0], -1) 
    
    # 2. Normalize
    X_flat = X_flat / 255.0

    # Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=y_train_val
    )

    train_ds = QRDataset(X_train, y_train)
    val_ds = QRDataset(X_val, y_val)
    test_ds = QRDataset(X_test, y_test)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )

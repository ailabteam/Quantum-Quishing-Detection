import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd

from src.data_loader_img import load_image_data
from src.models.q_resnet import ClassicResNet, QResNet
from src.utils.logger import ExperimentLogger

# CẤU HÌNH
DATA_DIR = "data/raw/kaggle_qr" # Đảm bảo đúng đường dẫn
EPOCHS = 5      # 5 Epochs là đủ cho Transfer Learning (99%)
BATCH_SIZE = 128 # 4090 VRAM 24GB thoải mái chạy batch lớn
LR = 0.0001     # Learning rate nhỏ cho fine-tuning

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return total_loss / len(loader), 100 * correct / total

def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Softmax để lấy xác suất lớp 1 (Malicious)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    auc = roc_auc_score(all_labels, all_probs)
    return acc, auc

def run_training(model_name, model, loaders, device):
    print(f"\n================ TRAINING {model_name} ================")
    train_loader, val_loader, _ = loaders
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    logger = ExperimentLogger(f"kaggle_{model_name}", {"model": model_name, "epochs": EPOCHS})
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_auc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f}")
        
        logger.log_metrics({"epoch": epoch+1, "train_loss": train_loss, "val_acc": val_acc, "val_auc": val_auc})
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{logger.exp_dir}/best_{model_name}.pth")
            
    total_time = time.time() - start_time
    print(f"--> Total Time: {total_time:.1f}s")
    print(f"--> Best Val Acc: {best_acc:.2f}%")
    return f"{logger.exp_dir}/best_{model_name}.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using Device: {device}")
    
    # 1. Load Data
    loaders = load_image_data(DATA_DIR, batch_size=BATCH_SIZE)
    
    # 2. Train Classical Baseline
    resnet = ClassicResNet(pretrained=True).to(device)
    path_classic = run_training("Classic_ResNet18", resnet, loaders, device)
    
    # Giải phóng VRAM
    del resnet
    torch.cuda.empty_cache()
    
    # 3. Train Quantum Proposed
    qresnet = QResNet(n_qubits=4, n_layers=2, pretrained=True).to(device)
    path_quantum = run_training("Quantum_ResNet18", qresnet, loaders, device)
    
    print("\n>>> TRAINING DONE. READY FOR ROBUSTNESS TEST.")
    print(f"Classic Model: {path_classic}")
    print(f"Quantum Model: {path_quantum}")

if __name__ == "__main__":
    main()

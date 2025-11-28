import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Import mới
from src.data_loader import load_data_dense
from src.models.hybrid_quantum import DenseQuantumNet
from src.utils.logger import ExperimentLogger

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    auc = roc_auc_score(all_targets, all_probs)
    acc = accuracy_score(all_targets, all_preds)
    return auc, acc, all_probs, all_preds, all_targets

def main():
    config = {
        "exp_name": "dense_quantum_mlp_v1",
        "data_path": "data/raw/qr_codes_29.pickle",
        "labels_path": "data/raw/qr_codes_29_labels.pickle",
        "batch_size": 64, # Batch lớn ổn định hơn
        "lr": 0.0005,     # LR nhỏ thôi
        "epochs": 25,
        "n_qubits": 4,
        "n_layers": 3
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = ExperimentLogger(config["exp_name"], config)
    
    # Load Data Dense
    train_loader, val_loader, test_loader = load_data_dense(
        config["data_path"], config["labels_path"], batch_size=config["batch_size"]
    )
    
    # Model Dense
    model = DenseQuantumNet(n_qubits=config["n_qubits"], n_layers=config["n_layers"]).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # Thêm Scheduler để giảm LR khi loss đi ngang (Quan trọng!)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_auc = 0.0
    print(f"--> [INFO] Start training Dense-Quantum MLP on {device}...")
    
    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_auc, val_acc, _, _, _ = evaluate(model, val_loader, device)
        
        # Step scheduler
        scheduler.step(val_auc)
        
        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        logger.log_metrics({"epoch": epoch+1, "val_auc": val_auc})
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"{logger.exp_dir}/best_model.pth")

    # TEST
    model.load_state_dict(torch.load(f"{logger.exp_dir}/best_model.pth"))
    test_auc, test_acc, test_probs, test_preds, test_targets = evaluate(model, test_loader, device)
    
    print(f"\n================ FINAL RESULTS (DENSE QUANTUM) ================")
    print(f"Test AUC: {test_auc:.4f}") 
    print(f"Test Acc: {test_acc:.4f}")
    print(f"===============================================================\n")
    
    # Save predictions
    test_indices = np.arange(len(test_targets))
    logger.save_predictions(test_indices, test_targets, test_probs, test_preds)

if __name__ == "__main__":
    main()

import torch
import pickle
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from src.models.hybrid_quantum import DenseQuantumNet # Đảm bảo class này vẫn ở đó

# --- CẤU HÌNH ---
DATA_PATH = "data/raw/qr_codes_29.pickle"
LABEL_PATH = "data/raw/qr_codes_29_labels.pickle"

# Đường dẫn đến model Quantum tốt nhất bạn đã train (SỬA LẠI ĐƯỜNG DẪN NẾU CẦN)
QUANTUM_MODEL_PATH = "experiments/20251128_193757_dense_quantum_mlp_v1/best_model.pth"

def add_noise(images, noise_level):
    """Thêm nhiễu Gaussian vào ảnh"""
    noise = np.random.normal(0, noise_level, images.shape)
    noisy_images = images + noise
    # Clip về [0, 1]
    return np.clip(noisy_images, 0, 1)

def main():
    print("--> [INFO] Loading Data...")
    with open(DATA_PATH, 'rb') as f: X = pickle.load(f)
    with open(LABEL_PATH, 'rb') as f: y = pickle.load(f)
    
    # Chuẩn bị dữ liệu
    X_flat = X.reshape(X.shape[0], -1) / 255.0
    
    # Split (Giữ nguyên random_state để test set giống lúc train)
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 1. TRAIN LẠI XGBOOST (Để có baseline so sánh) ---
    print("--> [INFO] Retraining XGBoost Baseline...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, 
        tree_method='hist', device='cuda', random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # --- 2. LOAD QUANTUM MODEL ---
    print("--> [INFO] Loading Pre-trained Quantum Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_model = DenseQuantumNet(input_dim=4761, n_qubits=4, n_layers=3).to(device)
    try:
        q_model.load_state_dict(torch.load(QUANTUM_MODEL_PATH))
        q_model.eval()
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file model tại {QUANTUM_MODEL_PATH}")
        return

    # --- 3. STRESS TEST LOOP ---
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    xgb_aucs = []
    qnn_aucs = []
    
    print("\n--- STARTING ROBUSTNESS TEST ---")
    print(f"{'Noise':<10} | {'XGBoost AUC':<15} | {'Quantum AUC':<15} | {'Delta':<10}")
    
    for sigma in noise_levels:
        # Tạo dữ liệu nhiễu
        X_test_noisy = add_noise(X_test, sigma)
        
        # XGBoost Predict
        y_xgb_prob = xgb_model.predict_proba(X_test_noisy)[:, 1]
        auc_xgb = roc_auc_score(y_test, y_xgb_prob)
        xgb_aucs.append(auc_xgb)
        
        # Quantum Predict
        X_tensor = torch.tensor(X_test_noisy, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = q_model(X_tensor)
            y_qnn_prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        auc_qnn = roc_auc_score(y_test, y_qnn_prob)
        qnn_aucs.append(auc_qnn)
        
        print(f"{sigma:<10.2f} | {auc_xgb:<15.4f} | {auc_qnn:<15.4f} | {auc_qnn - auc_xgb:<10.4f}")

    # --- 4. PLOT & SAVE ---
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, xgb_aucs, 'r-o', label='XGBoost (Baseline)', linewidth=2)
    plt.plot(noise_levels, qnn_aucs, 'b-s', label='Dense Quantum Net', linewidth=2)
    plt.xlabel('Gaussian Noise Level (Sigma)')
    plt.ylabel('AUC Score')
    plt.title('Robustness Test: Quantum vs Classical under Noise')
    plt.legend()
    plt.grid(True)
    plt.savefig('robustness_comparison.png', dpi=300)
    print("\n--> [INFO] Saved plot to robustness_comparison.png")

if __name__ == "__main__":
    main()

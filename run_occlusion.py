import torch
import pickle
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from src.models.hybrid_quantum import DenseQuantumNet

# --- CẤU HÌNH ---
DATA_PATH = "data/raw/qr_codes_29.pickle"
LABEL_PATH = "data/raw/qr_codes_29_labels.pickle"
# Đảm bảo đường dẫn này đúng tới file best_model.pth của DenseQuantumNet (AUC ~0.88)
QUANTUM_MODEL_PATH = "experiments/20251128_193757_dense_quantum_mlp_v1/best_model.pth"

def apply_occlusion(images, block_size):
    """
    Che khuất một vùng ngẫu nhiên trên ảnh bằng màu xám (0.5) hoặc đen (0).
    block_size: kích thước vùng bị che (ví dụ 10x10)
    """
    occluded_images = images.copy()
    N, H, W = images.shape
    
    # Với mỗi ảnh, chọn vị trí ngẫu nhiên để che
    for i in range(N):
        # Chọn tọa độ góc trên bên trái của khối che
        x = np.random.randint(0, H - block_size)
        y = np.random.randint(0, W - block_size)
        
        # Che vùng đó lại (gán bằng 0 hoặc 0.5 - giá trị trung tính)
        occluded_images[i, x:x+block_size, y:y+block_size] = 0.0 
        
    return occluded_images

def main():
    print("--> [INFO] Loading Data...")
    with open(DATA_PATH, 'rb') as f: X = pickle.load(f)
    with open(LABEL_PATH, 'rb') as f: y = pickle.load(f)
    
    # Chuẩn bị dữ liệu cho XGBoost (Flatten)
    X_flat = X.reshape(X.shape[0], -1) / 255.0
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. TRAIN XGBOOST
    print("--> [INFO] Retraining XGBoost Baseline...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, 
        tree_method='hist', device='cuda', random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # 2. LOAD QUANTUM MODEL
    print("--> [INFO] Loading Quantum Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_model = DenseQuantumNet(input_dim=4761, n_qubits=4, n_layers=3).to(device)
    q_model.load_state_dict(torch.load(QUANTUM_MODEL_PATH))
    q_model.eval()

    # 3. OCCLUSION TEST LOOP
    # Kích thước khối bị che: 0 (sạch), 5x5, 10x10, 15x15, 20x20 pixels
    block_sizes = [0, 5, 8, 12, 15, 20] 
    
    xgb_aucs = []
    qnn_aucs = []
    
    print("\n--- STARTING OCCLUSION ROBUSTNESS TEST ---")
    print(f"{'Block Size':<10} | {'XGBoost AUC':<15} | {'Quantum AUC':<15} | {'Delta':<10}")
    
    # X_test gốc (để apply occlusion lên dạng ảnh 69x69 trước khi flatten)
    X_test_img = X_test.reshape(-1, 69, 69) 
    
    for bs in block_sizes:
        if bs == 0:
            X_test_occ_flat = X_test
        else:
            X_test_occ = apply_occlusion(X_test_img, bs)
            X_test_occ_flat = X_test_occ.reshape(X_test_occ.shape[0], -1) # Flatten lại
        
        # XGBoost Predict
        y_xgb_prob = xgb_model.predict_proba(X_test_occ_flat)[:, 1]
        auc_xgb = roc_auc_score(y_test, y_xgb_prob)
        xgb_aucs.append(auc_xgb)
        
        # Quantum Predict
        X_tensor = torch.tensor(X_test_occ_flat, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = q_model(X_tensor)
            y_qnn_prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        auc_qnn = roc_auc_score(y_test, y_qnn_prob)
        qnn_aucs.append(auc_qnn)
        
        # Delta: Quantum - XGBoost (Dương là Quantum thắng)
        delta = auc_qnn - auc_xgb
        print(f"{bs:<10} | {auc_xgb:<15.4f} | {auc_qnn:<15.4f} | {delta:+.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(block_sizes, xgb_aucs, 'r-o', label='XGBoost', linewidth=2)
    plt.plot(block_sizes, qnn_aucs, 'b-s', label='Dense Quantum Net', linewidth=2)
    plt.xlabel('Occlusion Block Size (pixels)')
    plt.ylabel('AUC Score')
    plt.title('Robustness: Occlusion Attack')
    plt.legend()
    plt.grid(True)
    plt.savefig('occlusion_robustness.png')
    print("\n--> [INFO] Saved plot to occlusion_robustness.png")

if __name__ == "__main__":
    main()

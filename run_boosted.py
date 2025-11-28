import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

def main():
    print("--> [INFO] Loading Data...")
    
    # 1. Load ảnh gốc (Pixel features)
    with open("data/raw/qr_codes_29.pickle", 'rb') as f:
        X_pixels = pickle.load(f)
        X_pixels = X_pixels.reshape(X_pixels.shape[0], -1) / 255.0 # Flatten (N, 4761)
        
    # 2. Load Quantum Features (vừa tạo ở bước 1)
    try:
        X_quantum = np.load("data/processed/qr_quantum_features.npy")
    except FileNotFoundError:
        print("LỖI: Chưa chạy file gen_features.py!")
        return
        
    # 3. Load Labels
    with open("data/raw/qr_codes_29_labels.pickle", 'rb') as f:
        y = pickle.load(f)
        
    # 4. KẾT HỢP (CONCATENATE)
    # XGBoost sẽ nhìn thấy: [4761 pixels] + [12 quantum features]
    X_combined = np.hstack((X_pixels, X_quantum))
    
    print(f"--> [INFO] Combined Data Shape: {X_combined.shape}")
    
    # 5. Split (Giống hệt baseline để so sánh công bằng)
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Train XGBoost (Dùng lại tham số tốt nhất của baseline)
    print("--> [INFO] Training Quantum-Enhanced XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.05,
        tree_method='hist',
        device='cuda', # Dùng GPU
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 7. Evaluate
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_probs)
    acc = accuracy_score(y_test, y_preds)
    
    print(f"\n================ RESULT ================")
    print(f"Baseline AUC (Paper): ~0.9138")
    print(f"Quantum-Enhanced AUC: {auc:.5f}")
    print(f"Accuracy: {acc:.5f}")
    
    if auc > 0.9138:
        print("\n>>> VICTORY! WE BEAT THE BASELINE! <<<")
    else:
        print("\n>>> Result is comparable. Dataset might be saturated.")

if __name__ == "__main__":
    main()

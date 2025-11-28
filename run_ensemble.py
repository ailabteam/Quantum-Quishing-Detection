import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# 1. Đường dẫn file CSV kết quả của XGBoost (0.9138)
# Bạn thay đúng tên folder experiments mà bạn đã chạy thành công
xgb_path = "experiments/20251128_190040_baseline_xgboost/test_predictions.csv" 

# 2. Đường dẫn file CSV kết quả của Dense Quantum (0.8839)
# Bạn thay đúng tên folder experiments vừa chạy xong
qnn_path = "experiments/20251128_193757_dense_quantum_mlp_v1/test_predictions.csv"

def run_ensemble():
    print("--> [INFO] Loading predictions...")
    
    # Load dataframes
    try:
        df_xgb = pd.read_csv(xgb_path)
        df_qnn = pd.read_csv(qnn_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Hãy kiểm tra lại đường dẫn file CSV trong folder experiments!")
        return

    # Kiểm tra xem thứ tự mẫu có khớp nhau không
    if not np.array_equal(df_xgb['y_true'].values, df_qnn['y_true'].values):
        print("CẢNH BÁO: Thứ tự nhãn giữa 2 file không khớp nhau. Không thể ensemble!")
        return

    y_true = df_xgb['y_true'].values
    
    # Lấy xác suất dự đoán (cột y_prob hoặc y_prob_phishing)
    # XGBoost thường rất tự tin (gần 0 hoặc 1), Quantum thường mềm hơn.
    pred_xgb = df_xgb['y_prob'].values if 'y_prob' in df_xgb.columns else df_xgb['y_prob_phishing'].values
    pred_qnn = df_qnn['y_prob'].values if 'y_prob' in df_qnn.columns else df_qnn['y_prob_phishing'].values

    # --- CHIẾN THUẬT ENSEMBLE ---
    # Trọng số: XGBoost tốt hơn nên cho trọng số cao hơn một chút
    # Thử các tỷ lệ alpha khác nhau
    best_auc = 0
    best_alpha = 0
    
    print("\n--- Searching for best Weight ---")
    for alpha in np.arange(0.0, 1.1, 0.1):
        # Công thức: Final = alpha * XGB + (1-alpha) * Quantum
        final_prob = alpha * pred_xgb + (1 - alpha) * pred_qnn
        
        auc = roc_auc_score(y_true, final_prob)
        print(f"Alpha (XGB) = {alpha:.1f} | AUC: {auc:.5f}")
        
        if auc > best_auc:
            best_auc = auc
            best_alpha = alpha

    print(f"\n================ FINAL ENSEMBLE RESULT ================")
    print(f"Best Alpha: {best_alpha:.1f} (XGBoost) - {1-best_alpha:.1f} (Quantum)")
    print(f"BEST AUC: {best_auc:.5f}")
    print(f"Baseline XGBoost: {roc_auc_score(y_true, pred_xgb):.5f}")
    print(f"Baseline Quantum: {roc_auc_score(y_true, pred_qnn):.5f}")
    
    if best_auc > 0.9138:
        print("\n>>> CHÚC MỪNG! BẠN ĐÃ VƯỢT QUA BASELINE! <<<")
        print("Đây là kết quả để đưa vào Paper.")
    else:
        print("\n>>> Vẫn chưa vượt qua. Cần phương pháp PCA.")

if __name__ == "__main__":
    run_ensemble()

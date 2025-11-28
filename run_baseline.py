import yaml
import argparse
import numpy as np
from src.data_loader import load_data_numpy
from src.models.baseline import XGBoostBaseline
from src.utils.logger import ExperimentLogger

def main():
    # 1. Load config
    # Ở đây fix cứng cho nhanh, sau này load từ file yaml
    config = {
        "model_name": "xgboost_baseline",
        "data_path": "data/raw/qr_codes_29.pickle",
        "labels_path": "data/raw/qr_codes_29_labels.pickle",
        "test_size": 0.2,
        "seed": 42,
        "params": {
            "n_estimators": 200,
            "max_depth": 10, 
            "learning_rate": 0.05,
            "device": "cuda"
        }
    }

    # 2. Init Logger
    logger = ExperimentLogger("baseline_xgboost", config)

    # 3. Load Data
    X_train, X_test, y_train, y_test = load_data_numpy(
        config["data_path"], 
        config["labels_path"], 
        test_size=config["test_size"], 
        seed=config["seed"]
    )
    
    # 4. Train Model
    model = XGBoostBaseline(config["params"])
    model.train(X_train, y_train, X_test, y_test) # Dùng test làm val để monitor luôn cho tiện

    # 5. Evaluate
    results = model.evaluate(X_test, y_test)
    
    print(f"\n================ RESULTS ================")
    print(f"AUC: {results['auc']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"=========================================\n")

    # 6. Log Metrics & Save Predictions
    logger.log_metrics({
        "epoch": "final",
        "train_loss": 0, # XGBoost sklearn API khó lấy loss từng epoch ra ngoài
        "val_auc": results['auc'],
        "val_acc": results['accuracy']
    })
    
    # Tạo indices giả (0, 1, 2...) cho test set
    test_indices = np.arange(len(y_test))
    logger.save_predictions(
        test_indices, 
        y_test, 
        results['y_probs'], 
        results['y_preds']
    )

if __name__ == "__main__":
    main()

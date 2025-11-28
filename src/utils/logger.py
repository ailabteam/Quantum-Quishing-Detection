import os
import csv
import json
import datetime
import pandas as pd

class ExperimentLogger:
    def __init__(self, exp_name, config, base_dir="experiments"):
        """
        Khởi tạo logger.
        - Tạo folder riêng cho mỗi lần chạy: experiments/YYYYMMDD_HHMMSS_exp_name
        - Lưu config.json
        - Tạo file CSV log training
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, f"{timestamp}_{exp_name}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        print(f"--> [INFO] Experiment Output Directory: {self.exp_dir}")

        # 1. Lưu Config
        with open(os.path.join(self.exp_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # 2. Tạo file Training Log (CSV)
        self.log_file = os.path.join(self.exp_dir, "training_log.csv")
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            # Header mặc định, có thể thêm cột tùy model
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc", "lr", "time"])

        # 3. Path file dự đoán (sẽ ghi sau)
        self.pred_file = os.path.join(self.exp_dir, "test_predictions.csv")

    def log_metrics(self, metrics_dict):
        """
        Ghi một dòng vào training_log.csv
        metrics_dict: {"epoch": 1, "train_loss": 0.5, ...}
        """
        # Đọc header để biết thứ tự cột
        with open(self.log_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
        
        row = [metrics_dict.get(col, "") for col in header]
        
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def save_predictions(self, indices, y_true, y_probs, y_preds):
        """
        Lưu kết quả dự đoán ra CSV để vẽ ROC sau này.
        """
        df = pd.DataFrame({
            "index": indices,
            "y_true": y_true,
            "y_prob_phishing": y_probs,
            "y_pred_label": y_preds
        })
        df.to_csv(self.pred_file, index=False)
        print(f"--> [INFO] Predictions saved to {self.pred_file}")


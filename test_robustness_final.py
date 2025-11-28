import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader_img import load_image_data
from src.models.q_resnet import ClassicResNet, QResNet

# ================= CẤU HÌNH (CẬP NHẬT SAU KHI TRAIN XONG) =================
DATA_DIR = "data/raw/kaggle_qr"
BATCH_SIZE = 64

# Bạn hãy thay thế 2 đường dẫn dưới đây bằng đường dẫn thực tế sau khi train xong
# Ví dụ: experiments/20251128_204521_kaggle_Classic_ResNet18/best_Classic_ResNet18.pth
PATH_CLASSIC = "experiments/20251128_204521_kaggle_Classic_ResNet18/best_Classic_ResNet18.pth"
PATH_QUANTUM = "experiments/20251128_205147_kaggle_Quantum_ResNet18/best_Quantum_ResNet18.pth"
# ==========================================================================

def add_gaussian_noise(tensor, std):
    if std == 0: return tensor
    return tensor + torch.randn(tensor.size()) * std

def apply_occlusion(tensor, block_size):
    if block_size == 0: return tensor
    occ_tensor = tensor.clone()
    _, _, h, w = occ_tensor.shape
    # Che ngẫu nhiên 1 vùng trên mỗi ảnh trong batch
    for i in range(occ_tensor.shape[0]):
        x = np.random.randint(0, h - block_size)
        y = np.random.randint(0, w - block_size)
        occ_tensor[i, :, x:x+block_size, y:y+block_size] = 0 # Che màu đen
    return occ_tensor

def run_test(model, loader, device, attack_type, levels):
    model.eval()
    results = []

    print(f"--> Testing {attack_type}...")

    for level in levels:
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                # Apply Attack
                if attack_type == "Noise":
                    inputs = add_gaussian_noise(inputs, std=level)
                elif attack_type == "Occlusion":
                    inputs = apply_occlusion(inputs, block_size=int(level))

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"   Level {level}: Accuracy = {acc:.2f}%")
        results.append(acc)
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data (Chỉ cần test set)
    _, _, test_loader = load_image_data(DATA_DIR, batch_size=BATCH_SIZE)

    # 2. Load Models
    print("\n[1] Loading Classical Model...")
    classic = ClassicResNet(pretrained=False).to(device)
    classic.load_state_dict(torch.load(PATH_CLASSIC))

    print("\n[2] Loading Quantum Model...")
    quantum = QResNet(n_qubits=4, n_layers=2, pretrained=False).to(device)
    quantum.load_state_dict(torch.load(PATH_QUANTUM))

    # 3. Define Experiments
    # Noise levels (Standard Deviation)
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Occlusion levels (Pixel size of black block)
    occlusion_levels = [0, 20, 40, 60, 80, 100]

    # 4. Run & Collect Data
    data_records = []

    # --- Experiment A: Gaussian Noise ---
    print("\n=== EXPERIMENT A: GAUSSIAN NOISE ===")
    acc_c_noise = run_test(classic, test_loader, device, "Noise", noise_levels)
    acc_q_noise = run_test(quantum, test_loader, device, "Noise", noise_levels)

    for i, lvl in enumerate(noise_levels):
        data_records.append({"Model": "Classic_ResNet", "Attack": "Gaussian_Noise", "Severity": lvl, "Accuracy": acc_c_noise[i]})
        data_records.append({"Model": "Quantum_ResNet", "Attack": "Gaussian_Noise", "Severity": lvl, "Accuracy": acc_q_noise[i]})

    # --- Experiment B: Occlusion ---
    print("\n=== EXPERIMENT B: OCCLUSION ===")
    acc_c_occ = run_test(classic, test_loader, device, "Occlusion", occlusion_levels)
    acc_q_occ = run_test(quantum, test_loader, device, "Occlusion", occlusion_levels)

    for i, lvl in enumerate(occlusion_levels):
        data_records.append({"Model": "Classic_ResNet", "Attack": "Occlusion", "Severity": lvl, "Accuracy": acc_c_occ[i]})
        data_records.append({"Model": "Quantum_ResNet", "Attack": "Occlusion", "Severity": lvl, "Accuracy": acc_q_occ[i]})

    # 5. Save to CSV
    df = pd.DataFrame(data_records)
    csv_filename = "final_paper_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n[SUCCESS] Results saved to {csv_filename}")
    print("Use this CSV to create tables and plots for your paper.")

if __name__ == "__main__":
    main()

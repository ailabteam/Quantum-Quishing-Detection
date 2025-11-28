import pickle
import numpy as np
import pennylane as qml
import torch
import cv2
from tqdm import tqdm

# --- CẤU HÌNH ---
DATA_PATH = "data/raw/qr_codes_29.pickle"
SAVE_PATH = "data/processed/qr_quantum_features.npy"
N_QUBITS = 4  # Số qubit nhỏ để chạy nhanh
IMG_SIZE = 14 # Resize ảnh về 14x14

# --- 1. ĐỊNH NGHĨA MẠCH LƯỢNG TỬ ---
# Chúng ta dùng mạch này như một "Bộ lọc" (Filter) để trích xuất đặc trưng
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_filter(inputs):
    # inputs là 4 giá trị pixel (2x2 block)
    # Mã hóa dữ liệu vào góc quay
    qml.AngleEmbedding(inputs * np.pi, wires=range(N_QUBITS))
    
    # Tạo sự vướng víu (Entanglement) để bắt mối tương quan phi tuyến
    qml.BasicEntanglerLayers(weights=[[0.1]*N_QUBITS]*2, wires=range(N_QUBITS))
    
    # Đo đạc thống kê (Expectation Value)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(N_QUBITS)]

def extract_q_features(images):
    q_features_list = []
    
    print(f"--> [INFO] Start generating Quantum Features for {len(images)} images...")
    
    # Dùng tqdm để hiện thanh tiến trình
    for img in tqdm(images):
        # 1. Resize về kích thước nhỏ (14x14) để xử lý nhanh
        # img gốc là (69, 69)
        img_small = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img_small = img_small / 255.0 # Normalize [0, 1]
        
        # 2. Quét mạch lượng tử qua ảnh (Sliding Window)
        # Chúng ta chia ảnh 14x14 thành các block 2x2. Tổng cộng 7x7 = 49 blocks.
        # Mỗi block ra 4 giá trị lượng tử -> Flatten lại.
        
        vals = []
        # Chỉ lấy đại diện 4 vùng quan trọng (Góc trái trên, phải trên, trái dưới, giữa)
        # Để tiết kiệm thời gian, ta không quét hết.
        
        # Vùng 1: Top-Left (Finder Pattern) -> Block 2x2 tại (1,1)
        block1 = img_small[1:3, 1:3].flatten()
        
        # Vùng 2: Center (Data) -> Block 2x2 tại giữa
        mid = IMG_SIZE // 2
        block2 = img_small[mid:mid+2, mid:mid+2].flatten()
        
        # Vùng 3: Random Noise Area (Góc dưới phải)
        block3 = img_small[-3:-1, -3:-1].flatten()
        
        # Chuyển sang Tensor
        inputs = torch.tensor([block1, block2, block3], dtype=torch.float32)
        
        # Chạy Quantum Circuit cho 3 vùng này
        # output shape: (3 vùng, 4 giá trị đo)
        with torch.no_grad():
            q_out1 = quantum_filter(inputs[0])
            q_out2 = quantum_filter(inputs[1])
            q_out3 = quantum_filter(inputs[2])
            
        # Ghép lại thành vector đặc trưng (12 chiều)
        feat_vec = np.concatenate([q_out1, q_out2, q_out3])
        q_features_list.append(feat_vec)

    return np.array(q_features_list)

def main():
    # Load data gốc
    with open(DATA_PATH, 'rb') as f:
        X = pickle.load(f) # (N, 69, 69)
        
    # Extract
    X_quantum = extract_q_features(X)
    
    print(f"--> [INFO] Features generated. Shape: {X_quantum.shape}")
    print(f"--> [INFO] Saving to {SAVE_PATH}...")
    
    # Lưu lại để dùng cho bước sau
    np.save(SAVE_PATH, X_quantum)

if __name__ == "__main__":
    main()

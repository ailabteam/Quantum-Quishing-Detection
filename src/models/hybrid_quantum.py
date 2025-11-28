import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

class DenseQuantumNet(nn.Module):
    def __init__(self, input_dim=4761, n_qubits=4, n_layers=2):
        super(DenseQuantumNet, self).__init__()
        
        # PHẦN CỔ ĐIỂN: Nén dữ liệu nhưng giữ thông tin vị trí
        self.classical_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4), # Dropout cao để chống học vẹt
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, n_qubits), # Nén xuống 4 chiều cho Quantum
            nn.Tanh() # Tanh để đưa về [-1, 1] (tốt cho góc quay)
        )
        
        # PHẦN LƯỢNG TỬ
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # Encoding: Angle Embedding (Nhúng góc)
            # Nhân với PI để phủ hết vòng tròn lượng tử
            qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits))
            
            # Layer học: Strong Entanglement (Mạnh hơn Basic)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        
        # StronglyEntanglingLayers cần shape (n_layers, n_qubits, 3)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Output Final
        self.fc_final = nn.Linear(n_qubits, 2)

    def forward(self, x):
        # x: [Batch, 4761]
        pre_quantum = self.classical_net(x)
        q_out = self.quantum_layer(pre_quantum)
        return self.fc_final(q_out)

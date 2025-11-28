import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml
import numpy as np

# --- 1. BASELINE: CLASSICAL RESNET-18 ---
class ClassicResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ClassicResNet, self).__init__()
        # Load pre-trained ResNet18
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Thay lớp cuối cùng (FC) để output ra 2 lớp (Benign/Malicious)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.base_model(x)

# --- 2. PROPOSED: QUANTUM RESNET ---
class QResNet(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, pretrained=True):
        super(QResNet, self).__init__()

        # Backbone
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity() # Bỏ lớp FC cũ

        # Cầu nối: Giảm chiều từ 512 -> 4 (số qubits)
        self.pre_quantum = nn.Linear(in_features, n_qubits)

        # Quantum Layer
        # Dùng lightning.gpu cho RTX 4090
        try:
            dev = qml.device("default.qubit", wires=n_qubits) # Dùng CPU cho an toàn
        except:
            dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # Encoding: Dùng tanh ép về [-1,1] rồi nhân PI -> [-pi, pi]
            qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits))
            # Variational Layer: Strong Entanglement
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # Measurement
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # Classifier cuối
        self.post_quantum = nn.Linear(n_qubits, 2)

    def forward(self, x):
        features = self.base_model(x) # [Batch, 512]

        # Nén xuống 4 chiều
        q_in = torch.tanh(self.pre_quantum(features))

        # Qua mạch lượng tử
        q_out = self.quantum_layer(q_in)

        # Phân loại
        return self.post_quantum(q_out)

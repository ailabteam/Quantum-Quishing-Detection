"""Model zoo for the ablation requested by both reviewers.

The central question (R1-2, R2-3) is whether the robustness gain comes from the
Variational Quantum Circuit (VQC) or merely from the classical 512->4 bottleneck
plus the tanh nonlinearity. To isolate this we keep an identical ResNet-18
backbone and vary ONLY the classification head:

  classic_fc        : FC 512->2                       (original linear baseline)
  bottleneck_fc     : Linear 512->4 -> tanh -> 4->2   (bottleneck + tanh, NO VQC)
  mlp_head          : Linear 512->4 -> tanh -> 4->4 -> ReLU -> 4->2  (classical nonlinear, param-matched)
  qresnet           : Linear 512->4 -> tanh -> VQC(4q) -> 4->2       (proposed)

Head trainable-parameter budgets (for the fair-comparison table):
  classic_fc     : 512*2 + 2                         = 1026
  bottleneck_fc  : (512*4+4) + (4*2+2)               = 2062
  mlp_head       : (512*4+4) + (4*4+4) + (4*2+2)     = 2082
  qresnet        : (512*4+4) + (2*4*3) + (4*2+2)     = 2086   (VQC adds 24 params)

So bottleneck_fc / mlp_head / qresnet are parameter-matched to within ~24 params;
any robustness difference between them is attributable to the VQC, not capacity.

Normalization is applied INSIDE the wrapper (NormalizedModel) so that callers can
feed raw [0,1] images and apply perturbations before normalization.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

from .perturbations import IMAGENET_MEAN, IMAGENET_STD


def _backbone(pretrained: bool = True) -> nn.Module:
    weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
    net = tvm.resnet18(weights=weights)
    net.fc = nn.Identity()  # expose the 512-d feature vector
    return net


class _Head(nn.Module):
    """Marker base so heads can advertise their input feature dim."""

    in_features = 512


class ClassicFCHead(_Head):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 2)

    def forward(self, f):
        return self.fc(f)


class BottleneckFCHead(_Head):
    """512->4 -> tanh -> 4->2. Removes the VQC but keeps bottleneck + tanh."""

    def __init__(self, bottleneck=4):
        super().__init__()
        self.pre = nn.Linear(512, bottleneck)
        self.post = nn.Linear(bottleneck, 2)

    def forward(self, f):
        x = torch.tanh(self.pre(f))
        return self.post(x)


class MLPHead(_Head):
    """Classical nonlinear head, parameter-matched to the VQC head."""

    def __init__(self, bottleneck=4, hidden=4):
        super().__init__()
        self.pre = nn.Linear(512, bottleneck)
        self.hidden = nn.Linear(bottleneck, hidden)
        self.post = nn.Linear(hidden, 2)

    def forward(self, f):
        x = torch.tanh(self.pre(f))
        x = torch.relu(self.hidden(x))
        return self.post(x)


class VQCHead(_Head):
    """Quantum Bridge (512->n_qubits, tanh) -> VQC -> n_qubits->2.

    Uses R_y angle encoding to match Eq. (2) of the paper (the original code used
    AngleEmbedding's default RX). Strongly entangling ansatz, configurable depth.
    """

    def __init__(self, n_qubits=4, n_layers=2, rotation="Y", ansatz="strong"):
        super().__init__()
        import pennylane as qml

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        self.pre = nn.Linear(512, n_qubits)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation=rotation)
            if ansatz == "strong":
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            elif ansatz == "basic":
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            else:
                raise ValueError(f"unknown ansatz: {ansatz}")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        if ansatz == "strong":
            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        else:  # basic entangler: one rotation angle per qubit per layer
            weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.post = nn.Linear(n_qubits, 2)

    def forward(self, f):
        x = torch.tanh(self.pre(f))
        q = self.qlayer(x)
        return self.post(q)


class NormalizedModel(nn.Module):
    """Wrap backbone + head and normalize raw [0,1] inputs inside forward().

    This lets the evaluation pipeline corrupt images in [0,1] space before
    normalization, instead of corrupting already-normalized tensors.
    """

    def __init__(self, head: _Head, pretrained: bool = True):
        super().__init__()
        self.backbone = _backbone(pretrained)
        self.head = head
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.head(self.backbone(x))


_HEADS = {
    "classic_fc": lambda **kw: ClassicFCHead(),
    "bottleneck_fc": lambda **kw: BottleneckFCHead(bottleneck=kw.get("n_qubits", 4)),
    "mlp_head": lambda **kw: MLPHead(bottleneck=kw.get("n_qubits", 4), hidden=kw.get("hidden", 4)),
    "qresnet": lambda **kw: VQCHead(
        n_qubits=kw.get("n_qubits", 4),
        n_layers=kw.get("n_layers", 2),
        rotation=kw.get("rotation", "Y"),
        ansatz=kw.get("ansatz", "strong"),
    ),
}


def build_model(name: str, pretrained: bool = True, **kw) -> NormalizedModel:
    if name not in _HEADS:
        raise ValueError(f"unknown model '{name}'. choices: {list(_HEADS)}")
    return NormalizedModel(_HEADS[name](**kw), pretrained=pretrained)


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def head_param_count(model: NormalizedModel) -> int:
    return count_params(model.head)


MODEL_NAMES = list(_HEADS)

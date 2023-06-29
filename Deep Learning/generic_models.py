import torch
import torch.nn as nn
from typing import List, Tuple


class BinaryClassifierHead(nn.Module):
    """
    Final output layer used for binary classification tasks
    In dimension is the out dimension of the previous layer

    EG if ffnn with architecture 100x1000, 1000x100 is the
    previous layer the classifier head expects in_dim=100
    Output of this layer is the raw projection to 1D subspace
    """

    def __init__(
        self,
        in_dim,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.01)

    def forward(self, x):
        return self.proj(x)


class ClassifierHead(nn.Module):
    """
    Final output layer used for classification tasks
    In dimension is the out dimension of the previous layer
    EG if ffnn with architecture 100x1000, 1000x100 is the
    previous layer the classifier head expects in_dim=100

    supported activations
        "softmax: softmax activation
        "sigmoid": sigmoid activation [default]
        "": Identity
    """

    def __init__(self, in_dim, num_classes, act="sigmoid") -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, num_classes)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.01)

        if act == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act == "sigmoid":
            self.act = torch.sigmoid()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        val = self.act(x)
        return val


class MLP(nn.Module):
    """
    classic ffnn

    eg mlp(architecture = "(100,500), (500,100), ...", act="relu")

    supported activations between linear layers
        "relu": rectified linear unit activation
        "leakyrelu": leaky rectified linear unit activation
        "tanh": hyperbolic tangent
        "": Gaussian Error Linear Unit [default]
    """

    def __init__(self, architecture: List[Tuple], act="Gelu", dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU()
        elif act == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.GELU()

        self.layers = nn.ModuleList(self._parse_architecture(architecture))

    def _parse_architecture(self, architecture):
        tmp = [nn.Linear(*x) for x in architecture]
        layers = []
        for linear in tmp:
            layers.extend([linear, self.act, self.dropout])
        return layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

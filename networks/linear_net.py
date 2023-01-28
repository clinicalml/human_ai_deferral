import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Linear_net_sig(nn.Module):
    """
    Linear binary classifier
    """

    def __init__(self, input_dim, out_dim=1):
        super(Linear_net_sig, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


class LinearNetDefer(nn.Module):
    """
    Linear Classifier with out+1 units and no softmax
    """

    def __init__(self, input_dim, out_dim):
        super(LinearNetDefer, self).__init__()
        # an affine operation: y = Wx + b
        self.fc = nn.Linear(input_dim, out_dim + 1)

    def forward(self, x):
        out = self.fc(x)
        return out


class LinearNet(nn.Module):
    """
    Linear Classifier with out units and no softmax
    """

    def __init__(self, input_dim, out_dim):
        super(LinearNet, self).__init__()
        # an affine operation: y = Wx + b
        self.fc = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        out = self.fc(x)
        return out

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class NonLinearNet(nn.Module):
    """
    NonLinear Classifier
    """

    def __init__(self, input_dim, out_dim):
        super(NonLinearNet, self).__init__()
        self.fc_all = nn.Sequential(
            nn.Linear(input_dim, 250),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, out_dim),
        )
        # add dropout

    def forward(self, x):

        out = self.fc_all(x)

        return out

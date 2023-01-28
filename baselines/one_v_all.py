import copy
import math
from pyexpat import model
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
import shutil
import time
import torch.utils.data as data
import sys
import logging
from tqdm import tqdm

sys.path.append("..")
from helpers.utils import *
from helpers.metrics import *
from .basemethod import BaseMethod, BaseSurrogateMethod

eps_cst = 1e-8


class OVASurrogate(BaseSurrogateMethod):
    """Method of OvA surrogate from Calibrated Learning to Defer with One-vs-All Classifiers https://proceedings.mlr.press/v162/verma22c/verma22c.pdf"""

    # from https://github.com/rajevv/OvA-L2D/blob/main/losses/losses.py
    def LogisticLossOVA(self, outputs, y):
        outputs[torch.where(outputs == 0.0)] = (-1 * y) * (-1 * np.inf)
        l = torch.log2(1 + torch.exp((-1 * y) * outputs + eps_cst) + eps_cst)
        return l


    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        """
        outputs: network outputs
        m: cost of deferring to expert cost of classifier predicting  hum_preds == target
        labels: target
        """
        human_correct = (hum_preds == data_y).float()
        human_correct = torch.tensor(human_correct).to(self.device)
        
        batch_size = outputs.size()[0]
        l1 = self.LogisticLossOVA(outputs[range(batch_size), data_y], 1)
        l2 = torch.sum(
            self.LogisticLossOVA(outputs[:, :-1], -1), dim=1
        ) - self.LogisticLossOVA(outputs[range(batch_size), data_y], -1)
        l3 = self.LogisticLossOVA(outputs[range(batch_size), -1], -1)
        l4 = self.LogisticLossOVA(outputs[range(batch_size), -1], 1)

        l5 = human_correct * (l4 - l3)

        l = l1 + l2 + l3 + l5

        return torch.mean(l)
    
    
    

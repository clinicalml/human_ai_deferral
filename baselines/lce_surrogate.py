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
import pickle
import logging
from tqdm import tqdm

sys.path.append("..")
from helpers.utils import *
from helpers.metrics import *
from .basemethod import BaseMethod, BaseSurrogateMethod

eps_cst = 1e-8


class LceSurrogate(BaseSurrogateMethod):
    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        """
        Implmentation of L_{CE}^{\alpha}
        """
        outputs = F.softmax(outputs, dim=1)
        human_correct = (hum_preds == data_y).float()
        m2 = self.alpha * human_correct + (1 - human_correct)
        human_correct = torch.tensor(human_correct).to(self.device)
        m2 = torch.tensor(m2).to(self.device)
        batch_size = outputs.size()[0]  # batch_size
        loss = -human_correct * torch.log2(
            outputs[range(batch_size), -1] + eps_cst
        ) - m2 * torch.log2(
            outputs[range(batch_size), data_y] + eps_cst
        )  # pick the values corresponding to the labels
        return torch.sum(loss) / batch_size

    # fit with hyperparameter tuning over alpha
    def fit_hyperparam(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        scheduler=None,
        verbose=True,
        test_interval=5,
    ):
        alpha_grid = [0, 0.5, 1]
        best_alpha = 0
        best_acc = 0
        model_dict = copy.deepcopy(self.model.state_dict())
        for alpha in tqdm(alpha_grid):
            self.alpha = alpha
            self.model.load_state_dict(model_dict)
            self.fit(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs,
                optimizer = optimizer,
                lr = lr,
                verbose = verbose,
                test_interval = test_interval,
                scheduler = scheduler,
            )["system_acc"]
            accuracy = compute_deferral_metrics(self.test(dataloader_val))["system_acc"]
            logging.info(f"alpha: {alpha}, accuracy: {accuracy}")
            if accuracy > best_acc:
                best_acc = accuracy
                best_alpha = alpha
        self.alpha = best_alpha
        self.model.load_state_dict(model_dict)
        fit = self.fit(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs,
                optimizer = optimizer,
                lr = lr,
                verbose = verbose,
                test_interval = test_interval,
                scheduler = scheduler,
            )
        test_metrics = compute_deferral_metrics(self.test(dataloader_test))
        return test_metrics

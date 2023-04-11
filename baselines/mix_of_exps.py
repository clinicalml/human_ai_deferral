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
from .basemethod import BaseMethod

eps_cst = 1e-8

# This really doesn't work well on no bencmark,



class MixtureOfExperts(BaseMethod):
    """Implementation of Madras et al., 2018"""

    def __init__(self, model, device, plotting_interval=100):
        self.plotting_interval = plotting_interval
        self.model = model
        self.device = device

    def mixtures_of_experts_loss(self, outputs, human_is_correct, labels):
        """
        Implmentation of Mixtures of Experts loss from Madras et al., 2018
        """

        batch_size = outputs.size()[0]  # batch_size
        human_loss = torch.cuda.FloatTensor(1 - human_is_correct * 1.0).to(self.device)
        rejector_probability = torch.sigmoid(
            outputs[:, -1] + eps_cst
        )  # probability of rejection
        outputs_class = F.softmax(outputs[:, :-1], dim=1)
        classifier_loss = -torch.log2(
            outputs_class[range(batch_size), labels] + eps_cst
        )
        loss = (
            classifier_loss * (1 - rejector_probability)
            + human_loss * rejector_probability
        )
        return torch.sum(loss) / batch_size

    def fit_epoch(self, dataloader, optimizer, verbose=True, epoch=1):
        """
        Fit the model for one epoch
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        self.model.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            m = (hum_preds == data_y) * 1
            m = torch.tensor(m).to(self.device)
            outputs = self.model(data_x)
            # apply softmax to outputs

            loss = self.mixtures_of_experts_loss(outputs, m, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning(f"NAN LOSS")
                break
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        verbose=True,
        test_interval=5,
        scheduler=None,
    ):
        optimizer = optimizer(self.model.parameters(), lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer, len(dataloader_train)*epochs) 
        for epoch in tqdm(range(epochs)):
            self.fit_epoch(dataloader_train, optimizer, verbose, epoch)
            if verbose and epoch % test_interval == 0 and epoch > 1:
                data_test = self.test(dataloader_val)
                logging.info(compute_deferral_metrics(data_test))
            if scheduler is not None:
                scheduler.step()
        final_test = self.test(dataloader_test)
        return compute_deferral_metrics(final_test)

    def test(self, dataloader):
        defers_all = []
        truths_all = []
        hum_preds_all = []
        rej_score = []
        predictions_all = []  # classifier only
        self.model.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                outputs = self.model(data_x)
                outputs_soft = F.softmax(outputs[:, :-1], dim=1)
                _, predicted_class = torch.max(outputs_soft.data, 1)
                predictions_all.extend(predicted_class.cpu().numpy())
                rejector_outputs = torch.sigmoid(outputs[:, -1])
                defers_all.extend((rejector_outputs.cpu().numpy() >= 0.5).astype(int))
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                rej_score.extend(rejector_outputs.cpu().numpy())
        # convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score,
        }
        return data

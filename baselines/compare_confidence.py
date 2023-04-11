import argparse
import copy
import logging
import math
import os
import pickle
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from pyexpat import model
from tqdm import tqdm

from .basemethod import BaseMethod

sys.path.append("..")
from helpers.metrics import *
from helpers.utils import *

eps_cst = 1e-8


class CompareConfidence(BaseMethod):
    """Method trains classifier indepedently on cross entropy,
    and expert model on whether human prediction is equal to ground truth.
    Then, at each test point we compare the confidence of the classifier
    and the expert model.
    """

    def __init__(self, model_class, model_expert, device, plotting_interval=100):
        """
        Args:
            model_class (pytorch model): _description_
            model_expert (pytorch model): _description_
            device (str): device
            plotting_interval (int, optional): _description_. Defaults to 100.
        """
        self.model_class = model_class
        self.model_expert = model_expert
        self.device = device
        self.plotting_interval = plotting_interval

    def fit_epoch_class(self, dataloader, optimizer, verbose=True, epoch=1):
        """
        train classifier for single epoch
        Args:
            dataloader (dataloader): _description_
            optimizer (optimizer): _description_
            verbose (bool, optional): to print loss or not. Defaults to True.
            epoch (int, optional): _description_. Defaults to 1.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        loss_fn = nn.CrossEntropyLoss()

        self.model_class.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            outputs = self.model_class(data_x)
            # cross entropy loss
            loss = F.cross_entropy(outputs, data_y)
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

    def fit_epoch_expert(self, dataloader, optimizer, verbose=True, epoch=1):
        """train expert model for single epoch

        Args:
            dataloader (_type_): _description_
            optimizer (_type_): _description_
            verbose (bool, optional): _description_. Defaults to True.
            epoch (int, optional): _description_. Defaults to 1.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        loss_fn = nn.CrossEntropyLoss()

        self.model_expert.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            hum_equal_to_y = (hum_preds == data_y).long()
            hum_equal_to_y = torch.cuda.LongTensor(hum_equal_to_y).to(self.device)
            outputs = self.model_expert(data_x)
            # cross entropy loss
            loss = loss_fn(outputs, hum_equal_to_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, hum_equal_to_y, topk=(1,))[0]
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
        scheduler=None,
        verbose=True,
        test_interval=5,
    ):
        """fits classifier and expert model

        Args:
            dataloader_train (_type_): train dataloader
            dataloader_val (_type_): val dataloader
            dataloader_test (_type_): _description_
            epochs (_type_): training epochs
            optimizer (_type_): optimizer function
            lr (_type_): learning rate
            scheduler (_type_, optional): scheduler function. Defaults to None.
            verbose (bool, optional): _description_. Defaults to True.
            test_interval (int, optional): _description_. Defaults to 5.

        Returns:
            dict: metrics on the test set
        """
        optimizer_class = optimizer(self.model_class.parameters(), lr=lr)
        optimizer_expert = optimizer(self.model_expert.parameters(), lr=lr)
        if scheduler is not None:
            scheduler_class = scheduler(optimizer_class, len(dataloader_train) * epochs)
            scheduler_expert = scheduler(
                optimizer_expert, len(dataloader_train) * epochs
            )
        best_acc = 0
        # store current model dict
        best_model = [copy.deepcopy(self.model_class.state_dict()), copy.deepcopy(self.model_expert.state_dict())]
        for epoch in tqdm(range(epochs)):
            self.fit_epoch_class(
                dataloader_train, optimizer_class, verbose=verbose, epoch=epoch
            )
            self.fit_epoch_expert(
                dataloader_train, optimizer_expert, verbose=verbose, epoch=epoch
            )
            if epoch % test_interval == 0 and epoch > 1:
                data_test = self.test(dataloader_val)
                val_metrics = compute_deferral_metrics(data_test)
                if val_metrics["classifier_all_acc"] >= best_acc: 
                    best_acc = val_metrics["classifier_all_acc"]
                    best_model = [copy.deepcopy(self.model_class.state_dict()), copy.deepcopy(self.model_expert.state_dict())]

            if scheduler is not None:
                scheduler_class.step()
                scheduler_expert.step()
        self.model_class.load_state_dict(best_model[0])
        self.model_expert.load_state_dict(best_model[1])

        return compute_deferral_metrics(self.test(dataloader_test))

    def test(self, dataloader):
        defers_all = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        self.model_expert.eval()
        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                outputs_expert = self.model_expert(data_x)
                outputs_expert = F.softmax(outputs_expert, dim=1)
                max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                class_probs_all.extend(outputs_class.cpu().numpy())
                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                defers = []
                for i in range(len(data_y)):
                    rej_score_all.extend(
                        [outputs_expert[i, 1].item() - max_class_probs[i].item()]
                    )
                    if outputs_expert[i, 1] > max_class_probs[i]:
                        defers.extend([1])
                    else:
                        defers.extend([0])
                defers_all.extend(defers)
        # convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score_all,
            "class_probs": class_probs_all,
        }
        return data

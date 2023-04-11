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


def weighted_cross_entropy_loss(outputs, labels, weights):
    """
    Weigthed cross entropy loss
    outputs: network outputs with softmax
    labels: target
    weights: weights for each example

    return: weighted cross entropy loss as scalar
    """
    outputs = weights * F.cross_entropy(outputs, labels, reduction="none")  # regular CE
    return torch.sum(outputs) / torch.sum(weights)


class DifferentiableTriage(BaseMethod):
    def __init__(
        self,
        model_class,
        model_rejector,
        device,
        weight_low=0.00,
        strategy="human_error",
        plotting_interval=100,
    ):
        """Method from the paper 'Differentiable Learning Under Triage' adapted to this setting
        Args:
            model_class (_type_): _description_
            model_rejector (_type_): _description_
            device (_type_): _description_
            weight_low (float in [0,1], optional): weight for points that are deferred so that classifier trains less on them
            strategy (_type_): pick between "model_first", "human_error"
                "model_first" means that the rejector is 1 only if the human is correct and the model is wrong
                "human_error": the rejector is 1 if the human gets it right, otherwise 0
            plotting_interval (int, optional): _description_. Defaults to 100.

        """
        self.model_class = model_class
        self.model_rejector = model_rejector
        self.device = device
        self.weight_low = weight_low
        self.plotting_interval = plotting_interval
        self.strategy = strategy

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

    def find_machine_samples(self, model_outputs, data_y, hum_preds):
        """

        Args:
            model_outputs (_type_): _description_
            data_y (_type_): _description_
            hum_preds (_type_): _description_

        Returns:
            array:  binary array of size equal to the input indicating whether to train or not on each poin
        """
        max_class_probs, predicted_class = torch.max(model_outputs.data, 1)
        model_error = predicted_class != data_y
        hum_error = hum_preds != data_y
        rejector_labels = []
        soft_weights_classifier = []
        if self.strategy == "model_first":
            for i in range(len(model_outputs)):
                if not model_error[i]:
                    rejector_labels.append(0)
                    soft_weights_classifier.append(1)
                elif not hum_error[i]:
                    rejector_labels.append(1)
                    soft_weights_classifier.append(self.weight_low)
                else:
                    rejector_labels.append(0)
                    soft_weights_classifier.append(1.0)
        else:
            for i in range(len(model_outputs)):
                if not hum_error[i]:
                    rejector_labels.append(1)
                    soft_weights_classifier.append(self.weight_low)
                else:
                    rejector_labels.append(0)
                    soft_weights_classifier.append(1.0)

        rejector_labels = torch.cuda.LongTensor(rejector_labels).to(self.device)
        soft_weights_classifier = torch.tensor(soft_weights_classifier).to(self.device)
        return rejector_labels, soft_weights_classifier

    def fit_epoch_class_triage(self, dataloader, optimizer, verbose=True, epoch=1):
        """
        Fit the model for classifier for one epoch
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        self.model_class.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            outputs = self.model_class(data_x)
            # cross entropy loss

            rejector_labels, soft_weights_classifier = self.find_machine_samples(
                outputs, data_y, hum_preds
            )

            loss = weighted_cross_entropy_loss(outputs, data_y, soft_weights_classifier)
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

    def fit_epoch_rejector(self, dataloader, optimizer, verbose=True, epoch=1):
        """
        Fit the rejector for one epoch
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        loss_fn = nn.CrossEntropyLoss()

        self.model_rejector.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            outputs_class = self.model_class(data_x)
            rejector_labels, soft_weights_classifier = self.find_machine_samples(
                outputs_class, data_y, hum_preds
            )
            outputs = self.model_rejector(data_x)
            # cross entropy loss
            loss = F.cross_entropy(outputs, rejector_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, rejector_labels, topk=(1,))[0]
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
        optimizer_class = optimizer(self.model_class.parameters(), lr=lr)
        optimizer_rejector = optimizer(self.model_rejector.parameters(), lr=lr)
        if scheduler is not None:
            scheduler_class = scheduler(optimizer_class, len(dataloader_train) * epochs)
            scheduler_rejector = scheduler(optimizer_rejector, len(dataloader_train) * epochs)
        self.model_class.train()
        self.model_rejector.train()

        logging.info("Re-training classifier on data based on the formula")
        for epoch in tqdm(range(int(epochs))):
            self.fit_epoch_class_triage(
                dataloader_train, optimizer_class, verbose=verbose, epoch=epoch
            )
            if verbose and epoch % test_interval == 0:
                logging.info(compute_classification_metrics(self.test(dataloader_val)))
            if scheduler is not None:
                scheduler_class.step()
        # now fit rejector

        logging.info("Fitting rejector on all data")
        best_acc = 0
        best_model = copy.deepcopy(self.model_rejector.state_dict())

        for epoch in tqdm(range(int(epochs))):
            self.fit_epoch_rejector(
                dataloader_train, optimizer_rejector, verbose=verbose, epoch=epoch
            )
            if verbose and epoch % test_interval == 0:
                logging.info(compute_deferral_metrics(self.test(dataloader_val)))
            if epoch % test_interval == 0 and epoch > 1:
                data_test = self.test(dataloader_val)
                val_metrics = compute_deferral_metrics(data_test)
                if val_metrics["system_acc"] >= best_acc:
                    best_acc = val_metrics["system_acc"]
                    best_model = copy.deepcopy(self.model_rejector.state_dict())

            if scheduler is not None:
                scheduler_rejector.step()
        self.model_rejector.load_state_dict(best_model)
        return compute_deferral_metrics(self.test(dataloader_test))

    def fit_hyperparam(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        verbose=True,
        test_interval=5,
        scheduler  = None,
    ):
        weight_low_grid = [0,  1]
        best_weight = 0
        best_acc = 0
        model_rejector_dict = copy.deepcopy(self.model_rejector.state_dict())
        model_class_dict = copy.deepcopy(self.model_class.state_dict())
        for weight in tqdm(weight_low_grid):
            self.weight_low = weight
            self.model_rejector.load_state_dict(model_rejector_dict)
            self.model_class.load_state_dict(model_class_dict)
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
            logging.info(f"weight low : {weight}, accuracy: {accuracy}")
            if accuracy > best_acc:
                best_acc = accuracy
                best_weight = weight
        self.weight_low = best_weight
        self.model_rejector.load_state_dict(model_rejector_dict)
        self.model_class.load_state_dict(model_class_dict)
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

    def test(self, dataloader):
        defers_all = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        self.model_rejector.eval()
        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                outputs_rejector = self.model_rejector(data_x)
                outputs_rejector = F.softmax(outputs_rejector, dim=1)
                _, predictions_rejector = torch.max(outputs_rejector.data, 1)
                max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                defers_all.extend(predictions_rejector.cpu().numpy())
                rej_score_all.extend(outputs_rejector[:, 1].cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())
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

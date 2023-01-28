import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
import shutil
import time
from sklearn.metrics.pairwise import rbf_kernel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.utils.data as data
import sys
from matplotlib import pyplot as plt
import pickle
from matplotlib import cm
import torch.optim as optim
import torch.distributions as D
import logging
from tqdm import tqdm
sys.path.append("..")
sys.path.append(".")
from helpers.metrics import *

# file is not used, consider deleting or moving to deprecated

def train_deferral_single_model(Method, dataloader_train, dataloader_test, epochs, lr, verbose = True, test_interval = 5,  include_scheduler = False):

    optimizer = torch.optim.SGD(Method.model.parameters(), lr,
                                    weight_decay=5e-4)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader_train) * epochs)
    for epoch in tqdm(range(epochs)):
        Method.fit_epoch( dataloader_train, optimizer, verbose, epoch)
        if verbose and epoch % test_interval == 0:
            data_test = Method.test(dataloader_test)
            print(compute_deferral_metrics(data_test))
        if include_scheduler:
            scheduler.step()
    
    final_test = Method.test(dataloader_test)
    return compute_deferral_metrics(final_test)


def train_single_model(Method, model, fit, dataloader_train, dataloader_test, epochs, verbose = True, test_interval = 5):
    '''
    Method: the method class
    model: model in method
    fit: fit method in Method class
    '''
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader_train) * epochs)
    for epoch in tqdm(range(epochs)):
        Method.fit(epoch, dataloader_train, optimizer, verbose, epoch)
        if epoch % test_interval == 0:
            data_test = Method.test(dataloader_test)
            print(compute_classification_metrics(data_test))
        scheduler.step()
    final_test = Method.test(dataloader_test)
    return compute_classification_metrics(final_test)

import logging
import os
import pickle
import sys

import torch
import torch.optim as optim

sys.path.append("../")
import sys

import torch
import torch.nn as nn

sys.path.append("../")
import argparse
import datetime
# allow logging to print everything
import logging

from baselines.lce_surrogate import *
from datasetsdefer.synthetic_data import SyntheticData
from helpers.metrics import *
from networks.linear_net import *

logging.basicConfig(level=logging.DEBUG)
import datetime

import torch.optim as optim
from baselines.compare_confidence import *
from baselines.differentiable_triage import *
from baselines.lce_surrogate import *
from baselines.mix_of_exps import *
from baselines.one_v_all import *
from baselines.selective_prediction import *
from datasetsdefer.broward import *
from datasetsdefer.chestxray import *
from datasetsdefer.cifar_h import *
from datasetsdefer.cifar_synth import *
from datasetsdefer.generic_dataset import *
from datasetsdefer.hatespeech import *
from datasetsdefer.imagenet_16h import *
from datasetsdefer.synthetic_data import *
from methods.milpdefer import *
from methods.realizable_surrogate import *
from networks.cnn import *
from networks.cnn import DenseNet121_CE, NetSimple, WideResNet


def main():

    # check if there exists directory ../exp_data
    if not os.path.exists("../exp_data"):
        os.makedirs("../exp_data")
        os.makedirs("../exp_data/data")
        os.makedirs("../exp_data/plots")
        os.makedirs("../exp_data/models")

    else:
        if not os.path.exists("../exp_data/data"):
            os.makedirs("../exp_data/data")
        if not os.path.exists("../exp_data/plots"):
            os.makedirs("../exp_data/plots")
        if not os.path.exists("../exp_data/models"):    
            os.makedirs("../exp_data/models")

    date_now = datetime.datetime.now()
    date_now = date_now.strftime("%Y-%m-%d_%H%M%S")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # {'classifier_all_acc': 0.9211, 'human_all_acc': 0.367, 'coverage': 0.9904, 'classifier_nondeferred_acc': 0.9262924071082391, 'human_deferred_acc': 0.3541666666666667, 'system_acc': 0.9208}
    
    data_dir = '../data'
    

    optimizer = optim.AdamW
    scheduler = None
    lr = 0.01
    max_trials = 20 
    total_epochs = 500# 100
    errors_lce = []
    errors_rs = []
    errors_one_v_all = []
    errors_selective = []
    errors_compare_confidence = []
    errors_differentiable_triage = []
    errors_mixofexps = []
    for trial in range(max_trials):

        # generate data
        dataset = BrowardDataset(data_dir, test_split = 0.2, val_split = 0.1)
        
        model = LinearNet(dataset.d,3).to(device)
        RS = RealizableSurrogate(1, 300, model, device, True)
        RS.fit_hyperparam(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=5,
        )
        rs_metrics = compute_coverage_v_acc_curve(RS.test(dataset.data_test_loader))

        model = LinearNet(dataset.d,3).to(device)
        mixofexps = MixtureOfExperts(model, device)
        mixofexps.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=5,
        )
        mixofexps_metrics = compute_coverage_v_acc_curve(
            mixofexps.test(dataset.data_test_loader)
        )

        model = LinearNet(dataset.d,3).to(device)
        LCE = LceSurrogate(1, 300, model, device)
        LCE.fit_hyperparam(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=5,
        )
        lce_metrics = compute_coverage_v_acc_curve(LCE.test(dataset.data_test_loader))

        model_class = LinearNet(dataset.d,2).to(device)
        model_expert = LinearNet(dataset.d,2).to(device)
        compareconfidence = CompareConfidence(model_class, model_expert, device)
        compareconfidence.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=5,
        )
        compare_metrics = compute_coverage_v_acc_curve(
            compareconfidence.test(dataset.data_test_loader)
        )


        model = LinearNet(dataset.d,3).to(device)
        OVA = OVASurrogate(1, 300, model, device)
        OVA.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=5,
        )
        ova_metrics = compute_coverage_v_acc_curve(OVA.test(dataset.data_test_loader))

        model = LinearNet(dataset.d,2).to(device)
        SP = SelectivePrediction(model, device)
        SP.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=5,
        )
        sp_metrics = compute_coverage_v_acc_curve(SP.test(dataset.data_test_loader))

        model_class = LinearNet(dataset.d,2).to(device)
        model_rejector = LinearNet(dataset.d,2).to(device)
        diff_triage = DifferentiableTriage(
            model_class, model_rejector, device, 1, "human_error"
        )
        diff_triage.fit_hyperparam(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=5,
        )
        diff_triage_metrics = compute_coverage_v_acc_curve(
            diff_triage.test(dataset.data_test_loader)
        )

        errors_mixofexps.append(mixofexps_metrics)
        errors_lce.append(lce_metrics)
        errors_rs.append(rs_metrics)
        errors_one_v_all.append(ova_metrics)
        errors_selective.append(sp_metrics)
        errors_compare_confidence.append(compare_metrics)
        errors_differentiable_triage.append(diff_triage_metrics)
        
        all_data = {
            "max_trials": max_trials,
            "mixofexp": errors_mixofexps,
            "lce": errors_lce,
            "rs": errors_rs,
            "one_v_all": errors_one_v_all,
            "selective": errors_selective,
            "compare_confidence": errors_compare_confidence,
            "differentiable_triage": errors_differentiable_triage,
        }
        # dump data into pickle file
        with open("../exp_data/data/compass_exp_" + date_now + ".pkl", "wb") as f:
            pickle.dump(all_data, f)


if __name__ == "__main__":
    main()

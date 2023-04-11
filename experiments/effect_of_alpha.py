import logging
import os
import pickle
import sys
import torch
import torch.optim as optim

sys.path.append("../")
import sys

import torch

sys.path.append("../")
import datetime

# allow logging to print everything
import logging
import argparse
from baselines.lce_surrogate import *
from datasetsdefer.synthetic_data import SyntheticData
from helpers.metrics import *
from networks.linear_net import *

logging.basicConfig(level=logging.DEBUG)
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
from datasetsdefer.generic_dataset import *
from datasetsdefer.hatespeech import *
from datasetsdefer.imagenet_16h import *
from datasetsdefer.cifar_synth import *
from datasetsdefer.synthetic_data import *
from methods.milpdefer import *
from methods.realizable_surrogate import *
from networks.cnn import *
import datetime
from networks.cnn import NetSimple

def main():

    # check if there exists directory ../exp_data
    if not os.path.exists("../exp_data"):
        os.makedirs("../exp_data")
        os.makedirs("../exp_data/data")
        os.makedirs("../exp_data/plots")
    else:
        if not os.path.exists("../exp_data/data"):
            os.makedirs("../exp_data/data")
        if not os.path.exists("../exp_data/plots"):
            os.makedirs("../exp_data/plots")

    date_now = datetime.datetime.now()
    date_now = date_now.strftime("%Y-%m-%d_%H%M%S")

    expert_k = 5
    alphas = [0, 0.1, 0.2 , 0.5 ,0.7,0.9,1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam
    scheduler = None
    lr = 0.001
    max_trials = 10 
    total_epochs = 50# 100


    errors_lce = []
    errors_rs = []
    errors_one_v_all = []
    errors_selective = []
    errors_compare_confidence = []
    errors_differentiable_triage = []
    errors_mixofexps = []
    for trial in range(max_trials):
        errors_lce_trial = []
        errors_rs_trial = []
        errors_one_v_all_trial = []
        errors_selective_trial = []
        errors_compare_confidence_trial = []
        errors_differentiable_triage_trial = []
        errors_mixofexps_trial = []
        for alpha in alphas:
            # generate data
            dataset = CifarSynthDataset(expert_k, False, batch_size=512)

            model = NetSimple(11, 50, 50, 100, 20).to(device)
            RS = RealizableSurrogate(alpha, 300, model, device, True)
            RS.fit(
                dataset.data_train_loader,
                dataset.data_val_loader,
                dataset.data_test_loader,
                epochs=total_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                lr=lr,
                verbose=False,
                test_interval=2,
            )
            rs_metrics = compute_deferral_metrics(RS.test(dataset.data_test_loader))


            errors_rs_trial.append(rs_metrics)

        errors_rs.append(errors_rs_trial)

        all_data = {
            "max_trials": max_trials,
            "ks": alphas,
            "rs": errors_rs,
        }
        # dump data into pickle file
        with open("../exp_data/data/alphacifark_" + date_now + ".pkl", "wb") as f:
            pickle.dump(all_data, f)


if __name__ == "__main__":
    main()

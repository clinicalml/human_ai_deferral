import copy
import torch
import sys
import logging
from tqdm import tqdm
sys.path.append("..")
from helpers.utils import *
from helpers.metrics import *
from baselines.basemethod import BaseSurrogateMethod


eps_cst = 1e-8



class RealizableSurrogate(BaseSurrogateMethod):
    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        """ Implementation of our RealizableSurrogate loss function
        """
        human_correct = (hum_preds == data_y).float()
        human_correct = torch.tensor(human_correct).to(self.device)
        batch_size = outputs.size()[0]  # batch_size
        outputs_exp = torch.exp(outputs)
        new_loss = -torch.log2(
            (
                human_correct * outputs_exp[range(batch_size), -1]
                + outputs_exp[range(batch_size), data_y]
            )
            / (torch.sum(outputs_exp, dim=1) + eps_cst)
        )  # pick the values corresponding to the labels
        ce_loss = -torch.log2(
            (outputs_exp[range(batch_size), data_y])
            / (torch.sum(outputs_exp[range(batch_size), :-1], dim=1) + eps_cst)
        )
        loss = self.alpha * new_loss + (1 - self.alpha) * ce_loss
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
        verbose=True,
        test_interval=5,
        scheduler=None,
        alpha_grid=[0, 0.1, 0.3, 0.5, 0.9, 1],
    ):
        # np.linspace(0,1,11)
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
                epochs = epochs,
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
                epochs = epochs,
                optimizer = optimizer,
                lr = lr,
                verbose = verbose,
                test_interval = test_interval,
                scheduler = scheduler,
            )
        test_metrics = compute_deferral_metrics(self.test(dataloader_test))
        return test_metrics

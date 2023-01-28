import logging
import sys
import time
import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from helpers.metrics import *
from helpers.utils import *
sys.path.append("..")
from baselines.basemethod import BaseMethod

class MILPDefer(BaseMethod):
    """ Our MILP for linear learning to defer """
    def __init__(
        self,
        n_classes,
        time_limit=-1,
        add_regularization=False,
        lambda_reg=1,
        verbose=False,
    ):
        '''

        Args:
            n_classes (_type_): number of classes in label
            time_limit (int, optional): limit of training time, -1 is no limit. Defaults to -1.
            add_regularization (bool, optional): add l1 regularization. Defaults to False.
            lambda_reg (int, optional): lambda for l1 reg. Defaults to 1.
            verbose (bool, optional): be verbose. Defaults to False.
        '''    

        self.n_classes = n_classes
        self.time_limit = time_limit
        self.verbose = verbose
        self.add_regularization = add_regularization
        self.lambda_reg = lambda_reg

    def fit(self, dataloader_train, dataloader_val, dataloader_test):
        if self.n_classes == 2:
            self.fit_binary(dataloader_train, dataloader_val, dataloader_test)
        else:
            self.fit_multiclass(dataloader_train, dataloader_val, dataloader_test)
        test_metrics = compute_deferral_metrics(self.test(dataloader_test))
        return test_metrics

    def fit_hyperparam(self, dataloader_train, dataloader_val, dataloader_test):
        lambda_grid = [1e-5, 1e-3, 1e-2, 1e-1, 1, 5]
        best_lambda = 0
        best_acc = 0
        for lambda_reg in lambda_grid:
            self.lambda_reg = lambda_reg
            fit = self.fit(dataloader_train, dataloader_val, dataloader_test)
            val_metrics = compute_deferral_metrics(self.test(dataloader_val))

            if val_metrics["system_acc"] > best_acc:
                best_acc = val_metrics["system_acc"]
                best_lambda = lambda_reg
        self.lambda_reg = best_lambda
        fit = self.fit(dataloader_train, dataloader_val, dataloader_test)
        return fit

    def fit_binary(self, dataloader_train, dataloader_val, dataloader_test):
        def cb(model, where):
            """
            Callback function to print the current solution for Gurobi
            """
            if where == GRB.Callback.MIPNODE:
                # Get model objective
                obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)

                # Has objective changed?
                if abs(obj - model._cur_obj) > 1e-6:
                    # If so, update incumbent and time
                    model._cur_obj = obj
                    # model._time = time.time()

                if time.time() - model._time > 30:
                    model._time = time.time()
                    # H = model.getVarByName("H")
                    # R = model.getVarByName("R")
                    error_v = 0
                    rejs = 0
                    for i in range(max_data):
                        rej_raw = np.sum(
                            [R[j].Xn * data_x[i][j] for j in range(dimension)]
                        )
                        pred_raw = np.sum(
                            [H[j].Xn * data_x[i][j] for j in range(dimension)]
                        )
                        if rej_raw > 0:
                            rejs += 1
                            error_v += (data_y[i] != hum_preds[i]) * 1.0
                        else:
                            pred = pred_raw > 0
                            error_v += (data_y[i] != (2 * pred - 1)) * 1.0
                    logging.info(
                        f"Current solution {time.time()-model._time0:.2f}s: Coverage is {1-rejs/max_data:.2f} and system error is {error_v/max_data*100:.2f}% "
                    )
            # Terminate if objective has not improved in 10mins
            # if time.time() - model._time > 60*5:
            #    model.terminate()

        data_x = dataloader_train.dataset.tensors[0]
        data_y = dataloader_train.dataset.tensors[1]
        human_predictions = dataloader_train.dataset.tensors[2]

        C = 1
        gamma = 0.00001
        Mi = C + gamma
        Ki = C + gamma
        max_data = len(data_x)
        hum_preds = 2 * np.array(human_predictions) - 1
        # add extra dimension to x
        data_x_original = torch.clone(data_x)
        norm_scale = max(torch.norm(data_x_original, p=1, dim=1))
        last_time = time.time()
        # normalize data_x and then add dimension
        data_x = torch.cat(
            (torch.ones((len(data_x)), 1), data_x / norm_scale), dim=1
        ).numpy()
        data_y = 2 * data_y - 1  # covert to 1, -1
        max_data = max_data  # len(data_x)
        dimension = data_x.shape[1]

        model = gp.Model("milp_deferral")
        model.Params.IntFeasTol = 1e-9
        model.Params.MIPFocus = 0
        if self.time_limit != -1:
            model.Params.TimeLimit = self.time_limit

        H = model.addVars(dimension, lb=[-C] * dimension, ub=[C] * dimension, name="H")
        Hnorm = model.addVars(
            dimension, lb=[0] * dimension, ub=[C] * dimension, name="Hnorm"
        )
        Rnorm = model.addVars(
            dimension, lb=[0] * dimension, ub=[C] * dimension, name="Rnorm"
        )
        R = model.addVars(dimension, lb=[-C] * dimension, ub=[C] * dimension, name="R")
        phii = model.addVars(max_data, vtype=gp.GRB.CONTINUOUS, lb=0)
        psii = model.addVars(max_data, vtype=gp.GRB.BINARY)
        ri = model.addVars(max_data, vtype=gp.GRB.BINARY)

        equal = np.array(data_y) == hum_preds * 1.0
        human_err = 1 - equal

        if self.add_regularization:
            model.setObjective(
                gp.quicksum([phii[i] + ri[i] * human_err[i] for i in range(max_data)])
                / max_data
                + self.lambda_reg * gp.quicksum([Hnorm[j] for j in range(dimension)])
                + self.lambda_reg * gp.quicksum([Rnorm[j] for j in range(dimension)])
            )
        else:
            model.setObjective(
                gp.quicksum([phii[i] + ri[i] * human_err[i] for i in range(max_data)])
                / max_data
            )
        for i in range(max_data):
            model.addConstr(phii[i] >= psii[i] - ri[i], name="phii" + str(i))
            model.addConstr(
                Mi * psii[i]
                >= gamma
                - data_y[i]
                * gp.quicksum(H[j] * data_x[i][j] for j in range(dimension)),
                name="psii" + str(i),
            )
            model.addConstr(
                gp.quicksum([R[j] * data_x[i][j] for j in range(dimension)])
                >= Ki * (ri[i] - 1) + gamma * ri[i],
                name="Riub" + str(i),
            )
            model.addConstr(
                gp.quicksum([R[j] * data_x[i][j] for j in range(dimension)])
                <= Ki * ri[i] + gamma * (ri[i] - 1),
                name="Rilb" + str(i),
            )
            model.update()
        if self.add_regularization:
            for j in range(dimension):
                model.addConstr(Hnorm[j] >= H[j], name="Hnorm1" + str(j))
                model.addConstr(Hnorm[j] >= -H[j], name="Hnorm2" + str(j))
                model.addConstr(Rnorm[j] >= R[j], name="Rnorm1" + str(j))
                model.addConstr(Rnorm[j] >= -R[j], name="Rnorm2" + str(j))

        model.ModelSense = 1  # minimize
        model._time = time.time()
        model._time0 = time.time()
        model._cur_obj = float("inf")
        # model.write('model.lp')
        if self.verbose:
            model.optimize(callback=cb)
        else:
            model.optimize()
        # check if halspace solution has 0 error
        error_v = 0
        rejs = 0
        for i in range(max_data):
            rej_raw = np.sum([R[j].X * data_x[i][j] for j in range(dimension)])
            pred_raw = np.sum([H[j].X * data_x[i][j] for j in range(dimension)])
            if rej_raw > 0:
                rejs += 1
                error_v += data_y[i] * hum_preds[i] != 1
            else:
                pred = pred_raw > 0
                error_v += data_y[i] != (2 * pred - 1)
        if self.verbose:
            logging.info(
                f"MILP Training: Coverage is {1-rejs/max_data:.2f} and system error is {error_v/max_data*100:.2f}% and runtime {model.Runtime}"
            )

        self.H = [H[j].X for j in range(dimension)]
        self.R = [R[j].X for j in range(dimension)]
        self.run_time = model.Runtime
        self.norm_scale = norm_scale
        self.train_error = error_v / max_data

    def fit_multiclass(self, dataloader_train, dataloader_val, dataloader_test):
        def cb(model, where):
            if where == GRB.Callback.MIPNODE:
                # Get model objective
                obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)

                # Has objective changed?
                if abs(obj - model._cur_obj) > 1e-6:
                    # If so, update incumbent and time
                    model._cur_obj = obj
                    # model._time = time.time()

                if time.time() - model._time > 30:
                    model._time = time.time()
                    # H = model.getVarByName("H")
                    # R = model.getVarByName("R")
                    error_v = 0
                    rejs = 0
                    for i in range(max_data):
                        rej_raw = np.sum(
                            [R[j].Xn * data_x[i][j] for j in range(dimension)]
                        )
                        pred_raw = [
                            np.sum(
                                [H[j, l].Xn * data_x[i][j] for j in range(dimension)]
                            )
                            for l in range(n_classes)
                        ]
                        pred_raw = np.array(pred_raw)
                        if rej_raw > 0:
                            rejs += 1
                            error_v += data_y[i] != hum_preds[i]
                        else:
                            pred = np.argmax(pred_raw)
                            error_v += data_y[i] != pred
                    logging.info(
                        f"Current solution {time.time()-model._time0:.2f}s: Coverage is {1-rejs/max_data:.2f} and system error is {error_v/max_data*100:.2f}% "
                    )
            # Terminate if objective has not improved in 10mins
            # if time.time() - model._time > 60*5:
            #    model.terminate()

        C = 1
        gamma = 0.00001
        Mi = C + gamma
        Ki = C + gamma
        max_data = len(data_x)

        data_x = dataloader_train.dataset.tensors[0]
        data_y = dataloader_train.dataset.tensors[1]
        human_predictions = dataloader_train.dataset.tensors[2]

        hum_preds = np.array(human_predictions)
        # add extra dimension to x
        data_x_original = torch.clone(data_x)
        norm_scale = max(torch.norm(data_x_original, p=1, dim=1))
        last_time = time.time()
        # normalize data_x and then add dimension
        data_x = torch.cat(
            (torch.ones((len(data_x)), 1), data_x / norm_scale), dim=1
        ).numpy()
        data_y = data_y
        max_data = max_data
        dimension = data_x.shape[1]

        model = gp.Model("milp_multi")
        model.Params.IntFeasTol = 1e-9
        model.Params.MIPFocus = 0
        if self.time_limit != -1:
            model.Params.TimeLimit = self.time_limit

        H = model.addVars(
            dimension,
            self.n_classes,
            lb=-C * np.ones((dimension, self.n_classes)),
            ub=C * np.ones((dimension, self.n_classes)),
            name="H",
        )
        Hnorm = model.addVars(
            dimension,
            self.n_classes,
            lb=-0 * np.ones((dimension, self.n_classes)),
            ub=C * np.ones((dimension, self.n_classes)),
            name="Hnorm",
        )
        Rnorm = model.addVars(
            dimension,
            lb=-0 * np.ones((dimension)),
            ub=C * np.ones((dimension)),
            name="Rnorm",
        )
        R = model.addVars(dimension, lb=[-C] * dimension, ub=[C] * dimension, name="R")
        phii = model.addVars(max_data, vtype=gp.GRB.CONTINUOUS, lb=0)
        psii = model.addVars(max_data, vtype=gp.GRB.BINARY)
        cil = model.addVars(self.n_classes, max_data, vtype=gp.GRB.BINARY)
        ri = model.addVars(max_data, vtype=gp.GRB.BINARY)

        equal = np.array(data_y) == hum_preds * 1.0
        human_err = 1 - equal

        if self.add_regularization:
            model.setObjective(
                gp.quicksum([psii[i] for i in range(max_data)])
                + self.lambda_reg * gp.quicksum([R[j] for j in range(dimension)])
                + self.lambda_reg
                * gp.quicksum(
                    [H[j, l] for j in range(dimension) for l in range(self.n_classes)]
                )
            )
        else:
            model.setObjective(
                gp.quicksum([phii[i] + ri[i] * human_err[i] for i in range(max_data)])
                / max_data
            )
        for i in range(max_data):
            model.addConstr(phii[i] >= psii[i] - ri[i], name="phii" + str(i))
            for l in range(self.n_classes):
                if l == data_y[i].item():
                    continue
                model.addConstr(
                    gp.quicksum(
                        (H[j, data_y[i].item()] - H[j, l]) * data_x[i][j]
                        for j in range(dimension)
                    )
                    >= Mi * (cil[l, i] - 1) + gamma * cil[l, i]
                )
                model.addConstr(
                    gp.quicksum(
                        (H[j, data_y[i].item()] - H[j, l]) * data_x[i][j]
                        for j in range(dimension)
                    )
                    <= Mi * (cil[l, i]) + gamma * (cil[l, i] - 1)
                )

            model.addConstr(
                psii[i]
                >= (
                    self.n_classes
                    - 1
                    - gp.quicksum(cil[l, i] for l in range(self.n_classes))
                )
                / (self.n_classes - 1),
                name="psii" + str(i),
            )

            model.addConstr(
                gp.quicksum([R[j] * data_x[i][j] for j in range(dimension)])
                >= Ki * (ri[i] - 1) + gamma * ri[i],
                name="Riub" + str(i),
            )
            model.addConstr(
                gp.quicksum([R[j] * data_x[i][j] for j in range(dimension)])
                <= Ki * ri[i] + gamma * (ri[i] - 1),
                name="Rilb" + str(i),
            )
            model.update()
        if self.add_regularization:
            for j in range(dimension):
                model.addConstr(Rnorm[j] >= R[j], name="Rnorm1" + str(j))
                model.addConstr(Rnorm[j] >= -R[j], name="Rnorm2" + str(j))
            for k in range(self.n_classes):
                for j in range(dimension):
                    model.addConstr(
                        Hnorm[j, k] >= H[j, k], name="Hnorm1" + str(j) + str(k)
                    )
                    model.addConstr(
                        Hnorm[j, k] >= -H[j, k], name="Hnorm2" + str(j) + str(k)
                    )

        model.ModelSense = 1  # minimize
        model._time = time.time()
        model._time0 = time.time()
        model._cur_obj = float("inf")

        if self.verbose:
            model.optimize(callback=cb)
        else:
            model.optimize()
        # check if halspace solution has 0 error
        error_v = 0
        rejs = 0
        for i in range(max_data):
            rej_raw = np.sum([R[j].X * data_x[i][j] for j in range(dimension)])
            pred_raw = [
                np.sum([H[j, l].X * data_x[i][j] for j in range(dimension)])
                for l in range(self.n_classes)
            ]
            pred_raw = np.array(pred_raw)
            if rej_raw > 0:
                rejs += 1
                error_v += data_y[i] != hum_preds[i]
            else:
                pred = np.argmax(pred_raw)
                error_v += data_y[i] != pred
        logging.info(
            f"MILP Training: Coverage is {1-rejs/max_data:.2f} and system error is {error_v/max_data*100:.2f}% and runtime {model.Runtime}"
        )

        self.H = [[H[j, l].X for j in range(dimension)] for l in range(self.n_classes)]
        self.R = [R[j].X for j in range(dimension)]
        self.runtime = model.Runtime
        self.norm_scale = norm_scale
        self.train_error = error_v / max_data

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def test(self, dataloader):

        defers_all = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        data_x = dataloader.dataset.tensors[0]
        data_y = dataloader.dataset.tensors[1]
        human_predictions = dataloader.dataset.tensors[2]
        rej_scores = []
        class_probs = []
        if self.n_classes == 2:

            # normalize data_x and then add dimension
            data_x = torch.cat(
                (torch.ones((len(data_x)), 1), data_x / self.norm_scale), dim=1
            ).numpy()
            # convert torch tensor to numpy array
            data_y = data_y.numpy()
            hum_preds = human_predictions.numpy()
            dimension = data_x.shape[1]

            for i in range(len(data_x)):
                rej_raw = np.sum([self.R[j] * data_x[i][j] for j in range(dimension)])
                pred_raw = np.sum([self.H[j] * data_x[i][j] for j in range(dimension)])
                rej_pred = rej_raw > 0
                pred = pred_raw > 0
                rej_scores.append(rej_raw)
                # sigmoid on pred_raw
                class_probs.append([1 - self.sigmoid(pred_raw), self.sigmoid(pred_raw)])
                defers_all.append(rej_pred)
                predictions_all.append(pred)
                truths_all.append(data_y[i])
                hum_preds_all.append(hum_preds[i])

        else:

            hum_preds = np.array(hum_preds)
            # normalize data_x and then add dimension
            data_x = torch.cat(
                (torch.ones((len(data_x)), 1), data_x / self.norm_scale), dim=1
            ).numpy()
            data_y = data_y.numpy()
            dimension = data_x.shape[1]

            for i in range(len(data_y)):
                rej_raw = np.sum([self.R[j].X * data_x[i][j] for j in range(dimension)])
                pred_raw = [
                    np.sum([self.H[j, l].X * data_x[i][j] for j in range(dimension)])
                    for l in range(self.n_classes)
                ]
                pred_raw = np.array(pred_raw)
                rej_pred = rej_raw > 0
                pred = np.argmax(pred_raw)
                defers_all.append(rej_pred)
                predictions_all.append(pred)
                truths_all.append(data_y[i])
                hum_preds_all.append(hum_preds[i])
                rej_scores.append(rej_raw)
                class_probs.append(self.softmax(pred_raw))

        # convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        rej_scores = np.array(rej_scores)
        class_probs = np.array(class_probs)

        data = {
            "defers": defers_all,
            "labels": truths_all,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_scores,
            "class_probs": class_probs,
        }
        return data

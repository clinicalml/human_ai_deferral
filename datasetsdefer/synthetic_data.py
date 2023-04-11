import torch
import numpy as np
import sys
import torch
import torch.distributions as D
import logging
import sys
sys.path.append('../')
sys.path.append('../networks')

from .basedataset import BaseDataset
from networks.linear_net import Linear_net_sig

class SyntheticData(BaseDataset):
    """ Synthetic dataset introduced in our work """	
    def __init__(self,  train_samples = 1000, test_samples = 1000, data_distribution = "mix_of_guassians", d = 10,
                mean_scale = 1, expert_deferred_error = 0, expert_nondeferred_error = 0.5, machine_nondeferred_error = 0, num_of_guassians = 10,
              val_split = 0.1, batch_size = 1000, transforms = None):
        '''
        
        total_samples: total number of samples in the dataset
        data_distribution: the distribution of the data. mix_of_guassians, or uniform
        d: dimension of the data
        mean_scale: the scale of the means of the guassians, or uniform
        expert_deferred_error: the error of the expert when the data is deferred
        expert_nondeferred_error: the error of the expert when the data is nondeferred
        machine_nondeferred_error: the error of the machine when the data is nondeferred
        num_of_guassians: the number of guassians in the mix of guassians
        '''
        self.val_split = val_split
        self.batch_size = batch_size
        self.train_split = 1 - val_split
        self.transforms = transforms
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.total_samples = train_samples + test_samples
        self.data_distribution = data_distribution
        self.d = d
        self.n_dataset = 2
        self.mean_scale = mean_scale
        self.expert_deferred_error = expert_deferred_error
        self.expert_nondeferred_error = expert_nondeferred_error
        self.machine_nondeferred_error = machine_nondeferred_error
        self.num_of_guassians = num_of_guassians
        self.generate_data()


    
    def generate_data(self):
        if self.data_distribution == "uniform":
            data_x = torch.rand((self.total_samples,self.d))*self.mean_scale
        else:
            mix = D.Categorical(torch.ones(self.num_of_guassians,))
            comp = D.Independent(D.Normal(
                        torch.randn(self.num_of_guassians,self.d), torch.rand(self.num_of_guassians,self.d)), 1)
            gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
            data_x = gmm.sample((self.total_samples,))*self.mean_scale
        # get random labels
        mean_rej_prop = 0
        # make sure ramdom rejector rejects between 20 and 80% of the time (variable)
        while not (mean_rej_prop>=0.2 and mean_rej_prop<=0.8):
            net_rej_opt = Linear_net_sig(self.d)
            with torch.no_grad():
                outputs = net_rej_opt(data_x)
                predicted = torch.round(outputs.data)
                mean_rej_prop = np.mean([predicted[i][0] for i in range(len(predicted))]) 
        # get rejector preds on x
        opt_rej_preds = []
        with torch.no_grad():
            outputs = net_rej_opt(data_x)
            predicted = torch.round(outputs.data)   
            opt_rej_preds = [predicted[i][0] for i in range(len(predicted))]
        # get classifier that is 1 at least 20% and at most 80% on non-deferred side
        mean_class_prop = 0
        net_mach_opt = Linear_net_sig(self.d)
        while not (mean_class_prop>=0.2 and mean_class_prop<=0.8):
            net_mach_opt = Linear_net_sig(self.d)
            with torch.no_grad():
                outputs = net_mach_opt(data_x)
                predicted = torch.round(outputs.data)
                predicted_class = [predicted[i][0]*(1-opt_rej_preds[i]) for i in range(len(predicted))]
                mean_class_prop = np.sum(predicted_class)/(len(opt_rej_preds)-np.sum(opt_rej_preds))
        # get classifier preds on x
        opt_mach_preds = []
        with torch.no_grad():
            outputs = net_mach_opt(data_x)
            predicted = torch.round(outputs.data)   
            opt_mach_preds = [predicted[i][0] for i in range(len(predicted))]

        # get random labels
        data_y = torch.randint(low=0,high=2,size=(self.total_samples,))
        # make labels consistent with net_mach_opt on non-deferred side with error specified
        for i in range(len(data_y)):
            if opt_rej_preds[i] == 0:
                coin = np.random.binomial(1,1-self.machine_nondeferred_error,1)[0]
                if coin == 1:
                    data_y[i] = opt_mach_preds[i]


        # make expert 1-expert_deferred_error accurate on deferred side and 1-expert_nondeferred_error accurate otherwise
        human_predictions = [0]*len(data_y)
        for i in range(len(data_y)):
            if opt_rej_preds[i] == 1:
                coin = np.random.binomial(1,1-self.expert_deferred_error,1)[0]
                if coin == 1:
                    human_predictions[i] = data_y[i]
                else:
                    human_predictions[i] = 1 - data_y[i]

            else:
                coin = np.random.binomial(1,1-self.expert_nondeferred_error,1)[0]
                if coin == 1:
                    human_predictions[i] = data_y[i]
                else:
                    human_predictions[i] = 1 - data_y[i]

        human_predictions = torch.tensor(human_predictions)
        # split into train, val, test
        train_size = int(self.train_samples * self.train_split)
        val_size = int(self.train_samples * self.val_split)
        test_size = len(data_x) - train_size - val_size # = self.test_samples

        self.train_x, self.val_x, self.test_x = torch.utils.data.random_split(data_x, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
        self.train_y, self.val_y, self.test_y = torch.utils.data.random_split(data_y, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
        self.train_h, self.val_h, self.test_h = torch.utils.data.random_split(human_predictions, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
        logging.info("train size: ", len(self.train_x))
        logging.info("val size: ", len(self.val_x))
        logging.info("test size: ", len(self.test_x))
        self.data_train = torch.utils.data.TensorDataset(self.train_x.dataset.data[self.train_x.indices], self.train_y.dataset.data[self.train_y.indices], self.train_h.dataset.data[self.train_h.indices])
        self.data_val = torch.utils.data.TensorDataset(self.val_x.dataset.data[self.val_x.indices], self.val_y.dataset.data[self.val_y.indices], self.val_h.dataset.data[self.val_h.indices])
        self.data_test = torch.utils.data.TensorDataset(self.test_x.dataset.data[self.test_x.indices], self.test_y.dataset.data[self.test_y.indices], self.test_h.dataset.data[self.test_h.indices])
        

        self.data_train_loader = torch.utils.data.DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)
        self.data_val_loader = torch.utils.data.DataLoader(self.data_val, batch_size=self.batch_size, shuffle=True)
        self.data_test_loader = torch.utils.data.DataLoader(self.data_test, batch_size=self.batch_size, shuffle=True)


        # double check if the solution we got is actually correct
        error_optimal_= 0
        for i in range(len(data_y)):
            if opt_rej_preds[i] == 1:
                error_optimal_ += human_predictions[i] != data_y[i]
            else:
                error_optimal_ += opt_mach_preds[i] != data_y[i]
        error_optimal_ = error_optimal_/len(data_y)
        self.error_optimal = error_optimal_
        logging.info(f'Data optimal: Accuracy Train {100-100*error_optimal_:.3f} with rej {mean_rej_prop*100} \n \n')





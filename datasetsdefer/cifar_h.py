import torch
import numpy as np
import os
import random
import sys
import torch
import logging
import sys

sys.path.append("../")
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasetsdefer.generic_dataset import GenericImageExpertDataset
import requests
from .basedataset import BaseDataset


# https://github.com/jcpeterson/cifar-10h
class Cifar10h(BaseDataset):
    """CIFAR-10H dataset with seperate human annotations on the test set of CIFAR-10"""

    def __init__(
        self,
        use_data_aug,
        data_dir,
        test_split=0.2,
        val_split=0.1,
        batch_size=1000,
        transforms=None,
    ):
        """
        data_dir: where to save files for model
        use_data_aug: whether to use data augmentation (bool)
        test_split: percentage of test data
        val_split: percentage of data to be used for validation (from training set)
        batch_size: batch size for training
        transforms: data transforms
        """
        self.data_dir = data_dir
        self.use_data_aug = use_data_aug
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 10
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.generate_data()

    def metrics_cifar10h(self, exp_preds, labels):
        correct = 0
        total = 0
        j = 0
        self.class_conditional_acc = [0] * 10
        class_counts = [0] * 10
        for i in range(len(exp_preds)):
            total += 1
            j += 1
            correct += exp_preds[i] == labels[i]
            self.class_conditional_acc[labels[i]] += exp_preds[i] == labels[i]
            class_counts[labels[i]] += 1
        for i in range(0, 10):
            self.class_conditional_acc[i] = (
                100 * self.class_conditional_acc[i] / class_counts[i]
            )
        self.human_accuracy = 100 * correct / total

    def generate_data(self):
        """
        generate data for training, validation and test sets
        : "airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9
        """
        # download cifar10h data
        # check if file already exists
        # check if file already exists
        if not os.path.exists(self.data_dir + "/cifar10h-probs.npy"):
            logging.info("Downloading cifar10h data")
            r = requests.get(
                "https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy",
                allow_redirects=True,
            )
            with open(self.data_dir + "/cifar10h-probs.npy", "wb") as f:
                f.write(r.content)
            logging.info("Finished Downloading cifar10h data")
            try:
                cifar10h = np.load(self.data_dir + "/cifar10h-probs.npy")
            except:
                logging.error("Failed to load cifar10h data")
                raise
        else:
            logging.info("Loading cifar10h data")
            try:
                cifar10h = np.load(self.data_dir + "/cifar10h-probs.npy")
            except:
                logging.error("Failed to load cifar10h data")
                raise

        human_predictions = np.array(
            [
                np.argmax(np.random.multinomial(1, cifar10h[i]))
                for i in range(len(cifar10h))
            ]
        )

        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        if self.use_data_aug:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: F.pad(
                            x.unsqueeze(0), (4, 4, 4, 4), mode="reflect"
                        ).squeeze()
                    ),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        dataset = "cifar10"
        kwargs = {"num_workers": 0, "pin_memory": True}

        train_dataset_all = datasets.__dict__[dataset.upper()](
            "../data", train=False, download=True, transform=transform_test
        )
        labels_all = train_dataset_all.targets
        self.metrics_cifar10h(human_predictions, labels_all)

        test_size = int(self.test_split * len(train_dataset_all))
        val_size = int(self.val_split * len(train_dataset_all))
        train_size = len(train_dataset_all) - test_size - val_size

        train_x = train_dataset_all.data
        train_y = train_dataset_all.targets
        train_y = np.array(train_y)
        random_seed = random.randrange(10000)

        train_x, val_x, test_x = torch.utils.data.random_split(
            train_x,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_y, val_y, test_y = torch.utils.data.random_split(
            train_y,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_h, val_h, test_h = torch.utils.data.random_split(
            human_predictions,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )

        data_train = GenericImageExpertDataset(
            train_x.dataset[train_x.indices],
            train_y.dataset[train_y.indices],
            train_h.dataset[train_h.indices],
            transform_train,
        )
        data_val = GenericImageExpertDataset(
            val_x.dataset[val_x.indices],
            val_y.dataset[val_y.indices],
            val_h.dataset[val_h.indices],
            transform_test,
        )
        data_test = GenericImageExpertDataset(
            test_x.dataset[test_x.indices],
            test_y.dataset[test_y.indices],
            test_h.dataset[test_h.indices],
            transform_test,
        )

        self.data_train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

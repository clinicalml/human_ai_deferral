import logging
import sys

import numpy as np
import torch
from .basedataset import BaseDataset

sys.path.append("../")
from PIL import Image
from torch.utils.data import Dataset
logging.getLogger("PIL").setLevel(logging.WARNING)


class GenericImageExpertDataset(Dataset):
    def __init__(self, images, targets, expert_preds, transforms_fn, to_open=False):
        """

        Args:
            images (list): List of images
            targets (list): List of labels
            expert_preds (list): List of expert predictions
            transforms_fn (function): Function to apply to images
            to_open (bool): Whether to open images or not (RGB reader)
        """
        self.images = images
        self.targets = np.array(targets)
        self.expert_preds = np.array(expert_preds)
        self.transforms_fn = transforms_fn
        self.to_open = to_open

    def __getitem__(self, index):
        """Take the index of item and returns the image, label, expert prediction and index in original dataset"""
        label = self.targets[index]
        if self.transforms_fn is not None and self.to_open:
            image_paths = self.images[index]
            image = Image.open(image_paths).convert("RGB")
            image = self.transforms_fn(image)
        elif self.transforms_fn is not None:
            image = self.transforms_fn(self.images[index])
        else:
            image = self.images[index]
        expert_pred = self.expert_preds[index]
        return torch.FloatTensor(image), label, expert_pred

    def __len__(self):
        return len(self.targets)


class GenericDatasetDeferral(BaseDataset):
    def __init__(
        self,
        data_train,
        data_test=None,
        test_split=0.2,
        val_split=0.1,
        batch_size=100,
        transforms=None,
    ):
        """
        
        data_train: training data expectd as dict with keys 'data_x', 'data_y', 'hum_preds'
        data_test: test data expectd as dict with keys 'data_x', 'data_y', 'hum_preds'
        test_split: fraction of training data to use for test
        val_split: fraction of training data to use for validation
        batch_size: batch size for dataloaders
        transforms: transforms to apply to images
        """
        self.data_train = data_train
        self.data_test = data_test
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.generate_data()

    def generate_data(self):
        train_x = self.data_train["data_x"]
        train_y = self.data_train["data_y"]
        train_hum_preds = self.data_train["hum_preds"]
        if self.data_test is not None:
            test_x = self.data_test["data_x"]
            test_y = self.data_test["data_y"]
            test_h = self.data_test["hum_preds"]
            train_size = int((1 - self.val_split) * self.total_samples)
            val_size = int(self.val_split * self.total_samples)
            train_x, val_x = torch.utils.data.random_split(
                train_x,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            train_y, val_y = torch.utils.data.random_split(
                train_y,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            train_h, val_h = torch.utils.data.random_split(
                train_hum_preds,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            self.data_train = torch.utils.data.TensorDataset(
                train_x.dataset.data[train_x.indices],
                train_y.dataset.data[train_y.indices],
                train_h.dataset.data[train_h.indices],
            )
            self.data_val = torch.utils.data.TensorDataset(
                val_x.dataset.data[val_x.indices],
                val_y.dataset.data[val_y.indices],
                val_h.dataset.data[val_h.indices],
            )
            self.data_test = torch.utils.data.TensorDataset(test_x, test_y, test_h)

        else:
            train_size = int(self.train_split * self.total_samples)
            val_size = int(self.val_split * self.total_samples)
            test_size = self.total_samples - train_size - val_size
            train_x, val_x, test_x = torch.utils.data.random_split(
                train_x,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )
            train_y, val_y, test_y = torch.utils.data.random_split(
                train_y,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )
            train_h, val_h, test_h = torch.utils.data.random_split(
                train_hum_preds,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

            self.data_train = torch.utils.data.TensorDataset(
                train_x.dataset.data[train_x.indices],
                train_y.dataset.data[train_y.indices],
                train_h.dataset.data[train_h.indices],
            )
            self.data_val = torch.utils.data.TensorDataset(
                val_x.dataset.data[val_x.indices],
                val_y.dataset.data[val_y.indices],
                val_h.dataset.data[val_h.indices],
            )
            self.data_test = torch.utils.data.TensorDataset(
                test_x.dataset.data[test_x.indices],
                test_y.dataset.data[test_y.indices],
                test_h.dataset.data[test_h.indices],
            )

        self.data_train_loader = torch.utils.data.DataLoader(
            self.data_train, batch_size=self.batch_size, shuffle=True
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            self.data_val, batch_size=self.batch_size, shuffle=True
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            self.data_test, batch_size=self.batch_size, shuffle=True
        )

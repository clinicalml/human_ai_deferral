import torch
import numpy as np
import os
import random
import pandas as pd
import sys
import torch
import sys
import torch.nn as nn
sys.path.append("../")
sys.path.append("../networks")
from networks.cnn import DenseNet121_CE
import torchvision.transforms as transforms
from datasetsdefer.generic_dataset import GenericImageExpertDataset
from .basedataset import BaseDataset


# https://osf.io/2ntrf/
# https://www.pnas.org/doi/10.1073/pnas.2111547119


class ImageNet16h(BaseDataset):
    def __init__(
        self,
        use_data_aug,
        data_dir,
        noise_version,
        test_split=0.2,
        val_split=0.1,
        batch_size=1000,
        get_embeddings=False,
        transforms=None,
    ):
        """
        Must go to  https://osf.io/2ntrf/ , click on OSF Storage, download zip, unzip it, and write the path of the folder in data_dir
        data_dir: where to save files for model
        noise_version: noise version to use from 080,095, 110,125 (From imagenet16h paper)
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
        self.n_dataset = 16
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.noise_version = noise_version
        self.get_embeddings = get_embeddings
        self.d = 1024
        if self.noise_version not in ["080", "095", "110", "125"]:
            raise ValueError(
                'Noise version not supported, only pick from ["080","095","110","125"]'
            )
        self.generate_data()

    def generate_data(self):
        """
        generate data for training, validation and test sets

        """
        # check if the folder data_dir has everything we need

        if not os.path.exists(
            self.data_dir
            + "/Behavioral Data/human_only_classification_6per_img_export.csv"
        ):
            raise ValueError(
                "cant find csv, Please download the data from https://osf.io/2ntrf/ , unzip it, and construct the path of the folder in data_dir"
            )
        if not os.path.exists(
            self.data_dir + "/Noisy Images/phase_noise_" + self.noise_version
        ):
            raise ValueError(
                "cant find image folder, Please download the data from https://osf.io/2ntrf/ , unzip it, and construct the path of the folder in data_dir"
            )

        # load the csv file
        data_behavioral = pd.read_csv(
            self.data_dir
            + "/Behavioral Data/human_only_classification_6per_img_export.csv"
        )

        data_behavioral = data_behavioral[
            data_behavioral["noise_level"] == int(self.noise_version)
        ]
        data_behavioral = data_behavioral[
            [
                "participant_id",
                "image_id",
                "image_name",
                "image_category",
                "participant_classification",
                "confidence",
            ]
        ]

        # get unique categories
        categories = data_behavioral["image_category"].unique()
        # get mapping from category to index
        self.category_to_idx = {categories[i]: i for i in range(len(categories))}

        imagenames_categories = dict(
            zip(data_behavioral["image_name"], data_behavioral["image_category"])
        )
        # for each image name, get all the participant classifications
        image_name_to_participant_classifications = {}
        for image_name in data_behavioral["image_name"].unique():
            image_name_to_participant_classifications[image_name] = data_behavioral[
                data_behavioral["image_name"] == image_name
            ]["participant_classification"].values

        # sample a single classification from the participant classifications
        image_name_to_single_participant_classification = {}
        for image_name in image_name_to_participant_classifications:
            image_name_to_single_participant_classification[
                image_name
            ] = np.random.choice(image_name_to_participant_classifications[image_name])

        image_names = os.listdir(
            self.data_dir + "/Noisy Images/phase_noise_" + self.noise_version
        )
        image_names = [x for x in image_names if x.endswith(".png")]
        # remove png extension
        image_names = [x[:-4] for x in image_names]
        image_paths = np.array(
            [
                "/data/ml2/shared/mozannar/improved_deferral/data/osfstorage-archive/Noisy Images/phase_noise_080/"
                + x
                + ".png"
                for x in image_names
            ]
        )
        # get label for image names
        image_names_labels = np.array(
            [self.category_to_idx[imagenames_categories[x]] for x in image_names]
        )
        # get prediction for image names
        image_names_human_predictions = np.array(
            [
                self.category_to_idx[image_name_to_single_participant_classification[x]]
                for x in image_names
            ]
        )

        transform_train = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test_tensor = transforms.Compose(
            [
            transforms.ToTensor()
            ]
        )

        test_size = int(self.test_split * len(image_paths))
        val_size = int(self.val_split * len(image_paths))
        train_size = len(image_paths) - test_size - val_size
        random_seed = random.randrange(10000)

        train_x, val_x, test_x = torch.utils.data.random_split(
            image_paths,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_y, val_y, test_y = torch.utils.data.random_split(
            image_names_labels,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_h, val_h, test_h = torch.utils.data.random_split(
            image_names_human_predictions,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )




        data_train = GenericImageExpertDataset(
            train_x.dataset[train_x.indices],
            train_y.dataset[train_y.indices],
            train_h.dataset[train_h.indices],
            transform_train,
            to_open=True,
        )
        data_val = GenericImageExpertDataset(
            val_x.dataset[val_x.indices],
            val_y.dataset[val_y.indices],
            val_h.dataset[val_h.indices],
            transform_test,
            to_open=True,
        )
        data_test = GenericImageExpertDataset(
            test_x.dataset[test_x.indices],
            test_y.dataset[test_y.indices],
            test_h.dataset[test_h.indices],
            transform_test,
            to_open=True,
        )

        self.data_train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=self.batch_size,
            shuffle=False,
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

        if self.get_embeddings:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model_linear = DenseNet121_CE(16).to(device)
            model_linear.densenet121.classifier = nn.Sequential(*list(model_linear.densenet121.classifier.children())[:-1])
            # get embeddings of train-val-test data
            def get_embeddings(model, data_loader):
                model.eval()
                with torch.no_grad():
                    embeddings = []
                    for i, (x, y, h) in enumerate(data_loader):
                        x = x.to(device)
                        y = y.to(device)
                        h = h.to(device)
                        x = model(x)
                        embeddings.append(x.cpu().numpy())
                return np.concatenate(embeddings, axis=0)
            
            train_embeddings = torch.FloatTensor(get_embeddings(model_linear, self.data_train_loader))
            val_embeddings = torch.FloatTensor(get_embeddings(model_linear, self.data_val_loader))
            test_embeddings = torch.FloatTensor(get_embeddings(model_linear, self.data_test_loader))

            
            data_train = torch.utils.data.TensorDataset(
                train_embeddings,
                torch.from_numpy(train_y.dataset[train_y.indices]),
                torch.from_numpy(train_h.dataset[train_h.indices]),
            )
            data_val = torch.utils.data.TensorDataset(
                val_embeddings,
                torch.from_numpy(val_y.dataset[val_y.indices]),
                torch.from_numpy(val_h.dataset[val_h.indices]),
            )
            data_test = torch.utils.data.TensorDataset(
                test_embeddings,
                torch.from_numpy(test_y.dataset[test_y.indices]),
                torch.from_numpy(test_h.dataset[test_h.indices]),
            )

            self.data_train_loader = torch.utils.data.DataLoader(
                data_train,
                batch_size=1000,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            self.data_val_loader = torch.utils.data.DataLoader(
                data_val,
                batch_size=1000,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            self.data_test_loader = torch.utils.data.DataLoader(
                data_test,
                batch_size=1000,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

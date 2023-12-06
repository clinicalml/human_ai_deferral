from .basedataset import BaseDataset
import torch
import numpy as np
import os
import random
import shutil
import sys
import torch
import logging
import pandas as pd
import sys
import torch.nn as nn

sys.path.append("../")
import torchvision.transforms as transforms
from datasetsdefer.generic_dataset import GenericImageExpertDataset
import requests
import urllib.request
import tarfile
sys.path.append("../networks")
from networks.cnn import DenseNet121_CE

class ChestXrayDataset(BaseDataset):
    """Chest X-ray dataset from NIH with multiple radiologist annotations per point from Google Research"""
    def __init__(
        self,
        non_deferral_dataset,
        use_data_aug,
        data_dir,
        label_chosen,
        test_split=0.3,
        val_split=0.2,
        batch_size=1000,
        get_embeddings = False,
        transforms=None,
    ):
        """
        See https://nihcc.app.box.com/v/ChestXray-NIHCC and
        non_deferral_dataset (bool): if True, the dataset is the non-deferral dataset, meaning it is the full NIH dataset without the val-test of the human labeled, otherwise it is the deferral dataset that is only 4k in size total
        data_dir: where to save files for model
        label_chosen (int in 0,1,2,3): if non_deferral_dataset = False: which label to use between 0,1,2,3 which correspond to Fracture, Pneumotheras,  Airspace Opacity, and Nodule/Mass; if true: then it's NoFinding or not, Pneumotheras, Effusion, Nodule/Mass
        use_data_aug: whether to use data augmentation (bool)
        test_split: percentage of test data
        val_split: percentage of data to be used for validation (from training set)
        batch_size: batch size for training
        transforms: data transforms
        """
        self.non_deferral_dataset = non_deferral_dataset
        self.data_dir = data_dir
        self.use_data_aug = use_data_aug
        self.label_chosen = label_chosen
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 2
        self.get_embeddings = get_embeddings
        self.d = 1024
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.generate_data()

    def generate_data(self):
        """
        generate data for training, validation and test sets
        """

        links = [
            "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
            "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
            "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
            "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
            "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
            "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
            "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
            "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
            "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
            "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
            "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
            "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
        ]
        max_links = 12  # 12 is the limit
        links = links[:max_links]

        if not os.path.exists(self.data_dir + "/images_nih"):
            logging.info("Downloading NIH dataset")
            for idx, link in enumerate(links):
                if not os.path.exists(
                    self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                ):
                    fn = self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                    logging.info("downloading " + fn + "...")
                    urllib.request.urlretrieve(link, fn)  # download the zip file

            logging.info("Download complete. Please check the checksums")

            # make directory
            if not os.path.exists(self.data_dir + "/images_nih"):
                os.makedirs(self.data_dir + "/images_nih")

            # extract files
            for idx in range(max_links):
                fn = self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                logging.info("Extracting " + fn + "...")
                # os.system('tar -zxvf '+fn+' -C '+self.data_dir+'/images_nih')
                file = tarfile.open(fn)
                file.extractall(self.data_dir + "/images_nih")
                file.close()
                fn = self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                os.remove(fn)
                logging.info("Done")
        else:
            # double check that all files are there and extracted
            # get number of files in directory
            # if not equal to 102120, then download again
            num_files = len(
                [
                    name
                    for name in os.listdir(self.data_dir + "/images_nih")
                    if os.path.isfile(os.path.join(self.data_dir + "/images_nih", name))
                ]
            )
            if num_files != 102120:  # acutal is 112120
                logging.info("Files missing. Re-downloading...")
                shutil.rmtree(self.data_dir + "/images_nih")

                for idx, link in enumerate(links):
                    # check if file exists
                    fn = self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                    if not os.path.exists(
                        self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                    ):
                        logging.info("downloading " + fn + "...")
                        urllib.request.urlretrieve(link, fn)

                logging.info("Download complete. Please check the checksums")

                # make directory
                if not os.path.exists(self.data_dir + "/images_nih"):
                    os.makedirs(self.data_dir + "/images_nih")

                # extract files
                for idx in range(max_links):
                    fn = self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                    logging.info("Extracting " + fn + "...")
                    # os.system('tar -zxvf '+fn+' -C '+self.data_dir+'/images_nih')
                    file = tarfile.open(fn)
                    file.extractall(self.data_dir + "/images_nih")
                    file.close()
                    fn = self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                    os.remove(fn)
                    logging.info("Done")

        # DOWNLOAD CSV DATA FOR LABELS

        if (
            not os.path.exists(
                self.data_dir + "/four_findings_expert_labels_individual_readers.csv"
            )
            or not os.path.exists(
                self.data_dir + "/four_findings_expert_labels_test_labels.csv"
            )
            or not os.path.exists(
                self.data_dir + "/four_findings_expert_labels_validation_labels.csv"
            )
            or not os.path.exists(self.data_dir + "/Data_Entry_2017_v2020.csv")
        ):
            logging.info("Downloading readers NIH data")
            r = requests.get(
                "https://storage.googleapis.com/gcs-public-data--healthcare-nih-chest-xray-labels/four_findings_expert_labels/individual_readers.csv",
                allow_redirects=True,
            )

            with open(
                self.data_dir + "/four_findings_expert_labels_individual_readers.csv",
                "wb",
            ) as f:
                f.write(r.content)
            r = requests.get(
                "https://storage.googleapis.com/gcs-public-data--healthcare-nih-chest-xray-labels/four_findings_expert_labels/test_labels.csv",
                allow_redirects=True,
            )
            with open(
                self.data_dir + "/four_findings_expert_labels_test_labels.csv", "wb"
            ) as f:
                f.write(r.content)
            r = requests.get(
                "https://storage.googleapis.com/gcs-public-data--healthcare-nih-chest-xray-labels/four_findings_expert_labels/validation_labels.csv",
                allow_redirects=True,
            )
            with open(
                self.data_dir + "/four_findings_expert_labels_validation_labels.csv",
                "wb",
            ) as f:
                f.write(r.content)

            logging.info("Finished Downloading  readers NIH data")
            r = requests.get("https://raw.githubusercontent.com/raj713335/AI-IN-MEDICINE-SPECIALIZATION/master/DATA/Data_Entry_2017_v2020.csv")
            with open(self.data_dir + "/Data_Entry_2017_v2020.csv", "wb") as f:
                f.write(r.content)

            try:
                readers_data = pd.read_csv(
                    self.data_dir
                    + "/four_findings_expert_labels_individual_readers.csv"
                )
                test_data = pd.read_csv(
                    self.data_dir + "/four_findings_expert_labels_test_labels.csv"
                )
                validation_data = pd.read_csv(
                    self.data_dir + "/four_findings_expert_labels_validation_labels.csv"
                )
                all_dataset_data = pd.read_csv(
                    self.data_dir + "/Data_Entry_2017_v2020.csv"
                )

            except:
                logging.error("Failed to load readers NIH data")
                raise

        else:
            logging.info("Loading readers NIH data")
            try:
                readers_data = pd.read_csv(
                    self.data_dir
                    + "/four_findings_expert_labels_individual_readers.csv"
                )
                test_data = pd.read_csv(
                    self.data_dir + "/four_findings_expert_labels_test_labels.csv"
                )
                validation_data = pd.read_csv(
                    self.data_dir + "/four_findings_expert_labels_validation_labels.csv"
                )
                all_dataset_data = pd.read_csv(
                    self.data_dir + "/Data_Entry_2017_v2020.csv"
                )
            except:
                logging.error("Failed to load readers NIH data")
                raise

        data_labels = {}
        for i in range(len(validation_data)):
            labels = [
                validation_data.iloc[i]["Fracture"],
                validation_data.iloc[i]["Pneumothorax"],
                validation_data.iloc[i]["Airspace opacity"],
                validation_data.iloc[i]["Nodule or mass"],
            ]
            # covert YES to 1 and otherwise to 0
            labels = [1 if x == "YES" else 0 for x in labels]
            data_labels[validation_data.iloc[i]["Image Index"]] = labels
        for i in range(len(test_data)):
            labels = [
                test_data.iloc[i]["Fracture"],
                test_data.iloc[i]["Pneumothorax"],
                test_data.iloc[i]["Airspace opacity"],
                test_data.iloc[i]["Nodule or mass"],
            ]
            # covert YES to 1 and otherwise to 0
            labels = [1 if x == "YES" else 0 for x in labels]
            data_labels[test_data.iloc[i]["Image Index"]] = labels

        data_human_labels = {}
        for i in range(len(readers_data)):
            labels = [
                readers_data.iloc[i]["Fracture"],
                readers_data.iloc[i]["Pneumothorax"],
                readers_data.iloc[i]["Airspace opacity"],
                readers_data.iloc[i]["Nodule/mass"],
            ]
            # covert YES to 1 and otherwise to 0
            labels = [1 if x == "YES" else 0 for x in labels]
            if readers_data.iloc[i]["Image ID"] in data_human_labels:
                data_human_labels[readers_data.iloc[i]["Image ID"]].append(labels)
            else:
                data_human_labels[readers_data.iloc[i]["Image ID"]] = [labels]

        # for each key in data_human_labels, we have a list of lists, sample only one list from each key
        data_human_labels = {
            k: random.sample(v, 1)[0] for k, v in data_human_labels.items()
        }

        labels_categories = [
            "Fracture",
            "Pneumothorax",
            "Airspace opacity",
            "Nodule/mass",
        ]
        self.label_to_idx = {
            labels_categories[i]: i for i in range(len(labels_categories))
        }

        image_to_patient_id = {}
        for i in range(len(readers_data)):
            image_to_patient_id[readers_data.iloc[i]["Image ID"]] = readers_data.iloc[
                i
            ]["Patient ID"]

        patient_ids = list(set(image_to_patient_id.values()))

        data_all_nih_label = {}
        # the original dataset has the following labels ['Atelectasis' 'Cardiomegaly' 'Consolidation' 'Edema' 'Effusion' 'Emphysema' 'Fibrosis' 'Hernia' 'Infiltration' 'Mass' 'No Finding' 'Nodule' 'Pleural_Thickening' 'Pneumonia' 'Pneumothorax']
        for i in range(len(all_dataset_data)):
            if not all_dataset_data["Patient ID"][i] in patient_ids:
                labels = [0, 0, 0, 0]
                if "Pneumothorax" in all_dataset_data["Finding Labels"][i]:
                    labels[1] = 1
                if "Effusion" in all_dataset_data["Finding Labels"][i]:
                    labels[2] = 1
                if (
                    "Mass" in all_dataset_data["Finding Labels"][i]
                    or "Nodule" in all_dataset_data["Finding Labels"][i]
                ):
                    labels[3] = 1
                if "No Finding" in all_dataset_data["Finding Labels"][i]:
                    labels[0] = 0
                else:
                    labels[0] = 1
                data_all_nih_label[all_dataset_data["Image Index"][i]] = labels

        # depending on non_deferral_dataset
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
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
        if self.non_deferral_dataset == True:
            # iterate over key, value in data_all_nih_label
            data_y = []
            data_expert = []
            image_paths = []
            for key, value in list(data_all_nih_label.items()):
                image_path = self.data_dir + "/images_nih/" + key
                # check if the file exists
                if os.path.isfile(image_path):
                    data_y.append(value[self.label_chosen])
                    image_paths.append(self.data_dir + "/images_nih/" + key)
                    data_expert.append(value[self.label_chosen])  # nonsense expert

            data_y = np.array(data_y)
            data_expert = np.array(data_expert)
            image_paths = np.array(image_paths)

            random_seed = random.randrange(10000)

            test_size = int(self.test_split * len(image_paths))
            val_size = int(self.val_split * len(image_paths))
            train_size = len(image_paths) - test_size - val_size

            train_x, val_x, test_x = torch.utils.data.random_split(
                image_paths,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(random_seed),
            )
            train_y, val_y, test_y = torch.utils.data.random_split(
                data_y,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(random_seed),
            )
            train_h, val_h, test_h = torch.utils.data.random_split(
                data_expert,
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

        else:
            # split patient_ids into train and test, val
            random.shuffle(patient_ids, random.random)
            # split using 80% for trarain, 10% for test and 10% for validation
            train_patient_ids = patient_ids[: int(len(patient_ids) * self.train_split)]
            test_patient_ids = patient_ids[
                int(len(patient_ids) * self.train_split) : int(
                    len(patient_ids) * (self.train_split + self.test_split)
                )
            ]
            val_patient_ids = patient_ids[
                int(len(patient_ids) * (self.train_split + self.test_split)) :
            ]
            # go from patient ids to image ids
            train_image_ids = np.array(
                [k for k, v in image_to_patient_id.items() if v in train_patient_ids]
            )
            val_image_ids = np.array(
                [k for k, v in image_to_patient_id.items() if v in val_patient_ids]
            )
            test_image_ids = np.array(
                [k for k, v in image_to_patient_id.items() if v in test_patient_ids]
            )
            # remove images that are not in the directory
            train_image_ids = np.array(
                [
                    k
                    for k in train_image_ids
                    if os.path.isfile(self.data_dir + "/images_nih/" + k)
                ]
            )
            val_image_ids = np.array(
                [
                    k
                    for k in val_image_ids
                    if os.path.isfile(self.data_dir + "/images_nih/" + k)
                ]
            )
            test_image_ids = np.array(
                [
                    k
                    for k in test_image_ids
                    if os.path.isfile(self.data_dir + "/images_nih/" + k)
                ]
            )

            logging.info("Finished splitting data into train, test and validation")
            # print sizes
            logging.info("Train size: {}".format(len(train_image_ids)))
            logging.info("Test size: {}".format(len(test_image_ids)))
            logging.info("Validation size: {}".format(len(val_image_ids)))

            train_y = np.array(
                [data_labels[k][self.label_chosen] for k in train_image_ids]
            )
            val_y = np.array([data_labels[k][self.label_chosen] for k in val_image_ids])
            test_y = np.array(
                [data_labels[k][self.label_chosen] for k in test_image_ids]
            )
            train_h = np.array(
                [data_human_labels[k][self.label_chosen] for k in train_image_ids]
            )
            val_h = np.array(
                [data_human_labels[k][self.label_chosen] for k in val_image_ids]
            )
            test_h = np.array(
                [data_human_labels[k][self.label_chosen] for k in test_image_ids]
            )
            train_image_ids = np.array(
                [self.data_dir + "/images_nih/" + k for k in train_image_ids]
            )
            val_image_ids = np.array(
                [self.data_dir + "/images_nih/" + k for k in val_image_ids]
            )
            test_image_ids = np.array(
                [self.data_dir + "/images_nih/" + k for k in test_image_ids]
            )

            data_train = GenericImageExpertDataset(
                train_image_ids, train_y, train_h, transform_train, to_open=True
            )
            data_val = GenericImageExpertDataset(
                val_image_ids, val_y, val_h, transform_test, to_open=True
            )
            data_test = GenericImageExpertDataset(
                test_image_ids, test_y, test_h, transform_test, to_open=True
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
    
        if self.get_embeddings:

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            path_model = '../exp_data/models/chextxray_dn121_3epochs.pt'
            model_linear =  DenseNet121_CE(2).to(device)
            # torch load
            model_linear.load_state_dict(torch.load(path_model))

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
                torch.from_numpy(train_y),
                torch.from_numpy(train_h),
            )
            data_val = torch.utils.data.TensorDataset(
                val_embeddings,
                torch.from_numpy(val_y),
                torch.from_numpy(val_h),
            )
            data_test = torch.utils.data.TensorDataset(
                test_embeddings,
                torch.from_numpy(test_y),
                torch.from_numpy(test_h),
            )

            self.data_train_loader = torch.utils.data.DataLoader(
                data_train,
                batch_size=3000,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            self.data_val_loader = torch.utils.data.DataLoader(
                data_val,
                batch_size=3000,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            self.data_test_loader = torch.utils.data.DataLoader(
                data_test,
                batch_size=3000,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )


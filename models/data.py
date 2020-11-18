import pandas as pd
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../utils")
from features import mel, get_vggish_embedding
import augment as au
import torchvision.transforms as transforms

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


class LungDataset(Dataset):
    def __init__(self, label_file, base_dir, task, split="train", transform=None, train_prop=1.0, df=None):
        if df is None:
            df = pd.read_csv(label_file)
            splits_dir = os.path.join(base_dir, "splits")

            if split == "train":
                df = self.get_split(df, os.path.join(splits_dir, "train.txt"), train_prop=train_prop)
            elif split == "pre-train":
                df = self.get_split(df, os.path.join(splits_dir, "pre-train.txt"), train_prop=train_prop)
            elif split == "test":
                df = self.get_split(df, os.path.join(splits_dir, "test.txt"))
            else:
                raise Exception("Invalid split value. Must be train or test.")
        self.split = split
        self.task = task
        self.labels = df
        for idx, row in df.iterrows():
            df.at[idx, 'y'] = self.get_class_val(row)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        cycle = row["cycle"]
        filename = cycle + ".wav"
        # Read in the sound file
        dirs = ["COPD", "Healthy", "Other"]
        parent_path = self.base_dir + "/processed/"

        # Get correpsonding processed audio file (using Mel)
        X = None
        for d in dirs:
            file_path = parent_path + d + "/" + filename
            if os.path.isfile(file_path):
                X = file_path
                break
        if X is None:
            raise Exception(f"Could not find filename {filename} in {parent_path}.")
        # Get label
        y = self.get_class_val(row)
        return process_data(self.split, self.transform, X, y)

    def get_split(self, df, split_file_path, train_prop=1.0):
        # Takes in a DataFrame and a path to a file of only Ints denoting Patient ID.
        # Returns a DataFrame of only samples with Patient IDs contained in the split.
        IDs = set()
        with open(split_file_path, "r") as f:
            IDs = set([line.strip() for line in f])
            IDs = set(random.sample(IDs, int(train_prop * len(IDs))))
        return df[df.ID.isin(IDs)]

    def get_class_val(self, row):
        if self.task == "symptom":
            # Takes in a single row in a DataFrame, return an Int.
            # 0: None, 1: Wheeze, 2: Crackle, 3: Both
            wheeze = row["wheezes"]
            crackle = row["crackles"]
            if wheeze and crackle:
                return 3
            elif crackle:
                return 2
            elif wheeze:
                return 1
            else:
                return 0
        # elif self.task == "disease":
        #     # Takes in a row, return an Int.
        #     # 0: Healthy, 1: COPD, 2: Other
        #     label = row["diagnosis"]
        #     if label == 'Healthy':
        #         return 0
        #     elif label == 'COPD':
        #         return 1
        #     else:
        #         return 2
        elif self.task == "disease":
            # Takes in a single row in a DataFrame, return an Int.
            # 0: Abnormal, 1: Normal
            label = row["diagnosis"]
            if label == -1:
                return 1
            else:
                return 0

        elif self.task == "crackle":
            # Takes in a single row in a DataFrame, return an Int.
            # 0: Abnormal, 1: Normal
            label = row["diagnosis"]
            if label == -1:
                return 1
            else:
                return 0

        elif self.task == "wheeze":
            # Takes in a single row in a DataFrame, return an Int.
            # 0: Abnormal, 1: Normal
            label = row["diagnosis"]
            if label == -1:
                return 1
            else:
                return 0


class HeartDataset(Dataset):

    def __init__(self, label_file, base_dir, task, split="train", transform=None, train_prop=1.0, df=None):
        if df is None:
            df = pd.read_csv(label_file)
            splits_dir = os.path.join(base_dir, "splits")

            if split == "train":
                df = self.get_split(df, os.path.join(splits_dir, "train.txt"), train_prop=train_prop)
            elif split == "pre-train":
                df = self.get_split(df, os.path.join(splits_dir, "pre-train.txt"), train_prop=train_prop)
            elif split == "test":
                df = self.get_split(df, os.path.join(splits_dir, "test.txt"))
            else:
                raise Exception("Invalid split value. Must be train or test.")
        self.split = split
        self.task = task
        self.labels = df
        for idx, row in df.iterrows():
            df.at[idx, 'y'] = self.get_class_val(row)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        cycle = row["ID"]
        filename = cycle + ".wav"
        # Read in the sound file
        dirs = ["Normal", "Abnormal"]
        parent_path = self.base_dir + "/processed/"

        # Get correpsonding processed audio file (using Mel)
        X = None
        for d in dirs:
            file_path = parent_path + d + "/" + filename
            if os.path.isfile(file_path):
                X = file_path
                break
        if X is None:
            raise Exception(f"Could not find filename {filename} in {parent_path}.")
        # Get label
        y = self.get_class_val(row)
        return process_data(self.split, self.transform, X, y)

    def get_split(self, df, split_file_path, train_prop=1.0):
        # Takes in a DataFrame and a path to a file of only Ints denoting Patient ID.
        # Returns a DataFrame of only samples with Patient IDs contained in the split.
        IDs = set()
        with open(split_file_path, "r") as f:
            IDs = set([line.strip() for line in f])
            IDs = set(random.sample(IDs, int(train_prop * len(IDs))))
        return df[df.ID.isin(IDs)]

    def get_class_val(self, row):
        if self.task == "heart":
            # Takes in a single row in a DataFrame, return an Int.
            # 0: Abnormal, 1: Normal
            label = row["label"]
            if label == -1:
                return 1
            else:
                return 0


class HeartChallengeDataset(Dataset):
    def __init__(self, label_file, base_dir, task, split="train", transform=None, train_prop=1.0, df=None):
        if df is None:
            df = pd.read_csv(label_file)
            splits_dir = os.path.join(base_dir, "splits")

            if split == "train":
                df = self.get_split(df, os.path.join(splits_dir, "train.txt"), train_prop=train_prop)
            elif split == "pre-train":
                df = self.get_split(df, os.path.join(splits_dir, "pre-train.txt"), train_prop=train_prop)
            elif split == "test":
                df = self.get_split(df, os.path.join(splits_dir, "test.txt"))
            else:
                raise Exception("Invalid split value. Must be train or test.")
        self.split = split
        self.task = task
        self.labels = df
        for idx, row in df.iterrows():
            df.at[idx, 'y'] = self.get_class_val(row)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        cycle = row["ID"]
        filename = cycle + ".wav"
        # Read in the sound file
        dirs = ["Normal", "Artifact", "Extrasound", "Extrastole", "Murmur"]
        parent_path = self.base_dir + "/processed/"

        # Get correpsonding processed audio file (using Mel)
        X = None
        for d in dirs:
            file_path = parent_path + d + "/" + filename
            if os.path.isfile(file_path):
                X = file_path
                break
        if X is None:
            raise Exception(f"Could not find filename {filename} in {parent_path}.")
        # Get label
        y = self.get_class_val(row)
        return process_data(self.split, self.transform, X, y)

    def get_split(self, df, split_file_path, train_prop=1.0):
        # Takes in a DataFrame and a path to a file of only Ints denoting Patient ID.
        # Returns a DataFrame of only samples with Patient IDs contained in the split.
        IDs = set()
        with open(split_file_path, "r") as f:
            IDs = set([line.strip() for line in f])
            IDs = set(random.sample(IDs, int(train_prop * len(IDs))))
        return df[df.ID.isin(IDs)]

    def get_class_val(self, row):
        if self.task == "heartchallenge":
            # # Takes in a single row in a DataFrame, return an Int.
            # # 0: None, 1: Wheeze, 2: Crackle, 3: Both
            # label = row["label"]
            # if label == "Artifact":
            #     return 4
            # elif label == "Extrasound":
            #     return 3
            # elif label == "Extrastole":
            #     return 2
            # elif label == "Murmur":
            #     return 1
            # else:
            #     return 0
            # Takes in a single row in a DataFrame, return an Int.
            # 0: Abnormal, 1: Normal
            label = row["label"]
            if label == -1:
                return 1
            else:
                return 0

def transform(augment, X):
    
    if augment is None:
        return torch.Tensor(mel(X))
    if augment == "split":
        split = au.Split()
        return torch.Tensor(split(mel(X)))
    if augment == "spec":
        spec = au.SpectralAugment()
        return torch.Tensor(spec(mel(X)))
    if augment == "raw":
        raw = au.RawAugment()
        return torch.Tensor(mel(X, raw))
    if augment == "raw+spec":
        raw = au.RawAugment()
        spec = au.SpectralAugment()
        return torch.Tensor(spec(mel(X, raw)))


def process_data(mode, augment, X, y):
    if mode== "pre-train":
        xi = transform(augment, X)
        xj = transform(augment, X)
        return xi, xj

    X = transform(augment, X)
    return X, y



def get_dataset(task, label_file, base_dir, split="train", train_prop=1.0, df=None, transform=None):
    dataset = []
    if task == "crackle" or task == "disease" or task == "wheeze":
        dataset = LungDataset(label_file, base_dir, task, split=split, transform=transform, train_prop=train_prop,
                              df=df)
    elif task == "heart":
        dataset = HeartDataset(label_file, base_dir, task, split=split, transform=transform,
                               train_prop=train_prop,
                               df=df)
    elif task == "heartchallenge":
        dataset = HeartChallengeDataset(label_file, base_dir, task, split=split, transform=transform,
                                        train_prop=train_prop, df=df)
    return dataset


def get_data_loader(task, label_file, base_dir, batch_size=128, split="train", transform=None, df=None):
    dataset = get_dataset(task, label_file, base_dir, split=split, df=df, transform=transform)
    shuffle = True
    if split == "test":
        shuffle = False
    return DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True)


def get_scikit_loader(device, task, label_file, base_dir, split="train", df=None, encoder=None):
    dataset = get_dataset(task, label_file, base_dir, split=split, df=df)
    X = []
    y = []
    for data in dataset:
        x = data[0]
        if encoder is not None:
            x = x.view(1, 1, data[0].shape[0], data[0].shape[1]).to(device)
            x = encoder(x)
        X.append(x.cpu().detach().numpy())
        y.append(data[1])
    return X, y

import pandas as pd
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../utils")
from features import Mel, get_vggish_embedding, preprocess
import augment as au
import torchvision.transforms as transforms
import time
import numpy as np
import h5py as h5

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


class LungDataset(Dataset):
    def __init__(self, label_file, base_dir, task, split="train", transform=None, train_prop=1.0, df=None, data=None):
        if df is None:
            df = pd.read_csv(label_file)
            splits_dir = os.path.join(base_dir, "splits")

            if split == "train":
                df = self.get_split(df, os.path.join(splits_dir, "train.txt"), train_prop=train_prop)
            elif split == "pretrain":
                df = self.get_split(df, os.path.join(splits_dir, "pretrain.txt"), train_prop=train_prop)
            elif split == "test":
                df = self.get_split(df, os.path.join(splits_dir, "test.txt"))
            else:
                raise Exception("Invalid split value. Must be train or test.")
        if data is None:
            try:
                file=base_dir+'/processed/'+task+'_1.0.h5'
                file = h5.File(file,'r')
                data = file[split][df.index.tolist()]
            except:
                raise Exception("Data not found")
        self.data=data
        self.split = split
        self.task = task
        self.labels = df
        for idx, row in df.iterrows():
            df.at[idx, 'y'] = self.get_class_val(row)
        self.base_dir = base_dir
        self.transform = get_transform(transform)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        X=self.data[idx]
        y = self.get_class_val(row)
        if self.split == "test":
            X, y = process_data(self.split, self.transform, X, y)
            return row["cycle"], X, y
        return process_data(self.split, self.transform, X, y)

    def get_split(self, df, split_file_path, train_prop=1.0):
        # Takes in a DataFrame and a path to a file of only Ints denoting Patient ID.
        # Returns a DataFrame of only samples with Patient IDs contained in the split.
        IDs = set()
        with open(split_file_path, "r") as f:
            IDs = set([line.strip() for line in f])
            df=df[df.ID.isin(IDs)]
            df=df.reset_index()
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
                return 0
            else:
                return 1

        elif self.task == "crackle":
            # Takes in a single row in a DataFrame, return an Int.
            # 0: Abnormal, 1: Normal
            label = row["diagnosis"]
            if label == -1:
                return 0
            else:
                return 1

        elif self.task == "wheeze":
            # Takes in a single row in a DataFrame, return an Int.
            # 0: Abnormal, 1: Normal
            label = row["diagnosis"]
            if label == -1:
                return 0
            else:
                return 1


class HeartDataset(Dataset):

    def __init__(self, label_file, base_dir, task, split="train", transform=None, train_prop=1.0, df=None, data=None):
        if df is None:
            df = pd.read_csv(label_file)
            splits_dir = os.path.join(base_dir, "splits")

            if split == "train":
                df = self.get_split(df, os.path.join(splits_dir, "train.txt"), train_prop=train_prop)
            elif split == "pretrain":
                df = self.get_split(df, os.path.join(splits_dir, "pretrain.txt"), train_prop=train_prop)
            elif split == "test":
                df = self.get_split(df, os.path.join(splits_dir, "test.txt"))
            else:
                raise Exception("Invalid split value. Must be pretrain or train or test.")
        if data is None:
            try:
                file=base_dir+'/processed/heart_1.0.h5'
                file = h5.File(file,'r')
                data = file[split][df.index.tolist()]
            except:
                raise Exception("Data not found")
        self.data=data
        self.split = split
        self.task = task
        self.labels = df
        for idx, row in df.iterrows():
            df.at[idx, 'y'] = self.get_class_val(row)
        self.base_dir = base_dir
        self.transform = get_transform(transform)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        X = self.data[idx]
        # Get label
        y = self.get_class_val(row)
        if self.split == "test":
            X, y = process_data(self.split, self.transform, X, y)
            return row["ID"], X, y
        return process_data(self.split, self.transform, X, y)

    def get_split(self, df, split_file_path, train_prop=1.0):
        # Takes in a DataFrame and a path to a file of only Ints denoting Patient ID.
        # Returns a DataFrame of only samples with Patient IDs contained in the split.
        IDs = set()
        with open(split_file_path, "r") as f:
            IDs = set([line.strip() for line in f])
            df=df[df.ID.isin(IDs)]
            df=df.reset_index()
            IDs = set(random.sample(IDs, int(train_prop * len(IDs))))
        return df[df.ID.isin(IDs)]

    def get_class_val(self, row):
        if self.task == "heart":
            # Takes in a single row in a DataFrame, return an Int.
            # 0: Abnormal, 1: Normal
            label = row["label"]
            if label == -1:
                return 0
            else:
                return 1


class HeartChallengeDataset(Dataset):
    def __init__(self, label_file, base_dir, task, split="train", transform=None, train_prop=1.0, df=None, data=None):
        if df is None:
            df = pd.read_csv(label_file)
            splits_dir = os.path.join(base_dir, "splits")

            if split == "train":
                df = self.get_split(df, os.path.join(splits_dir, "train.txt"), train_prop=train_prop)
            elif split == "pretrain":
                df = self.get_split(df, os.path.join(splits_dir, "pretrain.txt"), train_prop=train_prop)
            elif split == "test":
                df = self.get_split(df, os.path.join(splits_dir, "test.txt"))
            else:
                raise Exception("Invalid split value. Must be train or test.")
        if data is None:
            try:
                file=base_dir+'/processed/heartchallenge_1.0.h5'
                file = h5.File(file,'r')
                data = file[split][df.index.tolist()]
            except:
                raise Exception("Data not found")
        self.data=data

        self.split = split
        self.task = task
        self.labels = df
        for idx, row in df.iterrows():
            df.at[idx, 'y'] = self.get_class_val(row)
        self.base_dir = base_dir

        self.transform = get_transform(transform)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        X=self.data[idx]
        # Get label
        y = self.get_class_val(row)
        if self.split == "test":
            X, y = process_data(self.split, self.transform, X, y)
            return row["ID"], X, y
        return process_data(self.split, self.transform, X, y)

    def get_split(self, df, split_file_path, train_prop=1.0):
        # Takes in a DataFrame and a path to a file of only Ints denoting Patient ID.
        # Returns a DataFrame of only samples with Patient IDs contained in the split.
        IDs = set()
        with open(split_file_path, "r") as f:
            IDs = set([line.strip() for line in f])
            df=df[df.ID.isin(IDs)]
            df=df.reset_index()
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
                return 0
            else:
                return 1


def get_transform(augment=None):
    if augment is None:
        mel = Mel()
        return mel
    if augment == "split":
        mel = Mel()
        split = au.Split(mel)
        return split
    if augment == "spec":
        mel = Mel()
        spec = au.SpectralAugment(mel)
        return spec
    if augment == "raw":
        raw = au.RawAugment()
        mel = Mel(raw)
        return mel
    if augment == "spec+split":
        mel = Mel()
        spec = au.SpectralAugment(mel)
        split = au.Split(spec)
        return split


def process_data(mode, augment, X, y):
    if mode == "pretrain":
        xi = torch.Tensor(augment(X))
        xj = torch.Tensor(augment(X))
        return xi, xj

    X = torch.Tensor(augment(X))
    return X, y


def get_dataset(task, label_file, base_dir, split="train", train_prop=1.0, df=None, transform=None, data=None):
    dataset = []
    if task == "crackle" or task == "disease" or task == "wheeze":
        dataset = LungDataset(label_file, base_dir, task, split=split, transform=transform, train_prop=train_prop,
                              df=df, data=data)
    elif task == "heart":
        dataset = HeartDataset(label_file, base_dir, task, split=split, transform=transform,
                               train_prop=train_prop, df=df, data=data)
    elif task == "heartchallenge":
        dataset = HeartChallengeDataset(label_file, base_dir, task, split=split, transform=transform,
                                        train_prop=train_prop, df=df, data=data)
    return dataset


def get_data_loader(task, label_file, base_dir, batch_size=128, split="train", transform=None, df=None, data=None):
    dataset = get_dataset(task, label_file, base_dir, split=split, df=df, transform=transform, data=data)
    shuffle = True
    if split == "test":
        shuffle = False
    return DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True)


def get_scikit_loader(device, task, label_file, base_dir, split="train", df=None, encoder=None, data=None):
    dataset = get_dataset(task, label_file, base_dir, split=split, df=df, data=data)
    X = []
    y = []
    id=[]
    for data in dataset:
        if split == "test":
            x = data[1]
            y.append(data[2])
            id.append(data[0])
        else:
            y.append(data[1])
            x=data[0]
        if encoder is not None:
            x = x.view(1, 1, x.shape[0], x.shape[1]).to(device)
            x = encoder(x)
        X.append(x.cpu().detach().numpy())

    if split == "test":
        return id, X, y
    return X, y


def h5ify(base_dir, label_file, train_prop):
    def get_split(df, split_file_path, train_prop=1.0):
        # Takes in a DataFrame and a path to a file of only Ints denoting Patient ID.
        # Returns a DataFrame of only samples with Patient IDs contained in the split.
        IDs = set()
        with open(split_file_path, "r") as f:
            IDs = set([line.strip() for line in f])
            IDs = set(random.sample(IDs, int(train_prop * len(IDs))))
        return df[df.ID.isin(IDs)]

    def get_data(cycle):
        filename = cycle + ".wav"
        # Read in the sound file
        dirs = ["Normal", "Abnormal"]
        parent_path = base_dir + "/processed/"

        # Get correpsonding processed audio file (using Mel)
        X = None
        for d in dirs:
            file_path = parent_path + d + "/" + filename
            if os.path.isfile(file_path):
                X = file_path
                break
        if X is None:
            raise Exception(f"Could not find filename {filename} in {parent_path}.")
        return preprocess(X)

    filename = task + "_" + str(train_prop) + ".h5"
    print('building: ' + filename)

    splits_dir = os.path.join(base_dir, "splits")
    __splits__ = ['train', 'test', 'pretrain']
    h5_dir = base_dir + "/processed/" + filename
    with h5.File(h5_dir,'w') as f:
        for split in __splits__:
            audio_samples = []
            try:
                df = pd.read_csv(label_file)
                if split == "train":
                    df = get_split(df, os.path.join(splits_dir, "train.txt"), train_prop=train_prop)
                elif split == "pretrain":
                    df = get_split(df, os.path.join(splits_dir, "pretrain.txt"), train_prop=train_prop)
                elif split == "test":
                    df = get_split(df, os.path.join(splits_dir, "test.txt"))
                else:
                    raise Exception("Invalid split value. Must be train or test.")
            except:
                print('split not found and will not be added to archive')

            if 'cycle' in df.columns:
                for idx, row in df.iterrows():
                    audio, sample_rate = get_data(row['cycle'])
                    audio_samples.append(audio)

            else:
                for idx, row in df.iterrows():
                    audio, sample_rate = get_data(row['ID'])
                    audio_samples.append(audio)
            audio_samples=np.array(audio_samples)
            f.create_dataset(split,data=audio_samples)


if __name__ == '__main__':
    __tasks__ = ['heart', 'disease', 'crackle', 'wheeze', 'heartchallenge']
    __train_props__ = [.01, .1, 1.0]
    for task in __tasks__:
        if task == 'disease' or task == 'crackle' or task == 'wheeze':
            base_dir = '../data'
        else:
            base_dir = '../' + task
        label_file = os.path.join(base_dir, "processed", "{}_labels.csv".format(task))
        for train_prop in __train_props__:
            h5ify(base_dir, label_file, train_prop)

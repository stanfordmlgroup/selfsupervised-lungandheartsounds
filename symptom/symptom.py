import os
import time
import numpy as np
import pandas as pd
import datetime
import argparse

import sys

sys.path.append("../utils")

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, MSELoss, Conv2d, LeakyReLU, MaxPool2d, Dropout
from torch.utils.data import Dataset, DataLoader

from features import mel

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)

# TODO: K FOLD CROSS VAL
# TODO: K FOLD CROSS VAL
# TODO: K FOLD CROSS VAL
# TODO: K FOLD CROSS VAL


class SymptomDataset(Dataset):
    def __init__(self, label_file, base_dir, split="train", transform=None):
        df = pd.read_csv(label_file)
        splits_dir = os.path.join(base_dir, "splits")

        if split == "train":
            df = self.get_split(df, os.path.join(splits_dir, "train.txt"))
        elif split == "test":
            df = self.get_split(df, os.path.join(splits_dir, "test.txt"))
        else:
            raise Exception("Invalid split value. Must be train or test.")

        self.labels = df
        self.base_dir = base_dir

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
                X = torch.Tensor(mel(file_path))
                break
        if X is None:
            raise Exception(f"Could not find filename {filename} in {parent_path}.")

        # Get symptom result
        y = self.get_class_val(row)
        return X, y

    def get_split(self, df, split_file_path):
        # Takes in a DataFrame and a path to a file of only Ints denoting Patient ID.
        # Returns a DataFrame of only samples with Patient IDs contained in the split.
        IDs = set()
        with open(split_file_path, "r") as f:
            IDs = set([line.strip() for line in f])
        return df[~df.ID.isin(IDs)]

    def get_class_val(self, row):
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


def get_data_loader(label_file, base_dir, batch_size=128, split="train"):
    dataset = SymptomDataset(label_file, base_dir, split=split)
    return DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_layers = Sequential(
            Conv2d(1, 128, kernel_size=[7,11], stride=2, padding=1),
            LeakyReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(128, 256, kernel_size=5, padding=1),
            LeakyReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(256, 256, kernel_size=1, padding=1),
            Conv2d(256, 256, kernel_size=3, padding=1),
            LeakyReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(256, 512, kernel_size=1, padding=1),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=1, padding=1),
            Conv2d(512, 512, kernel_size=3, padding=1),
            LeakyReLU(inplace=True),
            MaxPool2d(2),
        )

        self.linear_layers = Sequential(
            Linear(30720, 4096), ReLU(inplace=True), Dropout(0.5), Linear(4096, 512), ReLU(inplace=True), Linear(512, 4)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def train(epoch, arch, model, loader, optimizer, device):
    model.train()

    y_true = []
    y_pred = []

    for i, data in enumerate(loader):
        print(i)
        X, y = data
        X, y = X.view(128, 1, 259, 128).to(device), y.to(device)

        optimizer.zero_grad()
        if arch == "CNN":
            output = model(X)
        loss = F.cross_entropy(output, y)
        y_true.extend(y.tolist())
        y_pred.extend(output.tolist())
        loss.backward()
        optimizer.step()

    return loss, y_true, y_pred


@torch.no_grad()
def test(arch, model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    for i, data in enumerate(loader):
        X, y = data
        X, y = X.view(128, 1, 8, 16).to(device), y.to(device)

        if arch == "CNN":
            output = model(X)

        loss = F.cross_entropy(output, y)
        y_true.extend(y.tolist())
        y_pred.extend(output.tolist())

    return loss, y_true, y_pred


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)


def get_accuracy(labels, preds):
    return sum([np.argmax(pred) == label for label, pred in zip(labels, preds)]) / len(labels)


def train_(architecture, base_dir, device, log_dir, seed=None, test_mode=False):
    log_file = os.path.join(log_dir, f"train_log.txt")

    num_epochs = 500
    batch_size = 128
    learning_rate = 0.001
    label_file = os.path.join(base_dir, "processed", "symptoms_labels.csv")

    train_loader = get_data_loader(label_file, base_dir, batch_size=batch_size)
    test_loader = get_data_loader(label_file, base_dir, batch_size=batch_size, split="test")

    if not os.path.exists(os.path.join(log_dir, "params.txt")):
        with open(os.path.join(log_dir, "params.txt"), "w") as f:
            f.write(f"Model: {architecture}\n")
            f.write(f"Epochs: {num_epochs}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Learning rate: {learning_rate}")

    if architecture == "CNN":
        model = CNN().to(device)

    model.to(device)
    best_train_loss = 999
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss, train_true, train_pred = train(epoch, architecture, model, train_loader, optimizer, device)
        if train_loss < best_train_loss:
            save_weights(model, os.path.join(log_dir, "best_weights.pt"))
            best_train_loss = train_loss
        elapsed = time.time() - start
        print("Epoch: {:03d}, Time: {:.3f} s".format(epoch, elapsed))
        print("\tTrain CE: {:.7f}".format(train_loss))
        ce, test_true, test_pred = test(architecture, model, test_loader, device)
        train_accuracy = get_accuracy(train_true, train_pred)
        test_accuracy = get_accuracy(test_true, test_pred)
        print("\tTrain Acc: {:.7f}\tTest Acc: {:.7f}\n".format(train_accuracy, test_accuracy))
        with open(log_file, "a+") as log:
            log.write(
                "Epoch: {:03d}\tLoss: {:.7f}\tTrain Acc: {:.7f}\tTest Acc: {:.7f}\n".format(
                    epoch, train_loss, train_accuracy, test_accuracy
                )
            )

    if test_mode:
        test_file = os.path.join(log_dir, f"test_results.txt")
        model.load_state_dict(torch.load(os.path.join(log_dir, "best_weights.pt")))
        ce, y_true, y_pred = test(architecture, model, test_loader, device)
        print("Test CE: {:.7f}".format(ce))
        with open(test_file, "a+") as out:
            out.write("{}\t{:.7f}\n".format(seed, ce))

    return best_train_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices={"train", "test"})
    parser.add_argument("--architecture", type=str, default="CNN", choices={"CNN"})
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    base_dir = os.getcwd() + "/../data"
    log_dir = args.log_dir

    if args.mode == "train":
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(base_dir, "logs", now)
        else:
            log_dir = os.path.join(base_dir, "logs", log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train_(args.architecture, base_dir, device, log_dir)
    elif args.mode == "test":
        for seed in np.random.randint(0, 1000, size=3):
            print("seed:", seed)
            log_dir = os.path.join(base_dir, "logs", f"test_{seed}")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train_(args.architecture, base_dir, device, log_dir, seed, test_mode=True)

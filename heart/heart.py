import os
import time
import numpy as np
import pandas as pd
import datetime
import argparse
import sys
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, MSELoss, Conv2d, LeakyReLU, MaxPool2d, Dropout
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report

sys.path.append("../utils")
from features import mel
from glob import glob
from scipy.special import softmax

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


max_len = 259


class HeartDataset(Dataset):
    def __init__(self, label_file, base_dir, task, split="train", df=None):
        if df is None:
            df = pd.read_csv(label_file)
            splits_dir = os.path.join(base_dir, "splits")
            if split == "train":
                df = self.get_split(df, os.path.join(splits_dir, "train.txt"))
            elif split == "test":
                df = self.get_split(df, os.path.join(splits_dir, "validation.txt")) #TODO get real test set
            else:
                raise Exception("Invalid split value. Must be train or test.")
        self.task = task
        self.labels = df
        for idx, row in df.iterrows():
            df.at[idx, 'y'] = self.get_class_val(row)
        self.base_dir = base_dir
        # print(f'{df=}')
        # print(f'{task=}')
        # print(f'{self.labels =}')
        # print(f'{base_dir=}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        ID = row["ID"]
        filename = ID + ".wav"
        group = ID[0]
        # Read in the sound file
        parent_path = os.path.join(self.base_dir, f"training-{group}")

        # Get correpsonding processed audio file (using Mel)
        file_path = parent_path + "/" + filename
        target = torch.zeros(max_len, 128)
        source = torch.Tensor(mel(file_path)) if os.path.isfile(file_path) else None
        if source is None:
            raise Exception(f"Could not find filename {filename} in {parent_path}.")
        target[:min(source.shape[0],max_len), :] = source[:min(source.shape[0],max_len), :]
        X = target
        # Get label
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
        if self.task == "heart":
            # 0: Normal, 1: Abnormal
            val = row["abnormal"]
            if val == -1:
                return 0
            else:
                return 1



class LungDataset(Dataset):
    def __init__(self, label_file, base_dir, task, split="train", df=None):
        if df is None:
            df = pd.read_csv(label_file)
            splits_dir = os.path.join(base_dir, "splits")
            if split == "train":
                df = self.get_split(df, os.path.join(splits_dir, "train.txt"))
            elif split == "test":
                df = self.get_split(df, os.path.join(splits_dir, "test.txt"))
            else:
                raise Exception("Invalid split value. Must be train or test.")
        self.task = task
        self.labels = df
        for idx, row in df.iterrows():
            df.at[idx, 'y'] = self.get_class_val(row)
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

        # Get label
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
        elif self.task == "disease":
            # Takes in a row, return an Int.
            # 0: Healthy, 1: COPD, 2: Other
            label = row["diagnosis"]
            if label == 'Healthy':
                return 2
            elif label == 'COPD':
                return 1
            else:
                return 0


def get_data_loader(task, label_file, base_dir, batch_size=128, split="train", df=None):
    dataset = None
    if task == "heart":
        dataset = HeartDataset(os.path.join(base_dir, "ALL_REFERENCE.csv"), base_dir, task, split=split, df=df)
    else:
        dataset = LungDataset(label_file, base_dir, task, split=split, df=df)
    return DataLoader(dataset, batch_size, shuffle=True, drop_last=True) # Careful with drop_last, make sure no empty data loader returned, decrease batch size if needed


class CNN(torch.nn.Module):
    def __init__(self, classes):
        super(CNN, self).__init__()
        self.cnn_layers = Sequential(
            Conv2d(1, 128, kernel_size=[7, 11], stride=2, padding=1),
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
            Linear(30720, 4096), ReLU(inplace=True), Dropout(0.5), Linear(4096, 512), ReLU(inplace=True),
            Linear(512, classes)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def train(epoch, arch, task, model, loader, optimizer, device):
    model.train()

    y_true = []
    y_pred = []
    loss = None
    for i, data in enumerate(loader):
        X, y = data
        if task == "heart":
            X, y = X.view(16, 1, max_len, 128).to(device), y.to(device) #TODO change if heart tensor sizes ever
        else:
            X, y = X.view(128, 1, 259, 128).to(device), y.to(device)

        optimizer.zero_grad()
        if arch == "CNN":
            output = model(X)
        
        loss = F.cross_entropy(output, y)
        y_true.extend(y.tolist())
        y_pred.extend(output.tolist())
        loss.backward()
        optimizer.step()


    # ce = F.cross_entropy(torch.tensor(y_pred), torch.tensor(y_true))

    return loss, y_true, y_pred # ce, y_true, y_pred


@torch.no_grad()
def test(arch, task, model, loader, device):
    model.eval()

    y_true = []
    y_pred = []
    loss = None
    # print(f'{loader=}')
    for i, data in enumerate(loader):
        X, y = data
        if task == "heart":
            X, y = X.view(16, 1, max_len, 128).to(device), y.to(device)
        else:
            X, y = X.view(128, 1, 259, 128).to(device), y.to(device)
        if arch == "CNN":
            output = model(X)
        loss = F.cross_entropy(output, y)
        y_true.extend(y.tolist())
        y_pred.extend(output.tolist())
    #     print(f'TEST {y=}')
    #     print(f'TEST {output=}')
    #     print(f'TEST {y_pred=}')
    #     print(f'TEST {y_true=}')

    # print(f'TEST RET {y_pred=}')
    # print(f'TEST RET {y_true=}')
    # ce = F.cross_entropy(torch.tensor(y_pred), torch.tensor(y_true))

    return loss, y_true, y_pred # ce, y_true, y_pred


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)


def weights_init(m):
    if isinstance(m, Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)


def get_accuracy(labels, preds):
    # print(f'ACC {labels=}')
    # print(f'ACC {preds=}')
    return sum([np.argmax(pred) == label for label, pred in zip(labels, preds)]) / len(labels)


def train_(task, architecture, base_dir, device, log_dir, folds=5):
    log_file = os.path.join(log_dir, f"train_log.txt")

    n_splits = folds
    num_epochs = 50
    batch_size = 16
    learning_rate = 0.001
    label_file = os.path.join(base_dir, "processed", "{}_labels.csv".format(task))

    if not os.path.exists(os.path.join(log_dir, "params.txt")):
        with open(os.path.join(log_dir, "params.txt"), "w") as f:
            f.write(f"Model: {architecture}\n")
            f.write(f"Folds: {n_splits}\n")
            f.write(f"Epochs: {num_epochs}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Learning rate: {learning_rate}")

    classes = -1
    if task == "heart":
        classes = 2
    elif task == "disease":
        classes = 3
    elif task == "symptom":
        classes = 4

    if architecture == "CNN":
        model = CNN(classes=classes).to(device)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = None
    if task == "heart":
        dataset = HeartDataset(os.path.join(base_dir, "ALL_REFERENCE.csv"), base_dir, task, split="train")
    else:
        dataset = LungDataset(label_file, base_dir, task, split="train")
    df = dataset.labels
    total_train_acc = 0
    total_test_acc = 0
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=345)
    for fold, (train_idx, test_idx) in enumerate(kf.split(df, df["y"])):
        best_test_loss = 999

        model.apply(weights_init)
        start_fold = time.time()
        print("Fold: {:03d}".format(fold))
        with open(log_file, "a+") as log:
            log.write("Fold: {:03d}".format(fold))

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        # print(f'{test_df=}')
        train_loader = get_data_loader(task, label_file, base_dir, batch_size=batch_size, df=train_df)
        test_loader = get_data_loader(task, label_file, base_dir, batch_size=batch_size, df=test_df)

        fold_train_acc = 0
        fold_test_acc = 0
        for epoch in range(1, num_epochs + 1):
            start = time.time()
            train_loss, train_true, train_pred = train(epoch, architecture, task, model, train_loader, optimizer, device)
            test_loss, test_true, test_pred = test(architecture, task, model, test_loader, device)
            train_accuracy = get_accuracy(train_true, train_pred)
            test_accuracy = get_accuracy(test_true, test_pred)
            if test_loss < best_test_loss:
                save_weights(model, os.path.join(log_dir, "best_weights_" + str(fold) + ".pt"))
                best_test_loss = test_loss
            elapsed = time.time() - start
            print("\tEpoch: {:03d}, Time: {:.3f} s".format(epoch, elapsed))
            print("\t\tTrain CE: {:.7f}\tVal CE: {:.7f}".format(train_loss, test_loss))
            print("\t\tTrain Acc: {:.7f}\tVal Acc: {:.7f}\n".format(train_accuracy, test_accuracy))
            with open(log_file, "a+") as log:
                log.write(
                    "\tEpoch: {:03d}\tTrain Loss: {:.7f}\tVal Loss: {:.7f}\tTrain Acc: {:.7f}\tVal Acc: {:.7f}\n".format(
                        epoch, train_loss, test_loss, train_accuracy, test_accuracy
                    )
                )

            fold_train_acc += train_accuracy
            fold_test_acc += test_accuracy

        elapsed_fold = time.time() - start_fold

        fold_train_acc /= float(num_epochs)
        fold_test_acc /= float(num_epochs)
        total_train_acc += fold_train_acc
        total_test_acc += fold_test_acc
        print(
            "Fold: {:03d}, Time: {:.3f} s\tFold Train Acc: {:.7f}\tFold Val Acc: {:.7f}\n".format(
                fold, elapsed_fold, fold_train_acc, fold_test_acc
            )
        )
        with open(log_file, "a+") as log:
            log.write(
                "Fold: {:03d}, Time: {:.3f} s\tFold Train Acc: {:.7f}\tFold Val Acc: {:.7f}\n".format(
                    fold, elapsed_fold, fold_train_acc, fold_test_acc
                )
            )

    total_train_acc /= float(n_splits)
    total_test_acc /= float(n_splits)
    print(
        "Total Cross Val Train Acc: {:.7f}\tTotal Cross Val Test Acc: {:.7f}\n".format(total_train_acc, total_test_acc))
    with open(log_file, "a+") as log:
        log.write(
            "Total Cross Val Train Acc: {:.7f}\tTotal Cross Val Test Acc: {:.7f}\n".format(
                total_train_acc, total_test_acc
            )
        )

    return best_test_loss


def test_(task, architecture, base_dir, device, log_dir, seed=None):
    batch_size = 16
    label_file = os.path.join(base_dir, "processed", task + "_labels.csv")

    whole_test_loader = get_data_loader(task, label_file, base_dir, batch_size=batch_size, split="test")
    classes = -1
    if task == "heart":
        classes = 2
        labels = ["Normal", "Abnormal"]
    elif task == "disease":
        classes = 3
        labels = ["Other", "COPD", "Healthy"]
    elif task == "symptom":
        classes = 4
        labels = ["None", "Wheezes", "Crackles", "Both"]

    if architecture == "CNN":
        model = CNN(classes=classes).to(device)

    model.to(device)
    test_file = os.path.join(log_dir, f"test_results.txt")
    _y_pred = []
    with open(test_file, "w+") as out:

        for i, model_weights in enumerate(glob(os.path.join(os.path.join(log_dir, "best_weights_*.pt")))):
            model.load_state_dict(torch.load(model_weights))
            ce, y_true, y_pred = test(architecture, task, model, whole_test_loader, device)
            print()
            _y_pred.append(y_pred)
            print("Model {} Test CE: {:.7f}".format(i, ce))
            out.write("Model {} Test CE: {:.7f}\n".format(i, ce))

        _y_pred = np.average(_y_pred, axis=0)
        loss = get_accuracy(y_true, _y_pred)
        roc_score = roc_auc_score(y_true, softmax(_y_pred, axis=1), multi_class="ovr")
        report = classification_report(y_true, np.argmax(_y_pred, axis=1), target_names=labels)
        print("Ensemble Accuracy {:.7f}\nEnsemble AUC-ROC {:.7f}\n{}\n".format(loss, roc_score, report))
        out.write(
                "Seed: {}\tFolds: {}\nEnsemble Accuracy {:.7f}\nEnsemble AUC-ROC {:.7f}\n{}\n".format(seed, i, loss, roc_score, report))

    dataset_train = HeartDataset(label_file, base_dir, task, split="train") if task == "heart" else LungDataset(label_file, base_dir, task, split="train")
    dataset_test = HeartDataset(label_file, base_dir, task, split="test") if task == "heart" else LungDataset(label_file, base_dir, task, split="test")

    return ce


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices={"train", "test"})
    parser.add_argument("--architecture", type=str, default="CNN", choices={"CNN"})
    parser.add_argument("--task", type=str, default=None, choices={"symptom", "disease", "heart"})
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--data", type=str, default="../data")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    base_dir = os.path.join(os.getcwd(), args.data)
    log_dir = args.log_dir

    if args.mode == "train":
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(base_dir, "logs", now)
        else:
            log_dir = os.path.join(base_dir, "logs", log_dir)
        print(f"Log Dir: {log_dir}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train_(args.task, args.architecture, base_dir, device, log_dir, args.folds)
    elif args.mode == "test":
        seed = np.random.randint(0, 1000)
        print("seed:", seed)
        if log_dir is not None:
            log_dir = os.path.join(base_dir, "logs", log_dir)
        else:
            raise Exception("Testing requires log dir")
        print(f"Log Dir: {log_dir}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        np.random.seed(seed)
        torch.manual_seed(seed)
        test_(args.task, args.architecture, base_dir, device, log_dir, seed)
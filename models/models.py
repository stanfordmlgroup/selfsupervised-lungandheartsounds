import os
import time
import numpy as np
import pandas as pd
import datetime
import argparse
import sys
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, MSELoss, Conv2d, LeakyReLU, MaxPool2d, Dropout, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.dummy import DummyClassifier

sys.path.append("../utils")
from features import mel
from glob import glob
from scipy.special import softmax
import lightgbm as lgbm
import joblib
import labels as la

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


class LungDataset(Dataset):
    def __init__(self, label_file, base_dir, task, split="train", transform=None, df=None):
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
        elif self.task == "disease":
            # Takes in a row, return an Int.
            # 0: Healthy, 1: COPD, 2: Other
            label = row["diagnosis"]
            if label == 'Healthy':
                return 0
            elif label == 'COPD':
                return 1
            else:
                return 2


def get_data_loader(task, label_file, base_dir, batch_size=128, split="train", df=None):
    dataset = LungDataset(label_file, base_dir, task, split=split, df=df)
    return DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


def get_scikit_loader(task, label_file, base_dir, split="train", df=None):
    dataset = LungDataset(label_file, base_dir, task, split=split, df=df)
    X = []
    y = []
    for data in dataset:
        X.append(data[0].numpy())
        y.append(data[1])
    return X, y


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


def train(epoch, arch, model, loader, optimizer, device, loss):
    model.train()
    y_true = []
    y_pred = []

    for i, data in enumerate(loader):
        X, y = data
        X, y = X.view(128, 1, 259, 128).to(device), y.to(device)

        optimizer.zero_grad()
        if arch == "CNN":
            output = model(X)
        train_loss = loss(output, y)
        y_true.extend(y.tolist())
        y_pred.extend(output.tolist())
        train_loss = train_loss.cuda()
        train_loss.backward()
        optimizer.step()

    ce = loss(torch.tensor(y_pred).to(device), torch.tensor(y_true).to(device))

    return ce, y_true, y_pred


@torch.no_grad()
def test(arch, model, loader, device, loss):
    model.eval()
    y_true = []
    y_pred = []

    for i, data in enumerate(loader):
        X, y = data
        X, y = X.view(128, 1, 259, 128).to(device), y.to(device)

        if arch == "CNN":
            output = model(X)
        y_true.extend(y.tolist())
        y_pred.extend(output.tolist())
    ce = loss(torch.tensor(y_pred).to(device), torch.tensor(y_true).to(device))

    return ce, y_true, y_pred


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)


def weights_init(m):
    if isinstance(m, Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)


def get_accuracy(labels, preds):
    return sum([np.argmax(pred) == label for label, pred in zip(labels, preds)]) / len(labels)


def train_(task, architecture, base_dir, device, log_dir, folds=5):
    log_file = os.path.join(log_dir, f"train_log.txt")

    n_splits = folds
    num_epochs = 50
    batch_size = 128
    learning_rate = 0.001
    label_file = os.path.join(base_dir, "processed", "{}_labels.csv".format(task))

    if not os.path.exists(os.path.join(log_dir, "params.txt")):
        with open(os.path.join(log_dir, "params.txt"), "w") as f:
            f.write(f"Model: {architecture}\n")
            f.write(f"Folds: {n_splits}\n")
            f.write(f"Epochs: {num_epochs}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Learning rate: {learning_rate}")

    if task == "disease":
        classes = 3
    elif task == "symptom":
        classes = 4

    dataset = LungDataset(label_file, base_dir, task, split="train")
    df = dataset.labels
    total_train_acc = 0
    total_test_acc = 0
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=345)

    weights = torch.as_tensor(la.class_distribution(task, label_file)).float().to(device)
    weights = 1.0 / weights
    weights = weights / weights.sum()
    loss = CrossEntropyLoss(weight=weights).to(device)
    if architecture == "lgbm":
        X, y = get_scikit_loader(task, label_file, base_dir, split="train", df=df)
        X = np.asarray(X)
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        y = np.asarray(y)
        params = {"random_state": 345, "objective": "multiclass", "num_classes": classes, "verbose": -1,"gpu_platform_id":0,"gpu_device_id":0}
        results = lgbm.cv(params, lgbm.Dataset(np.asarray(X), label=np.asarray(y), categorical_feature="auto"),
                          folds=kf, metrics=["multiclass"])
        model = lgbm.LGBMClassifier(**params, class_weight="balanced", n_estimators=len(results['multi_logloss-mean']))
        for fold, (train_idx, test_idx) in enumerate(kf.split(df, df["y"])):
            start_fold = time.time()
            print("Fold: {:03d}".format(fold))
            with open(log_file, "a+") as log:
                log.write("Fold: {:03d}".format(fold))
            model.fit(X[train_idx], y[train_idx])
            train_pred = model.predict_proba(X[train_idx])[:, 1]
            test_pred = model.predict_proba(X[test_idx])[:, 1]
            joblib.dump(model, os.path.join(log_dir, "lgbm_fold_" + str(fold) + ".pkl"))
            fold_train_acc = get_accuracy(y[train_idx], train_pred)
            fold_test_acc = get_accuracy(y[test_idx], test_pred)
            total_train_acc += fold_train_acc
            total_test_acc += fold_test_acc
            elapsed_fold = time.time() - start_fold

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
            "Total Cross Val Train Acc: {:.7f}\tTotal Cross Val Test Acc: {:.7f}\n".format(total_train_acc,
                                                                                           total_test_acc))
        with open(log_file, "a+") as log:
            log.write(
                "Total Cross Val Train Acc: {:.7f}\tTotal Cross Val Test Acc: {:.7f}\n".format(
                    total_train_acc, total_test_acc
                )
            )

        return total_test_acc
    elif architecture == "CNN":
        model = CNN(classes=classes).to(device)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for fold, (train_idx, test_idx) in enumerate(kf.split(df, df["y"])):
            best_test_loss = 999

            model.apply(weights_init)
            start_fold = time.time()
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            train_loader = get_data_loader(task, label_file, base_dir, batch_size=batch_size, df=train_df)
            test_loader = get_data_loader(task, label_file, base_dir, batch_size=batch_size, df=test_df)

            fold_train_acc = 0
            fold_test_acc = 0
            for epoch in range(1, num_epochs + 1):
                start = time.time()
                train_loss, train_true, train_pred = train(epoch, architecture, model, train_loader, optimizer, device,
                                                           loss)
                test_loss, test_true, test_pred = test(architecture, model, test_loader, device, loss)
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
            "Total Cross Val Train Acc: {:.7f}\tTotal Cross Val Test Acc: {:.7f}\n".format(total_train_acc,
                                                                                           total_test_acc))
        with open(log_file, "a+") as log:
            log.write(
                "Total Cross Val Train Acc: {:.7f}\tTotal Cross Val Test Acc: {:.7f}\n".format(
                    total_train_acc, total_test_acc
                )
            )

        return best_test_loss

def auc_per_cat(true,pred,labels):
    output={}
    for i, label in enumerate(labels):
        y=(true==i).astype(int)
        output[label]=roc_auc_score(y,pred[:,i])
    return output

def test_(task, architecture, base_dir, device, log_dir, seed=None):
    batch_size = 128
    label_file = os.path.join(base_dir, "processed", task + "_labels.csv")

    if task == "disease":
        classes = 3
        labels = ["Healthy", "COPD", "Other"]
    elif task == "symptom":
        classes = 4
        labels = ["None", "Wheezes", "Crackles", "Both"]

    test_file = os.path.join(log_dir, f"test_results.txt")
    _y_pred = []
    weights = torch.as_tensor(la.class_distribution(task, label_file)).double().to(device)
    weights = 1.0 / weights
    weights = weights / weights.sum()
    loss = CrossEntropyLoss(weight=weights).to(device)
    with open(test_file, "w+") as out:
        if architecture == "CNN":
            whole_test_loader = get_data_loader(task, label_file, base_dir, batch_size=batch_size, split="test")
            model = CNN(classes=classes).to(device)
            model.to(device)
            for i, model_weights in enumerate(glob(os.path.join(os.path.join(log_dir, "best_weights_*.pt")))):
                model.load_state_dict(torch.load(model_weights))
                ce, y, y_pred = test(architecture, model, whole_test_loader, device, loss)
                _y_pred.append(y_pred)
                print("Model {} Test CE: {:.7f}".format(i, ce))
                out.write("Model {} Test CE: {:.7f}\n".format(i, ce))
        elif architecture == "lgbm":
            for i, model_weights in enumerate(glob(os.path.join(os.path.join(log_dir, "lgbm_fold_*.pkl")))):
                X, y = get_scikit_loader(task, label_file, base_dir, split="test")
                X = np.asarray(X)
                X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
                y = np.asarray(y)
                model = joblib.load(model_weights)
                y_pred = model.predict_proba(X)
                _y_pred.append(y_pred)
                loss=loss.cuda()
                ce = loss(torch.as_tensor(y_pred).to(device), torch.as_tensor(y).long().to(device))
                print("Model {} Test CE: {:.7f}".format(i, ce))
                out.write("Model {} Test CE: {:.7f}\n".format(i, ce))

        _y_pred = np.average(_y_pred, axis=0)
        auc_cat=auc_per_cat(y,softmax(_y_pred, axis=1),labels)
        out.write("\nPer category AUC:")
        print("\nPer category AUC:")
        for key in auc_cat.keys():
            out.write("{}: {:.3f}\n".format(key,auc_cat[key]))
            print("{}: {:.3f}\n".format(key,auc_cat[key]))
        roc_score = roc_auc_score(y, softmax(_y_pred, axis=1), multi_class="ovr")
        report = classification_report(y, np.argmax(_y_pred, axis=1), target_names=labels)
        print("Ensemble {}:\nEnsemble AUC-ROC: {:.7f}\n{}\n".format(architecture, roc_score, report))
        out.write(
            "Seed: {}\tFolds: {}\nEnsemble {}:\nEnsemble AUC-ROC: {:.7f}\n{}\n".format(seed, i + 1, architecture,
                                                                                       roc_score, report))

        scikit_X, scikit_y = get_scikit_loader(task, label_file, base_dir, split="test")
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(scikit_X, scikit_y)
        baseline_pred = baseline.predict(scikit_X)
        report = classification_report(scikit_y, baseline_pred, target_names=labels, zero_division=0)
        baseline_pred_as_class = []
        for pred in baseline_pred:
            pred_as_class = np.zeros((classes))
            pred_as_class[pred] = 1.0
            baseline_pred_as_class.append(pred_as_class)
        roc_score = roc_auc_score(scikit_y, baseline_pred_as_class, multi_class="ovr")
        print("Baseline:\nAUC-ROC: {:.7f}\n{}\n".format(roc_score, report))
        out.write("Baseline:\nAUC-ROC: {:.7f}\n{}\n".format(roc_score, report))

    return ce


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices={"train", "test"})
    parser.add_argument("--architecture", type=str, default="CNN", choices={"CNN", "lgbm"})
    parser.add_argument("--task", type=str, default=None, choices={"symptom", "disease"})
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

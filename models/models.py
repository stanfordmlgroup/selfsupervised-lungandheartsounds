import os
import time
import numpy as np
import datetime
import argparse
from torch.autograd import Variable
from torch.optim import Adam, LBFGS
import torch
from torch.nn import Sequential, Linear, ReLU, DataParallel, Conv2d, LeakyReLU, MaxPool2d, Dropout, CrossEntropyLoss, \
    Softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from glob import glob
from scipy.special import softmax
import lightgbm as lgbm
import joblib
import torchvision.models as models
import torch.nn.functional as F
from data import get_data_loader, get_dataset, get_scikit_loader
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../utils")
import loss as lo
import labels as la

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


class Logistic(torch.nn.Module):
    def __init__(self, input_dim):
        super(Logistic, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.linear(x)
        x = x.view(-1)
        x = torch.clamp(x,min=-10,max=10)
        return x


class SSL(torch.nn.Module):
    def __init__(self, encoder):
        super(SSL, self).__init__()
        self.encoder = encoder
        self.feature_length = encoder.num_ftrs
        self.linear_layers = Sequential(
            torch.nn.Linear(self.feature_length, self.feature_length, bias=False),
            torch.nn.BatchNorm1d(self.feature_length),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.feature_length, 1, bias=True)
        )

    def forward(self, x):
        x = self.encoder(x, tune=True)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.linear_layers(x)
        x = x.view(-1)
        return x


class ResNetSimCLR(torch.nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.num_ftrs = resnet.fc.in_features

        self.features = Sequential(*list(resnet.children())[:-1])
        self.out_dim = out_dim
        # projection MLP
        self.l1 = Linear(self.num_ftrs, self.num_ftrs)
        self.l2 = Linear(self.num_ftrs, self.out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x, tune=False):
        h = self.features(x)
        h = h.squeeze()
        if not tune:
            x = self.l1(h)
            x = F.relu(x)
            x = self.l2(x)
            return x
        else:
            return h

class CNN(torch.nn.Module):
    def __init__(self, task, classes):
        self.task = task
        super(CNN, self).__init__()
        self.cnn_layers = Sequential(
            Conv2d(1, 128, kernel_size=[7, 11], stride=2, padding=1),
            LeakyReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(128, 256, kernel_size=5, padding=1),
            LeakyReLU(inplace=True),
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
            MaxPool2d(4),
        )
        hidden_dim = 0
        if self.task == "disease" or self.task == "symptom" or self.task=='crackle' or self.task=='wheeze':
            hidden_dim = 10240
        elif self.task == "heart":
<<<<<<< HEAD
            hidden_dim = 140250 #Change for distill
=======
            hidden_dim = 90112 #Change for distill
>>>>>>> 3760e9d760a80cb5903e6b8213021a3cd4db2668
        # elif self.task == "heart_distill":
        #     hidden_dim = 2805
        elif self.task == "heartchallenge":
            hidden_dim = 2048

        self.linear_layers = Sequential(
            # 118272 for disease/symptom, 157696 for heart, 219648 for heart challenge
            Linear(hidden_dim, 256), ReLU(inplace=True), Linear(256, 128), ReLU(inplace=True),
            Linear(128, classes)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.linear_layers(x)
        return x


class CNNlight(torch.nn.Module):
    def __init__(self, task, classes):
        self.task = task
        super(CNNlight, self).__init__()
        self.cnn_layers = Sequential(
            Conv2d(1, 64, kernel_size=[7, 11], stride=2, padding=1),
            LeakyReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(64, 128, kernel_size=3, padding=1),
            LeakyReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(128, 128, kernel_size=1, padding=1),
            LeakyReLU(inplace=True),
            MaxPool2d(2),

        )
        hidden_dim = 0
        if self.task == "disease" or self.task == "symptom":
            hidden_dim = 17408
        elif self.task == "heart":
            hidden_dim = 4096
        elif self.task == "heartchallenge":
            hidden_dim = 1024
        self.linear_layers = Sequential(
            # 11904 for disease/symptom, 119040 for heart, 35712 for heart challenge
            Linear(hidden_dim, 4096), ReLU(inplace=True), Dropout(0.5), Linear(4096, 512), ReLU(inplace=True),
            Linear(512, classes)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear_layers(x)
        return x


def train(epoch, arch, model, loader, optimizer, device, loss):
    if arch == "CNN" or arch == "CNNlight":
        model.train()
    y_true = []
    y_pred = []

    for i, data in enumerate(loader):
        X, y = data
        if arch == "CNN" or arch == "CNNlight":
            X, y = X.view(X.shape[0], 1, X.shape[1], X.shape[2]).to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            train_loss = loss(output, y)
            y_true.extend(y.tolist())
            y_pred.extend(output.tolist())
            train_loss = train_loss.cuda()
            train_loss.backward()
            optimizer.step()
        if arch == "lgbm":
            X, y = np.asarray(X).reshape(X.shape[0], X.shape[1] * X.shape[2]), np.asarray(y)
            if epoch == 1 and i == 0:
                model.fit(X, y)
            else:
                model.fit(X, y, init_model=model)

            output = model.predict(X, raw_score=True)
            y_true.extend(y.tolist())
            y_pred.extend(output.tolist())
    ce = loss(torch.tensor(y_pred).to(device), torch.tensor(y_true).to(device))

    return ce, y_true, y_pred


@torch.no_grad()
def test(arch, model, loader, device, loss):
    if arch == "CNN" or arch == "CNNlight":
        model.eval()
    y_true = []
    y_pred = []

    for i, data in enumerate(loader):
        X, y = data
        if arch == "CNN" or arch == "CNNlight":
            X, y = X.view(X.shape[0], 1, X.shape[1], X.shape[2]).to(device), y.to(device)
            output = model(X)
            y_true.extend(y.tolist())
            y_pred.extend(output.tolist())
        elif arch == "lgbm":
            X, y = np.asarray(X).reshape(X.shape[0], X.shape[1] * X.shape[2]), np.asarray(y)
            output = model.predict_proba(X, raw_score=True)
            y_true.extend(y.tolist())
            y_pred.extend(output.tolist())
    ce = loss(torch.tensor(y_pred).to(device), torch.tensor(y_true).to(device))

    return ce, y_true, y_pred


def train_(task, architecture, base_dir, device, log_dir, folds=5, train_prop=1):
    log_file = os.path.join(log_dir, f"train_log.txt")

    n_splits = folds
    num_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    label_file = os.path.join(base_dir, "processed", "{}_labels.csv".format(task))

    with open(os.path.join(log_dir, "params.txt"), "w") as f:
        f.write(f"Model: {architecture}\n")
        f.write(f"Folds: {n_splits}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}")
        f.write(f"Proportion of training data: {train_prop}")

    if task == "disease":
        classes = 3
    elif task == "symptom":
        classes = 4
    elif task == "heart":
        classes = 2
    elif task == "heartchallenge":
        classes = 5

    dataset = get_dataset(task, label_file, base_dir, split="train", train_prop=train_prop)
    df = dataset.labels
    total_train_acc = 0
    total_test_acc = 0
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=345)

    weights = torch.as_tensor(la.class_distribution(task, label_file)).float().to(device)
    weights = 1.0 / weights
    weights = weights / weights.sum()
    loss = CrossEntropyLoss(weight=weights).to(device)

    for fold, (train_idx, test_idx) in enumerate(kf.split(df, df["y"])):
        if architecture == "CNN":
            model = DataParallel(CNN(task=task, classes=classes))
        elif architecture == "CNNlight":
            model = CNNlight(task=task, classes=classes).to(device)
        elif architecture == "lgbm":
            params = {"objective": "multiclass", "num_classes": classes,  # "verbose": -1,
                      "force_col_wise": True}
            model = lgbm.LGBMClassifier(**params, class_weight="balanced")
        if architecture == "CNN" or architecture == "CNNlight":
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = None
        #    model.apply(weights_init)

        start_fold = time.time()
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        train_loader = get_data_loader(task, label_file, base_dir, batch_size=batch_size, df=train_df)
        test_loader = get_data_loader(task, label_file, base_dir, batch_size=batch_size, df=test_df)

        fold_train_acc = 0
        fold_test_acc = 0
        for epoch in range(1, num_epochs + 1):
            best_test_loss = 999999
            start = time.time()
            train_loss, train_true, train_pred = train(epoch, architecture, model, train_loader, optimizer, device,
                                                       loss)
            test_loss, test_true, test_pred = test(architecture, model, test_loader, device, loss)
            train_accuracy = lo.get_accuracy(train_true, train_pred)
            test_accuracy = lo.get_accuracy(test_true, test_pred)

            if test_loss < best_test_loss:
                if architecture == "CNN" or architecture == "CNNlight":
                    lo.save_weights(model, os.path.join(log_dir, "best_weights_" + str(fold) + ".pt"))
                elif architecture == "lgbm":
                    joblib.dump(model, os.path.join(log_dir, "lgbm_fold_" + str(fold) + ".pkl"))
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


def test_(task, architecture, base_dir, device, log_dir, seed=None):
    label_file = os.path.join(base_dir, "processed", task + "_labels.csv")
    batch_size = 1306
    if task == "disease":
        classes = 3
        labels = ["Healthy", "COPD", "Other"]
    elif task == "symptom":
        classes = 4
        labels = ["None", "Wheezes", "Crackles", "Both"]
    elif task == "heart":
        classes = 2
        labels = ["Abnormal", "Normal"]
        batch_size = 700
    elif task == "heartchallenge":
        classes = 5
        labels = ["Normal", "Murmur", "Extrastole", "Extrasound", "Artifact"]
        batch_size = 192
    test_file = os.path.join(log_dir, f"test_results.txt")
    _y_pred = []
    weights = torch.as_tensor(la.class_distribution(task, label_file)).float().to(device)
    weights = 1.0 / weights
    weights = weights / weights.sum()
    loss = CrossEntropyLoss(weight=weights).to(device)
    with open(test_file, "w+") as out:
        if architecture == "CNN" or architecture == "CNNlight":
            whole_test_loader = get_data_loader(task, label_file, base_dir, batch_size=batch_size, split="test")
            if architecture == "CNN":
                model = DataParallel(CNN(task=task, classes=classes).to(device))
            elif architecture == "CNNlight":
                model = CNNlight(task=task, classes=classes).to(device)
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
                loss = loss.cuda()
                print(torch.as_tensor(y_pred).shape)
                ce = loss(torch.as_tensor(y_pred).float().to(device), torch.as_tensor(y).long().to(device))
                print("Model {} Test CE: {:.7f}".format(i, ce))
                out.write("Model {} Test CE: {:.7f}\n".format(i, ce))

        _y_pred = np.average(_y_pred, axis=0)
        auc_cat = lo.auc_per_cat(y, softmax(_y_pred, axis=1), labels)
        out.write("\nPer category AUC:")
        print("\nPer category AUC:")
        for key in auc_cat.keys():
            out.write("{}: {:.3f}\n".format(key, auc_cat[key]))
            print("{}: {:.3f}\n".format(key, auc_cat[key]))

        if task == "heart":
            roc_score = roc_auc_score(y, softmax(_y_pred, axis=1)[:, 0])
        else:
            roc_score = roc_auc_score(y, softmax(_y_pred, axis=1), multi_class="ovr")
        report = classification_report(y, np.argmax(_y_pred, axis=1), target_names=labels)
        conf_matrix = confusion_matrix(y, np.argmax(_y_pred, axis=1))
        print("Ensemble {}:\nEnsemble AUC-ROC: {:.7f}\n{}\nConfusion Matrix:\n{}\n".format(architecture, roc_score,
                                                                                           report, conf_matrix))
        out.write(
            "Seed: {}\tFolds: {}\nEnsemble {}:\nEnsemble AUC-ROC: {:.7f}\n{}\nConfusion Matrix:\n{}\n".format(seed,
                                                                                                              i + 1,
                                                                                                              architecture,
                                                                                                              roc_score,
                                                                                                              report,
                                                                                                              conf_matrix))

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
    parser.add_argument("--architecture", type=str, default="CNN", choices={"CNN", "lgbm", "CNNlight"})
    parser.add_argument("--task", type=str, default=None, choices={"symptom", "disease", "heartchallenge", "heart"})
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--data", type=str, default="../data")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train_prop", type=float, default=1.0)
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
        train_(args.task, args.architecture, base_dir, device, log_dir, args.folds, args.train_prop)
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

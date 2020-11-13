from models import ResNetSimCLR, SLDropout
import os
import time
import numpy as np
import datetime
import argparse
import torch
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from glob import glob
from scipy.special import softmax
import torch.nn.functional as F
from data import get_data_loader, get_dataset, get_scikit_loader, DataTransform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../utils")
import loss as lo
import labels as la
from loss import NTXentLoss
import random


class ContrastiveLearner(object):

    def __init__(self, dataset, epochs, batch_size, log_dir, model=None):
        self.device = self._get_device()
        self.dataset = dataset
        self.epochs = epochs
        self.log_dir = log_dir
        self.nt_xent_criterion = NTXentLoss(self.device, batch_size, .5, use_cosine_similarity=True)
        self.model = model
        self.batch_size = batch_size

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def get_model(self, out_dim):
        model = ResNetSimCLR(out_dim=out_dim, base_model="resnet18").to(self.device)
        model = self._load_pre_trained_weights(model)
        return model

    def pre_train(self, log_file, task, label_file):
        df = self.dataset.labels.reset_index()
        train_list = random.sample(range(0, len(df.index)), int(.8 * len(df.index)))
        train_df = df[df.index.isin(train_list)]
        test_df = df[~df.index.isin(train_list)]
        train_loader = get_data_loader(task, label_file, base_dir, batch_size=self.batch_size, split="pre-train",
                                       df=train_df)
        valid_loader = get_data_loader(task, label_file, base_dir, batch_size=self.batch_size, split="pre-train",
                                       df=test_df)

        if self.model is not None:
            model = self.model
        else:
            model = self.get_model(out_dim=256)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=10e-6)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        best_valid_loss = np.inf

        for epoch_counter in range(1, self.epochs+1):
            start = time.time()
            epoch_loss = 0
            num_batches = len(train_loader)
            for xis, xjs in train_loader:
                optimizer.zero_grad()
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            epoch_loss /= float(num_batches)
            # validate the model
            valid_loss = self._validate(model, valid_loader)
            if valid_loss < best_valid_loss:
                # save the model weights
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(self.log_dir, 'encoder.pth'))
            elapsed = time.time() - start
            print("\tEpoch: {:03d}, Time: {:.3f} s".format(epoch_counter, elapsed))
            print("\t\tTrain loss: {:.7f}\tVal loss: {:.7f}".format(epoch_loss, valid_loss))
            with open(log_file, "a+") as log:
                log.write(
                    "\tEpoch: {:03d}\tTrain Loss: {:.7f}\tVal Loss: {:.7f}\n".format(
                        epoch_counter, epoch_loss, valid_loss
                    )
                )

            if epoch_counter % 5 == 0:
                encoder = model.eval()
                train_X, train_y = get_scikit_loader(self.device, task, label_file, base_dir, "pre-train", df=train_df,
                                                     encoder=encoder)
                test_X, test_y = get_scikit_loader(self.device, task, label_file, base_dir, "pre-train", df=test_df, encoder=encoder)
                train_X = np.asarray(train_X)
                train_y = np.asarray(train_y)
                test_X = np.asarray(test_X)
                test_y = np.asarray(test_y)
                evaluator = KNeighborsClassifier(n_neighbors=10)
                evaluator.fit(train_X, train_y)
                fold_train_acc = evaluator.score(test_X, test_y)
                print(
                    "Pre-train KNN Evaluation Score: {:.7f}\n".format(fold_train_acc))
                with open(log_file, "a+") as log:
                    log.write("Pre-train KNN Evaluation Score: {:.7f}\n".format(fold_train_acc))

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            cosine_lr_decay = scheduler.get_lr()[0]
            print("\t\tcosine lr decay: {:.7f}".format(cosine_lr_decay))
            with open(log_file, "a+") as log:
                log.write("\tcosine lr decay: {:.7f}\n".format(cosine_lr_decay))

        return model

    def fine_tune(self, n_splits, task, label_file, log_file, encoder=None, evaluator_type=None, learning_rate=0.0):
        df = self.dataset.labels
        total_train_acc = 0
        total_test_acc = 0
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=345)

        weights = torch.as_tensor(la.class_distribution(task, label_file)).float().to(self.device)
        weights = 1.0 / weights
        weights = weights / weights.sum()
        loss = CrossEntropyLoss(weight=weights).to(self.device)

        if encoder is not None:
            total_train_acc = 0
            for fold, (train_idx, test_idx) in enumerate(kf.split(df, df["y"])):
                start_fold = time.time()
                train_df = df.iloc[train_idx]
                test_df = df.iloc[~train_idx]

                if evaluator_type == "linear" or evaluator_type == "knn":
                    encoder.eval()
                    train_X, train_y = get_scikit_loader(self.device, task, label_file, base_dir, "train", train_df, encoder)
                    test_X, test_y = get_scikit_loader(self.device, task, label_file, base_dir, "test", test_df, encoder)
                    train_X = np.asarray(train_X)
                    train_y = np.asarray(train_y)
                    test_X = np.asarray(test_X)
                    test_y = np.asarray(test_y)
                    if evaluator_type == "linear":
                        evaluator = LogisticRegression(class_weight="balanced", max_iter=10000)
                    elif evaluator_type == "knn":
                        evaluator = KNeighborsClassifier(n_neighbors=10)
                    evaluator.fit(train_X, train_y)
                    fold_train_acc = evaluator.socer(train_X, train_y)
                    fold_test_acc = evaluator.score(test_X, test_y)
                    joblib.dump(evaluator, os.path.join(log_dir, "evaluator_" + str(fold) + ".pkl"))
                elif evaluator_type == "fine-tune":
                    model = SLDropout(encoder).to(self.device)
                    train_df = df.iloc[train_idx]
                    test_df = df.iloc[test_idx]
                    train_loader = get_data_loader(task, label_file, base_dir, self.batch_size, "train", train_df)
                    test_loader = get_data_loader(task, label_file, base_dir, 1, "test", test_df)
                    # learning_rate = learning_rate * 10.0
                    # print("LR: {:.7f}".format(learning_rate))
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    fold_train_acc = 0
                    fold_test_acc = 0
                    for epoch in range(1, self.epochs + 1):
                        best_test_loss = 999999
                        start = time.time()
                        train_loss, train_true, train_pred = self._train(model, train_loader, optimizer, self.device,
                                                                         loss)
                        test_loss, test_true, test_pred = self._test(model, test_loader, self.device, loss)
                        train_accuracy = lo.get_accuracy(train_true, train_pred)
                        test_accuracy = lo.get_accuracy(test_true, test_pred)

                        if test_loss < best_test_loss:
                            lo.save_weights(model, os.path.join(log_dir, "evaluator_" + str(fold) + ".pt"))

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

                    fold_train_acc /= float(self.epochs)
                    fold_test_acc /= float(self.epochs)

                elapsed_fold = time.time() - start_fold
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

        else:
            print("No encoder provided. Supervised Training...\n")
            for fold, (train_idx, test_idx) in enumerate(kf.split(df, df["y"])):
                start_fold = time.time()
                model = self.get_model(2)
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                train_loader = get_data_loader(task, label_file, base_dir, self.batch_size, "train", train_df)
                test_loader = get_data_loader(task, label_file, base_dir, 1, "test", test_df)
                # learning_rate = learning_rate * 10.0
                # print("LR: {:.7f}".format(learning_rate))
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                fold_train_acc = 0
                fold_test_acc = 0
                for epoch in range(1, self.epochs + 1):
                    best_test_loss = 999999
                    start = time.time()
                    train_loss, train_true, train_pred = self._train(model, train_loader, optimizer, self.device,
                                                                     loss)
                    test_loss, test_true, test_pred = self._test(model, test_loader, self.device, loss)
                    train_accuracy = lo.get_accuracy(train_true, train_pred)
                    test_accuracy = lo.get_accuracy(test_true, test_pred)

                    if test_loss < best_test_loss:
                        lo.save_weights(model, os.path.join(log_dir, "evaluator_" + str(fold) + ".pt"))

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

                fold_train_acc /= float(self.epochs)
                fold_test_acc /= float(self.epochs)
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

    def test(self, task, label_file, log_file, encoder, evaluator_dir):
        _y_pred = []
        weights = torch.as_tensor(la.class_distribution(task, label_file)).float().to(self.device)
        weights = 1.0 / weights
        weights = weights / weights.sum()
        loss = CrossEntropyLoss(weight=weights).to(self.device)
        labels = ["Abnormal", "Normal"]

        scikit_eval = len(glob(os.path.join(evaluator_dir, "evaluator_*.pkl"))) > 0
        with open(log_file, "w+") as out:
            if scikit_eval:
                encoder.eval()
                X, y = get_scikit_loader(self.device, task, label_file, base_dir, split="test", encoder=encoder)
                X = np.asarray(X)
                y = np.asarray(y)
                y_true = y
                for i, model_weights in enumerate(glob(os.path.join(evaluator_dir, "evaluator_*.pkl"))):
                    model = joblib.load(model_weights)
                    y_pred = model.predict_proba(X)
                    _y_pred.append(y_pred)
                    loss = loss.cuda()
                    ce = loss(torch.as_tensor(y_pred).float().to(self.device),
                              torch.as_tensor(y).long().to(self.device))
                    print("Model {} Test CE: {:.7f}".format(i, ce))
                    out.write("Model {} Test CE: {:.7f}\n".format(i, ce))
            else:
                for i, model_weights in enumerate(glob(os.path.join(evaluator_dir, "evaluator_*.pt"))):
                    loader = get_data_loader(task, label_file, base_dir, batch_size=self.batch_size, split="test")
                    if encoder is None:
                        model = self.get_model(2)
                        state_dict = torch.load(model_weights)
                        model.load_state_dict(state_dict)
                    else:
                        model = SLDropout(encoder).to(self.device)
                    model.eval()
                    ce, y_true, y_pred = self._test(model, loader, self.device, loss)
                    _y_pred.append(y_pred)
                    print("Model {} Test CE: {:.7f}".format(i, ce))
                    out.write("Model {} Test CE: {:.7f}\n".format(i, ce))
            _y_pred = np.average(_y_pred, axis=0)
            auc_cat = lo.auc_per_cat(y_true, softmax(_y_pred, axis=1), labels)
            out.write("\nPer category AUC:")
            print("\nPer category AUC:")
            for key in auc_cat.keys():
                out.write("{}: {:.3f}\n".format(key, auc_cat[key]))
                print("{}: {:.3f}\n".format(key, auc_cat[key]))
            roc_score = 1 - roc_auc_score(y_true, softmax(_y_pred, axis=1)[:, 0])

            report = classification_report(y_true, np.argmax(_y_pred, axis=1), target_names=labels)
            conf_matrix = confusion_matrix(y_true, np.argmax(_y_pred, axis=1))
            print("Ensemble AUC-ROC: {:.7f}\n{}\nConfusion Matrix:\n{}\n".format(roc_score, report, conf_matrix))
            out.write(
                "Seed: {}\tFolds: {}\nEnsemble AUC-ROC: {:.7f}\n{}\nConfusion Matrix:\n{}\n".format(seed, i + 1,
                                                                                                    roc_score, report,
                                                                                                    conf_matrix))

            scikit_X, scikit_y = get_scikit_loader(self.device, task, label_file, base_dir, split="test")
            baseline = DummyClassifier(strategy="most_frequent")
            baseline.fit(scikit_X, scikit_y)
            baseline_pred = baseline.predict(scikit_X)
            report = classification_report(scikit_y, baseline_pred, target_names=labels, zero_division=0)
            roc_score = roc_auc_score(scikit_y, baseline_pred)
            print("Baseline:\nAUC-ROC: {:.7f}\n{}\n".format(roc_score, report))
            out.write("Baseline:\nAUC-ROC: {:.7f}\n{}\n".format(roc_score, report))
        return loss

    def _load_pre_trained_weights(self, model):
        try:
            state_dict = torch.load(os.path.join(self.log_dir, 'encoder.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found.")

        return model

    def _train(self, model, loader, optimizer, device, loss):
        model.train()
        y_true = []
        y_pred = []

        for i, data in enumerate(loader):
            X, y = data
            X, y = X.view(X.shape[0], 1, X.shape[1], X.shape[2]).to(device), y.to(device)
            optimizer.zero_grad()
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
    def _test(arch, model, loader, device, loss):

        model.eval()
        y_true = []
        y_pred = []
        for i, data in enumerate(loader):
            X, y = data
            X, y = X.view(X.shape[0], 1, X.shape[1], X.shape[2]).to(device), y.to(device)
            output = model(X)
            y_true.extend(y.tolist())
            if X.shape[0] == 1:
                y_pred.append(output.tolist())
            else:
                y_pred.extend(output.tolist())
        ce = loss(torch.tensor(y_pred).to(device), torch.tensor(y_true).to(device))
        print ()
        return ce, y_true, y_pred

    def _step(self, model, xis, xjs):

        # get the representations and the projections
        zis = model(xis)  # [N,C]

        # get the representations and the projections
        zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for xis, xjs in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss


def pretrain_(task, base_dir, log_dir, train_prop=1):
    log_file = os.path.join(log_dir, f"pretraintrain_log.txt")

    num_epochs = 25
    batch_size = 16
    learning_rate = 0.0001

    with open(os.path.join(log_dir, "pretrain_params.txt"), "w") as f:
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Proportion of training data: {train_prop}\n")

    label_file = os.path.join(base_dir, "processed", "{}_labels.csv".format(task))
    dataset = get_dataset(task, label_file, base_dir, split="pre-train", train_prop=train_prop)

    learner = ContrastiveLearner(dataset, num_epochs, batch_size, log_dir)
    learner.pre_train(log_file, task, label_file)


def train_(task, base_dir, log_dir, evaluator, folds=5, train_prop=1):
    log_file = os.path.join(log_dir, f"train_log.txt")

    num_epochs = 5
    batch_size = 16
    learning_rate = 0.0001

    with open(os.path.join(log_dir, "train_params.txt"), "w") as f:
        f.write(f"Folds: {folds}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Proportion of training data: {train_prop}\n")
        f.write(f"Evaluator: {evaluator}\n")

    label_file = os.path.join(base_dir, "processed", "{}_labels.csv".format(task))
    dataset = get_dataset(task, label_file, base_dir, split="train", train_prop=train_prop)

    learner = ContrastiveLearner(dataset, num_epochs, batch_size, log_dir)
    try:
        state_dict = torch.load(os.path.join(log_dir, 'encoder.pth'))
        encoder = learner.get_model(256)
        encoder.load_state_dict(state_dict)
    except FileNotFoundError:
        encoder = None
    learner.fine_tune(folds, task, label_file, log_file, encoder, evaluator, learning_rate)


def test_(task, base_dir, log_dir, seed=None):
    log_file = os.path.join(log_dir, f"test_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Folds: {seed}\n")
    label_file = os.path.join(base_dir, "processed", "{}_labels.csv".format(task))
    dataset = get_dataset(task, label_file, base_dir, split="test")
    learner = ContrastiveLearner(dataset, 0, 1, log_dir)
    try:
        state_dict = torch.load(os.path.join(log_dir, 'encoder.pth'))
        encoder = learner.get_model(256)
        encoder.load_state_dict(state_dict)
    except FileNotFoundError:
        encoder = None
    learner.test(task, label_file, log_file, encoder, log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices={"pre-train", "train", "test"})
    parser.add_argument("--task", type=str, default=None, choices={"lung", "heartchallenge", "heart"})
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--data", type=str, default="../data")
    parser.add_argument("--evaluator", type=str, default=None, choices={"knn", "linear", "fine-tune"})
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train_prop", type=float, default=1.0)
    args = parser.parse_args()

    base_dir = os.path.join(os.getcwd(), args.data)
    log_dir = args.log_dir

    if args.mode == "pre-train":
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(base_dir, "logs", now)
        else:
            log_dir = os.path.join(base_dir, "logs", log_dir)
        print(f"Log Dir: {log_dir}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        pretrain_(args.task, base_dir, log_dir, args.train_prop)
    elif args.mode == "train":
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(base_dir, "logs", now)
        else:
            log_dir = os.path.join(base_dir, "logs", log_dir)
        print(f"Log Dir: {log_dir}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train_(args.task, base_dir, log_dir, args.evaluator, args.folds, args.train_prop)
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
        test_(args.task, base_dir, log_dir, seed)

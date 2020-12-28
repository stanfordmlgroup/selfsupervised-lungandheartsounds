from models import ResNetSimCLR, SSL
import os
import time
import numpy as np
import datetime
import argparse
import torch
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.dummy import DummyClassifier
from glob import glob
from scipy.special import expit
import torch.nn.functional as F
from data import get_data_loader, get_dataset, get_scikit_loader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import sys
import copy

from sklearn import preprocessing
from matplotlib import pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../utils")
import loss as lo
import labels as la
import file as fi
from loss import NTXentLoss, WeightedFocalLoss
import random


class ContrastiveLearner(object):

    def __init__(self, dataset, epochs, batch_size, log_dir, model=None):
        self.device = self._get_device()
        self.dataset = dataset
        self.epochs = epochs
        self.log_dir = log_dir
        self.nt_xent_criterion = NTXentLoss()
        self.model = model
        self.batch_size = batch_size
        try:
            self.exp = dataset.exp
        except:
            self.exp = None

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def get_model(self, out_dim):
        model = ResNetSimCLR(out_dim=out_dim, base_model="resnet18").to(self.device)
        model = self._load_pre_trained_weights(model)
        return model

    def pre_train(self, log_file, task, label_file, augment=None, learning_rate=0.):
        df = self.dataset.labels.reset_index()
        data = self.dataset.data
        # scaler = preprocessing.StandardScaler()
        # scaler.fit(data.reshape((data.shape[0], -1)))
        # scaler.transform(data)
        train_list = random.sample(range(0, len(df.index)), int(.8 * len(df.index)))
        select = np.in1d(range(data.shape[0]), train_list)
        train_df = df[df.index.isin(train_list)]
        train_data = data[select]
        test_data = data[~select]
        test_df = df[~df.index.isin(train_list)]
        if self.exp in [2, 4, 5, 6]:
            self.batch_size = 1
        train_loader = get_data_loader(task, label_file, base_dir, batch_size=self.batch_size, split="pretrain",
                                       df=train_df, transform=augment, data=train_data, exp=self.exp)
        valid_loader = get_data_loader(task, label_file, base_dir, batch_size=self.batch_size, split="pretrain",
                                       df=test_df, transform=augment, data=test_data, exp=self.exp)

        if self.model is not None:
            model = self.model
        else:
            model = self.get_model(out_dim=256)

        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=10e-6)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        best_valid_loss = np.inf
        best_auc = np.inf
        fi.make_path(os.path.join(log_dir, 'checkpoints'))
        for epoch_counter in range(1, self.epochs + 1):
            start = time.time()
            epoch_loss = 0
            num_batches = len(train_loader)
            if self.exp in [2, 4, 5, 6]:
                for data in train_loader:
                    xis, xjs = [], []
                    for sample in data:
                        xis.append(sample[0])
                        xjs.append(sample[1])
                    xis = torch.stack(tuple(xis))
                    xis = xis.view(xis.shape[0], xis.shape[2], xis.shape[3])
                    xjs = torch.stack(tuple(xjs))
                    xjs = xjs.view(xjs.shape[0], xjs.shape[2], xjs.shape[3])
                    xis = xis.to(self.device)
                    xjs = xjs.to(self.device)
                    loss = self._step(model, xis, xjs)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss
            elif self.exp == 3:
                for xis, xjs, y in train_loader:
                    optimizer.zero_grad()
                    xis = xis.to(self.device)
                    xjs = xjs.to(self.device)

                    loss = self._step(model, xis, xjs, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss
            else:
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
                torch.save(model.state_dict(),
                           os.path.join(self.log_dir, 'checkpoints', 'encoder_{}.pth'.format(epoch_counter)))

            elapsed = time.time() - start
            print("\tEpoch: {:03d}, Time: {:.3f} s".format(epoch_counter, elapsed))
            print("\t\tTrain loss: {:.7f}\tVal loss: {:.7f}".format(epoch_loss, valid_loss))
            with open(log_file, "a+") as log:
                log.write(
                    "\tEpoch: {:03d}\tTrain Loss: {:.7f}\tVal Loss: {:.7f}\n".format(
                        epoch_counter, epoch_loss, valid_loss
                    )
                )

            encoder = model.eval()
            train_X, train_y = get_scikit_loader(self.device, task, label_file, base_dir, "train", df=train_df,
                                                 encoder=encoder, data=train_data)
            test_X, test_y = get_scikit_loader(self.device, task, label_file, base_dir, "train", df=test_df,
                                               encoder=encoder, data=test_data)
            train_X = np.asarray(train_X)
            train_y = np.asarray(train_y)
            test_X = np.asarray(test_X)
            test_y = np.asarray(test_y)
            evaluator = KNeighborsClassifier(n_neighbors=10)
            evaluator.fit(train_X, train_y)
            fold_train_acc = evaluator.score(test_X, test_y)
            roc_score = roc_auc_score(test_y, evaluator.predict_proba(test_X)[:, 1])
            model.train()

            if roc_score < best_auc:
                # save the model weights
                best_auc = roc_score
                torch.save(model.state_dict(),
                           os.path.join(self.log_dir, 'encoder.pth'))

            print(
                "pretrain KNN Acc: {:.3f}\t KNN AUC: {:.3f}\n".format(fold_train_acc, roc_score))
            with open(log_file, "a+") as log:
                log.write("pretrain KNN Acc: {:.3f}\t KNN AUC: {:.3f}\n".format(fold_train_acc, roc_score))

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            cosine_lr_decay = scheduler.get_lr()[0]
            print("\t\tcosine lr decay: {:.7f}".format(cosine_lr_decay))
            with open(log_file, "a+") as log:
                log.write("\tcosine lr decay: {:.7f}\n".format(cosine_lr_decay))
            # if counter > 10:
            #     print("Early stop...")
            #     break

        return model

    def fine_tune(self, n_splits, task, label_file, log_file, augment=None, encoder=None, evaluator_type=None,
                  learning_rate=0.0):
        df = self.dataset.labels
        data = self.dataset.data
        # scaler = preprocessing.StandardScaler()
        # scaler.fit(data.reshape((data.shape[0], -1)))
        # scaler.transform(data)
        total_train_acc = 0
        total_test_acc = 0
        if len(df.index) > 10:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=345)
            indices = kf.split(df, df["y"])
        else:
            train_idx = random.sample(range(0, len(df.index)), int(.8 * len(df.index)))
            test_idx = []
            for i in range(0, len(df.index)):
                if i not in train_idx:
                    test_idx.append(i)
            indices = [(train_idx, test_idx)]
            self.batch_size = 4
        weights = torch.as_tensor(la.class_distribution(task, label_file)).float().to(self.device)
        # weights = 1.0 / weights
        # weights = weights / weights.sum()
        pos_weight = torch.tensor(weights[1].item() / (weights[0].item() + weights[1].item())).to(self.device)
        # loss = BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        loss = WeightedFocalLoss(alpha=pos_weight).to(self.device)
        if encoder is not None:
            total_train_acc = 0
            base_encoder = encoder
            for fold, (train_idx, test_idx) in enumerate(indices):
                start_fold = time.time()

                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                train_data = data[train_idx]
                test_data = data[test_idx]
                if evaluator_type == "linear" or evaluator_type == "knn":
                    encoder.eval()
                    train_X, train_y = get_scikit_loader(self.device, task, label_file, base_dir, "train", train_df,
                                                         encoder, data=train_data)
                    id, test_X, test_y = get_scikit_loader(self.device, task, label_file, base_dir, "test", test_df,
                                                           encoder, data=test_data)
                    train_X = np.asarray(train_X)
                    train_y = np.asarray(train_y)
                    test_X = np.asarray(test_X)
                    test_y = np.asarray(test_y)
                    if evaluator_type == "linear":
                        evaluator = LogisticRegression(class_weight="balanced", max_iter=1000000, penalty="none")
                    elif evaluator_type == "knn":
                        n_neighbors = 10
                        if n_neighbors > self.batch_size:
                            n_neighbors = self.batch_size
                        evaluator = KNeighborsClassifier(n_neighbors=n_neighbors)
                    evaluator.fit(train_X, train_y)
                    fold_train_acc = evaluator.score(train_X, train_y)
                    fold_test_acc = evaluator.score(test_X, test_y)
                    joblib.dump(evaluator, os.path.join(log_dir, "evaluator_" + str(fold) + ".pkl"))

                elif evaluator_type == "fine-tune":
                    encoder = copy.deepcopy(base_encoder)
                    model = SSL(encoder).to(self.device)
                    train_df = df.iloc[train_idx]
                    test_df = df.iloc[test_idx]
                    train_loader = get_data_loader(task, label_file, base_dir, self.batch_size, "train", df=train_df,
                                                   data=train_data)
                    test_loader = get_data_loader(task, label_file, base_dir, 1, "test", df=test_df, data=test_data)
                    # learning_rate = learning_rate * 10.0
                    # print("LR: {:.7f}".format(learning_rate))
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    fold_train_acc = 0
                    fold_test_acc = 0
                    counter = 0
                    best_test_loss = np.inf
                    epoch = 0
                    for epoch in range(1, self.epochs + 1):
                        start = time.time()
                        train_loss, train_true, train_pred = self._train(model, train_loader, optimizer, self.device,
                                                                         loss)
                        test_loss, test_true, test_pred = self._test(model, test_loader, self.device, loss)
                        train_pred, test_pred = expit(train_pred), expit(test_pred)
                        train_accuracy = lo.get_accuracy(train_true, train_pred)
                        test_accuracy = lo.get_accuracy(test_true, test_pred)

                        if test_loss < best_test_loss:
                            lo.save_weights(model, os.path.join(log_dir, "evaluator_" + str(fold) + ".pt"))
                            best_test_loss = test_loss
                            counter = 0
                        else:
                            counter += 1

                        elapsed = time.time() - start
                        print("\tEpoch: {:03d}, Time: {:.3f} s".format(epoch, elapsed))
                        print("\t\tTrain BCE: {:.7f}\tVal BCE: {:.7f}".format(train_loss, test_loss))
                        print("\t\tTrain Acc: {:.7f}\tVal Acc: {:.7f}\n".format(train_accuracy, test_accuracy))
                        with open(log_file, "a+") as log:
                            log.write(
                                "\tEpoch: {:03d}\tTrain Loss: {:.7f}\tVal Loss: {:.7f}\tTrain Acc: {:.7f}\tVal Acc: {:.7f}\n".format(
                                    epoch, train_loss, test_loss, train_accuracy, test_accuracy
                                )
                            )

                        fold_train_acc += train_accuracy
                        fold_test_acc += test_accuracy
                        if counter >= 10:
                            print("Early Stop...")
                            break

                    fold_train_acc /= float(epoch)
                    fold_test_acc /= float(epoch)

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
            for fold, (train_idx, test_idx) in enumerate(indices):
                start_fold = time.time()
                model = self.get_model(1)
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                train_data = data[train_idx]
                test_data = data[test_idx]
                train_loader = get_data_loader(task, label_file, base_dir, self.batch_size, "train", df=train_df,
                                               transform=augment, data=train_data)
                test_loader = get_data_loader(task, label_file, base_dir, 1, "test", df=test_df, data=test_data)
                # learning_rate = learning_rate * 10.0
                # print("LR: {:.7f}".format(learning_rate))
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                fold_train_acc = 0
                fold_test_acc = 0
                best_test_loss = np.inf
                counter = 0
                for epoch in range(1, self.epochs + 1):
                    start = time.time()
                    train_loss, train_true, train_pred = self._train(model, train_loader, optimizer, self.device,
                                                                     loss)
                    test_loss, test_true, test_pred = self._test(model, test_loader, self.device, loss)
                    train_pred, test_pred = expit(train_pred), expit(test_pred)
                    train_accuracy = lo.get_accuracy(train_true, train_pred)
                    test_accuracy = lo.get_accuracy(test_true, test_pred)

                    if test_loss < best_test_loss:
                        lo.save_weights(model, os.path.join(log_dir, "evaluator_" + str(fold) + ".pt"))
                        best_test_loss = test_loss
                        counter = 0
                    else:
                        counter += 1

                    elapsed = time.time() - start
                    print("\tEpoch: {:03d}, Time: {:.3f} s".format(epoch, elapsed))
                    print("\t\tTrain BCE: {:.7f}\tVal BCE: {:.7f}".format(train_loss, test_loss))
                    print("\t\tTrain Acc: {:.7f}\tVal Acc: {:.7f}\n".format(train_accuracy, test_accuracy))
                    with open(log_file, "a+") as log:
                        log.write(
                            "\tEpoch: {:03d}\tTrain Loss: {:.7f}\tVal Loss: {:.7f}\tTrain Acc: {:.7f}\tVal Acc: {:.7f}\n".format(
                                epoch, train_loss, test_loss, train_accuracy, test_accuracy
                            )
                        )

                    fold_train_acc += train_accuracy
                    fold_test_acc += test_accuracy
                    if counter > 10:
                        print("Early Stop...")
                        break

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
        # weights = 1.0 / weights
        # weights = weights / weights.sum()
        pos_weight = torch.tensor(weights[1].item() / (weights[0].item() + weights[1].item())).to(self.device)
        # loss = BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        loss = WeightedFocalLoss(alpha=pos_weight).to(self.device)

        labels = ["Normal", "Abnormal"]

        # scaler = preprocessing.StandardScaler()
        data = self.dataset.data
        # scaler.fit(data.reshape((data.shape[0], -1)))
        # scaler.transform(data)

        scikit_eval = len(glob(os.path.join(evaluator_dir, "evaluator_*.pkl"))) > 0

        with open(log_file, "a+") as out:
            if scikit_eval:
                print('SK Model Test')
                encoder.eval()
                id, X, y = get_scikit_loader(self.device, task, label_file, base_dir, split="test", encoder=encoder,
                                             data=data)
                X = np.asarray(X)
                y = np.asarray(y)
                y_true = y
                for i, model_weights in enumerate(glob(os.path.join(evaluator_dir, "evaluator_*.pkl"))):
                    model = joblib.load(model_weights)
                    y_pred = model.predict_proba(X)[:, 1]
                    _y_pred.append(y_pred)
                    # loss = loss.cuda()
                    # ce = loss(torch.as_tensor(y_pred).to(self.device).view(-1).float(),
                    #           torch.as_tensor(y).to(self.device).view(-1).float())
                    # print("Model {} Test BCE: {:.7f}".format(i, ce))
                    # out.write("Model {} Test BCE: {:.7f}\n".format(i, ce))
            else:
                for i, model_weights in enumerate(glob(os.path.join(evaluator_dir, "evaluator_*.pt"))):
                    loader = get_data_loader(task, label_file, base_dir, batch_size=self.batch_size, split="test",
                                             data=data)
                    if encoder is None:
                        model = self.get_model(1)
                    else:
                        model = SSL(encoder).to(self.device)
                    state_dict = torch.load(model_weights)
                    model.load_state_dict(state_dict)
                    model.eval()
                    ce, y_true, y_pred = self._test(model, loader, self.device, loss, log_file)
                    _y_pred.append(expit(y_pred))
                    print("Model {} Test BCE: {:.7f}".format(i, ce))
                    out.write("Model {} Test BCE: {:.7f}\n".format(i, ce))
            _y_pred = np.average(_y_pred, axis=0)
            # auc_cat = lo.auc_per_cat(y_true, expit(_y_pred), labels)
            # out.write("\nPer category AUC:")
            # print("\nPer category AUC:")
            # for key in auc_cat.keys():
            # out.write("{}: {:.3f}\n".format(key, auc_cat[key]))
            # print("{}: {:.3f}\n".format(key, auc_cat[key]))
            # print(y_true[:10], _y_pred[:10], min(_y_pred), max(_y_pred))
            roc_score = roc_auc_score(y_true, _y_pred)

            # fpr, tpr, _ = roc_curve(y_true, _y_pred)
            # plt.plot(fpr, tpr, marker='.')
            # plt.show()

            report = classification_report(y_true, np.round(_y_pred), target_names=labels)
            conf_matrix = confusion_matrix(y_true, np.round(_y_pred))
            print("Ensemble AUC-ROC: {:.7f}\n{}\nConfusion Matrix:\n{}\n".format(roc_score, report, conf_matrix))
            out.write(
                "Seed: {}\tFolds: {}\nEnsemble AUC-ROC: {:.7f}\n{}\nConfusion Matrix:\n{}\n".format(seed, i + 1,
                                                                                                    roc_score, report,
                                                                                                    conf_matrix))

            id, scikit_X, scikit_y = get_scikit_loader(self.device, task, label_file, base_dir, split="test")
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
            print("Loaded pretrained model with success.")
        except FileNotFoundError:
            print("pretrained weights not found.")

        return model

    def _train(self, model, loader, optimizer, device, loss):
        model.train()
        y_true = []
        y_pred = []
        for i, data in enumerate(loader):
            X, y = data
            X, y = X.view(X.shape[0], 1, X.shape[1], X.shape[2]).to(device), y.to(device).float()
            optimizer.zero_grad()
            output = model(X).float()
            train_loss = loss(output.view(-1), y.view(-1))
            y_true.extend(y.tolist())
            y_pred.extend(output.tolist())
            train_loss = train_loss.cuda()
            train_loss.backward()
            optimizer.step()
        ce = loss(torch.tensor(y_pred).to(device).float().view(-1), torch.tensor(y_true).to(device).float().view(-1))

        return ce, y_true, y_pred

    @torch.no_grad()
    def _test(arch, model, loader, device, loss, log_file=None):

        model.eval()
        y_true = []
        y_pred = []
        if log_file is not None:
            with open(log_file, 'w') as f:
                f.write("ID,pred_proba,label\n")
        for i, data in enumerate(loader):
            id, X, y = data
            X, y = X.view(X.shape[0], 1, X.shape[1], X.shape[2]).to(device), y.to(device)
            output = model(X)

            if log_file is not None:
                with open(log_file, 'a+') as f:
                    f.write('{},{:.3f},{}\n'.format(id[0], expit(output.cpu()).item(), y.item()))

            y_true.extend(y.tolist())
            y_pred.extend(output.tolist())

        ce = loss(torch.tensor(y_pred).to(device).float().view(-1),
                  torch.tensor(y_true).to(device).unsqueeze(1).float().view(-1))
        return ce, y_true, y_pred

    def _step(self, model, xis, xjs, y=None):
        xis = xis.view(xis.shape[0], 1, xis.shape[1], xis.shape[2]).to(self.device)
        xjs = xjs.view(xjs.shape[0], 1, xjs.shape[1], xjs.shape[2]).to(self.device)
        # get the representations and the projections
        zis = model(xis)  # [N,C]
        # get the representations and the projections
        zjs = model(xjs)  # [N,C]
        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        inputs = torch.cat((zis, zjs)).view(zis.shape[0], 2, -1)
        loss = self.nt_xent_criterion(inputs, labels=y)
        return loss

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            counter = 0
            if self.exp in [2, 4, 5, 6]:
                for data in valid_loader:
                    xis, xjs = [], []
                    for sample in data:
                        xis.append(sample[0])
                        xjs.append(sample[1])
                    xis = torch.stack(tuple(xis))
                    xis = xis.view(xis.shape[0], xis.shape[2], xis.shape[3])
                    xjs = torch.stack(tuple(xjs))
                    xjs = xjs.view(xjs.shape[0], xjs.shape[2], xjs.shape[3])
                    xis = xis.to(self.device)
                    xjs = xjs.to(self.device)
                    loss = self._step(model, xis, xjs)
                    valid_loss += loss.item()
                    counter += 1
                valid_loss /= counter
            elif self.exp == 3:
                for xis, xjs, y in valid_loader:
                    xis = xis.to(self.device)
                    xjs = xjs.to(self.device)
                    loss = self._step(model, xis, xjs, y)
                    valid_loss += loss.item()
                    counter += 1
                valid_loss /= counter
            else:
                for xis, xjs in valid_loader:
                    xis = xis.to(self.device)
                    xjs = xjs.to(self.device)
                    loss = self._step(model, xis, xjs)
                    valid_loss += loss.item()
                    counter += 1
                valid_loss /= counter
        model.train()
        return valid_loss


def pretrain_(epochs, task, base_dir, log_dir, augment, train_prop=1, exp=None):
    log_file = os.path.join(log_dir, f"pretraintrain_log.txt")

    num_epochs = epochs
    batch_size = 16
    if train_prop == .01:
        batch_size = 5
    learning_rate = .0001

    with open(os.path.join(log_dir, "pretrain_params.txt"), "w") as f:
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Proportion of training data: {train_prop}\n")

    label_file = os.path.join(base_dir, "processed", "{}_labels.csv".format(task))
    dataset = get_dataset(task, label_file, base_dir, split="pretrain", train_prop=train_prop, exp=exp)

    learner = ContrastiveLearner(dataset, num_epochs, batch_size, log_dir)
    learner.pre_train(log_file, task, label_file, augment, learning_rate)


def train_(epochs, task, base_dir, log_dir, evaluator, augment, folds=5, train_prop=1, full_data=False):
    log_file = os.path.join(log_dir, f"train_log.txt")

    num_epochs = epochs
    batch_size = 16
    learning_rate = 0.0001
    if evaluator is not None:
        print("Evaluator: " + evaluator)
    with open(os.path.join(log_dir, "train_params.txt"), "w") as f:
        f.write(f"Folds: {folds}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Proportion of training data: {train_prop}\n")
        f.write(f"Evaluator: {evaluator}\n")

    label_file = os.path.join(base_dir, "processed", "{}_labels.csv".format(task))
    if not full_data:
        dataset = get_dataset(task, label_file, base_dir, split="train", train_prop=train_prop)
    else:
        dataset = get_dataset(task, label_file, base_dir, split="pretrain", train_prop=train_prop)

    learner = ContrastiveLearner(dataset, num_epochs, batch_size, log_dir)
    try:
        state_dict = torch.load(os.path.join(log_dir, 'encoder.pth'))
        encoder = learner.get_model(256)
        encoder.load_state_dict(state_dict)
    except FileNotFoundError:
        encoder = None
    learner.fine_tune(folds, task, label_file, log_file, augment, encoder, evaluator, learning_rate)


def test_(task, base_dir, log_dir, seed=None):
    log_file = os.path.join(log_dir, f"test_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Seed: {seed}\n")
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
    parser.add_argument("--mode", type=str, default="train", choices={"pretrain", "train", "test"})
    parser.add_argument("--task", type=str, default=None,
                        choices={"disease", "demo", "wheeze", "crackle", "heartchallenge", "heart"})
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--data", type=str, default="../data")
    parser.add_argument("--evaluator", type=str, default=None, choices={"knn", "linear", "fine-tune"})
    parser.add_argument("--augment", type=str, default=None, choices={"split", "raw", "spec", "spec+split"})
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train_prop", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--full_data", default=False)
    parser.add_argument("--exp", type=int, default=None)
    args = parser.parse_args()

    base_dir = os.path.join(os.getcwd(), args.data)
    log_dir = args.log_dir

    if args.mode == "pretrain":
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(base_dir, "logs", now)
        else:
            log_dir = os.path.join(base_dir, "logs", log_dir)
        print(f"Log Dir: {log_dir}")
        fi.make_path(log_dir)
        pretrain_(args.epochs, args.task, base_dir, log_dir, args.augment, args.train_prop, args.exp)
    elif args.mode == "train":
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(base_dir, "logs", now)
        else:
            log_dir = os.path.join(base_dir, "logs", log_dir)
        print(f"Log Dir: {log_dir}")
        fi.make_path(log_dir)
        train_(args.epochs, args.task, base_dir, log_dir, args.evaluator, args.augment, args.folds, args.train_prop,
               args.full_data)
    elif args.mode == "test":
        seed = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("seed:", seed)
        if log_dir is not None:
            log_dir = os.path.join(base_dir, "logs", log_dir)
        else:
            raise Exception("Testing requires log dir")
        print(f"Log Dir: {log_dir}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        test_(args.task, base_dir, log_dir, seed)
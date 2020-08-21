###This is defunct

import os
import time
import logging
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import argparse

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, MSELoss, Conv2d
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from classifier_dataloader import lungsounds_dataloader



class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(32, 4, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
        )

        self.linear_layers = Sequential(
            Linear(4 * 6 * 128, 3)
        )
    def forward(self,x):
        x=self.cnn_layers(x)
        x=x.view(x.size(0),-1)
        x=self.linear_layers(x)
        return x
def train(epoch, arch, model, loader, optimizer, device):
    model.train()

    # if epoch == 51:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    total = 0
    for i, data in enumerate(loader):
        datax = torch.unsqueeze(data['x'],1).to(device)
        optimizer.zero_grad()
        if arch == 'CNN':
            output = model(datax)

        loss = F.cross_entropy(output, data['y'].to(device))
        loss.backward()
        optimizer.step()
        if i==10:
            break

    return loss


@torch.no_grad()
def test(arch, model, loader, device):
    model.eval()

    loss_all = 0
    total = 0

    y_true = []
    y_pred = []

    for i, data in enumerate(loader):
        if arch == 'CNN':
            output = model(torch.unsqueeze(data['x'],1).to(device))

        loss = F.cross_entropy(output, data['y'].to(device))
        y_true.extend(data['y'].to(device).tolist())
        y_pred.extend(output.tolist())
        if i==10:
            break

    return loss, y_true, y_pred

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train_(architecture, base_dir, device, log_dir, seed=None, test_mode=False):
    logger = logging.getLogger('lungsounds_log')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    num_epochs = 100
    batch_size = 10
    learning_rate = 1e-4
    train_split = "train"
    val_split = "val"
    test_split = "test"
    data_dir=base_dir+"Respiratory_Sound_Database\\output"
    train_loader = lungsounds_dataloader(batch_size, data_dir,split_name=train_split)
    val_loader = lungsounds_dataloader(batch_size, data_dir, split_name=val_split)
    test_loader = lungsounds_dataloader(batch_size, data_dir, split_name=test_split)

    if not os.path.exists(os.path.join(log_dir, 'params.txt')):
        with open(os.path.join(log_dir, 'params.txt'), 'w') as f:
            f.write(f'Model: {architecture}\n')
            f.write(f'Epochs: {num_epochs}\n')
            f.write(f'Batch size: {batch_size}\n')
            f.write(f'Learning rate: {learning_rate}')



    if architecture == 'CNN':
        model = CNN().to(device)

    model.to(device)

    best_val_loss = 999

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs+1):
        start = time.time()
        train_loss = train(epoch, architecture, model, train_loader, optimizer, device)
        val_loss, y_true, y_pred = test(architecture, model, val_loader, device)
        if val_loss < best_val_loss:
            save_weights(model, os.path.join(log_dir, 'best_weights.pt'))
            best_val_loss = val_loss
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print('\tTrain CE: {:.7f}, Val CE: {:.7f}'.format(train_loss, val_loss))
        print(y_pred,y_true)
        logger.info('{:03d}\t{:.7f}\t{:.7f}\n'.format(epoch, train_loss, val_loss))

    if test_mode:
        test_file = os.path.join(log_dir, f'test_results.txt')
        model.load_state_dict(torch.load(os.path.join(log_dir, 'best_weights.pt')))
        ce, y_true, y_pred = test(architecture, model, test_loader, device)
        print('Test CE: {:.7f}'.format(ce))
        with open(test_file, 'a+') as out:
            out.write('{}\t{:.7f}\n'.format(seed, ce))



    return best_val_loss

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--architecture', type=str, default='CNN')
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    base_dir = os.getcwd()+'\\..\\data\\'
    log_dir = args.log_dir


    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(base_dir, 'logs', now)
        else:
            log_dir = os.path.join(base_dir, 'logs', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train_(args.architecture, base_dir, device, log_dir)
    elif args.mode == 'test':
        for seed in np.random.randint(0, 1000, size=3):
            print('seed:', seed)
            log_dir = os.path.join(base_dir, 'logs', f'test_{args.split}_{seed}')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train_(args.architecture, base_dir, device, log_dir, seed, test_mode=True)

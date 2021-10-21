import time
import numpy as np
import pandas as pd
import data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
import torch
import argparse
from models import SSL, ResNetSimCLR
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


def visualize_embeddings(task, base_dir, log_dir, model_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = ResNetSimCLR(out_dim=256, base_model="resnet18")
    state_dict = torch.load(os.path.join(log_dir, model_file))
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    label_file = os.path.join(base_dir, "processed", "{}_labels.csv".format(task))

    _, X, y = data.get_scikit_loader(device, task, label_file, base_dir, split='test', encoder=encoder)
    X = np.asarray(X)
    y = np.asarray(y)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X.reshape((X.shape[0], -1)))
    scaler.transform(X)

    feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    del X, y
    data_subset = df[feat_cols].values

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    df['tsne-pca50-one'] = tsne_pca_results[:, 0]
    df['tsne-pca50-two'] = tsne_pca_results[:, 1]
    plt.figure(figsize=(5, 4))
    sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df,
        legend="full",
        alpha=0.3,
    )
    plt.savefig(os.path.join(log_dir, model_file[:-4] + '_viz.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    args = parser.parse_args()
    log_dir = os.path.join(args.base_dir, "logs", args.log_dir)
    visualize_embeddings(args.task, args.base_dir, log_dir, args.model_file)

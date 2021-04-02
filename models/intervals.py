import glob
import sklearn
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib
import itertools
import os
from functools import partial
from multiprocessing import Pool
import torch
import random
from scipy.special import expit
from models import SSL, ResNetSimCLR, Logistic
from data import get_data_loader, get_dataset, get_scikit_loader
from contrastive import ContrastiveLearner

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

small_fin = [f'evaluator_{i}' for i in range(20)]
small_lin = [f'evaluator_{i}' for i in range(20, 40)]
medium_fin = [f'evaluator_{i}' for i in range(40, 50)]
medium_lin = [f'evaluator_{i}' for i in range(50, 60)]
large_fin = [f'evaluator_{i}' for i in range(60, 65)]
large_lin = [f'evaluator_{i}' for i in range(65, 70)]


def model_group(name):
    if name in small_fin or name in medium_fin or name in large_fin:
        return 'fine-tune'
    else:
        return 'linear'


class Model():
    def __init__(self, path, data, scikit, encoder):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.type = os.path.basename(os.path.dirname(path))
        evaluator = model_group(self.name)
        if evaluator == 'fine-tune':
            if encoder is None:
                model = ResNetSimCLR(out_dim=1, base_model="resnet18").to(device)
            else:
                model = SSL(encoder).to(device)

        else:
            data = scikit
            model = Logistic(data[1][0].shape[0]).to(device)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model.eval()

        if evaluator == 'fine-tune':
            self.scores = []
            for sample in data:
                _, X, y = sample
                X = X.view(1, 1, X.shape[0], X.shape[1]).to(device)
                self.scores.extend(expit(model(X).cpu().detach().numpy()))

        else:
            _, X, y = data
            X = torch.Tensor(X).to(device)
            self.scores = expit(model(X).cpu().detach()).tolist()


def single_replicate_all_models(gt, task, models, metric_str, replicate_num):
    sample_ids = np.random.choice(len(gt), size=len(gt), replace=True)
    performances = {}
    y_true = pd.Series(gt)[sample_ids].to_list()

    for model in models:
        y_score = pd.Series(model.scores)[sample_ids].to_list()
        performance = sklearn.metrics.roc_auc_score(y_true, y_score)
        performances[model.name] = performance
    return performances


def multiple_replicate_all_models(gt, task, models, num_replicates, metric_str):
    p = Pool(10)
    func = partial(single_replicate_all_models, gt, task, models, metric_str)
    results = p.map(func, range(num_replicates))
    return results


class ConfidenceGenerator():
    def __init__(self, confidence_level):
        self.records = []
        self.confidence_level = confidence_level

    @staticmethod
    def compute_cis(series, confidence_level):
        sorted_perfs = series.sort_values()
        lower_index = int(confidence_level / 2 * len(sorted_perfs)) - 1
        upper_index = int((1 - confidence_level / 2) * len(sorted_perfs)) - 1
        lower = sorted_perfs.iloc[lower_index].round(3)
        upper = sorted_perfs.iloc[upper_index].round(3)
        mean = sorted_perfs.mean().round(3)
        return lower, mean, upper

    def create_ci_record(self, perfs, name, perf_type):
        lower, mean, upper = ConfidenceGenerator.compute_cis(
            perfs, self.confidence_level)
        record = {"name": name,
                  "type": perf_type,
                  "lower": lower,
                  "mean": mean,
                  "upper": upper}
        self.records.append(record)

    def generate_cis(self, df):
        cols = {'small_fin': small_fin,
                'small_lin': small_lin,
                'medium_fin': medium_fin,
                'medium_lin': medium_lin,
                'large_fin': large_fin,
                'large_lin': large_lin,
                }

        for key in cols.keys():
            cols[key] = list(set(cols[key]).intersection(set(df.columns)))

        for group in cols.keys():
            for name in cols[group]:
                self.create_ci_record(df[name], name, 'individual')
            try:
                grp_avg = df[cols[group]].mean(axis=1)
                self.create_ci_record(grp_avg, group, 'average')
            except:
                pass
        # for (group1, group2) in itertools.combinations(cols.keys(), 2):
        #   group1_avg = df[cols[group1]].mean(axis=1)
        #  group2_avg = df[cols[group2]].mean(axis=1)
        # diff = group1_avg - group2_avg
        # self.create_ci_record(diff,
        #                     f'{group1}-{group2}',
        #                    'average difference')

        df = pd.DataFrame.from_records(self.records)

        return df


def run_stage_1_models(models, gt, task, num_replicates, metric_str, save_path):
    evaluations = multiple_replicate_all_models(
        gt,
        task,
        models,
        num_replicates,
        metric_str)
    df_task = pd.DataFrame.from_records(evaluations)
    if save_path:
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        df_task.to_csv(f"{save_path}/{task}.csv", index=False)


def run_stage1(task, num_replicates, metric_str, read_path, save_path=None, encoder=None, gt=None, scikit=None,
               data=None):
    model_paths = glob.glob(f'{read_path}/*.pt', recursive=True)

    models = [Model(path, data, scikit, encoder) for path in model_paths]

    run_stage_1_models(models, gt, task, num_replicates, metric_str, save_path)


def run_stage2(task, confidence_level, read_path, save_path=None):
    perfs = pd.read_csv(f"{read_path}/{task}.csv")
    cb = ConfidenceGenerator(confidence_level=confidence_level)
    df_task = cb.generate_cis(perfs)
    if save_path:
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        df_task.to_csv(f"{save_path}/{task}.csv", index=False)


def run_stage3(metric, read_path, save_path):
    results_paths = glob.glob(f'{read_path}/*.csv', recursive=True)

    dfs = []
    for path in results_paths:
        df = pd.read_csv(path)
        df['task'] = path.split('/')[-1].replace('.csv', '')
        dfs.append(df)

    concat_dfs = pd.concat(dfs)
    if save_path:
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        concat_dfs.to_csv(f"{save_path}/{metric}.csv", index=False)


def main(working_dir, dataset, num_replicates):
    metrics = 'auc'
    directories = glob.glob(os.path.join(working_dir, '*'))

    base_dir = os.path.split(working_dir)[0][:-5]
    label_file = os.path.join(f'{base_dir}', "processed", "{}_labels.csv".format(dataset))
    data = get_dataset(dataset, label_file, base_dir, split="test")

    for direct in directories:
        print(direct)
        learner = ContrastiveLearner(dataset, 0, 1, direct)

        try:
            state_dict = torch.load(os.path.join(direct, 'encoder.pth'))
            encoder = learner.get_model(256, restore=True)
        except FileNotFoundError:
            encoder = None

        scikit = id, X, y = get_scikit_loader(device, dataset, label_file, base_dir, split="test",
                                              encoder=encoder)

        stage_1_save_path = f'{direct}/results/raw_{metrics}'
        run_stage1(dataset,
                   num_replicates=num_replicates,
                   metric_str=metrics,
                   read_path=f'{direct}',
                   save_path=stage_1_save_path,
                   encoder=encoder,
                   gt=y,
                   data=data,
                   scikit=scikit
                   )

        stage_2_save_path = f'{direct}/results/processed_{metrics}'
        run_stage2(dataset,
                   read_path=stage_1_save_path,
                   save_path=stage_2_save_path,
                   confidence_level=0.05,
                   )

        stage_3_save_path = f'{direct}/results/'
        run_stage3(metrics,
                   read_path=stage_2_save_path,
                   save_path=stage_3_save_path
                   )


if __name__ == "__main__":
    num_replicates = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main('../data/logs/3_27', 'disease', num_replicates)
    #main('../heart/logs/3_27', 'heart', num_replicates)

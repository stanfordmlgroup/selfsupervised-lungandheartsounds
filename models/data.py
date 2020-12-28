import pandas as pd
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../utils")
from features import Mel, get_vggish_embedding, preprocess
from file import get_location
import augment as au
import torchvision.transforms as transforms
import time
import numpy as np
import h5py as h5
import copy

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


class LungDatasetExp3(Dataset):
    def __init__(self, label_file, base_dir, task, split="train", transform=None, train_prop=1.0, df=None, data=None,
                 exp=None):
        if df is None:
            df = pd.read_csv(label_file)
            splits_dir = os.path.join(base_dir, "splits")

            if split == "pretrain":
                df = self.get_split(df, os.path.join(splits_dir, "pretrain.txt"), train_prop=train_prop)
            else:
                raise Exception("Invalid split value. Must be pretrain.")
        if data is None:
            try:
                file = base_dir + '/processed/' + 'disease' + '_1.0.h5'
                file = h5.File(file, 'r')
                data = file[split][df.index.tolist()]
            except:
                raise Exception("Data not found")
        self.data = data
        self.split = split
        self.task = task
        self.labels = df
        counter = 0
        for idx, row in df.iterrows():
            self.labels.at[idx, 'y'] = self.get_class_val(row)
            self.labels.at[idx, 'location'] = get_location(df.at[idx, 'cycle'])
            self.labels.at[idx, 'index'] = counter
            counter += 1
        self.ID_list = self.labels.ID.unique()
        self.base_dir = base_dir
        self.transform = get_transform(transform)
        self.norm_func = transforms.Normalize([3.273], [100.439])

        self.exp = exp
        if self.exp in [4, 5, 6]:
            self.demo = pd.read_csv(os.path.join(base_dir, "demographics_MICE.csv"))
            self.demo = self.demo[self.demo.pt_num.isin(list(self.ID_list))]
            if self.exp == 4:
                self.only_adult = self.demo[self.demo['adult']]
                self.only_child = self.demo[self.demo['adult']]
            elif self.exp == 5 or self.exp == 6:
                self.only_female = self.demo[self.demo['sex_female'] == 1]
                self.only_male = self.demo[self.demo['sex_female'] == 0]
                if self.exp == 6:
                    self.only_male_adult = self.only_male[self.only_male['adult']]
                    self.only_male_child = self.only_male[self.only_male['adult']]
                    self.only_female_adult = self.only_female[self.only_female['adult']]
                    self.only_female_child = self.only_female[self.only_female['adult']]

    def __len__(self):
        return len(self.ID_list)

    def __getitem__(self, idx):
        id = self.ID_list[idx]

        # Positive pairs are from the same locations in the lung. Negative pairs are from different patients in the same
        # or different locations.
        if self.exp == 0:
            cycles = self.labels[self.labels['ID'] == id].drop(columns=['level_0'])
            cycles = cycles.reset_index()
            cycles_copy = copy.deepcopy(cycles)
            num_cycle = len(cycles)
            if num_cycle == 0:
                raise Exception('must have at least one cycle')
            elif num_cycle < 2:
                first_ind = 0
                second_ind = 0
            else:
                first_ind = random.randint(0, num_cycle - 1)
                first_cycle_loc = cycles.at[first_ind, 'location']
                # Find all cycles with the same location:
                cycles_copy.drop([first_ind])
                same_loc_list = cycles_copy[cycles_copy['location'] == first_cycle_loc]
                if len(same_loc_list) > 0:
                    second_ind = random.choice(same_loc_list.index)
                # No other cycles in the same location:
                else:
                    if len(cycles_copy) > 0:
                        second_ind = random.choice(cycles_copy.index)
                    else:
                        second_ind = first_ind  # Covered above but just in case

            first_cycle = self.data[int(cycles.at[first_ind, 'index'])]
            y = self.get_class_val(cycles.iloc[first_ind])
            second_cycle = self.data[int(cycles.at[second_ind, 'index'])]
            first_X, y1 = process_data("train", self.transform, first_cycle, y, self.norm_func)
            second_X, y1 = process_data("train", self.transform, second_cycle, y, self.norm_func)
            return first_X, second_X

        # Positive pairs are from different locations in the lung. Negative pairs are from different patients in the
        # same or different locations
        elif self.exp == 1:
            cycles = self.labels[self.labels['ID'] == id].drop(columns=['level_0'])
            cycles = cycles.reset_index()
            cycles_copy = copy.deepcopy(cycles)
            num_cycle = len(cycles)
            if num_cycle == 0:
                raise Exception('must have at least one cycle')
            elif num_cycle < 2:
                first_ind = 0
                second_ind = 0
            else:
                first_ind = random.randint(0, num_cycle - 1)
                first_cycle_loc = cycles.at[first_ind, 'location']
                # Find all cycles with the same location:
                cycles_copy.drop([first_ind])
                same_loc_list = cycles_copy[cycles_copy['location'] != first_cycle_loc]
                if len(same_loc_list) > 0:
                    second_ind = random.choice(same_loc_list.index)
                # No other cycles in the same location:
                else:
                    if len(cycles_copy) > 0:
                        second_ind = random.choice(cycles_copy.index)
                    else:
                        second_ind = first_ind  # Covered above but just in case

            first_cycle = self.data[int(cycles.at[first_ind, 'index'])]
            y = self.get_class_val(cycles.iloc[first_ind])
            second_cycle = self.data[int(cycles.at[second_ind, 'index'])]
            first_X, y1 = process_data("train", self.transform, first_cycle, y, self.norm_func)
            second_X, y1 = process_data("train", self.transform, second_cycle, y, self.norm_func)
            return first_X, second_X

        # Positive pairs are from the same locations in the lung. Negative pairs are from different patients in the
        # same location
        elif self.exp == 2:
            pair_list = []
            cycles = self.labels[self.labels['ID'] == id].drop(columns=['level_0'])
            cycles = cycles.reset_index()
            cycles_copy = copy.deepcopy(cycles)
            num_cycle = len(cycles)
            if num_cycle == 0:
                raise Exception('must have at least one cycle')
            elif num_cycle < 2:
                first_ind = 0
                first_cycle_loc = cycles.at[first_ind, 'location']
                second_ind = 0
            else:
                first_ind = random.randint(0, num_cycle - 1)
                first_cycle_loc = cycles.at[first_ind, 'location']

                # Find all cycles with the same location:
                cycles_copy.drop([first_ind])
                same_loc_list = cycles_copy[cycles_copy['location'] == first_cycle_loc]
                if len(same_loc_list) > 0:
                    second_ind = random.choice(same_loc_list.index)

                # No other cycles in the same location: (just grab same one twice, don't want to risk diff loc)
                else:
                    second_ind = first_ind

            first_cycle = self.data[int(cycles.at[first_ind, 'index'])]
            y = self.get_class_val(cycles.iloc[first_ind])
            second_cycle = self.data[int(cycles.at[second_ind, 'index'])]

            first_X, y1 = process_data("train", self.transform, first_cycle, y, self.norm_func)
            second_X, y1 = process_data("train", self.transform, second_cycle, y, self.norm_func)
            pair_list.append((first_X, second_X))

            # Generate rest of the batch:
            candidates = self.labels[self.labels['location'] == first_cycle_loc]
            candidates = candidates[candidates['ID'] != id]
            batch_patients = random.sample(list(candidates.ID.unique()), 15)
            for pat_id in batch_patients:
                cycles = candidates[candidates['ID'] == pat_id].drop(columns=['level_0'])
                cycles = cycles.reset_index()  # Get all the cycles for this person
                cycles_copy = copy.deepcopy(cycles)
                num_cycle = len(cycles)
                if num_cycle == 0:
                    raise Exception('must have at least one cycle')
                if num_cycle < 2:
                    first_ind = 0
                    second_ind = 0
                else:
                    first_ind = random.randint(0, num_cycle - 1)
                    first_cycle_loc = cycles.at[first_ind, 'location']

                    # Find all cycles with the same location:
                    cycles_copy.drop([first_ind])
                    same_loc_list = cycles_copy[cycles_copy['location'] == first_cycle_loc]
                    if len(same_loc_list) > 0:
                        second_ind = random.choice(same_loc_list.index)

                    # No other cycles in the same location: (just grab same one twice, don't want to risk diff loc)
                    else:
                        second_ind = first_ind

                first_cycle = self.data[int(cycles.at[first_ind, 'index'])]
                y = self.get_class_val(cycles.iloc[first_ind])
                second_cycle = self.data[int(cycles.at[second_ind, 'index'])]

                first_X, y1 = process_data("train", self.transform, first_cycle, y, self.norm_func)
                second_X, y1 = process_data("train", self.transform, second_cycle, y, self.norm_func)
                pair_list.append((first_X, second_X))
            return pair_list

        # Positive pair = same age band (child-child, adult-adult), negative pair = different age bands (child-adult)
        elif self.exp == 3:
            curr_demo = self.demo[self.demo['pt_num'] == id]

            # Choose two random cycles from this person:
            cycles = self.labels[self.labels['ID'] == id].drop(columns=['level_0'])
            cycles = cycles.reset_index()
            num_cycle = len(cycles)
            if num_cycle == 0:
                raise Exception('must have at least one cycle')
            elif num_cycle < 2:
                first_ind = 0
                second_ind = 0
            else:
                poss = [num for num in range(num_cycle)]
                sample_indexes = random.sample(poss, 2)
                first_ind = sample_indexes[0]
                second_ind = sample_indexes[1]

            first_cycle = self.data[int(cycles.at[first_ind, 'index'])]
            second_cycle = self.data[int(cycles.at[second_ind, 'index'])]
            if list(curr_demo.adult)[0]:  # Query format?
                y_val = 0
            else:
                y_val = 1

            first_X, y1 = process_data("train", self.transform, first_cycle, y_val, self.norm_func)
            second_X, y1 = process_data("train", self.transform, second_cycle, y_val, self.norm_func)
            return first_X, second_X, y_val

        # Positive pair = same recording, negative pair = different recording w/ similar age (child vs. adult)
        elif self.exp == 4:
            curr_demo = self.demo[self.demo['pt_num'] == id]
            if list(curr_demo.adult)[0]:
                is_adult = True
            else:
                is_adult = False

            pair_list = []  # (x1, x2, y) for 16
            # Generate batch of 16 people (15 others) with same age
            if is_adult:
                sample_df = copy.deepcopy(self.only_adult)
            else:
                sample_df = copy.deepcopy(self.only_child)
            sample_df = sample_df[sample_df.pt_num != id]
            batch_demo = sample_df.sample(15)
            # dummy y
            y_val = 0

            batch_ids = set(list(batch_demo.pt_num))  # Query format? (Want to get all the pt_nums)
            batch_ids.add(id)
            batch_ids = list(batch_ids)
            for curr_id in batch_ids:
                cycles = self.labels[self.labels['ID'] == curr_id].drop(columns=['level_0'])
                cycles = cycles.reset_index()
                cycles_copy = copy.deepcopy(cycles)
                num_cycle = len(cycles)
                if num_cycle == 0:
                    print(curr_id)
                    raise Exception('must have at least one cycle')
                elif num_cycle < 2:
                    first_ind = 0
                    second_ind = 0
                else:
                    first_ind = random.randint(0, num_cycle - 1)
                    first_cycle_name = cycles.at[first_ind, 'cycle']
                    first_cycle_name = first_cycle_name[:first_cycle_name.rfind('_')]
                    # Find all cycles with the same location:
                    cycles_copy.drop([first_ind])
                    same_loc_list = cycles_copy[cycles_copy['cycle'].str.contains(first_cycle_name)]
                    if len(same_loc_list) > 0:
                        second_ind = random.choice(same_loc_list.index)
                    # No other cycles in the same location:
                    else:
                        second_ind = first_ind  # Covered above but just in case

                first_cycle = self.data[int(cycles.at[first_ind, 'index'])]
                second_cycle = self.data[int(cycles.at[second_ind, 'index'])]

                first_X, y1 = process_data("train", self.transform, first_cycle, y_val, self.norm_func)
                second_X, y1 = process_data("train", self.transform, second_cycle, y_val, self.norm_func)

                pair_list.append((first_X, second_X))
            return pair_list

        # Positive pair = same recording, negative pair = different recording w/ similar sex attributes
        elif self.exp == 5:
            curr_demo = self.demo[self.demo['pt_num'] == id]
            if list(curr_demo.sex_female)[0] == 1:
                is_female = True
            else:
                is_female = False

            pair_list = []  # (x1, x2, y) for 16
            # Generate batch of 16 people (15 others) with same sex
            if is_female:
                sample_df = copy.deepcopy(self.only_female)
            else:
                sample_df = copy.deepcopy(self.only_male)
            sample_df = sample_df[sample_df.pt_num != id]
            batch_demo = sample_df.sample(15)
            # dummy y
            y_val = 0

            batch_ids = set(list(batch_demo.pt_num))  # Query format? (Want to get all the pt_nums)
            batch_ids.add(id)
            batch_ids = list(batch_ids)
            for curr_id in batch_ids:
                cycles = self.labels[self.labels['ID'] == curr_id].drop(columns=['level_0'])
                cycles = cycles.reset_index()
                cycles_copy = copy.deepcopy(cycles)
                num_cycle = len(cycles)
                if num_cycle == 0:
                    raise Exception('must have at least one cycle')
                elif num_cycle < 2:
                    first_ind = 0
                    second_ind = 0
                else:
                    first_ind = random.randint(0, num_cycle - 1)
                    first_cycle_name = cycles.at[first_ind, 'cycle']
                    first_cycle_name = first_cycle_name[:first_cycle_name.rfind('_')]
                    # Find all cycles with the same location:
                    cycles_copy.drop([first_ind])
                    same_loc_list = cycles_copy[cycles_copy['cycle'].str.contains(first_cycle_name)]
                    if len(same_loc_list) > 0:
                        second_ind = random.choice(same_loc_list.index)
                    # No other cycles in the same location:
                    else:
                        second_ind = first_ind  # Covered above but just in case

                first_cycle = self.data[int(cycles.at[first_ind, 'index'])]
                second_cycle = self.data[int(cycles.at[second_ind, 'index'])]

                first_X, y1 = process_data("train", self.transform, first_cycle, y_val, self.norm_func)
                second_X, y1 = process_data("train", self.transform, second_cycle, y_val, self.norm_func)

                pair_list.append((first_X, second_X))
            return pair_list

        # Positive pair = same recording, negative pair = different recording w/ similar age & sex attributes
        elif self.exp == 6:
            curr_demo = self.demo[self.demo['pt_num'] == id]

            if list(curr_demo.sex_female)[0] == 1:
                is_female = True
            else:
                is_female = False

            if list(curr_demo.adult)[0]:
                is_adult = True
            else:
                is_adult = False

            pair_list = []  # (x1, x2, y) for 16
            y_val = 0
            # Generate batch of 16 people (15 others) with same age
            if is_female:
                if is_adult:
                    sample_df = copy.deepcopy(self.only_female_adult)
                else:
                    sample_df = copy.deepcopy(self.only_female_child)
            else:
                if is_adult:
                    sample_df = copy.deepcopy(self.only_male_adult)
                else:
                    sample_df = copy.deepcopy(self.only_male_child)
            sample_df = sample_df[sample_df.pt_num != id]
            try:
                batch_demo = sample_df.sample(15)
            except:
                batch_demo = sample_df.sample(7)
            # dummy y
            y_val = 0

            batch_ids = set(list(batch_demo.pt_num))  # Query format? (Want to get all the pt_nums)
            batch_ids.add(id)
            batch_ids = list(batch_ids)
            for curr_id in batch_ids:
                cycles = self.labels[self.labels['ID'] == curr_id].drop(columns=['level_0'])
                cycles = cycles.reset_index()
                cycles_copy = copy.deepcopy(cycles)
                num_cycle = len(cycles)
                if num_cycle == 0:
                    raise Exception('must have at least one cycle')
                elif num_cycle < 2:
                    first_ind = 0
                    second_ind = 0
                else:
                    first_ind = random.randint(0, num_cycle - 1)
                    first_cycle_name = cycles.at[first_ind, 'cycle']
                    first_cycle_name = first_cycle_name[:first_cycle_name.rfind('_')]
                    # Find all cycles with the same location:
                    cycles_copy.drop([first_ind])
                    same_loc_list = cycles_copy[cycles_copy['cycle'].str.contains(first_cycle_name)]
                    if len(same_loc_list) > 0:
                        second_ind = random.choice(same_loc_list.index)
                    # No other cycles in the same location:
                    else:
                        second_ind = first_ind  # Covered above but just in case

                first_cycle = self.data[int(cycles.at[first_ind, 'index'])]
                second_cycle = self.data[int(cycles.at[second_ind, 'index'])]

                first_X, y1 = process_data("train", self.transform, first_cycle, y_val, self.norm_func)
                second_X, y1 = process_data("train", self.transform, second_cycle, y_val, self.norm_func)

                pair_list.append((first_X, second_X))
            return pair_list

    def get_class_val(self, row):
        if self.task == "demo":
            # Takes in a single row in a DataFrame, return an Int.
            # 0: Abnormal, 1: Normal
            label = row["diagnosis"]
            if label == -1:
                return 0
            else:
                return 1

    def get_split(self, df, split_file_path, train_prop=1.0):
        # Takes in a DataFrame and a path to a file of only Ints denoting Patient ID.
        # Returns a DataFrame of only samples with Patient IDs contained in the split.
        IDs = set()
        with open(split_file_path, "r") as f:
            IDs = set([line.strip() for line in f])
            df = df[df.ID.isin(IDs)]
            df = df.reset_index(drop=True)
            cycles = set(list(df.cycle))
            cycles = set(random.sample(cycles, int(train_prop * len(cycles))))
        return df[df.cycle.isin(cycles)].reset_index(drop=True)


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
                file = base_dir + '/processed/' + task + '_1.0.h5'
                file = h5.File(file, 'r')
                data = file[split][df.index.tolist()]
            except:
                raise Exception("Data not found")
        self.data = data
        self.split = split
        self.task = task
        self.labels = df
        for idx, row in df.iterrows():
            df.at[idx, 'y'] = self.get_class_val(row)
        self.base_dir = base_dir
        self.transform = get_transform(transform)
        self.norm_func = transforms.Normalize([3.273], [100.439])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        X = self.data[idx]
        y = self.get_class_val(row)
        if self.split == "test":
            X, y = process_data(self.split, self.transform, X, y, self.norm_func)
            return row["cycle"], X, y
        return process_data(self.split, self.transform, X, y, self.norm_func)

    def get_class_val(self, row):
        if self.task == "disease":
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

    def get_split(self, df, split_file_path, train_prop=1.0):
        # Takes in a DataFrame and a path to a file of only Ints denoting Patient ID.
        # Returns a DataFrame of only samples with Patient IDs contained in the split.
        IDs = set()
        with open(split_file_path, "r") as f:
            IDs = set([line.strip() for line in f])
            df = df[df.ID.isin(IDs)]
            df = df.reset_index()
            cycles = set(list(df.cycle))
            cycles = set(random.sample(cycles, int(train_prop * len(cycles))))
        return df[df.cycle.isin(cycles)]


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
                file = base_dir + '/processed/heart_1.0.h5'
                file = h5.File(file, 'r')
                data = file[split][df.index.tolist()]
            except:
                raise Exception("Data not found")
        self.data = data
        self.split = split
        self.task = task
        self.labels = df
        for idx, row in df.iterrows():
            df.at[idx, 'y'] = self.get_class_val(row)
        self.base_dir = base_dir
        self.transform = get_transform(transform)
        self.norm_func = transforms.Normalize([.7196], [32.07])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        X = self.data[idx]
        # Get label
        y = self.get_class_val(row)
        if self.split == "test":
            X, y = process_data(self.split, self.transform, X, y, self.norm_func)
            return row["ID"], X, y
        return process_data(self.split, self.transform, X, y, self.norm_func)

    def get_class_val(self, row):
        if self.task == "heart":
            # Takes in a single row in a DataFrame, return an Int.
            # 0: Abnormal, 1: Normal
            label = row["label"]
            if label == -1:
                return 0
            else:
                return 1

    def get_split(self, df, split_file_path, train_prop=1.0):
        # Takes in a DataFrame and a path to a file of only Ints denoting Patient ID.
        # Returns a DataFrame of only samples with Patient IDs contained in the split.
        IDs = set()
        with open(split_file_path, "r") as f:
            IDs = set([line.strip() for line in f])
            df = df[df.ID.isin(IDs)]
            df = df.reset_index()
            IDs = set(random.sample(IDs, int(train_prop * len(IDs))))
        return df[df.ID.isin(IDs)]


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
                file = base_dir + '/processed/heartchallenge_1.0.h5'
                file = h5.File(file, 'r')
                data = file[split][df.index.tolist()]
            except:
                raise Exception("Data not found")
        self.data = data

        self.split = split
        self.task = task
        self.labels = df
        for idx, row in df.iterrows():
            df.at[idx, 'y'] = self.get_class_val(row)
        self.base_dir = base_dir
        self.transform = get_transform(transform)
        self.norm_func = transforms.Normalize([.9364], [33.991])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        X = self.data[idx]
        # Get label
        y = self.get_class_val(row)
        if self.split == "test":
            X, y = process_data(self.split, self.transform, X, y, self.norm_func)
            return row["ID"], X, y
        return process_data(self.split, self.transform, X, y, self.norm_func)

    def get_class_val(self, row):
        if self.task == "heartchallenge":
            label = row["label"]
            if label == -1:
                return 0
            else:
                return 1

    def get_split(self, df, split_file_path, train_prop=1.0):
        # Takes in a DataFrame and a path to a file of only Ints denoting Patient ID.
        # Returns a DataFrame of only samples with Patient IDs contained in the split.
        IDs = set()
        with open(split_file_path, "r") as f:
            IDs = set([line.strip() for line in f])
            df = df[df.ID.isin(IDs)]
            df = df.reset_index()
            IDs = set(random.sample(IDs, int(train_prop * len(IDs))))
        return df[df.ID.isin(IDs)]


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


def process_data(mode, augment, X, y, norm_func):
    if mode == "pretrain":  # mode: pretrain, train, test
        xi = norm_func(torch.Tensor(augment(X)).unsqueeze(0)).squeeze(0)
        xj = norm_func(torch.Tensor(augment(X)).unsqueeze(0)).squeeze(0)
        return xi, xj

    X = norm_func(torch.Tensor(augment(X)).unsqueeze(0)).squeeze(0)
    return X, y


def get_dataset(task, label_file, base_dir, split="train", train_prop=1.0, df=None, transform=None, data=None,
                exp=None):
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
    elif task == "demo":
        if split == "pretrain":
            dataset = LungDatasetExp3(label_file, base_dir, task, split=split, transform=transform,
                                      train_prop=train_prop, df=df, data=data, exp=exp)
        else:
            dataset = LungDataset(label_file, base_dir, "disease", split=split, transform=transform,
                                  train_prop=train_prop,
                                  df=df, data=data)
    return dataset


def get_data_loader(task, label_file, base_dir, batch_size=128, split="train", transform=None, df=None, data=None,
                    exp=None):
    dataset = get_dataset(task, label_file, base_dir, split=split, df=df, transform=transform, data=data, exp=exp)
    shuffle = True
    if split == "test":
        shuffle = False
    return DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True)


def get_scikit_loader(device, task, label_file, base_dir, split="train", df=None, encoder=None, data=None):
    dataset = get_dataset(task, label_file, base_dir, split=split, df=df, data=data)
    X = []
    y = []
    id = []
    for data in dataset:
        if split == "test":
            x = data[1]
            y.append(data[2])
            id.append(data[0])
        else:
            y.append(data[1])
            x = data[0]
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
    with h5.File(h5_dir, 'w') as f:
        for split in __splits__:
            print(split)
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
            audio_samples = np.array(audio_samples)
            f.create_dataset(split, data=audio_samples)


if __name__ == '__main__':
    __tasks__ = ['disease', 'crackle', 'wheeze', 'heartchallenge']
    __train_props__ = [1.0]
    for task in __tasks__:
        if task == 'disease' or task == 'crackle' or task == 'wheeze':
            base_dir = '../data'
        else:
            base_dir = '../' + task
        label_file = os.path.join(base_dir, "processed", "{}_labels.csv".format(task))
        for train_prop in __train_props__:
            h5ify(base_dir, label_file, train_prop)

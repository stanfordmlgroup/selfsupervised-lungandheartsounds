import random
from collections import Counter
import os
import pandas as pd
import argparse
import file as fi


# create split files
def split(input_path='../data', split_dir='../data/splits', seed=252,
          distribution_dict=None):
    if distribution_dict is None or not distribution_dict:
        distribution_dict = {'COPD': 10, 'Healthy': 10, 'URTI': 3, 'Bronchiectasis': 2, 'Bronchiolitis': 2,
                             'Pneumonia': 2, 'LRTI': 1, 'Asthma': 0}
    random.seed(seed)
    # load diagnosis cv and extract unqiue diagnoses
    diag_csv = os.path.join(input_path, 'patient_diagnosis.csv')
    diagnosis = pd.read_csv(diag_csv, names=['pId', 'diagnosis'])
    ds = list(diagnosis['diagnosis'].unique())
    pIdbydiagnosis = {}
    train_list = []

    for diag in ds:
        # create dictionary of lists of files; split by diagnosis
        pIdbydiagnosis[diag] = (list(diagnosis[diagnosis["diagnosis"] == diag]["pId"]))
        # one list of all the files
        train_list.extend(list(diagnosis[diagnosis["diagnosis"] == diag]["pId"]))

    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    train_split = os.path.join(split_dir, 'train.txt')
    test_split = os.path.join(split_dir, 'test.txt')
    with open(test_split, "w") as test:
        test_list = []
        # take a random sample for each label
        for diag in ds:
            test_list.extend(random.sample(pIdbydiagnosis[diag], distribution_dict[diag]))
        for pId in test_list:
            test.write(str(pId) + "\n")
        print("Test samples: " + str(len(test_list)))

    with open(train_split, "w") as train:
        # create train split by subtracting out test files
        train_list = list(Counter(train_list) - Counter(test_list))
        for pId in train_list:
            train.write(str(pId) + "\n")
        print("Train samples: " + str(len(train_list)))


if __name__ == "__main__":
    # create splits based 30 tests of 10 patients each, OPTIONAL: provide a seed number for reproducibility and
    # custom split dictionary with keys as strings matching raw diagnoses and values equaling desired number of that
    # class in test set.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../data', help='data path')
    parser.add_argument('--split', default='../data/splits', help='location of output files')
    parser.add_argument('--seed', default=252, help='seed for sampling')
    parser.add_argument('--distribution', default=None, help='dict containing desired test samples from each '
                                                             'diagnosis class')
    # distribution_dict = {'COPD': 10, 'Healthy': 10, 'URTI': 3, 'Bronchiectasis': 2, 'Bronchiolitis': 2,
    # 'Pneumonia': 2, 'LRTI': 1, 'Asthma': 0}
    args = parser.parse_args()

    fi.make_path(args.split)
    split(args.data, args.split, args.seed, args.distribution)

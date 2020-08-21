import random
from collections import Counter
import os
import pandas as pd


# create split files
def split(input_path='../data/processed', split_dir='../data/splits', seed=252):
    random.seed(seed)
    # load diagnosis cv and extract unqiue diagnoses
    diag_csv = os.path.join(input_path, '../patient_diagnosis.csv')
    diagnosis = pd.read_csv(diag_csv, names=['pId', 'diagnosis'])
    ds = list(diagnosis['diagnosis'].unique())
    pIdbydiagnosis = []
    train_list = []

    for diag in ds:
        # create list of lists of files; split by diagnosis
        pIdbydiagnosis.append(list(diagnosis[diagnosis["diagnosis"] == diag]["pId"]))
        # one list of all the files
        train_list.extend(list(diagnosis[diagnosis["diagnosis"] == diag]["pId"]))

    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    train_split = os.path.join(split_dir, 'train.txt')
    test_split = os.path.join(split_dir, 'test.txt')
    with open(test_split, "w") as test:
        test_list = []
        # take a random sample for each disease
        for diag in ds:
            if diag == "COPD":
                test_list.extend(random.sample(pIdbydiagnosis[3], 10))
            elif diag == "Healthy":
                test_list.extend(random.sample(pIdbydiagnosis[1], 10))
            elif diag == "LRTI":
                test_list.extend(random.sample(pIdbydiagnosis[4], 1))
            elif diag == "URTI":
                test_list.extend(random.sample(pIdbydiagnosis[0], 3))
            elif diag == "Bronchiectasis":
                test_list.extend(random.sample(pIdbydiagnosis[5], 2))
            elif diag == "Pneumonia":
                test_list.extend(random.sample(pIdbydiagnosis[6], 2))
            elif diag == "Bronchiolitis":
                test_list.extend(random.sample(pIdbydiagnosis[7], 2))
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
    split()

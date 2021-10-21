import os
import pandas as pd
import numpy as np


def get_diag_label(pt_id, label_df):
    """gets the diagnosis label for a given patient id within the supplied label dataframe"""
    return label_df[label_df['pt_id'] == int(pt_id)]['label'].iloc[0]


def get_symptom_label(cycle, label_df):
    """gets the symptom label for a given cycle within the supplied label dataframe"""
    crackles = label_df[label_df['cycle'] == cycle]['crackles'].iloc[0]
    wheezes = label_df[label_df['cycle'] == cycle]['wheezes'].iloc[0]
    return [crackles, wheezes]


def process_diag_labels(data_path='../data'):
    """loads the diagnosis csv, applies a mapping to get labels, and saves the labels"""
    processed_dir = os.path.join(data_path, 'processed')
    diag_csv = os.path.join(data_path, 'patient_diagnosis.csv')
    diagnosis = pd.read_csv(diag_csv, names=['pId', 'diagnosis'])
    diagnosis['diagnosis'] = diagnosis['diagnosis'].apply(diag_map)
    diagnosis.to_csv(os.path.join(processed_dir, 'disease_labels.csv'), index=False, header=["pt_id", "label"])
    return diagnosis


def diag_map(diag):
    """map function to generate labels from raw csv"""
    if diag == "Healthy":
        return diag
    if diag == "COPD":
        return diag
    else:
        return diag.replace(diag, "Other")


def diag_one_hot(label):
    """one hot embedding for diagnosis labels"""
    if label == "Healthy":
        return 0
    elif label == 'COPD':
        return 1
    else:  # Other
        return 2


def symptom_one_hot(label_list):
    if label_list == [1, 1]:
        return 3
    elif label_list == [1, 0]:
        return 2
    elif label_list == [0, 1]:
        return 1
    else:
        return 0


def class_distribution(task, label_file):
    labelsdf = pd.read_csv(label_file)
    IDs = set()

    if task == 'disease' or task == 'crackle' or task == 'wheeze' or task == "demo":
        base_dir = '../data'
    else:
        base_dir = '../' + task
    split_file_path = os.path.join(base_dir, "splits", 'train.txt')

    with open(split_file_path, "r") as f:
        IDs = set([line.strip() for line in f])

    labelsdf=labelsdf[labelsdf.ID.isin(IDs)]

    distribution = []
    if task == "symptom":
        labelsdf = labelsdf[["crackles", "wheezes"]]
        labelsdf["combined"] = labelsdf.values.tolist()
        labelsdf = labelsdf["combined"].apply(symptom_one_hot)
    # elif task == "disease":
    #     labelsdf = labelsdf["diagnosis"].apply(diag_one_hot)
    elif task == "heart" or task == "heartchallenge":
        labelsdf = labelsdf["label"].apply(heart_recover_label).apply(heart_one_hot)
    elif task == "disease" or task == "wheeze" or task == "crackle" or task == "demo":
        labelsdf = labelsdf["diagnosis"].apply(heart_recover_label).apply(heart_one_hot)
    # elif task == "heartchallenge":
    #     labelsdf = labelsdf["label"].apply(heartchallenge_one_hot)
    classes = np.sort(labelsdf.unique())
    for class_ in classes:
        distribution.append(labelsdf[labelsdf == class_].count())
    return distribution


def heart_challenge_row_label(row):
    for key in row.keys():
        if row[key] == 1:
            return key


def heartchallenge_one_hot(label):
    if label == "Artifact":
        return 4
    elif label == "Extrasound":
        return 3
    elif label == "Extrastole":
        return 2
    elif label == "Murmur":
        return 1
    else:
        return 0


def heart_one_hot(label):
    if label == "Normal":
        return 0
    elif label == "Abnormal":
        return 1


def heart_recover_label(label):
    if label == -1:
        return "Normal"
    elif label == 1:
        return "Abnormal"

import os
import pandas as pd


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

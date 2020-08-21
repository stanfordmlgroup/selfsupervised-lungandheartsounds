# gets the diagnosis label for a given patient id within the supplied label dataframe
def get_diag_label(pt_id, label_df):
    return label_df[label_df['pt_id'] == int(pt_id)]['label'].iloc[0]


# gets the symptom label for a given cycle within the supplied label dataframe
def get_symptom_label(cycle, label_df):
    crackles = label_df[label_df['cycle'] == cycle]['crackles'].iloc[0]
    wheezes = label_df[label_df['cycle'] == cycle]['wheezes'].iloc[0]
    return [crackles, wheezes]


# one hot embedding for diagnosis labels
def diag_one_hot(label):
    if label == "Healthy":
        return 0
    elif label == 'COPD':
        return 1
    else:  # Other
        return 2

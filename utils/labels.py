
def get_label(pt_id, label_df):
    return label_df[label_df['pt_id'] == int(pt_id)]['label'].iloc[0]

def diag_one_hot(label):
    if label=="Healthy":
        return 0
    elif label == 'COPD':
        return 1
    else:#Other
        return 2


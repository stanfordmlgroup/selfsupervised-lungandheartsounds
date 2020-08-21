import splitfolders

def split(input_path='../data/processed', output_path='../data/output', seed=252):
    splitfolders.ratio(input_path,output=output_path,seed=seed, ratio=(0.8,0.0,0.2))

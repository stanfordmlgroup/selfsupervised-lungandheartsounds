import os
import torch
import pandas as pd
import numpy as np
import math
import librosa as lb # https://librosa.github.io/librosa/
import soundfile as sf # https://pysoundfile.readthedocs.io/en/latest/
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats

from torch.utils.data import Dataset, DataLoader
os.environ['KMP_DUPLICATE_LIB_OK']='True'
vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
vggish.eval()
from glob import glob
import splitfolders

class LungSoundsDataset(Dataset):
    def __init__(self,root,label_file,transform=None):
        self.labels=pd.read_csv(label_file)
        self.root=root
        self.wav_idx_dict=self.get_idx_mapping()
        self.idx_wav_dict={v:k for k,v in self.wav_idx_dict.items()}

    @property
    def raw_wavs(self):
        return glob(self.root+"\\*\\*.wav")

    def get_idx_mapping(self):
        self.wav_idx_dict = {}
        i = 0
        for file in self.raw_wavs:
            wav=file
            self.wav_idx_dict[wav] = i
            i += 1
        return self.wav_idx_dict

    def __len__(self):
        return len(self.raw_wavs)

    def __getitem__(self,idx):
        wav_dir=self.idx_wav_dict[idx]
        x=vggish.forward(wav_dir)
        pt_id=self.idx_wav_dict[idx].split("\\")[-1].split("_")[0]
        y=np.array(map(get_label(pt_id, self.labels)))

        x=torch.as_tensor(x)
        y=torch.as_tensor(y, dtype=torch.long)


        return {"x":x,"y":y}
    def get_label(self, pt_id, label_df):
        return label_df[label_df['pt_id'] == int(pt_id)]['label'].iloc[0]

    def map(self,label):
        out=None
        if label == 'Healthy':
            out=0
        elif label == 'COPD':
            out = 1
        elif label == 'Other':
            out = 2
        return out


def lungsounds_dataloader(batch_size,data_dir,label_file="B:\\Users\\psoni\\Projects\\lungsounds\\torchvggish\\data\\Respiratory_Sound_Database\\patient_diagnosis.csv",split_name=None):
    dataset=LungSoundsDataset(label_file=label_file, root=os.path.join(data_dir,split_name))

    return DataLoader(dataset,batch_size,shuffle=True)

def tokenize_file(filename):
    return filename.split('_')

def slice_data(start, end, raw_data,  sample_rate):
    max_ind = len(raw_data)
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]

def compute_len(samp_rate=22050, time=0, acquisition_mode=0):
    '''Computes the supposed length of sliced data
        samp_size = sample size from the data
        samp_rate = sampling rate. by default since we're working on 24-bit files, we'll use 96kHz
        time = length of time for the audio file. by default we'll use the max we have which is 5.48
        acquisition_mode = either mono or stereo. 0 for mono, 1 for stereo
    '''
    if acquisition_mode == 1: #ac mode is single channel which means it's 'mono'
        comp_len = samp_rate * time
    else: #stereo
        comp_len = (samp_rate * time)

    return comp_len

def process(data_path='../../data'):
    diag_csv = os.path.join(data_path,'patient_diagnosis.csv')
    diagnosis = pd.read_csv(diag_csv, names=['pId', 'diagnosis'])
    diagnosis['diagnosis']=diagnosis['diagnosis'].apply(diag_map)

    diagnosis.to_csv(os.path.join(os.getcwd(),'../data/disease_labels.csv'), index=False, header=["pt_id","label"])
    ds = diagnosis['diagnosis'].unique()
    audio_text_loc = os.path.join(data_path,'raw')
    files = [s.split('.')[0] for s in os.listdir(path=audio_text_loc) if '.txt' in s]
    files_ = []

    for f in files:
        df = pd.read_csv(audio_text_loc + '/' + f + '.txt', sep='\t', names=['start', 'end', 'crackles', 'wheezes'])
        df['filename'] = f
        # get filename features
        f_features = tokenize_file(f)
        df['pId'] = f_features[0]
        df['ac_mode'] = f_features[3]

        files_.append(df)

    files_df = pd.concat(files_)
    files_df.reset_index()
    files_df['pId'] = files_df['pId'].astype('float64')
    files_df = pd.merge(files_df, diagnosis, on='pId')
    files_df['len_per_slice'] = files_df['end'].sub(files_df['start'], axis=0)
    force_max_len = math.ceil(boxplot_stats(files_df['len_per_slice'])[0]['whishi'])
    os.makedirs('../data/processed')
    for d in ds:
        path = os.path.join('../data/processed', d)
        os.makedirs(path)
    i = 0  # iterator for file naming

    for idx, row in files_df.iterrows():
        filename = row['filename']
        start = row['start']
        end = row['end']
        diag = row['diagnosis']

        # check len and force to 6 sec if more than that
        if force_max_len < end - start:
            end = start + force_max_len

        aud_loc = audio_text_loc + '/' + f + '.wav'

        if idx != 0:
            if files_df.iloc[idx - 1]['filename'] == filename:
                i = i + 1
            else:
                i = 0
        n_filename = filename + '_' + str(i) + '.wav'
        path = '../data/processed/' + diag + '/' + n_filename

        print('processing ' + n_filename + '...')

        data, samplingrate = lb.load(aud_loc)
        sliced_data = slice_data(start=start, end=end, raw_data=data, sample_rate=samplingrate)

        # pad audio if < forced_max_len
        a_len = compute_len(samp_rate=samplingrate, time=force_max_len, acquisition_mode=row['ac_mode'] == 'sc')
        padded_data = lb.util.pad_center(sliced_data, a_len)

        sf.write(file=path, data=padded_data, samplerate=samplingrate)

def diag_map(diag):
    if diag=="Healthy":
        return diag
    if diag=="COPD":
        return diag
    else:
        return diag.replace(diag, "Other")

def split(input_path='../data/processed', output_path='../data/output', seed=252):
    splitfolders.ratio(input_path,output=output_path,seed=seed, ratio=(0.8,0.0,0.2))

if __name__=="__main__":
    process()
    split()


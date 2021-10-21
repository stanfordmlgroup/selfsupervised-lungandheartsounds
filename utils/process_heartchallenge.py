import os
import pandas as pd
from tqdm import tqdm
import math
from matplotlib.cbook import boxplot_stats
import librosa as lb  # https://librosa.github.io/librosa/
import soundfile as sf  # https://pysoundfile.readthedocs.io/en/latest/
import file as fi
import labels as la
import argparse
from glob import glob
import numpy as np

def slice_data(start, end, raw_data, sample_rate):
    """splices audio data to desired start, end timestamps"""
    max_ind = len(raw_data)
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]


def compute_len(samp_rate=22050, time=0):
    """Computes the supposed length of sliced data

    samp_size = sample size from the data
    samp_rate = sampling rate. by default since we're working on 24-bit files, we'll use 96kHz
    time = length of time for the audio file. by default we'll use the max we have which is 5.48
    acquisition_mode = either mono or stereo. 0 for mono, 1 for stereo
    """
    comp_len = samp_rate * time

    return comp_len


def get_labeldf(audio_loc):
    data = []
    folders = glob(audio_loc + "/*")
    for folder in folders:
        if fi.split_path(folder) == "Test":
            testA_df = pd.read_excel(audio_loc + "/../Challenge2_evaluation_sheet.xlsx", sheet_name=0)[
                ["Dataset A", "Normal", "Murmur", "Extra sound", "Artifact"]].head(52).rename(columns={'Extra sound': 'Extrasound'})
            for idx, row in testA_df.iterrows():
                file=row["Dataset A"].split(".")[0]
                audio=sf.SoundFile(os.path.join(folder, file+".wav"))
                end=len(audio)/audio.samplerate
                data.append([file, la.heart_challenge_row_label(row),0,end])
            testB_df = pd.read_excel(audio_loc + "/../Challenge2_evaluation_sheet.xlsx", sheet_name=1)[
                ["Dataset B", "Normal", "Murmur", "Extrastole"]].head(195)
            for idx, row in testB_df.iterrows():
                file = row["Dataset B"].split(".")[0]
                audio = sf.SoundFile(os.path.join(folder, file + ".wav"))
                end = len(audio) / audio.samplerate
                data.append([file, la.heart_challenge_row_label(row), 0, end])
        else:
            files = glob(folder + "/*")
            for file in files:
                file = fi.split_path(file).split(".")[0]
                audio = sf.SoundFile(os.path.join(folder, file + ".wav"))
                end = len(audio) / audio.samplerate
                data.append([file, fi.split_path(folder), 0, end])
    df=pd.DataFrame(data=data, columns=['ID', 'label','start','end'])
    return df

def max_length(files_df):
    files_df['len_per_slice'] = files_df['end'].sub(files_df['start'], axis=0)
    return math.ceil(boxplot_stats(files_df['len_per_slice'])[0]['whishi'])


def process(data_path='../heartchallenge', labels_only=False):
    """Process the raw wavs to get each slice

    INPUT: A data dir where audio files and cycle info is stored in data_path/audio_txt_files
    OUTPUTS: data_path/processed with each recording split into the respiratory cycles
             disease_labels.csv: file with patient ids and their disease diagnoses mapped by Healthy, COPD, or Other
             symptoms_labels.csv: file with respiratory cycle ids and presence of crackles and wheezes

    The function parses through each of the txt files and filenames to identify start and end for each cycle. A good
    clipping value is then calculated. Each slice from the raw audios is processed so that they are of uniform length by
    cropping/padding as appropriate. Files are then saved.
    """
    processed_dir = os.path.join(data_path, 'processed')
    fi.make_path(processed_dir)

    audio_loc = os.path.join(data_path, "audio_loc")
    df=get_labeldf(audio_loc)
    df.to_csv(os.path.join(processed_dir,'heartchallenge_labels.csv'), columns=['ID', 'label'], index=False)
    if not labels_only:
        ds = df['label'].unique()
        force_max_len = max_length(df)

        # make paths for the spliced files
        for d in ds:
            path = os.path.join(processed_dir, d)
            fi.make_path(path)
        split_dir=os.path.join(data_path,"splits")
        fi.make_path(split_dir)
        with open(os.path.join(split_dir,"train.txt"),"w+") as train_splits, open(os.path.join(split_dir,"test.txt"),"w+") as test_splits:
            for idx, row in tqdm(df.iterrows(),total=df.shape[0]):
                filename = row['ID']
                start = row['start']
                end = row['end']
                label=row["label"]
                if force_max_len < end - start:
                    end = start + force_max_len
                save_name = filename + ".wav"
                try:
                    aud_loc = os.path.join(audio_loc,label,filename + '.wav')
                    data, samplingrate = lb.load(aud_loc)
                    train_splits.write(filename+"\n")
                except:
                    aud_loc = os.path.join(audio_loc,"Test",filename + '.wav')
                    data, samplingrate = lb.load(aud_loc)
                    test_splits.write(filename+"\n")
                sliced_data = slice_data(start=start, end=end, raw_data=data, sample_rate=samplingrate)

                # pad audio if < forced_max_len
                a_len = compute_len(samp_rate=samplingrate, time=force_max_len)
                padded_data = lb.util.pad_center(sliced_data, a_len)
                path = os.path.join(processed_dir, label, save_name)
                sf.write(file=path, data=padded_data, samplerate=samplingrate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../heartchallenge', help='data path')
    parser.add_argument("--labels_only", default=False, help="if True does not process and only creates label files.")
    args = parser.parse_args()
    process(args.data, args.labels_only)

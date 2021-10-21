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


def max_length(files_df):
    files_df['len_per_slice'] = files_df['end'].sub(files_df['start'], axis=0)
    return math.ceil(boxplot_stats(files_df['len_per_slice'])[0]['whishi'])


def process(data_path='../heart', labels_only=False):
    """Process the raw wavs to get each slice

    INPUT: A data dir where audio files and cycle info is stored in data_path/audio_txt_files
    OUTPUTS: data_path/processed with each recording split into the respiratory cycles
             disease_labels.csv: file with patient ids and their disease diagnoses mapped by Healthy, COPD, or Other
             symptoms_labels.csv: file with respiratory cycle ids and presence of crackles and wheezes

    The function parses through each of the txt files and filenames to identify start and end for each cycle. A good
    clipping value is then calculated. Each slice from the raw audios is processed so that they are of uniform length by
    cropping/padding as appropriate. Files are then saved.
    """

    if not labels_only:
        processed_dir = os.path.join(data_path, 'processed')
        fi.make_path(processed_dir)

        audio_loc = os.path.join(data_path, "audio_loc")
        df = pd.read_csv(os.path.join(data_path, "heart_labels.csv"), header=0, names=["ID", "label"])
        df["label"] = df["label"].apply(la.heart_recover_label)
        ds = df['label'].unique()
        for idx, file in enumerate(list(df["ID"])):
            df.loc[idx, "start"] = 0
            audio = sf.SoundFile(os.path.join(audio_loc, file + ".wav"))
            end = len(audio) / audio.samplerate
            df.loc[idx, "end"] = end
        force_max_len = max_length(df)

        # make paths for the spliced files
        for d in ds:
            path = os.path.join(processed_dir, d)
            fi.make_path(path)
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            filename = row['ID']
            start = row['start']
            end = row['end']
            label = row["label"]
            if force_max_len < end - start:
                end = start + force_max_len
            save_name = filename + ".wav"
            aud_loc = os.path.join(audio_loc, filename + '.wav')
            data, samplingrate = lb.load(aud_loc)

            sliced_data = slice_data(start=start, end=end, raw_data=data, sample_rate=samplingrate)

            # pad audio if < forced_max_len
            a_len = compute_len(samp_rate=samplingrate, time=force_max_len)
            padded_data = lb.util.pad_center(sliced_data, a_len)
            path = os.path.join(processed_dir, label, save_name)
            sf.write(file=path, data=padded_data, samplerate=samplingrate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../heart', help='data path')
    parser.add_argument("--labels_only", default=False, help="if True does not process and only creates label files.")
    args = parser.parse_args()
    process(args.data, args.labels_only)

import os
import pandas as pd
from tqdm import tqdm
import math
from matplotlib.cbook import boxplot_stats
import librosa as lb  # https://librosa.github.io/librosa/
import soundfile as sf  # https://pysoundfile.readthedocs.io/en/latest/
import file as fi
import split as sp
import labels as la
import argparse

# splices audio data to desired start, end timestamps
def slice_data(start, end, raw_data, sample_rate):
    max_ind = len(raw_data)
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]


def compute_len(samp_rate=22050, time=0, acquisition_mode=0):
    """Computes the supposed length of sliced data
        samp_size = sample size from the data
        samp_rate = sampling rate. by default since we're working on 24-bit files, we'll use 96kHz
        time = length of time for the audio file. by default we'll use the max we have which is 5.48
        acquisition_mode = either mono or stereo. 0 for mono, 1 for stereo
    """
    if acquisition_mode == 1:  # ac mode is single channel which means it's 'mono'
        comp_len = samp_rate * time
    else:  # stereo
        comp_len = samp_rate * time

    return comp_len


def get_cycledf(audio_text_loc):
    # identify all cycle information files
    files = [fi.split_path(s).split('.')[0] for s in fi.get_filenames(audio_text_loc, ["txt"])]
    files_ = []
    # create dataframe from information contained in wav file names
    for f in files:
        df = pd.read_csv(audio_text_loc + '/' + f + '.txt', sep='\t', names=['start', 'end', 'crackles', 'wheezes'])
        df['filename'] = f
        # get filename features
        f_features = fi.tokenize_file(f)
        df['pId'] = f_features[0]
        df['ac_mode'] = f_features[3]

        files_.append(df)

    return files_


def mergedf(cycle_info, diagnosis):
    # process the dataframes and merge so that we have start and end info for each slice
    files_df = pd.concat(cycle_info)
    files_df.reset_index()
    files_df['pId'] = files_df['pId'].astype('float64')
    files_df = pd.merge(files_df, diagnosis, on='pId')
    return files_df


def max_length(files_df):
    files_df['len_per_slice'] = files_df['end'].sub(files_df['start'], axis=0)
    return math.ceil(boxplot_stats(files_df['len_per_slice'])[0]['whishi'])


def process(data_path='../data'):
    """
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

    diagnosis = la.process_diag_labels(data_path)
    # find unique labels-->we will need this later
    ds = diagnosis['diagnosis'].unique()

    audio_text_loc = os.path.join(data_path, 'audio_and_txt_files')
    files_df = mergedf(get_cycledf(audio_text_loc), diagnosis)

    # determine a max length for clips
    force_max_len = max_length(files_df)

    # make paths for the spliced files
    for d in ds:
        path = os.path.join(processed_dir, d)
        fi.make_path(path)

    # for each original file we splice by timestamps, and save as well as constructing a label file for the symptoms
    i = 0  # iterator for file naming
    with open(os.path.join(processed_dir, 'symptoms_labels.csv'), "a") as out:
        out.write("cycle,crackles,wheezes\n")

        for idx, row in tqdm(files_df.iterrows(), total=files_df.shape[0]):
            filename = row['filename']
            start = row['start']
            end = row['end']
            diag = row['diagnosis']
            crackles = row['crackles']
            wheezes = row['wheezes']

            # check len and force to 6 sec if more than that
            if force_max_len < end - start:
                end = start + force_max_len

            # reset index for each original file
            if idx != 0:
                if files_df.iloc[idx - 1]['filename'] == filename:
                    i = i + 1
                else:
                    i = 0

            n_filename = filename + '_' + str(i) + '.wav'
            path = os.path.join(processed_dir, diag, n_filename)

            aud_loc = audio_text_loc + '/' + filename + '.wav'
            data, samplingrate = lb.load(aud_loc)
            sliced_data = slice_data(start=start, end=end, raw_data=data, sample_rate=samplingrate)

            # pad audio if < forced_max_len
            a_len = compute_len(samp_rate=samplingrate, time=force_max_len, acquisition_mode=row['ac_mode'] == 'sc')
            padded_data = lb.util.pad_center(sliced_data, a_len)

            sf.write(file=path, data=padded_data, samplerate=samplingrate)

            cycle = n_filename[:-4]
            out.write(cycle + "," + str(crackles) + "," + str(wheezes) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../data', help='data path')

    process(parser.parse_args().data)


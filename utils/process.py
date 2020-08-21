import os
import pandas as pd
from tqdm import tqdm
import math
from matplotlib.cbook import boxplot_stats
import librosa as lb  # https://librosa.github.io/librosa/
import soundfile as sf  # https://pysoundfile.readthedocs.io/en/latest/
import file as fi
import split as sp


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
        comp_len = (samp_rate * time)

    return comp_len


# processes raw data consisting of wavs and cycle timestamps
def process(data_path='../data'):
    processed_dir = os.path.join(data_path, 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    # loads the diagnosis csv, applies a mapping to get labels, and saves the labels
    diag_csv = os.path.join(data_path, 'patient_diagnosis.csv')
    diagnosis = pd.read_csv(diag_csv, names=['pId', 'diagnosis'])
    diagnosis['diagnosis'] = diagnosis['diagnosis'].apply(diag_map)
    diagnosis.to_csv(os.path.join(processed_dir, 'disease_labels.csv'), index=False, header=["pt_id", "label"])
    # find unique labels-->we will need this later
    ds = diagnosis['diagnosis'].unique()
    # identify all cycle information files
    audio_text_loc = os.path.join(data_path, 'audio_and_txt_files')
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
    # process the df
    files_df = pd.concat(files_)
    files_df.reset_index()
    files_df['pId'] = files_df['pId'].astype('float64')
    files_df = pd.merge(files_df, diagnosis, on='pId')
    # determine length info
    files_df['len_per_slice'] = files_df['end'].sub(files_df['start'], axis=0)
    force_max_len = math.ceil(boxplot_stats(files_df['len_per_slice'])[0]['whishi'])

    # make paths for the spliced files
    for d in ds:
        path = os.path.join(processed_dir, d)
        if not os.path.exists(path):
            os.makedirs(path)

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

            aud_loc = audio_text_loc + '/' + f + '.wav'
            # reset index for each original file
            if idx != 0:
                if files_df.iloc[idx - 1]['filename'] == filename:
                    i = i + 1
                else:
                    i = 0

            n_filename = filename + '_' + str(i) + '.wav'
            path = os.path.join(processed_dir, diag, n_filename)

            # print('processing ' + n_filename + '...')

            data, samplingrate = lb.load(aud_loc)
            sliced_data = slice_data(start=start, end=end, raw_data=data, sample_rate=samplingrate)

            # pad audio if < forced_max_len
            a_len = compute_len(samp_rate=samplingrate, time=force_max_len, acquisition_mode=row['ac_mode'] == 'sc')
            padded_data = lb.util.pad_center(sliced_data, a_len)

            sf.write(file=path, data=padded_data, samplerate=samplingrate)

            cycle = n_filename[:-4]
            out.write(cycle + "," + str(crackles) + "," + str(wheezes) + "\n")


# map function to generate labels from raw csv
def diag_map(diag):
    if diag == "Healthy":
        return diag
    if diag == "COPD":
        return diag
    else:
        return diag.replace(diag, "Other")


if __name__ == "__main__":
    # splice files
    process()
    if not os.path.exists('../data/splits'):
        os.makedirs('../data/splits')
    # create splits based 30 tests of 10 patients each, OPTIONAL: provide a seed number for reproducibility.
    sp.split('../data/processed', '../data/splits')

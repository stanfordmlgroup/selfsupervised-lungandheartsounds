import os
from glob import glob


def tokenize_file(filename):
    return filename.split('_')


def get_pt_id(wav_path):
    if os.name == "nt":
        return tokenize_file(wav_path.split("\\")[-1])[0]
    else:
        return tokenize_file(wav_path.split("/")[-1])[0]


def get_filenames(search_dir, type_list):
    file_list = []
    for type in type_list:
        file_list.extend(glob(os.path.join(search_dir, "*.", type)))
    return file_list

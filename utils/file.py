import os
from glob import glob


# split by _
def tokenize_file(filename):
    return filename.split('_')


# gives last term in path split by os specific delineater
def split_path(path):
    if os.name == "nt":
        return path.split("\\")[-1]
    else:
        return path.split("/")[-1]


# takes raw path and gives patient id
def get_pt_id(path):
    return tokenize_file(split_path(path))[0]


# given a search directory and list of types (eg .txt would be input as ['txt']), outputs a list of all matching files
def get_filenames(search_dir, type_list):
    file_list = []
    for type in type_list:
        file_list.extend(glob(os.path.join(search_dir, "*." + type), recursive=True))
    return file_list

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
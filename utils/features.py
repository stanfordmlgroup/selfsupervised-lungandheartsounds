import os
import torch
import numpy as np
import soundfile as sf
import librosa
import argparse

def get_vggish_embedding(filename=None):
    # takes in a raw wav file and converts it to a x*128 embedding, where x is the number of seconds in the clip.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
    vggish.eval()
    return vggish.forward(filename)


def preprocess(file_name=None):
    # preprocessing of raw file for the below features
    x, sample_rate = sf.read(file_name, dtype='float32')

    if x.ndim > 1: x = x[:, 0]
    x = x.T
    return x


def stft(file_name=None):

    x = preprocess(file_name)
    # short term fourier transform
    stft = np.abs(librosa.stft(x))
    return stft


def mfccs(file_name=None):
    if file_name:
        _, sample_rate = sf.read(file_name, dtype='float32')
    x = preprocess(file_name)
    # mfcc (mel-frequency cepstrum)
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs


def chroma(file_name=None):
    if file_name:
        _, sample_rate = sf.read(file_name, dtype='float32')
    stft_in = stft(file_name)
    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft_in, sr=sample_rate).T, axis=0)
    return chroma


def mel(file_name=None):
    if file_name:
        _, sample_rate = sf.read(file_name, dtype='float32')
    x = preprocess(file_name)
    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(x, sr=sample_rate).T, axis=0)
    return mel


def contrast(file_name=None):
    if file_name:
        _, sample_rate = sf.read(file_name, dtype='float32')
    stft_in= stft(file_name)
    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft_in, sr=sample_rate).T, axis=0)
    return contrast


def tonnetz(file_name=None):
    if file_name:
        _, sample_rate = sf.read(file_name, dtype='float32')
    x = preprocess(file_name)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sample_rate).T, axis=0)
    return tonnetz

def example(args):
    print("Feature Extraction Examples:")
    example_file =args.file
    print("Filename: 101_1b1_Al_sc_Meditron.wav")
    print("Size of VGGish embedding: " + str(get_vggish_embedding(example_file).shape))
    print("Size of STFT: " + str(stft(example_file).shape))
    print("Size of MFCCS: " + str(mfccs(example_file).shape))
    print("Size of Chroma: " + str(chroma(example_file).shape))
    print("Size of Mel: " + str(mel(example_file).shape))
    print("Size of Contrast: " + str(contrast(example_file).shape))
    print("Size of Tonnetz: " + str(tonnetz(example_file).shape))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=os.path.join(os.getcwd(),'../data/audio_and_txt_files/101_1b1_Al_sc_Meditron.wav'), help='example filename')
    example(parser.parse_args())


import os
import torch
import numpy as np
import soundfile as sf
import librosa
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # prevents weird error with matplotlib
vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
    
def get_vggish_embedding(filename=None):
    """takes in a raw wav file and converts it to a x*128 embedding, where x is the number of seconds in the clip."""
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vggish.to(device)
    vggish.eval()
    x = vggish.forward(filename)
    return x.detach().numpy() 

def preprocess(file_name=None):
    """preprocessing of raw file for the below features"""
    x, sample_rate = sf.read(file_name, dtype='float32')

    if x.ndim > 1: x = x[:, 0]
    x = x.T
    return x


def stft(file_name=None):
    """short term fourier transform"""
    x = preprocess(file_name)
    stft = np.abs(librosa.stft(x))
    return stft


def mfccs(file_name=None):
    """mfcc (mel-frequency cepstrum)"""
    if file_name:
        _, sample_rate = sf.read(file_name, dtype='float32')
    x = preprocess(file_name)
    mfccs = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T
    return mfccs


def chroma(file_name=None):
    """chroma"""
    if file_name:
        _, sample_rate = sf.read(file_name, dtype='float32')
    stft_in = stft(file_name)
    chroma = librosa.feature.chroma_stft(S=stft_in, sr=sample_rate).T
    return chroma


def mel(file_name=None):
    """Mel spectrogram"""
    if file_name:
        _, sample_rate = sf.read(file_name, dtype='float32')
    x = preprocess(file_name)
    mel = librosa.feature.melspectrogram(x, sr=sample_rate).T
    return mel


def contrast(file_name=None):
    """spectral contrast"""
    if file_name:
        _, sample_rate = sf.read(file_name, dtype='float32')
    stft_in = stft(file_name)
    contrast = librosa.feature.spectral_contrast(S=stft_in, sr=sample_rate).T
    return contrast


def tonnetz(file_name=None):
    """tonnetz"""
    if file_name:
        _, sample_rate = sf.read(file_name, dtype='float32')
    x = preprocess(file_name)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sample_rate).T
    return tonnetz


def example(example_file):
    """function for showing example feature shapes"""
    print("Feature Extraction Examples:")
    print("Filename: 101_1b1_Al_sc_Meditron.wav")
    print("Size of VGGish embedding: " + str(get_vggish_embedding(example_file).shape))
    print("Size of STFT: " + str(stft(example_file).shape))
    print("Size of MFCCS: " + str(mfccs(example_file).shape))
    print("Size of Chroma: " + str(chroma(example_file).shape))
    print("Size of Mel: " + str(mel(example_file).shape))
    print("Size of Contrast: " + str(contrast(example_file).shape))
    print("Size of Tonnetz: " + str(tonnetz(example_file).shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',
                        default=os.path.join(os.getcwd(), '../data/audio_and_txt_files/101_1b1_Al_sc_Meditron.wav'),
                        help='example filename')
    example(parser.parse_args().file)

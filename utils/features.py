import os
import torch
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf


def get_vggish_embedding(filename):
    # takes in a raw wav file and converts it to a x*128 embedding, where x is the number of seconds in the clip.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
    vggish.eval()
    return vggish.forward(filename)


def preprocess(file_name=None):
    # preprocessing of raw file for the below features
    if file_name:
        # print('Extracting', file_name)
        X, sample_rate = sf.read(file_name, dtype='float32')
    else:
        device_info = sd.query_devices(None, 'input')
        sample_rate = int(device_info['default_samplerate'])
        q = queue.Queue()

        def callback(i, f, t, s):
            q.put(i.copy())

        data = []
        with sd.InputStream(samplerate=sample_rate, callback=callback):
            while True:
                if len(data) < 100000:
                    data.extend(q.get())
                else:
                    break
        X = np.array(data)

    if X.ndim > 1: X = X[:, 0]
    X = X.T
    return X


def stft(file_name=None):

    X = preprocess(file_name)
    # short term fourier transform
    stft = np.abs(librosa.stft(X))
    return stft


def mfccs(file_name=None):
    if file_name:
        _, sample_rate = sf.read(file_name, dtype='float32')
    X = preprocess(file_name)
    # mfcc (mel-frequency cepstrum)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
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
    X = preprocess(file_name)
    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
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
    X = preprocess(file_name)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return tonnetz

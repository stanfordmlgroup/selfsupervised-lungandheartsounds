import os
import torch
import numpy as np
import soundfile as sf
import librosa
import argparse
import scipy.signal

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
    return x, sample_rate


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


class Mel(object):
    def __init__(self, raw_aug=None):
        self.raw_aug = raw_aug

    def __call__(self, X, sample_rate=22050.0):
        if self.raw_aug is not None:
            X = self.raw_aug(X, sample_rate)
        #mel = librosa.feature.melspectrogram(X, sr=sample_rate)
        mel_log=sample2MelSpectrum(X, sample_rate=sample_rate)
        return mel_log


# vtlp_params = (alpha, f_high)
def sample2MelSpectrum(cycle_info, sample_rate, n_filters=50):
    n_rows = 175  # 7500 cutoff
    n_window = 512  # ~25 ms window
    (f, t, Sxx) = scipy.signal.spectrogram(cycle_info, fs=sample_rate, nfft=n_window, nperseg=n_window)
    Sxx = Sxx[:n_rows, :].astype(np.float32)  # sift out coefficients above 7500hz, Sxx has 196 columns
    mel_log = FFT2MelSpectrogram(f[:n_rows], Sxx, n_filters)[1]
    mel_min = np.min(mel_log)
    mel_max = np.max(mel_log)
    diff = mel_max - mel_min
    norm_mel_log = (mel_log - mel_min) / diff if (diff > 0) else np.zeros(shape=(n_filters, Sxx.shape[1]))
    if (diff == 0):
        print('Error: sample data is completely empty')
    return np.reshape(norm_mel_log, (n_filters, Sxx.shape[1])).astype(np.float32)


def Freq2Mel(freq):
    return 1125 * np.log(1 + freq / 700)


def Mel2Freq(mel):
    exponents = mel / 1125
    return 700 * (np.exp(exponents) - 1)

# mel_space_freq: the mel frequencies (HZ) of the filter banks, in addition to the two maximum and minimum frequency values
# fft_bin_frequencies: the bin freqencies of the FFT output
# Generates a 2d numpy array, with each row containing each filter bank
def GenerateMelFilterBanks(mel_space_freq, fft_bin_frequencies):
    n_filters = len(mel_space_freq) - 2
    coeff = []
    # Triangular filter windows
    # ripped from http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
    for mel_index in range(n_filters):
        m = int(mel_index + 1)
        filter_bank = []
        for f in fft_bin_frequencies:
            if (f < mel_space_freq[m - 1]):
                hm = 0
            elif (f < mel_space_freq[m]):
                hm = (f - mel_space_freq[m - 1]) / (mel_space_freq[m] - mel_space_freq[m - 1])
            elif (f < mel_space_freq[m + 1]):
                hm = (mel_space_freq[m + 1] - f) / (mel_space_freq[m + 1] - mel_space_freq[m])
            else:
                hm = 0
            filter_bank.append(hm)
        coeff.append(filter_bank)
    return np.array(coeff, dtype=np.float32)


# Transform spectrogram into mel spectrogram -> (frequencies, spectrum)
# vtlp_params = (alpha, f_high), vtlp will not be applied if set to None
def FFT2MelSpectrogram(f, Sxx, n_filterbanks):
    (max_mel, min_mel) = (Freq2Mel(max(f)), Freq2Mel(min(f)))
    mel_bins = np.linspace(min_mel, max_mel, num=(n_filterbanks + 2))
    # Convert mel_bins to corresponding frequencies in hz
    mel_freq = Mel2Freq(mel_bins)

    filter_banks = GenerateMelFilterBanks(mel_freq, f)
    mel_spectrum = np.matmul(filter_banks, Sxx)
    return (mel_freq[1:-1], np.log10(mel_spectrum + float(10e-12)))

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
    print("Size of Mel: " + str(Mel(example_file).shape))
    print("Size of Contrast: " + str(contrast(example_file).shape))
    print("Size of Tonnetz: " + str(tonnetz(example_file).shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',
                        default=os.path.join(os.getcwd(), '../data/audio_and_txt_files/101_1b1_Al_sc_Meditron.wav'),
                        help='example filename')
    example(parser.parse_args().file)

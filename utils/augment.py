import random
from audiomentations import *
import h5py as h5
import librosa
import librosa.display
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
from spec_augment import spec_augment

def visualization_spectrogram(mel_spectrogram, title):
    """visualizing result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
# Performs randomized masking
class BaseInputMask(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, pixel_array):
        if random.random() < self.p:
            self.num_rows, self.num_cols = pixel_array.shape
            start_row = random.randint(0, self.num_rows - 1)
            end_row = random.randint(start_row + 1, self.num_rows)
            start_col = random.randint(0, self.num_cols - 1)
            end_col = random.randint(start_col + 1, self.num_cols)
            pixel_array[start_row:end_row, start_col:end_col] = 0

        return pixel_array


# Splits into two slices with replacement (no masking)
class Split(object):
    def __init__(self, spec_func):
        self.spec_func = spec_func

    def __call__(self, X):
        pixel_array = self.spec_func(X)
        num_rows, num_cols = pixel_array.shape
        slice_col_width = num_rows // 2
        start_col_one = random.randint(0, slice_col_width)
        end_col_one = start_col_one + slice_col_width
        return pixel_array[start_col_one:end_col_one, :]



#Performs augment on raw audio data as specified in https://arxiv.org/ftp/arxiv/papers/2007/2007.07966.pdf
class RawAugment(object):
    def __init__(self):
        pass

    def __call__(self, audio_data, sample_rate):
        augment = Compose([
            Gain(p=.5),
            PitchShift(p=.5),
            Shift(rollover=False, p=.5),
            TimeStretch(p=.5),
            AddGaussianNoise(p=1.0),
        ])
        augment_sample = augment(samples=audio_data, sample_rate=sample_rate)
        return augment_sample

#Performs spectral augment as specified in https://arxiv.org/ftp/arxiv/papers/2007/2007.07966.pdf
#Need to fix spectral import issue
class SpectralAugment(object):
    def __init__(self, spec_func):
        self.spec_func = spec_func

    def __call__(self, X):
        spectrogram = self.spec_func(X)
        #augment = Compose([
           # BaseInputMask(p=.5),
            # SpecChannelShuffle(),
            # spectrogram_shuffle(spectrogram),
            # Shift(min_fraction=-.5, max_fraction=.5, p=.5),
            # SpecFrequencyMask(),
            # spectrogram_freq_mask(spectrogram),
            # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)
        #])
        spectrogram=torch.Tensor(spectrogram)
        spectrogram=spectrogram.view(1,spectrogram.shape[0],spectrogram.shape[1])
        augment_sample = spec_augment(mel_spectrogram=spectrogram, time_warping_para=60)
        augment_sample=augment_sample.squeeze().cpu().numpy()
        return augment_sample



def spectrogram_freq_mask(spectrogram):
    spec_mask = SpecFrequencyMask()
    spec_mask.randomize_parameters(spectrogram)
    return spec_mask.apply(spectrogram)


def spectrogram_shuffle(spectrogram):
    spec_shuffle = SpecChannelShuffle(spectrogram)
    spec_shuffle.randomize_parameters(spectrogram)
    return spec_shuffle.apply(spectrogram)


# def spec_augment(spectrogram):
#     augment = Compose([
#         SpecChannelShuffle(),
#         # spectrogram_shuffle(spectrogram),
#         Shift(min_fraction=-.5, max_fraction=.5, p=.5),
#         SpecFrequencyMask(),
#         # spectrogram_freq_mask(spectrogram),
#         AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)
#     ])
#     augment_sample = augment(samples=spectrogram)
#     return augment_sample


if __name__ == '__main__':
    from skimage import metrics
    import sys
    import os
    sys.path.append('../models')
    from data import get_transform
    import file as fi
    import soundfile as sf

    file = '../heart/processed/heart_1.0.h5'
    file = h5.File(file, 'r')
    data = file['test']
    num_samples = data.shape[0]
    __transforms__ = ["spec", "raw", "raw+spec"]
    mel = get_transform()
    path=fi.make_path(os.path.join(os.getcwd(),'../utils/output'))
    sf.write(os.path.join(path, 'original.wav'), data[0],samplerate=22050)
    for transform in __transforms__:
        augment = get_transform(transform)
        avg_psnr=0
        avg_ssim=0
        for i in range(0,1):
            raw=data[i]
            mel_ = mel(raw)
            augment_ = augment(raw)
            avg_psnr += metrics.peak_signal_noise_ratio(mel_, augment_, data_range=mel_.max()-mel_.min()) / num_samples
            avg_psnr += metrics.structural_similarity(mel_,augment_) / num_samples
        print("{}, Average PSNR: {:.2f}, Average SSIM: {:.3f}\n".format(transform,avg_psnr,avg_ssim))
        sf.write(os.path.join(path, transform+'.wav'), librosa.feature.inverse.mel_to_audio(augment_.T),samplerate=22050)

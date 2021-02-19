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
import time


def visualization_spectrogram(mel_spectrogram, title):
    """visualizing result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :], ref=np.max), y_axis='mel', fmax=8000,
                             x_axis='time')
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


# Performs augment on raw audio data as specified in https://arxiv.org/ftp/arxiv/papers/2007/2007.07966.pdf
class RawAugment(object):
    def __call__(self, audio_data, sample_rate=22050):
        augment = Compose([
            Gain(p=1),
            PitchShift(p=1),
            Shift(min_fraction=-.25, max_fraction=.25, rollover=False, p=1),
            TimeStretch(min_rate=.9, max_rate=1.1, p=1),
            #AddGaussianNoise(min_amplitude=.00001, max_amplitude=.00005, p=1),
        ])
        augment_sample = augment(samples=audio_data, sample_rate=sample_rate)
        return augment_sample


# Performs spectral augment as specified in https://arxiv.org/ftp/arxiv/papers/2007/2007.07966.pdf
# Need to fix spectral import issue
class SpectralAugment(object):
    def __init__(self, spec_func):
        self.spec_func = spec_func

    def __call__(self, X):
        spectrogram = self.spec_func(X)
        spectrogram = torch.Tensor(spectrogram)
        spectrogram = spectrogram.view(1, spectrogram.shape[0], spectrogram.shape[1])
        augment_sample = spec_augment(mel_spectrogram=spectrogram,
                                      frequency_mask_num=2, time_mask_num=5)
        augment_sample = augment_sample.squeeze().cpu().numpy()
        return augment_sample


class FreqAugment(object):
    def __init__(self, spec_func):
        self.spec_func = spec_func

    def __call__(self, X):
        spectrogram = self.spec_func(X)
        spectrogram = torch.Tensor(spectrogram)
        spectrogram = spectrogram.view(1, spectrogram.shape[0], spectrogram.shape[1])
        augment_sample = spec_augment(mel_spectrogram=spectrogram, frequency_mask_num=2,
                                      time_mask_num=0)
        augment_sample = augment_sample.squeeze().cpu().numpy()
        return augment_sample


class TimeAugment(object):
    def __init__(self, spec_func):
        self.spec_func = spec_func

    def __call__(self, X):
        spectrogram = self.spec_func(X)
        spectrogram = torch.Tensor(spectrogram)
        spectrogram = spectrogram.view(1, spectrogram.shape[0], spectrogram.shape[1])
        augment_sample = spec_augment(mel_spectrogram=spectrogram, frequency_mask_num=0,
                                      time_mask_num=5)
        augment_sample = augment_sample.squeeze().cpu().numpy()
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
    import matplotlib.pyplot as plt
    import noise

    def plot(orig, augment, path=None):
        plt.figure(figsize=(20, 10))

        plt.subplot(2, 2, 1)
        plt.title('original mel')
        #print(orig.min(), orig.mean(), orig.max(), orig.std())
        plt.imshow(orig, aspect='auto')

        plt.subplot(2, 2, 2)
        plt.title('augmented mel')
        #print(augment.min(), augment.mean(), augment.max(), orig.std(), "\n")
        plt.imshow(augment, aspect='auto')

        plt.subplot(2, 2, 3)
        plt.title('original mel hist')
        plt.hist(np.ma.masked_equal(orig.reshape(-1),0), bins=25)

        plt.subplot(2, 2, 4)
        plt.title('augmented mel hist')
        plt.hist(np.ma.masked_equal(augment.reshape(-1),0), bins=25)

        if path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()


    def post_process(sound):
        #less_sound = noise.reduce_noise_centroid_mb(sound, 22050)
        less_sound = sound
        db = librosa.core.amplitude_to_db(less_sound).mean()
        #print(db)
        #gain = Gain(min_gain_in_db=-30 - db, max_gain_in_db=-30 - db, p=1)
        #output = gain(less_sound, sample_rate=22050)
        #print(librosa.core.amplitude_to_db(output).mean(), "\n")
        return sound


    # file = '../heart/processed/heart_1.0.h5'
    # file = '../data/processed/disease_1.0.h5'
    # #file = '../heartchallenge/processed/heartchallenge_1.0.h5'
    #
    # file = h5.File(file, 'r')
    # data = file['test']
    # num_samples = 0
    # __transforms__ = ["spec"]
    # mel = get_transform()
    # path = fi.make_path(os.path.join(os.getcwd(), '../utils/output'))
    # sf.write(os.path.join(path, 'original.wav'),
    #          post_process(data[27]), samplerate=22050)
    # sample = [27]
    # for transform in __transforms__:
    #     start = time.time()
    #     augment = get_transform(transform)
    #     avg_psnr = 0
    #     avg_ssim = 0
    #     for i in sample:
    #         raw = data[i]
    #         mel_ = mel(raw)
    #         augment_ = augment(raw)
    #         avg_psnr += metrics.peak_signal_noise_ratio(mel_, augment_, data_range=mel_.max() - mel_.min())
    #         avg_ssim += metrics.structural_similarity(mel_, augment_)
    #         num_samples += 1
    #
    #         output = post_process(librosa.feature.inverse.mel_to_audio(augment_, n_fft=512))
    #     sf.write(os.path.join(path, transform + '.wav'), output,
    #              samplerate=22050)
    #     print("{}, Compute Time: {:.3f} s, Average PSNR: {:.2f}, Average SSIM: {:.3f}\n".format(transform,
    #                                                                                             time.time() - start,
    #                                                                                                 avg_psnr / num_samples,
    #                                                                                             avg_ssim / num_samples))
    #     plot(mel_, augment_)

    __task__ = ['disease', 'heart', 'heartchallenge']
    path = fi.make_path(os.path.join(os.getcwd(), '../utils/output'))

    mel = get_transform()
    rawaug = RawAugment()
    __transforms__ = ["spec", "raw", "time", "freq"]

    repitions=3
    for task in __task__:
        if task != 'disease':
            file = '../{}/processed/{}_1.0.h5'.format(task, task)
        else:
            file = '../data/processed/disease_1.0.h5'
        file = h5.File(file, 'r')
        data = file['test']
        task_path = fi.make_path(os.path.join(path, task))
        sample = random.sample(range(data.shape[0]), 10)
        for i in sample:
            audio = data[i]
            samp_path = fi.make_path(os.path.join(task_path, str(i)))
            sf.write(os.path.join(samp_path, 'original.wav'), post_process(audio), samplerate=22050)
            mel_ = mel(audio)
            for rep in range(repitions):
                for transform in __transforms__:
                    augment = get_transform(transform)
                    if transform == 'raw':
                        sf.write(os.path.join(samp_path, 'raw_{}.wav'.format(rep)), rawaug(audio), samplerate=22050)
                    augment_ = augment(audio)
                    plot(mel_, augment_, os.path.join(samp_path, '{}_report_{}.png'.format(transform,rep)))
import random
# from keras.preprocessing.image import ImageDataGenerator,
# array_to_img, img_to_array, load_img
from audiomentations import *
from torchvision import transforms

# from audiomentations import SpecChannelShuffle
# from audiomentations import spectrogram_transforms.py
# from audiomentations import Compose, Gain, PitchShift, TimeStretch, Shift
# from audiomentations.core.transforms_interface import BaseSpectrogramTransform

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
    def __init__(self):
        pass

    def __call__(self, pixel_array):
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
    def __init__(self):
        pass

    def __call__(self, spectrogram):
        augment = Compose([
            BaseInputMask(p=.5),
            # SpecChannelShuffle(),
            # spectrogram_shuffle(spectrogram),
            # Shift(min_fraction=-.5, max_fraction=.5, p=.5),
            # SpecFrequencyMask(),
            # spectrogram_freq_mask(spectrogram),
            # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)
        ])
        augment_sample = augment(samples=spectrogram)
        return augment_sample



def spectrogram_freq_mask(spectrogram):
    spec_mask = SpecFrequencyMask()
    spec_mask.randomize_parameters(spectrogram)
    return spec_mask.apply(spectrogram)


def spectrogram_shuffle(spectrogram):
    spec_shuffle = SpecChannelShuffle(spectrogram)
    spec_shuffle.randomize_parameters(spectrogram)
    return spec_shuffle.apply(spectrogram)


def spec_augment(spectrogram):
    augment = Compose([
        SpecChannelShuffle(),
        # spectrogram_shuffle(spectrogram),
        Shift(min_fraction=-.5, max_fraction=.5, p=.5),
        SpecFrequencyMask(),
        # spectrogram_freq_mask(spectrogram),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)
    ])
    augment_sample = augment(samples=spectrogram)
    return augment_sample

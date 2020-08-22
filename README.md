# aihc-sum20-lung-sounds

## Utility scripts & methods w/ relevant usage examples

### process.py
Provides functionality to take raw audio and text files and generate a processed dir (_path_/processed) with audio files spliced by respiratory cycle. Also generates disease_labels.csv (mapped disease diagnoses) and symptoms_labels.csv (presence of crackles/wheezes for each respiratory file)  

#### Running script from terminal
```
$ python process.py [--data "data directory"]
```
DEAFULTS:

data: /data/

### split.py
Creates train test splits. Each patient id is assigned to exactly  

#### Running script from terminal
```
$ python split.py [--data "data directory"] [--splits "desired location of splits"] [--seed "seed for random sampling"] [--distribution "dict of classes and desired number of test files from each class"]
```
DEFAULTS:

data: /data/ 

splits: /data/splits

seed: 252

distribution: {'COPD': 10, 'Healthy': 10, 'URTI': 3, 'Bronchiectasis': 2, 'Bronchiolitis': 2,
                             'Pneumonia': 2, 'LRTI': 1, 'Asthma': 0}
<hr>

### labels.py
Functions to handle label processing from raw and label retrieval
<hr>

### features.py
Methods to extract numerical features from raw audio files.

Currently implemented:

1. VGGish Embedding 
2. STFT
3. MFCCS
4. Chroma
5. MEL
6. Contrast
7. Tonnetz

#### Running script from terminal
```
$ python features.py [--file "example filename"]
```
Runs each of the feature methods on an example file (optionally can be provided by user). Returns shapes of each feature extraction.

#### VGGish
```
get_vggish_embedding(filename): returns a x*128 embedding, where x is the number of seconds in the clip.
```

A `torch`-compatible port of [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset) <sup>[1]</sup>, 
a feature embedding frontend for audio classification models. The weights are ported directly from the tensorflow model, so embeddings created using `torchvggish` will be identical.

[1]  S. Hershey et al., ‘CNN Architectures for Large-Scale Audio Classification’,\
    in International Conference on Acoustics, Speech and Signal Processing (ICASSP),2017\
    Available: https://arxiv.org/abs/1609.09430, https://ai.google/research/pubs/pub45611
#### STFT
```
stft(filename): returns a 1025*x embedding, where x is the number of frames in the clip
```
#### MFCCS
```
mfccs(filename): returns a feature of length 40
```
#### Chroma
```
chroma(filename): returns a feature of length 12
```
#### Mel
```
mel(filename): returns a feature of length 128
```
#### Contrast
```
contrast(filename): returns a feature of length 7
```
#### Tonnetz
```
tonnetz(filename): returns a feature of length 6
```
<hr>

### file.py
General functions for dealing with file reading and manipulation



# aihc-sum20-lung-sounds
## Experiment Reproduction
All scripts for reporoducing experimental results can be found in the scripts folder. The test script enables quickly running testing.

models/intervals.py has the functionality to analyze bootstrapped confidence intervals from trained models.
## Datasets
- Heart (Physionet)
    - Normal v Abnormal (2575/665)
        - Abnormal: "The patients suffer from a variety of illnesses (which we do not provide on a case-by-case basis), but typically they are heart valve defects and coronary artery disease patients. Heart valve defects include mitral valve prolapse, mitral regurgitation, aortic stenosis and valvular surgery. All the recordings from the patients were generally labeled as abnormal. We do not provide more specific classification for these abnormal recordings."
    - Extest (648 samples {324 of each}), Train (2592 samples) {Internal Test/Fine-Tune (648 samples {324 of each})}
    - [dataset](https://physionet.org/content/challenge-2016/1.0.0/)
    - Mel mean: .7196, std: 32.07 
- Heart (Peter J Bentley)
    - Normal v Abnormal (493/271)
        - Abnormal: Artifact: 56, Extrasound: 27, Extrastole: 66, Murmur: 178 (Artifacts dropped)
    - Extest (152 samples {76 of each}), Internal Test/Fine-Tune (612 samples)
    - [dataset](http://www.peterjbentley.com/heartchallenge/)
    - Mel mean: .9364, std: 33.991  
- Lung sounds (disease)
    - Normal v Abnormal (26/100)
        - Abnormal: COPD: 64, Healthy: 26, URTI: 14, Bronchiectasis: 7, Bronchiolitis: 6, Pneumonia: 6, LRTI: 2, Asthma: 1
    - Extest (30 samples {15 of each}), Internal Test/Fine-Tune (96 samples) {Internal Test/Fine-Tune (30 samples {15 of each})}
    - [dataset](https://www.kaggle.com/vbookshelf/respiratory-sound-database)
    - Mel mean: 3.273, std: 100.439 
- Lung sounds (crackles)
    - Normal v Abnormal (4528/2370)
    - Extest (30 samples {15 of each}), Internal Test/Fine-Tune (96 samples)
    - *Splits by patient not by sample*
    - [dataset](https://www.kaggle.com/vbookshelf/respiratory-sound-database)
- Lung sounds (wheezes)
    - Normal v Abnormal (5506/1392)
    - Extest (30 samples {15 of each}), Internal Test/Fine-Tune (96 samples)
    - *Splits by patient not by sample*
    - [dataset](https://www.kaggle.com/vbookshelf/respiratory-sound-database)
## Utility scripts & methods w/ relevant usage examples
### process.py
Provides functionality to take raw audio and text files and generate a processed dir (_path_/processed) with audio files spliced by respiratory cycle. Also generates disease_labels.csv (mapped disease diagnoses) and symptoms_labels.csv (presence of crackles/wheezes for each respiratory file)  
#### Running script from terminal
```
$ python process.py [--data "data directory"] [--labels_only "if True, doesn't process only makes labels"] 
```
DEFAULTS:

data: /data/
labels_only: False
### split.py
Creates train test splits. Each patient id is assigned to exactly one of train/test.  

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
mfccs(filename): returns a feature of length x*40, where x is the number of frames in the clip
```
#### Chroma
```
chroma(filename): returns a feature of length x*12, where x is the number of frames in the clip
```
#### Mel
```
mel(filename): returns a feature of length x*128, where x is the number of frames in the clip
```
#### Contrast
```
contrast(filename): returns a feature of length x*7, where x is the number of frames in the clip
```
#### Tonnetz
```
tonnetz(filename): returns a feature of length x*6, where x is the number of frames in the clip
```
<hr>

### file.py
General functions for dealing with file reading and manipulation


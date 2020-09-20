import os
from glob import glob

ids=glob(os.path.join(os.getcwd(),"*\\"))
for id in ids:
    wavs=glob(id+'*')
    for idx, wav in enumerate(wavs):
        os.rename(wav,id+str(idx)+".wav")
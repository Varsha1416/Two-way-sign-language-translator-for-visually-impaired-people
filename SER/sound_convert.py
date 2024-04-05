# Importing the basic libraries that we need

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile as wav
import numpy as np
from numpy.lib import stride_tricks
import librosa
import librosa.display
import os
CATEGORIES=[]
for class_name in os.listdir("dataset"):
    CATEGORIES.append(class_name)
DATADIR = "dataset"
DATADIR1 = "data"

def read_audio(conf, pathname, trim_long_data):
    print(pathname)
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples: # long enough
        if trim_long_data:
            y = y[0:0+conf.samples]
    else: # pad blank
        padding = conf.samples - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y

# Thanks to the librosa library, generating the mel-spectogram from the audio file

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

# Adding both previous function together

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    return mels

# A set of settings that you can adapt to fit your audio files (frequency, average duration, number of Fourier transforms...)

class conf:
    # Preprocessing settings
    sampling_rate = 44100
    duration = 1
    hop_length = 347*duration # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration
def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Scale to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


# To generate the image dataset, we first need to remove the .wav extension and add the .jpg

def rename_file(img_name):
    print(img_name)
    #img_name = img_name.split("/")[2]
    img_name = img_name[:-4]
    img_name += ".jpg"
    return img_name


import gc
lp=0
import os
for category in CATEGORIES :
    path = os.path.join(DATADIR, category)
    path=path+"/"
    path1 = os.path.join(DATADIR1, category)
    path1=path1+"/"
    for i, fn in enumerate(os.listdir(path)):
        print(i)
        print(fn)
        pathi = path + fn
        filename = rename_file(pathi)
        x = read_as_melspectrogram(conf, pathi, trim_long_data=False, debug_display=True)
        #x_color = mono_to_color(x)
        lp=lp+1
        filename=str(lp)
        plt.imshow(x, interpolation='nearest')
        for ju in range(0,2):
            plt.savefig(path1 + str(ju)+filename)
        #plt.show()
        
        plt.close()
        del x
        gc.collect()
    

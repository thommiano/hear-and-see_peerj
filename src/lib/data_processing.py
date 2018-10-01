import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa
import librosa.display

import numpy as np
import os
import time

import torch
from torch.autograd import Variable

import sys, inspect
sys.path.append("/audio") # Current docker images installs this here
import torchaudio


# There's currently a bug with torchaudio.save
# See: https://github.com/pytorch/audio/issues/14
def load_sound_tensor(filepath):
    '''Load .wav file returning a torch FloatTensor and the sample rate.'''
    sound_tensor, sr = torchaudio.load(filepath)
    return sound_tensor, sr


# There's currently a bug with torchaudio.save
# See: https://github.com/pytorch/audio/issues/14
def pad_sound_tensor(sound_tensor,sr,threshold=4.0,mode='edge'):
    '''Make sound tensor duration equal to threshold (in seconds) by adding padding when necessary.'''
    sound_np = sound_tensor.numpy()[:,0]
    duration = librosa.core.get_duration(sound_np, sr=sr)
    if duration < threshold:
        padding = int( ((threshold-duration) * sr) / 2)
        padded_sound = np.pad(sound_np,(padding,),mode)
        padded_sound_tensor = torchaudio.transforms.torch.FloatTensor(padded_sound).view(-1,1)
        return padded_sound_tensor
    else:
        return sound_tensor


# Torchaudio has its own spectrogram class, look in to that
def create_tensor_spectrogram(sound_tensor,sr):
    '''Convert sound tensor into mel-spectrogram.'''
    sound_np = sound_tensor.numpy()[:,0]
    spectrogram = librosa.feature.melspectrogram(y=sound_np, sr=sr)
    return spectrogram


def pad_sound_array(sound_array,sr,threshold=4.0,mode='edge'):
    '''Make sound array duration equal to threshold (in seconds) by adding padding when necessary.'''
    duration = librosa.core.get_duration(sound_array, sr=sr)
    if duration < threshold:
        padding = int( ((threshold-duration) * sr) / 2)
        sound_array = np.pad(sound_array,(padding,),mode)
    return sound_array


def save_spectrogram(spectrogram,write_path):
    '''Writes spectrogram to disk.'''
    plt.ioff()
    librosa.display.specshow(librosa.power_to_db(spectrogram,ref=np.max))
    plt.axis('off') # Removes black border
    plt.tight_layout()
    plt.savefig(write_path,bbox_inches='tight',pad_inches=-0.05,transparency=True, format='jpg')
    plt.clf() # clear figure to control memory problems
    plt.close()


def roll_spectrogram(spectrogram,roll_proportion=.5):
    '''Roll spectrogram by distance calculated from roll_proportion.'''
    roll_distance = int(round(spectrogram.shape[1]*roll_proportion))
    rolled_spectrogram = np.roll(spectrogram,roll_distance)
    return rolled_spectrogram


# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''
from __future__ import print_function

import copy

import librosa

from gan_hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
import ipdb

def get_spectrograms(sound_file): 
    '''Extracts melspectrogram and log magnitude from given `sound_file`.
    Args:
      sound_file: A string. Full path of a sound file.

    Returns:
      Transposed S: A 2d array. A transposed melspectrogram with shape of (length/T, n_mels)
      Transposed magnitude: A 2d array.Has shape of (length/T, 1+hp.n_fft//2)
      
    sr = 22050 # Sampling rate. Paper => 24000
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 30 # Number of inversion iterations 
    use_log_magnitude = True # if False, use magnitude
    '''
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=hp.sr) # or set sr to hp.sr.
    
    # stft. D: (1+n_fft//2, length/T)
    D = librosa.stft(y=y,
                     n_fft=hp.n_fft, 
                     hop_length=hp.hop_length, 
                     win_length=hp.win_length) 
    
    # magnitude spectrogram
    magnitude = np.abs(D) #(1+n_fft/2, length/T)
    
    # power spectrogram
    power = magnitude**2 #(1+n_fft/2, length/T) 
    
    # mel spectrogram
    S = librosa.feature.melspectrogram(S=power, n_mels=hp.n_mels) #(n_mels, length)

    return np.transpose(S.astype(np.float32)), np.transpose(magnitude.astype(np.float32)) # (length, n_mels), (length, 1+n_fft/2)
            
def shift_by_one(inputs):
    '''Shifts the content of `inputs` to the right by one 
      so that it becomes the decoder inputs.
      
    Args:
      inputs: A 3d tensor with shape of [N, length/T, C]
    Returns:
      A 3d tensor with the same shape and dtype as `inputs`.
    '''
    return tf.concat((tf.zeros_like(inputs[:, :1, :]), inputs[:, :-1, :]), 1)

def reduce_frames(arry, overlap, r):
    '''Reduces and adjust the shape and content of `arry` according to r.
    
    Args:                             T 
      arry: A 2d array with shape of [length/T, C]
      overlap: An int. Overlapping span.
      r: Reduction factor
     
    Returns:
      A 2d array with shape of [-1, C*r]
    '''
    length = arry.shape[0]
    # target shape is overlap*r
    num_padding = (overlap*r) - (length % (overlap*r)) if length % (overlap*r) !=0 else 0
    # zero pad at the end
    arry = np.pad(arry, [[0, num_padding], [0, 0]], 'constant', constant_values=(0, 0))
    length, C = arry.shape
    sliced = np.split(arry, list(range(overlap, length, overlap)), axis=0)
    
    started = False
    for i in range(0, len(sliced), r):
        if not started:
            reshaped = np.hstack(sliced[i:i+r])
            started = True
        else:
            reshaped = np.vstack((reshaped, np.hstack(sliced[i:i+r])))
            
    return reshaped

def spectrogram2wav(spectrogram):
    '''
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    spectrogram = spectrogram.T  # [f, t]
    X_best = copy.deepcopy(spectrogram)  # [f, t]
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)  # [f, t]
        phase = est / np.maximum(1e-8, np.abs(est))  # [f, t]
        X_best = spectrogram * phase  # [f, t]
    X_t = invert_spectrogram(X_best)

    return np.real(X_t)

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def restore_shape(arry, overlap, r):
    '''Reduces and adjust the shape and content of `arry` according to r.
    
    Args:
      arry: A 2d array with shape of [length/T, C]
      overlap: An int. Overlapping span.
      r: Reduction factor
     
    Returns:
      A 2d array with shape of [-1, C*r]
    '''
    length, C = arry.shape
    sliced = np.split(arry, list(range(overlap, length, overlap)), axis=0)
    
    started = False
    for s in sliced:
        if not started:
            restored = np.vstack(np.split(s, r, axis=1))
            started = True
        else:    
            restored = np.vstack((restored, np.vstack(np.split(s, r, axis=1))))
    
    # Trim zero paddings
    restored = restored[:np.count_nonzero(restored.sum(axis=1))]    
    return restored

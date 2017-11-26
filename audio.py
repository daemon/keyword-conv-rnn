from __future__ import print_function
import argparse
import os
import random
import sys
import wave

from PIL import Image
import librosa
import numpy as np
import pyaudio

def set_speech_format(f):
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)

def preprocess_audio(data, n_mels):
    data = librosa.feature.melspectrogram(data, sr=16000, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    data = data / (np.sqrt(np.var(data)) + 1E-7)
    data = data / (np.max(np.abs(data), 0) + 1E-6)
    data = np.transpose(data)
    return data.astype(dtype=np.float32)

def draw_spectrogram(spectrogram):
    spectrogram = np.transpose(spectrogram)
    spectrogram = np.abs(spectrogram)
    im = Image.fromarray((spectrogram * 256).astype(np.int8))
    im = im.resize((np.shape[0] * 5, 40 * 5))
    im.show()

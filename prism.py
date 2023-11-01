#!/usr/bin/env python3

import click
import sys
from scipy.io import wavfile
from tqdm import tqdm
from scipy.signal import spectrogram
import numpy as np
import os

import matplotlib.pyplot as plt

from math import sin, pi
import random
import typing

from PIL import Image

NPERSEG = None

def inverse_spectrogram(f: np.ndarray, t: np.ndarray, Sxx: np.ndarray, fs: int) -> np.ndarray:
  """inverse_spectrogram calculates the inverse spectrogram
  from frequencies f, intervals t, magnitude matrix Sxx using the sample rate fs.
Returns signal as a np.array."""
  #you can not hear the phase they say
  phases = [random.uniform(0,2*pi) for j in range(len(f))]
  #that +1 is here for off by one rounding errors i dont care to chase
  length= int(t[-1]*fs+1)
  #time axis
  out = np.zeros(length)
  for i in tqdm(range(1,len(t))):
    duration = t[i] - t[i-1]
    start =  t[i-1] * fs
    tics = int(duration*fs)
    for tic in range(tics):
      index = int(start + tic)
      #frequency axis
      for j in range(len(f)):
        magnitude = Sxx[j][i]
        #everyone loves the sin, because it starts from the zero
        out[index] += sin(2*pi/fs*f[j]*index+ phases[j]) * magnitude
  return out

@click.group()
def cli():
  pass

def float_to_int(arr):
  return (255 * (arr + 1) / 2).astype(np.uint8)

def int_to_float(arr):
  return (arr.astype(np.float64) / 255) * 2 - 1

@cli.command(name='convert')
@click.argument("filename")
def convert(filename):
  if not filename.endswith(".wav"):
    raise UserException("input audio file must be in WAV format.")

  rate, data = wavfile.read(filename)

  print(len(data))
  
  # make mono
  try:
    data = data[:,0]
  except IndexError:
    pass

  data = data.astype(np.float32)
  data = data / max(data)

  channels = 1

  if channels == 1:

    bins, t, spectrum = spectrogram(data, 44100, mode = "magnitude")
    
    print("************** len(bins), len(t)")
    print(len(bins))
    print(len(t))
      
      
    print(spectrum.shape)

    spectrum = spectrum / 0.02
    spectrum = spectrum * 256
    

    plt.plot(spectrum)
    plt.show()
    spectrum = np.rint(spectrum).astype(np.uint8)
    print(spectrum)

    plt.plot(spectrum.flatten())
    plt.show()

    print("spectrum is")
    print(spectrum)

    #spectrum = np.log(spectrum)
    #spectrum = spectrum / data.shape[0]

    print(len(data))
    
    im = Image.fromarray(spectrum, mode="L")
    im.save(os.path.basename(filename)[:-3] + "bmp")
  #elif channels == 2:
  #  r_bins, r_t, red = spectrogram(data[:,0], rate, mode = "magnitude")
  #  b_bins, b_t, blue = spectrogram(data[:,1], rate, mode = "magnitude")
  #  green = np.zeros(red.shape)
  #  im = Image.fromarray(np.stack((red, blue, green), axis=2), mode="RGB")
  #  print("original length: " + str(len(data)))
  #  print("new shape: " + str(red.shape))
  #  im.save(os.path.basename(filename)[:-3] + "bmp")
  #else:
  #  print("[error]: unable to handle number of channels in wav file.")


@cli.command(name='invert')
@click.argument("filename")
@click.argument("samples", type=click.INT)
def invert(filename, samples):
  im = np.asarray(Image.open(filename)).astype(np.float32)

  im = im / 256
  im = im * 0.02

  print(im)

  # idk if this is the best way
  ref_data = np.zeros(samples)
  
  f, t, _ = spectrogram(ref_data, 44100, nperseg = NPERSEG)

  print("************")
  print("len t, len f")
  print(len(t))
  print(len(f))

  import pickle
  with open("invert.pickle", "wb") as h:
    pickle.dump((f, t, im), h)

  
  signal = inverse_spectrogram(f, t, im, 44100)

  #signal = signal / max(signal)

  print("signal is")
  print(signal)

  plt.plot(signal)
  plt.show()
  
  wavfile.write("inverse.wav", 44100, signal)

@cli.command(name='test')
@click.argument("filename")
def test(filename):
  fs, data = wavfile.read(filename)

  left = data

  f, t, Sxx = spectrogram(left, fs, mode='magnitude')

  print("Sxx=")
  plt.plot(Sxx.flatten())
  plt.show()
  
  lInverse = inverse_spectrogram(f, t, Sxx, fs)
  
  wavfile.write("inverse.wav", fs, lInverse)

  
if __name__ == '__main__':
    cli()

    

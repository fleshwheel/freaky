import click
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from numba import jit

from PIL import Image
from scipy.signal import resample
from scipy.io import wavfile

#RESAMPLE = 2
#SAMPLE_RATE = 44_100 * RESAMPLE
#WINDOW_SIZE = 256
#FREQ_MIN = 1
#FREQ_MAX = SAMPLE_RATE / 2
FREQ_MAX = 15_000

@click.command()
@click.argument("in_file", required=True)
@click.argument("out_file", required=True)
@click.option("-r", "--sample-rate", default=44100, help="Sample rate of output audio WAV file.")
@click.option("-f", "--resample-factor", default=1, help="Resample input data before analysis.")
@click.option("-w", "--window-length", default=64, help="Width of analysis windows.")
def decode_wrapper(in_file, out_file, sample_rate, resample_factor, window_length):
    spectra = np.asarray(Image.open(in_file)).T.astype(np.float64) / 255
    result = decode(spectra, sample_rate * resample_factor, window_length)
    result = resample(result, len(result) // resample_factor)
    wavfile.write(out_file, sample_rate, result)

#@jit
def decode(spectra, sample_rate, window_length):

#    freqs = np.logspace(0, np.log(FREQ_MAX), spectra.shape[1], base=np.e).astype(np.uint)
    
    freqs = (FREQ_MAX * np.linspace(0, 1, spectra.shape[1]) ** 2) #.astype(np.uint)
#    print(freqs)
    
    spectra = spectra * spectra
#    spectra = np.exp(spectra)

    length = spectra.shape[0] * window_length
    T = np.linspace(0, length / sample_rate, spectra.shape[0] * window_length)

    components = np.zeros((len(freqs), length))
    phases = np.random.random(len(freqs)) * sample_rate * 4
    for i in range(len(freqs)):
        components[i] = freqs[i] * T
        if freqs[i] != 0:
            components[i] += phases[i] / freqs[i]
    
    components = np.sin(2 * np.pi * components)
#    for i in range(len(freqs)):
#        if freqs[i] != 0:
#            components[i] /= freqs[i]
        
    chunks = []
    last_spectrum = spectra[0]
    for window_idx in range(spectra.shape[0]):
#        spec_col = spectra[window_idx].copy().reshape(-1, 1)
#        chunk = np.repeat(spec_col, window_length, axis=1) => (32, 1024)
#        print(chunk.shape)
        chunk = np.linspace(last_spectrum, spectra[window_idx], window_length).T
#        chunk = np.linspace(spectra[window_idx], spectra[window_idx], window_length).T
        print(chunk.shape)
        chunks.append(chunk)
        last_spectrum = spectra[window_idx]
        
    coeffs = np.hstack(chunks)
   
    result = np.sum(np.multiply(coeffs, components), axis = 0)
    result /= max(result.flatten())

    return result
    

if __name__ == "__main__":
    decode_wrapper()

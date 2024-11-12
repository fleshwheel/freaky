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

def decode(spectra, sample_rate, window_length):
    
    freqs = np.linspace(0, sample_rate // 2, spectra.shape[1]).astype(np.uint)
    
#    spectra = spectra * spectra

    length = spectra.shape[0] * window_length
    T = np.linspace(0, length / sample_rate, spectra.shape[0] * window_length)

    components = np.zeros((len(freqs), length))
    phases = np.random.random(len(freqs)) * 2 * np.pi
    for i in range(len(freqs)):
        components[i] = freqs[i] * T + phases[i]
    components = np.sin(2 * np.pi * components)

    chunks = []
    last_spectrum = spectra[0]
    for window_idx in tqdm(range(spectra.shape[0])):
        spec_col = np.array(spectra[window_idx], ndmin=2).T
        chunk = np.linspace(last_spectrum, spectra[window_idx], window_length).T
        chunks.append(chunk)
        
    coeffs = np.hstack(chunks)
   
    result = np.sum(np.multiply(coeffs, components), axis = 0)
    result /= max(result.flatten())

    return result
    

if __name__ == "__main__":
    decode_wrapper()

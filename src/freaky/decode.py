#!/usr/bin/env python3

import click
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from numba import jit, prange

from PIL import Image
from scipy.signal import resample
from scipy.io import wavfile

# default values
FREQ_MAX = 20_000
STRIDE = 64
SAMPLE_RATE = 192000

@click.command()
@click.argument("in_file", required=True)
@click.argument("out_file", required=True)
@click.option("-x", "--resample-factor", default=1, help="Resample input data before analysis.")
@click.option("-s", "--stride", default=64, help="Space between centers of consecutive analysis windows.")
@click.option("-r", "--sample-rate", default=192000, help="Sample rate of output audio.")
@click.option("-2", "--stereo", is_flag=True)
def decode_cli(in_file, out_file, sample_rate, resample_factor, stride, stereo):
    decode(in_file, out_file, sample_rate, resample_factor, stride, stereo)

def decode(in_file, out_file,
           sample_rate = SAMPLE_RATE, stride = STRIDE,
           stereo = False):
    im = Image.open(in_file)

    if stereo:
        im = im.convert("RGB")
    else:
        im = im.convert("L")
    
    image_data = np.asarray(im, dtype=np.float64).T / 255

    if stereo:
        spectra_l = image_data[1]
        spectra_r = image_data[0]

        audio_l = _decode(spectra_l, sample_rate * resample_factor, stride)
        audio_r = _decode(spectra_r, sample_rate * resample_factor, stride)

        audio_l = resample(audio_l, len(audio_l) // resample_factor)
        audio_r = resample(audio_r, len(audio_r) // resample_factor)
        audio = np.array([audio_l, audio_r]).T

    else:
        audio = _decode(image_data, sample_rate * resample_factor, stride)
        audio = resample(audio, len(audio) // resample_factor)

    wavfile.write(out_file, sample_rate, audio)

@jit(nopython=True, parallel=True, nogil=True)
def _decode(spectra, sample_rate, stride):

    num_windows = spectra.shape[0]
    num_freqs = spectra.shape[1]
    num_samples = num_windows * stride
    
    freqs = np.linspace(0, FREQ_MAX, num_freqs)
    
    spectra = spectra * spectra

    length = num_windows * stride
    T = np.linspace(0, length / sample_rate, num_windows * stride)

    result = np.zeros(num_samples, dtype=np.float64)

    phases = np.random.random(len(freqs)) * 4 * np.pi
    phases[0] = 0

    for i_f in prange(len(freqs)):
        term = np.zeros(num_samples)
        
        component = freqs[i_f] * T
        component = np.sin(2 * np.pi * component + phases[i_f])

        last_coeff = spectra[0][i_f]
        for i_w in prange(num_windows):
            coeff = spectra[i_w][i_f]
            window = np.linspace(last_coeff, coeff, stride)
            wstart = i_w * stride
            wend = wstart + stride
            term[wstart: wend] = np.multiply(window, component[wstart: wend])
            last_coeff = coeff
        result += term

    return result
   

if __name__ == "__main__":
    decode_cli()

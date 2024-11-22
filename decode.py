import click
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from numba import jit, prange

from PIL import Image
from scipy.signal import resample
from scipy.io import wavfile

FREQ_MAX = 20_000

@click.command()
@click.argument("in_file", required=True)
@click.argument("out_file", required=True)
@click.option("-x", "--resample-factor", default=1, help="Resample input data before analysis.")
@click.option("-w", "--window-length", default=64, help="Space between centers of consecutive analysis windows.")
@click.option("-r", "--sample-rate", default=44100, help="Sample rate of output audio.")
@click.option("-2", "--stereo", is_flag=True)
def decode_wrapper(in_file, out_file, sample_rate, resample_factor, window_length, stereo):
    im = Image.open(in_file)

    if stereo:
        im = im.convert("RGB")
    else:
        im = im.convert("L")
    
    image_data = np.asarray(im, dtype=np.float64).T / 255

    if stereo:
        spectra_l = image_data[0]
        spectra_r = image_data[2]

        audio_l = decode(spectra_l, sample_rate *resample_factor, window_length)
        audio_r = decode(spectra_r, sample_rate * resample_factor, window_length)

        #audio_l = resample(audio_l, len(audio_l) // resample_factor)
        #audio_r = resample(audio_r, len(audio_r) // resample_factor)
        audio = np.array([audio_l, audio_r]).T

    else:
        audio = decode(image_data, sample_rate * resample_factor, window_length)
        audio = resample(audio, len(audio) // resample_factor)

    wavfile.write(out_file, sample_rate, audio)

@jit(nopython=True, parallel=True, nogil=True)
def decode(spectra, sample_rate, window_length):

    num_windows = spectra.shape[0]
    num_freqs = spectra.shape[1]
    num_samples = num_windows * window_length
    
    freqs = np.linspace(0, FREQ_MAX, num_freqs)
    
#    spectra = spectra * spectra
#    spectra = np.exp(spectra)

    length = num_windows * window_length
    T = np.linspace(0, length / sample_rate, num_windows * window_length)

    result = np.zeros(num_samples, dtype=np.float64)

    print(result.dtype)

    #components = np.zeros((len(freqs), length))
    phases = np.random.random(len(freqs)) * 1000

    for i_f in prange(len(freqs)):
        term = np.zeros(num_samples)
        
        component = freqs[i_f] * T
        #if freqs[i_f] != 0:
        #    component += phases[i_f] / freqs[i_f]
        if i_f % 2 == 0:
            component = np.cos(2 * np.pi * component)
        else:
            component = np.sin(2 * np.pi * component)

        last_coeff = spectra[0][i_f]
        for i_w in prange(num_windows):
            coeff = spectra[i_w][i_f]
            window = np.linspace(last_coeff, coeff, window_length)
            wstart = i_w * window_length
            wend = wstart + window_length
            term[wstart: wend] = np.multiply(window, component[wstart: wend])
            last_coeff = coeff
        result += term
  
    result /= max(np.abs(result.flatten()))
    
    return result
   

if __name__ == "__main__":
    decode_wrapper()
